"""
timingtask.variants -- named training configurations.
==================================================================

One generator, one trainer, flags. Nothing here forks the task: every entry is
a dict of overrides applied to :class:`TimingTaskConfig`, :class:`SchedulerConfig`
and :class:`RLTrainer`, so a variant is a row in a table rather than a copy of
the code, and two variants can never drift apart in the parts they share.

The problem these address: with the ITI restart rule on, an agent that licks
freely almost never reaches a cue, so it cannot experience cue->reward and
cannot learn why it should stop licking. The round-1 runs below attack that
trade-off from different sides; none of them worked.

    G  order-of-magnitude larger per-lick ITI penalty, restart ON
    H  restart OFF, large per-lick ITI penalty
    I  restart ON, penalty on P(lick) at every ITI step
    J  restart OFF + per-step P(lick) penalty

Usage
-----
    python -m timingtask.variants G --steps 400
    python -m timingtask.variants all --steps 400 --out runs/

or from a notebook::

    from timingtask.variants import run, RUNS
    hist = run("G", steps=400)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import TimingTaskConfig, SchedulerConfig, ObservationConfig
from .rl import (RLTrainer, ActorCritic, summarise,
                 summarise_by_delay, format_delay_report)

__all__ = ["BASE", "RUNS", "ROUND2", "ROUND3", "ROUND4", "ROUND6",
           "ROUND7", "TESTS", "PHASES", "build", "run", "run_all", "describe",
           "config_matrix"]


# --------------------------------------------------------------------------
# BASE = configuration F from the curriculum sweep, plus the relaxed promotion
# thresholds. Everything a variant does is stated as a delta from this, so the
# comparison across runs is a comparison of one or two numbers.
# --------------------------------------------------------------------------
BASE: Dict[str, Dict[str, Any]] = dict(
    task=dict(
        dt=0.02,
        answer_window=0.8,        # short: lick-immediately-after-cue is the
                                  # behaviour the cue-association stage teaches
        post_lick=1.5,
        cue_duration=0.6,
        cue_mode="pulse",
        # A long stop-licking period makes free licking expensive in time as
        # well as in reward, and gives the ITI penalty something to act on.
        iti_mean=3.0, iti_min=2.0, iti_max=5.0,
        iti_timeout=20.0,
        no_cue_prob=0.1,
        # --- reinforcement, round 5 -------------------------------------
        # Every lick that is not a correctly timed one costs the same -2.0,
        # wherever it happens: in the stop-licking period, on a catch trial, or
        # before the delay elapsed. Different prices for the same mistake were
        # three free parameters expressing one idea.
        reward=15.0,              # was 10.0
        early_penalty=-2.0,       # was -1.0
        no_cue_lick_penalty=-2.0, # was -0.1
        iti_lick_penalty=-2.0,
        miss_penalty=-2.0,
        iti_timeout_penalty=-10.0,
        time_penalty=-0.05,
        time_penalty_in_post=False,
        # Across-trial subjective value: a run of failures makes water worth
        # more and a wasted trial hurt more; a run of successes makes both
        # worth less. Scales `reward` and `early_penalty` together.
        reward_rate_gain=1.0,
        reward_rate_window=100,
        reward_rate_ref=0.5,
        # Temporal discounting REMOVED. With it, water at 1.8 s was worth 4.1
        # against 9.5 at 0.1 s, and combined with the step cost the return
        # difference between solving the trial and licking immediately fell to
        # 0.74 -- against a batch-to-batch swing in the return of about 30. The
        # task stopped preferring the correct answer at exactly the delays the
        # curriculum was pushing toward. Waiting is already punished once, by
        # the step cost; twice made the curriculum unlearnable.
        discount_rate=0.0,        # was 0.5
        iti_restart_on_lick=False,
        seed=0,
    ),
    sched=dict(
        mode="cue_autolearn",
        cue_association_delay=0.1,
        cue_association_response_window=0.6,
        cue_association_window=100,
        cue_association_min_trials=100,
        # RELAXED (user request): promotion no longer needs the agent to solve
        # both halves of the trade-off at once.
        cue_association_success_threshold=0.30,
        # ITI criterion OFF (None). It was a fraction-of-trials measure that
        # could not see a 10x fall in ITI licking, and the restart rule it
        # was paired with is off too. Set a licks/second number to re-enable.
        cue_association_max_iti_lick_hz=None,
        initial_delay=0.1, delay_step=0.1, min_delay=0.1, max_delay=2.0,
    ),
    obs=dict(n_lags=1),
    trainer=dict(
        n_envs=16, lr=4e-3, gamma=1.0,
        reward_scale=0.1,
        # The pre-clip gradient norm has been running 6-40 against this, so
        # every update was throttled and the effective step size was a constant
        # rather than anything responsive. 0 turns clipping off.
        grad_clip=1.0,
        iti_readout_penalty=0.0,
        activity_penalty=0.0,
        activity_deriv_penalty=0.0,
        activity_scope="iti",
        # OFF. It was 0.02 PER 20 ms STEP, which over a 240-step trial gives
        # P(at least one floor lick) = 0.992 -- the floor was not a floor, it
        # was the policy. Measured: the late-training lick distribution was a
        # clean exponential with a 0.96 s decay constant against the floor's
        # own 1.0/s hazard, and shuffling lick times against delays changed the
        # reward rate by 2%, i.e. the timing carried no information. If
        # exploration is reintroduced it must be sized per TRIAL: ~2e-4 per
        # step is a 5% chance of one exploratory lick in a trial.
        min_action_prob=0.0,
        seed=0,
    ),
    # tau is the unit time constant in MILLISECONDS (same units as dt = 20).
    # 100 ms is the standard rate-unit value, taken to stand for NMDA-dominated
    # synaptic decay; a membrane time constant is 10-20 ms. Slow behaviour has
    # to come from the RECURRENT CONNECTIVITY, not from the leak -- Yang et
    # al.'s model integrates for seconds out of 10 ms units, purely because its
    # striatal pair has an eigenvalue of exactly 1.
    # A (low, high) pair gives log-uniform time constants across units; the
    # machinery is there but is NOT in use, and any range should be justified
    # against measured intrinsic timescales before it is.
    # g and rec_init set the initial recurrent spectrum, which is what decides
    # how many SLOW modes the network starts with. Measured at N=128, a=0.2,
    # over 10 seeds -- max|lambda_J| above 1 means spontaneously chaotic:
    #
    #   gaussian   g=1.0   max 0.995   3.4 modes > 1 s
    #   gaussian   g=1.2   max 1.031  10.7 modes > 1 s   CHAOTIC
    #   orthogonal g=1.0   max 1.000  16.0 modes > 1 s
    #   orthogonal g=1.2   max 1.040  28.8 modes > 1 s   CHAOTIC
    #
    # Orthogonal at g=1.0 dominates gaussian at g=1.2 on both axes: 5x the slow
    # modes of the default and it does NOT leave the unit disc, because all
    # |lambda_W| = 1 exactly and the slow ones are those near angle 0.
    model=dict(hidden=128, tau=100.0, noise=0.05, train_tau=False,
               g=1.0, rec_init="gaussian"),
)


# --------------------------------------------------------------------------
# The variants. Each value is a nested dict of deltas from BASE.
# --------------------------------------------------------------------------
RUNS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------ round 1
    # Kept for reproducibility only. F, G and I never reached a single cue in
    # 400 updates and cannot: completing one stop-licking period needs ~150
    # consecutive non-lick steps, which has probability 7e-46 at the initial
    # lick probability of 0.5. Do not re-run them.
    # They restate the round-1 reward values EXPLICITLY, because BASE has since
    # moved: the restart rule is off there now, and the discount is gone. A
    # variant defined as a delta from a moving BASE stops meaning what it meant.
    "F": dict(task=dict(iti_restart_on_lick=True, iti_lick_penalty=-0.05,
                        discount_rate=0.5, reward=10.0, early_penalty=-1.0,
                        no_cue_lick_penalty=-0.1, time_penalty_in_post=True)),
    "G": dict(task=dict(iti_restart_on_lick=True, iti_lick_penalty=-2.0,
                        discount_rate=0.5, reward=10.0, early_penalty=-1.0,
                        no_cue_lick_penalty=-0.1, time_penalty_in_post=True)),
    "H": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                        discount_rate=0.5, reward=10.0, early_penalty=-1.0,
                        no_cue_lick_penalty=-0.1, time_penalty_in_post=True)),
    "I": dict(task=dict(iti_restart_on_lick=True, iti_lick_penalty=-0.05,
                        discount_rate=0.5, reward=10.0, early_penalty=-1.0,
                        no_cue_lick_penalty=-0.1, time_penalty_in_post=True),
              trainer=dict(iti_readout_penalty=1.0)),
    "J": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-0.05,
                        discount_rate=0.5, reward=10.0, early_penalty=-1.0,
                        no_cue_lick_penalty=-0.1, time_penalty_in_post=True),
              trainer=dict(iti_readout_penalty=1.0)),

    # ------------------------------------------------------------------ round 2
    # Every entry below switches the restart rule OFF; round 1 established that
    # nothing is learnable with it on. Names are words, not letters, because
    # "H" was also the entropy column in the training log.

    # H and J together: the per-lick price of round 1's H, plus round 1's J
    # per-step pressure on the lick probability.
    "combo": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                  trainer=dict(iti_readout_penalty=1.0)),

    # A missed answer window costs 10x more. Pushes toward licking, against the
    # never-lick absorbing state that killed round 1's H.
    "miss": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                           miss_penalty=-20.0),
                 trainer=dict(iti_readout_penalty=1.0)),

    # Activity instead of readout: penalise mean ||h||^2 during the ITI, so the
    # network is pushed toward a quiet baseline state rather than toward a small
    # lick readout specifically. Coefficient chosen so the term is ~0.2 at
    # initialisation -- same order as the policy loss, not dominant. This is the
    # first number to sweep.
    "act": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                trainer=dict(iti_readout_penalty=0.0,
                             activity_penalty=20.0)),

    # The derivative version: penalise ||h_t - h_{t-1}||^2 during the ITI, so
    # what is discouraged is CHANGE in the state rather than its size. Leaves
    # a large quiet state permissible and only forbids drift.
    "dact": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                 trainer=dict(iti_readout_penalty=0.0,
                              activity_deriv_penalty=200.0)),

    # Water worth 10x more, so one correct post-cue lick outweighs a long run
    # of ITI penalties and the cue->reward association has a large gradient.
    "bigrew": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                             reward=100.0),
                   trainer=dict(iti_readout_penalty=1.0)),

    # `combo` plus the probability floor, from before the floor became the
    # default. Kept for reproducibility.
    "floor": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                  trainer=dict(iti_readout_penalty=1.0, min_action_prob=0.02)),
    # `act` with the floor switched OFF -- the ablation of the mechanism that
    # replaced the entropy bonus.
    # ---------------------------------------------------------------- round 6
    # EVERY entry below declares `parent="act"`, so it is `act` plus one
    # change. The first round-6 sweep did not: the six variants were deltas
    # from BASE, which has activity_penalty = 0.0, so none of them carried the
    # activity penalty and the sweep compared six BASE runs against one `act`
    # run. Its results are void. Do not write a round-N variant as a bare
    # delta from BASE again unless BASE is genuinely the control.

    # Ablations of the heterogeneous time constants.
    "act_traintau": dict(parent="act", model=dict(train_tau=True)),

    # ---- the cue as a tonic step, not just a transient --------------------
    # With tau = 100 ms a 0.6 s pulse is gone within ~100 ms of cue offset, so
    # the network free-runs with NO input through the part of the trial it has
    # to time. A step held from cue onset gives a near-integrator something to
    # accumulate, which is how Yang et al.'s model produces a ramp at all.
    "act_step": dict(parent="act", task=dict(cue_mode="step")),
    "act_both": dict(parent="act", task=dict(cue_mode="both")),
    # Everything that currently looks promising, at once: the transient-plus-
    # tonic input, the initialisation with the most slow modes, and no gradient
    # throttling. If the isolated changes each do a little and this does a lot,
    # they interact; if this matches the best single change, they do not.
    "act_all": dict(parent="act",
                    task=dict(cue_mode="both"),
                    model=dict(rec_init="orthogonal", g=1.0),
                    trainer=dict(grad_clip=0.0)),

    # More slow modes at initialisation, two ways. `act_g12` leaves the unit
    # disc and is expected to be chaotic; `act_orth` does not.
    "act_g12": dict(parent="act", model=dict(g=1.2)),
    "act_orth": dict(parent="act", model=dict(rec_init="orthogonal", g=1.0)),

    # No gradient clipping. The pre-clip norm has been running 6-40 against a
    # clip of 1.0, so EVERY update was throttled and the effective step size
    # was a constant rather than anything responsive to the gradient.
    "act_noclip": dict(parent="act", trainer=dict(grad_clip=0.0)),

    # ---------------------------------------------------------------- round 7
    # `act_all` minus the orthogonal initialisation. Round 6 gave the
    # orthogonal spectrum no credit anywhere it was tested alone (`act_orth`
    # 0.63 correct against `act`'s 0.69), while the two changes it was bundled
    # with -- the tonic cue and unthrottled gradients -- are each traceable to
    # a mechanism. If this matches `act_all`, the orthogonal init is dropped.
    # NOTE: `act_noclip` ALONE collapsed to lick-immediately. Unclipped
    # gradients are only survivable with the tonic cue, which is why the pair
    # is tested and not the clip change on its own.
    "act_both_noclip": dict(parent="act",
                            task=dict(cue_mode="both"),
                            trainer=dict(grad_clip=0.0)),


    # ------------------------------------------------------------------ round 3
    # `act` was the only run in rounds 1-2 that learned monotonically and did
    # not collapse (cue success 0.84 at update 400, ITI licks 55 -> 6, entropy
    # stable at 0.22). Round 3 asks what part of it was necessary: the
    # magnitude penalty, the derivative penalty, or the restriction to the ITI.
    # All six share `act`'s task settings and differ only in the two
    # coefficients and the scope.

    # Both penalties, ITI only: low activity AND low change, cue response free.
    "act_dact_iti": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="iti",
                     activity_penalty=20.0, activity_deriv_penalty=200.0)),

    # Magnitude penalty over the WHOLE trial: a metabolic budget. Prices the
    # cue-evoked response too, so it may suppress the behaviour it is meant to
    # permit.
    # Coefficient 5.0, not `act`'s 20.0, ON PURPOSE: post-cue activity is
    # larger, so mean ||h||^2 over the whole trial is 0.0367 at initialisation
    # against 0.0091 over the ITI alone. At a shared coefficient this run would
    # differ from `act` in scope AND in strength, and a failure could not be
    # attributed to either. 5.0 matches the initial term magnitude (~0.18).
    "act_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_penalty=5.0)),

    # Derivative penalty over the whole trial: pay for CHANGING the state, not
    # for holding it. The closest thing here to a coding-efficiency constraint.
    "dact_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_deriv_penalty=200.0)),

    # ------------------------------------------------------------------ round 4
    # `act`'s lick times are wide. One candidate cause left: the price of
    # waiting is too low. 3x the per-step time cost.
    "act_time": dict(
        task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                  time_penalty=-0.15),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="iti",
                     activity_penalty=20.0)),

    # Both, whole trial. Same magnitude matching as act_full; the derivative
    # term needs none (0.00127 full against 0.00114 ITI).
    "act_dact_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_penalty=5.0, activity_deriv_penalty=200.0)),
}

# The round-2 set, for `--name round2`.
ROUND2 = ["combo", "miss", "act", "dact", "bigrew", "floor"]

# Round 3: `act` is included as the reference, and is the only one already run.
ROUND3 = ["act", "act_dact_iti", "act_full", "dact_full", "act_dact_full"]

# Round 4: why are the lick times wide? Two candidate causes, one run each.
ROUND4 = ["act", "act_time"]

# Round 6: does a spread of time constants give the network a timer to build?
# Round 6: one control, each candidate change isolated against it, and the
# full stack. Three factors -- the recurrent initialisation, gradient clipping,
# and whether the cue persists -- so a full factorial would be twelve runs;
# this is the main effects plus the combination.
ROUND6 = ["act",          # control: gaussian g=1, clip 1.0, transient cue
          "act_g12",      # gaussian g=1.2   (chaotic: max|lambda_J| = 1.04)
          "act_orth",     # orthogonal g=1.0 (19 slow modes vs 5, still stable)
          "act_noclip",   # gradient clipping off
          "act_step",     # cue as a tonic step instead of a transient
          "act_both",     # transient AND tonic, as Yang et al. have it
          "act_all"]      # orthogonal + no clip + both cue channels

# Round 7: the full curriculum, to 1.8 s rather than stopping at 1.0. Run with
#   --target-delay 1.8 --probe-max-delay 2.0
# The candidate is first; the other two are the controls it has to match.
ROUND7 = ["act_both_noclip",   # gaussian g=1, no clip, transient + tonic cue
          "act_all",           # the same plus the orthogonal init
          "act_both"]          # the same with gradient clipping back on

# --------------------------------------------------------------------------
# TRAINING PHASES. Weights keep learning through all of them; the schedule
# changes between them and the Adam state, the records and the weight
# snapshots carry over.
#
# The reason this exists: the growing curriculum presents ONE delay at a time
# and only ever raises it, so a memorised interval is the optimal policy and
# that is exactly what round 6 produced -- `act_all` licked at 1.30 s whether
# the required delay was 1.0 s or 1.8 s. An agent can only be expected to
# infer the delay if it is TRAINED where the delay changes under it. Phase 3
# is that training; phases 1 and 2 get it to a long delay first.
# --------------------------------------------------------------------------
PHASES: Dict[str, list] = {
    "r7": [
        # 1. the published curriculum, 0.1 s -> 1.8 s in 0.1 s steps
        dict(name="curriculum", sched=None, steps=12000,
             target_delay=1.8, patience=60, score="delay+correct"),
        # 2. hold 1.8 s. Consolidates the long delay before anything moves.
        dict(name="fixed18", steps=3000, patience=40, score="correct",
             sched=dict(mode="fixed", fixed_delay=1.8)),
        # 3. blocks of 1.0 s and 1.8 s, 50-150 trials each, unsignalled. The
        #    delay is knowable only from the outcome of recent trials, so this
        #    is the first phase in which timing beats a fixed interval.
        dict(name="switch", steps=6000, patience=80, score="correct",
             sched=dict(mode="block", block_delays=[1.0, 1.8],
                        block_min_trials=50, block_max_trials=150)),
    ],
    # The old single-phase behaviour, for reproducing rounds 1-6.
    "curriculum": [dict(name="curriculum", sched=None, steps=8000,
                        target_delay=1.0, patience=40,
                        score="delay+correct")],
}

# Frozen-weight test conditions, run after training. Each puts the agent at a
# delay it did not necessarily train at: two it has to hold, and one that
# changes under it. Testing at the delay training happened to stop on cannot
# separate an agent that times from one that memorised an interval.
#
# `fixed1s` and `fixed18` are a matched pair and are the cleanest test there
# is: the SAME frozen network at two delays 0.8 s apart. An agent that times
# shifts its lick by 0.8 s between them; one that memorised an interval does
# not move.
TESTS: Dict[str, Dict[str, Any]] = {
    "fixed1s": dict(mode="fixed", fixed_delay=1.0),
    "fixed18": dict(mode="fixed", fixed_delay=1.8),
    "switch": dict(mode="block", block_delays=[1.0, 1.8],
                   block_min_trials=50, block_max_trials=150),
}


# Sections of a spec. Anything else at the top level of a RUNS entry is a
# directive, not a block of overrides; `parent` is the only one.
_SECTIONS = ("task", "sched", "obs", "trainer", "model")


def _merge(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for section, fields in delta.items():
        if section not in _SECTIONS:
            continue                      # `parent`, handled by _chain
        out.setdefault(section, {}).update(fields)
    return out


def _chain(name: str) -> list:
    """Ancestry of a variant, oldest first, e.g. ``["act", "act_orth"]``.

    A variant may name a `parent`, in which case it is a delta from THAT
    variant's resolved spec rather than from BASE. Round 6 is the reason this
    exists: `act_orth` and its five siblings were written as deltas from BASE,
    so every one of them silently inherited `activity_penalty = 0.0` instead of
    `act`'s 20.0 -- the one ingredient that had ever made a run learn. Seven
    runs of a supposed ablation of `act` contained no `act` at all.
    """
    seen, order, cur = set(), [], name
    while cur is not None:
        if cur in seen:
            raise ValueError(f"Cyclic parent chain at {cur!r}: {order}")
        if cur not in RUNS:
            raise KeyError(f"Unknown variant {cur!r}. Available: {sorted(RUNS)}")
        seen.add(cur)
        order.append(cur)
        cur = RUNS[cur].get("parent")
    return list(reversed(order))


def _resolve(name: str) -> Dict[str, Any]:
    """Full spec for a variant: BASE, then each ancestor, then the variant."""
    spec = copy.deepcopy(BASE)
    for anc in _chain(name):
        spec = _merge(spec, RUNS[anc])
    return spec


def _flat_deltas(name: str) -> Dict[str, Any]:
    """Every field this variant changes relative to BASE, ancestors included."""
    out: Dict[str, Any] = {}
    for anc in _chain(name):
        for sec, fields in RUNS[anc].items():
            if sec in _SECTIONS:
                out.update(fields)
    return out


def _make_core(input_size: int, hidden: int, tau, noise: float,
               train_tau: bool = False, g: float = 1.0,
               rec_init: str = "gaussian"):
    """The recurrent core. Import is deferred and tolerant because the model
    module has moved once already; a variant sweep should not fail on that."""
    try:
        from .models import VanillaRNN                       # type: ignore
    except Exception:                                          # pragma: no cover
        from timingtask.models import VanillaRNN         # type: ignore
    return VanillaRNN(input_size, hidden, 1, tau=tau, dt=20.0, noise=noise,
                      train_tau=train_tau, g=g, rec_init=rec_init)


def build(name: str, **overrides):
    """Return ``(trainer, model, spec)`` for a named variant.

    ``overrides`` are flat keyword arguments routed to whichever section owns
    the field, so ``build("G", iti_lick_penalty=-5.0, activity_penalty=5.0)``
    works without knowing which dataclass a name lives on.
    """
    if name not in RUNS:
        raise KeyError(f"Unknown variant {name!r}. Available: {sorted(RUNS)}")
    spec = _resolve(name)

    for k, v in overrides.items():
        for section, fields in spec.items():
            if k in fields:
                fields[k] = v
                break
        else:
            raise KeyError(f"Unknown field {k!r}")

    task = TimingTaskConfig(**spec["task"])
    sched = SchedulerConfig(**spec["sched"])
    obs = ObservationConfig(**spec["obs"])

    tr_kw = dict(spec["trainer"])
    trainer = RLTrainer(task, sched, obs, **tr_kw)

    m = spec["model"]
    torch.manual_seed(int(spec["trainer"].get("seed", 0)))
    tau = m["tau"]
    tau = tuple(tau) if isinstance(tau, (tuple, list)) else float(tau)
    core = _make_core(trainer.obs_size, int(m["hidden"]), tau,
                      float(m["noise"]), bool(m.get("train_tau", False)),
                      float(m.get("g", 1.0)), str(m.get("rec_init", "gaussian")))
    model = ActorCritic(core, hidden_size=int(m["hidden"]))
    return trainer, model, spec


def run(name: str, *, steps: int = 400, log_every: int = 25,
        verbose: bool = True, out_dir: Optional[str] = None,
        n_eval: int = 400, n_probe: int = 10000, n_test: int = 10000,
        probe_max_delay: float = 1.8, target_delay: Optional[float] = 1.0,
        patience: Optional[int] = 40, plots: bool = True,
        phases: Optional[list] = None, **overrides):
    """Train one variant. Returns ``(hist, trainer, model)``.

    With ``out_dir`` set, writes for each run: the training history, the model
    weights, every trial record (training and evaluation), and three figures --
    the optimiser view over updates, behaviour over trials during training, and
    behaviour of the finished agent at a frozen delay.
    """
    trainer, model, spec = build(name, **overrides)
    if verbose:
        chain = " -> ".join(["BASE"] + _chain(name))
        print(f"=== {name} ===   [{chain}]")
        print(f"    iti_lick_penalty   {spec['task']['iti_lick_penalty']}")
        print(f"    iti_restart_on_lick{'':1} {spec['task']['iti_restart_on_lick']}")
        print(f"    iti_readout_penalty {spec['trainer']['iti_readout_penalty']}")
        print(f"    activity_pen        {spec['trainer']['activity_penalty']}"
              f"   deriv {spec['trainer']['activity_deriv_penalty']}"
              f"   scope {spec['trainer']['activity_scope']}")
        print(f"    miss_penalty        {spec['task']['miss_penalty']}"
              f"   reward {spec['task']['reward']}")
        print(f"    min_action_prob     {spec['trainer']['min_action_prob']}")
        print(f"    cue_mode            {spec['task']['cue_mode']}")
        print(f"    tau {spec['model']['tau']}  g {spec['model'].get('g', 1.0)}"
              f"  rec_init {spec['model'].get('rec_init', 'gaussian')}"
              f"  grad_clip {spec['trainer']['grad_clip']}")
        print(f"    reward_rate_gain    {spec['task']['reward_rate_gain']}"
              f"   window {spec['task']['reward_rate_window']}")

    # One phase or several. A single-phase run is exactly the old behaviour.
    if phases is None:
        phases = [dict(name="curriculum", sched=None, steps=steps,
                       target_delay=target_delay, patience=patience,
                       score="delay+correct")]
    hist: Dict[str, list] = {}
    boundaries: list = []
    offset = 0
    for k, ph in enumerate(phases):
        if verbose:
            sch = ph.get("sched")
            print(f"  -- phase {k + 1}/{len(phases)}: {ph['name']}  "
                  f"({'curriculum unchanged' if sch is None else sch})  "
                  f"steps<={ph.get('steps', steps)}", flush=True)
        h = trainer.train(
            model, steps=int(ph.get("steps", steps)), log_every=log_every,
            verbose=False, sched=ph.get("sched"), phase=ph["name"],
            reset=(k == 0), score=ph.get("score", "delay+correct"),
            target_delay=ph.get("target_delay"), patience=ph.get("patience"),
            on_log=_printer(trainer, ph["name"]) if verbose else None)
        # One continuous update axis across phases, so the training dashboard
        # reads as one run with marked boundaries rather than three plots.
        for key, v in h.items():
            hist.setdefault(key, []).extend(
                [x + offset for x in v] if key == "step" else v)
        offset += (h["step"][-1] if h.get("step") else 0)
        boundaries.append(offset)
        if verbose:
            print(f"     phase {ph['name']} ended: {trainer.stop_reason} "
                  f"(cumulative update {offset})", flush=True)
    boundaries = boundaries[:-1]          # the last one is the end of the run
    phase_records = {}
    for r in trainer.records:
        phase_records.setdefault(r.get("phase", "curriculum"), []).append(r)
    if verbose:
        for pname, recs in phase_records.items():
            print(format_delay_report(summarise_by_delay(recs),
                                      f"training phase {pname}"), flush=True)

    # Behaviour of the finished agent, at the delay training left it at, with
    # no further learning. Separate from the training records because those are
    # a moving curriculum and a raster over them mixes several tasks.
    eval_records = []
    if n_eval:
        eval_records = trainer.evaluate(model, n_trials=int(n_eval),
                                        collect_states=True)
        if verbose:
            e = summarise(eval_records)
            print(f"    eval ({len(eval_records)} trials, delay frozen): "
                  f"engaged {e['p_engaged']:.2f}  correct {e['p_correct']:.2f}  "
                  f"early {e['p_early']:.2f}  miss {e['p_miss']:.2f}  "
                  f"ITI licks/trial {e['n_iti_licks']:.2f}  "
                  f"first lick {e['first_lick']:.3f}s", flush=True)

    test_records: Dict[str, list] = {}
    if n_test:
        for label, sched in TESTS.items():
            test_records[label] = trainer.test(model, n_trials=int(n_test),
                                               sched=sched, collect_states=True)
            if verbose:
                t = summarise(test_records[label])
                print(f"    test {label:8s} ({len(test_records[label])} trials): "
                      f"engaged {t['p_engaged']:.2f}  correct {t['p_correct']:.2f} "
                      f" early {t['p_early']:.2f}  miss {t['p_miss']:.2f}  "
                      f"first lick {t['first_lick']:.3f}s  "
                      f"delay {t['delay']:.2f}s", flush=True)
                rep = summarise_by_delay(test_records[label])
                if len(rep["rows"]) > 1:
                    print(format_delay_report(rep, f"test {label}"), flush=True)

    # The matched pair: the SAME frozen network at 1.0 s and at 1.8 s. Pooling
    # them puts both delays in one table with a single shared cut-off, which is
    # the cleanest available statement of whether the lick time follows the
    # delay or is a memorised constant.
    if verbose and {"fixed1s", "fixed18"} <= set(test_records):
        rep = summarise_by_delay(test_records["fixed1s"]
                                 + test_records["fixed18"])
        print(format_delay_report(rep, "fixed 1.0s vs fixed 1.8s"), flush=True)

    # The probe: weights frozen, curriculum still running, delay allowed up to
    # probe_max_delay. A fixed-delay evaluation only asks whether ONE memorised
    # interval is right; this asks whether the agent can follow a delay that
    # keeps moving. Cheap -- no gradients.
    probe_records = []
    if n_probe:
        # After a phased run the live schedule is the last phase's; the probe
        # needs a growing curriculum, so hand it one explicitly.
        probe_sched = (dict(mode="autolearn", initial_delay=0.1)
                       if len(phases) > 1 else None)
        probe_records = trainer.probe(model, n_trials=int(n_probe),
                                      max_delay=float(probe_max_delay),
                                      sched=probe_sched, collect_states=True)
        if verbose and probe_records:
            reached = max(r["delay"] for r in probe_records)
            last = probe_records[-min(300, len(probe_records)):]
            p = summarise(last)
            print(f"    probe ({len(probe_records)} trials, delay free to "
                  f"{probe_max_delay:.2f}s): reached {reached:.2f}s  "
                  f"last300 engaged {p['p_engaged']:.2f}  "
                  f"correct {p['p_correct']:.2f}  "
                  f"first lick {p['first_lick']:.3f}s  "
                  f"delay {p['delay']:.2f}s", flush=True)
            print(format_delay_report(summarise_by_delay(probe_records),
                                      "probe"), flush=True)


    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}_hist.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.items()}, f)
        torch.save(model.state_dict(), os.path.join(out_dir, f"{name}_model.pt"))
        with open(os.path.join(out_dir, f"{name}_records.json"), "w") as f:
            json.dump({"train": _jsonable(trainer.records),
                       "eval": _jsonable(eval_records),
                       "probe": _jsonable(probe_records),
                       **{f"test_{k}": _jsonable(v) for k, v in
                          test_records.items()}}, f)
        # Hidden states are kept out of the JSON -- (T, 128) per trial is
        # megabytes of text -- but they are the only input to the internals
        # figure, so without this the network panels could not be redrawn
        # without retraining. One ragged npz, trial by trial.
        arrays = {}
        for tag, recs in [("eval", eval_records), ("probe", probe_records),
                          *[(f"test_{k}", v) for k, v in test_records.items()]]:
            for i, r in enumerate(recs):
                for field in ("hidden", "readout", "value"):
                    a = r.get(field)
                    if a is not None:
                        arrays[f"{tag}/{field}/{i}"] = np.asarray(a, np.float32)
        if arrays:
            np.savez_compressed(os.path.join(out_dir, f"{name}_states.npz"),
                                **arrays)
        if plots:
            from .plots import run_report
            run_report(hist, trainer.records, eval_records,
                       probe_records=probe_records, tests=test_records,
                       phases=phase_records if len(phase_records) > 1 else None,
                       boundaries=boundaries,
                       snapshots=getattr(trainer, "weight_snapshots", None),
                       name=name, out_dir=out_dir, clip=trainer.grad_clip)
    return hist, trainer, model


def _jsonable(records):
    """Trial records contain numpy scalars and arrays; json does not."""
    out = []
    for r in records:
        d = {}
        for k, v in r.items():
            # Per-step arrays never go into JSON. 30k trials x ~200 steps of
            # text produced a 379 MB records file; they live in the npz.
            if k in ("hidden", "readout", "value"):
                continue
            if isinstance(v, np.ndarray):
                d[k] = [float(x) for x in v.ravel()]
            elif isinstance(v, (list, tuple)):
                d[k] = [float(x) for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            else:
                d[k] = v
        out.append(d)
    return out


def _printer(trainer, phase: str = ""):
    """One line per log point, with the two metrics promotion depends on."""
    tag = f"{phase[:6]:>6s} " if phase else ""
    def _on_log(i, s):
        g = trainer.gens[0].scheduler
        cs = g.cue_success_rate()
        print(f"{tag}{i:5d}  ret {s.get('return', float('nan')):7.2f}  "
              f"eng {s['p_engaged']:.2f}  corr {s['p_correct']:.2f}  "
              f"cue {s['p_cue_success']:.2f}  itiLick {s['p_iti_lick']:.2f}  "
              f"nITI {s['n_iti_licks']:5.2f}  H {s['entropy']:.3f}  "
              f"delay {s['delay']:.2f}  gain {s['value_gain']:.2f}"
              f"  stage {g.get_training_stage()}"
              + (f"  cs100 {cs:.2f}" if cs is not None else ""), flush=True)
    return _on_log


def run_all(names=None, *, steps: int = 8000, out_dir: Optional[str] = None,
            **kw) -> Dict[str, Dict[str, list]]:
    # `phases` rides through **kw untouched.
    """Run several variants and print a final comparison table."""
    names = list(names or RUNS)
    print(config_matrix(names))
    hists = {}
    for n in names:
        hists[n], _, _ = run(n, steps=steps, out_dir=out_dir, **kw)
    print("\n" + "-" * 74)
    print(f"{'run':>4}  {'cue_succ':>9}  {'iti_lick':>9}  {'n_iti':>7}  "
          f"{'engaged':>8}  {'correct':>8}  {'H':>6}")
    for n, h in hists.items():
        tail = lambda k: float(np.mean(h[k][-3:])) if h.get(k) else float("nan")
        print(f"{n:>4}  {tail('p_cue_success'):9.2f}  {tail('p_iti_lick'):9.2f}  "
              f"{tail('n_iti_licks'):7.2f}  {tail('p_engaged'):8.2f}  "
              f"{tail('p_correct'):8.2f}  {tail('entropy'):6.3f}")
    print("-" * 74)
    return hists


def config_matrix(names) -> str:
    """Every field that differs across a set of variants, as a table.

    Printed before a sweep runs. Round 6 was void because six variants
    silently carried `activity_penalty = 0.0` while the control carried 20.0,
    and nothing in the output made that visible until the runs were finished.
    A column here that varies when it was not meant to is the confound, seen
    before the compute is spent.
    """
    names = list(names)
    specs = {n: _resolve(n) for n in names}
    rows = []
    for sec in _SECTIONS:
        for k in sorted(BASE.get(sec, {})):
            vals = [repr(specs[n][sec].get(k)) for n in names]
            if len(set(vals)) > 1:
                rows.append((f"{sec}.{k}", vals))
    if not rows:
        return "\n[config matrix] all variants resolve to identical configs\n"
    w = max(len(r[0]) for r in rows)
    cw = [max(len(n), *(len(v[i]) for _, v in rows)) for i, n in enumerate(names)]
    L = ["", "[config matrix] fields that differ across the sweep "
             "(everything else is shared)",
         "  " + " " * w + "  " + "  ".join(n.ljust(cw[i])
                                           for i, n in enumerate(names))]
    for k, vals in rows:
        L.append("  " + k.ljust(w) + "  "
                 + "  ".join(v.ljust(cw[i]) for i, v in enumerate(vals)))
    L.append("")
    return "\n".join(L)


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", nargs="?", default="act",
                   help="variant name, 'round2'..'round7', or 'all'")
    p.add_argument("--describe", action="store_true",
                   help="print the generated configuration reference and exit")
    p.add_argument("--steps", type=int, default=8000,
                   help="ceiling on updates; the run usually exits earlier on "
                        "--target-delay or --patience")
    p.add_argument("--target-delay", type=float, default=1.0,
                   help="stop once the batch-mean delay reaches this (0 = off)")
    p.add_argument("--patience", type=int, default=40,
                   help="stop after this many log points with no gain (0 = off)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", default=None)
    p.add_argument("--n-eval", type=int, default=400,
                   help="trials of frozen-delay evaluation after training")
    p.add_argument("--n-probe", type=int, default=10000,
                   help="frozen-weight trials with the curriculum still running")
    p.add_argument("--probe-max-delay", type=float, default=1.8)
    p.add_argument("--n-test", type=int, default=10000,
                   help="frozen-weight trials per test condition "
                        "(fixed 1.0 s, and switching 1.0/1.8 s)")
    p.add_argument("--phases", default=None,
                   help="named training-phase sequence: 'r7' trains the "
                        "curriculum to 1.8s, then AT a fixed 1.8s, then on "
                        "switching 1.0/1.8s blocks -- weights learning "
                        "throughout. Omit for the single-phase curriculum.")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    if a.describe:
        print(describe(a.name if a.name in RUNS else "act"))
        return
    if a.phases and a.phases not in PHASES:
        raise SystemExit(f"--phases must be one of {sorted(PHASES)}")
    ph = PHASES[a.phases] if a.phases else None
    kw = dict(steps=a.steps, log_every=a.log_every, out_dir=a.out,
              n_eval=a.n_eval, n_probe=a.n_probe, n_test=a.n_test,
              probe_max_delay=a.probe_max_delay, plots=not a.no_plots,
              phases=ph, target_delay=a.target_delay or None,
              patience=a.patience or None)
    if a.name in ("all", "round2", "round3", "round4", "round6", "round7"):
        names = {"round2": ROUND2, "round3": ROUND3, "round4": ROUND4,
                 "round6": ROUND6, "round7": ROUND7}.get(a.name)
        run_all(names, **kw)
    else:
        run(a.name, **kw)




# --------------------------------------------------------------------------- #
# The reference table -- GENERATED, never hand-written
# --------------------------------------------------------------------------- #
# Everything below reads the live dataclasses, the live BASE, and the live
# variant deltas. A hand-maintained table of thirty numbers is stale the day
# after it is written, which is exactly the problem it was meant to solve.
#
#   python -m timingtask.variants --describe > docs/timing_config_reference.md

_WHAT: Dict[str, str] = {
    # task
    "dt": "bin width, seconds",
    "cue_duration": "how long the transient cue channel stays at 1",
    "cue_mode": "pulse (transient only) | step (tonic, held to trial end) | "
                "both (Yang et al.: a transient cue AND a tonic step)",
    "answer_window": "seconds after the delay in which a lick is rewarded",
    "post_lick": "wind-down after the decisive lick; no contingency",
    "lick_refractory": "minimum gap between licks (blocks 2 steps)",
    "iti_mean": "stop-licking period, truncated-exponential mean",
    "iti_min": "stop-licking period, lower truncation",
    "iti_max": "stop-licking period, upper truncation",
    "iti_timeout": "abandon a trial stuck in the ITI",
    "iti_restart_on_lick": "AD HOC RULE: resample the whole ITI on any lick",
    "no_cue_prob": "fraction of catch trials (no cue, no timer)",
    "reward": "water, on a correctly timed lick",
    "early_penalty": "lick after the cue but before the delay elapsed",
    "miss_penalty": "answer window expired with no lick",
    "iti_lick_penalty": "each accepted lick in the stop-licking period",
    "no_cue_lick_penalty": "each lick on a catch trial",
    "iti_timeout_penalty": "trial abandoned in the ITI",
    "time_penalty": "per step, while the trial runs",
    "time_penalty_in_post": "AD HOC RULE: charge the step cost during wind-down",
    "reward_rate_gain": "across-trial value gain on reward AND early penalty "
                        "(0 = off)",
    "reward_rate_window": "trials the reward rate is estimated over",
    "reward_rate_ref": "reward rate at which the gain is 1.0",
    "reward_rate_min": "floor on the gain",
    "reward_rate_max": "ceiling on the gain",
    "discount_rate": "reward x exp(-rate * lick_time) (0 = off)",
    "seed": "environment RNG",
    # scheduler
    "mode": "how the required delay is chosen",
    "fixed_delay": "delay in `fixed` mode",
    "cue_association_delay": "delay held during stage 1",
    "cue_association_window": "trials in the promotion window",
    "cue_association_min_trials": "minimum trials before promotion can fire",
    "cue_association_response_window": "a lick within this of the cue counts "
                                       "as a cue response",
    "cue_association_success_threshold": "THRESHOLD: cue-response rate needed "
                                         "to leave stage 1",
    "cue_association_max_iti_lick_hz": "THRESHOLD: ITI lick rate ceiling for "
                                       "promotion (None = criterion off)",
    "initial_delay": "delay at the start of stage 2",
    "delay_step": "how much the delay grows on each promotion",
    "min_delay": "floor on the delay",
    "max_delay": "ceiling on the delay",
    "perf_window": "trials in the stage-2 promotion window",
    "min_trials_per_delay": "minimum trials at a delay before promotion",
    "success_threshold": "THRESHOLD: rewarded fraction needed to raise the delay",
    # trainer
    "n_envs": "parallel environments; one trial each per update",
    "lr": "Adam learning rate",
    "gamma": "return discount across steps",
    "value_coef": "weight on the value loss",
    "grad_clip": "gradient-norm clip (0 = off)",
    "reward_scale": "global rescaling of every environment reward",
    "iti_readout_penalty": "shaped reward: charge P(lick) at every ITI step",
    "activity_penalty": "LOSS TERM: mean ||h||^2 / N",
    "activity_deriv_penalty": "LOSS TERM: mean ||h_t - h_(t-1)||^2 / N",
    "activity_scope": "which steps the activity terms average over",
    "min_action_prob": "eps: floor under every action probability. PER STEP, "
                       "so size it per trial -- at 0.02 it produced a lick on "
                       "99% of trials and became the behaviour (0 = off)",
    # model / run
    "hidden": "recurrent units",
    "tau": "unit time constants, MILLISECONDS: a number = all identical, a "
           "(low, high) pair = log-uniform over that range",
    "train_tau": "learn the time constants too",
    "g": "gain on the recurrent initialisation",
    "rec_init": "gaussian (g/sqrt(N) i.i.d.) or orthogonal (all |eigenvalue| = g)",
    "noise": "private recurrent noise SD",
}


def _rows(d: Dict[str, Any], deltas: Dict[str, Any]):
    out = []
    for k, v in d.items():
        note = _WHAT.get(k, "")
        changed = " **changed**" if k in deltas else ""
        out.append(f"| `{k}` | `{v!r}` | {note}{changed} |")
    return out


def describe(name: str = "act") -> str:
    """Markdown reference for one variant, generated from the live code."""
    import datetime
    spec = _resolve(name)
    deltas = _flat_deltas(name)
    chain = _chain(name)
    L = [f"# Timing task — configuration reference (`{name}`)", "",
         (f"Inheritance: `" + "` -> `".join(["BASE"] + chain) + "`.") if
         len(chain) > 1 else "Delta from `BASE`.", "",
         f"Generated {datetime.date.today().isoformat()} by "
         f"`python -m timingtask.variants --describe`. "
         f"Do not edit by hand; regenerate.", "",
         "Rows marked **changed** differ from `variants.BASE`. "
         "`AD HOC RULE` marks a mechanism with no counterpart in the real task. "
         "`THRESHOLD` marks a promotion criterion. `LOSS TERM` marks something "
         "added to the optimiser's objective rather than to the reward.", ""]
    for title, d in (("Task and reinforcement", spec["task"]),
                     ("Curriculum", spec["sched"]),
                     ("Optimiser and loss terms", spec["trainer"]),
                     ("Network", spec["model"]),
                     ("Observations", spec["obs"])):
        L += [f"## {title}", "", "| parameter | value | what it does |",
              "|---|---|---|", *_rows(d, deltas), ""]
    L += ["## Run control (CLI defaults)", "",
          "| flag | default | what it does |", "|---|---|---|",
          "| `--steps` | 8000 | ceiling on updates |",
          "| `--target-delay` | 1.0 | stop once the batch-mean delay reaches it |",
          "| `--patience` | 40 | stop after this many log points with no gain |",
          "| `--log-every` | 25 | updates between log lines |",
          "| `--n-eval` | 400 | frozen-delay trials (source of the internals figure) |",
          "| `--n-probe` | 10000 | frozen weights, curriculum still running |",
          "| `--n-test` | 10000 | frozen weights, per test condition |",
          "| `--probe-max-delay` | 1.8 | ceiling for the probe |", "",
          "## Frozen-weight test conditions", "",
          "| name | schedule |", "|---|---|",
          *[f"| `{k}` | `{v!r}` |" for k, v in TESTS.items()], "",
          "## Variants", "", "| name | delta from BASE |", "|---|---|",
          *[f"| `{k}` | `{v!r}` |" if v else f"| `{k}` | (BASE unchanged) |"
            for k, v in RUNS.items()], ""]
    return "\n".join(L)

if __name__ == "__main__":       # pragma: no cover
    _cli()
