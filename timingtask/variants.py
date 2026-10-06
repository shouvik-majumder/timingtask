"""
timingtask.variants: named training configurations and the command line.

A variant is a dict of overrides applied to :class:`TimingTaskConfig`,
:class:`SchedulerConfig`, :class:`ObservationConfig`, the :class:`RLTrainer`
arguments and the network settings. All variants share one generator and one
trainer; they differ only in the values below.

Command line::

    python -m timingtask.variants act --out runs/
    python -m timingtask.variants act --phases staged --out runs/
    python -m timingtask.variants --describe > docs/configuration.md

Python::

    from timingtask.variants import run
    hist, trainer, model = run("act", steps=2000)
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

__all__ = ["BASE", "RUNS", "SWEEPS", "TESTS", "PHASES", "build", "run",
           "run_all", "describe", "config_matrix"]


# --------------------------------------------------------------------------
# BASE: the reference configuration. Every variant is a delta from this.
# --------------------------------------------------------------------------
BASE: Dict[str, Dict[str, Any]] = dict(
    task=dict(
        dt=0.02,
        answer_window=0.8,
        post_lick=1.5,
        cue_duration=0.6,
        cue_mode="pulse",
        iti_mean=3.0, iti_min=2.0, iti_max=5.0,
        iti_timeout=20.0,
        no_cue_prob=0.1,
        # Every lick outside the answer window carries the same penalty,
        # whether in the stop-licking period, on a catch trial or during
        # the delay.
        reward=15.0,
        early_penalty=-2.0,
        no_cue_lick_penalty=-2.0,
        iti_lick_penalty=-2.0,
        miss_penalty=-2.0,
        iti_timeout_penalty=-10.0,
        time_penalty=-0.05,
        time_penalty_in_post=False,
        reward_rate_gain=1.0,
        reward_rate_window=100,
        reward_rate_ref=0.5,
        discount_rate=0.0,
        iti_restart_on_lick=False,
        seed=0,
    ),
    sched=dict(
        mode="cue_autolearn",
        cue_association_delay=0.1,
        cue_association_response_window=0.6,
        cue_association_window=100,
        cue_association_min_trials=100,
        cue_association_success_threshold=0.30,
        cue_association_max_iti_lick_hz=None,
        initial_delay=0.1, delay_step=0.1, min_delay=0.1, max_delay=2.0,
    ),
    obs=dict(n_lags=1),
    trainer=dict(
        n_envs=16, lr=4e-3, gamma=1.0,
        reward_scale=0.1,
        grad_clip=1.0,
        iti_readout_penalty=0.0,
        activity_penalty=0.0,
        activity_deriv_penalty=0.0,
        activity_scope="iti",
        min_action_prob=0.0,
        seed=0,
    ),
    # tau is in milliseconds, matching dt = 20 ms. A (low, high) pair gives
    # log-uniform per-unit time constants.
    model=dict(hidden=128, tau=100.0, noise=0.05, train_tau=False,
               g=1.0, rec_init="gaussian"),
)


# --------------------------------------------------------------------------
# Variants. Each value is a nested dict of deltas from BASE, or from the
# variant named by ``parent``.
# --------------------------------------------------------------------------
RUNS: Dict[str, Dict[str, Any]] = {

    # ---- ITI restart rule and lick penalties ------------------------------
    # These restate the reward values they were defined with, so that they
    # remain fixed if BASE changes.
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

    # ---- reward shaping -----------------------------------------------------
    # Per-lick ITI penalty plus a per-step penalty on the lick probability.
    "combo": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                  trainer=dict(iti_readout_penalty=1.0)),
    # Larger miss penalty.
    "miss": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                           miss_penalty=-20.0),
                 trainer=dict(iti_readout_penalty=1.0)),
    # Larger reward.
    "bigrew": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                             reward=100.0),
                   trainer=dict(iti_readout_penalty=1.0)),
    # ``combo`` with a probability floor under every action.
    "floor": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                  trainer=dict(iti_readout_penalty=1.0, min_action_prob=0.02)),

    # ---- activity regularisation -------------------------------------------
    # Penalty on mean squared hidden activity during the ITI.
    "act": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                trainer=dict(iti_readout_penalty=0.0,
                             activity_penalty=20.0)),
    # Penalty on the one-step change in hidden activity during the ITI.
    "dact": dict(task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0),
                 trainer=dict(iti_readout_penalty=0.0,
                              activity_deriv_penalty=200.0)),
    # Both penalties, ITI only.
    "act_dact_iti": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="iti",
                     activity_penalty=20.0, activity_deriv_penalty=200.0)),
    # Activity penalty over the whole trial. The coefficient is lower than
    # in ``act`` so that the term has the same initial magnitude.
    "act_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_penalty=5.0)),
    # Derivative penalty over the whole trial.
    "dact_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_deriv_penalty=200.0)),
    # Both penalties over the whole trial.
    "act_dact_full": dict(
        task=dict(),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="full",
                     activity_penalty=5.0, activity_deriv_penalty=200.0)),
    # ``act`` with three times the per-step time cost.
    "act_time": dict(
        task=dict(iti_restart_on_lick=False, iti_lick_penalty=-2.0,
                  time_penalty=-0.15),
        trainer=dict(iti_readout_penalty=0.0, activity_scope="iti",
                     activity_penalty=20.0)),

    # ---- network and input, as deltas from ``act`` --------------------------
    "act_traintau": dict(parent="act", model=dict(train_tau=True)),
    # Cue as a tonic step, or as transient plus tonic.
    "act_step": dict(parent="act", task=dict(cue_mode="step")),
    "act_both": dict(parent="act", task=dict(cue_mode="both")),
    # Recurrent initialisation.
    "act_g12": dict(parent="act", model=dict(g=1.2)),
    "act_orth": dict(parent="act", model=dict(rec_init="orthogonal", g=1.0)),
    # Gradient clipping off.
    "act_noclip": dict(parent="act", trainer=dict(grad_clip=0.0)),
    "act_both_noclip": dict(parent="act",
                            task=dict(cue_mode="both"),
                            trainer=dict(grad_clip=0.0)),
    # Transient plus tonic cue, orthogonal initialisation, no clipping.
    "act_all": dict(parent="act",
                    task=dict(cue_mode="both"),
                    model=dict(rec_init="orthogonal", g=1.0),
                    trainer=dict(grad_clip=0.0)),
}

# Named groups of variants for ``run_all`` and the command line.
SWEEPS: Dict[str, list] = {
    "shaping": ["combo", "miss", "act", "dact", "bigrew", "floor"],
    "activity": ["act", "act_dact_iti", "act_full", "dact_full", "act_dact_full"],
    "timecost": ["act", "act_time"],
    "network": ["act", "act_g12", "act_orth", "act_noclip", "act_step",
                "act_both", "act_all"],
    "cue": ["act_both_noclip", "act_all", "act_both"],
}

# --------------------------------------------------------------------------
# Training phases. The weights learn throughout; the delay schedule changes
# between phases and the optimiser state, records and weight snapshots carry
# over.
# --------------------------------------------------------------------------
PHASES: Dict[str, list] = {
    # Curriculum to 1.8 s, then a held 1.8 s delay, then unsignalled blocks of
    # 1.0 s and 1.8 s. The final phase trains the agent where the required
    # delay changes and must be inferred from recent outcomes.
    "staged": [
        dict(name="curriculum", sched=None, steps=12000,
             target_delay=1.8, patience=60, score="delay+correct"),
        dict(name="fixed18", steps=3000, patience=40, score="correct",
             sched=dict(mode="fixed", fixed_delay=1.8)),
        dict(name="switch", steps=6000, patience=80, score="correct",
             sched=dict(mode="block", block_delays=[1.0, 1.8],
                        block_min_trials=50, block_max_trials=150)),
    ],
    # The curriculum alone, to 1.0 s.
    "curriculum": [dict(name="curriculum", sched=None, steps=8000,
                        target_delay=1.0, patience=40,
                        score="delay+correct")],
}

# Frozen-weight test conditions run after training: two held delays and a
# switching schedule.
TESTS: Dict[str, Dict[str, Any]] = {
    "fixed1s": dict(mode="fixed", fixed_delay=1.0),
    "fixed18": dict(mode="fixed", fixed_delay=1.8),
    "switch": dict(mode="block", block_delays=[1.0, 1.8],
                   block_min_trials=50, block_max_trials=150),
}


# Sections of a spec. ``parent`` is the only other key a RUNS entry may have.
_SECTIONS = ("task", "sched", "obs", "trainer", "model")


def _merge(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for section, fields in delta.items():
        if section not in _SECTIONS:
            continue                      # ``parent``, handled by _chain
        out.setdefault(section, {}).update(fields)
    return out


def _chain(name: str) -> list:
    """Ancestry of a variant, oldest first, e.g. ``["act", "act_orth"]``."""
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
    """Build the recurrent core."""
    from .models import VanillaRNN
    return VanillaRNN(input_size, hidden, 1, tau=tau, dt=20.0, noise=noise,
                      train_tau=train_tau, g=g, rec_init=rec_init)


def build(name: str, **overrides):
    """Return ``(trainer, model, spec)`` for a named variant.

    ``overrides`` are flat keyword arguments routed to whichever section
    defines the field, e.g. ``build("act", iti_lick_penalty=-5.0, lr=1e-3)``.
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
    """Train one variant and run the post-training evaluations. Returns
    ``(hist, trainer, model)``.

    With ``out_dir`` set, writes the training history (JSON), the model
    weights, every trial record (JSON), the per-step time series of the
    evaluation, test and probe trials (NPZ), and the diagnostic figures.
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
        # One continuous update axis across phases.
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

    # Evaluation at the delay training ended on, weights frozen.
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

    # Test conditions, weights frozen.
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

    # The two held delays pooled into one table with a shared cut-off.
    if verbose and {"fixed1s", "fixed18"} <= set(test_records):
        rep = summarise_by_delay(test_records["fixed1s"]
                                 + test_records["fixed18"])
        print(format_delay_report(rep, "fixed 1.0s vs fixed 1.8s"), flush=True)

    # Probe: weights frozen, curriculum still advancing.
    probe_records = []
    if n_probe:
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
        # Per-step time series are stored as a ragged NPZ rather than in the
        # JSON records.
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
    """Trial records without per-step arrays, with numpy types converted."""
    out = []
    for r in records:
        d = {}
        for k, v in r.items():
            if k in ("hidden", "readout", "value", "obs"):
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
    """One log line per log point."""
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
    """Run several variants and print a comparison table."""
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
    """Table of every field that differs across a set of variants."""
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
    p = argparse.ArgumentParser(
        description="Train an agent on the cue-triggered lick-timing task.")
    p.add_argument("name", nargs="?", default="act",
                   help="variant name, a sweep name "
                        f"({', '.join(SWEEPS)}), or 'all'")
    p.add_argument("--describe", action="store_true",
                   help="print the configuration reference (Markdown) and exit")
    p.add_argument("--steps", type=int, default=8000,
                   help="maximum number of updates")
    p.add_argument("--target-delay", type=float, default=1.0,
                   help="stop once the batch-mean delay reaches this (0 = off)")
    p.add_argument("--patience", type=int, default=40,
                   help="stop after this many log points without improvement (0 = off)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--n-eval", type=int, default=400,
                   help="evaluation trials at the final delay")
    p.add_argument("--n-probe", type=int, default=10000,
                   help="probe trials (weights frozen, curriculum advancing)")
    p.add_argument("--probe-max-delay", type=float, default=1.8)
    p.add_argument("--n-test", type=int, default=10000,
                   help="trials per frozen-weight test condition")
    p.add_argument("--phases", default=None,
                   help=f"training-phase sequence: one of {sorted(PHASES)}. "
                        "Omit for a single curriculum phase.")
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
    if a.name == "all":
        run_all(None, **kw)
    elif a.name in SWEEPS:
        run_all(SWEEPS[a.name], **kw)
    else:
        run(a.name, **kw)


# --------------------------------------------------------------------------- #
# Configuration reference, generated from the live code
# --------------------------------------------------------------------------- #
#   python -m timingtask.variants --describe > docs/configuration.md

_WHAT: Dict[str, str] = {
    # task
    "dt": "step size, seconds",
    "cue_duration": "duration of the transient cue channel, seconds",
    "cue_mode": "pulse (transient) | step (tonic, held to trial end) | both",
    "answer_window": "seconds after the delay in which a lick is rewarded",
    "post_lick": "seconds the trial continues after the decisive lick",
    "lick_refractory": "minimum interval between accepted licks, seconds",
    "iti_mean": "stop-licking period: truncated-exponential mean, seconds",
    "iti_min": "stop-licking period: lower truncation, seconds",
    "iti_max": "stop-licking period: upper truncation, seconds",
    "iti_timeout": "abandon a trial that has not left the stop-licking period",
    "iti_restart_on_lick": "resample the stop-licking period on any lick",
    "no_cue_prob": "fraction of catch trials (no cue, no timer)",
    "reward": "reward for a lick inside the answer window",
    "early_penalty": "lick after the cue but before the delay elapsed",
    "miss_penalty": "answer window expired with no lick",
    "iti_lick_penalty": "each accepted lick in the stop-licking period",
    "no_cue_lick_penalty": "each accepted lick on a catch trial",
    "iti_timeout_penalty": "trial abandoned in the stop-licking period",
    "time_penalty": "per step while the trial runs",
    "time_penalty_in_post": "also charge the step cost during the post-lick period",
    "reward_rate_gain": "across-trial gain on reward and early penalty (0 = off)",
    "reward_rate_window": "trials over which the reward rate is estimated",
    "reward_rate_ref": "reward rate at which the gain is 1.0",
    "reward_rate_min": "lower clip on the gain",
    "reward_rate_max": "upper clip on the gain",
    "discount_rate": "reward x exp(-rate * lick_time) (0 = off)",
    "seed": "environment RNG seed",
    # scheduler
    "mode": "delay schedule",
    "fixed_delay": "delay in fixed mode, seconds",
    "cue_association_delay": "delay during the cue-association stage, seconds",
    "cue_association_window": "trials in the cue-association promotion window",
    "cue_association_min_trials": "minimum trials before promotion can occur",
    "cue_association_response_window": "a lick within this many seconds of the "
                                       "cue counts as a cue response",
    "cue_association_success_threshold": "cue-response rate required to leave "
                                         "the cue-association stage",
    "cue_association_max_iti_lick_hz": "ITI lick-rate ceiling for promotion "
                                       "(None = criterion off)",
    "initial_delay": "delay at the start of delay training, seconds",
    "delay_step": "delay increment on each promotion, seconds",
    "min_delay": "lower bound on the delay, seconds",
    "max_delay": "upper bound on the delay, seconds",
    "perf_window": "trials in the delay-training promotion window",
    "min_trials_per_delay": "minimum trials at a delay before promotion",
    "success_threshold": "rewarded fraction required to increase the delay",
    # trainer
    "n_envs": "parallel environments; one trial each per update",
    "lr": "Adam learning rate",
    "gamma": "return discount across steps",
    "value_coef": "weight on the value loss",
    "grad_clip": "gradient-norm clip (0 = off)",
    "reward_scale": "global multiplier on environment rewards",
    "iti_readout_penalty": "shaped reward: charge P(lick) at every ITI step",
    "activity_penalty": "loss term: coefficient on mean ||h||^2 / N",
    "activity_deriv_penalty": "loss term: coefficient on mean ||h_t - h_(t-1)||^2 / N",
    "activity_scope": "steps the activity terms average over (iti | full)",
    "min_action_prob": "floor under every action probability, per step (0 = off)",
    # model
    "hidden": "recurrent units",
    "tau": "unit time constant, milliseconds; a (low, high) pair gives "
           "log-uniform per-unit values",
    "train_tau": "learn the time constants",
    "g": "gain of the recurrent initialisation",
    "rec_init": "gaussian (i.i.d. N(0, g^2/N)) or orthogonal (scaled by g)",
    "noise": "private recurrent noise SD",
    # observation
    "n_lags": "number of previous trials reported in the observation",
}


def _rows(d: Dict[str, Any], deltas: Dict[str, Any]):
    out = []
    for k, v in d.items():
        note = _WHAT.get(k, "")
        changed = " **changed**" if k in deltas else ""
        out.append(f"| `{k}` | `{v!r}` | {note}{changed} |")
    return out


def describe(name: str = "act") -> str:
    """Markdown configuration reference for one variant, generated from the
    live dataclasses and variant tables."""
    import datetime
    spec = _resolve(name)
    deltas = _flat_deltas(name)
    chain = _chain(name)
    L = [f"# Configuration reference (`{name}`)", "",
         (f"Inheritance: `" + "` -> `".join(["BASE"] + chain) + "`.") if
         len(chain) > 1 else f"Variant `{name}` as a delta from `BASE`.", "",
         f"Generated {datetime.date.today().isoformat()} by "
         f"`python -m timingtask.variants --describe`. "
         f"Regenerate rather than editing by hand.", "",
         "Rows marked **changed** differ from `variants.BASE`.", ""]
    for title, d in (("Task and reinforcement", spec["task"]),
                     ("Delay schedule", spec["sched"]),
                     ("Optimiser and loss terms", spec["trainer"]),
                     ("Network", spec["model"]),
                     ("Observation", spec["obs"])):
        L += [f"## {title}", "", "| parameter | value | description |",
              "|---|---|---|", *_rows(d, deltas), ""]
    L += ["## Run control (CLI defaults)", "",
          "| flag | default | description |", "|---|---|---|",
          "| `--steps` | 8000 | maximum number of updates |",
          "| `--target-delay` | 1.0 | stop once the batch-mean delay reaches it |",
          "| `--patience` | 40 | stop after this many log points without improvement |",
          "| `--log-every` | 25 | updates between log points |",
          "| `--n-eval` | 400 | evaluation trials at the final delay |",
          "| `--n-probe` | 10000 | probe trials (weights frozen, curriculum advancing) |",
          "| `--n-test` | 10000 | trials per frozen-weight test condition |",
          "| `--probe-max-delay` | 1.8 | delay ceiling for the probe |", "",
          "## Frozen-weight test conditions", "",
          "| name | schedule |", "|---|---|",
          *[f"| `{k}` | `{v!r}` |" for k, v in TESTS.items()], "",
          "## Training phases", "",
          "| name | phases |", "|---|---|",
          *[f"| `{k}` | " + ", ".join(ph["name"] for ph in v) + " |"
            for k, v in PHASES.items()], "",
          "## Variants", "", "| name | delta from BASE |", "|---|---|",
          *[f"| `{k}` | `{v!r}` |" if v else f"| `{k}` | (BASE unchanged) |"
            for k, v in RUNS.items()], ""]
    return "\n".join(L)


if __name__ == "__main__":       # pragma: no cover
    _cli()
