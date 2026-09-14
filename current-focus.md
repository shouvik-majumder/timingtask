# Current focus — neuralgeom

> Overwrite this file, never append. It describes the present, not history.
> Lines marked [VERIFY] were inferred by Claude from the repo, not stated by me.

**Task:** Train an RNN by reinforcement learning on the cue-triggered
lick-timing task, get it to hold a delay of ~1 s, then ask whether its dynamics
are a line attractor (Yang et al.) or two point attractors (Majumder et al.).
The task, the agent, the curriculum and the whole diagnostic suite are built and
tested. **Nothing has yet learned to time.**

**Where:** `neuralgeom/tasks/timing/` — 4.9k lines across `config.py` (three
dataclasses), `generator.py` (the trial state machine, no torch), `scheduler.py`
(the curriculum), `rl.py` (REINFORCE with a learned baseline), `plots.py` (the
diagnostic figures), `variants.py` (named configurations + the CLI),
`circuits.py` (the two published models, re-derived numerically), plus `env.py`
(gymnasium wrapper), `supervised.py`, `monitor.py`. 115 timing tests in
`tests/test_timing_*.py`.

**Last thing I did:** Removed the entropy bonus and the thirst mechanism
entirely; made a probability floor the exploration mechanism; made the
previous-outcome channel signed (+1/−1/0); added an across-trial value gain on
reward and early penalty, driven by the recent reward rate.

## Where the science actually is

**Best run: `act` under the round-5 reward structure — a 0.70 s delay at 100%
rewarded**, held from update 525 to 875, then frozen. From update 925 it licks
on every trial and earns nothing, delay pinned at 0.70, for 550 updates.

That stall is **policy saturation, not an incentive problem**: at a 0.70 s delay
a correct lick is worth +13.25 against −3.62 for an early one, a margin of 16.9.
The score function `d log pi(lick)/d Delta = 1 − p` is what dies — at p = 0.99
it is 0.01 — so a certain policy has almost no gradient however bad the outcome,
and any term whose force carries the same p(1−p) factor dies with it. The fix
now in place is a **floor under every action probability**,
`min_action_prob = 0.02`, which keeps `wait` sampleable and carries a score of
~p(1−p)/eps on those samples. See `docs/timing_exploration_note.md`.

**The earlier failure mode, now superseded:** a cue-triggered step in lick
hazard. Measured from the policy readout — the lick-minus-wait logit gap, which
is the whole policy — it sits at −2.45 before the cue, jumps to about −0.4
within 150 ms of cue onset, and holds there until the cue ends at 0.6 s. That
−0.4 is a per-step lick probability of 0.41, so the first lick lands ~70 ms
after the cue whatever the required delay is. In the frozen-weight switching
test the 1.0 s and 1.8 s conditions gave *identical* lick-time distributions,
both with a median of 0.06 s. It follows the cue; it does not time from it.

**Why, and this is the live hypothesis.** At the 0.1 s delay of the
cue-association stage a step IS the optimal policy, and the reward structure
then removed the incentive to change: temporal discounting plus the per-step
cost meant the return difference between solving a 1.8 s trial and licking
immediately had fallen to 0.74, against a batch-to-batch swing in the return of
about 30. The signal that should teach "wait longer" was an order of magnitude
below its own noise. That is now fixed and untested.

## RESOLVED — two reward-structure bugs, both mine

**The post-lick wind-down was billed and trained on.** After the decisive lick
the trial runs 1.5 s more so the recorded data contains a peri-lick epoch;
nothing done there changes any outcome. It was charging the per-step time cost
(a flat −3.75 on every trial containing a lick, and nothing on a missed trial),
and its 75 steps per trial were entering every loss term. Roughly **30% of each
gradient was computed on steps where the action had no consequence.** Now:
`time_penalty_in_post=False`, and two masks — `m_data` (trial running, used for
the return) and `m` (`m_data` and not POST, used for every loss term).

**An annealed coefficient was indexed to `--steps`.** The entropy bonus, since
removed, ran on `β_i = β₀ + (β_f − β₀)·i/steps`, which made the same
configuration at 400 and at 1500 updates two different experiments — their
trajectories separated by update 50. Nothing is scheduled against the run length
any more; keep it that way if a schedule is ever added back.

## Open questions, in the order they block things

1. **Seed variance is unmeasured and everything rests on n = 1.** The same
   configuration has produced cue-success 0.84 and complete failure. Five seeds
   of `act` is ~25 min and no comparison means anything until it is run.
2. **Does the probability floor unstick the 0.70 s freeze?** It is the
   replacement for the entropy bonus and has not been tested on a policy that
   had already learned. `act_nofloor` is the ablation.
3. **Is the 0.1 s delay increment too coarse?** At the policy stochasticity the
   runs sit at, the lick-time spread is 0.2–0.35 s, so one promotion can move
   the criterion past the whole current distribution in a single step.

## Hazards — do not change without thinking

- **`iti_restart_on_lick` must stay False.** With it on, completing one
  stop-licking period needs ~150 consecutive non-lick steps, which has
  probability 7e-46 at the initial lick probability of 0.5. Three
  configurations never reached a single cue in 400 updates.
- **`delay` in the log is a batch mean over 16 independent schedulers**, each of
  which only ever increases. A value of 0.26 is a mixture of environments at 0.2
  and 0.3, not a broken 0.1 s step. `stage` and `cs100` come from environment 0
  alone.
- **`corr` divides by licked trials, `eng` by cued trials.** `eng 0.13,
  corr 1.00` looks like perfection and is the never-lick collapse.
- **Round-1 variants F–J restate their reward values explicitly.** A variant
  defined as a delta from a `BASE` that has since moved stops meaning what it
  meant.
- **`success_t-1` is signed: +1 rewarded, −1 responded and wrong, 0 no
  decision.** Zero is reserved for "no evidence", not for failure. A channel
  that is exactly 0.0 contributes a gradient of exactly `delta_i * 0` to its
  input weights, so under the old 1/0 coding a long unrewarded run froze that
  column of `W_in` completely — the weight did not die, the input did.
- **Never tell the network when to lick.** There is no target time anywhere in
  the loss and there must not be; the delay may only ever be inferred.
- **The across-trial value gain scales `reward` AND `early_penalty` together**
  and is not observable. The agent must infer it from its own outcome history.
- Loss-term averages (`H`, `policy_loss`, `value_loss`) are over loss-eligible
  steps, a set that shrank by 30% when POST was masked out. They are not
  comparable to anything logged before that change.

## Things tried and dropped

The ITI restart rule, temporal discounting, an entropy bonus, an accumulating
thirst multiplier on the step cost, a per-step readout penalty on P(lick), a
larger miss penalty, a larger reward, and the derivative-of-activity penalty
were all tried and none of them helped; the derivative penalty was actively
fatal twice, and the entropy bonus could not do the job it was added for
(`docs/timing_exploration_note.md`). Thirst and the entropy bonus are deleted
from the code; the rest survive as flags in `variants.RUNS` for
reproducibility. Results are in `claude/timing_variant_sweep_results.md`; there
is no reason to re-run them.

## Reference

- `docs/timing_config_reference.md` — **generated**, every parameter, penalty,
  threshold and ad-hoc rule in one table. Regenerate with
  `python -m timingtask.variants --describe > docs/timing_config_reference.md`
- `docs/timing_rl_formulation.tex` — the equations, every symbol defined
- `docs/timing_model_inventory.md` — what each term is for and what evidence
  supports it
- `claude/timing_variant_sweep_results.md` — what was run and what happened

**Machines:** WS1 `D:\dev\neuralgeom`, WS2 `C:\dev\neuralgeom`. The drive letter
is a per-machine accident; nothing in the repo depends on it.

**Data is NOT in the repo.** `data_dir.local` per machine points at
`Z:/Users/Shouvik/Modelling/SampleData`. Verify with
`python -c "from neuralgeom.paths import describe_paths; print(describe_paths())"`

**Untracked but kept on disk** (see `.gitignore`): `HANDOFF.md` is the full API
map. `docs/PROJECTIVE_RESULTS.md` and `docs/PROJECTIVE_ROADMAP.md` hold the
projective-geometry findings and plan.
