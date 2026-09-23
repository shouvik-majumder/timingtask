# timingtask — a cue-triggered lick-timing task and the agents that learn it

`timingtask` generates trials, trains agents on them, and writes out what the
agent did and what its units did. It does not analyse the result. That is the
whole design: **train here, analyse in [`neuralgeom`](../neuralgeom)**, and let
the two meet at a file rather than at an import.

```
  timingtask                              neuralgeom
  ──────────                              ──────────
  generator.py   trials                   geometry/   pullback metric, Jacobians
  models.py      agents        ──►  .h5   subspace/   Grassmannian trajectories
  rl.py          REINFORCE   Trajectory   topology/   persistent homology
  training.py    supervised    schema     dynamics/   fixed points, LDS
  export.py      the seam                 data/       Trajectory, the contract
```

`timingtask` imports nothing from `neuralgeom`, and `neuralgeom` imports nothing
from `timingtask`. `tests/test_standalone.py` is the guard rail on that, and it
checks imports nested inside functions too, because both packages live in the
same conda environment and an accidental coupling would work perfectly here and
break for everyone else.

## The science

Train a network that can time, then ask whether its dynamics form a **line
attractor** (Yang et al. 2025 — timing is the integral of a tonic input) or
**two point attractors** (Majumder et al. 2026 — timing is set by the initial
condition). `circuits.py` implements both from their released code, and
`classify()` separates them by counting zero eigenvalues of the Jacobian.

**The task.** A stop-licking period of random length; a cue; a required delay
measured from cue onset; a lick in the answer window earns water. Licking during
the stop-licking period, on a catch trial, or before the delay elapses all cost
the same. Nothing ever tells the network when to lick — there is no target time
in the loss, and the delay may only be inferred from the cue and from four
channels carrying the previous trial's outcome.

**The agent.** One recurrent network, 128 leaky tanh units, whose state is read
out by a policy head (a lick probability per 20 ms step) and a value head.
Trained by REINFORCE with a learned baseline over 16 parallel environments, each
running its own copy of the task and its own curriculum.

See `current-focus.md` for where the science actually stands, and
`docs/timing_rl_formulation.tex` for the equations.

## Install

```bash
pip install -e .                 # core: numpy, torch, h5py
pip install -e '.[full]'         # + matplotlib, scikit-learn, gymnasium
```

The core is deliberately thin — the task generator is a plain numpy state
machine and needs no gym; the trainers need torch; `export.py` needs h5py. The
figures, the PCA in the state plots and the gymnasium `Env` wrapper are extras.

It installs cleanly into the same environment as `neuralgeom` (the dependency
set is a subset of that one, and numpy stays pinned `<2` for exactly that
reason), so both are importable side by side and the handoff needs no second
environment.

## Layout

```
timingtask/
  config.py      three dataclasses; every number, no logic
  generator.py   the trial state machine — the TIMER and the CUE CHANNEL are
                 independent axes, not a sequence of phases
  scheduler.py   the delay curriculum
  monitor.py     JSONL / in-memory trial logging
  env.py         gymnasium wrapper                      (extra: gym)
  models.py      the agent zoo — recurrent cores behind one `step` interface
  contract.py    TaskSpec / TrialBatch / Task, the batched-trial interface
  rl.py          REINFORCE with a learned baseline; ActorCritic; attach_states
  training.py    the task-agnostic supervised trainer
  supervised.py  the task as a batched, supervised Task
  variants.py    named configurations + the CLI
  plots.py       the diagnostic figures                 (extra: plots)
  circuits.py    the two published timing models, re-derived numerically
  export.py      states + behaviour → Trajectory HDF5   ← the seam

tests/     153 tests; `-m "not slow"` skips the one that trains a network
docs/      the formulation, the model inventory, the generated config reference
examples/  timing_debug.py (step-by-step trial inspection and a supervised run, as
           `# %%` cells) and timing_rnn.ipynb (supervised and RL training walkthrough)
```

## Run it

```bash
python -m timingtask.variants act --out runs/
```

Writes the training history, the weights, every trial record, and eight figures:
the optimiser view, the two heads, behaviour during training, trial history and
lick-time distributions, network internals, and three frozen-weight test
conditions (a fixed 1 s delay, a switching 1.0/1.8 s block, and a probe where
the curriculum keeps running while the weights do not change).

```bash
python -m timingtask.variants --describe > docs/timing_config_reference.md
```

Regenerates the configuration reference from the live code.

## Handing a trained agent to the geometry side

```python
from timingtask.export import save_trajectory

recs = trainer.evaluate(model, n_trials=512, collect_states=True)
save_trajectory(recs, "runs/act_probe.h5",
                condition="delay",          # the per-trial label
                W=model.core.rec.weight.detach().cpu().numpy(),
                generator="timingtask.rl")
```

```python
from neuralgeom.data import load_trajectory

traj = load_trajectory("runs/act_probe.h5")
traj.X            # (n_trials, T, N)   recurrent states, cue-aligned
traj.inputs       # (n_trials, T, n_in) the drive the network saw
traj.outputs      # (n_trials, T, 2)   [policy readout, critic value]
traj.condition    # (n_trials,)        the required delay on that trial
traj.aux["n_steps"]   # per-trial length; everything past it is NaN
```

`collect_states=True` is required — without it the rollout is discarded and only
the behavioural summary survives, and `records_to_trajectory` says so rather
than writing an empty file.

Three things the export decides, and why:

- **Trials are padded, not truncated.** Trial length is an *outcome* here (a
  trial ends at the decisive lick), so truncating to the shortest would throw
  away exactly the trials where the agent waited.
- **Padding is NaN, never zero.** Zero is a perfectly good hidden state and
  would otherwise be analysed as one. `aux["n_steps"]` gives the mask.
- **Alignment defaults to the cue.** The stop-licking period is resampled every
  trial and restarts on every lick, so trial start sits at a random distance
  from the cue; averaging in that frame smears everything locked to the timer.
  Catch trials align to where the cue *would* have been, so they stay
  comparable. A trial that never reached a cue is dropped and counted in
  `meta["n_unaligned_dropped"]`.

## Adding to this repository

**A new agent** goes in `models.py` and in `MODELS`. Nothing that trains names a
class — `rl.ActorCritic` wraps whatever core it is handed and `training.train`
takes any model with the same `step` — so nothing else has to change.

**A new training method** goes in its own module next to `rl.py`, and should
attach its per-trial series with `rl.attach_states` (or the same key names) so
`export.py` works on its output unchanged.

**A new rollout path** must call `attach_states`. It used to be written out
three times inline, and two of the three copies quietly omitted the hidden
states, so `collect_states=True` returned nothing usable from `switching_test`
and `probe`.
