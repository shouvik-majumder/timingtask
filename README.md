# timingtask

A cue-triggered lick-timing task for training and evaluating recurrent
agents, with a gymnasium environment, a batched supervised interface,
reinforcement-learning and supervised trainers, and HDF5 export of
trained-agent states and behaviour.

The task is adapted from the flexible lick-timing paradigm used in
Yang et al. (2025) and Majumder et al. (2026). Head-fixed mice are trained to
withhold licking after an auditory cue for an unsignalled delay and are
rewarded for licking within an answer window after the delay. The delay is
learned through a two-stage curriculum and is never signalled on the trial;
it must be inferred from the outcomes of previous trials.

## Task

### Trial structure

```
 stop-licking period     cue      delay        answer window      post-lick
|----------------------|=======|------------|-------------------|-----------|
trial start            cue onset  delay      window              decisive
                                  elapses    expires             lick
```

| Epoch | Duration | Licking |
|---|---|---|
| Stop-licking period (ITI) | truncated exponential; optionally restarted by any lick | penalised |
| Delay | set by the schedule, measured from cue onset | ends the trial unrewarded (`early`) |
| Answer window | `answer_window` after the delay | first lick is rewarded (`rewarded`); no lick is a `miss` |
| Post-lick | `post_lick` after the decisive lick | no contingency |

The cue is an input channel that is high for `cue_duration` from cue onset.
Its duration is independent of the delay, so a lick after the delay elapses
but while the cue is still on is rewarded. On catch trials (`no_cue_prob`) no
cue is presented and no timer starts; licks are penalised.

### Observation

| Channel | Description |
|---|---|
| `cue` | transient cue, 1 while the cue is on |
| `cue_step` | tonic cue, 1 from cue onset to trial end (`cue_mode="step"` or `"both"`) |
| `reward_t-k` | total reward on the k-th previous trial |
| `action_t-k` | 1 if the k-th previous trial contained a decisive lick |
| `success_t-k` | +1 rewarded, -1 responded and unrewarded, 0 no response |
| `first_lick_t-k` | first-lick latency on the k-th previous trial (s) |

The previous-trial channels are held constant for the whole trial and are
repeated for `k = 1 ... n_lags`.

### Actions and reward

The action at each step is binary (wait or lick). A refractory period of
`lick_refractory` seconds limits the lick rate. The reward per step is the
sum of a per-step cost and the event-dependent terms below.

| Event | Parameter |
|---|---|
| Lick inside the answer window | `reward` |
| Lick during the delay | `early_penalty` |
| Answer window expires with no lick | `miss_penalty` |
| Lick during the stop-licking period | `iti_lick_penalty` |
| Lick on a catch trial | `no_cue_lick_penalty` |
| Stop-licking period exceeds `iti_timeout` | `iti_timeout_penalty` |
| Every step while the trial runs | `time_penalty` |

An optional across-trial gain (`reward_rate_gain`) scales the reward and
the early-lick penalty by a function of the recent reward rate.

### Delay schedules

| Mode | Description |
|---|---|
| `fixed` | constant delay |
| `variable` | a new delay on every trial, from a set or a uniform range |
| `block` | delays held for a random number of trials per block |
| `autolearn` | delay increases by `delay_step` when the rewarded fraction over `perf_window` trials exceeds `success_threshold` |
| `cue_autolearn` | the published protocol: a cue-association stage at a minimal delay, then `autolearn` |
| `manual` | an explicit per-trial sequence |

The `cue_autolearn` defaults follow the behavioural protocol: 0.1 s initial
delay, 0.1 s increments, promotion at 30% rewarded over the last 100 trials.

## Installation

```bash
pip install -e .            # core: numpy, torch, h5py
pip install -e '.[full]'    # adds gymnasium, matplotlib, scikit-learn
```

Python 3.10 or later. The `gym` extra is required for `TimingTaskEnv`, the
`plots` extra for `timingtask.plots`.

## Usage

### gymnasium environment

```python
from timingtask import TimingTaskConfig, SchedulerConfig
from timingtask.env import TimingTaskEnv

env = TimingTaskEnv(TimingTaskConfig(), SchedulerConfig(mode="fixed", fixed_delay=1.0),
                    trials_per_episode=100)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)   # 0 = wait, 1 = lick
```

An episode is a session of `trials_per_episode` trials. Episode end is
reported as `truncated`; the task has no terminal state. Completed trials
are delivered as records through `info["record"]` and through any attached
`Monitor`.

### Reinforcement learning

```python
import torch
from timingtask import TimingTaskConfig, SchedulerConfig, ObservationConfig
from timingtask.models import VanillaRNN
from timingtask.rl import ActorCritic, RLTrainer
from timingtask.export import save_trajectory

task = TimingTaskConfig(answer_window=0.8, iti_mean=3.0, iti_min=2.0, iti_max=5.0,
                        iti_restart_on_lick=False, reward=15.0, early_penalty=-2.0,
                        iti_lick_penalty=-2.0, no_cue_lick_penalty=-2.0)
sched = SchedulerConfig(mode="cue_autolearn", cue_association_success_threshold=0.3)

trainer = RLTrainer(task, sched, ObservationConfig(), n_envs=16, lr=4e-3,
                    activity_penalty=20.0)
core = VanillaRNN(trainer.obs_size, 128, 1, tau=100.0, dt=20.0, noise=0.05)
model = ActorCritic(core)

hist = trainer.train(model, steps=4000, target_delay=1.0, patience=40)

records = trainer.evaluate(model, n_trials=400, collect_states=True)
save_trajectory(records, "runs/agent.h5", condition="delay",
                W=model.core.rec.weight.detach().numpy())
```

The agent is a leaky tanh recurrent network with a policy head and a value
head, trained by REINFORCE with a learned baseline (Song, Yang and Wang,
2017). No target lick time enters the objective. `docs/formulation.tex`
gives the full specification of one training update.

### Supervised training

```python
from timingtask import TimingTaskConfig, SchedulerConfig
from timingtask.supervised import TimingTask
from timingtask.models import make_model
from timingtask.training import train

task = TimingTask(TimingTaskConfig(answer_window=2.0, post_lick=0.0),
                  SchedulerConfig(mode="variable", min_delay=0.4, max_delay=1.2))
model = make_model("vanilla", task.spec, hidden_size=128, dt=task.dt)
train(model, task, steps=2000)
```

The supervised task uses a fixed epoch structure with a withhold output and a
ramp-to-threshold lick output, following the convention of Yang et al.
(2019).

### Command line

```bash
python -m timingtask.variants act --out runs/
python -m timingtask.variants act --phases staged --out runs/
python -m timingtask.variants --describe > docs/configuration.md
```

`timingtask.variants` holds named configurations as deltas from a base
configuration. A run writes the training history, the model weights, every
trial record, the per-step time series of the evaluation trials and the
diagnostic figures to the output directory. `docs/configuration.md` lists
every parameter of the default configuration.

Example scripts are in `examples/`.

## Outputs

### Trial records

Every completed trial produces a flat dict with, among other fields,
`outcome`, `delay`, `cue_onset_step`, `first_lick_s`, `decisive_lick_s`,
`lick_times_s`, `n_iti_licks`, `trial_reward`, `trial_steps`,
`training_stage` and the previous-trial channels the agent observed.
Records can be kept in memory (`MemoryMonitor`), appended to a JSONL file
(`JSONLMonitor`) or converted to columnar arrays (`records_to_arrays`).

With `collect_states=True`, `RLTrainer.evaluate`, `test` and `probe`
attach the per-step hidden state, observation, policy readout and value
estimate to each record.

### HDF5 export

`timingtask.export.save_trajectory` writes records with attached states to a
single HDF5 file:

| Dataset / attribute | Shape | Content |
|---|---|---|
| `X` | `(n_trials, T, N)` | hidden states, NaN-padded |
| `time` | `(T,)` | time in seconds relative to cue onset |
| `inputs` | `(n_trials, T, n_in)` | observation vectors |
| `outputs` | `(n_trials, T, 2)` | policy readout (lick minus wait logit) and value |
| `condition` | `(n_trials,)` | per-trial label (default: required delay) |
| `W` | `(N, N)` | recurrent weights, if given |
| `aux/<key>` | `(n_trials, ...)` | per-trial behaviour; `aux/n_steps` is the number of valid steps |
| `dt`, `tau`, `meta` | attrs | step size, time constant, JSON metadata |

Trials are aligned to cue onset by default (`align="start"` aligns to trial
start). Catch trials are aligned to the step at which the cue would have
occurred.

## Package layout

```
timingtask/
  config.py      TimingTaskConfig, SchedulerConfig, ObservationConfig
  generator.py   trial state machine
  scheduler.py   delay schedules and the training curriculum
  monitor.py     trial-record logging
  env.py         gymnasium environment
  models.py      VanillaRNN, GRUModel, LSTMModel
  contract.py    TaskSpec, TrialBatch, Task
  rl.py          ActorCritic, RLTrainer, rollout, behavioural summaries
  training.py    supervised trainer
  supervised.py  TimingTask (supervised interface)
  variants.py    named configurations and command-line interface
  plots.py       behavioural, network and training figures
  circuits.py    published low-dimensional timing circuits
  export.py      HDF5 export
tests/           pytest suite (`-m "not slow"` skips the training test)
examples/        train_rl.py, train_supervised.py, inspect_task.py
docs/            formulation.tex, configuration.md
```

## Reference circuits

`timingtask.circuits` implements the candidate circuit models of Yang et al.
(2025, Extended Data Fig. 1) from the released connectivity matrices, and the
two-attractor model of Majumder et al. (2026, Fig. 4). `classify` reports the
dynamical class of each circuit from its Jacobian spectrum and
`perturbation_signature` reproduces the published perturbation protocols.

## References

- Yang, Z., Inagaki, M., Gerfen, C. R., Fontolan, L. and Inagaki, H. K.
  (2025). Integrator dynamics in the cortico-basal ganglia loop for flexible
  motor timing. *Nature* 649, 1244-1253.
  https://doi.org/10.1038/s41586-025-09778-2
- Majumder, S., Hirokawa, K., Yang, Z., Jain, A., Paletzki, R., Gerfen, C. R.,
  Fontolan, L., Romani, S., Yasuda, R. and Inagaki, H. K. (2026).
  Complementary roles of cell-type-specific plasticity in shaping neocortical
  dynamics for learning action timing. *Nature Communications* 17, 8353.
  https://doi.org/10.1038/s41467-026-74869-1
- Song, H. F., Yang, G. R. and Wang, X.-J. (2017). Reward-based training of
  recurrent neural networks for cognitive and value-based tasks. *eLife* 6,
  e21492. https://doi.org/10.7554/eLife.21492
- Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T. and Wang, X.-J.
  (2019). Task representations in neural networks trained to perform many
  cognitive tasks. *Nature Neuroscience* 22, 297-306.
  https://doi.org/10.1038/s41593-018-0310-2

## License

MIT. See [LICENSE](LICENSE).
