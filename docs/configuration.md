# Configuration reference (`act`)

Variant `act` as a delta from `BASE`.

Generated 2026-10-06 by `python -m timingtask.variants --describe`. Regenerate rather than editing by hand.

Rows marked **changed** differ from `variants.BASE`.

## Task and reinforcement

| parameter | value | description |
|---|---|---|
| `dt` | `0.02` | step size, seconds |
| `answer_window` | `0.8` | seconds after the delay in which a lick is rewarded |
| `post_lick` | `1.5` | seconds the trial continues after the decisive lick |
| `cue_duration` | `0.6` | duration of the transient cue channel, seconds |
| `cue_mode` | `'pulse'` | pulse (transient) | step (tonic, held to trial end) | both |
| `iti_mean` | `3.0` | stop-licking period: truncated-exponential mean, seconds |
| `iti_min` | `2.0` | stop-licking period: lower truncation, seconds |
| `iti_max` | `5.0` | stop-licking period: upper truncation, seconds |
| `iti_timeout` | `20.0` | abandon a trial that has not left the stop-licking period |
| `no_cue_prob` | `0.1` | fraction of catch trials (no cue, no timer) |
| `reward` | `15.0` | reward for a lick inside the answer window |
| `early_penalty` | `-2.0` | lick after the cue but before the delay elapsed |
| `no_cue_lick_penalty` | `-2.0` | each accepted lick on a catch trial |
| `iti_lick_penalty` | `-2.0` | each accepted lick in the stop-licking period **changed** |
| `miss_penalty` | `-2.0` | answer window expired with no lick |
| `iti_timeout_penalty` | `-10.0` | trial abandoned in the stop-licking period |
| `time_penalty` | `-0.05` | per step while the trial runs |
| `time_penalty_in_post` | `False` | also charge the step cost during the post-lick period |
| `reward_rate_gain` | `1.0` | across-trial gain on reward and early penalty (0 = off) |
| `reward_rate_window` | `100` | trials over which the reward rate is estimated |
| `reward_rate_ref` | `0.5` | reward rate at which the gain is 1.0 |
| `discount_rate` | `0.0` | reward x exp(-rate * lick_time) (0 = off) |
| `iti_restart_on_lick` | `False` | resample the stop-licking period on any lick **changed** |
| `seed` | `0` | environment RNG seed |

## Delay schedule

| parameter | value | description |
|---|---|---|
| `mode` | `'cue_autolearn'` | delay schedule |
| `cue_association_delay` | `0.1` | delay during the cue-association stage, seconds |
| `cue_association_response_window` | `0.6` | a lick within this many seconds of the cue counts as a cue response |
| `cue_association_window` | `100` | trials in the cue-association promotion window |
| `cue_association_min_trials` | `100` | minimum trials before promotion can occur |
| `cue_association_success_threshold` | `0.3` | cue-response rate required to leave the cue-association stage |
| `cue_association_max_iti_lick_hz` | `None` | ITI lick-rate ceiling for promotion (None = criterion off) |
| `initial_delay` | `0.1` | delay at the start of delay training, seconds |
| `delay_step` | `0.1` | delay increment on each promotion, seconds |
| `min_delay` | `0.1` | lower bound on the delay, seconds |
| `max_delay` | `2.0` | upper bound on the delay, seconds |

## Optimiser and loss terms

| parameter | value | description |
|---|---|---|
| `n_envs` | `16` | parallel environments; one trial each per update |
| `lr` | `0.004` | Adam learning rate |
| `gamma` | `1.0` | return discount across steps |
| `reward_scale` | `0.1` | global multiplier on environment rewards |
| `grad_clip` | `1.0` | gradient-norm clip (0 = off) |
| `iti_readout_penalty` | `0.0` | shaped reward: charge P(lick) at every ITI step **changed** |
| `activity_penalty` | `20.0` | loss term: coefficient on mean ||h||^2 / N **changed** |
| `activity_deriv_penalty` | `0.0` | loss term: coefficient on mean ||h_t - h_(t-1)||^2 / N |
| `activity_scope` | `'iti'` | steps the activity terms average over (iti | full) |
| `min_action_prob` | `0.0` | floor under every action probability, per step (0 = off) |
| `seed` | `0` | environment RNG seed |

## Network

| parameter | value | description |
|---|---|---|
| `hidden` | `128` | recurrent units |
| `tau` | `100.0` | unit time constant, milliseconds; a (low, high) pair gives log-uniform per-unit values |
| `noise` | `0.05` | private recurrent noise SD |
| `train_tau` | `False` | learn the time constants |
| `g` | `1.0` | gain of the recurrent initialisation |
| `rec_init` | `'gaussian'` | gaussian (i.i.d. N(0, g^2/N)) or orthogonal (scaled by g) |

## Observation

| parameter | value | description |
|---|---|---|
| `n_lags` | `1` | number of previous trials reported in the observation |

## Run control (CLI defaults)

| flag | default | description |
|---|---|---|
| `--steps` | 8000 | maximum number of updates |
| `--target-delay` | 1.0 | stop once the batch-mean delay reaches it |
| `--patience` | 40 | stop after this many log points without improvement |
| `--log-every` | 25 | updates between log points |
| `--n-eval` | 400 | evaluation trials at the final delay |
| `--n-probe` | 10000 | probe trials (weights frozen, curriculum advancing) |
| `--n-test` | 10000 | trials per frozen-weight test condition |
| `--probe-max-delay` | 1.8 | delay ceiling for the probe |

## Frozen-weight test conditions

| name | schedule |
|---|---|
| `fixed1s` | `{'mode': 'fixed', 'fixed_delay': 1.0}` |
| `fixed18` | `{'mode': 'fixed', 'fixed_delay': 1.8}` |
| `switch` | `{'mode': 'block', 'block_delays': [1.0, 1.8], 'block_min_trials': 50, 'block_max_trials': 150}` |

## Training phases

| name | phases |
|---|---|
| `staged` | curriculum, fixed18, switch |
| `curriculum` | curriculum |

## Variants

| name | delta from BASE |
|---|---|
| `F` | `{'task': {'iti_restart_on_lick': True, 'iti_lick_penalty': -0.05, 'discount_rate': 0.5, 'reward': 10.0, 'early_penalty': -1.0, 'no_cue_lick_penalty': -0.1, 'time_penalty_in_post': True}}` |
| `G` | `{'task': {'iti_restart_on_lick': True, 'iti_lick_penalty': -2.0, 'discount_rate': 0.5, 'reward': 10.0, 'early_penalty': -1.0, 'no_cue_lick_penalty': -0.1, 'time_penalty_in_post': True}}` |
| `H` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'discount_rate': 0.5, 'reward': 10.0, 'early_penalty': -1.0, 'no_cue_lick_penalty': -0.1, 'time_penalty_in_post': True}}` |
| `I` | `{'task': {'iti_restart_on_lick': True, 'iti_lick_penalty': -0.05, 'discount_rate': 0.5, 'reward': 10.0, 'early_penalty': -1.0, 'no_cue_lick_penalty': -0.1, 'time_penalty_in_post': True}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `J` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -0.05, 'discount_rate': 0.5, 'reward': 10.0, 'early_penalty': -1.0, 'no_cue_lick_penalty': -0.1, 'time_penalty_in_post': True}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `combo` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `miss` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'miss_penalty': -20.0}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `bigrew` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'reward': 100.0}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `floor` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 1.0, 'min_action_prob': 0.02}}` |
| `act` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_penalty': 20.0}}` |
| `dact` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_deriv_penalty': 200.0}}` |
| `act_dact_iti` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'iti', 'activity_penalty': 20.0, 'activity_deriv_penalty': 200.0}}` |
| `act_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_penalty': 5.0}}` |
| `dact_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_deriv_penalty': 200.0}}` |
| `act_dact_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_penalty': 5.0, 'activity_deriv_penalty': 200.0}}` |
| `act_time` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'time_penalty': -0.15}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'iti', 'activity_penalty': 20.0}}` |
| `act_traintau` | `{'parent': 'act', 'model': {'train_tau': True}}` |
| `act_step` | `{'parent': 'act', 'task': {'cue_mode': 'step'}}` |
| `act_both` | `{'parent': 'act', 'task': {'cue_mode': 'both'}}` |
| `act_g12` | `{'parent': 'act', 'model': {'g': 1.2}}` |
| `act_orth` | `{'parent': 'act', 'model': {'rec_init': 'orthogonal', 'g': 1.0}}` |
| `act_noclip` | `{'parent': 'act', 'trainer': {'grad_clip': 0.0}}` |
| `act_both_noclip` | `{'parent': 'act', 'task': {'cue_mode': 'both'}, 'trainer': {'grad_clip': 0.0}}` |
| `act_all` | `{'parent': 'act', 'task': {'cue_mode': 'both'}, 'model': {'rec_init': 'orthogonal', 'g': 1.0}, 'trainer': {'grad_clip': 0.0}}` |
