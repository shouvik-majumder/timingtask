# Timing task — configuration reference (`act`)

Delta from `BASE`.

Generated 2026-09-14 by `python -m timingtask.variants --describe`. Do not edit by hand; regenerate.

Rows marked **changed** differ from `variants.BASE`. `AD HOC RULE` marks a mechanism with no counterpart in the real task. `THRESHOLD` marks a promotion criterion. `LOSS TERM` marks something added to the optimiser's objective rather than to the reward.

## Task and reinforcement

| parameter | value | what it does |
|---|---|---|
| `dt` | `0.02` | bin width, seconds |
| `answer_window` | `0.8` | seconds after the delay in which a lick is rewarded |
| `post_lick` | `1.5` | wind-down after the decisive lick; no contingency |
| `cue_duration` | `0.6` | how long the transient cue channel stays at 1 |
| `cue_mode` | `'pulse'` | pulse (transient only) | step (tonic, held to trial end) | both (Yang et al.: a transient cue AND a tonic step) |
| `iti_mean` | `3.0` | stop-licking period, truncated-exponential mean |
| `iti_min` | `2.0` | stop-licking period, lower truncation |
| `iti_max` | `5.0` | stop-licking period, upper truncation |
| `iti_timeout` | `20.0` | abandon a trial stuck in the ITI |
| `no_cue_prob` | `0.1` | fraction of catch trials (no cue, no timer) |
| `reward` | `15.0` | water, on a correctly timed lick |
| `early_penalty` | `-2.0` | lick after the cue but before the delay elapsed |
| `no_cue_lick_penalty` | `-2.0` | each lick on a catch trial |
| `iti_lick_penalty` | `-2.0` | each accepted lick in the stop-licking period **changed** |
| `miss_penalty` | `-2.0` | answer window expired with no lick |
| `iti_timeout_penalty` | `-10.0` | trial abandoned in the ITI |
| `time_penalty` | `-0.05` | per step, while the trial runs |
| `time_penalty_in_post` | `False` | AD HOC RULE: charge the step cost during wind-down |
| `reward_rate_gain` | `1.0` | across-trial value gain on reward AND early penalty (0 = off) |
| `reward_rate_window` | `100` | trials the reward rate is estimated over |
| `reward_rate_ref` | `0.5` | reward rate at which the gain is 1.0 |
| `discount_rate` | `0.0` | reward x exp(-rate * lick_time) (0 = off) |
| `iti_restart_on_lick` | `False` | AD HOC RULE: resample the whole ITI on any lick **changed** |
| `seed` | `0` | environment RNG |

## Curriculum

| parameter | value | what it does |
|---|---|---|
| `mode` | `'cue_autolearn'` | how the required delay is chosen |
| `cue_association_delay` | `0.1` | delay held during stage 1 |
| `cue_association_response_window` | `0.6` | a lick within this of the cue counts as a cue response |
| `cue_association_window` | `100` | trials in the promotion window |
| `cue_association_min_trials` | `100` | minimum trials before promotion can fire |
| `cue_association_success_threshold` | `0.3` | THRESHOLD: cue-response rate needed to leave stage 1 |
| `cue_association_max_iti_lick_hz` | `None` | THRESHOLD: ITI lick rate ceiling for promotion (None = criterion off) |
| `initial_delay` | `0.1` | delay at the start of stage 2 |
| `delay_step` | `0.1` | how much the delay grows on each promotion |
| `min_delay` | `0.1` | floor on the delay |
| `max_delay` | `2.0` | ceiling on the delay |

## Optimiser and loss terms

| parameter | value | what it does |
|---|---|---|
| `n_envs` | `16` | parallel environments; one trial each per update |
| `lr` | `0.004` | Adam learning rate |
| `gamma` | `1.0` | return discount across steps |
| `reward_scale` | `0.1` | global rescaling of every environment reward |
| `grad_clip` | `1.0` | gradient-norm clip (0 = off) |
| `iti_readout_penalty` | `0.0` | shaped reward: charge P(lick) at every ITI step **changed** |
| `activity_penalty` | `20.0` | LOSS TERM: mean ||h||^2 / N **changed** |
| `activity_deriv_penalty` | `0.0` | LOSS TERM: mean ||h_t - h_(t-1)||^2 / N |
| `activity_scope` | `'iti'` | which steps the activity terms average over |
| `min_action_prob` | `0.0` | eps: floor under every action probability. PER STEP, so size it per trial -- at 0.02 it produced a lick on 99% of trials and became the behaviour (0 = off) |
| `seed` | `0` | environment RNG |

## Network

| parameter | value | what it does |
|---|---|---|
| `hidden` | `128` | recurrent units |
| `tau` | `100.0` | unit time constants, MILLISECONDS: a number = all identical, a (low, high) pair = log-uniform over that range |
| `noise` | `0.05` | private recurrent noise SD |
| `train_tau` | `False` | learn the time constants too |
| `g` | `1.0` | gain on the recurrent initialisation |
| `rec_init` | `'gaussian'` | gaussian (g/sqrt(N) i.i.d.) or orthogonal (all |eigenvalue| = g) |

## Observations

| parameter | value | what it does |
|---|---|---|
| `n_lags` | `1` |  |

## Run control (CLI defaults)

| flag | default | what it does |
|---|---|---|
| `--steps` | 8000 | ceiling on updates |
| `--target-delay` | 1.0 | stop once the batch-mean delay reaches it |
| `--patience` | 40 | stop after this many log points with no gain |
| `--log-every` | 25 | updates between log lines |
| `--n-eval` | 400 | frozen-delay trials (source of the internals figure) |
| `--n-probe` | 10000 | frozen weights, curriculum still running |
| `--n-test` | 10000 | frozen weights, per test condition |
| `--probe-max-delay` | 1.8 | ceiling for the probe |

## Frozen-weight test conditions

| name | schedule |
|---|---|
| `fixed1s` | `{'mode': 'fixed', 'fixed_delay': 1.0}` |
| `fixed18` | `{'mode': 'fixed', 'fixed_delay': 1.8}` |
| `switch` | `{'mode': 'block', 'block_delays': [1.0, 1.8], 'block_min_trials': 50, 'block_max_trials': 150}` |

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
| `act` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_penalty': 20.0}}` |
| `dact` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_deriv_penalty': 200.0}}` |
| `bigrew` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'reward': 100.0}, 'trainer': {'iti_readout_penalty': 1.0}}` |
| `floor` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0}, 'trainer': {'iti_readout_penalty': 1.0, 'min_action_prob': 0.02}}` |
| `act_traintau` | `{'parent': 'act', 'model': {'train_tau': True}}` |
| `act_step` | `{'parent': 'act', 'task': {'cue_mode': 'step'}}` |
| `act_both` | `{'parent': 'act', 'task': {'cue_mode': 'both'}}` |
| `act_all` | `{'parent': 'act', 'task': {'cue_mode': 'both'}, 'model': {'rec_init': 'orthogonal', 'g': 1.0}, 'trainer': {'grad_clip': 0.0}}` |
| `act_g12` | `{'parent': 'act', 'model': {'g': 1.2}}` |
| `act_orth` | `{'parent': 'act', 'model': {'rec_init': 'orthogonal', 'g': 1.0}}` |
| `act_noclip` | `{'parent': 'act', 'trainer': {'grad_clip': 0.0}}` |
| `act_both_noclip` | `{'parent': 'act', 'task': {'cue_mode': 'both'}, 'trainer': {'grad_clip': 0.0}}` |
| `act_dact_iti` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'iti', 'activity_penalty': 20.0, 'activity_deriv_penalty': 200.0}}` |
| `act_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_penalty': 5.0}}` |
| `dact_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_deriv_penalty': 200.0}}` |
| `act_time` | `{'task': {'iti_restart_on_lick': False, 'iti_lick_penalty': -2.0, 'time_penalty': -0.15}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'iti', 'activity_penalty': 20.0}}` |
| `act_dact_full` | `{'task': {}, 'trainer': {'iti_readout_penalty': 0.0, 'activity_scope': 'full', 'activity_penalty': 5.0, 'activity_deriv_penalty': 200.0}}` |

