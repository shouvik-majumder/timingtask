# Timing task — inventory of everything currently implemented

Purpose: one place to survey what the model contains, so terms that are not
earning their place can be removed. Status column is evidence from rounds 1–3,
not opinion.

- **load-bearing** — a run demonstrably depends on it
- **untested** — implemented, never varied, no evidence either way
- **inert** — currently disabled (value 0 or None); costs nothing but exists
- **suspect** — evidence it does nothing, or does harm

---

## 1. Loss terms

Everything the optimiser minimises. `L = L_pi + c_V L_V − beta H + (penalties)`.

| term | coefficient | value in `act` | enters via | status |
|---|---|---|---|---|
| policy gradient `L_pi` | — | — | sampled action × standardised advantage | load-bearing |
| value regression `L_V` | `value_coef` | 0.5 | squared error on the return | load-bearing (baseline; without it advantage is the raw return) |
| ~~entropy bonus~~ | — | **removed** | — | deleted; its force carries the same `p(1−p)` factor as the policy gradient's own score function, so it vanished exactly when needed (`docs/timing_exploration_note.md`) |
| activity `mean‖h‖²/N` | `activity_penalty` | 20.0 (ITI scope) | analytic, differentiable | **load-bearing** — the only term that produced a run that learned and did not collapse |
| activity derivative `mean‖h_t−h_{t−1}‖²/N` | `activity_deriv_penalty` | 0 in `act`; 200 in `dact` | analytic, differentiable | **suspect** alone — `dact` collapsed to never-lick at update 300. Untested in combination |
| ITI readout penalty `−lambda·sg[p_lick]` | `iti_readout_penalty` | 0 in `act`; 1.0 in `combo` | shaped reward, through the return | **suspect** — present in `combo`, `miss`, `bigrew`, `floor`; none learned |

Scope switch `activity_scope ∈ {"iti", "full"}` selects which steps the two
activity terms average over. Round 3 is the test of that switch.

### Coefficient scale at initialisation
Each penalty's contribution to the loss on the first update, for reference when
setting new coefficients (policy loss is O(1) by construction, since the
advantage is standardised):

| term | raw value | × coefficient |
|---|---|---|
| `mean‖h‖²/N`, ITI steps | 0.0091 | 0.183 (×20) |
| `mean‖h‖²/N`, all steps | 0.0367 | 0.183 (×5) |
| `mean‖dh‖²/N`, ITI steps | 0.00114 | 0.228 (×200) |
| `mean‖dh‖²/N`, all steps | 0.00127 | 0.253 (×200) |

---

## 2. Task parameters (`TimingTaskConfig`)

Value column is what `variants.BASE` uses, which differs from the dataclass
default where noted.

| parameter | BASE | default | what it does | status |
|---|---|---|---|---|
| `dt` | 0.02 | 0.02 | bin width, seconds | structural |
| `cue_duration` | 0.6 | 0.6 | how long the cue channel stays at 1 | load-bearing |
| `answer_window` | 0.8 | 5.0 | seconds after the delay in which a lick is rewarded | load-bearing (curriculum: short during cue association) |
| `post_lick` | 1.5 | 1.5 | trial continues this long after the first decisive lick | untested — exists so peri/post-lick epochs are in the data |
| `lick_refractory` | 0.05 | 0.05 | minimum inter-lick interval; blocks 2 steps | load-bearing (a held action would otherwise be one lick per bin) |
| `iti_mean` / `min` / `max` | 3.0 / 2.0 / 5.0 | 1.2 / 0.5 / 2.5 | truncated-exponential stop-licking period | load-bearing; the long version was needed to make ITI licking costly |
| `iti_timeout` | 20.0 | 20.0 | give up on a trial stuck in the ITI | **inert with restart off** — never reached once the counter always decrements |
| `iti_restart_on_lick` | **False** | True | resample the whole ITI on any lick | **load-bearing (as False)** — with True, nothing learned in 400 updates, in any configuration |
| `no_cue_prob` | 0.1 | 0.1 | catch trials, the zero-input control | untested — never varied, but needed for the control condition |
| `reward` | 10.0 | 10.0 | water, before discounting | `bigrew` (100.0) changed nothing |
| `early_penalty` | −1.0 | −1.0 | lick before the delay elapsed | untested |
| `miss_penalty` | −2.0 | −2.0 | answer window expired unlicked | `miss` (−20.0) changed nothing |
| `iti_lick_penalty` | −2.0 | −0.1 | per accepted ITI lick | ambiguous — present at −2.0 in every round-2 run, including the four that failed |
| `no_cue_lick_penalty` | −0.1 | −0.1 | lick on a catch trial | untested |
| `iti_timeout_penalty` | −10.0 | −10.0 | trial abandoned in the ITI | **inert with restart off** |
| `time_penalty` | −0.05 | −0.05 | per step, every phase | **suspect** — with restart off the ITI length is action-independent, so this term has no gradient there. After the cue it *pays* for early licking: an early lick saves up to 0.8 s ≈ 2.0 in step cost against an `early_penalty` of −1.0 |
| ~~`thirst_*`~~ | — | **removed** | across-trial multiplier on the step cost | deleted; never switched on in any run |
| `reward_rate_gain` | 1.0 | 0.0 | across-trial value gain, scaling reward AND early penalty together | **new, untested** |
| `reward_rate_window` | 100 | 100 | trials the reward rate is estimated over | new |
| `reward_rate_ref` | 0.5 | 0.5 | rate at which the gain is 1.0 | new |
| `discount_rate` | 0.5 | 0.0 | `reward × exp(−rate·t)`: pays for licking sooner | untested — introduced with the curriculum, never varied alone |
| `seed` | 0 | 0 | environment RNG | **every result so far is one seed** |

---

## 3. Scheduler parameters (`SchedulerConfig`)

| parameter | BASE | what it does | status |
|---|---|---|---|
| `mode` | `cue_autolearn` | two-stage curriculum: cue association → delay growth | load-bearing |
| `cue_association_delay` | 0.1 | delay held during stage 1 | load-bearing |
| `cue_association_response_window` | 0.6 | a lick within this of cue onset counts as a cue response | load-bearing (defines `cue_success`) |
| `cue_association_window` | 100 | trials in the promotion window | untested |
| `cue_association_min_trials` | 100 | minimum before promotion can fire | untested |
| `cue_association_success_threshold` | 0.30 | cue-response rate needed to promote | load-bearing (relaxed from 0.50) |
| `cue_association_max_iti_lick_hz` | **None** | ITI lick rate ceiling for promotion | **removed from the path** — was a fraction-of-trials measure that saturated at 1.00 and blocked `act` at 0.84 cue success. Now a rate, disabled by default |
| `initial_delay`, `delay_step`, `min_delay`, `max_delay` | 0.1 / 0.1 / 0.1 / 2.0 | stage-2 delay growth | **untested** — no run has ever reached stage 2 |
| `perf_window`, `min_trials_per_delay`, `success_threshold` | 100 / 100 / 0.30 | stage-2 promotion | untested, same reason |
| `block_*`, `manual_schedule`, `delay_set` | — | the `block` / `manual` / `variable` modes | unused in this line of work |

---

## 4. Observation channels (`ObservationConfig`) — 5 inputs

| channel | on | what it carries | status |
|---|---|---|---|
| `cue` | yes | 1 while the cue is audible | load-bearing |
| `reward_t-1` | yes | total reward on the previous trial | untested |
| `action_t-1` | yes | did the previous trial contain a decisive lick | untested |
| `success_t-1` | yes | previous outcome, SIGNED: +1 rewarded, −1 responded and wrong, 0 no decision | untested |
| `first_lick_t-1` | yes | previous first-lick latency, seconds | untested |
| `n_lags` | 1 | how many previous trials are exposed | untested |
| `trial_start`, `block`, `norm_delay` | no | optional extras | inert; `norm_delay` leaks the answer and must stay off |

The four history channels are per-trial constants, not events. They have never
been ablated — nothing yet shows the agent uses them at all, and an ablation is
cheap.

---

## 5. Network and training

| item | value | status |
|---|---|---|
| units `N` | 128 | untested |
| `tau` / `dt` → leak `alpha` | 100 ms / 20 ms → 0.2 | untested |
| noise SD | 0.05, scaled `sqrt(2/alpha)` | untested |
| nonlinearity | `tanh` | load-bearing by assumption; note a negative bias saturates rather than silences, unlike ReLU |
| `h0` | fixed at 0, not trained | structural |
| trained parameters | `W_in, b_in, W_rec, W_pi, b_pi, w_v, b_v` | the cue input weights are learned, so the network can become blind to the cue |
| `n_envs` | 16 | load-bearing — batch size is what determines whether the advantage has any contrast |
| `lr` | 4e-3 (Adam) | untested |
| `gamma` | 1.0 | untested |
| `grad_clip` | 1.0 | untested |
| `reward_scale` | 0.1 | untested |
| `min_action_prob` | 0.02 | **now the exploration mechanism** — replaced the entropy bonus; ablation is `act_nofloor` |

---

## 6. Candidates for removal, in order of confidence

1. **`iti_readout_penalty`** — in four failed runs, absent from the one that
   worked. Test: rerun `act` with it added; if nothing changes, delete.
2. ~~`min_action_prob`~~ — promoted to the exploration mechanism, not a
   deletion candidate.
3. ~~`thirst_*`~~ — deleted.
4. **`iti_timeout` / `iti_timeout_penalty`** — unreachable while the restart
   rule is off. Keep the mechanism as a safety net, drop the penalty from the
   story.
5. **`time_penalty`** — has no gradient during the ITI and pays for early
   licking after the cue. Needs an ablation before it is trusted, not removal
   on sight.
6. ~~`entropy_coef` / `entropy_final`~~ — deleted.

Everything above is a single-seed observation. Nothing here should be deleted on
the strength of one run.
