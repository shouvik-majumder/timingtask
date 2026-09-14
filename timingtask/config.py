"""
timingtask.config — configuration for the cue-triggered timing task.
================================================================================

Three dataclasses, no logic. Values here are conveniences; the trial *logic*
lives in ``generator.py`` and does not depend on any particular number.

Times are in SECONDS throughout. ``TimingTaskConfig.dt`` converts to the integer
step counts the generator actually runs on -- float time accumulated by repeated
subtraction drifts, and phase boundaries then land a step early or late.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["TimingTaskConfig", "SchedulerConfig", "ObservationConfig",
           "VARIANTS", "make_config"]


@dataclass
class TimingTaskConfig:
    """Trial structure and reinforcement.

    The task: the animal must withhold licking for the whole stop-licking
    period or it restarts; a cue then starts a timer; a lick before the delay
    elapses aborts the trial unrewarded; a lick anywhere in the answer window is
    rewarded (binary).
    """

    dt: float = 0.02

    # --- cue -------------------------------------------------------------
    # The cue is an OBSERVABLE SIGNAL, not a phase. It is high for this long
    # from cue onset and overlaps the timer arbitrarily. If the delay is shorter
    # than the cue, a lick after the delay but while the cue is still on IS
    # rewarded. Nothing downstream may assume a cue duration.
    cue_duration: float = 0.6

    # How the cue reaches the network.
    #
    #   "pulse" -- one channel, high for `cue_duration` from cue onset. This was
    #              the only option, and with tau = 100 ms nothing of it survives
    #              into the part of the trial where the delay has to be timed:
    #              a pulse into a leaky unit leaves a transient that is gone in
    #              ~100 ms, so the network free-runs with no input at all.
    #   "step"  -- one channel, high from cue onset until the trial ends. In a
    #              near-integrator a step produces a RAMP, which is the thing a
    #              threshold crossing can time. A constant step carries no
    #              elapsed-time information by itself; its integral does.
    #   "both"  -- two channels, the transient AND the tonic step. This is what
    #              Yang et al. actually use: the transient cue enters ALM unit 1
    #              OFF the integration manifold, while a tonic step held from
    #              cue onset drives the integrator, and the step's AMPLITUDE is
    #              their timing knob (slope/amplitude = 4.035, exactly linear).
    #              Here the amplitude is fixed at 1 and the network sets its own
    #              gain through W_in.
    #
    # Catch trials stay at zero on every cue channel, so the zero-input control
    # is unchanged.
    cue_mode: str = "pulse"

    # --- the timer -------------------------------------------------------
    answer_window: float = 5.0        # 3 s autolearn / 5 s fixed / 10 s switching
    post_lick: float = 1.5            # trial continues after the first lick so
                                      # peri_lick and post_lick epochs exist
    lick_refractory: float = 0.05     # minimum gap between licks in a bout

    # --- stop-licking period (ITI) ---------------------------------------
    # Resampled from a truncated exponential every trial so cue onset is
    # unpredictable. ANY lick restarts it.
    iti_mean: float = 1.2
    iti_min: float = 0.5
    iti_max: float = 2.5
    iti_timeout: float = 20.0         # give up on a trial stuck in ITI

    # If False the stop-licking period runs to completion regardless of licking.
    # The restart rule is what makes an agent that licks freely almost never
    # reach a cue -- so it cannot experience cue->reward, and cannot learn the
    # association it needs in order to stop licking for the right reason.
    # Turning it off breaks that trap at the cost of task fidelity.
    iti_restart_on_lick: bool = True

    no_cue_prob: float = 0.1          # catch trials: the zero-input control

    # --- reinforcement ---------------------------------------------------
    reward: float = 10.0              # binary; paid on the first eligible lick
    early_penalty: float = -1.0       # lick before the delay elapses
    miss_penalty: float = -2.0        # answer window expired with no lick
    iti_lick_penalty: float = -0.1    # lick during the stop-licking period
    no_cue_lick_penalty: float = -0.1
    iti_timeout_penalty: float = -10.0
    time_penalty: float = -0.05       # per STEP, while the trial runs

    # Does the per-step cost apply during the post-lick wind-down?
    # It used to, and that was a perverse incentive: the wind-down is 1.5 s of
    # steps that follow a decisive lick and carry no contingency, so licking
    # cost 3.75 more than NOT licking at the same moment -- the task paid the
    # agent to miss. False charges only the phases where an action still
    # matters.
    time_penalty_in_post: bool = False

    # --- subjective value: an ACROSS-TRIAL gain on reward and on the cost of
    # --- a wasted opportunity ----------------------------------------------
    # A run of failures should make water matter more and make throwing a trial
    # away hurt more; a run of successes should make both matter less. One
    # scalar does both, because they are the same quantity -- what this trial's
    # outcome is worth right now.
    #
    #   rate = rewarded fraction over the last `reward_rate_window` trials
    #   gain = clip(exp(reward_rate_gain * (reward_rate_ref - rate)),
    #               reward_rate_min, reward_rate_max)
    #   water paid       = reward       * gain
    #   early lick costs = early_penalty * gain
    #
    # Per environment, and NOT observable: the animal feels the value of water
    # change but is never told the number, so the agent has to infer it from
    # its own outcome history through the previous-trial channels.
    #
    # reward_rate_gain = 0 disables it. At 1.0 with ref 0.5 the gain runs from
    # 1.65 after a run of failures to 0.61 after a run of successes.
    reward_rate_gain: float = 0.0
    reward_rate_window: int = 100
    reward_rate_ref: float = 0.5
    reward_rate_min: float = 0.5
    reward_rate_max: float = 2.0

    # Exponential discount on the reward: reward * exp(-rate * lick_time).
    # Off by default -- it shapes WHEN the agent licks, which is the dependent
    # variable, so it is an explicit manipulation and never a silent default.
    discount_rate: float = 0.0

    seed: Optional[int] = 0

    # --- derived ---------------------------------------------------------
    def steps(self, seconds: float) -> int:
        return int(round(float(seconds) / float(self.dt)))


@dataclass
class SchedulerConfig:
    """How the required delay is chosen, trial to trial.

    ``cue_autolearn`` reproduces the published protocol: a cue-association stage
    at a minimal delay, promoted to delay-growing autolearn once the animal
    responds to the cue and stops licking in the ITI.
    """

    mode: str = "fixed"      # fixed | variable | autolearn | cue_autolearn | block | manual

    fixed_delay: float = 0.6
    initial_delay: float = 0.1        # autolearn starts at the cue-association delay
    delay_step: float = 0.1
    min_delay: float = 0.1
    max_delay: float = 2.0

    # autolearn: 30% rewarded in the last 100 trials at a given delay -> +0.1 s
    perf_window: int = 100
    min_trials_per_delay: int = 100
    success_threshold: float = 0.30

    # cue association: promote on cue-response rate AND low ITI licking
    cue_association_delay: float = 0.1
    cue_association_window: int = 100
    cue_association_min_trials: int = 100
    cue_association_response_window: float = 0.6
    cue_association_success_threshold: float = 0.50
    # Promotion out of cue association used to require the FRACTION OF TRIALS
    # containing at least one ITI lick to fall below a threshold. That number
    # cannot distinguish 55 ITI licks per trial from 6, so it stayed at 1.00
    # through the entire learning curve of the `act` run and blocked a agent
    # that had reached 0.84 cue success. It is now a RATE in licks per second,
    # and None disables the criterion entirely -- which is the default, since
    # the ITI restart rule it was paired with is also off.
    cue_association_max_iti_lick_hz: Optional[float] = None

    # VARIABLE: a fresh delay every trial. Empty set -> uniform(min, max).
    delay_set: List[float] = field(default_factory=list)

    block_delays: List[float] = field(default_factory=lambda: [1.0, 3.0])
    block_min_trials: int = 50
    block_max_trials: int = 150

    manual_schedule: Optional[List[float]] = None


@dataclass
class ObservationConfig:
    """What the agent sees. The cue channel is always present.

    The previous-trial channels are what let the agent modulate its timing by
    experience. They are held CONSTANT for the whole trial -- a static offset,
    not an event -- so the agent can use them to set a ramp slope from the first
    step. ``n_lags`` extends this to the last K trials; with the history carried
    explicitly in the input, the recurrent state does not have to bridge trials
    and an episode can be a single trial.
    """

    include_prev_reward: bool = True
    include_prev_action: bool = True
    include_prev_trial_success: bool = True
    include_prev_first_lick: bool = True
    n_lags: int = 1                   # how many previous trials to expose
    include_trial_start: bool = False
    include_block_context: bool = False
    include_normalized_delay: bool = False   # leaks the answer; off by default

    @property
    def per_lag(self) -> int:
        return sum(int(b) for b in (self.include_prev_reward,
                                    self.include_prev_action,
                                    self.include_prev_trial_success,
                                    self.include_prev_first_lick))


# Experiment variants. Only durations differ -- the logic is identical.
VARIANTS = {
    "autolearn": dict(answer_window=3.0, mode="cue_autolearn"),
    "fixed":     dict(answer_window=5.0, mode="fixed"),
    "switching": dict(answer_window=10.0, mode="block"),
}


def make_config(variant: str = "fixed", **overrides):
    """Return ``(TimingTaskConfig, SchedulerConfig)`` for a named variant."""
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}. Available: {sorted(VARIANTS)}")
    v = dict(VARIANTS[variant])
    task = TimingTaskConfig(answer_window=v.pop("answer_window"))
    sched = SchedulerConfig(mode=v.pop("mode"))
    for k, val in overrides.items():
        if hasattr(task, k):
            setattr(task, k, val)
        elif hasattr(sched, k):
            setattr(sched, k, val)
        else:
            raise KeyError(f"Unknown config field {k!r}")
    return task, sched
