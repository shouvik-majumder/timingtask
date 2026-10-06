"""
timingtask.config: configuration for the cue-triggered lick-timing task.

Three dataclasses hold every numerical setting. Trial logic lives in
``generator.py`` and does not depend on any particular value.

All durations are in seconds. ``TimingTaskConfig.steps`` converts a duration to
the integer step count the generator operates on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["TimingTaskConfig", "SchedulerConfig", "ObservationConfig",
           "VARIANTS", "make_config"]


@dataclass
class TimingTaskConfig:
    """Trial structure and reinforcement.

    A trial consists of a stop-licking period of random duration, a cue that
    starts a timer, a required delay measured from cue onset, and an answer
    window. A lick before the delay elapses ends the trial unrewarded; the
    first lick inside the answer window is rewarded.
    """

    dt: float = 0.02

    # --- cue -------------------------------------------------------------
    # The cue is an observable input signal, independent of the trial phase.
    # It is high for ``cue_duration`` from cue onset and may outlast the delay,
    # in which case a lick after the delay but during the cue is rewarded.
    cue_duration: float = 0.6

    # How the cue is presented to the agent.
    #   "pulse"  one channel, high for ``cue_duration`` from cue onset.
    #   "step"   one channel, high from cue onset until the trial ends.
    #   "both"   two channels: the transient pulse and the tonic step.
    # Catch trials hold every cue channel at zero.
    cue_mode: str = "pulse"

    # --- timer -----------------------------------------------------------
    answer_window: float = 5.0        # seconds after the delay in which a lick is rewarded
    post_lick: float = 1.5            # trial continues this long after the decisive lick
    lick_refractory: float = 0.05     # minimum interval between accepted licks

    # --- stop-licking period (ITI) ---------------------------------------
    # Drawn from a truncated exponential on every trial, so that cue onset is
    # unpredictable from trial start.
    iti_mean: float = 1.2
    iti_min: float = 0.5
    iti_max: float = 2.5
    iti_timeout: float = 20.0         # abandon a trial that has not left the ITI

    # If True, any lick during the stop-licking period resamples its duration.
    # If False, the period runs to completion regardless of licking.
    iti_restart_on_lick: bool = True

    no_cue_prob: float = 0.1          # fraction of catch trials (no cue, no timer)

    # --- reinforcement ---------------------------------------------------
    reward: float = 10.0              # paid on the first lick inside the answer window
    early_penalty: float = -1.0       # lick before the delay elapses
    miss_penalty: float = -2.0        # answer window expires with no lick
    iti_lick_penalty: float = -0.1    # each accepted lick in the stop-licking period
    no_cue_lick_penalty: float = -0.1 # each accepted lick on a catch trial
    iti_timeout_penalty: float = -10.0
    time_penalty: float = -0.05       # per step while the trial runs

    # Whether the per-step cost also applies during the post-lick period, in
    # which no action has any consequence.
    time_penalty_in_post: bool = False

    # --- across-trial value gain -------------------------------------------
    # A multiplicative gain on the reward and on the early-lick penalty that
    # depends on the recent reward rate:
    #
    #   rate = rewarded fraction over the last ``reward_rate_window`` trials
    #   gain = clip(exp(reward_rate_gain * (reward_rate_ref - rate)),
    #               reward_rate_min, reward_rate_max)
    #
    # The gain is computed per environment and is not observable; it reaches
    # the agent only through the magnitude of the rewards it receives.
    # ``reward_rate_gain = 0`` disables it.
    reward_rate_gain: float = 0.0
    reward_rate_window: int = 100
    reward_rate_ref: float = 0.5
    reward_rate_min: float = 0.5
    reward_rate_max: float = 2.0

    # Exponential discount on the reward: reward * exp(-rate * lick_time).
    # 0 disables it.
    discount_rate: float = 0.0

    seed: Optional[int] = 0

    # --- derived ---------------------------------------------------------
    def steps(self, seconds: float) -> int:
        """Convert a duration in seconds to an integer number of steps."""
        return int(round(float(seconds) / float(self.dt)))


@dataclass
class SchedulerConfig:
    """How the required delay is chosen from trial to trial.

    ``cue_autolearn`` implements the two-stage training protocol of the
    behavioural task: a cue-association stage at a minimal delay, followed by
    delay training in which the delay increases by ``delay_step`` whenever the
    rewarded fraction over a window of recent trials exceeds a criterion.
    """

    mode: str = "fixed"      # fixed | variable | autolearn | cue_autolearn | block | manual

    fixed_delay: float = 0.6
    initial_delay: float = 0.1
    delay_step: float = 0.1
    min_delay: float = 0.1
    max_delay: float = 2.0

    # autolearn promotion: rewarded fraction over ``perf_window`` trials at the
    # current delay must exceed ``success_threshold``.
    perf_window: int = 100
    min_trials_per_delay: int = 100
    success_threshold: float = 0.30

    # cue association: promotion requires a cue-response rate above threshold
    # and, optionally, an ITI lick rate below a ceiling.
    cue_association_delay: float = 0.1
    cue_association_window: int = 100
    cue_association_min_trials: int = 100
    cue_association_response_window: float = 0.6
    cue_association_success_threshold: float = 0.50
    # ITI lick-rate ceiling in licks per second. None disables the criterion.
    cue_association_max_iti_lick_hz: Optional[float] = None

    # variable: a new delay on every trial, drawn from ``delay_set`` or, if
    # empty, uniformly from [min_delay, max_delay].
    delay_set: List[float] = field(default_factory=list)

    # block: delays held constant for a random number of trials per block.
    block_delays: List[float] = field(default_factory=lambda: [1.0, 3.0])
    block_min_trials: int = 50
    block_max_trials: int = 150

    # manual: an explicit per-trial delay sequence (the last value is held).
    manual_schedule: Optional[List[float]] = None


@dataclass
class ObservationConfig:
    """Composition of the observation vector.

    The cue channel(s) are always present. The previous-trial channels report
    the outcome of the preceding trial(s) and are held constant for the whole
    of the current trial. ``n_lags`` sets how many previous trials are
    reported.
    """

    include_prev_reward: bool = True
    include_prev_action: bool = True
    include_prev_trial_success: bool = True
    include_prev_first_lick: bool = True
    n_lags: int = 1
    include_trial_start: bool = False
    include_block_context: bool = False
    include_normalized_delay: bool = False   # reveals the required delay; off by default

    @property
    def per_lag(self) -> int:
        return sum(int(b) for b in (self.include_prev_reward,
                                    self.include_prev_action,
                                    self.include_prev_trial_success,
                                    self.include_prev_first_lick))


# Named task variants. They differ only in the answer window and the delay
# schedule; the trial logic is identical.
VARIANTS = {
    "autolearn": dict(answer_window=3.0, mode="cue_autolearn"),
    "fixed":     dict(answer_window=5.0, mode="fixed"),
    "switching": dict(answer_window=10.0, mode="block"),
}


def make_config(variant: str = "fixed", **overrides):
    """Return ``(TimingTaskConfig, SchedulerConfig)`` for a named variant.

    Keyword overrides are applied to whichever dataclass defines the field.
    """
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
