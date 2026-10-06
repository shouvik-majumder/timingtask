"""
timingtask.generator: the trial state machine.

A step-driven state machine with no dependency on gymnasium or torch. It is
wrapped by :class:`timingtask.env.TimingTaskEnv` for reinforcement learning
and by :class:`timingtask.supervised.TimingTask` for supervised training.

Trial structure
---------------
Two independent processes define a trial:

1. A timer, started at cue onset, which determines reward eligibility:

       ITI  ->  WAITING  ->  ELIGIBLE  ->  (POST | expired)
                lick: early  lick: rewarded

2. The cue channel, high for ``cue_duration`` from cue onset. Its duration is
   independent of the required delay, so a lick after the delay elapses but
   while the cue is still on is rewarded.

The stop-licking period (ITI) is drawn from a truncated exponential on every
trial and, if ``iti_restart_on_lick`` is set, is resampled on every lick, so
that cue onset is unpredictable from trial start.

Time is counted in integer steps of ``dt`` seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .config import ObservationConfig, SchedulerConfig, TimingTaskConfig

__all__ = ["Phase", "StepResult", "TrialGenerator"]


class Phase:
    ITI = "iti"           # stop-licking period
    WAITING = "waiting"   # timer running, before the delay; a lick ends the trial unrewarded
    ELIGIBLE = "eligible" # timer past the delay; a lick is rewarded
    POST = "post"         # after the decisive lick; no further contingency
    NO_CUE = "no_cue"     # catch trial: no cue, no timer


OUTCOMES = ("ongoing", "rewarded", "early", "miss", "iti_timeout", "no_cue_complete")


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    trial_over: bool
    phase: str
    cue_on: bool
    info: Dict[str, Any] = field(default_factory=dict)
    record: Optional[Dict[str, Any]] = None   # set on the step that ends a trial


class TrialGenerator:
    """Step-driven timing-task state machine.

    >>> gen = TrialGenerator(TimingTaskConfig(), DelayScheduler(SchedulerConfig()))
    >>> res = gen.step(lick=False)
    >>> res.phase
    'iti'
    """

    def __init__(self, task: TimingTaskConfig, scheduler,
                 obs_cfg: Optional[ObservationConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = task
        self.scheduler = scheduler
        self.obs_cfg = obs_cfg or ObservationConfig()
        self.rng = rng if rng is not None else np.random.default_rng(task.seed)

        # step-count conversions, done once
        self._cue_steps = task.steps(task.cue_duration)
        self._answer_steps = task.steps(task.answer_window)
        self._post_steps = task.steps(task.post_lick)
        self._refractory_steps = max(1, task.steps(task.lick_refractory))
        self._iti_timeout_steps = task.steps(task.iti_timeout)

        self.trial_index = 0
        # Rewarded / not rewarded, one entry per completed cued trial, read
        # over a window of ``reward_rate_window`` trials.
        self._outcome_history: Deque[int] = deque(
            maxlen=max(1, int(task.reward_rate_window)))
        # Outcomes of the last ``n_lags`` trials, most recent first. Held
        # constant for the whole of the current trial.
        self._n_lags = max(1, int(self.obs_cfg.n_lags))
        self._history: List[Dict[str, float]] = [
            {"reward": 0.0, "action": 0.0, "success": 0.0, "first_lick": 0.0}
            for _ in range(self._n_lags)]
        self._start_trial()

    # -- history helpers --------------------------------------------------
    @property
    def prev_reward(self) -> float:
        return self._history[0]["reward"]

    @property
    def prev_action(self) -> float:
        return self._history[0]["action"]

    @property
    def prev_success(self) -> float:
        return self._history[0]["success"]

    @property
    def prev_first_lick(self) -> float:
        return self._history[0]["first_lick"]

    # -- observation ------------------------------------------------------
    @property
    def obs_size(self) -> int:
        c = self.obs_cfg
        n_cue = 2 if self.cfg.cue_mode == "both" else 1
        return (n_cue + c.per_lag * self._n_lags + sum(int(b) for b in (
            c.include_trial_start, c.include_block_context,
            c.include_normalized_delay)))

    def observation_labels(self) -> List[str]:
        """Names of the observation channels, in order."""
        c = self.obs_cfg
        mode = self.cfg.cue_mode
        out = []
        if mode in ("pulse", "both"):
            out.append("cue")
        if mode in ("step", "both"):
            out.append("cue_step")
        for k in range(self._n_lags):
            sfx = f"_t-{k + 1}"
            if c.include_prev_reward:
                out.append("reward" + sfx)
            if c.include_prev_action:
                out.append("action" + sfx)
            if c.include_prev_trial_success:
                out.append("success" + sfx)
            if c.include_prev_first_lick:
                out.append("first_lick" + sfx)
        for flag, name in ((c.include_trial_start, "trial_start"),
                           (c.include_block_context, "block"),
                           (c.include_normalized_delay, "norm_delay"),
                           ):
            if flag:
                out.append(name)
        return out

    def observe(self) -> np.ndarray:
        """The current observation vector, as float32."""
        c = self.obs_cfg
        mode = self.cfg.cue_mode
        v = []
        if mode in ("pulse", "both"):
            v.append(1.0 if self.cue_on else 0.0)
        if mode in ("step", "both"):
            v.append(1.0 if self.cue_step else 0.0)
        for h in self._history:
            if c.include_prev_reward:
                v.append(h["reward"])
            if c.include_prev_action:
                v.append(h["action"])
            if c.include_prev_trial_success:
                v.append(h["success"])
            if c.include_prev_first_lick:
                v.append(h["first_lick"])
        if c.include_trial_start:
            v.append(1.0 if self.step_index == 0 else 0.0)
        if c.include_block_context:
            v.append(float(np.tanh(getattr(self.scheduler, "block_index", 0) / 10.0)))
        if c.include_normalized_delay:
            lo, hi = self.scheduler.min_delay, self.scheduler.max_delay
            v.append(float(np.clip((self.delay - lo) / max(1e-8, hi - lo), 0, 1)))
        return np.asarray(v, dtype=np.float32)

    # -- across-trial value gain -------------------------------------------
    @property
    def reward_rate(self) -> float:
        """Rewarded fraction over the last ``reward_rate_window`` completed
        cued trials. An empty history reads as ``reward_rate_ref``."""
        h = self._outcome_history
        return (float(sum(h)) / len(h)) if h else float(self.cfg.reward_rate_ref)

    @property
    def value_gain(self) -> float:
        """Multiplicative gain applied to the reward and the early-lick
        penalty, as a function of the recent reward rate. See
        :class:`~timingtask.config.TimingTaskConfig`."""
        c = self.cfg
        if not c.reward_rate_gain:
            return 1.0
        g = float(np.exp(c.reward_rate_gain * (c.reward_rate_ref - self.reward_rate)))
        return float(np.clip(g, c.reward_rate_min, c.reward_rate_max))

    # -- cue channel ----------------------------------------------------------
    @property
    def cue_on(self) -> bool:
        """True while the transient cue is on. Depends only on the time since
        cue onset."""
        if self._cue_onset_step is None:
            return False
        return 0 <= (self.step_index - self._cue_onset_step) < self._cue_steps

    @property
    def cue_step(self) -> bool:
        """True from cue onset until the end of the trial."""
        if self._cue_onset_step is None:
            return False
        return self.step_index >= self._cue_onset_step

    @property
    def timer(self) -> Optional[float]:
        """Seconds since cue onset, or None before the cue."""
        if self._cue_onset_step is None:
            return None
        return (self.step_index - self._cue_onset_step) * self.cfg.dt

    # -- trial lifecycle --------------------------------------------------
    def _sample_iti_steps(self) -> int:
        c = self.cfg
        while True:
            s = float(self.rng.exponential(c.iti_mean))
            if c.iti_min <= s <= c.iti_max:
                return c.steps(s)

    def _start_trial(self) -> None:
        self.phase = Phase.ITI
        self.step_index = 0
        self.delay = float(self.scheduler.get_current_delay())
        self._delay_steps = self.cfg.steps(self.delay)
        self.is_no_cue = bool(self.rng.random() < self.cfg.no_cue_prob)

        self._iti_remaining = self._sample_iti_steps()
        self._iti_elapsed = 0
        self._cue_onset_step: Optional[int] = None
        self._virtual_cue_step: Optional[int] = None
        self._post_remaining = self._post_steps
        self._last_lick_step: Optional[int] = None

        self.lick_steps: List[int] = []     # every accepted lick, in step index
        self.iti_licked = False
        self.trial_reward = 0.0
        self.outcome = "ongoing"
        self.first_lick_s: Optional[float] = None
        self.decisive_lick_s: Optional[float] = None

    def _accept_lick(self) -> bool:
        """Apply the refractory period, so that a sustained lick action counts
        as one lick rather than one per step."""
        if self._last_lick_step is not None and \
                (self.step_index - self._last_lick_step) < self._refractory_steps:
            return False
        self._last_lick_step = self.step_index
        self.lick_steps.append(self.step_index)
        return True

    # -- the step ---------------------------------------------------------
    def step(self, lick: bool) -> StepResult:
        """Advance one step. ``lick`` is the agent's action."""
        cfg = self.cfg
        reward = (cfg.time_penalty
                  if (cfg.time_penalty_in_post or self.phase != Phase.POST)
                  else 0.0)
        licked = self._accept_lick() if lick else False
        trial_over = False
        record = None

        if self.phase == Phase.ITI:
            self._iti_elapsed += 1
            if licked:
                self.iti_licked = True
                reward += cfg.iti_lick_penalty
                if cfg.iti_restart_on_lick:
                    self._iti_remaining = self._sample_iti_steps()
                else:
                    self._iti_remaining -= 1
            else:
                self._iti_remaining -= 1
                if self._iti_remaining <= 0:
                    if self.is_no_cue:
                        self.phase = Phase.NO_CUE
                        # The step at which the cue would have occurred. No
                        # cue is presented and no timer starts; the step is
                        # recorded so that catch trials can be aligned to the
                        # same reference as cued trials.
                        self._virtual_cue_step = self.step_index + 1
                        self._no_cue_remaining = self._delay_steps + self._answer_steps
                    else:
                        self.phase = Phase.WAITING
                        # observe() is called after step_index is incremented,
                        # so the first observed cue step is the next one.
                        self._cue_onset_step = self.step_index + 1
            if self._iti_elapsed >= self._iti_timeout_steps:
                reward += cfg.iti_timeout_penalty
                self.outcome, trial_over = "iti_timeout", True

        elif self.phase == Phase.NO_CUE:
            if licked:
                reward += cfg.no_cue_lick_penalty
            self._no_cue_remaining -= 1
            if self._no_cue_remaining <= 0:
                self.outcome, trial_over = "no_cue_complete", True

        elif self.phase == Phase.WAITING:
            elapsed = self.step_index - self._cue_onset_step
            if licked:
                reward += cfg.early_penalty * self.value_gain
                self.decisive_lick_s = elapsed * cfg.dt
                self.outcome = "early"
                self.phase = Phase.POST
            elif elapsed + 1 >= self._delay_steps:
                self.phase = Phase.ELIGIBLE

        elif self.phase == Phase.ELIGIBLE:
            elapsed = self.step_index - self._cue_onset_step
            if licked:
                t = elapsed * cfg.dt
                r = cfg.reward
                if cfg.discount_rate:
                    r *= float(np.exp(-cfg.discount_rate * t))
                reward += r * self.value_gain
                self.decisive_lick_s = t
                self.outcome = "rewarded"
                self.phase = Phase.POST
            elif elapsed + 1 >= self._delay_steps + self._answer_steps:
                reward += cfg.miss_penalty
                self.outcome, trial_over = "miss", True

        elif self.phase == Phase.POST:
            # The trial continues for ``post_lick`` seconds after the decisive
            # lick so that peri-lick activity and lick bouts are recorded.
            self._post_remaining -= 1
            if self._post_remaining <= 0:
                trial_over = True

        else:                                                # pragma: no cover
            raise RuntimeError(f"unknown phase {self.phase!r}")

        if licked and self._cue_onset_step is not None and self.first_lick_s is None:
            fl = (self.step_index - self._cue_onset_step) * cfg.dt
            if fl >= 0:
                self.first_lick_s = fl

        self.trial_reward += reward
        self.step_index += 1

        if trial_over:
            record = self._finish_trial()

        return StepResult(obs=self.observe(), reward=float(reward),
                          trial_over=trial_over, phase=self.phase,
                          cue_on=self.cue_on,
                          info={"delay": self.delay, "outcome": self.outcome,
                                "trial": self.trial_index,
                                "timer": self.timer,
                                "no_cue": self.is_no_cue},
                          record=record)

    def _finish_trial(self) -> Dict[str, Any]:
        """Build the trial record, update the scheduler and the history
        channels, and start the next trial."""
        cfg = self.cfg
        onset = self._cue_onset_step
        lick_times = [(s - onset) * cfg.dt if onset is not None else s * cfg.dt
                      for s in self.lick_steps]
        rewarded = self.outcome == "rewarded"
        cue_success = (self.first_lick_s is not None and
                       self.first_lick_s <= self.scheduler.cue_response_window
                       ) if onset is not None else None

        # Catch trials are aligned to the step at which the cue would have occurred.
        align = onset if onset is not None else self._virtual_cue_step
        align_licks = ([(s - align) * cfg.dt for s in self.lick_steps]
                       if align is not None else lick_times)

        rec = {
            "trial": self.trial_index,
            "outcome": self.outcome,
            "n_iti_licks": sum(1 for s in self.lick_steps
                               if onset is None or s < onset),
            # Duration of the stop-licking period actually experienced, so
            # that ITI licking can be expressed as a rate.
            "iti_steps": int(onset if onset is not None
                             else (self._virtual_cue_step
                                   if self._virtual_cue_step is not None
                                   else self.step_index)),
            "rewarded": rewarded,
            "early": self.outcome == "early",
            "miss": self.outcome == "miss",
            "delay": self.delay,
            "answer_window": cfg.answer_window,
            "cue_duration": cfg.cue_duration,
            "no_cue_trial": self.is_no_cue,
            "cue_onset_step": onset,
            "align_step": align,
            "lick_times_aligned_s": align_licks,
            "first_lick_s": self.first_lick_s,
            "decisive_lick_s": self.decisive_lick_s,
            "lick_times_s": lick_times,
            "n_licks": len(self.lick_steps),
            "iti_licked": self.iti_licked,
            "cue_success": cue_success,
            "trial_reward": self.trial_reward,
            "trial_steps": self.step_index,
            "trial_duration_s": self.step_index * cfg.dt,
            "reward_rate": float(self.reward_rate),
            "value_gain": float(self.value_gain),
            "training_stage": self.scheduler.get_training_stage(),
            "dt": cfg.dt,
            # The previous-trial channels the agent observed during this
            # trial, copied from the observation history rather than inferred
            # from record order (records from parallel environments interleave).
            "prev_success": (float(self._history[0]["success"])
                             if self._history else None),
            "prev_reward": (float(self._history[0]["reward"])
                            if self._history else None),
            "prev_action": (bool(self._history[0]["action"])
                            if self._history else None),
            "prev_first_lick": (float(self._history[0]["first_lick"])
                                if self._history else None),
        }

        # The scheduler scores cued trials only.
        sched_success = None if self.is_no_cue or self.outcome in (
            "iti_timeout", "no_cue_complete") else rewarded
        self.scheduler.on_trial_end(sched_success, rec)
        rec["next_delay"] = float(self.scheduler.get_current_delay())
        rec["next_training_stage"] = self.scheduler.get_training_stage()
        rec["stage_transitioned"] = bool(self.scheduler.last_stage_transition)

        # Signed outcome channel: +1 rewarded, -1 responded and unrewarded,
        # 0 no response.
        signed = (1.0 if rewarded
                  else (0.0 if self.decisive_lick_s is None else -1.0))
        self._history.insert(0, {
            "reward": float(self.trial_reward),
            "action": 1.0 if self.decisive_lick_s is not None else 0.0,
            "success": signed,
            "first_lick": float(self.first_lick_s or 0.0)})
        del self._history[self._n_lags:]
        # Only cued trials that reached a cue count toward the reward rate.
        if not self.is_no_cue and self.outcome != "iti_timeout":
            self._outcome_history.append(int(bool(rewarded)))
        self.trial_index += 1
        self._start_trial()
        return rec
