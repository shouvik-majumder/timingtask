"""
timingtask.scheduler: delay schedules.

``cue_autolearn`` implements the two-stage training protocol of the
behavioural task. In the cue-association stage the delay is held at a minimal
value until the animal responds reliably to the cue. In the delay-training
stage the delay increases by ``delay_step`` whenever the rewarded fraction over
the last ``perf_window`` trials at the current delay exceeds
``success_threshold``. The defaults in :class:`SchedulerConfig` follow the
published protocol (0.1 s initial delay, 0.1 s increments, 30% rewarded over
100 trials).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from .config import SchedulerConfig

__all__ = ["DelayScheduler"]

MODES = ("fixed", "autolearn", "cue_autolearn", "block", "manual", "variable")


class DelayScheduler:
    """Chooses the required delay for each trial according to ``config.mode``.

    The generator calls :meth:`on_trial_end` after every trial and reads the
    delay for the next trial with :meth:`get_current_delay`. Setting
    ``frozen = True`` holds the delay and the training stage fixed.
    """

    def __init__(self, config: SchedulerConfig,
                 rng: Optional[np.random.Generator] = None):
        if config.mode.lower() not in MODES:
            raise ValueError(f"Unknown mode {config.mode!r}. Available: {MODES}")
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.reset()

    # -- attributes read by the generator -----------------------------------
    @property
    def min_delay(self) -> float:
        return float(self.config.min_delay)

    @property
    def max_delay(self) -> float:
        return float(self.config.max_delay)

    @property
    def cue_response_window(self) -> float:
        return float(self.config.cue_association_response_window)

    @property
    def block_index(self) -> int:
        return int(self._block_index)

    def get_current_delay(self) -> float:
        return float(self.current_delay)

    def get_training_stage(self) -> str:
        return str(self.training_stage)

    # -- metrics ------------------------------------------------------------
    def cue_success_rate(self) -> Optional[float]:
        """Fraction of recent cued trials with a lick inside the cue-response
        window."""
        w = self._cue_success_window
        return float(sum(w) / len(w)) if w else None

    def cue_iti_lick_rate(self) -> Optional[float]:
        """ITI licking over the recent window, in licks per second."""
        w = self._cue_iti_window
        if not w:
            return None
        licks = sum(n for n, _ in w)
        secs = sum(t for _, t in w)
        return float(licks / secs) if secs > 0 else None

    def delay_success_rate(self) -> Optional[float]:
        """Rewarded fraction over the recent window at the current delay."""
        w = self._autolearn_window
        return float(sum(w) / len(w)) if w else None

    def cue_metrics_ready(self) -> bool:
        c = self.config
        return (len(self._cue_success_window) >= int(c.cue_association_window)
                and self._cue_trials >= max(1, int(c.cue_association_min_trials)))

    def status(self) -> Dict[str, Any]:
        """Current delay, stage and promotion metrics, for logging."""
        return {"delay": self.get_current_delay(),
                "training_stage": self.get_training_stage(),
                "block_index": self.block_index,
                "trials_at_delay": self._trials_at_delay,
                "delay_success_rate": self.delay_success_rate(),
                "cue_success_rate": self.cue_success_rate(),
                "cue_iti_lick_rate": self.cue_iti_lick_rate(),
                "cue_metrics_ready": self.cue_metrics_ready(),
                "stage_transitioned": self.last_stage_transition}

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        c = self.config
        self.frozen = False
        self.trial_count = 0
        self.last_stage_transition = False
        self._trials_at_delay = 0
        self._autolearn_window: Deque[int] = deque(maxlen=int(c.perf_window))
        self._cue_success_window: Deque[int] = deque(maxlen=int(c.cue_association_window))
        # (n_iti_licks, iti_seconds) per trial
        self._cue_iti_window: Deque[Any] = deque(maxlen=int(c.cue_association_window))
        self._cue_trials = 0
        self._block_index = 0
        self._block_remaining = 0

        mode = c.mode.lower()
        self.training_stage = mode
        if mode == "fixed":
            self.current_delay = c.fixed_delay
        elif mode == "autolearn":
            self.current_delay = c.initial_delay
        elif mode == "cue_autolearn":
            self.current_delay = self._clip(c.cue_association_delay)
            self.training_stage = "cue_association"
        elif mode == "block":
            self.current_delay = self._sample_block_delay()
            self._block_remaining = self._sample_block_length()
        elif mode == "manual":
            self.current_delay = self._manual_delay(0)
        elif mode == "variable":
            self.current_delay = self._sample_variable_delay()

    def _sample_variable_delay(self) -> float:
        """A new delay on every trial, from ``delay_set`` or uniformly from
        ``[min_delay, max_delay]``."""
        c = self.config
        if c.delay_set:
            return float(c.delay_set[int(self.rng.integers(0, len(c.delay_set)))])
        return float(self.rng.uniform(c.min_delay, c.max_delay))

    def _clip(self, v: float) -> float:
        return float(np.clip(v, self.config.min_delay, self.config.max_delay))

    def _sample_block_length(self) -> int:
        c = self.config
        return int(self.rng.integers(c.block_min_trials, c.block_max_trials + 1))

    def _sample_block_delay(self) -> float:
        d = self.config.block_delays
        if not d:
            raise ValueError("block_delays must be non-empty in block mode")
        return float(d[int(self.rng.integers(0, len(d)))])

    def _manual_delay(self, i: int) -> float:
        s = self.config.manual_schedule
        if not s:
            return float(self.config.initial_delay)
        return float(s[i]) if i < len(s) else float(s[-1])

    # -- updates ----------------------------------------------------------
    def _apply_autolearn(self, success: Optional[bool]) -> None:
        c = self.config
        if success is not None:
            self._trials_at_delay += 1
            self._autolearn_window.append(int(bool(success)))
        if (len(self._autolearn_window) >= int(c.perf_window)
                and self._trials_at_delay >= max(1, int(c.min_trials_per_delay))):
            rate = sum(self._autolearn_window) / len(self._autolearn_window)
            if rate > c.success_threshold:
                nxt = self._clip(self.current_delay + c.delay_step)
                if nxt != self.current_delay:
                    self.current_delay = nxt
                    self._trials_at_delay = 0
                    self._autolearn_window.clear()

    def _update_cue_association(self, rec: Dict[str, Any]) -> None:
        if rec.get("no_cue_trial") or rec.get("cue_onset_step") is None:
            return
        self._cue_success_window.append(int(bool(rec.get("cue_success"))))
        self._cue_iti_window.append((int(rec.get("n_iti_licks", 0)),
                                     float(rec.get("iti_steps", 0))
                                     * float(rec.get("dt", 0.02))))
        self._cue_trials += 1

    def on_trial_end(self, success: Optional[bool],
                     record: Optional[Dict[str, Any]] = None) -> float:
        """Update the schedule after a trial. ``success`` is None for trials
        that are not scored (catch trials, ITI timeouts). Returns the delay
        for the next trial."""
        c = self.config
        rec = record or {}
        self.trial_count += 1
        self.last_stage_transition = False
        if getattr(self, "frozen", False):
            return float(self.current_delay)
        mode = c.mode.lower()

        if mode == "fixed":
            self.current_delay = c.fixed_delay

        elif mode == "autolearn":
            self._apply_autolearn(success)

        elif mode == "cue_autolearn":
            if self.training_stage == "cue_association":
                if rec.get("early"):
                    rec = {**rec, "cue_success": False}
                self._update_cue_association(rec)
                cs, iti = self.cue_success_rate(), self.cue_iti_lick_rate()
                iti_ok = (c.cue_association_max_iti_lick_hz is None
                          or (iti is not None
                              and iti <= c.cue_association_max_iti_lick_hz))
                if (cs is not None and self.cue_metrics_ready()
                        and cs >= c.cue_association_success_threshold
                        and iti_ok):
                    self.training_stage = "autolearn"
                    self.current_delay = self._clip(c.cue_association_delay)
                    self._trials_at_delay = 0
                    self._autolearn_window.clear()
                    self.last_stage_transition = True
            else:
                self._apply_autolearn(success)

        elif mode == "block":
            self._block_remaining -= 1
            if self._block_remaining <= 0:
                self._block_index += 1
                self.current_delay = self._sample_block_delay()
                self._block_remaining = self._sample_block_length()

        elif mode == "manual":
            self.current_delay = self._manual_delay(self.trial_count)

        elif mode == "variable":
            self.current_delay = self._sample_variable_delay()

        return self.get_current_delay()
