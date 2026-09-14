"""
timingtask.scheduler — how the required delay is chosen.
=====================================================================

Ported from the earlier ``delay_scheduler.py`` with the private methods the
environment was reaching into promoted to public API.

``cue_autolearn`` reproduces the published training protocol:

    "mice were trained to lick after the cue onset with a minimal delay (0.1 s,
    'cue association'). Second, the delay duration was gradually increased
    ('delay training'; reaching criterion performance, 30% rewarded trials in
    the last 100 trials with a given delay duration, resulted in a delay
    increase of 0.1 s)"

The defaults in ``SchedulerConfig`` match that protocol parameter for parameter.
Reference outcome: lick time reaches 1.36 +/- 0.11 s in 6 days (n = 34 mice).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from .config import SchedulerConfig

__all__ = ["DelayScheduler"]

MODES = ("fixed", "autolearn", "cue_autolearn", "block", "manual", "variable")


class DelayScheduler:
    def __init__(self, config: SchedulerConfig,
                 rng: Optional[np.random.Generator] = None):
        if config.mode.lower() not in MODES:
            raise ValueError(f"Unknown mode {config.mode!r}. Available: {MODES}")
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.reset()

    # -- public attributes the generator reads ----------------------------
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

    # -- promoted from private -------------------------------------------
    def cue_success_rate(self) -> Optional[float]:
        w = self._cue_success_window
        return float(sum(w) / len(w)) if w else None

    def cue_iti_lick_rate(self) -> Optional[float]:
        """ITI licking in LICKS PER SECOND over the recent window.

        A rate, not the fraction of trials containing a lick: the fraction
        saturates at 1.0 and stays there while the actual licking falls by an
        order of magnitude, which is exactly what blocked promotion in the
        `act` run."""
        w = self._cue_iti_window
        if not w:
            return None
        licks = sum(n for n, _ in w)
        secs = sum(t for _, t in w)
        return float(licks / secs) if secs > 0 else None

    def delay_success_rate(self) -> Optional[float]:
        w = self._autolearn_window
        return float(sum(w) / len(w)) if w else None

    def cue_metrics_ready(self) -> bool:
        c = self.config
        return (len(self._cue_success_window) >= int(c.cue_association_window)
                and self._cue_trials >= max(1, int(c.cue_association_min_trials)))

    def status(self) -> Dict[str, Any]:
        """Everything a monitor wants, in one call."""
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
        # (n_iti_licks, iti_seconds) per trial -- a rate needs both
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
        """A fresh delay every trial.

        This is what stops the supervised task being degenerate: with one fixed
        interval the network stores a single waveform and infers nothing. With
        the delay varying, and NOT signalled by any input, the network can at
        best learn the distribution -- which is the honest version of what the
        animal faces, since it is never told the criterion either.
        """
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
        c = self.config
        rec = record or {}
        self.trial_count += 1
        self.last_stage_transition = False
        # Frozen: hold the delay and the stage exactly where they are. Used to
        # evaluate a trained agent at ONE delay -- a raster taken while the
        # curriculum is still moving mixes several tasks in one picture.
        # A flag rather than a mode switch because all 16 environments share one
        # SchedulerConfig object but each is at its own delay.
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
