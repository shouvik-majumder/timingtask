"""
timingtask.env — the gymnasium face of the timing task.
====================================================================

A thin wrapper over :class:`~timingtask.generator.TrialGenerator`.
All trial logic lives in the generator; this file only adapts it to the
gymnasium API and fans records out to monitors.

EPISODE = SESSION, NOT TRIAL
----------------------------
The earlier implementation made one episode one trial, while the delay
scheduler, the performance tracker and the previous-trial observations were all
cross-trial state living on the env and surviving ``reset()``. That mismatch was
the source of most of its complexity, and of a live bug where ``reset(seed=...)``
silently wiped autolearn progress.

Here an episode is a SESSION of ``trials_per_episode`` trials, and ``reset()``
means "a new animal": a fresh generator, a fresh scheduler, learning progress
back to zero. Nothing survives a reset, so nothing can be silently destroyed by
one. The task has no terminal state, only a length limit, so episode end is
reported as ``truncated=True`` and never ``terminated=True``.

ACTIONS
-------
``action_mode="discrete"``    Discrete(2), 0 = wait, 1 = lick.
``action_mode="continuous"``  Box((1,)); a lick is emitted when the signal
                              crosses ``lick_threshold``. This is the
                              ramp-to-bound readout of the timing literature and
                              makes the RL face directly comparable to the
                              supervised one.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ObservationConfig, SchedulerConfig, TimingTaskConfig, make_config
from .generator import TrialGenerator
from .monitor import Monitor, MonitorList
from .scheduler import DelayScheduler

__all__ = ["TimingTaskEnv"]


class TimingTaskEnv(gym.Env):
    """Cue-triggered lick-timing task.

    >>> env = TimingTaskEnv(variant="fixed", trials_per_episode=10)
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(0)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 task_config: Optional[TimingTaskConfig] = None,
                 scheduler_config: Optional[SchedulerConfig] = None,
                 obs_config: Optional[ObservationConfig] = None,
                 *,
                 variant: Optional[str] = None,
                 trials_per_episode: int = 100,
                 action_mode: str = "discrete",
                 lick_threshold: float = 0.5,
                 monitors: Optional[Monitor] = None,
                 seed: Optional[int] = None):
        super().__init__()
        if variant is not None:
            if task_config is not None or scheduler_config is not None:
                raise ValueError("pass either `variant` or explicit configs, not both")
            task_config, scheduler_config = make_config(variant)

        self.cfg = task_config or TimingTaskConfig()
        self.scheduler_cfg = scheduler_config or SchedulerConfig()
        self.obs_cfg = obs_config or ObservationConfig()
        if seed is not None:
            self.cfg.seed = int(seed)

        if action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        self.action_mode = action_mode
        self.lick_threshold = float(lick_threshold)
        self.trials_per_episode = int(trials_per_episode)

        self.monitors = monitors if isinstance(monitors, MonitorList) \
            else MonitorList([monitors] if monitors is not None else [])

        self.action_space = (spaces.Discrete(2) if action_mode == "discrete"
                             else spaces.Box(-np.inf, np.inf, (1,), np.float32))
        self._build(self.cfg.seed)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.gen.obs_size,), np.float32)

    # -- construction -----------------------------------------------------
    def _build(self, seed) -> None:
        """A fresh animal: new rng, new scheduler, no learning history."""
        rng = np.random.default_rng(seed)
        self.scheduler = DelayScheduler(self.scheduler_cfg, rng)
        self.gen = TrialGenerator(self.cfg, self.scheduler, self.obs_cfg, rng)
        self._trials_done = 0

    def _to_lick(self, action) -> bool:
        if self.action_mode == "discrete":
            a = int(np.asarray(action).reshape(()))
            if a not in (0, 1):
                raise ValueError(f"invalid action {action!r}; expected 0 or 1")
            return bool(a)
        return bool(float(np.asarray(action).reshape(-1)[0]) > self.lick_threshold)

    # -- gymnasium API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._build(self.cfg.seed if seed is None else int(seed))
        info = {"phase": self.gen.phase, "trial": 0,
                "trials_per_episode": self.trials_per_episode,
                **self.scheduler.status()}
        self.monitors.on_reset(info)
        return self.gen.observe(), info

    def step(self, action):
        res = self.gen.step(self._to_lick(action))
        if res.record is not None:
            self._trials_done += 1
            self.monitors.on_trial(res.record)

        truncated = self._trials_done >= self.trials_per_episode
        info = {**res.info, "phase": res.phase, "cue_on": res.cue_on,
                "trials_done": self._trials_done,
                "training_stage": self.scheduler.get_training_stage()}
        if res.record is not None:
            info["record"] = res.record
        # No terminal state -- only a length limit. Hence truncated, never
        # terminated; an agent must not bootstrap a zero value at episode end.
        return res.obs, res.reward, False, bool(truncated), info

    def close(self):
        self.monitors.close()

    @property
    def delay(self) -> float:
        return self.gen.delay

    @property
    def training_stage(self) -> str:
        return self.scheduler.get_training_stage()
