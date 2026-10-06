"""
timingtask.supervised: the timing task as a batched supervised ``Task``.

Wraps the trial generator in the :class:`~timingtask.contract.Task` interface
so that :func:`timingtask.training.train` and
:func:`timingtask.models.make_model` apply directly::

    task  = TimingTask(variant="fixed")
    model = make_model("vanilla", task.spec, hidden_size=256, dt=task.dt)
    train(model, task, steps=4000)

Trial layout
------------
Following the convention of Yang et al. (2019) and related work, each trial
has a fixed epoch structure and nothing acts during training. The generator
is rolled forward with a non-licking observer to lay out the epochs (ITI, cue,
delay, answer window), and every trial runs to the end of its answer window.

Outputs
-------
Two output units:

    unit 0  WITHHOLD   high while licking is not permitted, low afterwards
    unit 1  LICK       ramps from cue onset and crosses ``threshold`` at the
                       target time

Cost mask
---------
The per-step loss weight is zero during a grace period at trial start and
around each epoch transition, ``w_answer`` inside the answer window and 1
elsewhere. The withhold unit is additionally weighted by ``w_withhold``.

Trial-to-trial variability
--------------------------
The ITI duration varies (truncated exponential), catch trials hold the
withhold target throughout, input and recurrent noise are scaled to be
dt-independent, and the target lick time may be jittered. Variation of the
required delay across trials comes from the scheduler (``mode="variable"``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .contract import Task, TaskSpec, TrialBatch
from .config import ObservationConfig, SchedulerConfig, TimingTaskConfig, make_config
from .generator import Phase, TrialGenerator
from .scheduler import DelayScheduler

__all__ = ["TimingTask", "WITHHOLD", "LICK"]

WITHHOLD, LICK = 0, 1          # output unit indices


class TimingTask(Task):
    """The timing task as a supervised, batched ``Task``.

    Parameters
    ----------
    target_offset : target lick time, in seconds after the delay elapses.
    target_jitter : SD of per-trial Gaussian jitter on the target time (s).
    threshold : level the lick unit must cross; the crossing is the lick time.
    plateau : level the lick target settles at after the crossing, as a
        multiple of ``threshold``. Must exceed 1 so that the target continues
        to rise through the threshold.
    hold_level : target of the withhold unit while withholding.
    w_answer, w_withhold : cost-mask weights.
    grace : seconds of zero loss weight after trial start and after each
        epoch transition.
    sigma_x : input noise SD.
    eval_noise : apply input noise in eval mode as well as training mode.
    """

    def __init__(self,
                 task_config: Optional[TimingTaskConfig] = None,
                 scheduler_config: Optional[SchedulerConfig] = None,
                 obs_config: Optional[ObservationConfig] = None,
                 *,
                 variant: Optional[str] = None,
                 target_offset: float = 0.15,
                 target_jitter: float = 0.0,
                 threshold: float = 1.0,
                 plateau: float = 1.5,
                 ramp_exponent: float = 1.0,
                 hold_level: float = 0.8,
                 w_answer: float = 5.0,
                 w_withhold: float = 2.0,
                 grace: float = 0.1,
                 sigma_x: float = 0.01,
                 eval_noise: bool = True,
                 seed: Optional[int] = None):
        if variant is not None:
            if task_config is not None or scheduler_config is not None:
                raise ValueError("pass either `variant` or explicit configs, not both")
            task_config, scheduler_config = make_config(variant)
        cfg = task_config or TimingTaskConfig()
        if plateau <= 1.0:
            raise ValueError("plateau must be > 1 so that the lick target "
                             "continues to rise through the threshold")

        # Task works in milliseconds; the configuration dataclasses use seconds.
        super().__init__(dt=cfg.dt * 1000.0, sigma=sigma_x, seed=seed)

        self.cfg = cfg
        self.scheduler_cfg = scheduler_config or SchedulerConfig()
        self.obs_cfg = obs_config or ObservationConfig()
        self.target_offset = float(target_offset)
        self.target_jitter = float(target_jitter)
        self.threshold = float(threshold)
        self.plateau = float(plateau)
        self.ramp_exponent = float(ramp_exponent)
        self.hold_level = float(hold_level)
        self.w_answer = float(w_answer)
        self.w_withhold = float(w_withhold)
        self.grace = float(grace)
        self.eval_noise = bool(eval_noise)

        self._rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.scheduler = DelayScheduler(self.scheduler_cfg, self._rng)
        self.gen = TrialGenerator(cfg, self.scheduler, self.obs_cfg, self._rng)

        self.spec = TaskSpec(
            name=f"timing:{self.scheduler_cfg.mode}",
            input_dim=self.gen.obs_size,
            output_dim=2,
            loss="mse",
            input_labels=("cue",),
            output_labels=("withhold", "lick"),
            description="cue-triggered lick timing; withhold + ramp-to-threshold",
        )

    # -- rollout ----------------------------------------------------------
    def _roll_trial(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Lay out one trial's epochs with a non-licking observer.

        The trial runs its full course (ITI, cue, delay, answer window). The
        resulting ``miss`` outcome is discarded; only the epoch timing is used.
        """
        obs: List[np.ndarray] = []
        for _ in range(200000):
            obs.append(self.gen.observe())
            res = self.gen.step(False)
            if res.record is not None:
                return np.asarray(obs, dtype=np.float32), res.record
        raise RuntimeError("trial did not complete")

    def _target_and_mask(self, n_steps: int, rec: Dict[str, Any]
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(y (T, 2), w (T,))``: targets and cost-mask weights."""
        dt = self.cfg.dt
        y = np.zeros((n_steps, 2), dtype=np.float32)
        w = np.zeros(n_steps, dtype=np.float32)
        grace_steps = max(1, int(round(self.grace / dt)))
        onset = rec.get("cue_onset_step")

        # WITHHOLD is high from trial start and falls once licking is permitted.
        y[:, WITHHOLD] = self.hold_level
        w[:] = 1.0
        w[:grace_steps] = 0.0

        if onset is None:                        # catch trial: withhold throughout
            return y, w

        t_star = float(rec["delay"]) + self.target_offset
        if self.target_jitter:
            t_star = max(dt, t_star + float(self._rng.normal(0, self.target_jitter)))
        star = max(1, int(round(t_star / dt)))
        idx = np.arange(n_steps)
        elapsed = idx - onset

        # LICK rises from cue onset, crosses ``threshold`` at t*, and continues
        # to ``plateau * threshold``.
        rising = elapsed >= 0
        frac = np.maximum(elapsed[rising], 0) / star
        y[rising, LICK] = np.minimum(
            self.threshold * frac ** self.ramp_exponent,
            self.threshold * self.plateau)
        delay_step = int(round(float(rec["delay"]) / dt))
        y[elapsed >= delay_step, WITHHOLD] = 0.05

        # Weights: the answer window is up-weighted; grace windows around each
        # transition carry zero weight.
        w[elapsed >= delay_step] = self.w_answer
        for edge in (0, delay_step, star):
            m = (elapsed >= edge - grace_steps // 2) & (elapsed < edge + grace_steps)
            w[m] = 0.0
        return y, w

    # -- Task API ---------------------------------------------------------
    def sample(self, batch_size: int) -> TrialBatch:
        seqs, ys, ws, recs = [], [], [], []
        for _ in range(batch_size):
            obs, rec = self._roll_trial()
            y, w = self._target_and_mask(len(obs), rec)
            seqs.append(obs); ys.append(y); ws.append(w); recs.append(rec)

        T = max(len(s) for s in seqs)
        B, n_in = batch_size, self.spec.input_dim
        x = torch.zeros(B, T, n_in)
        y = torch.zeros(B, T, 2)
        weight = torch.zeros(B, T)
        for i, (s, yy, ww) in enumerate(zip(seqs, ys, ws)):
            t = len(s)
            x[i, :t] = torch.from_numpy(s)
            y[i, :t] = torch.from_numpy(yy)
            weight[i, :t] = torch.from_numpy(ww)
        if self.sigma:
            x = x + self._noise(x.shape)

        def col(key, default=np.nan):
            return torch.tensor([float(r.get(key) if r.get(key) is not None
                                       else default) for r in recs])

        meta = {
            "delay": col("delay"),
            "target_time_s": torch.tensor(
                [float(r["delay"]) + self.target_offset for r in recs]),
            "trial_len": torch.tensor([len(s) for s in seqs]),
            "cue_onset_step": col("cue_onset_step", -1),
            "no_cue_trial": torch.tensor([bool(r["no_cue_trial"]) for r in recs]),
            "answer_window": col("answer_window"),
            "weight": weight,                    # the cost mask, as weights
        }
        # loss_mask is boolean for compatibility with masked_loss; the weights
        # are carried in meta and used by ``loss``.
        return TrialBatch(x, y, weight > 0, meta)

    def loss(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Weighted masked MSE using the cost-mask weights and the per-unit
        weight on the withhold unit."""
        w = batch.meta["weight"].to(outputs.device)                 # (B, T)
        unit_w = torch.ones(outputs.shape[-1], device=outputs.device)
        unit_w[WITHHOLD] = self.w_withhold
        se = ((outputs - batch.targets) ** 2 * unit_w).mean(-1)     # (B, T)
        return (se * w).sum() / w.sum().clamp_min(1)

    # -- readout ----------------------------------------------------------
    def crossing_time(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Time of the first threshold crossing of the lick unit after cue
        onset, in seconds. NaN if the unit never crosses."""
        z = outputs[..., LICK]
        B, T = z.shape
        idx = torch.arange(T, device=z.device).unsqueeze(0)
        onset = batch.meta["cue_onset_step"].to(z.device).unsqueeze(1)
        valid = (onset >= 0) & (idx >= onset) & \
                (idx < batch.meta["trial_len"].to(z.device).unsqueeze(1))
        above = (z >= self.threshold) & valid
        any_above = above.any(dim=1)
        first = torch.argmax(above.to(torch.int8), dim=1).to(torch.float32)
        t = (first - onset.squeeze(1).to(torch.float32)) * self.cfg.dt
        return torch.where(any_above, t, torch.full_like(t, float("nan")))

    def outcome_counts(self, outputs: Tensor, batch: TrialBatch) -> Dict[str, Any]:
        """Behavioural summary of a batch.

        ``p_engaged`` (fraction of cued trials with a lick) and ``p_correct``
        (fraction of licked trials that were rewarded) are reported separately,
        since a model that never licks makes no errors.
        """
        t = self.crossing_time(outputs, batch)
        delay = batch.meta["delay"].to(t.device)
        window = batch.meta["answer_window"].to(t.device)
        cued = ~batch.meta["no_cue_trial"].to(t.device)
        licked = cued & ~torch.isnan(t)
        early = licked & (t < delay)
        rew = licked & (t >= delay) & (t <= delay + window)
        miss = cued & ~licked
        n = int(cued.sum())
        catch = batch.meta["no_cue_trial"].to(t.device)
        fa = int((catch & ~torch.isnan(t)).sum())
        return {"rewarded": int(rew.sum()), "early": int(early.sum()),
                "miss": int(miss.sum()), "n_cued": n,
                "p_engaged": float(licked.sum()) / max(n, 1),
                "p_correct": float(rew.sum()) / max(int(licked.sum()), 1),
                "catch_false_alarms": fa}

    def accuracy(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Fraction of cued trials rewarded. See :meth:`outcome_counts` for
        the engagement / correctness split."""
        c = self.outcome_counts(outputs, batch)
        return torch.tensor(c["rewarded"] / max(c["n_cued"], 1))
