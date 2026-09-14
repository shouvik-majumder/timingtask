"""
timingtask.supervised — the supervised face of the timing task.
============================================================================

Wraps the trial generator in the ``Task`` interface from
:mod:`timingtask.contract`, so ``training.train`` and
``models.make_model`` work unchanged::

    task  = TimingTask(variant="fixed")
    model = make_model("vanilla", task.spec, hidden_size=256, dt=task.dt)
    train(model, task, steps=4000)

THE NETWORK NEVER ACTS DURING TRAINING, AND NOTHING SCRIPTED DOES EITHER
-----------------------------------------------------------------------
An earlier version rolled a scripted "perfect" licker through the environment
and trained on the trials it produced. That is behavioural cloning of a teacher,
and worse, it let the teacher's action decide when each trial ENDED -- so the
network only ever saw trial structure shaped by another agent's behaviour.

The convention in this literature (Yang et al. 2019; Wang et al. 2018; Sohn et
al. 2019; Dubreuil/Valente et al. 2022) is different and simpler: a trial has a
FIXED EPOCH STRUCTURE, nothing acts during training, and the network's output is
scored against a target defined over the whole window. Here that means the
generator is rolled forward with a no-lick observer purely to lay out the
epochs -- ITI, cue, delay, answer window -- and every trial runs to the end of
its answer window regardless of what any agent would have done.

OUTPUTS: A WITHHOLD UNIT AND A LICK UNIT
----------------------------------------
Two outputs, following the fixation/response convention of Yang et al. 2019:

    unit 0  WITHHOLD   high while the animal must not lick, low afterwards
    unit 1  LICK       rises to threshold across the answer window

A single ramp cannot express "actively hold still" -- it only says "not yet".
The two-unit form gives the withhold requirement its own gradient, which matters
here because the task asks the network to withhold twice for different reasons
(the ITI, and then the delay).

COST MASK
---------
Weighted, not boolean, following ``multitask/task.py::add_c_mask``:

  * a GRACE PERIOD at trial start and straddling each epoch transition, weight 0,
    so the network is not punished for finite response latency;
  * the answer window up-weighted (``w_answer``, default 5);
  * the withhold unit up-weighted (``w_withhold``, default 2), because a task
    whose main difficulty is not-responding needs the not-responding scored.

VARIABILITY
-----------
Degeneracy is the thing to avoid: if every trial is identical the network stores
one waveform and infers nothing. Sources here, all on by default:

  * variable ITI (truncated exponential -> flat hazard, so elapsed time carries
    no information about cue onset and the network must time FROM THE CUE);
  * catch trials, where the target holds withhold for the whole trial;
  * input noise ``sigma_x`` and recurrent noise (in the model), both scaled by
    sqrt(2/alpha) so their effect is dt-independent;
  * optional jitter on the target lick time.

Trial-to-trial variation of the required delay comes from the scheduler
(``mode="variable"``), which is what turns this from "reproduce one waveform"
into "read the interval off experience".
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
    target_offset : where in the answer window supervision aims, in seconds
        after the delay elapses.
    target_jitter : SD of per-trial jitter on that aim point (seconds). Small
        jitter stops the network locking onto one exact step. Note this is an
        augmentation, not a published technique -- Sohn et al. jitter the INPUT
        event times instead, which is the more defensible route when the point
        is scalar timing noise.
    threshold : level the lick unit must cross; the lick time is the crossing.
    plateau : where the lick target settles after the crossing, as a multiple of
        ``threshold``. Must be > 1 -- a target that flattens AT the threshold
        puts the loss's least-sensitive region on the decision boundary, and the
        network's accuracy then collapses while its loss keeps improving.
    hold_level : the withhold unit's target while withholding.
    w_answer, w_withhold : cost-mask weights.
    grace : seconds of zero weight after trial start and after each transition.
    sigma_x : input noise SD, scaled by sqrt(2/alpha) like the recurrent noise.
    eval_noise : if True, input noise is applied even when the model is in eval
        mode. Without it every trial is identical and there is no lick-time
        DISTRIBUTION to compare with an animal's.
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
            raise ValueError(
                "plateau must be > 1: a target that flattens at the threshold "
                "puts the loss's least-sensitive region on the decision boundary")

        # cognitive.Task works in MILLISECONDS; the timing configs use seconds.
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
        """Lay out ONE trial's epochs with a no-lick observer.

        Nothing acts. The observer never licks, so the trial runs its full
        course -- ITI, cue, delay, the whole answer window -- and the resulting
        structure is a property of the task, not of any agent's behaviour.
        The 'miss' outcome this produces is discarded; only the timing is used.
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
        """Return ``(y (T, 2), w (T,))`` -- targets and cost-mask weights."""
        dt = self.cfg.dt
        y = np.zeros((n_steps, 2), dtype=np.float32)
        w = np.zeros(n_steps, dtype=np.float32)
        grace_steps = max(1, int(round(self.grace / dt)))
        onset = rec.get("cue_onset_step")

        # WITHHOLD is high from trial start; it only falls once licking is allowed.
        y[:, WITHHOLD] = self.hold_level
        w[:] = 1.0
        w[:grace_steps] = 0.0                    # settle-in at trial start

        if onset is None:                        # catch trial: withhold throughout
            return y, w

        t_star = float(rec["delay"]) + self.target_offset
        if self.target_jitter:
            t_star = max(dt, t_star + float(self._rng.normal(0, self.target_jitter)))
        star = max(1, int(round(t_star / dt)))
        idx = np.arange(n_steps)
        elapsed = idx - onset

        # LICK unit: rises from cue onset, crosses `threshold` at t*, keeps
        # rising to `plateau * threshold` -- it must NOT flatten at the threshold.
        rising = elapsed >= 0
        frac = np.maximum(elapsed[rising], 0) / star
        y[rising, LICK] = np.minimum(
            self.threshold * frac ** self.ramp_exponent,
            self.threshold * self.plateau)
        # WITHHOLD falls once the delay has elapsed.
        delay_step = int(round(float(rec["delay"]) / dt))
        y[elapsed >= delay_step, WITHHOLD] = 0.05

        # weights: answer window matters most; grace windows straddle each
        # transition so response latency is never punished.
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
        # loss_mask stays boolean for compatibility with masked_loss; the
        # weights ride along in meta and are used by `weighted_loss` below.
        return TrialBatch(x, y, weight > 0, meta)

    def loss(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Weighted masked MSE. Use this instead of ``training.masked_loss`` to
        get the epoch and output-unit weighting; a plain boolean mask throws
        both away."""
        w = batch.meta["weight"].to(outputs.device)                 # (B, T)
        unit_w = torch.ones(outputs.shape[-1], device=outputs.device)
        unit_w[WITHHOLD] = self.w_withhold
        se = ((outputs - batch.targets) ** 2 * unit_w).mean(-1)     # (B, T)
        return (se * w).sum() / w.sum().clamp_min(1)

    # -- readout ----------------------------------------------------------
    def crossing_time(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """First crossing of the LICK unit past threshold after cue onset, in
        seconds. NaN if it never crosses."""
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
        """Behavioural summary, splitting ENGAGEMENT from CORRECTNESS.

        The two must be tracked separately. Withholding forever is a zero-cost
        policy that scores no errors, so a single accuracy number cannot tell an
        agent that has learned the timing from one that has learned to do
        nothing. Song et al. 2017 use exactly this split as their stopping
        criterion (``p_decision >= 0.99 and p_correct >= 0.8``).
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
        # false alarms on catch trials: licking with no cue at all
        catch = batch.meta["no_cue_trial"].to(t.device)
        fa = int((catch & ~torch.isnan(t)).sum())
        return {"rewarded": int(rew.sum()), "early": int(early.sum()),
                "miss": int(miss.sum()), "n_cued": n,
                "p_engaged": float(licked.sum()) / max(n, 1),
                "p_correct": float(rew.sum()) / max(int(licked.sum()), 1),
                "catch_false_alarms": fa}

    def accuracy(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Fraction of cued trials rewarded. Report ``outcome_counts`` alongside:
        this number alone cannot distinguish good timing from never licking."""
        c = self.outcome_counts(outputs, batch)
        return torch.tensor(c["rewarded"] / max(c["n_cued"], 1))
