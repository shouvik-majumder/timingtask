"""
timingtask.training — a task-agnostic supervised trainer.
=============================================================

Knows nothing about any specific task or model beyond the two interfaces:

    batch = task.sample(B)        -> TrialBatch (inputs, targets, loss_mask)
    out, H = model(batch.inputs)  -> (B, T, out), (B, T, hidden)

so a new task or a new architecture drops straight in.

    hist = train(model, task, steps=2000)
    print(hist["acc"][-1])
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .contract import Task, TrialBatch

__all__ = ["masked_loss", "train", "evaluate", "run_trials"]


def masked_loss(outputs: Tensor, batch: TrialBatch, kind: str = "cross_entropy"
                ) -> Tensor:
    """Loss averaged over the (B, T) positions where ``loss_mask`` is True."""
    m = batch.loss_mask
    if kind == "cross_entropy":
        B, T, K = outputs.shape
        ce = nn.functional.cross_entropy(
            outputs.reshape(B * T, K), batch.targets.reshape(B * T),
            reduction="none").reshape(B, T)
        return (ce * m).sum() / m.sum().clamp_min(1)
    if kind == "mse":
        se = (outputs - batch.targets).pow(2).mean(-1)
        return (se * m).sum() / m.sum().clamp_min(1)
    raise ValueError(f"Unknown loss {kind!r}")


def train(model: nn.Module, task: Task, *, steps: int = 2000,
          batch_size: int = 64, lr: float = 2e-3, grad_clip: float = 1.0,
          l2_rate: float = 0.0, l2_weight: float = 0.0,
          log_every: int = 200, device: str = "cpu",
          on_log: Optional[Callable[[int, Dict], None]] = None,
          verbose: bool = True) -> Dict[str, List[float]]:
    """Train ``model`` on ``task``. Returns a history dict.

    l2_rate : penalty on mean squared firing rate (keeps dynamics tame and
        makes fixed-point structure cleaner — standard in the RNN-for-neuro
        literature).
    l2_weight : penalty on recurrent weight magnitude.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist: Dict[str, List[float]] = {"step": [], "loss": [], "acc": []}
    t0 = time.time()

    for i in range(1, steps + 1):
        batch = task.sample(batch_size).to(device)
        out, H = model(batch.inputs)
        loss = masked_loss(out, batch, task.spec.loss)
        if l2_rate:
            loss = loss + l2_rate * H.pow(2).mean()
        if l2_weight:
            wsum = sum(p.pow(2).sum() for n, p in model.named_parameters()
                       if "rec" in n or "weight_hh" in n)
            loss = loss + l2_weight * wsum
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if i % log_every == 0 or i == 1 or i == steps:
            acc = float(evaluate(model, task, batch_size=256, device=device))
            hist["step"].append(i)
            hist["loss"].append(float(loss.detach()))
            hist["acc"].append(acc)
            if verbose:
                print(f"    step {i:5d}/{steps}  loss {hist['loss'][-1]:.4f}  "
                      f"acc {acc:.3f}  ({time.time() - t0:.0f}s)",
                      flush=True)
            if on_log:
                on_log(i, {"loss": hist["loss"][-1], "acc": acc})
            model.train()

    model.eval()
    return hist


@torch.no_grad()
def evaluate(model: nn.Module, task: Task, *, batch_size: int = 256,
             device: str = "cpu") -> Tensor:
    """Mean accuracy on a fresh batch (model set to eval → no noise).

    A task that defines its own ``accuracy`` gets to decide what "correct"
    means, and this defers to it. Only tasks that do NOT override it fall back
    to the categorical rule below.

    That fallback assumes ``targets`` is ``(B, T)`` integer class labels, which
    is true of every cross-entropy task but not of a regression task, whose
    targets are ``(B, T, out)`` — the shapes then fail to broadcast. It also
    scores by ``argmax``, which is meaningless for a scalar readout. The timing
    task, for instance, is "correct" when its readout crosses threshold inside
    the answer window; no amount of argmax expresses that.
    """
    was_training = model.training
    model.eval()
    batch = task.sample(batch_size).to(device)
    out, _ = model(batch.inputs)

    if type(task).accuracy is not Task.accuracy:      # task defines its own
        acc = task.accuracy(out, batch)
        acc = acc.mean() if getattr(acc, "ndim", 0) else acc
    else:
        # score only the decision epoch: positions where the target is nonzero
        dec = batch.loss_mask & (batch.targets > 0)
        pred = out.argmax(-1)
        acc = ((pred == batch.targets) & dec).sum().double() / dec.sum().clamp_min(1)

    if was_training:
        model.train()
    return acc


@torch.no_grad()
def run_trials(model: nn.Module, task: Task, batch_size: int = 256,
               device: str = "cpu"):
    """Sample a batch and run it (noise-free). Returns (batch, outputs, hidden)."""
    was_training = model.training
    model.eval()
    batch = task.sample(batch_size).to(device)
    out, H = model(batch.inputs)
    if was_training:
        model.train()
    return batch, out, H
