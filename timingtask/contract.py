"""
timingtask.contract: the batched-trial interface shared by tasks and trainers.

    TaskSpec    static description of a task: dimensions, loss type, labels
    TrialBatch  one batch of trials: inputs, targets, loss_mask, meta
    Task        base class; subclasses set ``spec`` and implement ``sample``

Conventions
-----------
* Time is discretised with step ``dt`` in milliseconds (the configuration
  dataclasses use seconds; ``supervised.py`` converts).
* Trials in a batch share one tensor length ``T``. Per-trial epoch boundaries
  vary and are carried by ``loss_mask``.
* Targets are ``(B, T)`` int64 class labels for cross-entropy, or
  ``(B, T, out)`` float for MSE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

__all__ = ["TaskSpec", "TrialBatch", "Task"]


@dataclass
class TaskSpec:
    """Static description of a task."""
    name: str
    input_dim: int
    output_dim: int
    loss: str = "cross_entropy"           # or "mse"
    input_labels: Sequence[str] = ()
    output_labels: Sequence[str] = ()
    description: str = ""


@dataclass
class TrialBatch:
    """One batch of trials."""
    inputs: Tensor                        # (B, T, input_dim)
    targets: Tensor                       # (B, T) long | (B, T, out) float
    loss_mask: Tensor                     # (B, T) bool
    meta: Dict[str, Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_steps(self) -> int:
        return self.inputs.shape[1]

    def to(self, device) -> "TrialBatch":
        return TrialBatch(
            self.inputs.to(device), self.targets.to(device),
            self.loss_mask.to(device),
            {k: v.to(device) for k, v in self.meta.items()},
        )


class Task:
    """Base class for batched tasks. Subclasses implement ``sample`` and set
    ``spec``."""

    spec: TaskSpec

    def __init__(self, dt: float = 20.0, sigma: float = 0.15,
                 seed: Optional[int] = None):
        self.dt = float(dt)
        self.sigma = float(sigma)          # input noise SD, scaled by dt in _noise
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))
        else:
            self._gen.seed()

    # -- helpers ---------------------------------------------------------- #
    def _steps(self, ms: float) -> int:
        return max(1, int(round(ms / self.dt)))

    def _randn(self, *shape) -> Tensor:
        return torch.randn(*shape, generator=self._gen)

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, generator=self._gen)

    def _randint(self, high: int, shape) -> Tensor:
        return torch.randint(high, shape, generator=self._gen)

    def _noise(self, shape) -> Tensor:
        """Input noise scaled so that its effect is independent of dt."""
        return self.sigma * math.sqrt(2.0 * 100.0 / self.dt) * self._randn(*shape)

    def sample(self, batch_size: int) -> TrialBatch:
        raise NotImplementedError

    def accuracy(self, outputs: Tensor, batch: TrialBatch) -> Tensor:
        """Per-trial fraction of masked positions whose argmax matches the
        target. Defined for classification tasks only.

        outputs : (B, T, out) logits.
        """
        if self.spec.loss != "cross_entropy":
            raise NotImplementedError("accuracy defined for classification tasks")
        pred = outputs.argmax(-1)                              # (B, T)
        m = batch.loss_mask
        correct = ((pred == batch.targets) & m).sum(1).double()
        return correct / m.sum(1).clamp_min(1).double()

    def __repr__(self) -> str:
        return f"{self.spec.name}(dt={self.dt}, sigma={self.sigma})"
