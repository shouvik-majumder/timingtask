"""
timingtask.models: recurrent network models.

Every model implements the same interface:

    out, H = model(inputs)          # (B,T,in) -> (B,T,out), (B,T,hidden)
    h1     = model.step(x_t, h)     # one step: (B,in), (B,hidden) -> (B,hidden)
    y      = model.readout(h)       # (B,hidden) -> (B,out)
    model.hidden_size, model.state_is_tuple

``step`` is a pure function of ``(x, h)`` with no side effects, so Jacobians
of the dynamics can be taken with ``torch.func``. The trainers in
:mod:`timingtask.rl` and :mod:`timingtask.training` accept any model with this
interface, and :class:`~timingtask.rl.ActorCritic` wraps any of them as the
recurrent core of an agent.

:class:`VanillaRNN` is the continuous-time ("leaky") tanh network standard in
computational neuroscience:

    h_{t+1} = (1 - alpha) h_t + alpha * tanh(W_rec h_t + W_in x_t + b + noise)
    alpha   = dt / tau
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["VanillaRNN", "GRUModel", "LSTMModel", "MODELS", "make_model"]


class _BaseRNN(nn.Module):
    state_is_tuple = False

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def readout(self, h: Tensor) -> Tensor:
        return self.out(h)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size,
                           device=device or self.out.weight.device,
                           dtype=dtype or self.out.weight.dtype)

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, inputs: Tensor, h0: Optional[Tensor] = None,
                return_hidden: bool = True) -> Tuple[Tensor, Tensor]:
        B, T, _ = inputs.shape
        h = self.init_state(B, inputs.device, inputs.dtype) if h0 is None else h0
        hs = []
        for t in range(T):
            h = self.step(inputs[:, t], h)
            hs.append(h)
        H = torch.stack(hs, dim=1)                      # (B, T, hidden)
        Y = self.readout(H)
        return (Y, H) if return_hidden else Y


class VanillaRNN(_BaseRNN):
    """Leaky tanh RNN (forward-Euler discretisation of a continuous-time
    rate network).

    Parameters
    ----------
    tau : unit time constant in milliseconds. A scalar gives every unit the
        same value; a ``(low, high)`` pair draws per-unit values log-uniformly
        from that range.
    dt : integration step in milliseconds; should match the task's ``dt``.
    noise : SD of the private recurrent noise, scaled by ``sqrt(2 / alpha)``
        so that the stationary variance of the noise-driven state is
        independent of ``dt``. Applied in training mode only.
    g : gain of the recurrent weight initialisation.
    rec_init : ``"gaussian"`` (i.i.d. N(0, g^2 / N)) or ``"orthogonal"``
        (orthogonal matrix scaled by ``g``).
    train_h0 : learn the initial state instead of fixing it at zero.
    train_tau : learn the (log) time constants.
    """

    def __init__(self, input_size, hidden_size, output_size, *,
                 tau=100.0, dt: float = 20.0, noise: float = 0.05,
                 g: float = 1.0, rec_init: str = "gaussian",
                 train_h0: bool = False, train_tau: bool = False,
                 nonlinearity=torch.tanh):
        super().__init__(input_size, hidden_size, output_size)
        if isinstance(tau, (tuple, list)):
            lo, hi = float(tau[0]), float(tau[1])
            t = torch.exp(torch.empty(hidden_size).uniform_(
                math.log(lo), math.log(hi)))
        else:
            t = torch.full((hidden_size,), float(tau))
        # tau must exceed dt, otherwise alpha > 1 and the update overshoots.
        log_tau = torch.log(t.clamp_min(float(dt) * 1.0001))
        if train_tau:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)
        self.dt = float(dt)
        self.noise = float(noise)
        self.phi = nonlinearity
        self.inp = nn.Linear(input_size, hidden_size, bias=True)
        self.rec = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            if rec_init == "orthogonal":
                nn.init.orthogonal_(self.rec.weight, gain=g)
            else:
                self.rec.weight.normal_(0.0, g / math.sqrt(hidden_size))
            self.inp.weight.normal_(0.0, 1.0 / math.sqrt(input_size))
            self.inp.bias.zero_()
        self.h0 = nn.Parameter(torch.zeros(hidden_size), requires_grad=train_h0)

    @property
    def alpha(self) -> Tensor:
        """Per-unit leak ``dt / tau`` as a ``(hidden,)`` tensor, clamped to
        at most 1."""
        return (self.dt / torch.exp(self.log_tau)).clamp(1e-4, 1.0)

    @property
    def tau(self) -> Tensor:
        return torch.exp(self.log_tau)

    def init_state(self, batch_size, device=None, dtype=None) -> Tensor:
        return self.h0.to(device=device or self.h0.device,
                          dtype=dtype or self.h0.dtype).expand(batch_size, -1)

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        a = self.alpha
        pre = self.rec(h) + self.inp(x)
        if self.training and self.noise > 0:
            pre = pre + torch.sqrt(2.0 / a) * self.noise * torch.randn_like(pre)
        return (1 - a) * h + a * self.phi(pre)

    def velocity(self, x: Tensor, h: Tensor) -> Tensor:
        """One-step state change ``F(h, x) - h``; zero at a fixed point."""
        return self.step(x, h) - h


class GRUModel(_BaseRNN):
    """Gated recurrent unit with the same interface."""

    def __init__(self, input_size, hidden_size, output_size, **kw):
        super().__init__(input_size, hidden_size, output_size)
        self.cell = nn.GRUCell(input_size, hidden_size)

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        return self.cell(x, h)

    def velocity(self, x: Tensor, h: Tensor) -> Tensor:
        return self.step(x, h) - h


class LSTMModel(_BaseRNN):
    """LSTM with the same interface. The state ``(h, c)`` is concatenated
    into a single vector of length ``2 * hidden_size``."""

    state_is_tuple = True

    def __init__(self, input_size, hidden_size, output_size, **kw):
        super().__init__(input_size, hidden_size, output_size)
        self.cell = nn.LSTMCell(input_size, hidden_size)

    def init_state(self, batch_size, device=None, dtype=None) -> Tensor:
        z = torch.zeros(batch_size, 2 * self.hidden_size,
                        device=device or self.out.weight.device,
                        dtype=dtype or self.out.weight.dtype)
        return z

    def step(self, x: Tensor, state: Tensor) -> Tensor:
        h, c = state[:, :self.hidden_size], state[:, self.hidden_size:]
        h, c = self.cell(x, (h, c))
        return torch.cat([h, c], dim=1)

    def readout(self, state: Tensor) -> Tensor:
        return self.out(state[..., :self.hidden_size])

    def velocity(self, x: Tensor, state: Tensor) -> Tensor:
        return self.step(x, state) - state


MODELS = {"vanilla": VanillaRNN, "gru": GRUModel, "lstm": LSTMModel}


def make_model(name: str, spec, hidden_size: int = 128, **kw):
    """Build a model from a ``TaskSpec``.

    >>> model = make_model("vanilla", task.spec, hidden_size=128, dt=task.dt)
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODELS)}")
    return MODELS[name](spec.input_dim, hidden_size, spec.output_dim, **kw)
