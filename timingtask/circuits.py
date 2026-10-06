"""
timingtask.circuits: published low-dimensional timing circuits.

Reference implementations of the candidate circuit models considered in
Yang et al. (2025, *Nature* 649:1244-1253, Extended Data Fig. 1) and of the
two-attractor model of Majumder et al. (2026, *Nat. Commun.* 17:8353,
Fig. 4). They are provided so that a task-trained network can be compared
against the published hypotheses with common tools.

The Yang et al. circuits are four-unit rate networks (units 1-2 cortex, units
3-4 striatum). Connectivity matrices and simulation parameters are transcribed
from the released code (``github.com/inagaki-lab/Yang_et_al_2024``,
``codeForEDF1_github/network.py``); :func:`verify` re-derives each model's
spectrum from those matrices.

Perturbation signatures
-----------------------
Yang et al. distinguish the circuits by how a transient perturbation shifts
the lick time as a function of trial duration:

  PAUSE   a constant time shift equal to the perturbation duration,
          independent of trial length (the state is preserved; the clock stops).
  REWIND  a constant amplitude setback, so the time shift grows with trial
          duration (the state is pushed back along the manifold).

:func:`perturbation_signature` reproduces both protocols for each model and
:func:`compare_all` tabulates the results.

    >>> from timingtask import circuits as C
    >>> print(C.classify("data").summary())
    >>> print(C.compare_all())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Circuit", "MODELS", "get", "simulate", "classify", "Spectrum",
           "lick_time", "perturbation_signature", "compare_all", "verify",
           "two_attractor_field", "two_attractor_trajectory"]

TAU, DT, RMAX, N_STEPS, CUE_IDX = 0.010, 0.001, 100.0, 3500, 499
INH_START, INH_DUR, INH_RAMP = 1100, 300, 300


@dataclass
class Circuit:
    """One candidate circuit. Units 1-2 are cortical (ALM), units 3-4 striatal."""
    name: str
    paper_name: str
    W: np.ndarray
    input_vector: np.ndarray
    h_init: np.ndarray
    trial_gains: Tuple[float, ...]
    alm_silencing: float          # current applied to units 1-2
    str_inhibition: float         # current applied to unit 3
    readout_unit: int = 2         # 0-based; the unit whose rate is thresholded
    ramp_input: bool = False      # externally_driven uses a ramp input, not a step
    verdict: str = ""
    note: str = ""

    def __post_init__(self):
        self.W = np.asarray(self.W, float)
        self.input_vector = np.asarray(self.input_vector, float)
        self.h_init = np.asarray(self.h_init, float)


# --------------------------------------------------------------------------- #
# The candidate circuits. Panel letters refer to Extended Data Fig. 1 of
# Yang et al. (2025).
# --------------------------------------------------------------------------- #
MODELS: Dict[str, Circuit] = {
    "externally_driven": Circuit(
        "externally_driven", "ED1a: externally driven",
        [[.4, -.3, .6, 0], [-.3, .4, .6, 0], [.6, 0, .4, -.3], [.6, 0, -.3, .4]],
        [0.5, 0, 0, 0], [5, 5, 5, 5], (8.5, 7, 5.8, 4.8, 4), -10.0, -0.4,
        ramp_input=True,
        verdict="Rejected: predicts no lick-time shift under either protocol.",
        note="No slow mode. The Jacobian is defective (eigenvalue -0.3 with "
             "algebraic multiplicity 3, geometric multiplicity 2); activity "
             "follows the input and returns to baseline when a perturbation ends."),
    "two_region_integrator": Circuit(
        "two_region_integrator", "ED1b: distributed integrator",
        [[.4, -.3, .3, .6], [-.3, .4, 0, .9], [.9, 0, .4, -.3], [.6, .3, -.3, .4]],
        [3, 0, 0, 0], [5, 5, 5, 5], (14, 11, 8.6, 7.2, 6), -0.25, -0.3,
        verdict="Rejected: cortical silencing rewinds rather than pauses.",
        note="Two zero eigenvalues: a two-dimensional plane attractor spanning "
             "both regions. The cortical silencing direction overlaps the "
             "integration manifold."),
    "two_integrator": Circuit(
        "two_integrator", "ED1c: redundant integrators",
        [[.7, -.3, .4, 0], [-.3, .7, .4, 0], [.4, 0, .7, -.3], [.4, 0, -.3, .7]],
        [3, 0, 0, 0], [5, 5, 5, 5], (14, 11, 8.6, 7.2, 6), -10.0, -0.1,
        verdict="Rejected: cortical silencing resets rather than pauses.",
        note="Two independent line attractors, one per region. Common-mode "
             "cortical inhibition is orthogonal to both left eigenvectors, "
             "but rectification drives the striatal projection to zero."),
    "one_integrator_follower_opposite": Circuit(
        "one_integrator_follower_opposite", "ED1d: cortical integrator, striatal follower",
        [[.7, -.3, .5, 0], [-.3, .7, .5, 0], [.5, 0, .6, -.3], [.5, 0, -.3, .6]],
        [2, 0, 0, 0], [5, 5, 5, 5], (14, 11, 8.6, 7.2, 6), -10.0, -0.2,
        verdict="Rejected: striatal inhibition has almost no effect.",
        note="Line attractor in cortex alone; the left eigenvector has zero "
             "striatal weight, so striatal input cannot affect the timer."),
    "one_integrator_follower_opposite_leaky": Circuit(
        "one_integrator_follower_opposite_leaky", "ED1e: cortical leaky integrator",
        [[.69, -.3, .3, .4], [-.3, .69, 0, .4], [0, 0, .4, 0], [.4, 0, 0, .3]],
        [0, 0, 10, 0], [5, 5, 5, 5], (14, 11, 8.6, 7.2, 6), -10.0, -0.2,
        readout_unit=3,
        verdict="Rejected: reproduces the striatal rewind but not the cortical pause.",
        note="Leaky integrator (slowest eigenvalue -0.01, leak time constant "
             "1.0 s). The only model whose input arrives in striatum."),
    "data": Circuit(
        "data", "ED1f: striatal integrator, cortical input",
        [[.4, 0, 0, 0], [0, .3, .4, 0], [.03, .4, .7, -.3], [0, .4, -.3, .7]],
        [50, 0, 0, 0], [0, 5, 5, 5], (14, 11, 8.6, 7.2, 6), -10.0, -0.07,
        verdict="Accepted: pauses under cortical silencing and rewinds under "
                "striatal inhibition.",
        note="Perfect integrator with one zero Jacobian eigenvalue. The "
             "integrated variable is the striatal difference mode, left "
             "eigenvector [0.035, 0, 0.707, -0.707]. Unit 1 relays the input "
             "to striatal unit 3 with weight 0.03, which sets the ramp slope."),
}
# Panel letters as aliases.
PANELS = {"a": "externally_driven", "b": "two_region_integrator",
          "c": "two_integrator", "d": "one_integrator_follower_opposite",
          "e": "one_integrator_follower_opposite_leaky", "f": "data"}


def get(name: str) -> Circuit:
    """Look up a circuit by name or by panel letter."""
    return MODELS[PANELS.get(name, name)]


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def simulate(c: Circuit, gain: float, *, inhibition: Optional[np.ndarray] = None,
             n_steps: int = N_STEPS) -> np.ndarray:
    """Simulate one trial. Returns rates of shape ``(4, n_steps)``.

    Dynamics follow the released code: ``tau dh/dt = -h + W r + I``,
    ``r = clip(h, 0, RMAX)``, tau = 10 ms, dt = 1 ms.
    """
    r0 = np.clip(c.h_init, 0, RMAX)
    Ithresh = -(-r0 + c.W @ r0)            # holds the baseline as a fixed point
    I = np.zeros((4, n_steps))
    amp = c.input_vector * gain / CUE_IDX
    if c.ramp_input:
        ramp = np.linspace(0, 1, n_steps - CUE_IDX)
        I[:, CUE_IDX:] = amp[:, None] * ramp[None, :]
    else:
        I[:, CUE_IDX:] = amp[:, None]
    inh = np.zeros((4, n_steps)) if inhibition is None else inhibition

    h = np.zeros((4, n_steps)); r = np.zeros((4, n_steps))
    h[:, 0] = c.h_init; r[:, 0] = np.clip(h[:, 0], 0, RMAX)
    for n in range(1, n_steps):
        tot = I[:, n] + Ithresh + inh[:, n]
        h[:, n] = h[:, n - 1] + DT / TAU * (-h[:, n - 1] + c.W @ r[:, n - 1] + tot)
        r[:, n] = np.clip(h[:, n], 0, RMAX)
    return r


def inhibition_current(c: Circuit, kind: str, n_steps: int = N_STEPS) -> np.ndarray:
    """Perturbation current: full strength for 300 ms, then a 300 ms linear
    ramp-down. ``kind`` is ``'alm'`` (units 1-2) or ``'str'`` (unit 3)."""
    inh = np.zeros((4, n_steps))
    s = c.alm_silencing if kind == "alm" else c.str_inhibition
    units = [0, 1] if kind == "alm" else [2]
    a, b = INH_START, INH_START + INH_DUR
    inh[units, a:b] = s
    inh[units, b:b + INH_RAMP] = s * np.linspace(1, 0, INH_RAMP)[None, :]
    return inh


def ramp_mode(r: np.ndarray, units: Tuple[int, int]) -> np.ndarray:
    """Project activity onto the unit-norm ramp direction within a region,
    defined as the (end - cue) difference vector."""
    sl = slice(units[0], units[1])
    v = r[sl, -1] - r[sl, CUE_IDX]
    n = np.linalg.norm(v)
    return (r[sl].T @ (v / n)) if n > 1e-12 else np.zeros(r.shape[1])


LICK_RATE = 10.0        # Hz; ``target_threshold`` in the released code


def lick_time(r: np.ndarray, readout_unit: int = 2,
              rate: float = LICK_RATE) -> float:
    """Time at which the readout unit's rate first reaches ``rate``, in
    seconds from the cue. NaN if it never does."""
    idx = np.flatnonzero(r[readout_unit, CUE_IDX:] >= rate)
    return float(idx[0] * DT) if idx.size else float("nan")


# --------------------------------------------------------------------------- #
# Spectral classification
# --------------------------------------------------------------------------- #
@dataclass
class Spectrum:
    name: str
    eig_W: np.ndarray
    eig_J: np.ndarray                       # Jacobian -I + W
    n_zero: int
    n_near_zero: int
    leak_tau: Optional[float]
    left: Optional[np.ndarray]              # left eigenvector of the slowest mode
    right: Optional[np.ndarray]
    alm_weight: float = 0.0
    str_weight: float = 0.0
    overlaps: Dict[str, float] = field(default_factory=dict)
    classification: str = ""

    def summary(self) -> str:
        e = ", ".join(f"{x:+.4f}" for x in np.sort_complex(self.eig_J).real)
        s = [f"{self.name}: {self.classification}",
             f"  eig(-I+W) = [{e}]",
             f"  zero modes: {self.n_zero} exact, {self.n_near_zero} near-zero"]
        if self.leak_tau is not None:
            s.append(f"  leak tau = {self.leak_tau:.3f} s")
        if self.left is not None:
            s.append(f"  slow-mode left eigvec = "
                     f"[{', '.join(f'{x:+.3f}' for x in self.left)}]")
            s.append(f"  carried by: ALM {self.alm_weight:.3f} / "
                     f"STR {self.str_weight:.3f}")
            for k, v in self.overlaps.items():
                tag = ("on-manifold" if abs(v) > 0.2 else
                       "off-manifold" if abs(v) < 0.05 else "weakly on")
                s.append(f"  overlap with {k:<16} = {v:+.4f}   {tag}")
        return "\n".join(s)


def classify(name: str, *, zero_tol: float = 1e-9,
             near_tol: float = 0.05) -> Spectrum:
    """Eigen-decompose a circuit's Jacobian and locate its integration
    manifold.

    A perfect integrator has exactly one zero Jacobian eigenvalue (a line
    attractor). The left eigenvector of that mode identifies the integrated
    variable; an input's overlap with it indicates whether the input is
    integrated (on-manifold) or leaves the timer unaffected (off-manifold).
    """
    c = get(name)
    w, vr = np.linalg.eig(c.W)
    J = -np.eye(4) + c.W
    ej, VR = np.linalg.eig(J)

    real = ej.real
    n_zero = int(np.sum(np.abs(ej) < zero_tol))
    n_near = int(np.sum(np.abs(ej) < near_tol)) - n_zero
    k = int(np.argmin(np.abs(ej)))
    slowest = real[k]
    leak = None if abs(slowest) < zero_tol else float(TAU / abs(slowest))

    left = right = None
    alm = strw = 0.0
    ov: Dict[str, float] = {}
    if abs(ej[k]) < near_tol:
        # Match the left eigenvector to this eigenvalue rather than by index.
        ejT, VLT = np.linalg.eig(J.T)
        kk = int(np.argmin(np.abs(ejT - ej[k])))
        left = np.real(VLT[:, kk])
        left = left / (np.linalg.norm(left) + 1e-12)
        right = np.real(VR[:, k]); right = right / (np.linalg.norm(right) + 1e-12)
        alm = float(np.linalg.norm(left[:2])); strw = float(np.linalg.norm(left[2:]))
        for label, v in (("input", c.input_vector),
                         ("ALM silencing", np.array([-1., -1., 0., 0.])),
                         ("STR inhibition", np.array([0., 0., -1., 0.]))):
            nv = np.linalg.norm(v)
            ov[label] = float(left @ v / nv) if nv > 1e-12 else 0.0

    if n_zero == 1:
        cl = "PERFECT INTEGRATOR: 1-D line attractor"
    elif n_zero >= 2:
        cl = f"{n_zero}-D PLANE ATTRACTOR"
    elif n_near >= 1:
        cl = f"LEAKY INTEGRATOR (tau = {leak:.2f} s)"
    else:
        cl = "NO SLOW MODE: externally driven"
    return Spectrum(name, w, ej, n_zero, n_near, leak, left, right, alm, strw,
                    ov, cl)


# --------------------------------------------------------------------------- #
# Perturbation signatures
# --------------------------------------------------------------------------- #
def perturbation_signature(name: str) -> Dict[str, object]:
    """Run both perturbation protocols on all trial types and classify the
    lick-time shift as PAUSE (state-independent), REWIND (state-dependent),
    RESET (state-independent but much longer than the perturbation), no
    effect, or abolished licking."""
    c = get(name)
    ctrl = [simulate(c, g) for g in c.trial_gains]
    base = np.array([lick_time(r, c.readout_unit) for r in ctrl])

    out: Dict[str, object] = {"model": name, "verdict": c.verdict,
                              "control_lick_times": base.tolist()}
    for kind in ("alm", "str"):
        inh = inhibition_current(c, kind)
        lt = np.array([lick_time(simulate(c, g, inhibition=inh), c.readout_unit)
                       for g in c.trial_gains])
        d = lt - base
        ok = np.isfinite(d) & np.isfinite(base)

        # Regress the shift on the control lick time. The slope measures
        # state dependence: ~0 for a pause, > 0 for a rewind.
        slope = float("nan")
        if ok.sum() >= 2:
            slope = float(np.polyfit(base[ok], d[ok], 1)[0])
        n_lick = int(np.isfinite(lt).sum())
        mean_shift = float(np.nanmean(d[ok])) if ok.any() else float("nan")

        # A pause delays action by about the perturbation duration; a
        # state-independent shift far exceeding it is a reset.
        dur = (INH_DUR + INH_RAMP / 2) * DT          # 0.45 s effective
        if not ok.any():
            sig = "abolishes licking"
        elif np.nanmax(np.abs(d[ok])) < 0.10:
            sig = "no effect"
        elif not np.isfinite(slope) or abs(slope) >= 0.15:
            sig = "REWIND (state-dependent)"
        elif mean_shift > 2.0 * dur:
            sig = "RESET (shift >> duration)"
        else:
            sig = "PAUSE (shift ~ duration)"
        out[kind] = {"lick_times": lt.tolist(), "shift": d.tolist(),
                     "mean_shift": mean_shift, "slope": slope,
                     "n_licked": n_lick, "signature": sig}
    return out


def compare_all() -> str:
    """Tabulate the dynamical class and both perturbation signatures of every
    model."""
    hdr = ("model", "dynamical class", "ALM silencing", "STR inhibition")
    w = (40, 32, 32, 32)
    rows = ["", "".join(h.ljust(k) for h, k in zip(hdr, w)), "-" * sum(w)]
    for n in MODELS:
        sp = classify(n)
        p = perturbation_signature(n)
        a_, t_ = p["alm"], p["str"]
        rows.append("".join(x.ljust(k) for x, k in zip(
            (n, sp.classification, a_["signature"], t_["signature"]), w)))
        det = lambda d: "  shift %+.2fs  slope %+.2f" % (d["mean_shift"], d["slope"])
        rows.append("".join(x.ljust(k) for x, k in zip(
            ("", "", det(a_), det(t_)), w)))
    rows += [
        "",
        "Slope = d(shift) / d(control lick time), the state-dependence measure.",
        "  ~0  the same shift on short and long trials: PAUSE.",
        "  >0  the shift grows with trial duration: REWIND.",
        "",
        "Yang et al. (2025) report a pause of ~0.47 s under ALM silencing and a",
        "state-dependent rewind of ~1.0 s under striatal inhibition; only the",
        "'data' model reproduces both.",
    ]
    return "\n".join(rows)


def verify() -> List[str]:
    """Re-derive the published spectral properties from the connectivity
    matrices. Raises ``AssertionError`` on a mismatch; returns a summary."""
    msgs = []
    d = classify("data")
    assert d.n_zero == 1, "the accepted model must be a perfect integrator"
    assert abs(abs(d.left[2]) - 0.7071) < 1e-3 and \
           abs(abs(d.left[3]) - 0.7071) < 1e-3, "striatal difference mode"
    assert abs(d.overlaps["ALM silencing"]) < 0.05, "ALM must be off-manifold"
    assert abs(d.overlaps["STR inhibition"]) > 0.5, "STR must be on-manifold"
    msgs.append("data: line attractor, striatal difference mode, "
                f"ALM overlap {d.overlaps['ALM silencing']:+.4f} (off-manifold), "
                f"STR overlap {d.overlaps['STR inhibition']:+.4f} (on-manifold)")
    for n, expect in (("two_region_integrator", 2), ("two_integrator", 2),
                      ("one_integrator_follower_opposite", 1)):
        s = classify(n)
        assert s.n_zero == expect, (n, s.n_zero, expect)
        msgs.append(f"{n}: {s.n_zero} zero mode(s): {s.classification}")
    e = classify("one_integrator_follower_opposite_leaky")
    assert e.n_zero == 0 and e.leak_tau is not None
    msgs.append(f"leaky: no exact zero, leak tau = {e.leak_tau:.3f} s")
    x = classify("externally_driven")
    assert x.n_zero == 0 and x.n_near_zero == 0
    msgs.append("externally_driven: no slow mode")
    return msgs


# --------------------------------------------------------------------------- #
# Majumder et al. (2026): the two-attractor model
# --------------------------------------------------------------------------- #
def two_attractor_field(x, y, *, a1=(1.5, 1.5), a2=(15.0, 15.0),
                        strength1=1.0, strength2=0.5, bias_y=0.0):
    """Flow field of the two-attractor model (Fig. 4). ``x`` is the cue mode
    and ``y`` the ramping mode.

    The field is normalised to unit magnitude before anisotropic gains
    (30, 10) are applied, so it is a direction field rather than a gradient
    flow; the lick time is set by the path from the initial condition to the
    lick attractor.
    """
    xs, ys = np.asarray(x) / 5.0, np.asarray(y) / 5.0
    a1x, a1y = a1[0] / 5.0, a1[1] / 5.0
    a2x, a2y = a2[0] / 5.0, a2[1] / 5.0
    g1 = np.exp(-((xs - a1x) ** 2 + (ys - a1y) ** 2))
    g2 = np.exp(-((xs - a2x) ** 2 + (ys - a2y) ** 2))
    vx = -strength1 * (xs - a1x) * g1 - strength1 * (xs - a2x) * g2
    vy = (-strength2 * (ys - a1y) * g1
          - strength2 * (ys - a2y + bias_y) * g2)
    mod = np.clip((ys - a1y) / max(a2y - a1y, 1e-9) * 2.0, 0, 2)
    vx = vx * mod
    n = np.sqrt(vx ** 2 + vy ** 2) + 1e-6
    return 30.0 * vx / n, 10.0 * vy / n


def two_attractor_trajectory(amp: float, angle: float, *, dt: float = 1e-3,
                             n_steps: int = 3000, cue_at: float = 0.5,
                             cue_dur: float = 0.2, y_thr: float = 15.0):
    """Simulate one trial of the two-attractor model. The cue is a transient
    kick of amplitude ``amp`` and direction ``angle``; the lick time is set by
    the resulting initial condition. Returns ``(x, y, lick_time)``."""
    x, y = 1.5, 1.5
    xs, ys = np.empty(n_steps), np.empty(n_steps)
    a, b = int(cue_at / dt), int((cue_at + cue_dur) / dt)
    lick = float("nan")
    for t in range(n_steps):
        u, v = two_attractor_field(x, y)
        sx = amp * np.cos(angle) if a <= t < b else 0.0
        sy = amp * np.sin(angle) if a <= t < b else 0.0
        x = max(1.0, x + (u + sx) * dt)
        y = max(1.0, y + (v + sy) * dt)
        xs[t], ys[t] = x, y
        if np.isnan(lick) and y >= 0.99 * y_thr:
            lick = (t * dt) - cue_at
    return xs, ys, lick
