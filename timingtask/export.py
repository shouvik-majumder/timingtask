"""
timingtask.export — hand the trained agent's states and behaviour to analysis.
==============================================================================

This module is the ONLY connection between this repository and the geometry
repository, and it is a FILE FORMAT, not an import. Nothing here imports
``neuralgeom``; nothing in ``neuralgeom`` imports this. Train an agent here,
write a ``.h5``, open it there.

    recs = trainer.evaluate(model, n_trials=512, collect_states=True)
    save_trajectory(recs, "runs/act_probe.h5", generator="timingtask.rl")

    # ...in the analysis repo, in the same conda environment:
    from neuralgeom.data import load_trajectory
    traj = load_trajectory("runs/act_probe.h5")
    traj.X          # (n_trials, T, N)  the recurrent states
    traj.condition  # (n_trials,)       the required delay on that trial

The schema
----------
The canonical ``Trajectory`` HDF5 layout, reproduced here so this file is
readable without the other repository at hand:

    X          (n_trials, T, N)      REQUIRED  hidden states
    time       (T,)                  REQUIRED  sample times, SECONDS
    dt, tau                          attrs     scalars
    inputs     (n_trials, T, n_in)   optional  the drive the network saw
    outputs    (n_trials, T, n_out)  optional  [policy readout, critic value]
    condition  (n_trials,)           optional  one label per trial
    W          (N, N)                optional  recurrent connectivity
    aux/<key>  (n_trials, ...)       optional  per-trial behaviour
    meta                             attr      JSON: generator id + config

Two decisions this module makes, and why
----------------------------------------
**Trials are padded to a common length, not truncated to the shortest.** Trial
length is an OUTCOME here -- a trial ends at the decisive lick -- so truncating
to the shortest would discard exactly the long trials, which are the ones where
the agent waited, which are the ones the analysis is about. Padding is ``NaN``,
never zero: zero is a perfectly good hidden state and would be silently
analysed as one. ``aux/n_steps`` carries each trial's true length so any
analysis can mask.

**Alignment defaults to the cue, not to trial start.** The stop-licking period
is resampled every trial and restarts on every lick, so trial start is at a
random distance from the cue and averaging across trials in that frame smears
everything that is locked to the timer. The record already carries
``align_step``, which is the cue onset, or for a catch trial the step where the
cue WOULD have been -- so catch trials land in the same frame and stay
comparable. Pass ``align="start"`` to opt out.

Nothing here assumes a cue duration; see the standing rule in
``generator.py``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["records_to_trajectory", "save_trajectory", "TrajectoryBundle"]


# Per-trial scalars worth carrying into the analysis. Anything numeric, boolean
# or None-able in the record; strings are handled separately below.
_AUX_SCALARS = (
    "trial", "delay", "answer_window", "cue_duration", "trial_steps",
    "n_licks", "n_iti_licks", "iti_steps", "first_lick_s", "decisive_lick_s",
    "trial_reward", "trial_duration_s", "reward_rate", "value_gain",
    "cue_onset_step", "align_step",
    "prev_success", "prev_reward", "prev_first_lick",
)
_AUX_BOOLS = ("rewarded", "early", "miss", "no_cue_trial", "iti_licked",
              "cue_success", "prev_action")
# Record fields that are strings. Stored as fixed-width bytes, which HDF5 takes
# natively and every reader decodes; do NOT quietly integer-code them, because
# the code would then mean nothing without a key that lives somewhere else.
_AUX_LABELS = ("outcome", "training_stage")

OUTCOME_CODES = {"rewarded": 0, "early": 1, "miss": 2, "iti_timeout": 3,
                 "no_cue_complete": 4, "ongoing": 5}


class TrajectoryBundle(dict):
    """The Trajectory fields as a plain dict, before they are written.

    A ``dict`` subclass rather than a dataclass so it can be handed straight to
    ``neuralgeom.data.Trajectory(**bundle)`` by anyone who has both packages
    installed, without this repository owning a copy of that class.
    """

    def summary(self) -> str:
        X = self["X"]
        n_finite = int(np.isfinite(X[..., 0]).sum())
        return (f"TrajectoryBundle: {X.shape[0]} trials x {X.shape[1]} steps "
                f"x {X.shape[2]} units, dt={self['dt']:.3g}s, "
                f"{n_finite} of {X.shape[0] * X.shape[1]} (trial, step) cells "
                f"occupied, aux={sorted(self['aux'])}")


def _scalar(value: Any) -> float:
    """Record fields are Optional[...] all over; None becomes NaN, not 0."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        raise TypeError(
            f"{value!r} is a string, not a number. String-valued record fields "
            f"belong in _AUX_LABELS, where they are stored as bytes; a numeric "
            f"code would be meaningless without a key stored alongside it.")
    return float(value)


def records_to_trajectory(
    records: Sequence[Dict[str, Any]],
    *,
    align: str = "cue",
    t_pre: float = 1.0,
    t_post: float = 3.0,
    condition: str = "delay",
    dt: Optional[float] = None,
    tau: Optional[float] = None,
    W: Optional[np.ndarray] = None,
    generator: str = "timingtask",
    config: Optional[Dict[str, Any]] = None,
) -> TrajectoryBundle:
    """Turn trial records into the Trajectory field set.

    Parameters
    ----------
    records
        What ``RLTrainer.evaluate`` / ``.switching_test`` / ``.probe`` return
        with ``collect_states=True``. Each must carry ``hidden``; ``obs``,
        ``readout`` and ``value`` are used if present.
    align
        ``"cue"`` (default) puts t=0 at cue onset, using the record's
        ``align_step`` so catch trials share the frame. ``"start"`` puts t=0 at
        trial start and ignores ``t_pre``.
    t_pre, t_post
        Seconds kept either side of t=0 under ``align="cue"``. Steps outside a
        trial's own extent are NaN.
    condition
        Which record field becomes the per-trial label. ``"delay"`` is the
        natural one for a timing task; ``"outcome"`` gives the integer codes in
        ``OUTCOME_CODES``.
    dt
        Seconds per step. Read from the records if not given.
    W
        Recurrent connectivity, if you want the analysis side to see it:
        ``model.core.rec.weight.detach().cpu().numpy()``.

    Returns
    -------
    TrajectoryBundle
        Under ``align="cue"`` this may hold FEWER trials than were passed
        in: a trial that never reached a cue has no alignment point and is
        dropped, with the count kept in ``meta['n_unaligned_dropped']``.
        Keys: ``X``, ``time``, ``dt``, ``tau``, ``inputs``, ``outputs``,
        ``condition``, ``W``, ``aux``, ``meta``.
    """
    recs = [r for r in records if r is not None and "hidden" in r]
    if not recs:
        raise ValueError(
            "no records carry hidden states. Pass collect_states=True to "
            "evaluate/switching_test/probe -- without it the rollout is "
            "discarded and only the behavioural summary survives.")
    if align not in ("cue", "start"):
        raise ValueError(f"align must be 'cue' or 'start', got {align!r}")

    # A trial that never reached a cue -- an ITI timeout, which an agent that
    # licks constantly produces by the hundred -- has no alignment point, so
    # under align="cue" it cannot be placed in the frame at all. DROP those
    # rather than emit an all-NaN row: a row of NaN is a trial in every count,
    # every shape and every index, and would quietly dilute any average taken
    # over trials. How many were dropped is recorded in meta.
    n_all = len(recs)
    if align == "cue":
        recs = [r for r in recs if r.get("align_step") is not None]
    n_unaligned = n_all - len(recs)
    if not recs:
        raise ValueError(
            f"none of the {n_all} trials ever reached a cue, so none can be "
            f"aligned to one. This is what an agent that licks through every "
            f"stop-licking period looks like. Use align='start' to export "
            f"them anyway.")

    dt = float(dt if dt is not None else recs[0].get("dt", 0.02))
    N = int(np.asarray(recs[0]["hidden"]).shape[1])
    has_obs = all("obs" in r for r in recs)
    has_out = all("readout" in r and "value" in r for r in recs)
    n_in = int(np.asarray(recs[0]["obs"]).shape[1]) if has_obs else 0

    # -- the common time frame ------------------------------------------- #
    if align == "cue":
        n_pre = int(round(t_pre / dt))
        n_post = int(round(t_post / dt))
        T = n_pre + n_post
        time = (np.arange(T) - n_pre) * dt
    else:
        n_pre = 0
        T = max(int(np.asarray(r["hidden"]).shape[0]) for r in recs)
        time = np.arange(T) * dt

    B = len(recs)
    X = np.full((B, T, N), np.nan, np.float32)
    inputs = np.full((B, T, n_in), np.nan, np.float32) if has_obs else None
    outputs = np.full((B, T, 2), np.nan, np.float32) if has_out else None

    for b, r in enumerate(recs):
        h = np.asarray(r["hidden"], np.float32)          # (T_b, N)
        T_b = h.shape[0]
        # Where step 0 of this trial lands in the common frame.
        if align == "cue":
            a = r.get("align_step")
            if a is None:                                # no cue ever arrived
                continue
            offset = n_pre - int(a)
        else:
            offset = 0
        src0, dst0 = max(0, -offset), max(0, offset)
        n = min(T_b - src0, T - dst0)
        if n <= 0:
            continue
        X[b, dst0:dst0 + n] = h[src0:src0 + n]
        if has_obs:
            inputs[b, dst0:dst0 + n] = np.asarray(r["obs"], np.float32)[src0:src0 + n]
        if has_out:
            outputs[b, dst0:dst0 + n, 0] = np.asarray(r["readout"], np.float32)[src0:src0 + n]
            outputs[b, dst0:dst0 + n, 1] = np.asarray(r["value"], np.float32)[src0:src0 + n]

    # -- the per-trial label --------------------------------------------- #
    if condition == "outcome":
        cond = np.array([OUTCOME_CODES.get(r.get("outcome"), -1) for r in recs],
                        np.int64)
    else:
        cond = np.array([_scalar(r.get(condition)) for r in recs], np.float64)

    # -- behaviour -------------------------------------------------------- #
    aux: Dict[str, np.ndarray] = {}
    for key in _AUX_SCALARS:
        if any(key in r for r in recs):
            aux[key] = np.array([_scalar(r.get(key)) for r in recs], np.float64)
    for key in _AUX_BOOLS:
        if any(key in r for r in recs):
            aux[key] = np.array([_scalar(r.get(key)) for r in recs], np.float64)
    for key in _AUX_LABELS:
        if any(key in r for r in recs):
            aux[key] = np.array([str(r.get(key) or "") for r in recs], dtype="S")
    aux["outcome_code"] = np.array(
        [OUTCOME_CODES.get(r.get("outcome"), -1) for r in recs], np.int64)
    # Steps of this trial that actually landed in the frame — the mask every
    # analysis needs in order to ignore the NaN padding.
    aux["n_steps"] = np.array([int(np.isfinite(X[b, :, 0]).sum())
                               for b in range(B)], np.int64)
    # Lick times are ragged; store them padded, with the count alongside.
    licks = [np.asarray(r.get("lick_times_aligned_s") or [], np.float64)
             for r in recs]
    n_lick_max = max((len(l) for l in licks), default=0)
    if n_lick_max:
        padded = np.full((B, n_lick_max), np.nan)
        for b, l in enumerate(licks):
            padded[b, :len(l)] = l
        aux["lick_times_aligned_s"] = padded
        aux["n_lick_times"] = np.array([len(l) for l in licks], np.int64)

    meta = {
        "generator": generator,
        "align": align,
        "t_pre": t_pre if align == "cue" else 0.0,
        "t_post": t_post if align == "cue" else T * dt,
        "condition_field": condition,
        "outcome_codes": OUTCOME_CODES,
        "n_trials": B,
        "n_unaligned_dropped": n_unaligned,
        "padding": "nan",
        "output_labels": ["policy_readout_lick_minus_wait", "critic_value"],
        "config": config or {},
    }

    return TrajectoryBundle(
        X=X, time=time, dt=dt, tau=tau, inputs=inputs, outputs=outputs,
        condition=cond, W=(None if W is None else np.asarray(W)),
        aux=aux, meta=meta)


def save_trajectory(records: Sequence[Dict[str, Any]], path: str, *,
                    compression: str = "gzip", **kwargs) -> str:
    """Build a bundle from ``records`` and write it as Trajectory HDF5.

    Keyword arguments are those of :func:`records_to_trajectory`. Returns the
    path. Needs ``h5py``; needs nothing from the geometry repository.
    """
    bundle = (records if isinstance(records, TrajectoryBundle)
              else records_to_trajectory(records, **kwargs))
    return write_bundle(bundle, path, compression=compression)


def write_bundle(bundle: Dict[str, Any], path: str, *,
                 compression: str = "gzip") -> str:
    """Write a bundle in the canonical Trajectory HDF5 schema."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=bundle["X"], compression=compression)
        f.create_dataset("time", data=bundle["time"])
        f.attrs["dt"] = float(bundle["dt"])
        if bundle.get("tau") is not None:
            f.attrs["tau"] = float(bundle["tau"])
        for name in ("inputs", "outputs"):
            if bundle.get(name) is not None:
                f.create_dataset(name, data=np.asarray(bundle[name]),
                                 compression=compression)
        for name in ("condition", "W"):
            if bundle.get(name) is not None:
                f.create_dataset(name, data=np.asarray(bundle[name]))
        for key, val in (bundle.get("aux") or {}).items():
            f.create_dataset(f"aux/{key}", data=np.asarray(val))
        f.attrs["meta"] = json.dumps(bundle.get("meta") or {}, default=str)
        # The flag ``neuralgeom.data.load_trajectory`` uses to tell the
        # canonical schema from the legacy rnn_{tag}.h5 one.
        f.attrs["neuralgeom_trajectory"] = True
    return path
