"""
timingtask.export: HDF5 export of agent states and behaviour.

Converts trial records that carry per-step time series (see
:func:`timingtask.rl.attach_states`) into a single HDF5 file of cue-aligned
hidden states, inputs, readouts and per-trial behaviour.

    recs = trainer.evaluate(model, n_trials=512, collect_states=True)
    save_trajectory(recs, "runs/agent.h5", generator="timingtask.rl")

Schema
------
    X          (n_trials, T, N)      required  hidden states
    time       (T,)                  required  sample times in seconds
    dt, tau                          attrs     scalars
    inputs     (n_trials, T, n_in)   optional  observation vectors
    outputs    (n_trials, T, n_out)  optional  [policy readout, value]
    condition  (n_trials,)           optional  one label per trial
    W          (N, N)                optional  recurrent weight matrix
    aux/<key>  (n_trials, ...)       optional  per-trial behaviour
    meta                             attr      JSON: generator id, alignment,
                                               configuration

Conventions
-----------
* Trials are padded to a common length with NaN; ``aux/n_steps`` gives each
  trial's number of valid steps.
* With ``align="cue"`` (default), ``t = 0`` is cue onset. Catch trials are
  aligned to the step at which the cue would have occurred. Trials that never
  reached a cue are dropped and counted in ``meta["n_unaligned_dropped"]``.
  With ``align="start"``, ``t = 0`` is trial start.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["records_to_trajectory", "save_trajectory", "TrajectoryBundle"]


# Per-trial scalar fields carried into ``aux``.
_AUX_SCALARS = (
    "trial", "delay", "answer_window", "cue_duration", "trial_steps",
    "n_licks", "n_iti_licks", "iti_steps", "first_lick_s", "decisive_lick_s",
    "trial_reward", "trial_duration_s", "reward_rate", "value_gain",
    "cue_onset_step", "align_step",
    "prev_success", "prev_reward", "prev_first_lick",
)
_AUX_BOOLS = ("rewarded", "early", "miss", "no_cue_trial", "iti_licked",
              "cue_success", "prev_action")
# String-valued fields, stored as fixed-width bytes.
_AUX_LABELS = ("outcome", "training_stage")

OUTCOME_CODES = {"rewarded": 0, "early": 1, "miss": 2, "iti_timeout": 3,
                 "no_cue_complete": 4, "ongoing": 5}


class TrajectoryBundle(dict):
    """The exported fields as a plain dict, before writing."""

    def summary(self) -> str:
        X = self["X"]
        n_finite = int(np.isfinite(X[..., 0]).sum())
        return (f"TrajectoryBundle: {X.shape[0]} trials x {X.shape[1]} steps "
                f"x {X.shape[2]} units, dt={self['dt']:.3g}s, "
                f"{n_finite} of {X.shape[0] * X.shape[1]} (trial, step) cells "
                f"occupied, aux={sorted(self['aux'])}")


def _scalar(value: Any) -> float:
    """Convert an optional record field to float; None becomes NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        raise TypeError(
            f"{value!r} is a string, not a number. String-valued record fields "
            f"belong in _AUX_LABELS.")
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
    """Convert trial records into the exported field set.

    Parameters
    ----------
    records
        Records returned by ``RLTrainer.evaluate`` / ``.test`` / ``.probe``
        with ``collect_states=True``. Each must carry ``hidden``; ``obs``,
        ``readout`` and ``value`` are used if present.
    align
        ``"cue"`` (default) puts t=0 at cue onset using the record's
        ``align_step``; ``"start"`` puts t=0 at trial start and ignores
        ``t_pre``.
    t_pre, t_post
        Seconds kept before and after t=0 under ``align="cue"``.
    condition
        Record field used as the per-trial label. ``"outcome"`` yields the
        integer codes in ``OUTCOME_CODES``.
    dt
        Seconds per step. Read from the records if not given.
    W
        Recurrent weight matrix to store alongside the states.

    Returns
    -------
    TrajectoryBundle
        Keys: ``X``, ``time``, ``dt``, ``tau``, ``inputs``, ``outputs``,
        ``condition``, ``W``, ``aux``, ``meta``. Under ``align="cue"`` trials
        without an alignment point are dropped and counted in
        ``meta["n_unaligned_dropped"]``.
    """
    recs = [r for r in records if r is not None and "hidden" in r]
    if not recs:
        raise ValueError(
            "no records carry hidden states. Pass collect_states=True to "
            "evaluate/test/probe.")
    if align not in ("cue", "start"):
        raise ValueError(f"align must be 'cue' or 'start', got {align!r}")

    n_all = len(recs)
    if align == "cue":
        recs = [r for r in recs if r.get("align_step") is not None]
    n_unaligned = n_all - len(recs)
    if not recs:
        raise ValueError(
            f"none of the {n_all} trials reached a cue, so none can be "
            f"cue-aligned. Use align='start' to export them.")

    dt = float(dt if dt is not None else recs[0].get("dt", 0.02))
    N = int(np.asarray(recs[0]["hidden"]).shape[1])
    has_obs = all("obs" in r for r in recs)
    has_out = all("readout" in r and "value" in r for r in recs)
    n_in = int(np.asarray(recs[0]["obs"]).shape[1]) if has_obs else 0

    # -- common time frame ------------------------------------------------- #
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
        if align == "cue":
            a = r.get("align_step")
            if a is None:
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

    # -- per-trial label --------------------------------------------------- #
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
    # Number of valid (non-padding) steps per trial.
    aux["n_steps"] = np.array([int(np.isfinite(X[b, :, 0]).sum())
                               for b in range(B)], np.int64)
    # Lick times are ragged; store them NaN-padded with the count alongside.
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
    """Build a bundle from ``records`` and write it to ``path`` as HDF5.

    Keyword arguments are those of :func:`records_to_trajectory`. Returns
    ``path``.
    """
    bundle = (records if isinstance(records, TrajectoryBundle)
              else records_to_trajectory(records, **kwargs))
    return write_bundle(bundle, path, compression=compression)


def write_bundle(bundle: Dict[str, Any], path: str, *,
                 compression: str = "gzip") -> str:
    """Write a bundle to HDF5 in the schema described in the module docstring."""
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
        f.attrs["schema"] = "trajectory/1"
        # Compatibility flag recognised by existing readers of this schema.
        f.attrs["neuralgeom_trajectory"] = True
    return path
