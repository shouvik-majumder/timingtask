"""
timingtask.monitor: per-trial record logging.

The generator emits one flat record per completed trial. Monitors subscribe to
that stream. A monitor cannot affect training: one that raises is reported
once and removed for the rest of the run.

    mons = MonitorList([JSONLMonitor("runs/fixed_01.jsonl")])
    env = TimingTaskEnv(monitors=mons)
    ...
    mons.close()
    records = read_jsonl("runs/fixed_01.jsonl")
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

__all__ = ["Monitor", "JSONLMonitor", "MemoryMonitor", "MonitorList",
           "read_jsonl", "records_to_arrays"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


class Monitor:
    """Subscriber interface. All hooks are optional."""

    def on_reset(self, info: Dict[str, Any]) -> None: ...
    def on_trial(self, record: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class MemoryMonitor(Monitor):
    """Keeps records in a list."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.resets: List[Dict[str, Any]] = []

    def on_reset(self, info):
        self.resets.append(dict(info))

    def on_trial(self, record):
        self.records.append(dict(record))


class JSONLMonitor(Monitor):
    """Appends records to a newline-delimited JSON file."""

    def __init__(self, path: str, flush_every: int = 20):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._flush_every = int(flush_every)
        self._n = 0

    def on_reset(self, info):
        self._write({"_event": "reset", **_jsonable(info)})

    def on_trial(self, record):
        self._write({"_event": "trial", **_jsonable(record)})

    def _write(self, obj):
        self._f.write(json.dumps(obj) + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.flush()
            self._f.close()


class MonitorList(Monitor):
    """Fans records out to several monitors, isolating failures.

    A monitor that raises is reported once and removed from the list.
    """

    def __init__(self, monitors: Optional[Iterable[Monitor]] = None):
        self.monitors: List[Monitor] = list(monitors or [])
        self.failed: List[str] = []

    def append(self, m: Monitor) -> None:
        self.monitors.append(m)

    def _fan(self, hook: str, *args) -> None:
        for m in list(self.monitors):
            try:
                getattr(m, hook)(*args)
            except Exception as e:                      # noqa: BLE001
                name = type(m).__name__
                self.failed.append(f"{name}.{hook}: {type(e).__name__}: {e}")
                print(f"[monitor] {name} disabled after error in {hook}: {e}")
                self.monitors.remove(m)

    def on_reset(self, info):
        self._fan("on_reset", info)

    def on_trial(self, record):
        self._fan("on_trial", record)

    def close(self):
        self._fan("close")


# --------------------------------------------------------------------------- #
# reading records back
# --------------------------------------------------------------------------- #
def read_jsonl(path: str, event: str = "trial") -> List[Dict[str, Any]]:
    """Read a JSONL run file. ``event=None`` returns every line."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if event is None or obj.get("_event") == event:
                out.append(obj)
    return out


def records_to_arrays(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Columnar view of a record list.

    Scalar fields become arrays; ``lick_times_s`` remains a list of lists
    because trials differ in their number of licks. Missing values become NaN.
    """
    if not records:
        return {}
    scalar_keys = [k for k, v in records[0].items()
                   if not isinstance(v, (list, dict)) and not k.startswith("_")]
    out: Dict[str, Any] = {}
    for k in scalar_keys:
        vals = [r.get(k) for r in records]
        if all(isinstance(v, (bool, np.bool_)) or v is None for v in vals):
            out[k] = np.array([bool(v) if v is not None else False for v in vals])
        elif all(isinstance(v, str) or v is None for v in vals):
            out[k] = np.array([v if v is not None else "" for v in vals], dtype=object)
        else:
            out[k] = np.array([np.nan if v is None else float(v) for v in vals],
                              dtype=float)
    out["lick_times_s"] = [list(r.get("lick_times_s") or []) for r in records]
    return out
