"""
timingtask — the cue-triggered lick-timing task, its agents, and their training.
================================================================================

A self-contained research package: generate trials, train agents on them by
whatever method, and export what the agent did and what its units did.

    config.py      three dataclasses; every number, no logic
    generator.py   the trial state machine (timer and cue as INDEPENDENT axes)
    scheduler.py   the delay curriculum
    monitor.py     JSONL / in-memory trial logging
    env.py         gymnasium wrapper          (gymnasium is an optional extra)
    models.py      the agent zoo — recurrent cores behind one ``step`` interface
    contract.py    TaskSpec / TrialBatch / Task, the batched-trial interface
    rl.py          REINFORCE with a learned baseline; ActorCritic
    training.py    the task-agnostic supervised trainer
    supervised.py  the task as a batched, supervised ``Task``
    variants.py    named configurations + the CLI
    plots.py       the diagnostic figures
    circuits.py    the two published timing models, re-derived numerically
    export.py      write states + behaviour as a Trajectory HDF5 file

This package depends on nothing but numpy / torch / h5py (and, lazily,
matplotlib, scikit-learn, gymnasium). In particular it does NOT import the
geometry library. The two meet at a FILE, not an import: ``export.py`` writes
the ``Trajectory`` HDF5 schema, and the geometry side reads it with
``neuralgeom.data.load_trajectory``. Train here; analyse there.
"""
from .config import (ObservationConfig, SchedulerConfig, TimingTaskConfig,
                     VARIANTS, make_config)
from .generator import Phase, StepResult, TrialGenerator
from .scheduler import DelayScheduler
from .monitor import (JSONLMonitor, MemoryMonitor, Monitor, MonitorList,
                      read_jsonl, records_to_arrays)
from .contract import Task, TaskSpec, TrialBatch
from .models import GRUModel, LSTMModel, MODELS, VanillaRNN, make_model
from .export import records_to_trajectory, save_trajectory

__version__ = "0.1.0"

__all__ = ["TimingTaskConfig", "SchedulerConfig", "ObservationConfig",
           "VARIANTS", "make_config", "TrialGenerator", "StepResult", "Phase",
           "DelayScheduler", "Monitor", "MonitorList", "MemoryMonitor",
           "JSONLMonitor", "read_jsonl", "records_to_arrays",
           "Task", "TaskSpec", "TrialBatch",
           "VanillaRNN", "GRUModel", "LSTMModel", "MODELS", "make_model",
           "records_to_trajectory", "save_trajectory",
           "__version__"]


def __getattr__(name):
    # gymnasium is an optional extra; importing the env should only fail for
    # someone who actually asks for it.
    if name == "TimingTaskEnv":
        from .env import TimingTaskEnv
        return TimingTaskEnv
    raise AttributeError(name)
