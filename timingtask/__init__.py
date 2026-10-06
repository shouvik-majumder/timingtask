"""
timingtask: a cue-triggered lick-timing task for recurrent agents.

The package provides the trial generator, a gymnasium environment, a batched
supervised interface, recurrent network models, reinforcement-learning and
supervised trainers, and an HDF5 exporter for trained-agent states and
behaviour.

Modules
-------
config       Configuration dataclasses for the task, the delay schedule and
             the observation vector.
generator    The trial state machine.
scheduler    Delay schedules, including the two-stage training curriculum.
monitor      Per-trial record logging (in memory or JSONL).
env          gymnasium ``Env`` wrapper (requires the ``gym`` extra).
models       Recurrent network models sharing one ``step`` interface.
contract     ``TaskSpec`` / ``TrialBatch`` / ``Task``, the batched-trial API.
rl           Actor-critic agent and REINFORCE trainer.
training     Task-agnostic supervised trainer.
supervised   The task as a batched supervised ``Task``.
variants     Named configurations and the command-line interface.
plots        Behavioural and network diagnostics (requires the ``plots`` extra).
circuits     Published low-dimensional timing circuits, for reference.
export       HDF5 export of hidden states, inputs, readouts and behaviour.

Core dependencies are numpy, torch and h5py. gymnasium, matplotlib and
scikit-learn are optional and imported lazily.
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

__version__ = "0.2.0"

__all__ = ["TimingTaskConfig", "SchedulerConfig", "ObservationConfig",
           "VARIANTS", "make_config", "TrialGenerator", "StepResult", "Phase",
           "DelayScheduler", "Monitor", "MonitorList", "MemoryMonitor",
           "JSONLMonitor", "read_jsonl", "records_to_arrays",
           "Task", "TaskSpec", "TrialBatch",
           "VanillaRNN", "GRUModel", "LSTMModel", "MODELS", "make_model",
           "records_to_trajectory", "save_trajectory",
           "__version__"]


def __getattr__(name):
    # gymnasium is an optional dependency; defer the import until requested.
    if name == "TimingTaskEnv":
        from .env import TimingTaskEnv
        return TimingTaskEnv
    raise AttributeError(name)
