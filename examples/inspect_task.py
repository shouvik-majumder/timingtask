"""Inspect the trial structure with scripted policies.

    python examples/inspect_task.py

Prints a step-by-step trace of single trials under three scripted policies,
then runs a session through the gymnasium environment and summarises the
trial records.
"""
from collections import Counter

import numpy as np

from timingtask import (DelayScheduler, MemoryMonitor, MonitorList,
                        ObservationConfig, SchedulerConfig, TimingTaskConfig,
                        TrialGenerator)
from timingtask.env import TimingTaskEnv
from timingtask.plots import print_trial_trace, trial_trace


def main():
    task = TimingTaskConfig(dt=0.02, cue_duration=0.6, answer_window=2.0,
                            post_lick=0.5, iti_mean=1.2, iti_min=0.5,
                            iti_max=2.5, no_cue_prob=0.0, seed=0)
    sched = SchedulerConfig(mode="fixed", fixed_delay=1.0)
    rng = np.random.default_rng(0)
    gen = TrialGenerator(task, DelayScheduler(sched, rng), ObservationConfig(), rng)
    print("observation channels:", gen.observation_labels())

    policies = {
        "lick 0.1 s after the delay": lambda g: (
            g.timer is not None and g.decisive_lick_s is None
            and g.timer >= g.delay + 0.1),
        "lick at cue onset (early)": lambda g: (
            g.timer is not None and g.decisive_lick_s is None),
        "never lick (miss)": lambda g: False,
    }
    for name, policy in policies.items():
        print(f"\n--- {name} ---")
        print_trial_trace(trial_trace(gen, policy=policy), transitions_only=True)

    # A session through the gymnasium environment with a scripted agent.
    mon = MemoryMonitor()
    env = TimingTaskEnv(TimingTaskConfig(seed=0),
                        SchedulerConfig(mode="fixed", fixed_delay=1.0),
                        trials_per_episode=200, monitors=MonitorList([mon]))
    env.reset(seed=0)
    aim = 0.2
    while True:
        g = env.gen
        lick = int(g.timer is not None and g.decisive_lick_s is None
                   and g.timer >= g.delay + aim)
        _, _, _, truncated, _ = env.step(lick)
        if truncated:
            break
    print(f"\n{len(mon.records)} trials:", dict(Counter(r["outcome"] for r in mon.records)))
    fl = [r["first_lick_s"] for r in mon.records if r["first_lick_s"] is not None]
    print(f"first lick: mean {np.mean(fl):.3f} s (delay 1.0 s, aim +{aim} s)")


if __name__ == "__main__":
    main()
