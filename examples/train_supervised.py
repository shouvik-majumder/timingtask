"""Train a recurrent network on the timing task with supervised targets.

    python examples/train_supervised.py --steps 1000

The network receives the same observations as the reinforcement-learning
agent and is trained to produce a withhold output and a ramp-to-threshold
lick output. The required delay varies from trial to trial
(``mode="variable"``), so the network must time from the cue.
"""
import argparse

import torch

from timingtask import SchedulerConfig, TimingTaskConfig
from timingtask.models import make_model
from timingtask.supervised import TimingTask
from timingtask.training import train, run_trials


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    task_cfg = TimingTaskConfig(dt=0.02, cue_duration=0.6, answer_window=2.0,
                                post_lick=0.0, iti_mean=1.2, iti_min=0.5,
                                iti_max=2.5, no_cue_prob=0.1, seed=args.seed)
    sched_cfg = SchedulerConfig(mode="variable", min_delay=0.4, max_delay=1.2)
    task = TimingTask(task_cfg, sched_cfg, seed=args.seed, target_offset=0.15)

    model = make_model("vanilla", task.spec, hidden_size=args.hidden,
                       tau=100.0, dt=task.dt, noise=0.05)
    train(model, task, steps=args.steps, batch_size=args.batch_size,
          log_every=100)

    batch, out, hidden = run_trials(model, task, batch_size=256)
    counts = task.outcome_counts(out, batch)
    print(f"outcomes: {counts}")
    print(f"hidden states: {tuple(hidden.shape)}  (trials, steps, units)")


if __name__ == "__main__":
    main()
