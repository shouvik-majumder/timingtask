"""Train an actor-critic agent on the timing task with REINFORCE and export
its hidden states.

    python examples/train_rl.py --steps 2000 --out runs/rl_demo

The agent is trained on the two-stage curriculum (cue association, then a
delay that increases by 0.1 s per promotion). After training, the agent is
evaluated with frozen weights at the final delay and its cue-aligned hidden
states, inputs and readouts are written to an HDF5 file.
"""
import argparse
import os

import torch

from timingtask import ObservationConfig, SchedulerConfig, TimingTaskConfig
from timingtask.export import save_trajectory
from timingtask.models import VanillaRNN
from timingtask.rl import ActorCritic, RLTrainer, summarise


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=2000, help="number of updates")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", default="runs/rl_demo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    task = TimingTaskConfig(
        dt=0.02, cue_duration=0.6, answer_window=0.8, post_lick=1.5,
        iti_mean=3.0, iti_min=2.0, iti_max=5.0, iti_restart_on_lick=False,
        no_cue_prob=0.1,
        reward=15.0, early_penalty=-2.0, iti_lick_penalty=-2.0,
        no_cue_lick_penalty=-2.0, miss_penalty=-2.0, time_penalty=-0.05,
        reward_rate_gain=1.0, seed=args.seed)
    sched = SchedulerConfig(
        mode="cue_autolearn", cue_association_delay=0.1,
        cue_association_success_threshold=0.3,
        initial_delay=0.1, delay_step=0.1, max_delay=2.0)
    obs = ObservationConfig(n_lags=1)

    trainer = RLTrainer(task, sched, obs, n_envs=args.n_envs, lr=4e-3,
                        reward_scale=0.1, activity_penalty=20.0, seed=args.seed)
    core = VanillaRNN(trainer.obs_size, args.hidden, 1, tau=100.0, dt=20.0,
                      noise=0.05)
    model = ActorCritic(core)

    hist = trainer.train(model, steps=args.steps, log_every=50,
                         target_delay=1.0, patience=40)

    records = trainer.evaluate(model, n_trials=400, collect_states=True)
    s = summarise(records)
    print(f"evaluation: delay {s['delay']:.2f} s  engaged {s['p_engaged']:.2f}  "
          f"correct {s['p_correct']:.2f}  first lick {s['first_lick']:.3f} s")

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    path = save_trajectory(
        records, os.path.join(args.out, "states.h5"),
        condition="delay",
        W=model.core.rec.weight.detach().cpu().numpy(),
        generator="timingtask.rl",
        config={"task": task.__dict__, "scheduler": sched.__dict__})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
