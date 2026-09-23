"""
Timing task + RNN training -- interactive debug script.
=======================================================

A `# %%` cell script: run it top to bottom, or open it in VS Code and run cells
one at a time (Shift+Enter) to get a notebook without the notebook.

PURPOSE IS DEBUGGING. Section 1 exists so the task structure can be *inspected*
rather than trusted -- a step-by-step trace of one trial, then timelines, then
the across-trial view. Read those before believing anything downstream.

    python examples/timing_debug.py    # from the repository root; runs everything, saves figures
"""
# %%
# ---------------------------------------------------------------- setup
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from timingtask import (DelayScheduler, MemoryMonitor, MonitorList,
                                     ObservationConfig, SchedulerConfig,
                                     TimingTaskConfig, TrialGenerator)
from timingtask import plots as P
from timingtask.env import TimingTaskEnv
from timingtask.supervised import TimingTask
from timingtask.models import make_model
from timingtask.training import train

OUT = os.path.join("outputs", "figures", "timing_debug")
os.makedirs(OUT, exist_ok=True)
SAVE = True          # False when working interactively
torch.manual_seed(0)
np.random.seed(0)


def show(fig, name):
    if SAVE:
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=130,
                    bbox_inches="tight")
        print(f"  saved {name}.png")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg"):
        plt.show()                      # interactive session
    else:
        plt.close(fig)                  # plain script: don't warn, don't leak


# The debug configuration: FIXED DELAY, no post-lick period (not needed until
# the geometry epochs matter), catch trials kept because they are the
# zero-input control and a good check that the timer never starts without a cue.
TASK = TimingTaskConfig(
    dt=0.02,
    cue_duration=0.6,
    answer_window=5.0,
    post_lick=0.0,          # set >0 to re-enable peri/post-lick epochs
    iti_mean=1.2, iti_min=0.5, iti_max=2.5,
    no_cue_prob=0.1,
    seed=0,
)
# Delay 0.4 s DELIBERATELY SHORTER than the 0.6 s cue: that is the structural
# case the old phase-machine could not express, and it should be visible in the
# timelines as reward becoming available while the cue is still audible.
SCHED = SchedulerConfig(mode="fixed", fixed_delay=0.4)
print("delay", SCHED.fixed_delay, "s | cue", TASK.cue_duration,
      "s | answer window", TASK.answer_window, "s | dt", TASK.dt)
print(f"NOTE cue {TASK.cue_duration}s > delay {SCHED.fixed_delay}s: reward "
      "becomes available while the cue is STILL AUDIBLE. Check the timelines.")

# %%
# =====================================================================
# 1a.  STEP-BY-STEP TRACE OF ONE TRIAL   <-- start here
# =====================================================================
# The most direct check on task structure. Watch for:
#   * phase goes iti -> waiting -> eligible, and NEVER has a separate cue phase
#   * cue_on is True from timer 0.000 for exactly cue_duration, independent of
#     which phase the timer is in
#   * eligible begins at timer == delay, to the step
#   * a lick in `waiting` gives early_penalty; in `eligible` gives the reward
# Catch trials are on, so force a CUED trial for the traces -- otherwise the
# demo may land on a no-cue trial where the timer never starts and the trace
# shows nothing. (That is itself a useful check: see 1c for a catch timeline.)
TRACE_TASK = TimingTaskConfig(**{**TASK.__dict__, "no_cue_prob": 0.0})
rng = np.random.default_rng(0)
gen = TrialGenerator(TRACE_TASK, DelayScheduler(SCHED, rng),
                     ObservationConfig(), rng)

print("\n--- perfect timer: withholds, then licks 0.1s into the answer window ---")
rows = P.trial_trace(
    gen, policy=lambda g: (g.timer is not None and g.decisive_lick_s is None
                           and g.timer >= g.delay + 0.1))
P.print_trial_trace(rows, transitions_only=True)

print("\n--- impatient: licks the moment the cue starts (should be `early`) ---")
rows_early = P.trial_trace(gen, policy=lambda g: g.timer is not None
                           and g.decisive_lick_s is None)
P.print_trial_trace(rows_early, transitions_only=True)

print("\n--- never licks (should be `miss` after the answer window) ---")
rows_miss = P.trial_trace(gen, policy=lambda g: False)
P.print_trial_trace(rows_miss, transitions_only=True)

# %%
# ---------------------------------------------------------------------
# 1b.  ITI restart: a lick must push the cue further away
# ---------------------------------------------------------------------
g2 = TrialGenerator(TRACE_TASK, DelayScheduler(SCHED, np.random.default_rng(1)),
                    ObservationConfig(), np.random.default_rng(1))
print(f"{'step':>5} {'iti_remaining':>14} {'licked':>7}")
for i in range(12):
    lick = i in (3, 4)
    print(f"{g2.step_index:>5} {g2._iti_remaining:>14} {str(lick):>7}"
          + ("   <-- lick RESTARTS the stop-licking period" if lick else ""))
    g2.step(lick)

# %%
# ---------------------------------------------------------------------
# 1c.  Trial timelines -- one panel per trial
# ---------------------------------------------------------------------
# Grey = ITI, orange = delay (lick aborts), blue = answer (lick rewards),
# dark bar at the top = the cue channel. The cue bar should visibly EXTEND
# past the dashed delay line whenever cue_duration >= delay.
mon = MemoryMonitor()
env = TimingTaskEnv(TASK, SCHED, trials_per_episode=400,
                    monitors=MonitorList([mon]))
env.reset(seed=0)


class Scripted:
    """A sloppy animal: picks ONE aim point per trial, then licks when the timer
    reaches it. Drawing the jitter fresh at every step instead would make this a
    per-step hazard -- the lick time becomes a first-passage over hundreds of
    draws and lands far earlier than the nominal aim. Easy mistake; it silently
    changes the behavioural model.
    """

    def __init__(self, mean=0.15, sd=0.25, seed=0):
        self.mean, self.sd = mean, sd
        self.rng = np.random.default_rng(seed)
        self.trial, self.aim = -1, None

    def __call__(self, e):
        if e.gen.trial_index != self.trial:            # new trial -> new aim
            self.trial = e.gen.trial_index
            self.aim = self.rng.normal(self.mean, self.sd)
        t = e.gen.timer
        if t is None or e.gen.decisive_lick_s is not None:
            return 0
        return int(t >= e.gen.delay + self.aim)


scripted = Scripted()


while True:
    _, _, _, trunc, _ = env.step(scripted(env))
    if trunc:
        break
print(f"{len(mon.records)} trials collected")

recs = mon.records
fig, _ = P.plot_trial_timelines(recs)
show(fig, "01_trial_timelines")

# %%
# ---------------------------------------------------------------------
# 1d.  Across-trial structure checks
# ---------------------------------------------------------------------
from collections import Counter
print("outcome counts:", Counter(r["outcome"] for r in recs))
cued = [r for r in recs if r["cue_onset_step"] is not None]
iti = np.array([r["cue_onset_step"] * TASK.dt for r in cued])
print(f"stop-licking period: mean {iti.mean():.2f}s  min {iti.min():.2f}  "
      f"max {iti.max():.2f}   (configured {TASK.iti_min}-{TASK.iti_max}, "
      f"mean {TASK.iti_mean}; longer than max => licks restarted it)")
fl = np.array([r["first_lick_s"] for r in cued if r["first_lick_s"] is not None])
print(f"first lick: mean {fl.mean():.3f}s  sd {fl.std():.3f}   "
      f"delay {SCHED.fixed_delay}")
early = [r for r in cued if r["early"]]
if early:
    e = np.array([r["decisive_lick_s"] for r in early])
    assert e.max() < SCHED.fixed_delay, "BUG: an 'early' lick landed after the delay"
    print(f"early licks all before the delay (max {e.max():.3f}) -- OK")
rew = [r for r in cued if r["rewarded"]]
if rew:
    w = np.array([r["decisive_lick_s"] for r in rew])
    assert w.min() >= SCHED.fixed_delay, "BUG: a reward was paid before the delay"
    print(f"rewarded licks all at/after the delay (min {w.min():.3f}) -- OK")

# %%
# =====================================================================
# 2.  BEHAVIOUR DASHBOARD -- does the monitoring work?
# =====================================================================
fig, _ = P.behaviour_dashboard(recs, title="scripted sloppy animal, fixed delay")
show(fig, "02_behaviour_scripted")

# %%
# =====================================================================
# 3.  RNN TRAINING, with the dashboards refreshed as it learns
# =====================================================================
task = TimingTask(TASK, SCHED, seed=0, target_offset=0.10,
                  ramp_exponent=1.0,      # linear: less late bias than 2.8
                  plateau=1.5,            # target keeps rising PAST threshold;
                                          # see supervised.py -- clamping it at
                                          # the threshold decouples loss from
                                          # accuracy and the network collapses
                  mask_mode="ramp")
model = make_model("vanilla", task.spec, hidden_size=128,
                   tau=100.0, dt=task.dt, noise=0.05)
print(f"input {task.spec.input_dim} -> hidden 128 -> output "
      f"{task.spec.output_dim} | alpha = dt/tau = {model.alpha}")

EVERY, TOTAL, BATCH = 100, 600, 16
history = {"step": [], "loss": [], "acc": []}
for block in range(TOTAL // EVERY):
    h = train(model, task, steps=EVERY, batch_size=BATCH,
              log_every=EVERY, verbose=True)
    done = (block + 1) * EVERY
    for k in history:
        history[k].extend(h[k] if k != "step" else [s + block * EVERY for s in h["step"]])

    model.eval()
    with torch.no_grad():
        b = task.sample(64)
        out, H = model(b.inputs)
    print(f"  [{done}] outcomes {task.outcome_counts(out, b)}")

    ct = task.crossing_time(out, b).numpy()
    onsets = b.meta["cue_onset_step"].numpy().astype(int)
    delay = b.meta["delay"].numpy()
    outc = np.where(np.isnan(ct), "miss",
                    np.where(ct < delay, "early", "rewarded"))
    fig, _ = P.model_dashboard(H.numpy(), out[..., 0].numpy(), onsets,
                               delays=delay / TASK.dt, outcomes=list(outc),
                               lengths=b.meta["trial_len"].numpy(),
                               threshold=task.threshold, pre=10, post=100,
                               title=f"RNN internals @ step {done}")
    show(fig, f"03_model_step{done:04d}")
    model.train()

# %%
# ---------------------------------------------------------------------
# 3b.  Learning curves
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].plot(history["step"], history["loss"], color="0.25", lw=1.6)
axs[0].set_yscale("log")
P._style(axs[0], "step", "masked MSE", "loss")
axs[1].plot(history["step"], history["acc"], color=P.OUTCOME_COLORS["rewarded"],
            lw=1.8)
axs[1].axhline(0.30, color="0.25", ls="--", lw=1.0, label="mouse expert ~30%")
axs[1].set_ylim(-.02, 1.02)
axs[1].legend(fontsize=7, frameon=False)
P._style(axs[1], "step", "fraction rewarded", "accuracy (task criterion)")
fig.tight_layout()
show(fig, "04_learning_curves")

# %%
# =====================================================================
# 4.  THE TRAINED NETWORK, behaving
# =====================================================================
# Push the network's own licks back through the environment, so the behaviour
# dashboard shows what the MODEL does rather than what a script does.
model.eval()
mon2 = MemoryMonitor()
env2 = TimingTaskEnv(TASK, SCHED, trials_per_episode=300,
                     action_mode="continuous", lick_threshold=task.threshold,
                     monitors=MonitorList([mon2]))
obs, _ = env2.reset(seed=1)
h = model.init_state(1)
with torch.no_grad():
    while True:
        x = torch.from_numpy(obs).float().unsqueeze(0)
        h = model.step(x, h)
        z = model.readout(h)
        obs, _, _, trunc, _ = env2.step(z.numpy().reshape(-1))
        if trunc:
            break
print(f"{len(mon2.records)} trials by the trained network")
fig, _ = P.behaviour_dashboard(mon2.records, title="trained RNN, fixed delay")
show(fig, "05_behaviour_trained")

from collections import Counter as _C
print("outcomes:", _C(r["outcome"] for r in mon2.records))

# %%
# =====================================================================
# 5.  POST-TRAINING ANALYSES
# =====================================================================
with torch.no_grad():
    b = task.sample(128)
    out, H = model(b.inputs)
Hn, Zn = H.numpy(), out[..., 0].numpy()
onsets = b.meta["cue_onset_step"].numpy().astype(int)
ct = task.crossing_time(out, b).numpy()

# trial_len is NOT optional: the batch pads every trial to the longest one with
# zeros and the RNN keeps running over the padding, so without it the PSTH,
# heatmap and readout all show the network responding to zero input.
lens = b.meta["trial_len"].numpy()
A, t = P.align_to_cue(Hn, onsets, pre=10, post=120, lengths=lens)
keep = onsets >= 0
# lick-time TERCILES, labelled by their median -- the repo's convention for
# lick-time conditions is to label by median, never "condition 1, 2, 3".
ctk = ct[keep]
q = np.nanquantile(ctk, [1 / 3, 2 / 3])
binlab = np.digitize(ctk, q)
# A tercile can be EMPTY -- if the network is deterministic every crossing is
# identical and all trials fall in one bin. nanmedian of an empty slice warns
# and returns nan, so label only the bins that exist.
med = {}
for k in np.unique(binlab):
    v = ctk[binlab == k]
    med[k] = f"{np.nanmedian(v):.2f}s" if np.isfinite(v).any() else "n/a"
groups = np.array([med[k] for k in binlab], dtype=object)
if len(med) == 1:
    print("  only one lick-time bin -- the network is deterministic, so there "
          "is nothing to split by. Leave noise on at evaluation to get a spread.")

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
P.plot_activity_heatmap(axs[0, 0], A, t)
P.plot_psth(axs[0, 1], A, t, groups=groups)
P.plot_pca_trajectories(axs[1, 0], A, t, groups=groups)
Za, _ = P.align_to_cue(Zn[..., None], onsets, pre=10, post=120, lengths=lens)
P.plot_readout(axs[1, 1], Za[..., 0], t, threshold=task.threshold,
               delays=b.meta["delay"].numpy()[keep] / TASK.dt)
fig.suptitle("trained RNN -- PSTH, trajectories, readout", fontsize=11,
             color="0.25", x=.02, ha="left")
fig.tight_layout()
show(fig, "06_analyses")

print(f"\ncrossing times: mean {np.nanmean(ct):.3f}s sd {np.nanstd(ct):.3f}  "
      f"| delay {SCHED.fixed_delay}s | target "
      f"{SCHED.fixed_delay + task.target_offset}s")
if np.nanstd(ct) < 1e-6:
    print("  NOTE sd == 0. model.eval() disables the recurrent noise and the "
          "fixed delay makes every cued trial's input identical, so the "
          "network is deterministic. There is no lick-time DISTRIBUTION to "
          "compare with an animal's until noise is left on at evaluation "
          "(model.train()) or the input itself varies.")
print(f"figures in {OUT}")
