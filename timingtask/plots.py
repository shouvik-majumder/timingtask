"""
timingtask.plots — rig-style monitoring and model diagnostics.
===========================================================================

Two dashboards, both drawing from the flat trial records the generator emits:

  * BEHAVIOUR -- what a lab watches while training an animal: lick raster,
    first-lick times against the criterion, rolling accuracy, outcome mix,
    lick-time distributions, ITI licking, reward.
  * MODEL -- what only a simulation can show: unit activity, the readout
    against its threshold, PSTHs, trajectories.

Every panel function takes an ``ax`` and draws into it, so panels compose into
whatever figure you want and nothing here opens or saves a file. The dashboard
helpers are thin wrappers that lay panels out.

Colours follow one rule: outcome is a CATEGORICAL variable with a FIXED slot
order, so "early" is the same colour in every figure, in every run, whatever
subset is present. The palette is Okabe-Ito, which is colourblind-safe
(validated: worst adjacent pair dE 11.0 deuteranopia). Anything ordered by
magnitude -- delay, lick-time bin, trial number -- uses a single-hue sequential
ramp instead, never a rainbow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "OUTCOME_COLORS", "OUTCOME_ORDER",
    "trial_trace", "print_trial_trace",
    "plot_trial_timeline", "plot_trial_timelines", "plot_lick_raster", "plot_first_lick_scatter",
    "plot_rolling_accuracy", "plot_outcome_proportions", "plot_first_lick_hist",
    "plot_iti_lick_rate", "plot_reward", "plot_iti_durations",
    "behaviour_dashboard", "plot_loss_terms", "plot_grad_norm",
    "plot_training_metrics", "plot_delay_spread", "training_dashboard",
    "plot_first_lick_hist_blocks", "plot_first_lick_hist_by_delay",
    "plot_history_split_hist", "plot_history_effect_over_trials",
    "history_regression", "plot_history_regression",
    "stack_readouts", "plot_readout_heatmap", "plot_readout_blocks",
    "history_dashboard", "trial_groups", "plot_weight_norms", "plot_weight_steps",
    "plot_readout_direction_drift", "plot_policy_bias", "plot_value_traces",
    "heads_dashboard", "plot_value_gain", "run_report",
    "align_to_cue", "plot_activity_heatmap", "plot_unit_traces",
    "_mark_licks",
    "plot_readout", "plot_psth", "plot_pca_trajectories", "model_dashboard",
]

# Fixed categorical slots. Never cycled, never reordered by frequency.
OUTCOME_ORDER = ("rewarded", "early", "miss", "no_cue_complete", "iti_timeout")
OUTCOME_COLORS = {
    "rewarded":        "#0072B2",   # blue
    "early":           "#D55E00",   # vermillion
    "miss":            "#E69F00",   # orange
    "no_cue_complete": "#009E73",   # green
    "iti_timeout":     "#CC79A7",   # purple
}
_GRID = dict(color="0.85", lw=0.6)
_INK = "0.25"


def _style(ax, xlabel="", ylabel="", title=""):
    """Recessive axes: the data should be the darkest thing in the frame."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.6")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="0.4", labelsize=8, length=3)
    ax.grid(True, axis="y", **_GRID)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=_INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=_INK)
    if title:
        ax.set_title(title, fontsize=9, color=_INK, loc="left")
    return ax


def _arr(records, key, default=np.nan):
    return np.array([r.get(key) if r.get(key) is not None else default
                     for r in records], dtype=float)


def _rolling(x, w):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    w = max(1, min(int(w), len(x)))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def _rolling_median(x, w):
    """Median in a sliding window. Preferred over the mean for lick times:
    the distribution is right-skewed and a single very late lick drags a mean
    by more than it should."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    w = max(1, min(int(w), len(x)))
    return np.array([np.median(x[i:i + w]) for i in range(len(x) - w + 1)])


# --------------------------------------------------------------------------- #
# 1. Task inspection -- the debugging tools
# --------------------------------------------------------------------------- #
def trial_trace(generator, policy=None, max_steps: int = 5000) -> List[Dict[str, Any]]:
    """Roll ONE trial and return a per-step trace.

    The first thing to look at when the task structure is in doubt: it shows,
    step by step, the phase, whether the cue is audible, the timer, the action
    and the reward -- so a phase boundary landing in the wrong place is visible
    directly rather than inferred from summary statistics.

    ``policy(generator) -> bool`` decides the lick; the default never licks.
    """
    if policy is None:
        policy = lambda g: False                                   # noqa: E731
    rows = []
    for _ in range(max_steps):
        g = generator
        lick = bool(policy(g))
        row = {"step": g.step_index, "phase": g.phase, "cue_on": g.cue_on,
               "timer": g.timer, "delay": g.delay, "lick": lick}
        res = g.step(lick)
        row["reward"] = res.reward
        row["outcome"] = res.info["outcome"]
        rows.append(row)
        if res.record is not None:
            row["record"] = res.record
            break
    return rows


def print_trial_trace(rows, every: int = 1, transitions_only: bool = False):
    """Print a trace. ``transitions_only`` collapses to the interesting steps:
    phase changes, cue on/off, licks, and the last step."""
    print(f"{'step':>5} {'phase':<9} {'cue':>4} {'timer':>7} {'delay':>6} "
          f"{'lick':>5} {'reward':>8}  outcome")
    prev = None
    for i, r in enumerate(rows):
        key = (r["phase"], r["cue_on"])
        interesting = (key != prev) or r["lick"] or (i == len(rows) - 1)
        prev = key
        if transitions_only and not interesting:
            continue
        if not transitions_only and i % every and not interesting:
            continue
        t = "  --  " if r["timer"] is None else f"{r['timer']:6.3f}"
        print(f"{r['step']:>5} {r['phase']:<9} {str(r['cue_on']):>4} {t:>7} "
              f"{r['delay']:6.2f} {str(r['lick']):>5} {r['reward']:8.3f}  "
              f"{r['outcome']}")


def plot_trial_timeline(ax, record: Dict[str, Any], *, show_legend=True,
                        xlim: Optional[Tuple[float, float]] = None):
    """One trial on a time axis: the timer's regions, the cue channel, licks.

    The cue is drawn as its own band ABOVE the timer regions, not as one of
    them, because it overlaps the timer arbitrarily -- if the delay is shorter
    than the cue, part of the rewarded window happens while the cue is still
    audible. A phase model cannot draw this picture.

    The shaded regions show the trial's STRUCTURE (what would have happened),
    which extends past where the trial actually ended; the dotted line marks the
    real end. Pass a common ``xlim`` so panels are comparable.
    """
    dt = record.get("dt", 0.02)
    delay = float(record["delay"])
    aw = float(record["answer_window"])
    cue_d = float(record.get("cue_duration", 0.6))
    onset = record.get("cue_onset_step")
    n = int(record["trial_steps"])

    if onset is None:                                  # catch trial
        t0, t1 = 0.0, n * dt
        ax.axvspan(t0, t1, color=OUTCOME_COLORS["no_cue_complete"], alpha=.12,
                   zorder=0)
        ax.text(.02, .72, "no-cue catch trial (zero-input control): no cue, "
                "no timer, nothing is rewardable",
                transform=ax.transAxes, fontsize=8, color=_INK)
        xlabel = "time from trial start (s)"
    else:
        t0, t1 = -onset * dt, (n - onset) * dt
        ax.axvspan(min(t0, xlim[0] if xlim else t0), 0, color="0.92", zorder=0)
        ax.axvspan(0, delay, color=OUTCOME_COLORS["early"], alpha=.12, zorder=0)
        ax.axvspan(delay, delay + aw, color=OUTCOME_COLORS["rewarded"],
                   alpha=.12, zorder=0)
        ax.axvline(0, color=_INK, lw=1.0, zorder=2)
        ax.axvline(delay, color=_INK, lw=1.0, ls="--", zorder=2)
        ax.axvline(t1, color=_INK, lw=.9, ls=":", zorder=2)      # trial ended
        # the cue band -- an independent axis, drawn above the timer regions
        ax.axvspan(0, cue_d, ymin=.86, ymax=1.0, color=_INK, alpha=.55, zorder=3)
        ax.text(cue_d / 2, .93, "cue", ha="center", va="center", fontsize=7,
                color="white", transform=ax.get_xaxis_transform(), zorder=4)
        xlabel = "time from cue (s)"

    licks = np.asarray(record.get("lick_times_s") or [], dtype=float)
    if licks.size:
        ax.vlines(licks, .05, .55, color=_INK, lw=1.2,
                  transform=ax.get_xaxis_transform(), zorder=5)
    fl = record.get("first_lick_s")
    if fl is not None:
        c = OUTCOME_COLORS.get(record["outcome"], _INK)
        ax.plot([fl], [.62], marker="v", ms=8, color=c, mec="white", mew=.8,
                transform=ax.get_xaxis_transform(), zorder=6, clip_on=False)

    ax.set_xlim(*(xlim if xlim else (t0, t1)))
    ax.set_yticks([])
    _style(ax, xlabel, "",
           f"trial {record['trial']} · {record['outcome']} · delay {delay:.2f}s"
           f" · cue {cue_d:.2f}s")
    ax.grid(False)
    if show_legend:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Patch(color="0.92", label="stop-licking period"),
            Patch(color=OUTCOME_COLORS["early"], alpha=.3, label="delay — lick aborts"),
            Patch(color=OUTCOME_COLORS["rewarded"], alpha=.3, label="answer — lick rewards"),
            Patch(color=_INK, alpha=.55, label="cue audible"),
            Line2D([], [], color=_INK, ls=":", label="trial end")],
            fontsize=7, frameon=False, ncol=5, loc="lower center",
            bbox_to_anchor=(.5, 1.18))
    return ax


def plot_trial_timelines(records, *, outcomes=OUTCOME_ORDER, figsize=(11, 2.1)):
    """One panel per requested outcome, on a COMMON time axis.

    Reports any outcome class that did not occur, rather than silently drawing
    fewer panels -- an absent class is usually the interesting part.
    """
    import matplotlib.pyplot as plt
    picks, missing = [], []
    for name in outcomes:
        hit = next((r for r in records if r["outcome"] == name), None)
        (picks.append(hit) if hit else missing.append(name))
    if missing:
        print(f"  no example of: {', '.join(missing)}")
    if not picks:
        raise ValueError("no trials to draw")
    cued = [r for r in picks if r.get("cue_onset_step") is not None]
    lo = min([-(r["cue_onset_step"]) * r.get("dt", .02) for r in cued] or [-1.5])
    hi = max([float(r["delay"]) + float(r["answer_window"]) for r in cued] or [3.])
    fig, axs = plt.subplots(len(picks), 1,
                            figsize=(figsize[0], figsize[1] * len(picks)))
    for i, (ax, r) in enumerate(zip(np.atleast_1d(axs), picks)):
        plot_trial_timeline(ax, r, show_legend=(i == 0),
                            xlim=(lo, min(hi, lo + 8.0)))
    fig.tight_layout()
    return fig, np.atleast_1d(axs)


# --------------------------------------------------------------------------- #
# 2. Behaviour dashboard -- the rig view
# --------------------------------------------------------------------------- #
def plot_lick_raster(ax, records, *, max_trials: int = 400):
    """Every lick, cue-aligned, trial on y, with the criterion staircase.

    The criterion drawn against the behaviour is the point: on an autolearn
    schedule you want to see the licks tracking the staircase up.
    """
    recs = [r for r in records if r.get("cue_onset_step") is not None][-max_trials:]
    for i, r in enumerate(recs):
        lt = np.asarray(r.get("lick_times_s") or [], dtype=float)
        if lt.size:
            ax.plot(lt, np.full_like(lt, i), ".", ms=1.6, color="0.55", zorder=1)
        fl = r.get("first_lick_s")
        if fl is not None:
            ax.plot(fl, i, ".", ms=3.2,
                    color=OUTCOME_COLORS.get(r["outcome"], _INK), zorder=2)
    if recs:
        d = _arr(recs, "delay")
        ax.step(d, np.arange(len(recs)), where="mid", color=_INK, lw=1.4, zorder=3,
                label="required delay")
        ax.axvline(0, color=_INK, lw=.8, ls=":", zorder=3)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style(ax, "time from cue (s)", "trial", "lick raster")
    ax.grid(False)
    return ax


def plot_first_lick_scatter(ax, records, *, window: int = 50,
                            stat: str = "median"):
    """First-lick time per trial, coloured by outcome, against the criterion."""
    recs = [r for r in records if r.get("cue_onset_step") is not None]
    idx = np.arange(len(recs))
    fl = _arr(recs, "first_lick_s")
    out = [r["outcome"] for r in recs]
    for name in OUTCOME_ORDER:                       # fixed slot order
        m = np.array([o == name for o in out])
        if m.any():
            ax.plot(idx[m], fl[m], ".", ms=3.5, color=OUTCOME_COLORS[name],
                    label=name, zorder=2)
    if len(recs):
        ax.step(idx, _arr(recs, "delay"), where="mid", color=_INK, lw=1.4,
                zorder=3, label="required delay")
        ok = ~np.isnan(fl)
        if ok.sum() > window:
            fn = _rolling_median if stat == "median" else _rolling
            ax.plot(idx[ok][window - 1:], fn(fl[ok], window), color=_INK,
                    lw=1.4, ls="--", zorder=4,
                    label=f"rolling {stat} ({window})")
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    _style(ax, "trial", "first lick (s from cue)", "first-lick time")
    return ax


def plot_rolling_accuracy(ax, records, *, window: int = 100,
                          criterion: float = 0.30):
    recs = [r for r in records if not r.get("no_cue_trial")]
    y = np.array([bool(r.get("rewarded")) for r in recs], float)
    if len(y) >= 2:
        w = min(window, len(y))
        ax.plot(np.arange(w - 1, len(y)), _rolling(y, w),
                color=OUTCOME_COLORS["rewarded"], lw=1.6, label=f"last {w}")
    ax.axhline(criterion, color=_INK, lw=1.0, ls="--",
               label=f"promotion criterion {criterion:.0%}")
    ax.set_ylim(-.02, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax, "trial", "fraction rewarded", "accuracy")
    return ax


def plot_outcome_proportions(ax, records, *, window: int = 50):
    """Stacked outcome mix over trials. Fixed slot order, so the bands mean the
    same thing between runs."""
    if len(records) < 2:
        return _style(ax, "trial", "proportion", "outcome mix")
    names = [r["outcome"] for r in records]
    w = min(window, len(names))
    series, present = [], []
    for name in OUTCOME_ORDER:
        y = np.array([n == name for n in names], float)
        if y.sum() == 0:
            continue
        series.append(_rolling(y, w))
        present.append(name)
    if series:
        x = np.arange(w - 1, len(names))
        ax.stackplot(x, *series, labels=present,
                     colors=[OUTCOME_COLORS[n] for n in present],
                     edgecolor="white", linewidth=.5)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.set_ylim(0, 1)
    _style(ax, "trial", "proportion", f"outcome mix (rolling {w})")
    return ax


def plot_first_lick_hist(ax, records, *, recent: int = 200, bins: int = 40):
    recs = [r for r in records if r.get("cue_onset_step") is not None]
    fl = _arr(recs, "first_lick_s")
    fl = fl[~np.isnan(fl)]
    if fl.size == 0:
        return _style(ax, "first lick (s)", "density", "lick-time distribution")
    edges = np.linspace(0, max(fl.max(), 1e-3) * 1.05, bins + 1)
    ax.hist(fl, bins=edges, density=True, color="0.75", label="all trials")
    tail = _arr(recs[-recent:], "first_lick_s")
    tail = tail[~np.isnan(tail)]
    if tail.size > 5:
        ax.hist(tail, bins=edges, density=True, histtype="step", lw=1.6,
                color=OUTCOME_COLORS["rewarded"], label=f"last {len(tail)}")
    if recs:
        ax.axvline(recs[-1]["delay"], color=_INK, lw=1.2, ls="--",
                   label="current delay")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "first lick (s from cue)", "density", "lick-time distribution")
    return ax


def plot_iti_lick_rate(ax, records, *, window: int = 50):
    """Pre-cue licking. Gates promotion out of cue association, and a run where
    this stays high is an agent that never really engages with the cue."""
    y = np.array([bool(r.get("iti_licked")) for r in records], float)
    if len(y) >= 2:
        w = min(window, len(y))
        ax.plot(np.arange(w - 1, len(y)), _rolling(y, w),
                color=OUTCOME_COLORS["early"], lw=1.6)
    ax.set_ylim(-.02, 1.02)
    _style(ax, "trial", "P(licked in ITI)", "pre-cue licking")
    return ax


def plot_reward(ax, records, *, window: int = 50):
    """Reward RATE, not just the cumulative curve -- a cumulative plot rises
    even while performance is falling."""
    r = _arr(records, "trial_reward")
    if len(r) >= 2:
        w = min(window, len(r))
        ax.plot(np.arange(w - 1, len(r)), _rolling(r, w), color=_INK, lw=1.5,
                label=f"per-trial (rolling {w})")
        ax.legend(fontsize=7, frameon=False)
    ax.axhline(0, color="0.7", lw=.8)
    _style(ax, "trial", "reward / trial", "reward rate")
    return ax


def plot_iti_durations(ax, records, *, bins: int = 30):
    """Time from trial start to cue. Should look like a truncated exponential;
    a heavy right tail means licking is restarting the clock a lot."""
    d = np.array([r["cue_onset_step"] * r.get("dt", .02) for r in records
                  if r.get("cue_onset_step") is not None], float)
    if d.size:
        ax.hist(d, bins=bins, color="0.75")
        ax.axvline(d.mean(), color=_INK, lw=1.2, ls="--",
                   label=f"mean {d.mean():.2f}s")
        ax.legend(fontsize=7, frameon=False)
    _style(ax, "trial start to cue (s)", "trials", "stop-licking period")
    return ax


def behaviour_dashboard(records, *, figsize=(15, 11), window: int = 50,
                        title: Optional[str] = None):
    """The rig view. Returns ``(fig, axes_dict)``."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    a = {"raster": axs[0, 0], "first_lick": axs[0, 1],
         "accuracy": axs[1, 0], "outcomes": axs[1, 1],
         "hist": axs[2, 0], "iti_lick": axs[2, 1],
         "reward": axs[3, 0], "iti_dur": axs[3, 1]}
    plot_lick_raster(a["raster"], records)
    plot_first_lick_scatter(a["first_lick"], records, window=window)
    plot_rolling_accuracy(a["accuracy"], records)
    plot_outcome_proportions(a["outcomes"], records, window=window)
    plot_first_lick_hist(a["hist"], records)
    plot_iti_lick_rate(a["iti_lick"], records, window=window)
    plot_reward(a["reward"], records, window=window)
    plot_iti_durations(a["iti_dur"], records)
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig, a


# --------------------------------------------------------------------------- #
# 3. Model internals
# --------------------------------------------------------------------------- #
def align_to_cue(H, onsets, *, pre: int = 10, post: int = 120, lengths=None,
                 min_trials: float = 0.5):
    """Re-slice ``(B, T, N)`` activity to a common cue-aligned window.

    Trials have different ITI lengths, so cue onset lands at a different index
    in every one. Averaging without aligning first smears the cue response into
    nothing -- the commonest way to make a PSTH look flat.

    ``lengths`` IS NOT OPTIONAL IN PRACTICE. A batch pads every trial out to the
    longest one with ZEROS, and a recurrent network keeps running over that
    padding, producing hidden states and a readout that are the network's
    response to zero input rather than anything about the task. The loss mask
    hides this during training; a PSTH does not. Symptom: every trace does the
    same thing at the same step -- a bump, a crash below zero, a slow drift --
    starting exactly where the shortest trials end. Pass ``batch.meta["trial_len"]``
    and it becomes NaN instead.

    ``min_trials`` trims timepoints where too few trials are still running.
    A value below 1 is a FRACTION of trials (the default, 0.5, keeps only
    timepoints where at least half the trials are live); 1 or more is an
    absolute count. Without it the tail of every average is carried by the few
    longest trials and shows up as a spike.

    Returns ``(A, t)``: ``A`` is ``(n_kept, n_steps, N)``, NaN outside each
    trial's real extent, and ``t`` is the step offset from cue onset.
    """
    H = np.asarray(H)
    onsets = np.asarray(onsets)
    B, T, N = H.shape
    if lengths is None:
        lengths = np.full(B, T)
    lengths = np.asarray(lengths).astype(int)
    out = np.full((B, pre + post, N), np.nan)
    for i in range(B):
        o = int(onsets[i])
        if o < 0:
            continue
        end = min(T, int(lengths[i]))          # real data stops here
        lo, hi = o - pre, o + post
        src_lo, src_hi = max(0, lo), min(end, hi)
        if src_hi > src_lo:
            out[i, src_lo - lo: src_hi - lo] = H[i, src_lo:src_hi]
    keep = ~np.isnan(out).all(axis=(1, 2))
    out = out[keep]
    t = np.arange(-pre, post)
    if min_trials and len(out):
        # A timepoint kept alive by a handful of long trials is not an average;
        # it produces a bright stripe at the tail of every PSTH and heatmap.
        # A FRACTION (< 1) is the sane default -- an absolute count scales
        # wrongly with batch size.
        need = (max(1, int(round(min_trials * len(out)))) if min_trials < 1
                else min(int(min_trials), len(out)))
        n_live = (~np.isnan(out[:, :, 0])).sum(axis=0)
        enough = n_live >= need
        if enough.any():
            lo, hi = int(np.argmax(enough)), int(len(enough) - np.argmax(enough[::-1]))
            out, t = out[:, lo:hi], t[lo:hi]
    return out, t


def _nanreduce(fn, a, axis=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _nanmin(a, axis=None):
    return _nanreduce(np.nanmin, a, axis)


def _nanmax(a, axis=None):
    return _nanreduce(np.nanmax, a, axis)


def _nanmean(a, axis=None):
    """np.nanmean without the all-NaN-slice warning: a fully-padded column is
    expected once trial lengths are respected, and NaN is the right answer."""
    return _nanreduce(np.nanmean, a, axis)


def _peak_sort(A, *, split_half: bool = True):
    """Order units by latency to peak. Sorting and displaying on the SAME data
    manufactures a diagonal out of noise, so sort on one half of the trials and
    display the other."""
    n = A.shape[0]
    if split_half and n >= 4:
        sort_src, show_src = A[0::2], A[1::2]
    else:
        sort_src = show_src = A
    m_sort = _nanmean(sort_src, axis=0)                      # (T, N)
    m_show = _nanmean(show_src, axis=0)
    finite = ~np.isnan(m_sort).all(axis=0)
    peak = np.zeros(m_sort.shape[1], int)
    peak[finite] = np.nanargmax(m_sort[:, finite], axis=0)
    order = np.argsort(peak)
    return m_show[:, order], order


def _mark_licks(ax, lick_steps, *, y=None, color=None, label="lick"):
    """Draw the lick times. Without these every panel is a picture of activity
    with no reference to the behaviour that produced it."""
    if lick_steps is None:
        return
    L = np.asarray([x for x in np.ravel(lick_steps) if np.isfinite(x)], float)
    if not L.size:
        return
    med = float(np.median(L))
    ax.axvline(med, color=color or OUTCOME_COLORS["early"], lw=1.4, ls="-",
               zorder=6, label=f"{label} (median {med:.0f})")
    lo, hi = np.percentile(L, [25, 75])
    ax.axvspan(lo, hi, color=color or OUTCOME_COLORS["early"], alpha=.10, zorder=0)


def plot_activity_heatmap(ax, A, t, *, split_half: bool = True, cmap="viridis",
                          lick_steps=None):
    """Units x time, sorted by latency to peak. Sequential ramp, single hue."""
    M, order = _peak_sort(A, split_half=split_half)
    # np.ptp and np.min do NOT skip NaN -- a single NaN in a column made the
    # whole normalised column NaN and the image rendered blank.
    lo = _nanmin(M, axis=0)
    hi = _nanmax(M, axis=0)
    z = (M - lo) / (hi - lo + 1e-9)
    im = ax.imshow(np.ma.masked_invalid(z.T), aspect="auto", origin="lower",
                   cmap=cmap, extent=[t[0], t[-1], 0, z.shape[1]],
                   vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(0, color="white", lw=1.0, ls="--")
    _mark_licks(ax, lick_steps, color="white")
    _style(ax, "steps from cue", "unit (sorted by peak)",
           "activity, peak-latency sorted" + (" (split-half)" if split_half else ""))
    ax.grid(False)
    return im


def plot_unit_traces(ax, A, t, *, units: Optional[Sequence[int]] = None,
                     n: int = 6, lick_steps=None):
    """A few single units. Sequential ramp by unit index, not a rainbow."""
    import matplotlib.cm as cm
    M = _nanmean(A, axis=0)
    if units is None:
        units = np.argsort(-np.ptp(M, axis=0))[:n]              # most modulated
    cols = cm.viridis(np.linspace(.15, .85, len(units)))
    for c, u in zip(cols, units):
        ax.plot(t, M[:, u], color=c, lw=1.4, label=f"u{u}")
    ax.axvline(0, color=_INK, lw=.9, ls="--")
    _mark_licks(ax, lick_steps)
    ax.legend(fontsize=6, frameon=False, ncol=3)
    _style(ax, "steps from cue", "activity", "example units (most modulated)")
    return ax


def plot_readout(ax, Z, t, *, threshold: float = 1.0, delays=None,
                 outcomes=None, max_traces: int = 60, lick_steps=None):
    """The readout against its threshold -- where ramp-to-bound is visible.

    Traces are coloured by outcome so early crossings are distinguishable from
    rewarded ones at a glance.
    """
    Z = np.asarray(Z)
    n = min(len(Z), max_traces)
    for i in range(n):
        c = _INK if outcomes is None else OUTCOME_COLORS.get(outcomes[i], _INK)
        ax.plot(t, Z[i], color=c, lw=.7, alpha=.5)
    ax.axhline(threshold, color=_INK, lw=1.3, ls="-",
               label=f"threshold {threshold:g}")
    ax.axvline(0, color=_INK, lw=.9, ls=":")
    if delays is not None and len(delays):
        ax.axvline(float(np.median(delays)), color=_INK, lw=1.1, ls="--",
                   label="delay (median)")
    _mark_licks(ax, lick_steps)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    rng = float(np.nanmax(np.nanmean(Z, 0)) - np.nanmin(np.nanmean(Z, 0)))
    _style(ax, "steps from cue", "readout z",
           f"readout vs threshold  (range {rng:.2f})")
    return ax


def plot_psth(ax, A, t, *, groups=None, group_name="lick-time bin",
              cmap="viridis", lick_steps=None):
    """Population-mean PSTH, optionally split by a group label.

    Group colours use a sequential ramp because the groups are ORDERED (early
    to late lick times); a categorical palette would imply they are not.
    """
    import matplotlib.cm as cm
    if groups is None:
        ax.plot(t, _nanmean(A, axis=(0, 2)), color=_INK, lw=1.8)
    else:
        g = np.asarray(groups)
        levels = [v for v in np.unique(g) if not (isinstance(v, float) and np.isnan(v))]
        cols = getattr(cm, cmap)(np.linspace(.15, .85, max(len(levels), 1)))
        for c, v in zip(cols, levels):
            m = g == v
            if m.sum() == 0:
                continue
            ax.plot(t, _nanmean(A[m], axis=(0, 2)), color=c, lw=1.6,
                    label=f"{v}")
        ax.legend(fontsize=7, frameon=False, title=group_name,
                  title_fontsize=7)
    ax.axvline(0, color=_INK, lw=.9, ls="--")
    _mark_licks(ax, lick_steps)
    _style(ax, "steps from cue", "mean activity", "population PSTH")
    return ax


def plot_pca_trajectories(ax, A, t, *, groups=None, n_show: int = 40,
                          cmap="viridis", lick_steps=None, center: bool = False):
    """Trajectories in the top two PCs of the cue-aligned activity.

    ``center=True`` removes each trial's own mean before the PCA. Use it when
    the leading PC is a STATIC per-trial offset rather than dynamics -- which
    happens whenever a constant input channel (the previous trial's outcome,
    held fixed for the whole trial) differs between trials. The symptom is
    scattered short arcs that look disconnected: the plot is showing where each
    trial sits, not where it goes. Compare the printed across/within ratio; if
    it is much greater than 1 the uncentred plot is dominated by the offset.
    """
    import matplotlib.cm as cm
    from sklearn.decomposition import PCA
    B, T, N = A.shape
    if center:
        A = A - _nanmean(A, axis=1)[:, None, :]
    flat = A.reshape(-1, N)
    ok = ~np.isnan(flat).any(1)
    if ok.sum() < 3:
        return _style(ax, "PC1", "PC2", "state-space trajectories (no data)")
    p = PCA(n_components=2).fit(flat[ok])
    Y = np.full((B * T, 2), np.nan)
    Y[ok] = p.transform(flat[ok])
    Y = Y.reshape(B, T, 2)

    if groups is None:
        cols = [_INK] * B
    else:
        g = np.asarray(groups)
        levels = list(np.unique(g))
        ramp = getattr(cm, cmap)(np.linspace(.15, .85, max(len(levels), 1)))
        cols = [ramp[levels.index(v)] for v in g]
    for i in range(min(B, n_show)):
        ax.plot(Y[i, :, 0], Y[i, :, 1], color=cols[i], lw=.8, alpha=.6)
    zero = int(np.argmin(np.abs(t)))
    ax.plot(Y[:n_show, zero, 0], Y[:n_show, zero, 1], "o", ms=4, color=_INK,
            mec="white", mew=.6, label="cue onset", zorder=5)
    if lick_steps is not None:
        xs, ys = [], []
        for i, L in enumerate(np.ravel(lick_steps)[:n_show]):
            if not np.isfinite(L):
                continue
            k = int(np.argmin(np.abs(t - L)))
            if np.isfinite(Y[i, k, 0]):
                xs.append(Y[i, k, 0]); ys.append(Y[i, k, 1])
        if xs:
            ax.plot(xs, ys, "v", ms=6, color=OUTCOME_COLORS["early"],
                    mec="white", mew=.6, label="lick", zorder=6)
    ax.legend(fontsize=7, frameon=False)
    evr = p.explained_variance_ratio_
    across = float(np.nanstd(_nanmean(Y[:, :, 0], axis=1)))
    within = float(_nanmean(np.nanstd(Y[:, :, 0], axis=1)))
    ratio = across / max(within, 1e-9)
    tag = " [centred]" if center else (f"  across/within {ratio:.0f}x"
                                       if ratio > 3 else "")
    _style(ax, f"PC1 ({evr[0]:.0%})", f"PC2 ({evr[1]:.0%})",
           f"state-space trajectories{tag}")
    return ax


def model_dashboard(H, Z, onsets, *, delays=None, outcomes=None, groups=None,
                    lengths=None, lick_steps=None, threshold: float = 1.0,
                    pre: int = 10, post: int = 120, min_trials: float = 0.5,
                    figsize=(15, 8), title: Optional[str] = None):
    """The model view. ``H`` is ``(B, T, N)`` hidden activity, ``Z`` is
    ``(B, T)`` readout, ``onsets`` the per-trial cue-onset step.

    Pass ``lengths=batch.meta["trial_len"]`` -- without it every panel here
    includes the network's response to post-trial zero padding. See
    :func:`align_to_cue`.
    """
    import matplotlib.pyplot as plt
    A, t = align_to_cue(H, onsets, pre=pre, post=post, lengths=lengths,
                        min_trials=min_trials)
    Za, _ = align_to_cue(np.asarray(Z)[..., None], onsets, pre=pre, post=post,
                         lengths=lengths, min_trials=min_trials)
    Za = Za[..., 0]
    keep = np.asarray(onsets) >= 0
    outc = None if outcomes is None else [o for o, k in zip(outcomes, keep) if k]
    grp = None if groups is None else np.asarray(groups)[keep]

    lick = None if lick_steps is None else np.asarray(lick_steps)[keep]
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    plot_activity_heatmap(axs[0, 0], A, t, lick_steps=lick)
    plot_unit_traces(axs[0, 1], A, t, lick_steps=lick)
    plot_readout(axs[0, 2], Za, t, threshold=threshold, delays=delays,
                 outcomes=outc, lick_steps=lick)
    plot_psth(axs[1, 0], A, t, groups=grp, lick_steps=lick)
    plot_pca_trajectories(axs[1, 1], A, t, groups=grp, lick_steps=lick)
    # Same data, per-trial mean removed: if the uncentred panel is dominated by
    # a static offset this is where the actual dynamics become visible.
    plot_pca_trajectories(axs[1, 2], A, t, groups=grp, lick_steps=lick,
                          center=True)
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig, axs


# --------------------------------------------------------------------------- #
# 4. Training diagnostics -- what the optimiser was doing, next to behaviour
# --------------------------------------------------------------------------- #
# Distinct from the outcome palette on purpose: these are loss terms, not
# behavioural categories, and colouring them from the same set invites reading
# a relationship that is not there.
_TERM_COLORS = {
    "policy_loss":  "#0072B2",
    "value_loss":   "#D55E00",
    "entropy":      "#009E73",
    "act_pen":      "#CC79A7",
    "dact_pen":     "#56B4E9",
    "grad_norm":    "0.25",
    "return":       "#E69F00",
}
_TERM_LABELS = {
    "policy_loss": "policy",
    "value_loss":  "value (MSE on return)",
    "entropy":     "entropy (nats)",
    "act_pen":     r"mean $\|h\|^2/N$",
    "dact_pen":    r"mean $\|\Delta h\|^2/N$",
}


def plot_loss_terms(ax, hist, *, keys=("policy_loss", "value_loss", "entropy",
                                       "act_pen", "dact_pen"), logy=True):
    """Each loss term against update index, on one axis.

    The terms are plotted RAW, not multiplied by their coefficients: the raw
    value is the quantity being controlled (mean squared activity, entropy in
    nats), and multiplying it by the coefficient hides whether the quantity
    moved or the price did. Log scale because they span three decades.
    """
    x = np.asarray(hist.get("step", []), float)
    any_plotted = False
    for k in keys:
        y = hist.get(k)
        if not y:
            continue
        y = np.asarray(y, float)
        n = min(len(x), len(y))
        ax.plot(x[:n], np.abs(y[:n]) if logy else y[:n],
                color=_TERM_COLORS.get(k, None), lw=1.4,
                label=_TERM_LABELS.get(k, k))
        any_plotted = True
    if logy and any_plotted:
        ax.set_yscale("log")
    if any_plotted:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax, "update", "|term|" if logy else "term",
           "loss terms (raw, before coefficients)")
    return ax


def plot_grad_norm(ax, hist, *, clip: Optional[float] = None):
    """Gradient norm before clipping. A trace pinned at the clip line means the
    update is being throttled every step and the effective learning rate is no
    longer the one that was set."""
    x, y = hist.get("step", []), hist.get("grad_norm", [])
    if y:
        n = min(len(x), len(y))
        ax.plot(np.asarray(x[:n]), np.asarray(y[:n], float),
                color=_TERM_COLORS["grad_norm"], lw=1.2)
        ax.set_yscale("log")
    if clip:
        ax.axhline(clip, color=OUTCOME_COLORS["early"], lw=1.2, ls="--",
                   label=f"clip = {clip:g}")
        ax.legend(fontsize=7, frameon=False)
    _style(ax, "update", "grad norm", "gradient norm (pre-clip)")
    return ax


def plot_training_metrics(ax, hist, *, keys=("p_engaged", "p_correct",
                                             "p_cue_success", "p_iti_lick")):
    """Behavioural summaries against update index, on a shared 0-1 axis."""
    x = np.asarray(hist.get("step", []), float)
    cols = {"p_engaged": "0.25", "p_correct": OUTCOME_COLORS["rewarded"],
            "p_cue_success": OUTCOME_COLORS["no_cue_complete"],
            "p_iti_lick": OUTCOME_COLORS["early"]}
    for k in keys:
        y = hist.get(k)
        if not y:
            continue
        n = min(len(x), len(y))
        ax.plot(x[:n], np.asarray(y[:n], float), lw=1.4,
                color=cols.get(k), label=k.replace("p_", ""))
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax, "update", "fraction", "behaviour during training")
    return ax


def plot_delay_and_iti(ax, hist):
    """Required delay (batch mean) and ITI licks per trial, twin axes.

    NB the delay here is the MEAN over the parallel environments, each of which
    carries its own scheduler and is at its own delay. A value of 0.26 is not a
    broken 0.1 s step -- it is a mixture of environments at 0.2 and at 0.3.
    """
    x = np.asarray(hist.get("step", []), float)
    d = hist.get("delay")
    if d:
        n = min(len(x), len(d))
        ax.plot(x[:n], np.asarray(d[:n], float), color=_INK, lw=1.6,
                label="required delay (batch mean)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ni = hist.get("n_iti_licks")
    if ni:
        n = min(len(x), len(ni))
        ax2.plot(x[:n], np.asarray(ni[:n], float),
                 color=OUTCOME_COLORS["early"], lw=1.2, alpha=0.8)
        ax2.set_yscale("symlog", linthresh=1.0)
        ax2.set_ylabel("ITI licks / trial", fontsize=8,
                       color=OUTCOME_COLORS["early"])
        ax2.tick_params(labelsize=7, colors=OUTCOME_COLORS["early"])
    _style(ax, "update", "delay (s)", "curriculum")
    return ax


def training_dashboard(hist, *, figsize=(13, 8), title=None, clip=None,
                       boundaries=None):
    """Optimiser view: loss terms, gradient, metrics, curriculum.

    ``boundaries`` marks the updates at which the training SCHEDULE changed --
    the curriculum handing over to a held long delay, then to switching blocks.
    Without them a phased run reads as one continuous curve and a jump caused
    by the task changing looks like something the optimiser did.
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    a = {"loss": axs[0, 0], "grad": axs[0, 1],
         "metrics": axs[1, 0], "curriculum": axs[1, 1]}
    plot_loss_terms(a["loss"], hist)
    plot_grad_norm(a["grad"], hist, clip=clip)
    plot_training_metrics(a["metrics"], hist)
    plot_delay_spread(a["curriculum"], hist)
    for ax in a.values():
        for b in (boundaries or []):
            ax.axvline(b, color="0.35", lw=1.0, ls=(0, (4, 3)), zorder=0)
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig, a


def run_report(hist, train_records=None, eval_records=None, *,
               probe_records=None, tests=None, phases=None, boundaries=None,
               snapshots=None, name="run",
               out_dir=None, clip=None, window: int = 50, n_blocks: int = 5,
               block: Optional[int] = None, dpi: int = 130):
    """Every diagnostic for one run. Returns {label: figure}.

    Separate figures because they answer separate questions: the optimiser over
    UPDATES; the two heads, also over updates, since weights move once per
    update and never within a trial; behaviour over TRIALS during training,
    which is where the lick timing is actually shaped; the probe, where the
    weights are fixed but the delay keeps growing; and each named test
    condition. A raster that mixes them is unreadable.

    ``eval_records`` no longer gets its own behaviour figure -- testing at
    whatever delay training happened to stop on cannot separate an agent that
    times from one that memorised an interval. It is kept because it is the
    source of the network-internals and value-head panels.
    """
    import matplotlib.pyplot as plt
    figs = {}
    figs["training"] = training_dashboard(
        hist, title=f"{name} -- optimiser", clip=clip,
        boundaries=boundaries)[0]
    if train_records:
        figs["behaviour_training"] = behaviour_dashboard(
            train_records, window=window,
            title=f"{name} -- behaviour during training "
                  f"({len(train_records)} trials)")[0]
        figs["history_training"] = history_dashboard(
            train_records, n_blocks=n_blocks, block=block,
            title=f"{name} -- distributions, trial history, readout "
                  f"(training)")[0]
    # One behaviour figure per TRAINING PHASE. The phases are different tasks
    # -- a growing delay, a held one, switching blocks -- and a raster that
    # pools them is three experiments in one panel.
    for pname, recs in (phases or {}).items():
        if not recs:
            continue
        figs[f"behaviour_train_{pname}"] = behaviour_dashboard(
            recs, window=window,
            title=f"{name} -- training phase: {pname} "
                  f"({len(recs)} trials, weights learning)")[0]
        figs[f"history_train_{pname}"] = history_dashboard(
            recs, n_blocks=n_blocks, block=block,
            title=f"{name} -- training phase: {pname}, trial history "
                  f"and lick timing")[0]
    figs["heads"] = heads_dashboard(
        hist, snapshots, eval_records, n_blocks=n_blocks, block=block,
        title=f"{name} -- policy and value heads")[0]
    if eval_records and any(r.get("hidden") is not None for r in eval_records):
        try:
            figs["model"] = _model_figure(eval_records, name)
        except Exception:          # geometry panels are optional diagnostics
            pass
    for label, recs in (tests or {}).items():
        if not recs:
            continue
        figs[f"behaviour_{label}"] = behaviour_dashboard(
            recs, window=window,
            title=f"{name} -- test: {label}, weights frozen "
                  f"({len(recs)} trials)")[0]
        figs[f"history_{label}"] = history_dashboard(
            recs, n_blocks=n_blocks, block=block,
            title=f"{name} -- test: {label}, trial history and lick timing")[0]
    if probe_records:
        figs["behaviour_probe"] = behaviour_dashboard(
            probe_records, window=window,
            title=f"{name} -- probe: weights frozen, delay free "
                  f"({len(probe_records)} trials)")[0]
        figs["history_probe"] = history_dashboard(
            probe_records, n_blocks=n_blocks, block=block,
            title=f"{name} -- probe: does lick time follow the delay?")[0]
    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        for label, fig in figs.items():
            fig.savefig(os.path.join(out_dir, f"{name}_{label}.png"),
                        dpi=dpi, bbox_inches="tight")
        plt.close("all")
    return figs


# --------------------------------------------------------------------------- #
# 5. Learning over trials -- distributions, trial history, the policy readout
# --------------------------------------------------------------------------- #
def _first_licks(records, cued_only=True):
    rs = [r for r in records
          if (not cued_only or r.get("cue_onset_step") is not None)
          and r.get("first_lick_s") is not None]
    return rs, np.array([r["first_lick_s"] for r in rs], float)


def trial_groups(n, *, n_blocks: int = 5, block: Optional[int] = None):
    """Split ``n`` trials into consecutive groups, as (start, stop) pairs.

    Two independent binnings exist in these figures and confusing them is easy:
    THIS one groups TRIALS (the curves, chronological), while ``bin_ms`` below
    is the width of the histogram bin on the TIME axis. They have nothing to do
    with each other.

    ``n_blocks`` gives equal-count groups -- quintiles by default, so every
    curve rests on the same number of trials however long the run. ``block``
    overrides it with a fixed number of trials per group, capped so a 10k-trial
    run cannot draw fifty unreadable curves.
    """
    if n <= 0:
        return []
    if block:
        b = max(int(block), int(np.ceil(n / 8)))
        edges = list(range(0, n, b)) + [n]
    else:
        edges = np.linspace(0, n, int(n_blocks) + 1).round().astype(int).tolist()
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 5]


def _smooth(y, k=3):
    """Moving average over k bins. Removes the single-bin spikes that a 50 ms
    histogram of a few hundred trials is mostly made of, without moving the
    mode the way a wider bin would."""
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def plot_first_lick_hist_blocks(ax, records, *, n_blocks: int = 5,
                                block: Optional[int] = None,
                                bin_ms: float = 50.0, smooth: int = 3,
                                xmax: Optional[float] = None):
    """First-lick density in consecutive groups of trials, dark to light.

    ``bin_ms`` is the width of the histogram bin on the TIME axis (50 ms).
    ``n_blocks`` is how many chronological groups of TRIALS get their own curve
    (5 equal-count groups). The two are unrelated; the title states both.

    A single 'last 200' histogram cannot show a distribution moving, which is
    the whole question when the required delay is growing underneath it.
    """
    rs, fl = _first_licks(records)
    if not len(fl):
        return _style(ax, "first lick (s from cue)", "density", "lick-time drift")
    hi = xmax if xmax else float(np.nanpercentile(fl, 99.5)) + 0.1
    edges = np.arange(0, hi + bin_ms / 1000., bin_ms / 1000.)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    import matplotlib.cm as cm
    groups = trial_groups(len(fl), n_blocks=n_blocks, block=block)
    for i, (a, b) in enumerate(groups):
        d, _ = np.histogram(fl[a:b], bins=edges, density=True)
        ax.plot(ctr, _smooth(d, smooth), lw=1.5,
                color=cm.viridis(0.12 + 0.78 * i / max(1, len(groups) - 1)),
                label=f"{a}-{b}")
    if groups:
        ax.legend(fontsize=6, frameon=False, ncol=2, title="trials",
                  title_fontsize=6)
    _style(ax, "first lick (s from cue)", "density",
           f"lick-time drift  |  x: {bin_ms:g} ms bins, smoothed {smooth}"
           f"  |  {len(groups)} trial groups")
    return ax


def plot_first_lick_hist_by_delay(ax, records, *, bin_ms: float = 50.0,
                                  smooth: int = 3, min_trials: int = 20,
                                  xmax: Optional[float] = None):
    """First-lick density separately for each required delay.

    The direct test of timing: if the agent is reading the trial rather than
    repeating one interval, each curve should sit to the right of the last by
    about the delay increment, and the dashed lines mark where each should be.
    """
    rs, fl = _first_licks(records)
    if not len(fl):
        return _style(ax, "first lick (s from cue)", "density", "by delay")
    dly = np.array([r["delay"] for r in rs], float)
    levels = sorted(set(np.round(dly, 3)))
    hi = xmax if xmax else float(np.nanpercentile(fl, 99.5)) + 0.1
    edges = np.arange(0, hi + bin_ms / 1000., bin_ms / 1000.)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    import matplotlib.cm as cm
    shown = 0
    for i, L in enumerate(levels):
        y = fl[np.round(dly, 3) == L]
        if len(y) < min_trials:
            continue
        c = cm.plasma(0.10 + 0.75 * i / max(1, len(levels) - 1))
        d, _ = np.histogram(y, bins=edges, density=True)
        ax.plot(ctr, _smooth(d, smooth), lw=1.5, color=c,
                label=f"{L:.2f}s  (n={len(y)}, med {np.median(y):.2f})")
        ax.axvline(L, color=c, lw=0.9, ls=":")
        shown += 1
    if shown:
        ax.legend(fontsize=6, frameon=False, title="required delay",
                  title_fontsize=6)
    _style(ax, "first lick (s from cue)", "density",
           f"lick time by required delay  |  x: {bin_ms:g} ms bins")
    return ax


def plot_value_gain(ax, hist):
    """The across-trial value gain and the reward rate that drives it.

    Above 1 the agent has been failing and water is worth more than baseline;
    below 1 it has been succeeding and water is worth less. Flat at 1.0 means
    `reward_rate_gain` is off."""
    x = np.asarray(hist.get("step", []), float)
    y = hist.get("value_gain")
    if y:
        n = min(len(x), len(y))
        ax.plot(x[:n], np.asarray(y[:n], float), lw=1.6, color=_INK,
                label="value gain")
        ax.axhline(1.0, color="0.7", lw=.8, ls="--")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    r = hist.get("reward_rate")
    if r:
        ax2 = ax.twinx()
        n = min(len(x), len(r))
        ax2.plot(x[:n], np.asarray(r[:n], float), lw=1.2,
                 color=OUTCOME_COLORS["rewarded"], alpha=.85)
        ax2.set_ylim(-0.03, 1.03)
        ax2.set_ylabel("reward rate (100 trials)", fontsize=8,
                       color=OUTCOME_COLORS["rewarded"])
        ax2.tick_params(labelsize=7, colors=OUTCOME_COLORS["rewarded"])
    _style(ax, "update", "gain", "subjective value of a trial")
    return ax


def plot_delay_spread(ax, hist):
    """Required delay against update, with an axis that starts at zero.

    Each parallel environment runs its own scheduler and each only ever
    increases its delay -- the promotion rule adds ``delay_step`` and has no
    branch that subtracts. Any wobble in this line is the BATCH MEAN moving
    because catch trials are excluded and different environments drop out of
    the average, not a delay going down. On an autoscaled axis a constant
    0.100 renders as a wandering line across a 0.005 range, which is why the
    limit is pinned here.
    """
    x = np.asarray(hist.get("step", []), float)
    d = hist.get("delay")
    if d:
        n = min(len(x), len(d))
        d = np.asarray(d[:n], float)
        ax.plot(x[:n], d, color=_INK, lw=1.8, label="required delay (batch mean)")
        ax.set_ylim(0, max(0.4, float(np.nanmax(d)) * 1.15))
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax, "update", "delay (s)", "curriculum")
    return ax


# ---- trial history --------------------------------------------------------
_HIST_COLORS = {"rewarded": OUTCOME_COLORS["rewarded"],
                "unrewarded": OUTCOME_COLORS["early"]}


def _prev_outcome(records):
    """(records with a first lick, boolean 'previous trial was rewarded').

    Uses the record's own ``success_t-1`` semantics: the previous trial in the
    SAME environment. Records arrive interleaved across the 16 parallel
    environments, so the neighbour in the list is usually a different animal --
    reconstructing 'previous' by list position would be wrong. Falls back to
    list order only if the field is absent.
    """
    rs = [r for r in records if r.get("first_lick_s") is not None
          and r.get("cue_onset_step") is not None]
    prev = []
    for r in rs:
        p = r.get("prev_success")
        prev.append(bool(p) if p is not None else None)
    return rs, prev


def plot_history_split_hist(ax, records, *, bin_ms: float = 50.0,
                            smooth: int = 3, recent: Optional[int] = None):
    """First-lick density split by whether the previous trial was rewarded.

    The expected signature: after reward the agent should lick earlier, after
    failure it should wait longer. A separation here is evidence the previous
    trial's outcome is being used; overlapping curves mean the history channels
    are being ignored.
    """
    rs, prev = _prev_outcome(records)
    if recent:
        rs, prev = rs[-recent:], prev[-recent:]
    fl = np.array([r["first_lick_s"] for r in rs], float)
    ok = np.array([p is not None for p in prev])
    if ok.sum() < 20:
        return _style(ax, "first lick (s from cue)", "density",
                      "trial history (no data)")
    pv = np.array([bool(p) for p in prev])
    hi = float(np.nanpercentile(fl, 99.5)) + 0.1
    edges = np.arange(0, hi + bin_ms / 1000., bin_ms / 1000.)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for label, m in (("rewarded", ok & pv), ("unrewarded", ok & ~pv)):
        if m.sum() < 10:
            continue
        d, _ = np.histogram(fl[m], bins=edges, density=True)
        ax.plot(ctr, _smooth(d, smooth), lw=1.6, color=_HIST_COLORS[label],
                label=f"prev {label} (n={m.sum()}, med {np.median(fl[m]):.3f})")
        ax.axvline(np.median(fl[m]), color=_HIST_COLORS[label], lw=0.9, ls=":")
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax, "first lick (s from cue)", "density",
           "lick time by previous outcome")
    return ax


def plot_history_effect_over_trials(ax, records, *, window: int = 400,
                                    step: int = 100):
    """Median lick time after a rewarded vs an unrewarded trial, in moving
    windows. Shows WHEN in training the trial-history effect appears, which a
    single pooled histogram cannot."""
    rs, prev = _prev_outcome(records)
    fl = np.array([r["first_lick_s"] for r in rs], float)
    pv = np.array([p if p is not None else np.nan for p in prev], float)
    # Shrink the window rather than draw nothing: a short run should still
    # show its (noisier) history effect instead of an empty axis.
    window = int(min(window, max(40, len(fl) // 3)))
    step = max(1, min(step, window // 2))
    xs, med_r, med_u = [], [], []
    for a in range(0, max(1, len(fl) - window + 1), step):
        y, p = fl[a:a + window], pv[a:a + window]
        r_, u_ = y[p == 1], y[p == 0]
        if len(r_) < 10 or len(u_) < 10:
            continue
        xs.append(a + window / 2)
        med_r.append(np.median(r_))
        med_u.append(np.median(u_))
    if xs:
        ax.plot(xs, med_r, color=_HIST_COLORS["rewarded"], lw=1.6,
                label="after reward")
        ax.plot(xs, med_u, color=_HIST_COLORS["unrewarded"], lw=1.6,
                label="after failure")
        ax.fill_between(xs, med_r, med_u, color="0.8", alpha=.5, zorder=0)
        ax.legend(fontsize=7, frameon=False)
    _style(ax, "trial", "median first lick (s)",
           f"trial-history effect (window {window})")
    return ax


def history_regression(records, *, window: int = 600, step: int = 150,
                       n_shuffle: int = 200, rng=None):
    """Regress first-lick time on the previous trial's outcome and latency, in
    moving windows, against a trial-order shuffle null.

    Model, per window:  first_lick ~ b0 + b1*prev_rewarded + b2*prev_first_lick

    The null shuffles trial ORDER within the window, which destroys the
    pairing between a trial and its predecessor while leaving the lick-time
    distribution and the outcome frequencies untouched. Without it a non-zero
    coefficient means little: lick times drift over training, and drift alone
    produces correlations with anything that also drifts.

    Returns a dict of arrays: ``trial``, ``b_reward``, ``b_prev_lick`` and the
    2.5/97.5 percentiles of each under the null.
    """
    rng = rng or np.random.default_rng(0)
    rs = [r for r in records if r.get("first_lick_s") is not None
          and r.get("prev_success") is not None
          and r.get("cue_onset_step") is not None]
    y_all = np.array([r["first_lick_s"] for r in rs], float)
    x1_all = np.array([float(bool(r["prev_success"])) for r in rs], float)
    x2_all = np.array([float(r.get("prev_first_lick", 0.0) or 0.0) for r in rs],
                      float)
    window = int(min(window, max(60, len(y_all) // 3)))
    step = max(1, min(step, window // 2))
    out = {k: [] for k in ("trial", "b_reward", "b_prev_lick",
                           "b_reward_lo", "b_reward_hi",
                           "b_prev_lick_lo", "b_prev_lick_hi")}

    def fit(y, X):
        return np.linalg.lstsq(X, y, rcond=None)[0]

    for a in range(0, max(1, len(y_all) - window + 1), step):
        y, x1, x2 = y_all[a:a + window], x1_all[a:a + window], x2_all[a:a + window]
        if len(y) < 40 or x1.std() < 1e-9:
            continue
        X = np.column_stack([np.ones_like(y), x1, x2])
        b = fit(y, X)
        null = np.empty((n_shuffle, 2))
        for s in range(n_shuffle):
            p = rng.permutation(len(y))          # break trial ORDER only
            null[s] = fit(y, np.column_stack([np.ones_like(y), x1[p], x2[p]]))[1:]
        out["trial"].append(a + window / 2)
        out["b_reward"].append(b[1])
        out["b_prev_lick"].append(b[2])
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        out["b_reward_lo"].append(lo[0]);  out["b_reward_hi"].append(hi[0])
        out["b_prev_lick_lo"].append(lo[1]); out["b_prev_lick_hi"].append(hi[1])
    return {k: np.asarray(v, float) for k, v in out.items()}


def plot_history_regression(ax, records, **kw):
    """Regression weights over training with the shuffle null as a grey band.
    A coefficient outside the band is a trial-history effect that trial order
    shuffling destroys; inside it, the agent is not using the history."""
    r = history_regression(records, **kw)
    if not len(r["trial"]):
        return _style(ax, "trial", "weight (s)", "trial-history regression")
    for key, lo, hi, col, lab in (
            ("b_reward", "b_reward_lo", "b_reward_hi",
             _HIST_COLORS["rewarded"], "prev rewarded"),
            ("b_prev_lick", "b_prev_lick_lo", "b_prev_lick_hi",
             OUTCOME_COLORS["no_cue_complete"], "prev first-lick")):
        ax.plot(r["trial"], r[key], color=col, lw=1.6, label=lab)
        ax.fill_between(r["trial"], r[lo], r[hi], color=col, alpha=.15, lw=0)
    ax.axhline(0, color=_INK, lw=0.8, ls="--")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "trial", "weight (s)",
           "lick time ~ previous trial (shaded: order-shuffle null)")
    return ax


# ---- the policy readout ---------------------------------------------------
def stack_readouts(records, *, pre: int = 25, post: int = 100):
    """(trials x time) array of the policy readout aligned to cue onset.

    The readout is the logit gap, lick minus wait: the single scalar the whole
    policy is a function of, with P(lick) = sigmoid(gap). Trials are padded with
    NaN where they are shorter than the window.
    """
    rows, keep = [], []
    for r in records:
        d = r.get("readout")
        if d is None:
            continue
        on = r.get("cue_onset_step") or r.get("align_step")
        if on is None:
            continue
        a, b = int(on) - pre, int(on) + post
        row = np.full(pre + post, np.nan)
        lo, hi = max(0, a), min(len(d), b)
        if hi > lo:
            row[lo - a:hi - a] = np.asarray(d)[lo:hi]
        rows.append(row)
        keep.append(r)
    if not rows:
        return np.zeros((0, pre + post)), [], np.arange(-pre, post)
    return np.vstack(rows), keep, np.arange(-pre, post)


def plot_readout_heatmap(ax, records, *, pre: int = 25, post: int = 100,
                         dt: float = 0.02, cmap: str = "RdBu_r"):
    """Policy readout, trials on y in training order, time from cue on x."""
    A, keep, t = stack_readouts(records, pre=pre, post=post)
    if not len(A):
        return _style(ax, "time from cue (s)", "trial", "policy readout")
    v = float(np.nanpercentile(np.abs(A), 99)) or 1.0
    ax.imshow(A, aspect="auto", origin="lower", cmap=cmap, vmin=-v, vmax=v,
              extent=[t[0] * dt, t[-1] * dt, 0, len(A)],
              interpolation="nearest")
    ax.axvline(0, color=_INK, lw=1.0)
    fl = np.array([r.get("first_lick_s", np.nan) for r in keep], float)
    ax.plot(fl, np.arange(len(fl)) + .5, ".", ms=1.6, color="k", alpha=.55)
    _style(ax, "time from cue (s)", "trial",
           "policy readout (lick − wait logit)")
    return ax


def plot_readout_blocks(ax, records, *, n_blocks: int = 5,
                        block: Optional[int] = None, pre: int = 25,
                        post: int = 100, dt: float = 0.02):
    """Mean policy readout per block of trials, dark to light with training.

    This is where a ramp would show up: an agent timing the interval should
    develop a readout that rises from cue onset and crosses zero (P(lick) = 0.5)
    near the required delay, and the crossing should move right as the delay
    grows.
    """
    A, keep, t = stack_readouts(records, pre=pre, post=post)
    if not len(A):
        return _style(ax, "time from cue (s)", "readout", "readout over training")
    import matplotlib.cm as cm
    groups = trial_groups(len(A), n_blocks=n_blocks, block=block)
    for i, (a, b) in enumerate(groups):
        ax.plot(t * dt, _nanmean(A[a:b], axis=0), lw=1.5,
                color=cm.viridis(0.12 + 0.78 * i / max(1, len(groups) - 1)),
                label=f"{a}-{b}")
    ax.axhline(0, color=_INK, lw=0.9, ls="--")
    ax.axvline(0, color=_INK, lw=0.9)
    if groups:
        ax.legend(fontsize=6, frameon=False, ncol=2, title="trials",
                  title_fontsize=6)
    _style(ax, "time from cue (s)", "lick − wait logit",
           "policy readout over training (0 = coin flip)")
    return ax


def history_dashboard(records, *, figsize=(13, 11), n_blocks: int = 5,
                      block: Optional[int] = None,
                      title: Optional[str] = None, window: int = 400):
    """Distributions, trial-history use, and the policy readout."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    a = {"drift": axs[0, 0], "by_delay": axs[0, 1],
         "hist_split": axs[1, 0], "hist_time": axs[1, 1],
         "regression": axs[2, 0], "readout": axs[2, 1]}
    plot_first_lick_hist_blocks(a["drift"], records, n_blocks=n_blocks,
                                block=block)
    plot_first_lick_hist_by_delay(a["by_delay"], records)
    plot_history_split_hist(a["hist_split"], records)
    plot_history_effect_over_trials(a["hist_time"], records, window=window)
    plot_history_regression(a["regression"], records,
                            window=max(300, window), step=max(75, window // 4))
    if any(r.get("readout") is not None for r in records):
        plot_readout_blocks(a["readout"], records, n_blocks=n_blocks,
                            block=block)
    else:
        plot_iti_lick_rate(a["readout"], records)
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig, a


def _model_figure(records, name, *, pre: int = 25, post: int = 120):
    """Network internals for the evaluation trials, using the existing
    activity plotters: population heatmap, single units, PSTH by lick-time
    bin, PCA trajectories -- plus the policy readout, which is the RL-specific
    part (``model_dashboard`` reads the supervised output head, which the
    actor-critic does not use)."""
    import matplotlib.pyplot as plt
    recs = [r for r in records if r.get("hidden") is not None
            and r.get("cue_onset_step") is not None]
    if not recs:
        raise ValueError("no hidden states")
    T = pre + post
    H = np.full((len(recs), T, recs[0]["hidden"].shape[1]), np.nan, np.float32)
    for i, r in enumerate(recs):
        h, on = np.asarray(r["hidden"]), int(r["cue_onset_step"])
        a, b = on - pre, on + post
        lo, hi = max(0, a), min(len(h), b)
        if hi > lo:
            H[i, lo - a:hi - a] = h[lo:hi]
    t = np.arange(-pre, post) * float(recs[0].get("dt", 0.02))
    fl = np.array([r.get("first_lick_s", np.nan) for r in recs], float)
    groups = np.digitize(fl, np.nanquantile(fl[~np.isnan(fl)], [.33, .66])) \
        if np.isfinite(fl).sum() > 10 else None

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    # These take the full (trials, T, N) stack, not a trial average: the
    # heatmap sorts units on one half of the trials and displays the other.
    plot_activity_heatmap(axs[0, 0], H, t)
    plot_unit_traces(axs[0, 1], H, t)
    plot_psth(axs[0, 2], H, t, groups=groups)
    plot_pca_trajectories(axs[1, 0], H, t, groups=groups)
    plot_readout_heatmap(axs[1, 1], recs)
    plot_readout_blocks(axs[1, 2], recs, block=max(20, len(recs) // 4))
    fig.suptitle(f"{name} -- network internals (evaluation trials)",
                 fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. The heads -- do the policy and value readouts sharpen, and do they settle?
# --------------------------------------------------------------------------- #
def plot_weight_norms(ax, hist):
    """Sizes of the four weight blocks against update index.

    ``w_delta`` is the policy readout direction, the difference of the two rows
    of the policy head. Its norm is the GAIN of the policy: the logit gap is
    w_delta . h, so this bounds how confident the policy can become for a state
    of a given size. A policy that sharpens does it either by growing this or by
    growing the state; the activity penalty forbids the second route, which is
    why the two curves should be read together.
    """
    x = np.asarray(hist.get("step", []), float)
    for k, c, lab in (("w_delta_norm", OUTCOME_COLORS["rewarded"],
                       r"$\|w_\Delta\|$ (policy readout)"),
                      ("w_value_norm", OUTCOME_COLORS["early"],
                       r"$\|w_V\|$ (value readout)"),
                      ("W_rec_norm", "0.25", r"$\|W_{rec}\|_F$"),
                      ("W_in_norm", OUTCOME_COLORS["no_cue_complete"],
                       r"$\|W_{in}\|_F$")):
        y = hist.get(k)
        if y:
            n = min(len(x), len(y))
            ax.plot(x[:n], np.asarray(y[:n], float), lw=1.5, color=c, label=lab)
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "update", "norm", "weight magnitudes")
    return ax


def plot_weight_steps(ax, hist):
    """Relative size of each update, per parameter block: ||dtheta||/||theta||.

    This is the settling diagnostic. A run that has converged has every curve
    decaying toward zero; one still being thrown around does not, however good
    its behaviour looks.
    """
    x = np.asarray(hist.get("step", []), float)
    keys = sorted(k for k in hist if k.startswith("d_"))
    import matplotlib.cm as cm
    for i, k in enumerate(keys):
        y = np.asarray(hist[k], float)
        n = min(len(x), len(y))
        ax.plot(x[:n], y[:n], lw=1.2,
                color=cm.viridis(0.1 + 0.8 * i / max(1, len(keys) - 1)),
                label=k[2:].replace("core.", ""))
    if keys:
        ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=False, ncol=2)
    _style(ax, "update", r"$\|\Delta\theta\| / \|\theta\|$",
           "relative update size (settling)")
    return ax


def plot_readout_direction_drift(ax, snapshots):
    """Cosine similarity of each readout DIRECTION to its final value.

    Separates two things the norm cannot: a readout that is still growing along
    a settled direction (this curve at 1.0, the norm still rising) from one
    whose direction is still being rewritten (this curve below 1.0). The second
    means the network has not decided what feature it is reading.
    """
    if not snapshots:
        return _style(ax, "update", "cosine to final", "readout direction")
    x = [s["step"] for s in snapshots]
    for key, c, lab in (("w_delta", OUTCOME_COLORS["rewarded"], "policy"),
                        ("w_value", OUTCOME_COLORS["early"], "value")):
        V = np.stack([np.asarray(s[key], float) for s in snapshots])
        f = V[-1] / (np.linalg.norm(V[-1]) + 1e-12)
        cos = V @ f / (np.linalg.norm(V, axis=1) + 1e-12)
        ax.plot(x, cos, lw=1.6, color=c, label=lab)
    ax.axhline(1.0, color=_INK, lw=.8, ls="--")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "update", "cosine to final", "readout direction, drift to final")
    return ax


def plot_policy_bias(ax, hist, snapshots):
    """The policy's resting lick drive: the bias difference alone, which is the
    logit gap the network would produce from a zero state. Plotted with the
    entropy it implies, so a bias that has run away is visible as an entropy
    that cannot recover."""
    if snapshots:
        x = [s["step"] for s in snapshots]
        b = np.array([s["b_delta"] for s in snapshots], float)
        ax.plot(x, b, lw=1.6, color=_INK, label=r"$b_{lick} - b_{wait}$")
        ax.axhline(0, color="0.7", lw=.8, ls="--")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    y = hist.get("entropy")
    if y:
        ax2 = ax.twinx()
        xs = np.asarray(hist.get("step", []), float)
        n = min(len(xs), len(y))
        ax2.plot(xs[:n], np.asarray(y[:n], float), lw=1.2,
                 color=OUTCOME_COLORS["no_cue_complete"], alpha=.85)
        ax2.axhline(np.log(2), color="0.8", lw=.8, ls=":")
        ax2.set_ylabel("entropy (nats)", fontsize=8,
                       color=OUTCOME_COLORS["no_cue_complete"])
        ax2.tick_params(labelsize=7, colors=OUTCOME_COLORS["no_cue_complete"])
    _style(ax, "update", "bias gap", "resting lick drive")
    return ax


def _stack_field(records, key, *, pre=25, post=120):
    rows = []
    for r in records:
        v = r.get(key)
        on = r.get("cue_onset_step") or r.get("align_step")
        if v is None or on is None:
            continue
        a, b = int(on) - pre, int(on) + post
        row = np.full(pre + post, np.nan)
        lo, hi = max(0, a), min(len(v), b)
        if hi > lo:
            row[lo - a:hi - a] = np.asarray(v)[lo:hi]
        rows.append(row)
    return (np.vstack(rows) if rows else np.zeros((0, pre + post)),
            np.arange(-pre, post))


def plot_value_traces(ax, records, *, n_blocks: int = 5,
                      block: Optional[int] = None, dt: float = 0.02,
                      pre: int = 25, post: int = 120):
    """Predicted value aligned to cue, averaged per block of trials.

    The value head estimates the return from here to the end of the trial. If it
    is doing its job the trace should rise toward the moment water becomes
    available and drop after the lick; a flat line means the baseline is
    carrying no information and the advantage is just the return.
    """
    A, t = _stack_field(records, "value", pre=pre, post=post)
    if not len(A):
        return _style(ax, "time from cue (s)", "predicted value", "value head")
    import matplotlib.cm as cm
    groups = trial_groups(len(A), n_blocks=n_blocks, block=block)
    for i, (a, b) in enumerate(groups):
        ax.plot(t * dt, _nanmean(A[a:b], axis=0), lw=1.5,
                color=cm.viridis(0.12 + 0.78 * i / max(1, len(groups) - 1)),
                label=f"{a}-{b}")
    ax.axvline(0, color=_INK, lw=.9)
    if groups:
        ax.legend(fontsize=6, frameon=False, ncol=2, title="trials",
                  title_fontsize=6)
    _style(ax, "time from cue (s)", "predicted value",
           "value head over training")
    return ax


def heads_dashboard(hist, snapshots=None, records=None, *, figsize=(13, 8),
                    title=None, n_blocks: int = 5, block: Optional[int] = None):
    """Policy and value heads: magnitude, direction, settling, and what the
    value function actually predicts."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    plot_weight_norms(axs[0, 0], hist)
    plot_readout_direction_drift(axs[0, 1], snapshots or [])
    plot_policy_bias(axs[0, 2], hist, snapshots or [])
    plot_value_gain(axs[1, 0], hist)
    if records:
        plot_value_traces(axs[1, 1], records, n_blocks=n_blocks, block=block)
        plot_readout_blocks(axs[1, 2], records, n_blocks=n_blocks, block=block)
    else:
        plot_grad_norm(axs[1, 1], hist)
        plot_training_metrics(axs[1, 2], hist)
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, x=.02, ha="left")
    fig.tight_layout()
    return fig, axs
