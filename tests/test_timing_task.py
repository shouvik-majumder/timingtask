"""Tests for the timing-task generator and scheduler."""
import numpy as np
import pytest
from timingtask import (DelayScheduler, ObservationConfig, Phase, SchedulerConfig,
                    TimingTaskConfig, TrialGenerator, make_config)


def build(**kw):
    task = TimingTaskConfig(seed=0, **{k: v for k, v in kw.items()
                                       if hasattr(TimingTaskConfig, k)})
    sched_kw = {k: v for k, v in kw.items() if not hasattr(TimingTaskConfig, k)}
    sc = SchedulerConfig(**sched_kw)
    rng = np.random.default_rng(0)
    return TrialGenerator(task, DelayScheduler(sc, rng), ObservationConfig(), rng)


def run(gen, policy, max_steps=200000):
    """Roll forward, returning completed trial records."""
    recs = []
    for _ in range(max_steps):
        r = gen.step(policy(gen))
        if r.record:
            recs.append(r.record)
    return recs


# --------------------------------------------------------------------- cue
def test_cue_is_a_signal_not_a_phase():
    """Cue duration and reward eligibility are independent axes: with a delay
    SHORTER than the cue, a lick after the delay but while the cue is still on
    must be REWARDED. This is the whole point of the restructure."""
    g = build(cue_duration=0.6, mode="fixed", fixed_delay=0.2, no_cue_prob=0.0)
    while g.phase != Phase.ELIGIBLE:
        g.step(False)
    assert g.cue_on, "cue should still be audible at delay=0.2 < cue=0.6"
    res = g.step(True)
    assert res.info["outcome"] == "rewarded"
    assert res.record is None, "trial continues past the lick (post-lick epoch)"
    rec = None
    while rec is None:
        rec = g.step(False).record
    assert rec["rewarded"] is True
    assert rec["first_lick_s"] >= 0.2 and rec["first_lick_s"] < 0.6


def test_cue_on_depends_only_on_time_since_onset():
    g = build(cue_duration=0.2, mode="fixed", fixed_delay=1.0, no_cue_prob=0.0)
    while g._cue_onset_step is None:
        g.step(False)
    seen = []
    for _ in range(30):
        seen.append((round(g.timer, 3), g.cue_on, g.phase))
        g.step(False)
    on = [t for t, c, _ in seen if c]
    assert min(on) == 0.0, "agent must see timer == 0 at cue onset"
    assert max(on) < 0.2
    assert len(on) == g.cfg.steps(0.2), "cue visible for exactly cue_steps steps"
    # cue goes off while still WAITING -- a phase model could not express this
    assert any(ph == Phase.WAITING and not c for _, c, ph in seen)


# ------------------------------------------------------------------- timer
def test_lick_before_delay_aborts_unrewarded():
    g = build(mode="fixed", fixed_delay=1.0, no_cue_prob=0.0)
    while g.phase != Phase.WAITING:
        g.step(False)
    res = g.step(True)
    assert res.info["outcome"] == "early"
    assert g.phase == Phase.POST


def test_miss_when_answer_window_expires():
    g = build(mode="fixed", fixed_delay=0.2, answer_window=0.4, no_cue_prob=0.0)
    recs = []
    for _ in range(2000):
        r = g.step(False)
        if r.record:
            recs.append(r.record)
            break
    assert recs[0]["outcome"] == "miss"


def test_delay_boundary_is_exact_in_steps():
    """dt=0.02, delay=1.0 -> eligible at exactly step 50 after cue onset."""
    g = build(dt=0.02, mode="fixed", fixed_delay=1.0, no_cue_prob=0.0)
    while g._cue_onset_step is None:
        g.step(False)
    onset = g._cue_onset_step
    while g.phase == Phase.WAITING:
        g.step(False)
    assert g.step_index - onset == 50


# --------------------------------------------------------------------- ITI
def test_iti_lick_restarts_the_stop_licking_period():
    g = build(mode="fixed", no_cue_prob=0.0)
    g.step(False)
    before = g._iti_remaining
    g.step(False)
    assert g._iti_remaining == before - 1
    g.step(True)                       # lick during ITI
    assert g._iti_remaining >= g.cfg.steps(g.cfg.iti_min) - 1
    assert g.iti_licked is True
    assert g.phase == Phase.ITI


def test_iti_timeout_terminates():
    """An agent that licks forever never reaches the cue and must be released."""
    g = build(mode="fixed", iti_timeout=1.0, no_cue_prob=0.0)
    recs = run_until(g, lambda _: True, 1)
    assert recs[0]["outcome"] == "iti_timeout"
    assert recs[0]["cue_onset_step"] is None


def run_until(gen, policy, n):
    recs = []
    for _ in range(200000):
        r = gen.step(policy(gen))
        if r.record:
            recs.append(r.record)
            if len(recs) >= n:
                return recs
    raise AssertionError(f"only {len(recs)}/{n} trials completed")


# -------------------------------------------------------------- post-lick
def test_trial_continues_after_the_rewarded_lick():
    """peri_lick and post_lick epochs must exist in generated data."""
    g = build(mode="fixed", fixed_delay=0.2, post_lick=1.0, no_cue_prob=0.0)
    while g.phase != Phase.ELIGIBLE:
        g.step(False)
    res = g.step(True)
    assert not res.trial_over, "trial must not end on the rewarded lick"
    assert g.phase == Phase.POST
    n = 0
    while not g.step(False).trial_over:
        n += 1
        assert n < 500
    assert n == pytest.approx(g.cfg.steps(1.0) - 1, abs=2)


def test_lick_bouts_recorded_with_refractory():
    g = build(mode="fixed", fixed_delay=0.2, lick_refractory=0.1, no_cue_prob=0.0)
    while g.phase != Phase.ELIGIBLE:
        g.step(False)
    rec = None
    for _ in range(400):
        r = g.step(True)               # hold the lick down
        if r.record:
            rec = r.record
            break
    assert rec["n_licks"] > 1, "a bout should record multiple licks"
    gaps = np.diff(rec["lick_times_s"])
    assert gaps.min() >= 0.1 - 1e-9, "refractory violated"


# --------------------------------------------------------------- no-cue
def test_no_cue_trials_never_start_the_timer():
    g = build(mode="fixed", no_cue_prob=1.0)
    rec = run_until(g, lambda _: False, 1)[0]
    assert rec["no_cue_trial"] and rec["cue_onset_step"] is None
    assert rec["outcome"] == "no_cue_complete"
    assert rec["first_lick_s"] is None


# ------------------------------------------------------------ scheduler
def test_fixed_delay_never_moves():
    g = build(mode="fixed", fixed_delay=0.7, no_cue_prob=0.0)
    recs = run_until(g, lambda gg: gg.phase == Phase.ELIGIBLE, 30)
    assert {r["delay"] for r in recs} == {0.7}


def test_autolearn_promotes_on_the_published_criterion():
    """30% rewarded over 100 trials at a delay -> +0.1 s."""
    sc = SchedulerConfig(mode="autolearn", initial_delay=0.1, delay_step=0.1,
                         perf_window=100, min_trials_per_delay=100,
                         success_threshold=0.30)
    s = DelayScheduler(sc, np.random.default_rng(0))
    assert s.get_current_delay() == pytest.approx(0.1)
    for i in range(100):                       # 40% rewarded -> above criterion
        s.on_trial_end(i % 10 < 4, {})
    assert s.get_current_delay() == pytest.approx(0.2)

    s2 = DelayScheduler(sc, np.random.default_rng(0))
    for i in range(100):                       # 20% -> below criterion
        s2.on_trial_end(i % 10 < 2, {})
    assert s2.get_current_delay() == pytest.approx(0.1)


def test_autolearn_ignores_catch_trials():
    sc = SchedulerConfig(mode="autolearn", perf_window=10, min_trials_per_delay=10)
    s = DelayScheduler(sc, np.random.default_rng(0))
    for _ in range(50):
        s.on_trial_end(None, {})               # catch trials score nothing
    assert s.delay_success_rate() is None
    assert s.get_current_delay() == pytest.approx(sc.initial_delay)


def test_cue_autolearn_two_stage_promotion():
    sc = SchedulerConfig(mode="cue_autolearn", cue_association_window=10,
                         cue_association_min_trials=10,
                         cue_association_success_threshold=0.5,
                         cue_association_max_iti_lick_hz=0.5)
    s = DelayScheduler(sc, np.random.default_rng(0))
    assert s.get_training_stage() == "cue_association"
    for _ in range(10):                        # responds to cue, no ITI licking
        s.on_trial_end(True, {"cue_success": True, "n_iti_licks": 0,
                              "iti_steps": 100, "dt": 0.02,
                              "cue_onset_step": 0, "no_cue_trial": False})
    assert s.get_training_stage() == "autolearn"
    assert s.last_stage_transition is True


def test_cue_autolearn_blocked_by_iti_licking():
    sc = SchedulerConfig(mode="cue_autolearn", cue_association_window=10,
                         cue_association_min_trials=10,
                         cue_association_max_iti_lick_hz=0.5)
    s = DelayScheduler(sc, np.random.default_rng(0))
    for _ in range(30):    # licks the cue but also the ITI: 20 licks in 2 s
        s.on_trial_end(True, {"cue_success": True, "n_iti_licks": 20,
                              "iti_steps": 100, "dt": 0.02,
                              "cue_onset_step": 0, "no_cue_trial": False})
    assert s.get_training_stage() == "cue_association"


def test_iti_criterion_is_a_rate_and_is_off_by_default():
    """The fraction-of-trials measure it replaced saturated at 1.0 and could
    not see a tenfold fall in ITI licking."""
    sc = SchedulerConfig(mode="cue_autolearn", cue_association_window=10,
                         cue_association_min_trials=10,
                         cue_association_success_threshold=0.5)
    assert sc.cue_association_max_iti_lick_hz is None
    s = DelayScheduler(sc, np.random.default_rng(0))
    for _ in range(10):        # every trial has ITI licks; promotes anyway
        s.on_trial_end(True, {"cue_success": True, "n_iti_licks": 20,
                              "iti_steps": 100, "dt": 0.02,
                              "cue_onset_step": 0, "no_cue_trial": False})
    assert s.get_training_stage() == "autolearn"
    assert abs(s.cue_iti_lick_rate() - 10.0) < 1e-6      # 20 licks / 2 s


def test_delay_is_clipped_to_max():
    sc = SchedulerConfig(mode="autolearn", initial_delay=1.9, max_delay=2.0,
                         delay_step=0.1, perf_window=5, min_trials_per_delay=5)
    s = DelayScheduler(sc, np.random.default_rng(0))
    for _ in range(200):
        s.on_trial_end(True, {})
    assert s.get_current_delay() <= 2.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        DelayScheduler(SchedulerConfig(mode="nonsense"))


# ------------------------------------------------------------- records
def test_record_shape_and_determinism():
    g = build(mode="fixed", fixed_delay=0.3, no_cue_prob=0.1)
    a = run_until(g, lambda gg: gg.phase == Phase.ELIGIBLE, 40)
    g2 = build(mode="fixed", fixed_delay=0.3, no_cue_prob=0.1)
    b = run_until(g2, lambda gg: gg.phase == Phase.ELIGIBLE, 40)
    assert [r["outcome"] for r in a] == [r["outcome"] for r in b]
    for k in ("trial", "outcome", "delay", "first_lick_s", "lick_times_s",
              "iti_licked", "no_cue_trial", "trial_duration_s", "dt",
              "training_stage", "next_delay"):
        assert k in a[0], k


def test_observation_size_matches_config():
    for cfg in (ObservationConfig(), ObservationConfig(include_trial_start=True),
                ObservationConfig(include_prev_reward=False,
                                  include_prev_action=False)):
        t, sc = TimingTaskConfig(seed=0), SchedulerConfig()
        g = TrialGenerator(t, DelayScheduler(sc), cfg)
        assert g.observe().shape == (g.obs_size,)
        assert g.observe().dtype == np.float32


def test_variants_differ_only_in_durations():
    for name in ("autolearn", "fixed", "switching"):
        t, s = make_config(name)
        assert t.cue_duration == 0.6
        assert t.dt == 0.02
    assert make_config("autolearn")[0].answer_window == 3.0
    assert make_config("fixed")[0].answer_window == 5.0
    assert make_config("switching")[0].answer_window == 10.0


# ------------------------------------------------------- ITI restart rule
def _free_licker(restart, n=120, p_lick=0.05, seed=0):
    """A constant-hazard licker: the failure mode the flag exists to escape."""
    rng = np.random.default_rng(seed)
    t = TimingTaskConfig(dt=0.02, answer_window=0.8, post_lick=0.5,
                         iti_mean=3.0, iti_min=2.0, iti_max=5.0,
                         iti_restart_on_lick=restart, no_cue_prob=0.0, seed=seed)
    g = TrialGenerator(t, DelayScheduler(SchedulerConfig(mode="cue_autolearn"), rng),
                       ObservationConfig(), rng)
    recs = []
    while len(recs) < n:
        g.observe()
        r = g.step(bool(rng.random() < p_lick))
        if r.record is not None:
            recs.append(r.record)
    return recs


def test_iti_restart_flag_controls_whether_the_cue_is_reachable():
    on, off = _free_licker(True), _free_licker(False)
    reach = lambda rs: sum(r.get("cue_onset_step") is not None for r in rs) / len(rs)
    # With the restart rule a free licker almost never reaches a cue, which is
    # exactly why it cannot learn the association. Without it, most trials do.
    assert reach(on) < 0.2
    assert reach(off) > 0.6


def test_iti_lick_count_is_recorded_and_restart_costs_more_licks():
    on, off = _free_licker(True), _free_licker(False)
    mean = lambda rs: float(np.mean([r["n_iti_licks"] for r in rs]))
    assert all("n_iti_licks" in r for r in on)
    assert mean(on) > mean(off)


# ---------------------------------------------------- cue as a tonic step
def _cue_columns(mode, delay=1.0, n=400):
    rng = np.random.default_rng(0)
    t = TimingTaskConfig(dt=0.02, cue_mode=mode, cue_duration=0.6,
                         answer_window=0.8, iti_restart_on_lick=False,
                         no_cue_prob=0.0, seed=0)
    g = TrialGenerator(t, DelayScheduler(SchedulerConfig(mode="fixed",
                                                        fixed_delay=delay), rng),
                       ObservationConfig(), rng)
    obs = []
    for _ in range(n):
        obs.append(g.observe().copy())
        if g.step(False).record is not None:
            break
    n_cue = 2 if mode == "both" else 1
    return np.array(obs)[:, :n_cue], g


def test_cue_step_persists_past_cue_offset_and_the_pulse_does_not():
    """With tau = 100 ms a 0.6 s pulse leaves nothing in the state by the time
    a 1 s delay elapses; a step gives a near-integrator something to
    accumulate."""
    pulse, _ = _cue_columns("pulse")
    step, _ = _cue_columns("step")
    assert int(pulse.sum()) == 30                 # cue_duration / dt
    assert int(step.sum()) >= 3 * int(pulse.sum())   # 90 vs 30 at delay 1.0


def test_both_gives_two_cue_channels_the_transient_and_the_tonic():
    """Yang et al.'s arrangement: a transient cue off the integration manifold
    plus a tonic step that drives the integrator."""
    cols, g = _cue_columns("both")
    assert g.observation_labels()[:2] == ["cue", "cue_step"]
    assert g.obs_size == 6
    assert int(cols[:, 0].sum()) == 30            # transient
    assert int(cols[:, 1].sum()) > 30             # tonic, still on
    # the tonic never switches off before the transient does
    assert cols[:, 1][cols[:, 0] == 1].all()


def test_catch_trials_stay_at_zero_on_every_cue_channel():
    """The zero-input control has to survive the change."""
    for mode in ("pulse", "step", "both"):
        rng = np.random.default_rng(1)
        t = TimingTaskConfig(dt=0.02, cue_mode=mode, no_cue_prob=1.0,
                             iti_restart_on_lick=False, seed=1)
        g = TrialGenerator(t, DelayScheduler(SchedulerConfig(mode="fixed"), rng),
                           ObservationConfig(), rng)
        n_cue = 2 if mode == "both" else 1
        for _ in range(300):
            assert not g.observe()[:n_cue].any()
            g.step(False)
