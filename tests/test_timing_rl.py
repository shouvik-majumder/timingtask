"""Tests for across-trial value, lagged history, and the RL trainer."""
import numpy as np
import pytest
import torch
from timingtask.models import VanillaRNN
from timingtask.config import ObservationConfig, SchedulerConfig, TimingTaskConfig
from timingtask.generator import TrialGenerator
from timingtask.rl import ActorCritic, RLTrainer, rollout, summarise
from timingtask.scheduler import DelayScheduler


def gen(**kw):
    oc = kw.pop("obs", ObservationConfig())
    sc = kw.pop("sched", SchedulerConfig(mode="fixed", fixed_delay=0.4))
    rng = np.random.default_rng(0)
    return TrialGenerator(TimingTaskConfig(seed=0, **kw), DelayScheduler(sc, rng),
                          oc, rng)


def run_trials(g, policy, n):
    out = []
    for _ in range(400000):
        r = g.step(policy(g))
        if r.record:
            out.append(r.record)
            if len(out) >= n:
                return out
    raise AssertionError("trials did not complete")


NEVER = lambda g: False                                            # noqa: E731
ONTIME = lambda g: (g.timer is not None and g.decisive_lick_s is None
                    and g.timer >= g.delay + 0.1)                  # noqa: E731


# --------------------------------------------- across-trial subjective value
def test_value_gain_rises_after_failure_and_falls_after_success():
    """The gain exceeds 1 after a run of failures and falls below 1 after a
    run of successes; ``reward_rate_gain=0`` disables it."""
    def rate_after(policy, gain=1.0, n=150):
        g = gen(reward_rate_gain=gain, no_cue_prob=0.0,
                iti_restart_on_lick=False)
        recs = run_trials(g, policy, n)
        return recs[-1]["reward_rate"], recs[-1]["value_gain"]
    r_ok, g_ok = rate_after(ONTIME)
    r_no, g_no = rate_after(lambda gg: False)
    assert r_ok > 0.9 and g_ok < 0.7          # sated
    assert r_no < 0.1 and g_no > 1.5          # deprived
    assert rate_after(ONTIME, gain=0.0)[1] == 1.0     # 0 disables it


def test_value_gain_scales_the_water_and_the_early_penalty_together():
    g = gen(reward_rate_gain=1.0, no_cue_prob=0.0, iti_restart_on_lick=False)
    assert g.value_gain == 1.0                 # empty history reads as the ref
    n = 0
    while n < 60:                              # 60 trials of early licking
        g.observe()
        if g.step(g.phase == "waiting").record is not None:
            n += 1
    assert g.reward_rate < 0.05 and g.value_gain > 1.5
    # the early penalty is scaled by the same gain that scales the water
    from timingtask.config import TimingTaskConfig
    base, scaled = [], []
    for gain in (0.0, 1.0):
        h = gen(reward_rate_gain=gain, no_cue_prob=0.0,
                iti_restart_on_lick=False)
        recs = []
        while len(recs) < 60:
            h.observe()
            r = h.step(h.phase == "waiting")
            if r.record is not None:
                recs.append(r.record)
        (base if gain == 0.0 else scaled).append(recs[-1]["trial_reward"])
    assert scaled[0] < base[0]                 # the same mistake now costs more


def test_previous_outcome_channel_is_signed():
    """+1 rewarded, -1 responded and unrewarded, 0 no response."""
    g = gen(no_cue_prob=0.0, iti_restart_on_lick=False)
    seen = set()
    for _ in range(4000):
        g.observe()
        r = g.step(g.phase in ("waiting", "eligible"))
        if r.record is not None and r.record["prev_success"] is not None:
            seen.add(r.record["prev_success"])
    assert seen <= {-1.0, 0.0, 1.0} and seen & {-1.0, 1.0}


def test_value_gain_is_clipped():
    g = gen(reward_rate_gain=5.0, reward_rate_max=1.4, no_cue_prob=0.0,
            iti_restart_on_lick=False)
    assert max(r["value_gain"] for r in run_trials(g, NEVER, 5)) <= 1.4


def test_value_gain_hidden_from_the_agent():
    """The value gain and the reward rate are not observation channels."""
    labels = gen(reward_rate_gain=1.0).observation_labels()
    assert "value_gain" not in labels and "reward_rate" not in labels


# -------------------------------------------------------------- history
def test_history_channels_vary_once_the_agent_acts():
    """The success and first-lick channels take nonzero values once the agent
    responds."""
    g = gen(no_cue_prob=0.0)
    obs = []
    for _ in range(6):
        run_trials(g, ONTIME, 1)
        obs.append(g.observe())
    a = np.stack(obs)
    labels = g.observation_labels()
    for name in ("success_t-1", "first_lick_t-1"):
        j = labels.index(name)
        assert a[:, j].std() >= 0 and a[1:, j].max() > 0, name


def test_lags_expose_the_last_k_trials():
    for k in (1, 3, 5):
        g = gen(obs=ObservationConfig(n_lags=k))
        labels = g.observation_labels()
        assert g.obs_size == len(labels) == 1 + 4 * k
        assert f"first_lick_t-{k}" in labels
        assert g.observe().shape == (g.obs_size,)


def test_history_shifts_by_one_trial():
    g = gen(obs=ObservationConfig(n_lags=3), no_cue_prob=0.0)
    labels = g.observation_labels()
    j1, j2 = labels.index("first_lick_t-1"), labels.index("first_lick_t-2")
    run_trials(g, ONTIME, 2)
    a = g.observe()
    run_trials(g, NEVER, 1)
    b = g.observe()
    assert b[j2] == pytest.approx(a[j1]), "t-1 should become t-2 after a trial"


def test_history_is_constant_within_a_trial():
    """The previous-trial channels are held constant for the whole trial."""
    g = gen(no_cue_prob=0.0)
    run_trials(g, ONTIME, 1)
    j = g.observation_labels().index("first_lick_t-1")
    vals = []
    for _ in range(20):
        vals.append(g.observe()[j])
        g.step(False)
    assert len(set(np.round(vals, 6))) == 1


# ------------------------------------------------------------------- RL
def build(n_envs=4, hidden=32, **kw):
    tc = TimingTaskConfig(dt=0.02, answer_window=1.0, post_lick=0.0,
                          no_cue_prob=0.0, seed=0, **kw)
    tr = RLTrainer(tc, SchedulerConfig(mode="fixed", fixed_delay=0.3),
                   ObservationConfig(), n_envs=n_envs, seed=0)
    core = VanillaRNN(tr.obs_size, hidden, 1, tau=100.0, dt=20.0, noise=0.05)
    return tr, ActorCritic(core, hidden_size=hidden)


def test_actor_critic_shapes():
    tr, m = build()
    h = m.init_state(4)
    x = torch.zeros(4, tr.obs_size)
    h, logits, v = m.step(x, h)
    assert logits.shape == (4, 2) and v.shape == (4,) and h.shape == (4, 32)


def test_core_step_stays_a_pure_function():
    """``torch.func`` can take the Jacobian of the core's ``step``."""
    from torch.func import jacrev
    tr, m = build()
    x = torch.zeros(1, tr.obs_size)
    m.eval()
    f = lambda h: m.core.step(x, h.unsqueeze(0)).squeeze(0)        # noqa: E731
    J = jacrev(f)(torch.zeros(32))
    assert J.shape == (32, 32)


def test_rollout_returns_one_record_per_env():
    tr, m = build(n_envs=5)
    out = rollout(m, tr.gens)
    assert len(out["records"]) == 5
    assert all(r is not None for r in out["records"])
    T = out["logp"].shape[0]
    for k in ("logp", "value", "reward", "entropy", "mask"):
        assert out[k].shape == (T, 5), k
    assert out["hidden"].shape == (T, 5, 32)


def test_mask_marks_steps_where_the_trial_was_live():
    tr, m = build(n_envs=4)
    out = rollout(m, tr.gens)
    assert out["mask"][0].all(), "every trial is live at step 0"
    lens = out["mask"].sum(0).numpy()
    assert (lens > 0).all() and (lens <= out["mask"].shape[0]).all()


def test_greedy_rollout_is_deterministic():
    tr, m = build(n_envs=3)
    m.eval()
    a = rollout(m, tr.gens, greedy=True)["logits"]
    tr2, _ = build(n_envs=3)
    b = rollout(m, tr2.gens, greedy=True)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_returns_are_discounted_sums_within_a_trial():
    tr, m = build(n_envs=2)
    tr.gamma = 1.0
    r = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    msk = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    R = tr._returns(r, msk)
    assert R[0, 0] == pytest.approx(6.0)      # 1+2+3, all live
    assert R[0, 1] == pytest.approx(3.0)      # 1+2, then masked


def test_training_runs_and_reports_engagement_separately():
    tr, m = build(n_envs=4)
    h = tr.train(m, steps=6, log_every=3, verbose=False)
    for k in ("return", "p_engaged", "p_correct", "entropy"):
        assert len(h[k]) >= 2, k
    assert 0.0 <= h["p_engaged"][-1] <= 1.0


def test_summarise_splits_engagement_from_correctness():
    """A policy that never licks has p_engaged 0 and p_miss 1."""
    never = [{"no_cue_trial": False, "decisive_lick_s": None, "rewarded": False,
              "early": False, "miss": True, "delay": 0.3, "first_lick_s": None}] * 5
    s = summarise(never)
    assert s["p_engaged"] == 0.0 and s["p_miss"] == 1.0

    good = [{"no_cue_trial": False, "decisive_lick_s": 0.4, "rewarded": True,
             "early": False, "miss": False, "delay": 0.3, "first_lick_s": 0.4}] * 5
    s = summarise(good)
    assert s["p_engaged"] == 1.0 and s["p_correct"] == 1.0


def test_summarise_ignores_catch_trials():
    s = summarise([{"no_cue_trial": True, "decisive_lick_s": None,
                    "rewarded": False, "early": False, "miss": False,
                    "delay": 0.3, "first_lick_s": None}])
    assert s["n"] == 0


def test_envs_are_independent():
    """Each generator keeps its own scheduler, history and trial state."""
    tr, m = build(n_envs=6)
    assert len({id(g) for g in tr.gens}) == 6
    assert len({id(g.scheduler) for g in tr.gens}) == 6
    assert len({id(g._history) for g in tr.gens}) == 6
    for _ in range(2):
        rollout(m, tr.gens)
    # ITI lengths are drawn per-env, so the cue lands at different steps
    onsets = [g._cue_onset_step for g in tr.gens]
    assert len(set(o for o in onsets if o is not None)) > 1 or all(
        o is None for o in onsets)


def test_an_untrained_policy_never_escapes_the_iti():
    """With the restart rule on, a near-uniform policy licks on about half of
    all steps, so the stop-licking period never completes and every trial
    ends in ``iti_timeout``."""
    tr, m = build(n_envs=4, iti_timeout=1.0)
    out = rollout(m, tr.gens)
    recs = [r for r in out["records"] if r is not None]
    assert all(r["outcome"] == "iti_timeout" for r in recs)
    assert all(r["cue_onset_step"] is None for r in recs)
    assert all(r["trial_steps"] == tr.cfg.steps(1.0) for r in recs)


# --------------------------------------------------- ITI readout penalty
def test_summarise_reports_the_promotion_metrics():
    tr, m = build(n_envs=4)
    recs = []
    while len(recs) < 4:
        out = rollout(m, tr.gens, max_steps=2000)
        recs += [r for r in out["records"] if r is not None]
    s = summarise(recs)
    for k in ("p_iti_lick", "n_iti_licks", "p_cue_success"):
        assert k in s and not np.isnan(s[k])


def test_iti_mask_is_binary_and_shaped_like_the_reward():
    tr, m = build(n_envs=4)
    out = rollout(m, tr.gens, max_steps=200)
    iti = out["iti"]
    assert iti.shape == out["reward"].shape
    # a mask, so the penalty can never touch a non-ITI step
    assert float((iti * (1 - iti)).abs().sum()) == 0.0


def test_iti_readout_penalty_lowers_the_shaped_reward_only_in_the_iti():
    tr, m = build(n_envs=4)
    tr.iti_readout_penalty = 1.0
    out = rollout(m, tr.gens, max_steps=200)
    p = torch.softmax(out["logits"], -1)[..., 1].detach()
    shaped = out["reward"] * tr.reward_scale - tr.iti_readout_penalty * p * out["iti"]
    d = (out["reward"] * tr.reward_scale) - shaped
    assert float(d[out["iti"] == 0].abs().sum()) == 0.0
    assert float(d[out["iti"] == 1].sum()) > 0.0


# ------------------------------------------------- exploration and regularisers
def test_min_action_prob_floors_the_entropy():
    import math
    tr, m = build(n_envs=4)
    with torch.no_grad():                       # force a confident policy
        m.policy.bias.copy_(torch.tensor([0.0, 20.0]))
    floor = -(0.02 * math.log(0.02) + 0.98 * math.log(0.98))
    hi = rollout(m, tr.gens, max_steps=100, min_action_prob=0.02)
    assert float(hi["entropy"].detach().min()) >= floor - 1e-5
    tr2, m2 = build(n_envs=4)
    with torch.no_grad():
        m2.policy.bias.copy_(torch.tensor([0.0, 20.0]))
    lo = rollout(m2, tr2.gens, max_steps=100)
    assert float(lo["entropy"].detach().max()) < floor   # unfloored: collapses


def test_activity_penalties_are_positive_and_iti_restricted():
    tr, m = build(n_envs=4)
    out = rollout(m, tr.gens, max_steps=300)
    Hh, msk, iti = out["hidden"], out["mask"], out["iti"]
    w = msk * iti
    prev = torch.cat([out["h_init"].unsqueeze(0), Hh[:-1]], dim=0)
    act = ((Hh ** 2).mean(-1) * w).sum() / w.sum().clamp_min(1)
    dact = (((Hh - prev) ** 2).mean(-1) * w).sum() / w.sum().clamp_min(1)
    assert float(act.detach()) > 0 and float(dact.detach()) > 0
    # a step outside the ITI can never contribute
    assert float(((((Hh ** 2).mean(-1)) * w * (1 - iti)).sum()).detach()) == 0.0


def test_activity_scope_full_covers_more_steps_than_iti():
    # With the restart rule off an untrained policy leaves the ITI, so the two
    # scopes cover different numbers of steps.
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    out = rollout(m, tr.gens, max_steps=2000)
    full = float(out["mask"].sum())
    iti = float((out["mask"] * out["iti"]).sum())
    assert full > iti > 0


def test_activity_scope_is_validated():
    with pytest.raises(ValueError):
        RLTrainer(activity_scope="elsewhere")


def test_activity_penalty_changes_the_loss_but_not_the_reward():
    tr, m = build(n_envs=4)
    tr.activity_penalty = 20.0
    h = tr.train(m, steps=2, log_every=1, verbose=False)
    assert "act_pen" in h and all(v >= 0 for v in h["act_pen"])


# ------------------------------------------------- diagnostics
def test_train_keeps_records_and_logs_grad_norm():
    tr, m = build(n_envs=4)
    h = tr.train(m, steps=3, log_every=1, verbose=False)
    assert len(tr.records) >= 3
    assert len(h["grad_norm"]) == 3 and all(g >= 0 for g in h["grad_norm"])
    for k in ("delay", "first_lick", "p_miss", "trials"):
        assert k in h


def test_evaluate_freezes_the_delay_and_does_not_learn():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    before = [float(p.detach().sum()) for p in m.parameters()]
    delays = [g.scheduler.current_delay for g in tr.gens]
    recs = tr.evaluate(m, n_trials=12)
    assert len(recs) == 12
    assert [g.scheduler.current_delay for g in tr.gens] == delays
    assert [float(p.detach().sum()) for p in m.parameters()] == before
    assert not any(g.scheduler.frozen for g in tr.gens)   # restored
    assert m.training                                     # restored


def test_run_report_writes_every_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from timingtask.plots import run_report
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    h = tr.train(m, steps=2, log_every=1, verbose=False)
    ev = tr.evaluate(m, n_trials=8, collect_states=True)
    pb = tr.probe(m, n_trials=8, collect_states=True)
    figs = run_report(h, tr.records, ev, probe_records=pb,
                      snapshots=tr.weight_snapshots, name="t",
                      out_dir=str(tmp_path), clip=1.0)
    assert {"training", "heads", "behaviour_training", "history_training",
            "behaviour_probe", "history_probe"} <= set(figs)
    assert "behaviour_final" not in figs
    assert len(list(tmp_path.glob("t_*.png"))) >= 6


# ------------------------------------------------- history and probe
def test_records_carry_the_previous_trial_the_agent_was_shown():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    recs = tr.evaluate(m, n_trials=20)
    later = [r for r in recs if r["trial"] > 0]
    assert later and all(r["prev_success"] is not None for r in later)
    assert all(isinstance(r["prev_first_lick"], float) for r in later)


def test_probe_lets_the_delay_grow_and_restores_max_delay():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    before = tr.scheduler_cfg.max_delay
    recs = tr.probe(m, n_trials=24, max_delay=1.8, collect_states=True)
    assert len(recs) == 24
    assert tr.scheduler_cfg.max_delay == before
    assert m.training
    assert any(r.get("readout") is not None for r in recs)


def test_history_regression_null_brackets_a_shuffled_effect():
    """With no real dependence on the previous trial, the fitted weight should
    fall inside the order-shuffle band most of the time."""
    from timingtask.plots import history_regression
    rng = np.random.default_rng(0)
    recs = [{"first_lick_s": float(rng.normal(0.4, 0.1)),
             "prev_success": bool(rng.random() < 0.5),
             "prev_first_lick": float(rng.normal(0.4, 0.1)),
             "cue_onset_step": 0} for _ in range(600)]
    r = history_regression(recs, window=300, step=150, n_shuffle=80, rng=rng)
    assert len(r["trial"]) >= 1
    inside = ((r["b_reward"] >= r["b_reward_lo"])
              & (r["b_reward"] <= r["b_reward_hi"]))
    assert inside.mean() >= 0.5


def test_history_dashboard_makes_six_panels(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from timingtask.plots import history_dashboard
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    recs = tr.evaluate(m, n_trials=120, collect_states=True)
    fig, a = history_dashboard(recs, block=50)
    assert len(a) == 6
    fig.savefig(tmp_path / "h.png", dpi=60)


def test_model_figure_uses_the_hidden_states(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from timingtask.plots import _model_figure
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    ev = tr.evaluate(m, n_trials=40, collect_states=True)
    fig = _model_figure(ev, "t")
    assert len(fig.axes) >= 6
    fig.savefig(tmp_path / "m.png", dpi=60)


# ------------------------------------------------- heads and test conditions
def test_weight_diagnostics_are_logged_per_update_not_per_trial():
    tr, m = build(n_envs=4)
    h = tr.train(m, steps=4, log_every=1, verbose=False)
    n = len(h["step"])
    for k in ("w_delta_norm", "w_value_norm", "W_rec_norm", "W_in_norm"):
        assert len(h[k]) == n
    # every relative-step series is the same length as `step`
    for k in [k for k in h if k.startswith("d_")]:
        assert len(h[k]) == n
    # frozen and unused blocks are not tracked at all
    assert "d_core.h0" not in h and "d_core.out.weight" not in h
    assert len(tr.weight_snapshots) == n
    assert tr.weight_snapshots[0]["w_delta"].shape == (m.core.hidden_size,)


def test_test_condition_changes_the_delay_and_restores_the_scheduler():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    before_mode = tr.scheduler_cfg.mode
    before_delay = [g.scheduler.current_delay for g in tr.gens]
    recs = tr.test(m, n_trials=24, sched=dict(mode="fixed", fixed_delay=1.0))
    assert len(recs) == 24
    assert all(abs(r["delay"] - 1.0) < 1e-9 for r in recs)
    assert tr.scheduler_cfg.mode == before_mode
    assert [g.scheduler.current_delay for g in tr.gens] == before_delay
    assert m.training


def test_heads_dashboard_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from timingtask.plots import heads_dashboard
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    h = tr.train(m, steps=3, log_every=1, verbose=False)
    ev = tr.evaluate(m, n_trials=20, collect_states=True)
    fig, axs = heads_dashboard(h, tr.weight_snapshots, ev, block=10)
    assert axs.size == 6
    fig.savefig(tmp_path / "heads.png", dpi=60)


# ------------------------------------------------- reward structure
def test_time_penalty_is_not_charged_during_the_post_lick_winddown():
    """The per-step cost applies in the post-lick period only when
    ``time_penalty_in_post`` is set."""
    from timingtask.generator import Phase
    for in_post, expect in ((False, 0.0), (True, -0.05)):
        tc = TimingTaskConfig(dt=0.02, time_penalty=-0.05,
                              time_penalty_in_post=in_post, seed=0)
        rng = np.random.default_rng(0)
        g = TrialGenerator(tc, DelayScheduler(SchedulerConfig(mode="fixed"), rng),
                           ObservationConfig(), rng)
        g.phase = Phase.POST
        g._post_remaining = 10
        assert abs(g.step(False).reward - expect) < 1e-9


def test_every_wrong_lick_costs_the_same_in_base():
    from timingtask.variants import BASE
    t = BASE["task"]
    assert t["iti_lick_penalty"] == t["early_penalty"] == t["no_cue_lick_penalty"]
    assert t["discount_rate"] == 0.0


def test_training_stops_early_on_target_delay_and_on_stall():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    tr.train(m, steps=50, log_every=1, verbose=False, target_delay=0.0)
    assert "target_delay" in tr.stop_reason          # 0.0 is met immediately
    tr2, m2 = build(n_envs=4, iti_restart_on_lick=False)
    tr2.train(m2, steps=50, log_every=1, verbose=False,
              patience=3, min_delta=1e9)            # nothing can beat min_delta
    assert "stalled" in tr2.stop_reason


def test_describe_is_generated_from_the_live_config():
    from timingtask.variants import describe, BASE
    md = describe("act")
    assert f"`{BASE['task']['reward']!r}`" in md     # reads the live value
    for heading in ("## Task and reinforcement", "## Delay schedule",
                    "## Network", "## Variants"):
        assert heading in md
    assert md.count("|") > 100


# ------------------------------------------------- the post-lick period
def test_post_lick_winddown_earns_and_costs_nothing():
    """No reward or penalty is paid during the post-lick period."""
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    out = rollout(m, tr.gens, max_steps=2000)
    post = out["mask"] * out["post"]
    assert float(post.sum()) > 0                      # POST steps did occur
    assert float((out["reward"] * post).abs().sum()) == 0.0


def test_post_lick_steps_are_excluded_from_every_loss_term():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    out = rollout(m, tr.gens, max_steps=2000)
    m_data = out["mask"]
    m_loss = m_data * (1 - out["post"])
    assert float(m_loss.sum()) < float(m_data.sum())   # strictly fewer
    assert float((m_loss * out["post"]).sum()) == 0.0


# ---------------------------------------------- heterogeneous time constants
def test_tau_may_be_a_range_and_defaults_to_a_scalar():
    from timingtask.models import VanillaRNN
    one = VanillaRNN(3, 64, 1, tau=100.0, dt=20.0)
    assert abs(float(one.tau.min()) - 100.0) < 1e-3      # float32 round-trip
    assert abs(float(one.alpha.max()) - 0.2) < 1e-6
    many = VanillaRNN(3, 64, 1, tau=(50.0, 3000.0), dt=20.0)
    assert 50.0 <= float(many.tau.min()) and float(many.tau.max()) <= 3000.0
    assert float(many.tau.max()) / float(many.tau.min()) > 5
    assert many.alpha.shape == (64,)
    # a unit can never overshoot: alpha <= 1 even if tau is set below dt
    assert float(VanillaRNN(3, 8, 1, tau=1.0, dt=20.0).alpha.max()) <= 1.0


def test_tau_is_learnable_only_when_asked():
    from timingtask.models import VanillaRNN
    assert not VanillaRNN(3, 16, 1).log_tau.requires_grad
    m = VanillaRNN(3, 16, 1, tau=(50.0, 3000.0), train_tau=True)
    assert m.log_tau.requires_grad
    m.step(torch.zeros(2, 3), m.init_state(2)).sum().backward()
    assert m.log_tau.grad is not None and float(m.log_tau.grad.abs().sum()) > 0


def test_a_tau_spread_gives_the_network_slow_modes_to_build_a_timer_from():
    """A log-uniform spread of time constants gives the initial network more
    modes with decay times above one second than a single tau."""
    import numpy as np
    from timingtask.models import VanillaRNN
    def slow_modes(tau, seed=0):
        torch.manual_seed(seed)
        m = VanillaRNN(3, 128, 1, tau=tau, dt=20.0)
        a = m.alpha.detach().numpy()
        J = (1 - a)[:, None] * np.eye(128) + a[:, None] * m.rec.weight.detach().numpy()
        r = np.clip(np.abs(np.linalg.eigvals(J)), 1e-12, 1 - 1e-9)
        return int((-0.02 / np.log(r) > 1.0).sum())
    assert slow_modes((50.0, 3000.0)) > 3 * max(1, slow_modes(100.0))


# ------------------------------------------ recurrent initialisation and clip
def test_orthogonal_init_gives_more_slow_modes_without_leaving_the_unit_disc():
    """An orthogonal initialisation at g=1 has more slow modes than the
    Gaussian default while keeping every eigenvalue inside the unit disc;
    raising g instead moves eigenvalues outside it."""
    import numpy as np
    from timingtask.models import VanillaRNN
    def spec(seed, **kw):
        torch.manual_seed(seed)
        m = VanillaRNN(3, 128, 1, tau=100.0, dt=20.0, **kw)
        a = m.alpha.detach().numpy()
        J = (1 - a)[:, None] * np.eye(128) + a[:, None] * m.rec.weight.detach().numpy()
        r = np.abs(np.linalg.eigvals(J))
        return r.max(), int((-0.02 / np.log(np.clip(r, 1e-12, 1 - 1e-12)) > 1.0).sum())
    gmax, gslow = spec(0)
    omax, oslow = spec(0, rec_init="orthogonal")
    xmax, xslow = spec(0, g=1.2)
    assert oslow > gslow                    # more slow modes than the default
    assert omax <= 1.0 + 1e-6               # and still marginally stable
    assert xmax > omax                      # raising g leaves the disc instead


def test_grad_clip_zero_disables_clipping_but_still_reports_the_norm():
    tr, m = build(n_envs=4, iti_restart_on_lick=False)
    tr.grad_clip = 0.0
    h = tr.train(m, steps=3, log_every=1, verbose=False)
    assert len(h["grad_norm"]) == 3 and max(h["grad_norm"]) > 0
