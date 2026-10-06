"""Tests for the supervised interface to the timing task."""
import numpy as np
import pytest
import torch
from timingtask.config import ObservationConfig, SchedulerConfig, TimingTaskConfig
from timingtask.supervised import LICK, WITHHOLD, TimingTask


def mk(**kw):
    tk = {k: v for k, v in kw.items() if hasattr(TimingTaskConfig, k)}
    rest = {k: v for k, v in kw.items() if k not in tk}
    tk.setdefault("answer_window", 2.0)
    tk.setdefault("post_lick", 0.0)
    return TimingTask(TimingTaskConfig(seed=0, **tk),
                      SchedulerConfig(mode="fixed", fixed_delay=0.4),
                      seed=0, **rest)


def test_spec_has_a_withhold_unit_and_a_lick_unit():
    t = mk(no_cue_prob=0.0)
    assert t.spec.output_dim == 2 and t.spec.loss == "mse"
    assert t.spec.output_labels == ("withhold", "lick")
    assert t.dt == pytest.approx(t.cfg.dt * 1000.0), "Task.dt is in milliseconds"


def test_batch_shapes():
    t = mk(no_cue_prob=0.0)
    b = t.sample(6)
    T = b.inputs.shape[1]
    assert b.inputs.shape == (6, T, t.spec.input_dim)
    assert b.targets.shape == (6, T, 2)
    assert b.meta["weight"].shape == (6, T)
    assert b.loss_mask.any(1).all()


def test_nothing_acts_during_training():
    """The generator is rolled with a non-licking observer to lay out the
    epochs, so every trial runs its full answer window and ends in 'miss'."""
    t = mk(no_cue_prob=0.0)
    _, rec = t._roll_trial()
    assert rec["outcome"] == "miss"
    assert rec["first_lick_s"] is None and rec["n_licks"] == 0


def test_withhold_unit_falls_when_the_delay_elapses():
    t = mk(no_cue_prob=0.0)
    b = t.sample(4)
    for i in range(4):
        o = int(b.meta["cue_onset_step"][i])
        d = int(round(float(b.meta["delay"][i]) / t.cfg.dt))
        w = b.targets[i, :, WITHHOLD]
        assert float(w[o + d - 3]) == pytest.approx(t.hold_level)
        assert float(w[o + d + 5]) < 0.1


def test_lick_target_keeps_rising_past_the_threshold():
    """The lick target equals the threshold at the target time and continues
    to rise afterwards, up to ``plateau * threshold``."""
    t = mk(no_cue_prob=0.0, plateau=1.5, ramp_exponent=1.0)
    b = t.sample(4)
    for i in range(4):
        o = int(b.meta["cue_onset_step"][i])
        star = int(round(float(b.meta["target_time_s"][i]) / t.cfg.dt))
        n = int(b.meta["trial_len"][i])
        if o + star + 2 >= n:
            continue
        z = b.targets[i, :, LICK]
        assert float(z[o + star]) == pytest.approx(1.0, abs=1e-5)
        assert float(z[o + star + 1]) > float(z[o + star])
    assert float(b.targets[..., LICK].max()) <= 1.5 + 1e-6


def test_plateau_must_exceed_one():
    with pytest.raises(ValueError):
        mk(plateau=1.0)


def test_cost_mask_is_weighted_not_boolean():
    """Grace windows at zero weight, answer window up-weighted."""
    t = mk(no_cue_prob=0.0, w_answer=5.0, grace=0.1)
    b = t.sample(4)
    w = b.meta["weight"]
    assert float(w.max()) == pytest.approx(5.0)
    assert float(w.min()) == 0.0
    g = int(round(0.1 / t.cfg.dt))
    for i in range(4):
        assert not w[i, :g].any(), "settle-in period at trial start"


def test_catch_trials_hold_withhold_throughout():
    """On catch trials the withhold target is held for the whole trial
    (checked within each trial's own length; the batch is zero-padded)."""
    t = mk(no_cue_prob=1.0)
    b = t.sample(4)
    assert bool(b.meta["no_cue_trial"].all())
    assert float(b.targets[..., LICK].abs().max()) == 0.0
    for i in range(4):
        n = int(b.meta["trial_len"][i])
        w = b.targets[i, :n, WITHHOLD]
        assert float(w.min()) == pytest.approx(t.hold_level)


def test_variability_sources_are_on():
    """The ITI varies across trials and input noise is applied."""
    t = mk(no_cue_prob=0.0, sigma_x=0.05)
    b = t.sample(16)
    assert len(set(b.meta["cue_onset_step"].tolist())) > 4, "ITI must vary"
    t0 = mk(no_cue_prob=0.0, sigma_x=0.0)
    assert float(t0.sample(8).inputs[..., 0].std()) < float(
        t.sample(8).inputs[..., 0].std())


def test_crossing_time_and_outcome_split():
    t = mk(no_cue_prob=0.0)
    b = t.sample(8)
    B, T = b.inputs.shape[0], b.inputs.shape[1]
    z = torch.zeros(B, T, 2)
    for i in range(B):
        o = int(b.meta["cue_onset_step"][i])
        k = o + int(round(0.6 / t.cfg.dt))
        if k < T:
            z[i, k:, LICK] = 1.0
    ct = t.crossing_time(z, b)
    assert torch.allclose(ct, torch.full_like(ct, 0.6), atol=t.cfg.dt)
    c = t.outcome_counts(z, b)
    assert c["rewarded"] == 8 and c["p_engaged"] == 1.0 and c["p_correct"] == 1.0

    silent = t.outcome_counts(torch.zeros(B, T, 2), b)
    assert silent["miss"] == 8 and silent["p_engaged"] == 0.0


def test_outcome_counts_reports_engagement_separately():
    """A silent output has p_engaged 0 and every cued trial counted as a miss."""
    t = mk(no_cue_prob=0.0)
    b = t.sample(6)
    c = t.outcome_counts(torch.zeros_like(b.targets), b)
    assert c["p_engaged"] == 0.0
    assert c["rewarded"] + c["early"] + c["miss"] == c["n_cued"]


def test_weighted_loss_uses_the_weights():
    t = mk(no_cue_prob=0.0)
    b = t.sample(4)
    perfect = t.loss(b.targets, b)
    wrong = t.loss(torch.zeros_like(b.targets), b)
    assert float(perfect) == pytest.approx(0.0, abs=1e-6)
    assert float(wrong) > 0.0


@pytest.mark.slow
def test_fixed_delay_is_learnable_as_a_smoke_test():
    """A fixed-delay task is learnable in a few hundred supervised steps."""
    from timingtask.models import VanillaRNN
    from timingtask.training import train
    torch.manual_seed(0)
    t = mk(no_cue_prob=0.0)
    m = VanillaRNN(t.spec.input_dim, 128, t.spec.output_dim, tau=100.0, dt=t.dt)
    train(m, t, steps=300, batch_size=16, log_every=1000, verbose=False)
    b = t.sample(32)
    with torch.no_grad():
        out, H = m(b.inputs)
    assert t.outcome_counts(out, b)["p_engaged"] > 0.8
    assert H.shape[:2] == b.inputs.shape[:2], "hidden is (B, T, N)"
