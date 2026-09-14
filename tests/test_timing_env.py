"""Tests for the gymnasium face and the monitor stream."""
import json
import numpy as np
import pytest
from timingtask import (JSONLMonitor, MemoryMonitor, Monitor, MonitorList,
                    ObservationConfig, SchedulerConfig, TimingTaskConfig,
                    read_jsonl, records_to_arrays)
from timingtask.env import TimingTaskEnv


def wait(env, n=100000, action=0):
    """Step until the episode truncates, returning (n_steps, last_info)."""
    for i in range(n):
        _, _, term, trunc, info = env.step(action)
        assert term is False, "task has no terminal state"
        if trunc:
            return i + 1, info
    raise AssertionError("episode never truncated")


# ------------------------------------------------------------------- gym API
def test_reset_step_contract():
    env = TimingTaskEnv(variant="fixed", trials_per_episode=3)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(r, float) and term is False and isinstance(trunc, bool)
    assert "phase" in info and "cue_on" in info


def test_episode_is_a_session_of_trials():
    env = TimingTaskEnv(variant="fixed", trials_per_episode=4)
    env.reset(seed=0)
    _, info = wait(env)
    assert info["trials_done"] == 4


def test_episode_end_is_truncated_not_terminated():
    """No terminal state -- an agent must not bootstrap a zero value here."""
    env = TimingTaskEnv(variant="fixed", trials_per_episode=2)
    env.reset(seed=0)
    _, info = wait(env)
    assert info["trials_done"] == 2


def test_reset_is_a_new_animal():
    """The old bug: reset(seed=) rebuilt the scheduler and silently wiped
    autolearn progress while other state survived. Now nothing survives."""
    sc = SchedulerConfig(mode="autolearn", perf_window=5, min_trials_per_delay=5,
                         initial_delay=0.1, delay_step=0.1)
    env = TimingTaskEnv(TimingTaskConfig(seed=0), sc, trials_per_episode=100000)
    env.reset(seed=0)
    for _ in range(20000):
        env.step(0)
        if env.scheduler.delay_success_rate() is not None:
            break
    env.reset(seed=0)
    assert env.scheduler.delay_success_rate() is None
    assert env.gen.trial_index == 0
    assert env.scheduler.get_current_delay() == pytest.approx(0.1)


def test_same_seed_same_episode():
    def rollout(seed):
        env = TimingTaskEnv(variant="fixed", trials_per_episode=5)
        env.reset(seed=seed)
        mon = MemoryMonitor()
        env.monitors.append(mon)
        wait(env)
        return [(r["outcome"], round(r["trial_duration_s"], 6)) for r in mon.records]
    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


# ---------------------------------------------------------------- actions
def test_discrete_action_validation():
    env = TimingTaskEnv(variant="fixed", trials_per_episode=2)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(7)


def test_continuous_action_is_threshold_crossing():
    """Ramp-to-bound: the readout is a scalar, a lick is a threshold crossing."""
    env = TimingTaskEnv(variant="fixed", trials_per_episode=2,
                        action_mode="continuous", lick_threshold=0.5)
    env.reset(seed=0)
    assert env.action_space.shape == (1,)
    assert env._to_lick(np.array([0.9], np.float32)) is True
    assert env._to_lick(np.array([0.1], np.float32)) is False
    assert env._to_lick(np.array([0.5], np.float32)) is False   # strict >
    env.step(np.array([0.0], np.float32))


def test_continuous_and_discrete_agree_given_the_same_licks():
    cfg = dict(trials_per_episode=6)
    a = TimingTaskEnv(TimingTaskConfig(seed=1), SchedulerConfig(mode="fixed"), **cfg)
    b = TimingTaskEnv(TimingTaskConfig(seed=1), SchedulerConfig(mode="fixed"),
                      action_mode="continuous", **cfg)
    ma, mb = MemoryMonitor(), MemoryMonitor()
    a.reset(seed=1); a.monitors.append(ma)
    b.reset(seed=1); b.monitors.append(mb)
    for _ in range(20000):
        _, _, _, ta, _ = a.step(0)
        _, _, _, tb, _ = b.step(np.array([0.0], np.float32))
        if ta or tb:
            break
    assert [r["outcome"] for r in ma.records] == [r["outcome"] for r in mb.records]


def test_variant_or_configs_not_both():
    with pytest.raises(ValueError):
        TimingTaskEnv(TimingTaskConfig(), variant="fixed")


# --------------------------------------------------------------- monitors
def test_memory_monitor_receives_every_trial():
    mon = MemoryMonitor()
    env = TimingTaskEnv(variant="fixed", trials_per_episode=7,
                        monitors=MonitorList([mon]))
    env.reset(seed=0)
    wait(env)
    assert len(mon.records) == 7
    assert len(mon.resets) == 1
    assert [r["trial"] for r in mon.records] == list(range(7))


def test_a_failing_monitor_cannot_take_down_the_run(capsys):
    class Bad(Monitor):
        def on_trial(self, record):
            raise RuntimeError("boom")

    good = MemoryMonitor()
    mons = MonitorList([Bad(), good])
    env = TimingTaskEnv(variant="fixed", trials_per_episode=5, monitors=mons)
    env.reset(seed=0)
    wait(env)
    assert len(good.records) == 5, "the healthy monitor must keep receiving"
    assert len(mons.monitors) == 1, "the failing monitor is dropped"
    assert any("boom" in f for f in mons.failed)


def test_jsonl_roundtrip_is_the_durable_record(tmp_path):
    p = tmp_path / "run.jsonl"
    mons = MonitorList([JSONLMonitor(str(p))])
    env = TimingTaskEnv(variant="fixed", trials_per_episode=6, monitors=mons)
    env.reset(seed=0)
    wait(env)
    env.close()

    recs = read_jsonl(str(p))
    assert len(recs) == 6
    assert read_jsonl(str(p), event=None)[0]["_event"] == "reset"
    for line in p.read_text().splitlines():
        json.loads(line)                       # every line valid JSON

    cols = records_to_arrays(recs)
    assert cols["delay"].shape == (6,)
    assert cols["rewarded"].dtype == bool
    assert len(cols["lick_times_s"]) == 6
    assert cols["outcome"].dtype == object


def test_records_to_arrays_handles_missing_values():
    recs = [{"trial": 0, "first_lick_s": None, "outcome": "miss",
             "rewarded": False, "lick_times_s": []},
            {"trial": 1, "first_lick_s": 0.4, "outcome": "rewarded",
             "rewarded": True, "lick_times_s": [0.4, 0.5]}]
    cols = records_to_arrays(recs)
    assert np.isnan(cols["first_lick_s"][0]) and cols["first_lick_s"][1] == 0.4
    assert cols["rewarded"].tolist() == [False, True]


def test_records_to_arrays_empty():
    assert records_to_arrays([]) == {}


# ------------------------------------------------------- behaviour sanity
def test_a_perfect_timer_gets_rewarded_and_promotes_the_delay():
    """An agent that licks exactly when eligible should hit 100% and drive the
    autolearn staircase upward."""
    sc = SchedulerConfig(mode="autolearn", initial_delay=0.1, delay_step=0.1,
                         perf_window=10, min_trials_per_delay=10, max_delay=0.5)
    mon = MemoryMonitor()
    env = TimingTaskEnv(TimingTaskConfig(seed=0, no_cue_prob=0.0), sc,
                        trials_per_episode=80, monitors=MonitorList([mon]))
    env.reset(seed=0)
    for _ in range(200000):
        act = 1 if env.gen.phase == "eligible" else 0
        _, _, _, trunc, _ = env.step(act)
        if trunc:
            break
    assert all(r["rewarded"] for r in mon.records)
    assert mon.records[-1]["delay"] > mon.records[0]["delay"]
    assert env.scheduler.get_current_delay() == pytest.approx(0.5)


def test_an_impatient_agent_licks_early_and_never_promotes():
    sc = SchedulerConfig(mode="autolearn", initial_delay=0.5, delay_step=0.1,
                         perf_window=10, min_trials_per_delay=10)
    mon = MemoryMonitor()
    env = TimingTaskEnv(TimingTaskConfig(seed=0, no_cue_prob=0.0), sc,
                        trials_per_episode=30, monitors=MonitorList([mon]))
    env.reset(seed=0)
    for _ in range(200000):
        act = 1 if env.gen.phase in ("waiting", "eligible") else 0
        _, _, _, trunc, _ = env.step(act)
        if trunc:
            break
    assert all(r["early"] for r in mon.records)
    assert env.scheduler.get_current_delay() == pytest.approx(0.5)
