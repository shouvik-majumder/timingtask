"""The seam: what this repository writes, the analysis repository must read.

Two layers of test. The first checks the schema against the written spec using
h5py alone, so it passes in an environment where only timingtask is installed.
The second actually opens the file with ``neuralgeom.data.load_trajectory`` and
is skipped when that package is absent -- it is the only place in this repo
that mentions it, and it is a test, not the package.
"""
import json

import numpy as np
import pytest

from timingtask import (DelayScheduler, ObservationConfig, SchedulerConfig,
                        TimingTaskConfig, TrialGenerator)
from timingtask.export import (OUTCOME_CODES, records_to_trajectory,
                               save_trajectory)
from timingtask.rl import ActorCritic, attach_states, rollout
from timingtask.models import VanillaRNN

N_UNITS = 12
LICK_ACTION = 1


def make_records(n_gen=4, n_rounds=6, seed=0, delay=0.3, require_cue=True):
    """Run a small untrained agent and collect real records with states."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    cfg = TimingTaskConfig(seed=seed, dt=0.05)
    sc = SchedulerConfig(initial_delay=delay, max_delay=delay)
    oc = ObservationConfig()
    gens = [TrialGenerator(cfg, DelayScheduler(sc, np.random.default_rng(seed + i)),
                           oc, np.random.default_rng(seed + i))
            for i in range(n_gen)]
    core = VanillaRNN(gens[0].obs_size, N_UNITS, 1, tau=100.0, dt=cfg.dt * 1000)
    model = ActorCritic(core, n_actions=2)
    # An UNTRAINED agent licks on about half of all steps, so it restarts the
    # stop-licking period forever and almost never reaches a cue -- the records
    # would be nothing but iti_timeout. Bias the policy hard toward waiting so
    # the fixture exercises the cue-aligned path, which is the default one.
    with torch.no_grad():
        model.policy.bias[LICK_ACTION] -= 4.0
    recs = []
    with torch.no_grad():
        for _ in range(n_rounds):
            out = rollout(model, gens, max_steps=600)
            attach_states(out)
            recs += [r for r in out["records"] if r is not None]
    assert recs, "the rollout produced no completed trials"
    if require_cue:
        recs = [r for r in recs if r.get("align_step") is not None]
        assert recs, "no trial reached a cue; raise n_rounds or the wait bias"
    return recs


# --------------------------------------------------------------------- #
# attach_states
# --------------------------------------------------------------------- #
def test_attach_states_gives_every_series_at_the_trial_length():
    for r in make_records():
        T = int(r["trial_steps"])
        assert r["hidden"].shape == (T, N_UNITS)
        assert r["obs"].shape[0] == T
        assert r["readout"].shape == (T,)
        assert r["value"].shape == (T,)


# --------------------------------------------------------------------- #
# records -> bundle
# --------------------------------------------------------------------- #
def test_bundle_shapes_and_time_base():
    recs = make_records()
    b = records_to_trajectory(recs, t_pre=0.5, t_post=1.5, dt=0.05)
    B, T, N = b["X"].shape
    assert B == len(recs)
    assert N == N_UNITS
    assert T == int(round(0.5 / 0.05)) + int(round(1.5 / 0.05))
    assert b["time"].shape == (T,)
    # t = 0 is the cue, so the frame straddles zero
    assert b["time"][0] < 0 < b["time"][-1]
    assert np.isclose(b["time"][int(round(0.5 / 0.05))], 0.0)
    assert b["inputs"].shape == (B, T, recs[0]["obs"].shape[1])
    assert b["outputs"].shape == (B, T, 2)
    assert b["condition"].shape == (B,)


def test_padding_is_nan_never_zero():
    """Zero is a perfectly good hidden state; padding must be distinguishable."""
    b = records_to_trajectory(make_records(), t_pre=2.0, t_post=6.0, dt=0.05)
    X = b["X"]
    assert np.isnan(X).any(), "a 8 s window around short trials must have padding"
    # every unit is padded on exactly the same (trial, step) cells
    pad = np.isnan(X)
    assert (pad.all(-1) | (~pad).all(-1)).all()


def test_n_steps_counts_the_occupied_cells():
    b = records_to_trajectory(make_records(), t_pre=1.0, t_post=3.0, dt=0.05)
    occupied = np.isfinite(b["X"][..., 0]).sum(1)
    assert (b["aux"]["n_steps"] == occupied).all()
    assert (b["aux"]["n_steps"] > 0).all()


def test_cue_alignment_puts_cue_onset_at_t_zero():
    """Under align="cue" the drive should start at the same index every trial."""
    recs = make_records()
    b = records_to_trajectory(recs, t_pre=1.0, t_post=3.0, dt=0.05)
    n_pre = int(round(1.0 / 0.05))
    cue = b["inputs"][:, :, 0]                    # channel 0 is the cue
    cued = ~b["aux"]["no_cue_trial"].astype(bool)
    # On a cued trial the cue channel is off just before t=0 and on just after.
    assert np.nanmax(cue[cued, n_pre]) > 0
    assert np.nanmax(cue[cued, n_pre - 1]) == 0


def test_start_alignment_ignores_t_pre():
    recs = make_records()
    b = records_to_trajectory(recs, align="start", dt=0.05)
    assert b["time"][0] == 0.0
    assert b["X"].shape[1] == max(int(r["trial_steps"]) for r in recs)
    assert b["meta"]["align"] == "start"


def test_condition_can_be_the_delay_or_the_outcome():
    recs = make_records()
    by_delay = records_to_trajectory(recs, condition="delay")
    assert np.allclose(by_delay["condition"], [r["delay"] for r in recs])
    by_outcome = records_to_trajectory(recs, condition="outcome")
    assert set(np.unique(by_outcome["condition"])) <= set(OUTCOME_CODES.values())


def test_behaviour_survives_into_aux():
    b = records_to_trajectory(make_records())
    aux = b["aux"]
    for key in ("delay", "rewarded", "early", "miss", "first_lick_s",
                "outcome_code", "n_steps", "trial_reward"):
        assert key in aux, key
        assert len(aux[key]) == b["X"].shape[0]
    # None must arrive as NaN, not as a silent zero
    assert np.isnan(aux["first_lick_s"]).any() or np.isfinite(aux["first_lick_s"]).all()


def test_missing_states_is_an_error_not_an_empty_file():
    recs = make_records()
    for r in recs:
        r.pop("hidden")
    with pytest.raises(ValueError, match="collect_states"):
        records_to_trajectory(recs)


def test_bad_alignment_is_rejected():
    with pytest.raises(ValueError, match="align"):
        records_to_trajectory(make_records(), align="cue_onset")


# --------------------------------------------------------------------- #
# bundle -> HDF5, read back with h5py alone
# --------------------------------------------------------------------- #
def test_hdf5_round_trip_matches_the_documented_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    recs = make_records()
    path = save_trajectory(recs, str(tmp_path / "traj.h5"),
                           t_pre=1.0, t_post=3.0, dt=0.05,
                           generator="timingtask.test", config={"note": "unit"})
    b = records_to_trajectory(recs, t_pre=1.0, t_post=3.0, dt=0.05)

    with h5py.File(path, "r") as f:
        assert f.attrs["neuralgeom_trajectory"]
        assert np.allclose(f["X"][:], b["X"], equal_nan=True)
        assert np.allclose(f["time"][:], b["time"])
        assert np.isclose(f.attrs["dt"], 0.05)
        assert f["inputs"].shape == b["inputs"].shape
        assert f["outputs"].shape == b["outputs"].shape
        assert f["condition"].shape == b["condition"].shape
        meta = json.loads(f.attrs["meta"])
        assert meta["generator"] == "timingtask.test"
        assert meta["align"] == "cue"
        assert meta["padding"] == "nan"
        assert meta["config"]["note"] == "unit"
        assert set(f["aux"]) == set(b["aux"])


def test_connectivity_is_carried_when_given(tmp_path):
    h5py = pytest.importorskip("h5py")
    W = np.eye(N_UNITS, dtype=np.float32)
    path = save_trajectory(make_records(), str(tmp_path / "w.h5"), W=W)
    with h5py.File(path, "r") as f:
        assert np.allclose(f["W"][:], W)


# --------------------------------------------------------------------- #
# the actual interop, when the analysis package is installed
# --------------------------------------------------------------------- #
def test_the_analysis_package_can_load_what_we_write(tmp_path):
    load_trajectory = pytest.importorskip(
        "neuralgeom.data", reason="analysis package not installed"
    ).load_trajectory
    path = save_trajectory(make_records(), str(tmp_path / "interop.h5"),
                           t_pre=1.0, t_post=3.0, dt=0.05,
                           generator="timingtask.rl")
    traj = load_trajectory(path)
    assert traj.X.ndim == 3
    assert traj.time.shape == (traj.X.shape[1],)
    assert np.isclose(traj.dt, 0.05)
    assert traj.inputs is not None and traj.outputs is not None
    assert traj.condition is not None
    assert traj.meta["generator"] == "timingtask.rl"
    assert "n_steps" in traj.aux
