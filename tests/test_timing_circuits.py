"""Tests for the published candidate circuits.

Every number here is re-derived from the released connectivity matrices, not
copied from the papers -- the published Methods contain no equations.
"""
import numpy as np
import pytest
from timingtask import circuits as C


def test_verify_passes():
    msgs = C.verify()
    assert any("line attractor" in m for m in msgs)


def test_accepted_model_is_a_perfect_integrator():
    """Exactly one zero Jacobian eigenvalue -- a 1-D line attractor. Not a point
    attractor, not a saddle."""
    s = C.classify("data")
    assert s.n_zero == 1
    ev = np.sort(np.abs(s.eig_J))
    assert ev[0] < 1e-9 and ev[1] > 0.2


def test_integrated_variable_is_the_striatal_difference_mode():
    s = C.classify("data")
    assert abs(abs(s.left[2]) - 0.7071) < 1e-3
    assert abs(abs(s.left[3]) - 0.7071) < 1e-3
    assert np.sign(s.left[2]) != np.sign(s.left[3]), "a DIFFERENCE mode"
    assert s.str_weight > 0.99 and s.alm_weight < 0.05


def test_alm_is_off_manifold_and_striatum_is_on():
    """The paper's core claim, checkable in three lines: the ALM common drive
    contributes nothing to the time representation while striatal input is
    integrated. This asymmetry is why one protocol pauses and the other rewinds."""
    s = C.classify("data")
    assert abs(s.overlaps["ALM silencing"]) < 0.05
    assert abs(s.overlaps["STR inhibition"]) > 0.5


def test_dynamical_classes_of_every_model():
    expected = {
        "externally_driven": (0, 0),                      # no slow mode
        "two_region_integrator": (2, 0),                  # plane attractor
        "two_integrator": (2, 0),                         # two line attractors
        "one_integrator_follower_opposite": (1, 0),       # ALM line attractor
        "one_integrator_follower_opposite_leaky": (0, 1),  # leaky
        "data": (1, 0),                                   # striatal line attractor
    }
    for name, (n_zero, n_near) in expected.items():
        s = C.classify(name)
        assert (s.n_zero, s.n_near_zero) == (n_zero, n_near), (name, s.classification)


def test_alm_integrator_model_cannot_be_touched_by_striatum():
    """ED1d's left eigenvector has EXACTLY zero striatal weight, so striatal
    inhibition cannot affect the timer. That null prediction is what the data
    falsify."""
    s = C.classify("one_integrator_follower_opposite")
    assert abs(s.left[2]) < 1e-9 and abs(s.left[3]) < 1e-9
    assert abs(s.overlaps["STR inhibition"]) < 1e-9


def test_leaky_model_has_a_one_second_leak():
    s = C.classify("one_integrator_follower_opposite_leaky")
    assert s.n_zero == 0
    assert s.leak_tau == pytest.approx(1.0, abs=0.01)


def test_control_lick_times_reproduce_the_released_model():
    """Ramp slope is set by the tonic input amplitude, and the five conditions
    span 0.9-2.1 s."""
    p = C.perturbation_signature("data")
    lt = np.array(p["control_lick_times"])
    assert np.allclose(lt, [0.901, 1.142, 1.455, 1.735, 2.078], atol=0.01)
    assert np.all(np.diff(lt) > 0), "weaker input -> later lick"


def test_ramp_slope_is_exactly_linear_in_the_input():
    """A perfect integrator must have slope proportional to input amplitude.
    This is the property a trained network either has or does not."""
    c = C.get("data")
    ratios = []
    for gain in c.trial_gains:
        r = C.simulate(c, gain)
        seg = r[2, C.CUE_IDX:C.CUE_IDX + 400]
        slope = (seg[-1] - seg[0]) / (400 * C.DT)
        ratios.append(slope / (c.input_vector[0] * gain / C.CUE_IDX))
    assert np.std(ratios) / np.mean(ratios) < 0.01, ratios


def test_the_accepted_model_pauses_then_rewinds():
    """The discriminating signature. ALM silencing shifts the lick time by about
    the silencing duration regardless of trial length (state-independent);
    striatal inhibition sets the state back by a constant amplitude, so the
    shift grows with trial duration (state-dependent)."""
    p = C.perturbation_signature("data")
    assert "PAUSE" in p["alm"]["signature"]
    assert abs(p["alm"]["slope"]) < 0.15
    assert 0.4 < p["alm"]["mean_shift"] < 0.9
    assert "REWIND" in p["str"]["signature"]
    assert p["str"]["slope"] > 0.3


def test_no_other_model_produces_that_pair():
    """The paper's argument, reproduced from the matrices."""
    ok = []
    for name in C.MODELS:
        p = C.perturbation_signature(name)
        if "PAUSE" in p["alm"]["signature"] and "REWIND" in p["str"]["signature"]:
            ok.append(name)
    assert ok == ["data"], ok


def test_externally_driven_model_snaps_back():
    """No slow mode at all: a perturbation leaves no trace once it ends."""
    s = C.classify("externally_driven")
    assert s.n_zero == 0 and s.n_near_zero == 0
    assert "NO SLOW MODE" in s.classification


def test_simulate_shapes_and_rectification():
    r = C.simulate(C.get("data"), 14)
    assert r.shape == (4, C.N_STEPS)
    assert r.min() >= 0.0 and r.max() <= C.RMAX


def test_panel_letters_resolve():
    assert C.get("f") is C.get("data")
    assert C.get("a").name == "externally_driven"


# --------------------------------------------------- Majumder two-attractor
def test_two_attractor_timing_is_set_by_the_cue_kick():
    """The flow field is identical on every trial; a larger kick licks earlier.
    No tonic drive and no trial-history term anywhere in that model."""
    lts = [C.two_attractor_trajectory(a, np.pi / 4)[2]
           for a in (150, 127.5, 105, 82.5)]
    fin = [x for x in lts if np.isfinite(x)]
    assert len(fin) >= 3
    assert all(b > a for a, b in zip(fin, fin[1:])), lts


def test_two_attractor_field_has_a_separatrix_between_the_basins():
    """Two basins, so the flow reverses somewhere between them. It does, near
    the midpoint: at (3,3) the field points back to baseline, at (12,12) on to
    the lick state."""
    u_lo, v_lo = C.two_attractor_field(3.0, 3.0)
    u_hi, v_hi = C.two_attractor_field(12.0, 12.0)
    assert u_lo < 0 and v_lo < 0, "inside the baseline basin, flow returns"
    assert u_hi > 0 and v_hi > 0, "inside the lick basin, flow advances"


def test_the_field_is_normalised_not_a_gradient_flow():
    """Implementation detail that changes the interpretation: the field is
    divided by its own magnitude before the anisotropic gains (30, 10) are
    applied. The intrinsic slowness between basins is therefore largely removed
    and the timing comes from geometry -- path direction and length -- so this
    is a discrete two-attractor model, not a shallow-basin one."""
    for pt in [(3.0, 3.0), (8.0, 8.0), (12.0, 12.0), (14.0, 14.0)]:
        u, v = C.two_attractor_field(*pt)
        assert np.hypot(u / 30.0, v / 10.0) == pytest.approx(1.0, abs=1e-3), pt
