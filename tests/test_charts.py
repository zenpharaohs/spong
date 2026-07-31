"""Phase-3 gates (SPONG_FOUNDING Part IV): the tricky branch at E ≤ 1e-12,
both separatrices ≤ 1e-11 with the √d asymptote, seam agreement, and clean
fold handling on a high-degree zoo."""

import numpy as np
import pytest

from spong import charts, model, sturm
from tests.test_enumeration import TRICKY_F


def test_closed_form_sym2_eigh_recovers_small_value_from_determinant():
    lam, frame = charts._sym2_eigh(np.diag((-1e-8, 1e8)))
    np.testing.assert_allclose(lam, (-1e-8, 1e8), rtol=2e-16)
    np.testing.assert_allclose(frame, np.eye(2), rtol=0, atol=0)


def test_native_hadamard_transform_matches_python_oracle():
    """The production fourth-order graph transform is native; Python pins it."""
    m = model.build([1.0, 1.0, 0.5], [1.0, 1.0, 0.5],
                    model.moments_uniform01(15))
    if m._native_kernel is None:
        pytest.skip("native kernel not built")
    b = np.linspace(4.0, 8.0, 1001)
    py_w, py_it, py_rel = charts._slow_fixed_point_python(m, b)
    c_w, c_it, c_rel = m._native_kernel.slow_fixed_point(b, 1e-13, 40)
    c_w = np.asarray(c_w)
    assert c_it == py_it
    assert np.allclose(c_w, py_w, rtol=5e-14, atol=1e-25)
    assert abs(c_rel - py_rel) < 1e-15


def test_native_curve_diagnostics_match_python_oracles():
    m = model.build([1.0, -0.5, 0.25], [0.75, 1.0, -0.2],
                    model.moments_uniform01(9))
    if m._native_kernel is None or not hasattr(
            m._native_kernel, "curve_diagnostics"):
        pytest.skip("native curve diagnostics not built")
    b = np.linspace(-2.0, 3.0, 2003)
    a = np.asarray(m.a_star(b))+1e-5*np.sin(17.0*b)
    curve = np.ascontiguousarray(np.column_stack((a, b)))
    for digits, start in ((1e3, 1), (1e6, 37), (1e12, 113)):
        expected_angle = charts._angle_energy_detail_python(
            m, curve, digits=digits, start=start)
        expected_backbone = charts._backbone_residual_python(
            m, curve, digits=digits, start=start)
        actual = m._native_kernel.curve_diagnostics(
            curve, digits, start)
        assert actual[1:3] == expected_angle[1:3]
        assert actual[0] == pytest.approx(
            expected_angle[0], rel=2e-13, abs=1e-28)
        assert actual[3] == pytest.approx(
            expected_backbone, rel=2e-13, abs=1e-28)


def _lowest_saddle_and_adjacent_min(e):
    saddles = sorted(e.saddles, key=lambda p: p.b)
    s = saddles[0]
    ups = sorted((p for p in e.minima if p.b > s.b), key=lambda p: p.b)
    return s, ups[0]


# ------------------------ gate: tricky unstable ------------------------ #

def test_tricky_unstable_branch(tricky):
    m, e = tricky
    s, t = _lowest_saddle_and_adjacent_min(e)
    assert s.b == pytest.approx(-2.738230515199397, abs=1e-9)
    assert t.b == pytest.approx(-0.7895860210707522, abs=1e-9)

    br = charts.trace_unstable(m, s.b, (t.a, t.b))
    assert br.term == "capture"
    assert br.certs["angle_energy"] <= 1e-12          # MATLAB got 5.3e-13
    assert br.diag["kappa_saddle"] > 1e8              # the stiffness is real
    assert br.diag["stiff_frac"] > 0.3                # shallow water is real
    assert "seam_residual" in br.certs
    assert br.certs["seam_residual"] < 1e-8
    # endpoint is the exact minimum (appended on capture)
    assert br.Y[-1, 0] == pytest.approx(t.a, abs=1e-12)
    assert br.Y[-1, 1] == pytest.approx(t.b, abs=1e-12)
    # loss decreases monotonically along the branch (descent)
    L = np.array([m.L(a, b) for a, b in br.Y])
    assert np.all(np.diff(L) < 1e-10)


# ---------------------- gate: tricky separatrices ---------------------- #

@pytest.mark.parametrize("sign", [+1, -1])
def test_tricky_separatrix(tricky, sign):
    m, e = tricky
    s, _ = _lowest_saddle_and_adjacent_min(e)
    box = (-25.0, 25.0, -12.0, 16.0)
    br = charts.trace_stable(m, s.b, sign, box=box)
    assert br.term == "box_exit"
    assert br.certs["angle_energy"] <= 1e-11          # MATLAB got 1.25e-12
    # ascent: loss increases along the separatrix
    L = np.array([m.L(a, b) for a, b in br.Y])
    assert np.all(np.diff(L) > -1e-10)
    # asymptote certificate: exit direction approaches b = ±sqrt(d)·a
    a_e, b_e = br.Y[-1]
    d = len(TRICKY_F) - 1                              # 11
    assert abs(b_e / a_e) == pytest.approx(np.sqrt(d), rel=0.05)


# ------------------- gate: d=2 oracle, whole skeleton ------------------- #

def test_d2_all_unstable_branches():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    box = (-10.0, 10.0, -12.0, 8.0)
    for s in e.saddles:
        for direction in (+1, -1):
            side = [p for p in e.minima
                    if (p.b > s.b if direction > 0 else p.b < s.b)]
            if side:
                t = min(side, key=lambda p: abs(p.b - s.b))
                br = charts.trace_unstable(m, s.b, (t.a, t.b), box=box)
                assert br.term == "capture", (s.b, direction, br.term)
                assert br.certs["angle_energy"] < 1e-9
            else:
                # unbounded branch: pseudo-target on the backbone at the box
                b_exit = box[3] if direction > 0 else box[2]
                a_exit = float(m.a_star(b_exit))
                br = charts.trace_unstable(m, s.b, (a_exit, b_exit), box=box)
                assert br.term in ("capture", "box_exit")
                assert br.certs["angle_energy"] < 1e-9


def test_d2_random_stiff_down_branch_capture():
    """Seed 2735729614: a quadratic f=g branch hit the HI threshold between
    grid nodes, causing engine/shallow ping-pong and abort_zone_limit."""
    f = [-0.27126925828072923, -0.7363598557165663, 0.7989868625933855]
    m = model.build(f, f, model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    s = sorted(e.saddles, key=lambda p: p.b)[0]
    t = [p for p in e.minima if p.b < s.b][0]
    br = charts.trace_unstable(
        m, s.b, (t.a, t.b),
        box=(-0.537726716445422, 1.998796912946715,
             -5.148979319205286, 3.1997597252090433))
    assert br.term == "capture"
    assert br.certs["angle_energy"] < 1e-9
    assert br.certs["seam_residual"] < 1e-9
    assert br.Y[-1, 0] == pytest.approx(t.a, abs=1e-12)
    assert br.Y[-1, 1] == pytest.approx(t.b, abs=1e-12)


# ------------------ gate: fold handling, high degree ------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_d17_zoo_clean_terminations(seed):
    """Every bounded unstable branch of a d=17 model terminates cleanly
    (capture or box exit) — folds are handed between charts, not fatal."""
    rng = np.random.default_rng(seed)
    d = 17
    f = rng.standard_normal(d + 1)
    g = rng.standard_normal(d + 1)
    m = model.build(f, g, model.moments_uniform01(2 * d + 1))
    e = sturm.enumerate_critical_points(m)
    if not e.morse:
        pytest.skip("non-Morse draw")
    box = (-60.0, 60.0, -60.0, 60.0)
    n_traced = 0
    for s in e.saddles:
        for direction in (+1, -1):
            side = [p for p in e.minima
                    if (p.b > s.b if direction > 0 else p.b < s.b)]
            if not side:
                continue
            t = min(side, key=lambda p: abs(p.b - s.b))
            br = charts.trace_unstable(m, s.b, (t.a, t.b), box=box,
                                       ds=abs(t.b - s.b) / 2000.0)
            assert br.term in ("capture", "box_exit"), \
                (seed, s.b, direction, br.term)
            assert np.all(np.isfinite(br.Y))
            n_traced += 1
    assert n_traced > 0
