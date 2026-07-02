"""Phase-4 gates (SPONG_FOUNDING Part IV): √d asymptote convergence, index
balance across the zoo, backbone-pole model matches the C_inf analysis."""

from fractions import Fraction as F

import numpy as np
import pytest

from spong import atlas, charts, model, sturm
from tests.test_enumeration import TRICKY_F


@pytest.fixture(scope="module")
def tricky():
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    e = sturm.enumerate_critical_points(m)
    return m, e


@pytest.fixture(scope="module")
def d2():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    return m, e


# ---------------- gate: backbone-pole model (C_inf) ------------------- #

@pytest.mark.parametrize("fix", ["d2", "tricky"])
def test_C_inf_matches_numeric_limit(fix, request):
    m, _ = request.getfixturevalue(fix)
    c = float(atlas.C_inf(m))
    # bdot = -u'(b) ~ -C_inf / b**2  =>  u'(b) * b**2 -> C_inf
    for b in (1e5, 1e6):
        assert m.u_p(b) * b**2 == pytest.approx(c, rel=1e-2)
    # and the approach improves with b (it is a genuine limit)
    e1 = abs(m.u_p(1e4) * 1e8 - c)
    e2 = abs(m.u_p(1e5) * 1e10 - c)
    assert e2 < e1


# ---------------- gate: √d asymptote via richardson3 ------------------ #

def test_asymptote_certificate_tricky(tricky):
    m, e = tricky
    s = min(e.saddles, key=lambda p: p.b)
    br = charts.trace_stable(m, s.b, +1, box=(-40.0, 40.0, -40.0, 40.0))
    assert br.term == "box_exit"
    cert = atlas.asymptote_certificate(m, br.Y)
    assert cert["target"] == pytest.approx(np.sqrt(11.0))
    # raw samples close within ~5%; the Aitken extrapolant much closer
    assert abs(cert["slope_samples"][-1] - cert["target"]) < 0.05 * cert["target"]
    assert cert["residual"] < 0.01


# ---------------- gate: index balance across the zoo ------------------- #

def test_index_balance_oracles(d2, tricky):
    for m, e in (d2, tricky):
        r = atlas.index_balance(m, e)
        assert r["balanced"], r


@pytest.mark.parametrize("seed", range(12))
def test_index_balance_zoo(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 6))
    f = rng.standard_normal(d + 1)
    g = rng.standard_normal(d + 1)
    m = model.build(f, g, model.moments_uniform01(2 * d + 1))
    e = sturm.enumerate_critical_points(m)
    if not e.morse:
        pytest.skip("non-Morse draw")
    r = atlas.index_balance(m, e)
    assert r["balanced"], (seed, r)


def test_index_balance_far_root_model():
    """The pinned far-root model: the balance circle must enclose b ~ 1e8."""
    m = model.build([1, 1, 1, F(1, 10**8)], [1, 1, 1, F(1, 10**8)],
                    model.moments_normal01(9))
    e = sturm.enumerate_critical_points(m)
    r = atlas.index_balance(m, e)
    assert r["b_bound"] > 1e8
    assert r["balanced"], r


# ---------------- genericity and degree drops -------------------------- #

def test_effective_degree_drop():
    # nominal degree 5 with three zero leading coefficients: d_eff = 2
    g = [1, 1, 1, 0, 0, 0]
    m = model.build([1, 1, 1], g, model.moments_uniform01(11))
    assert atlas.effective_degree(m) == 2
    gen = atlas.genericity(m)
    assert gen["generic"] and gen["d_eff"] == 2
    assert atlas.rim_directions(m)["diagonal_slopes"][0] == pytest.approx(
        np.sqrt(2.0))


# ---------------- box contract ----------------------------------------- #

def test_compute_box_contract(d2):
    m, e = d2
    view = (-2.0, 2.0, -3.0, 3.0)
    box = atlas.compute_box(m, e, view=view)
    # view strictly inside compute box
    assert box[0] < view[0] and box[1] > view[1]
    assert box[2] < view[2] and box[3] > view[3]
    # all nondegenerate critical points inside
    for p in e.points:
        if p.kind != "degenerate":
            assert box[0] <= p.a <= box[1] and box[2] <= p.b <= box[3]
    # legal max respected
    assert box[3] <= atlas.legal_max_b(m) and box[2] >= -atlas.legal_max_b(m)
