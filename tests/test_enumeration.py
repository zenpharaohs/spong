"""Phase-1 gates: certified critical-point enumeration.

Oracles: the two models whose critical points were established in the
mse-bundle (MATLAB) sessions, plus self-consistency gates on random zoos
(alternation EXACT; float sign-scan cross-check; N(0,1) heavy tails with no
overflow path).
"""

from fractions import Fraction as F

import numpy as np
import pytest

from spong import model, sturm


def scan_sign_changes(m, lo, hi, n=200001):
    """Independent float cross-check: sign changes of u' ~ B*N on a grid."""
    b = np.linspace(lo, hi, n)
    s = np.sign(m.B(b) * m.Nval(b))
    s = s[s != 0]
    return int(np.sum(s[:-1] != s[1:]))


# --------------------------------------------------------------------- #
# Oracle 1: d = 2, f = g = [1,1,1], U(0,1)   (MATLAB: 4 critical points) #
# --------------------------------------------------------------------- #

def test_oracle_d2():
    m = model.build([1, 1, 1], [1, 1, 1], model.moments_uniform01(5))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse and e.alternates

    # MATLAB found (b, type): (-9.44513, saddle), (-1.63767, min),
    # (-0.51720, saddle), (+1, min with a = 1 exactly)
    got = [(p.b, p.kind) for p in e.points]
    expect = [(-9.44513, "saddle"), (-1.63767, "min"),
              (-0.51720, "saddle"), (1.0, "min")]
    assert len(got) == len(expect)
    for (b, k), (be, ke) in zip(got, expect):
        assert k == ke
        assert b == pytest.approx(be, abs=2e-5)

    # the perfect-fit minimum is an EXACT rational root: f = g forces
    # N(1) = 0 identically in rational arithmetic
    from spong import _poly as P
    assert P.eval_at(m.N, F(1)) == 0
    last = e.points[-1]
    assert last.interval.lo <= F(1) <= last.interval.hi
    assert last.a == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------- #
# Oracle 2: the tricky d = 11 model (kappa = 8.5e8)                      #
# --------------------------------------------------------------------- #

TRICKY_F = [1.12873645202828, -0.289963040800028, 1.26155071814115,
            0.475424811707271, 1.17411675149371, 0.126947068043646,
            -0.656815928948082, -1.48139907157878, 0.155488995903894,
            0.818551368521001, -0.292588130834394, -0.540786416488526]


def test_oracle_tricky():
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse and e.alternates

    bs = {p.kind: [] for p in e.points}
    for p in e.points:
        bs.setdefault(p.kind, []).append(p.b)

    # MATLAB-established points (audit_tricky_branch):
    tricky_saddle = -2.738230515199397
    tricky_min = -0.7895860210707522
    assert any(abs(b - tricky_saddle) < 1e-9 for b in bs["saddle"])
    assert any(abs(b - tricky_min) < 1e-9 for b in bs["min"])
    # perfect-fit global minimum at (1, 1), exact
    assert any(abs(b - 1.0) < 1e-12 for b in bs["min"])

    # independent float cross-check of the count in the core zone
    lo, hi = -12.0, 16.0
    inside = [p for p in e.points if lo < p.b <= hi]
    assert scan_sign_changes(m, lo, hi) == len(inside)


# --------------------------------------------------------------------- #
# Gate: alternation + cross-checks on random zoos                        #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(40))
def test_zoo_uniform_alternation(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 7))
    f = rng.standard_normal(d + 1)
    g = rng.standard_normal(d + 1)
    m = model.build(f, g, model.moments_uniform01(2 * d + 1))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive
    if e.morse:
        assert e.alternates
        lo, hi = -50.0, 50.0
        inside = [p for p in e.points if lo < p.b <= hi]
        assert scan_sign_changes(m, lo, hi) == len(inside)


@pytest.mark.parametrize("seed", range(8))
def test_zoo_normal_d13(seed):
    """The heavy-tailed zoo that produced overflow-NaN in MATLAB: the exact
    layer has no overflow path, far roots included."""
    rng = np.random.default_rng(seed)
    d = 13
    f = rng.standard_normal(d - 1)
    m = model.build(f, f, model.moments_normal01(2 * d + 1))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive
    for p in e.points:
        assert np.isfinite(p.b) and np.isfinite(p.a)
        assert p.kind in ("min", "saddle", "degenerate")
    if e.morse:
        assert e.alternates
        # f = g: perfect-fit min at b = 1 exactly
        assert any(p.interval.lo <= F(1) <= p.interval.hi
                   and p.kind == "min" for p in e.points)


def test_far_root_classification():
    """Deterministic far-root gate through the full enumeration path.

    A tiny leading activation coefficient puts critical values at the
    coefficient-ratio scale (~1e8 here) — the regime whose polyval overflow
    produced NaN candidates in the MATLAB predecessor.  The exact layer has
    no overflow path; far roots are found, classified, and alternate.
    """
    m = model.build([1, 1, 1, F(1, 10**8)], [1, 1, 1, F(1, 10**8)],
                    model.moments_normal01(9))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse and e.alternates
    assert len(e.points) == 6
    far = sorted(p.b for p in e.points if abs(p.b) > 1e6)
    assert len(far) == 2
    assert far[0] == pytest.approx(-1.3333332641666678e8, rel=1e-9)
    assert far[1] == pytest.approx(1.4999998327777209e7, rel=1e-9)
    for p in e.points:
        assert np.isfinite(p.b) and p.kind in ("min", "saddle")


def test_complex_common_factor_does_not_make_real_loss_nonmorse():
    """Morse is a condition at real critical points, not complex roots.

    For X~N(0,1), f=1 and g=1+2x+2x² give
      A=(1+2b²)(1+6b²), B=1+2b².
    The common factor has no real root and cancels from B²/A.  The reduced
    u' numerator is proportional to b, so the sole real critical point is a
    nondegenerate minimum.
    """
    m = model.build([1], [1, 2, 2], model.moments_normal01(7))
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse and e.alternates
    assert len(e.points) == 1
    q = e.points[0]
    assert q.source == "H"
    assert q.b == pytest.approx(0.0)
    assert q.kind == "min" and q.u2_sign > 0
    assert q.local.spectral.determinant > 0


@pytest.mark.slow
def test_zoo_uniform_alternation_1000():
    """Founding Phase-1 gate at full width: alternation on 1000 random
    models (EXACT classification).  Run with `-m slow`."""
    bad = []
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        d = int(rng.integers(2, 7))
        f = rng.standard_normal(d + 1)
        g = rng.standard_normal(d + 1)
        m = model.build(f, g, model.moments_uniform01(2 * d + 1))
        e = sturm.enumerate_critical_points(m)
        if not e.psi_positive or (e.morse and not e.alternates):
            bad.append(seed)
    assert not bad, f"alternation/psi failures at seeds {bad[:10]}"
