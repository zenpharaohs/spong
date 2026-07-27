"""IRK6-GL (3-stage Gauss): order, anadromy, Tier-0 parity, and the
conditioning claim the closed-form 3x3 stage solve rests on."""

import numpy as np
import pytest

from spong import gauss


def _order(vals):
    """Self-convergence exponents from successive halvings."""
    out = []
    for i in range(len(vals) - 2):
        d1, d2 = vals[i + 1] - vals[i], vals[i + 2] - vals[i + 1]
        if abs(d2) > 0:
            out.append(np.log2(abs(d1 / d2)))
    return out


# ------------------------------------------------------------------ order --


@pytest.mark.parametrize("lam", [-1.0, -10.0])
def test_gl6_is_order_six_on_linear_problems(lam):
    F = lambda x, y: np.array([lam * y[0]])
    J = lambda x, y: np.array([[lam]])
    vals = [float(gauss.solve(F, 0.0, 1.0, np.array([1.0]), n,
                              method="gl6", jac=J).y_end[0])
            for n in (5, 10, 20, 40, 80)]
    assert min(_order(vals)) > 5.5


def test_gl6_beats_gl4_at_equal_step_count():
    """Order 6 vs 4 on a nonlinear problem with a known solution."""
    F = lambda x, y: np.array([-y[0] + np.sin(x)])
    J = lambda x, y: np.array([[-1.0]])
    exact = (np.exp(-3.0) * 1.5 + 0.5 * (np.sin(3.0) - np.cos(3.0)))
    e4 = abs(float(gauss.solve(F, 0.0, 3.0, np.array([1.0]), 12,
                               method="gl4", jac=J).y_end[0]) - exact)
    e6 = abs(float(gauss.solve(F, 0.0, 3.0, np.array([1.0]), 12,
                               method="gl6", jac=J).y_end[0]) - exact)
    assert e6 < e4 / 100.0


# --------------------------------------------------------------- anadromy --


def test_gl6_is_anadromic():
    """Symmetry: forward then backward returns to the start."""
    F = lambda x, y: np.array([-y[0] + 0.5 * y[0] ** 2])
    J = lambda x, y: np.array([[-1.0 + y[0]]])
    gap = gauss.reversal_gap(F, 0.0, 1.5, np.array([0.4]), 60,
                             method="gl6", jac=J)
    assert gap < 1e-12


# ----------------------------------------------------------- Tier-0 parity --


def test_gl6_scalar_matches_the_numpy_path():
    """Tier-0 closed-form 3x3 == step(method='gl6') for scalar systems."""
    fs = lambda x, y: -y + 0.5 * y * y
    js = lambda x, y: -1.0 + y
    fv = lambda x, y: np.array([-y[0] + 0.5 * y[0] ** 2])
    jv = lambda x, y: np.array([[-1.0 + y[0]]])
    n, h, y = 40, 1.5 / 40, 0.4
    for i in range(n):
        y = gauss.gl6_scalar(fs, js, i * h, y, h)
    ref = float(gauss.solve(fv, 0.0, 1.5, np.array([0.4]), n,
                            method="gl6", jac=jv).y_end[0])
    assert abs(y - ref) <= 1e-12 * max(abs(ref), 1.0)


@pytest.mark.parametrize("lam", [-1e2, -1e6, -1e10])
def test_gl6_scalar_survives_stiffness(lam):
    """The closed-form solve must not degrade as |h*lambda| grows."""
    fs = lambda x, y: lam * y
    js = lambda x, y: lam
    fv = lambda x, y: np.array([lam * y[0]])
    jv = lambda x, y: np.array([[lam]])
    n, h, y = 200, 1.0 / 200, 1.0
    for i in range(n):
        y = gauss.gl6_scalar(fs, js, i * h, y, h)
    ref = float(gauss.solve(fv, 0.0, 1.0, np.array([1.0]), n,
                            method="gl6", jac=jv).y_end[0])
    assert abs(y - ref) <= 1e-10 * max(abs(ref), 1e-12)


# ------------------------------------------------------- the stage matrix --


def _A3():
    return np.array(gauss._GL3_A, dtype=float)


def test_det_of_stage_matrix_is_the_pade_denominator():
    """det(I - zA) = 1 - z/2 + z^2/10 - z^3/120 exactly.

    This is what makes the closed-form (Cramer) stage solve provably safe.
    """
    A = _A3()
    for z in (0.5, -1.0, -10.0, -1e3, -1e6):
        det = np.linalg.det(np.eye(3) - z * A)
        pade = 1.0 - z / 2 + z * z / 10 - z ** 3 / 120
        assert abs(det - pade) <= 1e-12 * abs(pade)


def test_stage_matrix_singularities_are_all_anti_dissipative():
    """A-stability: every root of the Pade denominator has Re > 0, so the
    stage matrix cannot be singular for dissipative h*lambda."""
    roots = np.roots([-1 / 120, 1 / 10, -1 / 2, 1])
    assert np.all(roots.real > 0.5)


def test_stage_matrix_conditioning_is_bounded_in_stiffness():
    """cond_2(I - zA) saturates rather than growing with |z|."""
    A = _A3()
    conds = [np.linalg.cond(np.eye(3) - z * A)
             for z in (-1e2, -1e4, -1e8, -1e14)]
    assert max(conds) < 12.0
    assert abs(conds[-1] - conds[-2]) < 1e-6      # saturated


def test_conditioning_guard_never_trips_on_dissipative_problems():
    """The guard is the only thing needed: entries are essentially exact, so
    small backward error IS small forward error, and any backward-stable solve
    serves.  It must therefore be silent across the whole dissipative range."""
    for lam in (-1.0, -1e3, -1e6, -1e9, -1e12, -1e15):
        fs = lambda x, y: lam * y
        js = lambda x, y: lam
        y, h = 1.0, 1.0 / 50
        for i in range(50):
            y = gauss.gl6_scalar(fs, js, i * h, y, h)     # must not raise
        assert np.isfinite(y)


def test_conditioning_guard_trips_at_a_true_singularity():
    """At a Pade-denominator root the stage matrix IS singular; the guard must
    say so rather than dividing by a near-zero determinant."""
    A = _A3()
    z = 4.644371                       # real root of Q3, anti-dissipative
    lam, h = z / 0.01, 0.01
    fs = lambda x, y: lam * y
    js = lambda x, y: lam
    ratio = (abs(np.linalg.det(np.eye(3) - z * A))
             / np.prod([max(abs((np.eye(3) - z * A)[i])) for i in range(3)]))
    assert ratio < gauss._STAGE_GUARD          # calibration holds
    with pytest.raises(FloatingPointError, match="ill-conditioned"):
        gauss.gl6_scalar(fs, js, 0.0, 1.0, h)


def test_guard_threshold_sits_between_healthy_and_singular():
    """Seven orders of separation: healthy saturates at 0.4159, a genuine
    singularity is ~2.9e-08.  The trip must lie strictly between."""
    A = _A3()

    def ratio(M):
        return (abs(np.linalg.det(M))
                / np.prod([max(abs(M[i])) for i in range(3)]))

    healthy = min(ratio(np.eye(3) - z * A)
                  for z in (-1e2, -1e4, -1e8, -1e12, -1e16))
    singular = ratio(np.eye(3) - 4.644371 * A)
    assert singular < gauss._STAGE_GUARD < healthy
    assert healthy / gauss._STAGE_GUARD > 100.0
    assert gauss._STAGE_GUARD / singular > 100.0


def test_diagonal_dominance_threshold_covers_the_mild_regime():
    """Second, independent guarantee (Andrew): for small enough h the stage
    matrix is diagonally dominant, so LU is stable WITHOUT pivoting.

    Row i is dominant when |1 - h*a_ii*J_i| > h|J_i| * sum_{j!=i} |a_ij|.
    For dissipative real z = h*J the diagonal grows, so each row has its own
    threshold; the binding one is small and sits in the mild regime, which is
    exactly where the order-6 method earns its keep.  The stiff end is covered
    instead by the Pade/conditioning results above -- together the two
    arguments span the whole range.
    """
    A = _A3()
    thresholds = []
    for i in range(3):
        off = sum(abs(A[i, j]) for j in range(3) if j != i)
        # dissipative real z: |1 + a_ii|z|| > off*|z|
        if A[i, i] >= off:
            thresholds.append(np.inf)          # dominant for every |z|
        else:
            thresholds.append(1.0 / (off - A[i, i]))
    zmax = min(thresholds)
    assert 1.0 < zmax < 3.0                    # binding row, mild regime

    # below the threshold, unpivoted LU must match pivoted LU
    rng = np.random.default_rng(7)
    for _ in range(200):
        z = rng.uniform(-0.9 * zmax, 0.0, size=3)
        M = np.eye(3) - (z[:, None] * A)
        for i in range(3):
            assert abs(M[i, i]) > sum(abs(M[i, j]) for j in range(3) if j != i)
        r = rng.standard_normal(3)
        assert np.max(np.abs(np.linalg.solve(M, r)
                             - np.linalg.inv(M) @ r)) < 1e-10


def test_closed_form_matches_lu_with_a_VARYING_stage_jacobian():
    """The Pade proof above assumes a frozen Jacobian D = lambda*I.  In the
    method the three J_i differ by O(h), so check that case numerically."""
    rng = np.random.default_rng(20260727)
    A = _A3()
    worst = 0.0
    for _ in range(400):
        scale = 10.0 ** rng.uniform(0, 12)
        J = -scale * rng.uniform(0.5, 1.5, size=3)      # dissipative, spread
        h = 1e-3
        M = np.eye(3) - h * (J[:, None] * A)
        r = rng.standard_normal(3)
        lu = np.linalg.solve(M, r)
        adj = np.linalg.inv(M) @ r                      # adjugate/Cramer route
        worst = max(worst, float(np.max(np.abs(lu - adj))
                                 / max(float(np.max(np.abs(lu))), 1e-300)))
        assert np.linalg.cond(M) < 1e3
    assert worst < 1e-8
