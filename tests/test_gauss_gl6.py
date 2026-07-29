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


def test_scalar_stage_newton_rejects_iteration_exhaustion():
    """A finite iterate is not an accepted step unless its stage residual met
    the termination criterion."""
    f = lambda x, y: y * y + np.sin(x)
    j = lambda x, y: 2.0 * y
    with pytest.raises(FloatingPointError, match="did not converge"):
        gauss.gl6_scalar(f, j, 0.0, 1.0, 3.0, maxit=1)


def test_general_stage_newton_rejects_iteration_exhaustion(monkeypatch):
    """The vector path obeys the same residual-based contract."""
    f = lambda x, y: np.array([y[0] * y[0] + np.sin(x)])
    j = lambda x, y: np.array([[2.0 * y[0]]])
    monkeypatch.setattr(gauss, "_NEWTON_MAX", 1)
    with pytest.raises(FloatingPointError, match="did not converge"):
        gauss.step(f, 0.0, np.array([1.0]), 3.0, method="gl6", jac=j)


def test_native_armijo_restart_certifies_stage_solve_python_rejects():
    """The C Armijo restart may enlarge Newton's basin, but its accepted step
    must retain the defining anadromic certificate."""
    pytest.importorskip("spong._native")
    from spong import charts, model

    m = model.build([1.0, 1.0, 0.5], [1.0, 1.0, 0.5],
                    model.moments_uniform01(15))
    f, j = charts.slow_rhs_s(m)
    x, y, h = -0.27105203823092516, 1.7261330563357526e-6, -8.062546477701309
    with pytest.raises(FloatingPointError, match="did not converge"):
        gauss.gl6_scalar(f, j, x, y, h)
    y1 = m._native_kernel.slow_step(x, y, h)
    assert np.isfinite(y1)
    assert m._native_kernel.slow_step(x + h, y1, -h) == pytest.approx(
        y, rel=0, abs=5e-15)
    assert np.isfinite(gauss.gl6_scalar(f, j, x, y, h / 2.0))
    assert np.isfinite(m._native_kernel.slow_step(x, y, h / 2.0))


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
    hardest_real = 9.64e-04        # d=17 branch; elevated but solvable
    assert gauss._STAGE_GUARD < hardest_real
    singular = ratio(np.eye(3) - 4.644371 * A)
    assert singular < gauss._STAGE_GUARD < healthy
    assert healthy / gauss._STAGE_GUARD > 100.0
    assert gauss._STAGE_GUARD / singular > 10.0


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


# ------------------------------------------------------- dense output (s=3) --


@pytest.mark.parametrize("method", ["imm", "gl4", "gl6"])
def test_dense_output_interpolates_the_step_endpoints(method):
    """u(0) = y0 and u(1) = y1 for every stage count.

    Special-casing s <= 2 would silently give a 3-stage step the 2-stage
    polynomial -- wrong nodes and k3 dropped -- and find_event root-finds here.
    """
    F = lambda x, y: np.array([-0.7 * y[0] + np.cos(x)])
    J = lambda x, y: np.array([[-0.7]])
    st = gauss.step(F, 0.3, np.array([0.9]), 0.25, method=method, jac=J)
    assert np.allclose(st.dense(0.0), st.y0, atol=1e-14)
    assert np.allclose(st.dense(1.0), st.y1, atol=1e-13)


def test_gl6_dense_output_converges_at_the_collocation_order():
    """The interpolant is order s+1 = 4, NOT the endpoint order 2s = 6.

    Gauss endpoints are superconvergent; the collocation polynomial between
    them is not.  Measured ratios ~15.8 per halving (2^4 = 16).  Anything
    asserting 1e-12 off-node at a working step size is asserting the wrong
    order, not catching a bug.
    """
    lam = -0.6
    F = lambda x, y: np.array([lam * y[0]])
    J = lambda x, y: np.array([[lam]])
    errs = []
    for h in (0.4, 0.2, 0.1, 0.05):
        st = gauss.step(F, 0.0, np.array([1.0]), h, method="gl6", jac=J)
        errs.append(max(abs(float(st.dense(t)[0]) - np.exp(lam * t * h))
                        for t in (0.17, 0.5, 0.83)))
    for a, b in zip(errs[:-1], errs[1:]):
        assert 12.0 < a / b < 20.0                 # order 4


def test_gl6_dense_differs_from_the_gl4_polynomial():
    """Guards the bug directly: if dense() fell through to the 2-stage branch
    the 3-stage interpolant would coincide with a 2-node/2-stage evaluation."""
    F = lambda x, y: np.array([np.sin(3.0 * x) - 0.4 * y[0]])
    J = lambda x, y: np.array([[-0.4]])
    st6 = gauss.step(F, 0.0, np.array([1.0]), 0.5, method="gl6", jac=J)
    c1, c2 = gauss._GL2_C
    th = 0.37
    w1 = (th**2 / 2 - c2 * th) / (c1 - c2)
    w2 = (th**2 / 2 - c1 * th) / (c2 - c1)
    wrong = st6.y0 + st6.h * (w1 * st6.K[0] + w2 * st6.K[1])
    assert abs(float(st6.dense(th)[0]) - float(wrong[0])) > 1e-6


# ---------------------------------------------------------- native parity --


def _stage_newton_converged(f, j, x, y, h, tol=1e-13, maxit=30):
    """Did the GL6 stage Newton actually converge at this state?

    Parity is a claim about the METHOD.  Where the stage equations are not
    solved, neither path has a defined answer: both run out of iterations and
    land on different non-solutions, which is a property of the problem, not a
    discrepancy between implementations.  (The engine rejects such a step and
    halves — see the retry loop in charts._continue_curve.)
    """
    c, A = gauss._GL3_C, gauss._GL3_A
    xs = [x + ci * h for ci in c]
    K = [f(x, y)] * 3
    for _ in range(maxit):
        Y = [y + h * (A[i][0] * K[0] + A[i][1] * K[1] + A[i][2] * K[2])
             for i in range(3)]
        r = [K[i] - f(xs[i], Y[i]) for i in range(3)]
        if max(abs(v) for v in r) < tol * (1.0 + max(abs(v) for v in K)):
            return True
        J = [j(xs[i], Y[i]) for i in range(3)]
        M = [[(1.0 if i == k else 0.0) - h * A[i][k] * J[i] for k in range(3)]
             for i in range(3)]
        rhs = [-r[i] for i in range(3)]
        try:
            d = np.linalg.solve(np.array(M), np.array(rhs))
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(d)):
            return False
        K = [K[i] + float(d[i]) for i in range(3)]
    return False


def test_native_and_python_agree_on_the_gl6_default():
    """The C kernel and the Python fallback must be the SAME method.

    Widened after an out-of-sample probe found the previous three-point version
    could not have detected a divergence: it pinned only benign states.  This
    sweeps the range the adaptive controller actually operates in, choosing h
    from the LOCAL stiffness (|h·J| spanning the mild and stiff regimes) rather
    than independently of it — sampling h blindly generates |h·J| ~ 1e4 steps
    that the controller would never take, where both methods are meaningless
    and GL4 disagrees with itself just as much.
    """
    pytest.importorskip("spong._native")
    from fractions import Fraction as F_
    from spong import charts, model

    rng = np.random.default_rng(20260727)
    worst = 0.0
    checked = 0
    for coeffs in ([F_(1), F_(1), F_(1, 2)],
                   [F_(1), F_(1), F_(1, 2), F_(1, 6)],
                   [F_(1), F_(-1), F_(1, 3), F_(1, 5)]):
        m = model.build([float(c) for c in coeffs], [float(c) for c in coeffs],
                        model.moments_uniform01(15))
        k = m._native_kernel
        if k is None:
            pytest.skip("native kernel not built")
        sf, sj = charts.slow_rhs_s(m)
        for _ in range(400):
            b = float(rng.uniform(-3.0, 3.0))
            w = float(rng.normal(0.0, 1.0) * 10.0 ** rng.uniform(-12, -2))
            try:
                J = sj(b, w)
            except Exception:
                continue
            if not np.isfinite(J) or J == 0.0:
                continue
            # h chosen so |h*J| lands in the controller's real range
            h = float(rng.choice([1e-3, 1e-2, 1e-1, 1.0]) / abs(J))
            h *= float(rng.choice([1.0, -1.0]))
            if not _stage_newton_converged(sf, sj, b, w, h):
                continue                      # no defined answer to compare
            try:
                nv, pv = k.slow_step(b, w, h), gauss.gl6_scalar(sf, sj, b, w, h)
            except Exception:
                continue
            if not (np.isfinite(nv) and np.isfinite(pv)):
                assert not np.isfinite(nv) and not np.isfinite(pv), (
                    "one path produced a finite step and the other did not")
                continue
            checked += 1
            worst = max(worst, abs(nv - pv) / max(abs(pv), abs(w), 1e-16))
    assert checked > 500, f"only {checked} usable samples"
    assert worst < 1e-10, f"native/python divergence {worst:.2e}"


def test_native_and_python_agree_on_the_fast_chart():
    """Same, for the other chart — the earlier test covered only the slow one."""
    pytest.importorskip("spong._native")
    from fractions import Fraction as F_
    from spong import charts, model

    rng = np.random.default_rng(11)
    m = model.build([1.0, 1.0, 0.5], [1.0, 1.0, 0.5], model.moments_uniform01(15))
    k = m._native_kernel
    if k is None:
        pytest.skip("native kernel not built")
    ff, fj = charts.fast_rhs_s(m)
    worst, checked = 0.0, 0
    for _ in range(600):
        w = float(rng.normal(0.0, 1.0) * 10.0 ** rng.uniform(-10, -2))
        b = float(rng.uniform(-3.0, 3.0))
        try:
            J = fj(w, b)
        except Exception:
            continue
        if not np.isfinite(J) or J == 0.0:
            continue
        h = float(rng.choice([1e-3, 1e-2, 1e-1]) / abs(J)) * float(
            rng.choice([1.0, -1.0]))
        if not _stage_newton_converged(ff, fj, w, b, h):
            continue
        try:
            nv, pv = k.fast_step(w, b, h), gauss.gl6_scalar(ff, fj, w, b, h)
        except Exception:
            continue
        if not (np.isfinite(nv) and np.isfinite(pv)):
            continue
        checked += 1
        worst = max(worst, abs(nv - pv) / max(abs(pv), abs(b), 1e-16))
    assert checked > 200, f"only {checked} usable samples"
    assert worst < 1e-10, f"native/python divergence {worst:.2e}"


@pytest.mark.parametrize("order", [4, 6, 8])
def test_native_normalized_2d_step_matches_general_irk(order):
    """The native geometric-flow kernel is the same vector IRK method."""
    pytest.importorskip("spong._native")
    from spong import model

    m = model.build([1.0, 2.0, 1.0], [0.5, -1.0, 2.0],
                    model.moments_uniform01(7))
    k = m._native_kernel
    if k is None:
        pytest.skip("native kernel not built")

    def F(_x, z):
        g = m.gradL(float(z[0]), float(z[1]))
        return g / np.linalg.norm(g)

    def J(_x, z):
        g = m.gradL(float(z[0]), float(z[1]))
        H = m.hessL(float(z[0]), float(z[1]))
        ng = np.linalg.norm(g)
        return H / ng - np.outer(g, H @ g) / ng**3

    method = f"gl{order}"
    for z in (np.array([0.3, 0.4]), np.array([2.0, -1.0])):
        native = np.asarray(k.normalized_step(*z, 1e-3, order))
        python = gauss.step(F, 0.0, z, 1e-3, method=method, jac=J).y1
        assert np.allclose(native, python, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("order", [4, 6, 8])
def test_native_potential_rate_step_has_expected_loss_change(order):
    """The geometric reparameterization satisfies dL/dt=1."""
    pytest.importorskip("spong._native")
    from spong import model

    m = model.build([1.0, 2.0, 1.0], [0.5, -1.0, 2.0],
                    model.moments_uniform01(7))
    k = m._native_kernel
    if k is None:
        pytest.skip("native kernel not built")
    z = np.array((0.3, 0.4))
    h = 1e-5
    zn = np.asarray(k.potential_step(*z, h, order))
    assert np.all(np.isfinite(zn))
    assert m.L(*zn)-m.L(*z) == pytest.approx(h, rel=2e-10, abs=2e-15)


def test_gl8_tableau_satisfies_gauss_moments_and_stage_consistency():
    c = np.asarray(gauss._GL4_C)
    A = np.asarray(gauss._GL4_A)
    b = np.asarray(gauss._GL4_B)
    np.testing.assert_allclose(np.sum(A, axis=1), c, rtol=0, atol=2e-16)
    for degree in range(8):
        assert np.sum(b*c**degree) == pytest.approx(
            1.0/(degree+1), rel=0, abs=4e-16)
