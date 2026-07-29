import ast
from fractions import Fraction
from pathlib import Path

import numpy as np

from spong import _native, charts, gauss, inverse, model, sturm


def test_production_np_linalg_calls_are_explicitly_allowlisted():
    """Exact model structure must not silently return to generic LAPACK."""
    source_root = Path(__file__).parents[1] / "src" / "spong"
    calls = []
    forbidden_imports = []
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and node.module in {"numpy.linalg", "numpy.linalg.linalg"}):
                forbidden_imports.append((path.name, node.lineno))
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Attribute)
                    and isinstance(fn.value.value, ast.Name)
                    and fn.value.value.id == "np"
                    and fn.value.attr == "linalg"):
                calls.append((path.name, fn.attr))
    assert not forbidden_imports
    assert calls == [("local.py", "svd")]


def test_native_local_kernel_accepts_general_rotated_polynomial():
    # Rotation of a physical jet that is quadratic in ``a`` produces all
    # powers of both normal coordinates.  This degree-four example exercises
    # the storage shape required by the transformed Poincare polynomial.
    grad = (
        ((1.0,), (2.0,), (3.0,), (4.0,), (5.0,)),
        ((-1.0, 2.0), (0.5, -3.0), (2.0,), (-0.25,), (0.125,)),
    )
    kernel = _native.LocalKernel(grad)
    u, s = 0.3, -0.2
    expected0 = sum(row[0]*u**i for i, row in enumerate(grad[0]))
    expected1 = sum(
        sum(c*s**j for j, c in enumerate(row))*u**i
        for i, row in enumerate(grad[1]))
    np.testing.assert_allclose(
        kernel.gradient(u, s), (expected0, expected1), rtol=2e-16)


def test_native_centered_curve_step_removes_transverse_stiffness():
    """For da'=da, db'=-db the integral curves satisfy db=C/da."""
    kernel = _native.LocalKernel((
        ((0.0,), (1.0,)),
        ((0.0, -1.0),),
    ))
    db = kernel.curve_step(1.0, 1.0, 0.1, 0, 6)
    np.testing.assert_allclose(db, 1.0/1.1, rtol=2e-12)


def test_native_centered_raw_gl6_step_matches_linear_flow():
    kernel = _native.LocalKernel((
        ((0.0,), (2.0,)),
        ((0.0, 5.0),),
    ))
    z0 = np.array((0.3, -0.2))
    h = -1e-3
    got = kernel.raw_step(*z0, h, 6)
    want = z0*np.exp(h*np.array((2.0, 5.0)))
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-16)
    got8 = kernel.raw_step(*z0, h, 8)
    np.testing.assert_allclose(got8, want, rtol=2e-14, atol=2e-16)


def test_native_gl8_stage_solve_is_row_scaled_on_anisotropic_linear_jet():
    kernel = _native.LocalKernel((
        ((0.0,), (1e12,)),
        ((0.0, 1e-6),),
    ))
    z0 = np.array((1e-12, -2.0))
    h = -1e-10

    def F(_t, z):
        return np.array((1e12*z[0], 1e-6*z[1]))

    def J(_t, _z):
        return np.array(((1e12, 0.0), (0.0, 1e-6)))

    native = np.asarray(kernel.raw_step(*z0, h, 8))
    oracle = gauss.step(F, 0.0, z0, h, method="gl8", jac=J).y1
    assert np.all(np.isfinite(native))
    np.testing.assert_allclose(native, oracle, rtol=3e-13, atol=3e-16)


def test_centered_critical_jets_are_stationary_and_have_symmetric_hessian(d2):
    demo_model, e = d2
    for p in e.points:
        if p.kind == "degenerate":
            continue
        assert p.local is not None
        assert p.local.gradient(0.0, 0.0) == (0.0, 0.0)
        H = np.asarray(p.local.hessian)
        assert H[0, 1] == H[1, 0]
        assert p.local.spectral.eigenvalues[0] < 0 if p.kind == "saddle" \
            else p.local.spectral.eigenvalues[0] > 0
        if p.kind == "saddle":
            assert {x.manifold for x in p.local.poincare} == {
                "stable", "unstable"}
            for chart in p.local.poincare:
                assert chart.desired_reach > 0
                assert chart.normal_native is not None
                assert chart.conditioned_native is not None
                assert np.all(np.isfinite(chart.divisors))
                assert chart.physical(p.local, 0.0, 0.0) == (p.a, p.b)
                transformed = np.asarray(chart.transformed_quadratic)
                retained = np.asarray(chart.retained)
                assert np.max(np.abs(transformed[~retained]), initial=0.0) < 1e-10
                assert chart.velocity(p.local, 0.0, 0.0) == (0.0, 0.0)
                # Decimal and FP64 frames must use the same row/column
                # convention when the exact physical jet is rotated.
                R = np.asarray(chart.frame)
                normal = np.array((0.017, -0.023))
                physical = R @ normal
                np.testing.assert_allclose(
                    R @ chart.normal_native.gradient(*normal),
                    p.local.gradient(*physical), rtol=3e-13, atol=3e-14)
                u, s = normal
                pmap, qmap = chart.selected_map
                U = u+pmap[0]*u*u+pmap[1]*u*s+pmap[2]*s*s
                S = s+qmap[0]*u*u+qmap[1]*u*s+qmap[2]*s*s
                J = np.array((
                    (1+2*pmap[0]*u+pmap[1]*s,
                     pmap[1]*u+2*pmap[2]*s),
                    (2*qmap[0]*u+qmap[1]*s,
                     1+qmap[1]*u+2*qmap[2]*s)))
                pulled = np.asarray(
                    chart.conditioned_native.gradient(u, s))
                direct = np.asarray(chart.normal_native.gradient(U, S))
                np.testing.assert_allclose(
                    J @ pulled, np.linalg.det(J)*direct,
                    rtol=3e-13, atol=3e-14)


def test_centered_jet_matches_global_gradient_away_from_cancellation(d2):
    demo_model, e = d2
    for p in e.points:
        if p.local is None:
            continue
        da, db = 2e-4, -3e-4
        got = p.local.gradient(da, db)
        want = demo_model.gradL(p.a + da, p.b + db)
        np.testing.assert_allclose(got, want, rtol=2e-10, atol=2e-13)


def test_native_centered_gl6_step_agrees_with_global_kernel(d2):
    demo_model, e = d2
    p = e.saddles[0]
    assert p.local.native is not None
    da, db, h = 3e-4, -2e-4, 1e-5
    local = p.local.native.normalized_step(da, db, h, 6)
    global_ = demo_model._native_kernel.normalized_step(
        p.a + da, p.b + db, h, 6)
    np.testing.assert_allclose(
        np.asarray(local) + (p.a, p.b), global_, rtol=2e-10, atol=2e-13)


def test_centered_raw_arrival_captures_regular_minimum(d2):
    demo_model, e = d2
    p = e.minima[0]
    start = (p.a+2e-2, p.b-1e-2)
    diag = {}
    curve, term = charts._centered_raw_arrival(
        start, (p.a, p.b), p.local, 1e-5, diag)
    assert term == "capture"
    assert curve[-1] == (p.a, p.b)
    assert diag["centered_arrival"]["accepted_steps"] > 0
    centered_loss = np.array([
        p.local.potential(a-p.local.a, b-p.local.b) for a, b in curve[:-1]])
    assert np.all(np.diff(centered_loss) < 0.0)


def test_geometry_consumes_native_critical_chart(d2):
    demo_model, e = d2
    p = e.saddles[0]
    br = charts.trace_stable(
        demo_model, p.b, 1, box=(-10.0, 10.0, -12.0, 8.0),
        critical_local=p.local)
    assert br.diag["critical_chart"]
    assert br.diag["critical_order"] == 6
    assert br.diag["critical_launch_nonlinearity"] <= 1e-10
    assert br.diag["critical_steps"] > 1
    assert np.all(np.any(br.Y[1:] != br.Y[:-1], axis=1))


def test_native_hadamard_graph_is_invariant_and_grid_convergent(d2):
    _m, e = d2
    p = e.saddles[0]
    coarse, dc = p.local.unstable_graph(2e-3, n=257)
    fine, df = p.local.unstable_graph(2e-3, n=513)
    assert dc["relative_change"] < 1e-11
    assert df["relative_change"] < 1e-11
    np.testing.assert_allclose(coarse[-1], fine[-1], rtol=2e-8, atol=2e-12)

    tangent = np.gradient(fine, axis=0)
    velocity = np.asarray([p.local.gradient(*z) for z in fine])
    cross = tangent[:, 0] * velocity[:, 1] - tangent[:, 1] * velocity[:, 0]
    denom = np.linalg.norm(tangent, axis=1) * np.linalg.norm(velocity, axis=1)
    defect = np.abs(cross[2:]) / np.maximum(denom[2:], 1e-300)
    assert np.max(defect) < 2e-8


def test_native_hadamard_graph_resolves_extreme_stiff_saddle(tricky):
    _m, e = tricky
    p = max(e.saddles, key=lambda q:
            q.local.spectral.eigenvalues[1]
            / abs(q.local.spectral.eigenvalues[0]))
    lam = np.asarray(p.local.spectral.eigenvalues)
    assert lam[1] / abs(lam[0]) > 1e12
    coarse, dc = p.local.unstable_graph(1e-3, n=257, sign=-1)
    fine, df = p.local.unstable_graph(1e-3, n=513, sign=-1)
    assert dc["relative_change"] < 1e-11
    assert df["relative_change"] < 1e-11
    np.testing.assert_allclose(coarse[-1], fine[-1], rtol=1e-11, atol=1e-15)
    comparison = p.local.compare_graph_acceleration(
        1e-3, n=513, sign=-1, manifold="unstable")
    assert comparison["plain_iterations"] is not None
    assert comparison["rre_iterations"] is not None
    assert comparison["rre_residual"] <= comparison["plain_residual"]


def test_poincare_quadrature_is_stable_at_extreme_eigenvalue_ratios(tricky):
    _m, e = tricky
    p = max(e.saddles, key=lambda q:
            q.local.spectral.eigenvalues[1]
            / abs(q.local.spectral.eigenvalues[0]))
    for chart in p.local.poincare:
        coarse, dc = chart.graph(p.local, -1, n=257, reach=1e-4)
        fine, df = chart.graph(p.local, -1, n=513, reach=1e-4)
        assert dc["relative_change"] < 1e-10
        assert df["relative_change"] < 1e-10
        error = np.max(np.abs(
            np.asarray(dc["normal_h"])-np.asarray(df["normal_h"])[::2]))
        assert error < 1e-6 * 1e-4
        assert np.all(np.isfinite(coarse))
        assert np.all(np.isfinite(fine))


def test_materialized_stubs_are_certified_and_topologically_labeled(d2):
    m, e = d2
    enriched = sturm.materialize_stubs(m, e)
    assert all(len(p.stubs) == 4 for p in enriched.saddles)
    for p in enriched.saddles:
        assert {(s.manifold, s.orientation) for s in p.stubs} == {
            ("stable", -1), ("stable", 1),
            ("unstable", -1), ("unstable", 1)}
        unstable = [s for s in p.stubs if s.manifold == "unstable"]
        assert {s.b_direction for s in unstable} == {-1, 1}
        for stub in p.stubs:
            cert = dict(stub.certificates)
            assert cert["graph_change_error_fine"] < 1e-12
            assert cert["grid_error"] < 1e-6
            assert np.isfinite(cert["field_cosine"])
            assert cert["global_field_ready"] in (0, 1)
            assert cert["injectivity_margin"] > 1e-6
            assert cert["invariance_direction_error"] < 1e-5
            assert cert["global_resolution_margin"] > 0
            assert cert["monotone_failures"] == 0
            assert len(stub.curve) > 10
            if stub.manifold == "stable":
                assert stub.destination_kind == "infinity"


def test_exact_spectral_frame_recovers_large_radius_saddle():
    """Seed 2966286515: FP64 eigvalsh erases the small negative eigenvalue."""
    rng = np.random.default_rng(2966286515)
    dg = int(rng.integers(3, 7))
    g = [int(x) for x in rng.integers(-4, 5, size=dg+1)]
    if g[0] == 0:
        g[0] = int(rng.choice([-1, 1]))
    if g[-1] == 0:
        g[-1] = int(rng.choice([-1, 1]))
    radius = Fraction(int(rng.choice([-1, 1])) * 2**14)
    moments = model.moments_uniform01(2*(dg+1)+1)
    design = inverse.design([radius], g, moments, deg_f=dg+1)
    enumeration = sturm.enumerate_critical_points(design.model)
    saddle = max(enumeration.saddles, key=lambda p: abs(p.b))

    assert np.linalg.eigvalsh(np.asarray(saddle.local.hessian))[0] == 0.0
    lm, lp = saddle.local.spectral.eigenvalues
    assert lm < 0.0 < lp
    assert lm < -1e-26
    assert lp > 1e33
    assert lm*lp == np.float64(
        saddle.local.spectral.determinant.numerator
        / saddle.local.spectral.determinant.denominator)
    R = np.asarray(saddle.local.spectral.frame)
    np.testing.assert_allclose(R.T @ R, np.eye(2), atol=2e-16)

    enriched = sturm.materialize_stubs(design.model, enumeration)
    saddle = max(enriched.saddles, key=lambda p: abs(p.b))
    assert len(saddle.stubs) == 4
    assert all(dict(s.certificates)["poincare_conditioned"] == 1.0
               for s in saddle.stubs)
    assert all(c.normal_native is not None for c in saddle.local.poincare)
    # The exact saddle signature survives, but its unstable eigenvalue is too
    # small for any finite FP64 graph prefix to satisfy the invariance-angle
    # certificate.  This is now an explicit conditioning refusal, not a false
    # launch followed by a generic continuation failure.
    unstable = [s for s in saddle.stubs if s.manifold == "unstable"]
    assert all(dict(s.certificates)["graph_certified"] == 0.0
               for s in unstable)
    assert all(dict(s.certificates)["global_field_ready"] == 0.0
               for s in unstable)
    assert all(dict(s.certificates)["spectral_resolution_margin"] < 1.0
               for s in unstable)
    assert all(dict(s.certificates)["fp64_spectral_resolved"] == 0.0
               for s in unstable)
