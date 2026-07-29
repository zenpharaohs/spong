from fractions import Fraction as F

import pytest

from spong import model, qualification, sturm


def test_arithmetic_profile_uses_only_consumed_moment_prefix():
    mu = model.moments_uniform01(20)
    m = model.build([1, 2, 3], [2, -1], mu)
    q = qualification.arithmetic_profile(m)
    assert q["degrees"]["f"] == 2
    assert q["degrees"]["g"] == 1
    assert q["moments"]["highest_order"] == 4
    assert q["moments"]["count"] == 5
    assert q["moments"]["binary64_all_nonzero_finite"]


def test_raw_moment_scale_detects_scaled_entire_mgf():
    degree = 4
    base = model.moments_normal01(2*degree + 1)
    sigma = F(2)**30
    scaled = tuple(x*sigma**k for k, x in enumerate(base))
    m0 = model.build([1]*5, [1]*5, base)
    m1 = model.build([1]*5, [1]*5, scaled)
    q0 = qualification.moment_profile(m0)
    q1 = qualification.moment_profile(m1)
    assert q1["variance_log2"] == pytest.approx(
        q0["variance_log2"] + 60.0)
    # Standardization removes scale, as it should.
    assert q1["max_standardized_central_moment_log2"] == pytest.approx(
        q0["max_standardized_central_moment_log2"])
    assert q1["max_abs_log2"] > q0["max_abs_log2"] + 200.0


def test_polynomial_profile_not_fooled_by_exact_compensation():
    degree = 4
    sigma = F(2)**20
    base = model.moments_normal01(2*degree + 1)
    scaled = tuple(x*sigma**k for k, x in enumerate(base))
    coefficients = tuple(F(1, sigma**k) for k in range(degree + 1))
    m0 = model.build([1]*5, [1]*5, base)
    m1 = model.build(coefficients, coefficients, scaled)
    q0 = qualification.arithmetic_profile(m0)
    q1 = qualification.arithmetic_profile(m1)
    for name in ("A", "B", "N"):
        assert q1["polynomials"][name] == q0["polynomials"][name]


def test_reduced_backbone_derivative_matches_model_identity():
    from spong import _poly as P

    m = model.build([1, -2, 4, 1], [3, 1, -2],
                    model.moments_uniform01(7))
    r = qualification.reduced_backbone_polynomials(m)
    # u' = H/D² in the reduced representation and B*N/A² in Model.
    assert P.mul(r["derivative_numerator"], P.mul(m.alpha, m.alpha)) \
        == P.mul(P.mul(m.beta, m.N),
                 P.mul(r["denominator"], r["denominator"]))
    q = qualification.backbone_profile(m)
    assert q["critical_squarefree"]["degree"] <= \
        q["derivative_numerator"]["degree"]


def test_constant_backbone_has_zero_critical_polynomial():
    m = model.build([1], [1], model.moments_uniform01(1))
    q = qualification.backbone_profile(m)
    assert q["derivative_numerator"]["degree"] == -1
    assert q["critical_squarefree"]["degree"] == -1


def test_morse_preflight_has_cheap_and_cached_chain_levels():
    m = model.build([1, 1, 1], [1, 1, 1],
                    model.moments_uniform01(5))
    cheap = qualification.morse_preflight(m)
    assert cheap["max_degree"] >= 1
    assert cheap["max_primitive_height_bits"] >= 1
    assert "sturm_chains" not in cheap
    assert cheap["native_peak_coefficient_bits"] >= 1
    assert cheap["native_total_prs_steps"] >= 1
    assert all(x["status"] == 0
               for x in cheap["native_exact"].values())
    warm = qualification.morse_preflight(m, include_sturm_chain=True)
    assert warm["peak_sturm_coefficient_bits"] >= 1
    assert warm["total_sturm_coefficient_bits"] >= \
        warm["peak_sturm_coefficient_bits"]


def test_sturm_work_counters_are_observational():
    from spong import _poly as P

    p = P.poly([-2, 0, 1])
    plain = sturm.isolate_roots(p)
    stats = {}
    measured = sturm.isolate_roots(p, stats=stats)
    assert measured == plain
    assert stats["isolated_roots"] == 2
    assert stats["variation_evaluations"] > 0
    assert stats["subdivision_nodes"] > 0
    refined = sturm.refine(p, measured[0], stats=stats)
    assert refined.lo <= refined.hi
    assert stats["refinement_bisections"] > 0


def test_skeleton_profile_measures_exactly_enumerated_roots_and_stubs():
    m = model.build([1, 1, 1], [1, 1, 1],
                    model.moments_uniform01(5))
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    q = qualification.skeleton_profile(m, e)
    assert q["critical_count"] == 4
    assert q["morse_exact"] and q["alpha_positive_exact"]
    assert q["critical_coordinates_binary64_distinct"]
    assert q["min_adjacent_root_separation_ulps"] > 1.0
    assert q["morse_root_collision_margin_log2_eps"] > 0.0
    assert q["min_A_evaluation_margin_log2_eps"] > 0.0
    assert q["min_hessian_relative_nonsingularity"] > 0.0
    assert q["min_spectral_resolution_margin"] is not None
    assert q["max_gamma_l1_upper_log2"] is not None
    assert q["max_gamma_target_product_log2"] is not None
    assert len(q["critical_profiles"]) == q["critical_count"]
    assert all("gamma_l1_upper_log2" in x
               for x in q["critical_profiles"])


def test_gamma_bound_is_invariant_under_loss_scaling():
    m0 = model.build([1, -2, 1], [1, 1, 1],
                     model.moments_uniform01(5))
    m1 = model.build([7, -14, 7], [1, 1, 1],
                     model.moments_uniform01(5))
    e0 = sturm.enumerate_critical_points(m0)
    e1 = sturm.enumerate_critical_points(m1)
    # Scaling f alone does not generally scale L, so use corresponding points
    # only to verify that every exact local gamma profile is finite/nonnegative
    # where nonlinear terms exist.  True loss-scale invariance is algebraic in
    # H^-1 D^kF and is exercised directly below by scaling the stored jet.
    local = next(q.local for q in e0.points if q.local is not None)
    q0 = qualification.local_gamma_l1_profile(local)
    from dataclasses import replace
    scale = F(7)
    scaled = replace(
        local,
        exact_hessian=tuple(tuple(scale*x for x in row)
                            for row in local.exact_hessian),
        exact_grad=tuple(
            tuple(tuple(scale*x for x in row) for row in component)
            for component in local.exact_grad))
    q1 = qualification.local_gamma_l1_profile(scaled)
    assert q1["gamma_l1_upper_log2"] == pytest.approx(
        q0["gamma_l1_upper_log2"])
