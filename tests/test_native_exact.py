"""Qualification gates for the GMP exact-polynomial core."""

from fractions import Fraction
import random

from spong import _native
from spong import _poly as P
from spong import sturm
from spong import topology


def _python_analysis(p):
    trimmed = P.trim(p)
    gcd = P.gcd_poly(trimmed, P.deriv(trimmed))
    squarefree = (trimmed if P.degree(gcd) <= 0
                  else P.divmod_exact(trimmed, gcd)[0])
    repeated = (
        sturm._count_roots_python(gcd) if P.degree(gcd) > 0 else 0)
    return {
        "distinct_real_roots": sturm._count_roots_python(trimmed),
        "repeated_real_roots": repeated,
        "input_degree": P.degree(trimmed),
        "squarefree_degree": P.degree(squarefree),
    }


def _native_interval_count(p, lo, hi):
    result = _native.sturm_count_interval(
        P.int_primitive(p),
        None if lo is None else lo.numerator,
        None if lo is None else lo.denominator,
        None if hi is None else hi.numerator,
        None if hi is None else hi.denominator,
        0, 0, 0)
    assert result["status"] == _native.SPONG_EXACT_OK
    return result["count"]


def _python_sign_polynomial_at_root(root_polynomial, query, interval,
                                    max_bisections=160):
    """Independent Fraction oracle for the native algebraic-sign kernel."""
    if interval.exact:
        value = P.eval_at(query, interval.lo)
        return (value > 0) - (value < 0)
    squarefree = sturm.squarefree_part(root_polynomial)
    lo, hi = interval.lo, interval.hi
    sign_lo = P.eval_at(squarefree, lo)
    for bisections in range(max_bisections + 1):
        enclosed_lo, enclosed_hi = topology._polynomial_interval(
            query, sturm.RootInterval(lo, hi, False))
        if enclosed_lo > 0:
            return 1
        if enclosed_hi < 0:
            return -1
        if bisections == max_bisections:
            break
        mid = (lo+hi)/2
        sign_mid = P.eval_at(squarefree, mid)
        if sign_mid == 0:
            value = P.eval_at(query, mid)
            return (value > 0) - (value < 0)
        if (sign_mid > 0) == (sign_lo > 0):
            lo, sign_lo = mid, sign_mid
        else:
            hi = mid
    return None


def _native_isolate(p):
    result = _native.SturmPlan(P.int_primitive(p)).isolate()
    assert result["status"] == _native.SPONG_EXACT_OK
    return [
        sturm.RootInterval(
            Fraction(int(lo_num), int(lo_den)),
            Fraction(int(hi_num), int(hi_den)),
            bool(exact))
        for lo_num, lo_den, hi_num, hi_den, exact in result["intervals"]
    ]


def test_native_sturm_matches_fraction_oracle_on_random_integer_polynomials():
    rng = random.Random(20260729)
    for _ in range(160):
        degree = rng.randrange(1, 13)
        coefficients = [rng.randrange(-2**18, 2**18)
                        for _ in range(degree+1)]
        coefficients[-1] = rng.choice([-1, 1])*rng.randrange(1, 2**18)
        p = tuple(Fraction(x) for x in coefficients)
        expected = _python_analysis(p)
        actual = _native.sturm_analyze(
            P.int_primitive(p), 0, 0, 0)
        assert actual["status"] == _native.SPONG_EXACT_OK
        for key, value in expected.items():
            assert actual[key] == value
        assert actual["chain_polynomials"] > 0
        assert actual["chain_coefficients"] >= \
            actual["chain_polynomials"]


def test_native_sturm_matches_repeated_real_and_complex_factor_families():
    x_minus_one = (Fraction(-1), Fraction(1))
    x_plus_two = (Fraction(2), Fraction(1))
    x2_plus_one = (Fraction(1), Fraction(0), Fraction(1))
    families = [
        P.mul(P.mul(x_minus_one, x_minus_one), x_plus_two),
        P.mul(P.mul(x2_plus_one, x2_plus_one), x_minus_one),
        P.mul(P.mul(x2_plus_one, x2_plus_one), P.mul(
            x_minus_one, x_minus_one)),
    ]
    for p in families:
        expected = _python_analysis(p)
        actual = _native.sturm_analyze(P.int_primitive(p), 0, 0, 0)
        for key, value in expected.items():
            assert actual[key] == value


def test_native_exact_work_limit_refuses_instead_of_silently_truncating():
    p = (2, -3, 0, 1)
    limited = _native.sturm_analyze(p, 1, 0, 0)
    assert limited["status"] == _native.SPONG_EXACT_WORK_LIMIT
    assert limited["peak_coefficient_bits"] > 1
    limited = _native.sturm_analyze(p, 0, 2, 0)
    assert limited["status"] == _native.SPONG_EXACT_WORK_LIMIT
    limited = _native.sturm_analyze(p, 0, 0, 1)
    assert limited["status"] == _native.SPONG_EXACT_WORK_LIMIT


def test_native_bounded_sturm_matches_fraction_oracle_on_random_rationals():
    rng = random.Random(20260730)
    for _ in range(240):
        degree = rng.randrange(1, 13)
        coefficients = [rng.randrange(-2**14, 2**14)
                        for _ in range(degree + 1)]
        coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 2**14)
        p = tuple(Fraction(x) for x in coefficients)
        endpoints = sorted((
            Fraction(rng.randrange(-100, 101), rng.randrange(1, 32)),
            Fraction(rng.randrange(-100, 101), rng.randrange(1, 32))))
        lo, hi = endpoints
        expected = sturm._count_roots_python(p, lo, hi)
        assert _native_interval_count(p, lo, hi) == expected


def test_persistent_native_sturm_plan_reuses_chain_for_many_intervals():
    p = tuple(Fraction(x) for x in (
        1260, -1308, -5049, 4788, 5145, -4356, -1395, 1068, 0, -72, 9))
    plan = _native.SturmPlan(P.int_primitive(p))
    rng = random.Random(20260731)
    for _ in range(200):
        lo, hi = sorted((
            Fraction(rng.randrange(-80, 81), rng.randrange(1, 24)),
            Fraction(rng.randrange(-80, 81), rng.randrange(1, 24))))
        assert plan.count(lo, hi) == sturm._count_roots_python(p, lo, hi)
    assert plan.count(None, None) == sturm._count_roots_python(p)


def test_native_bounded_sturm_infinities_and_endpoint_convention():
    # Roots are -2, 0, and 1.  Sturm counting is on (lo, hi], so a root at
    # the lower endpoint is excluded and a root at the upper is included.
    p = P.mul(P.mul((2, 1), (0, 1)), (-1, 1))
    intervals = [
        (None, None),
        (None, Fraction(0)),
        (Fraction(0), None),
        (Fraction(-2), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(3)),
        (Fraction(-3), Fraction(-2)),
    ]
    for lo, hi in intervals:
        assert _native_interval_count(p, lo, hi) == \
            sturm._count_roots_python(p, lo, hi)


def test_native_isolation_matches_fraction_oracle_on_random_polynomials():
    rng = random.Random(20260801)
    for _ in range(180):
        degree = rng.randrange(1, 11)
        coefficients = [rng.randrange(-2**12, 2**12)
                        for _ in range(degree + 1)]
        coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 2**12)
        p = tuple(Fraction(x) for x in coefficients)
        assert _native_isolate(p) == sturm._isolate_roots_python(p)


def test_native_isolation_certifies_puncture_around_exact_root():
    # x(1_000_000 x - 1) has roots 0 and 1e-6.  The initial 2^-20
    # puncture around zero contains both, so it must be halved.
    p = (Fraction(0), Fraction(-1), Fraction(1_000_000))
    plan = _native.SturmPlan(P.int_primitive(p))
    result = plan.isolate()
    assert result["status"] == _native.SPONG_EXACT_OK
    assert len(result["intervals"]) == 2
    assert result["puncture_halvings"] >= 1
    assert _native_isolate(p) == sturm._isolate_roots_python(p)


def test_native_isolation_work_limit_refuses_without_partial_answer():
    result = _native.SturmPlan((-2, 0, 1)).isolate(1)
    assert result["status"] == _native.SPONG_EXACT_WORK_LIMIT
    assert result["intervals"] == []


def test_native_refinement_matches_fraction_oracle():
    rng = random.Random(20260803)
    rel = Fraction(1, 2**48)
    compared = 0
    for _ in range(120):
        degree = rng.randrange(2, 11)
        coefficients = [rng.randrange(-2**10, 2**10)
                        for _ in range(degree + 1)]
        coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 2**10)
        p = tuple(Fraction(x) for x in coefficients)
        plan = _native.SturmPlan(P.int_primitive(p))
        for iv in sturm._isolate_roots_python(p):
            expected = sturm._refine_python(p, iv, rel)
            result = plan.refine(iv.lo, iv.hi, rel)
            assert result["status"] == _native.SPONG_EXACT_OK
            lo_num, lo_den, hi_num, hi_den, exact = result["interval"]
            actual = sturm.RootInterval(
                Fraction(int(lo_num), int(lo_den)),
                Fraction(int(hi_num), int(hi_den)), bool(exact))
            assert actual == expected
            compared += 1
    assert compared > 100


def test_native_original_polynomial_sign_matches_fraction_evaluation():
    rng = random.Random(20260804)
    for _ in range(300):
        degree = rng.randrange(1, 12)
        coefficients = [rng.randrange(-2**12, 2**12)
                        for _ in range(degree + 1)]
        coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 2**12)
        p = tuple(Fraction(x) for x in coefficients)
        x = Fraction(rng.randrange(-100, 101), rng.randrange(1, 40))
        expected = P.eval_at(p, x)
        expected_sign = (expected > 0) - (expected < 0)
        assert _native.SturmPlan(P.int_primitive(p)).sign_at(x) == \
            expected_sign


def test_native_polynomial_sign_at_algebraic_root_matches_interval_oracle():
    rng = random.Random(20260806)
    compared = 0
    for _ in range(120):
        root_degree = rng.randrange(1, 7)
        root_coefficients = [rng.randrange(-64, 65)
                             for _ in range(root_degree + 1)]
        root_coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 65)
        root_polynomial = tuple(Fraction(x) for x in root_coefficients)
        plan = _native.SturmPlan(P.int_primitive(root_polynomial))
        for interval in sturm._isolate_roots_python(root_polynomial):
            query_degree = rng.randrange(0, 7)
            query = tuple(Fraction(rng.randrange(-64, 65))
                          for _ in range(query_degree + 1))
            query = P.trim(query)
            if not query:
                query = (Fraction(1),)
            expected = _python_sign_polynomial_at_root(
                root_polynomial, query, interval)
            result = plan.sign_polynomial_at_root(
                P.int_primitive(query), interval.lo, interval.hi,
                interval.exact, max_bisections=160)
            assert result["status"] == _native.SPONG_EXACT_OK
            assert result["sign"] == expected
            compared += 1
    assert compared > 100


def test_native_polynomial_sign_at_shared_algebraic_root_is_unresolved():
    root = (-2, 0, 1)
    plan = _native.SturmPlan(root)
    result = plan.sign_polynomial_at_root(
        root, Fraction(1), Fraction(2), False, max_bisections=24)
    assert result["status"] == _native.SPONG_EXACT_OK
    assert result["sign"] is None
    assert result["resolved"] == 0
    assert result["bisections"] == 24


def test_native_refinement_work_limit_refuses_without_partial_answer():
    plan = _native.SturmPlan((-2, 0, 1))
    result = plan.refine(
        Fraction(-3), Fraction(0), Fraction(1, 2**48), 1, 0)
    assert result["status"] == _native.SPONG_EXACT_WORK_LIMIT
    assert result["interval"] is None


def test_production_interval_sign_matches_fraction_oracle():
    rng = random.Random(20260805)
    for _ in range(300):
        degree = rng.randrange(1, 11)
        coefficients = [rng.randrange(-2**10, 2**10)
                        for _ in range(degree + 1)]
        coefficients[-1] = rng.choice([-1, 1]) * rng.randrange(1, 2**10)
        p = tuple(Fraction(x) for x in coefficients)
        lo, hi = sorted((
            Fraction(rng.randrange(-80, 81), rng.randrange(1, 30)),
            Fraction(rng.randrange(-80, 81), rng.randrange(1, 30))))
        iv = sturm.RootInterval(lo, hi, False)
        if sturm._count_roots_python(p, lo, hi) != 0:
            expected = None
        else:
            value = P.eval_at(p, iv.mid)
            expected = None if value == 0 else (1 if value > 0 else -1)
        assert sturm.interval_sign(p, iv) == expected
