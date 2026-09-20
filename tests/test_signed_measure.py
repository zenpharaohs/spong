"""The signed-measure (Krein) regime: indefinite form, A > 0, still Morse.

These assert the structural facts that DO NOT depend on the loss being
nonnegative, because that is precisely what a future positivity assumption
would break.  The backbone theory never used L >= 0; it used H11 = 2A with
A(b) > 0 certified exactly, which is why an indefinite moment functional
changes the interpretation of L and nothing about its critical structure.
"""

from fractions import Fraction

import numpy as np
import pytest

from spong import model, resolution, sturm, zoo


def _det_exact(rows):
    a = [row[:] for row in rows]
    n = len(a)
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        det *= a[col][col]
        inv = Fraction(1)/a[col][col]
        for r in range(col+1, n):
            factor = a[r][col]*inv
            if factor:
                for c in range(col, n):
                    a[r][c] -= factor*a[col][c]
    return det


@pytest.fixture(scope="module", params=zoo.signed_measure_names())
def case(request):
    return zoo.get_signed_measure(request.param)


@pytest.fixture(scope="module")
def built(case):
    m = case.build()
    return case, m, sturm.enumerate_critical_points(m)


def test_hankel_is_genuinely_indefinite(built):
    """The stored signature, by Jacobi's rule on leading principal minors.

    EXACT: no eigenvalues, no tolerance.  The number of negative eigenvalues
    is the number of sign changes in the minor sequence, which is valid
    because none of the minors vanishes -- asserted here, since a zero minor
    would make the rule inapplicable rather than merely inaccurate.
    """
    case, m, _e = built
    size = max(len(case.f), len(case.g))
    mu = case.moments(2*size - 1)
    minors = [Fraction(1)]
    for n in range(1, size+1):
        minors.append(_det_exact(
            [[mu[i+j] for j in range(n)] for i in range(n)]))
    assert all(x != 0 for x in minors), "Jacobi's rule needs nonzero minors"
    negative = sum(1 for i in range(size)
                   if (minors[i] > 0) != (minors[i+1] > 0))
    assert (size-negative, negative) == case.hankel_signature
    assert negative > 0, "the point of the case is that the form is indefinite"


def test_the_form_is_positive_where_the_model_evaluates_it(built):
    """A(b) > 0 for all real b, exactly -- the hypothesis that matters.

    A(b) = <g(b.), g(b.)> is the indefinite form on the one-dimensional span
    of g(b.), so this says the form is positive on exactly the subspace where
    a is optimised.  The negative cone misses the Veronese curve.
    """
    _case, m, e = built
    assert e.psi_positive
    assert sturm.is_positive(m.alpha)


def test_no_local_maximum_survives_indefiniteness(built):
    """H11 = 2A > 0, so the Hessian is never negative definite.

    This is the fact the whole backbone theory rests on, and it is a
    statement about A, not about the sign of L.  Checked at the critical
    points and at ordinary samples in between.
    """
    _case, m, e = built
    for point in e.points:
        H = m.hessL(float(point.a), float(point.b))
        assert H[0, 0] > 0.0
    for b in np.linspace(-6.0, 6.0, 61):
        for a in (-2.0, -0.3, 0.0, 0.7, 3.0):
            assert m.hessL(a, b)[0, 0] > 0.0
    assert all(point.kind in ("min", "saddle") for point in e.points)


def test_critical_points_lie_on_the_backbone_and_classify_by_u2(built):
    """det H = 2A u'', so minima are u'' > 0 and saddles u'' < 0.

    The identity is algebra in A, B and C and does not consult the
    signature; this pins that it still holds in the indefinite regime.
    """
    _case, m, e = built
    for point in e.points:
        a, b = float(point.a), float(point.b)
        assert a == pytest.approx(float(m.a_star(b)), rel=1e-9, abs=1e-12)
        H = m.hessL(a, b)
        det = H[0, 0]*H[1, 1] - H[0, 1]**2
        assert det == pytest.approx(2*m.A(b)*m.u_pp(b), rel=1e-7)
        if point.kind == "min":
            assert point.u2_sign > 0 and det > 0.0
        else:
            assert point.u2_sign < 0 and det < 0.0


def test_u2_still_alternates(built):
    """Theorem 2 is about the one-dimensional Morse function u, not L >= 0."""
    _case, _m, e = built
    assert e.morse
    assert e.alternates


def test_the_loss_is_still_bounded_below(built):
    """inf L = min over the critical points, attained, even with C < 0.

    L = u + A w^2 with A > 0, so at each b the fibre minimum is u(b); the
    global infimum is therefore inf over b of u, and u -> u_inf finitely at
    both ends.  A negative C would put that infimum below zero without
    making the landscape unbounded, so this asserts the BOUND, not a sign:
    no sampled point of the plane goes below the critical floor, and u out
    at |b| = 1e6 has already settled to its limit.
    """
    _case, m, e = built
    floor = min(float(m.L(float(q.a), float(q.b))) for q in e.points)
    for b in np.linspace(-40.0, 40.0, 401):
        for a in (-5.0, -1.0, -0.2, 0.0, 0.3, 2.0, 8.0):
            assert float(m.L(a, b)) >= floor - 1e-9
    # u_inf is the same finite limit at both ends, approached from within.
    ends = [float(m.u(b)) for b in (-1e5, -1e6, 1e5, 1e6)]
    assert all(np.isfinite(value) for value in ends)
    assert max(ends) - min(ends) < 1e-3*max(1.0, abs(ends[0]))
    assert min(ends) >= floor - 1e-9


def test_resolution_certifies_the_signed_case(built):
    """The public contract returns a certified portrait, not a refusal.

    An indefinite moment functional is a legitimate regime -- least squares
    in a Krein space -- and spong is expected to solve it, not decline it.
    """
    _case, m, _e = built
    r = resolution.resolve(m)
    assert r.status is resolution.ResolutionStatus.CERTIFIED_PORTRAIT
    assert r.reason is resolution.ResolutionReason.NONE
    assert r.exact_morse is True


def test_moments_are_exact_rationals_and_normalised(built):
    """mu_0 = 1 and every moment is a Fraction: no float enters the model."""
    case, m, _e = built
    mu = case.moments(2*max(len(case.f), len(case.g)) - 1)
    assert mu[0] == 1
    assert all(isinstance(value, Fraction) for value in mu)
    assert all(isinstance(c, Fraction) for c in m.alpha)
    assert all(isinstance(c, Fraction) for c in m.beta)
    assert isinstance(m.C, Fraction)


def test_a_positive_definite_control_has_no_negative_direction():
    """The same machinery on uniform01, to show the test can fail.

    Without this, every assertion above would pass on a case whose Hankel
    is positive definite and the suite would not notice.
    """
    mu = model.moments_uniform01(9)
    size = 5
    minors = [Fraction(1)]
    for n in range(1, size+1):
        minors.append(_det_exact(
            [[mu[i+j] for j in range(n)] for i in range(n)]))
    negative = sum(1 for i in range(size)
                   if (minors[i] > 0) != (minors[i+1] > 0))
    assert negative == 0
