"""Exact rational-response SPONG models on the two-point input measure.

For X equally likely to be -1 or +1, f(X)=X.  If P is odd, Q is even,
and g(z)=P(z)+gain*Q(z), then

    A=P(b)^2+gain^2 Q(b)^2,  B=P(b),  C=1,
    u(b)=gain^2 Q(b)^2 / (P(b)^2+gain^2 Q(b)^2).

This realizes the squared-response form 1/(1+(P/(gain*Q))^2) wherever
Q is nonzero.  Coprime P and Q ensure A>0 on the real line.  This is a
construction of SPONG instances, not a filter-design algorithm.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from . import _poly as poly, model


def rational_response_model(numerator, denominator=(1,), gain=1) -> model.Model:
    """Build the exact SPONG instance for an odd P / even Q response."""
    p = poly.poly(numerator)
    q = poly.poly(denominator)
    k = poly.as_fraction(gain)
    if not p or any(p[j] for j in range(0, len(p), 2)):
        raise ValueError("numerator must be a nonzero odd polynomial")
    if not q or any(q[j] for j in range(1, len(q), 2)):
        raise ValueError("denominator must be a nonzero even polynomial")
    if k == 0:
        raise ValueError("gain must be nonzero")
    if poly.degree(poly.gcd_poly(p, q)) > 0:
        raise ValueError("numerator and denominator must be coprime")
    g = poly.add(p, poly.scale(q, k))
    moments = model.moments_rademacher(2 * max(len(g), 2) - 1)
    return model.build((0, 1), g, moments)


def chebyshev_coefficients(degree: int) -> tuple[Fraction, ...]:
    """T_degree in ascending powers, using T_(n+1)=2z*T_n-T_(n-1)."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    previous, current = poly.poly((1,)), poly.poly((0, 1))
    if degree == 0:
        return previous
    for _ in range(1, degree):
        previous, current = current, poly.sub(
            poly.scale(poly.mul((Fraction(0), Fraction(1)), current),
                       Fraction(2)), previous)
    return current


def chebyshev_response_model(degree: int, gain=1) -> model.Model:
    """Odd-degree family with exactly equal finite backbone minima."""
    if isinstance(degree, bool) or not isinstance(degree, int) \
            or degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    return rational_response_model(chebyshev_coefficients(degree), gain=gain)


def gaussian_smoothed_moments(count: int, variance) -> tuple[Fraction, ...]:
    """Moments of 1/2 N(-1,v) + 1/2 N(+1,v), exactly for rational v.

    At v=0 this is the Rademacher measure.  Every v>0 gives a smooth,
    strictly positive density.  Only the moments needed by a finite-degree
    SPONG model are materialized.
    """
    v = poly.as_fraction(variance)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if v < 0:
        raise ValueError("variance must be nonnegative")
    out = []
    for k in range(count):
        if k % 2:
            out.append(Fraction(0))
            continue
        moment = Fraction(0)
        double_factorial = 1
        for j in range(k // 2 + 1):
            if j:
                double_factorial *= 2 * j - 1
            moment += comb(k, 2 * j) * double_factorial * v**j
        out.append(moment)
    return tuple(out)


def smoothed_chebyshev_model(degree: int, variance, gain=1) -> model.Model:
    """Chebyshev coefficients under a continuous Gaussian-mixture input law.

    Positive variance perturbs the exact equiripple backbone; it does not
    preserve the Rademacher moments or assert an unchanged global portrait.
    """
    # Validate the degree and gain using the discrete construction first.
    discrete = chebyshev_response_model(degree, gain)
    v = poly.as_fraction(variance)
    if v <= 0:
        raise ValueError("variance must be positive")
    moments = gaussian_smoothed_moments(2 * len(discrete.g) - 1, v)
    return model.build(discrete.f, discrete.g, moments)
