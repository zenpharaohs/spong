"""Exact univariate polynomial arithmetic over the rationals.

Polynomials are tuples of ``fractions.Fraction`` in ASCENDING power order.
IEEE floats are dyadic rationals, so ``Fraction(x)`` converts model inputs
exactly: everything here is EXACT over the model as given (SPONG_FOUNDING,
certificate semantics).  Private helper module; public API is spong.model
and spong.sturm.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

Poly = tuple[Fraction, ...]

ZERO: Poly = ()


def as_fraction(x) -> Fraction:
    """Exact conversion; floats convert via their dyadic value."""
    return x if isinstance(x, Fraction) else Fraction(x)


def poly(coeffs) -> Poly:
    """Ascending-order polynomial from any iterable of numbers, trimmed."""
    return trim(tuple(as_fraction(c) for c in coeffs))


def trim(p: Poly) -> Poly:
    n = len(p)
    while n and p[n - 1] == 0:
        n -= 1
    return tuple(p[:n])


def degree(p: Poly) -> int:
    """Degree; -1 for the zero polynomial."""
    return len(p) - 1


def add(p: Poly, q: Poly) -> Poly:
    n = max(len(p), len(q))
    return trim(tuple(
        (p[k] if k < len(p) else 0) + (q[k] if k < len(q) else 0)
        for k in range(n)))


def sub(p: Poly, q: Poly) -> Poly:
    return add(p, scale(q, Fraction(-1)))


def scale(p: Poly, c: Fraction) -> Poly:
    if c == 0:
        return ZERO
    return tuple(ck * c for ck in p)


def mul(p: Poly, q: Poly) -> Poly:
    if not p or not q:
        return ZERO
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi:
            for j, qj in enumerate(q):
                out[i + j] += pi * qj
    return trim(tuple(out))


def deriv(p: Poly) -> Poly:
    return trim(tuple(k * p[k] for k in range(1, len(p))))


def eval_at(p: Poly, x: Fraction) -> Fraction:
    """Exact Horner evaluation at a rational point."""
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def divmod_exact(f: Poly, g: Poly) -> tuple[Poly, Poly]:
    """Exact (quotient, remainder) over the rationals; g != 0."""
    if not g:
        raise ZeroDivisionError("polynomial division by zero")
    r = list(f)
    q = [Fraction(0)] * max(len(f) - len(g) + 1, 1)
    dg, lg = degree(g), g[-1]
    while len(r) - 1 >= dg and any(r):
        while r and r[-1] == 0:
            r.pop()
        if len(r) - 1 < dg:
            break
        c = r[-1] / lg
        k = len(r) - 1 - dg
        q[k] = c
        for j in range(len(g)):
            r[k + j] -= c * g[j]
        r.pop()
    return trim(tuple(q)), trim(tuple(r))


def int_primitive(p: Poly) -> tuple[int, ...]:
    """Primitive integer polynomial = p times a POSITIVE rational.

    Positive scaling preserves every sign Sturm theory looks at.
    """
    if not p:
        return ()
    den = 1
    for c in p:
        den = den * c.denominator // gcd(den, c.denominator)
    ints = [int(c * den) for c in p]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    g = g or 1
    return tuple(v // g for v in ints)


def gcd_poly(p: Poly, q: Poly) -> Poly:
    """Monic gcd over the rationals (exact Euclid, primitive PRS).

    Each remainder is rescaled to primitive integer form: gcd is only
    defined up to a constant, and the rescaling bounds coefficient growth.
    """
    def prim(r: Poly) -> Poly:
        return tuple(Fraction(v) for v in int_primitive(r))

    a, b = prim(trim(p)), prim(trim(q))
    while b:
        _, r = divmod_exact(a, b)
        a, b = b, prim(r)
    if not a:
        return ZERO
    return scale(a, Fraction(1) / a[-1])
