"""Certified enumeration — EXACT Sturm machinery and the critical-point set.

SPONG_FOUNDING Part II, section 4.  Companion-matrix roots are banned.

All counting, isolation, positivity and squarefree decisions here are EXACT:
rational arithmetic over the dyadic inputs (stdlib Fraction / bigint).  There
is no overflow path — Fractions do not overflow — so the far-root NaN
pathology of the MATLAB predecessor cannot occur by construction.  Floating
point appears only in the cosmetic float fields of results (RESIDUAL).

Critical set of the model (model.py):  u' = B·N/A², so the critical b-values
are the real roots of B UNION the real roots of N.  Classification at a
simple root:  u'' = (B'N + BN')/A² there, so

    root of N:  sign(u'') = sign(B·N')     (evaluated on the interval)
    root of B:  sign(u'') = sign(B'·N)

with u'' > 0 ⇒ minimum of L, u'' < 0 ⇒ saddle (det H = 2A·u''; H11 > 0).
Interval signs are certified: the interval is refined until the sign
polynomial has Sturm count zero inside, then its sign at the midpoint is its
sign at the root — EXACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from . import _poly as P
from ._poly import Poly
from .model import Model

# --------------------------------------------------------------------- #
# Sturm chains (exact)                                                   #
# --------------------------------------------------------------------- #


from functools import lru_cache  # noqa: E402


def _as_poly(ints: tuple[int, ...]) -> Poly:
    return tuple(Fraction(v) for v in ints)


@lru_cache(maxsize=512)
def sturm_chain(p: Poly) -> tuple[tuple[int, ...], ...]:
    """Sturm chain of p as primitive integer polynomials.

    Primitive PRS: each remainder is immediately rescaled by a POSITIVE
    rational to primitive integer form.  Positive per-element scaling
    preserves every sign Sturm theory uses, and bounds the coefficient
    growth that makes naive Fraction Euclid blow up at degree ~30+.
    """
    a = _as_poly(P.int_primitive(P.trim(p)))
    if not a:
        return ()
    chain = [P.int_primitive(a)]
    d = P.deriv(a)
    b = _as_poly(P.int_primitive(d)) if d else ()
    while b:
        chain.append(P.int_primitive(b))
        _, r = P.divmod_exact(a, b)
        a = b
        b = _as_poly(P.int_primitive(P.scale(r, Fraction(-1)))) if r else ()
    return tuple(chain)


def _sign_int(p: tuple[int, ...], x: Fraction) -> int:
    """Exact sign of an integer polynomial at a rational point (bigint).

    sign(p(n/d)) = sign(p(n/d)·d^deg) = sign(Horner over Z), d > 0.
    """
    if not p:
        return 0
    num, den = x.numerator, x.denominator
    deg = len(p) - 1
    acc = p[-1]
    for k in range(deg - 1, -1, -1):
        acc = acc * num + p[k] * den ** (deg - k)
    return (acc > 0) - (acc < 0)


def _variations(signs: list[int]) -> int:
    s = [v for v in signs if v != 0]
    return sum(1 for i in range(len(s) - 1) if s[i] != s[i + 1])


def _sign_at_inf(p: tuple[int, ...], positive: bool) -> int:
    if not p:
        return 0
    lc = p[-1]
    s = (lc > 0) - (lc < 0)
    if positive:
        return s
    return s if (len(p) - 1) % 2 == 0 else -s


def variations_at(chain, x: Fraction | None, positive_inf: bool = True) -> int:
    """Sign-variation count at a rational point, or at ±infinity (x=None)."""
    if x is None:
        signs = [_sign_at_inf(p, positive_inf) for p in chain]
    else:
        signs = [_sign_int(p, x) for p in chain]
    return _variations(signs)


def count_roots(p: Poly, lo: Fraction | None = None,
                hi: Fraction | None = None) -> int:
    """EXACT number of distinct real roots of p in (lo, hi].

    None bounds mean -infinity / +infinity.  For counting purposes p is
    replaced by its squarefree part, so multiple roots count once.
    """
    sf = squarefree_part(p)
    ch = sturm_chain(sf)
    if not ch:
        return 0
    v_lo = variations_at(ch, lo, positive_inf=False)
    v_hi = variations_at(ch, hi, positive_inf=True)
    return v_lo - v_hi


@lru_cache(maxsize=512)
def squarefree_part(p: Poly) -> Poly:
    g = P.gcd_poly(p, P.deriv(p))
    if P.degree(g) <= 0:
        return P.trim(p)
    q, _ = P.divmod_exact(p, g)
    return q


def is_squarefree(p: Poly) -> bool:
    """EXACT: gcd(p, p') is constant."""
    return P.degree(P.gcd_poly(p, P.deriv(p))) <= 0


def is_positive(p: Poly) -> bool:
    """EXACT: p(b) > 0 for all real b (no real roots + positive sample)."""
    if not p:
        return False
    return count_roots(p) == 0 and P.eval_at(p, Fraction(0)) > 0


def cauchy_bound(p: Poly) -> Fraction:
    """All real roots have |b| < bound (exact rational)."""
    q = P.trim(p)
    lc = abs(q[-1])
    m = max((abs(c) for c in q[:-1]), default=Fraction(0))
    return 1 + m / lc


@dataclass(frozen=True)
class RootInterval:
    lo: Fraction
    hi: Fraction               # lo == hi means an exact rational root
    exact: bool

    @property
    def mid(self) -> Fraction:
        return (self.lo + self.hi) / 2


def isolate_roots(p: Poly) -> list[RootInterval]:
    """Disjoint intervals, one distinct real root each.  EXACT.

    Rational roots hit by a bisection point are returned exactly.
    """
    sf = squarefree_part(p)
    ch = sturm_chain(sf)
    if not ch:
        return []
    bound = cauchy_bound(sf)
    out: list[RootInterval] = []

    def var(x: Fraction) -> int:
        return variations_at(ch, x)

    def rec(lo: Fraction, hi: Fraction, v_lo: int, v_hi: int):
        n = v_lo - v_hi
        if n == 0:
            return
        if n == 1:
            out.append(RootInterval(lo, hi, False))
            return
        mid = (lo + hi) / 2
        if P.eval_at(sf, mid) == 0:
            out.append(RootInterval(mid, mid, True))
            # Punctured split around the exact root.  The puncture must be
            # CERTIFIED to contain only that root: shrink eps until the
            # Sturm count across (left, right] is exactly 1 — otherwise a
            # nearby root inside the puncture is silently lost (found the
            # hard way: an exact root at 0 with a companion at ~2e-6,
            # caught by the index-balance winding certificate).
            eps = (hi - lo) / 2**20
            left, right = mid - eps, mid + eps
            while (P.eval_at(sf, left) == 0 or P.eval_at(sf, right) == 0
                   or var(left) - var(right) != 1):
                eps /= 2
                left, right = mid - eps, mid + eps
            rec(lo, left, v_lo, var(left))
            rec(right, hi, var(right), v_hi)
        else:
            v_mid = var(mid)
            rec(lo, mid, v_lo, v_mid)
            rec(mid, hi, v_mid, v_hi)

    rec(-bound, bound, var(-bound), var(bound))
    out.sort(key=lambda r: r.lo)
    return out


def refine(p: Poly, iv: RootInterval, rel: Fraction = Fraction(1, 2**48)
           ) -> RootInterval:
    """Bisect an isolating interval to relative width `rel`.  EXACT."""
    if iv.exact:
        return iv
    sf = squarefree_part(p)
    lo, hi = iv.lo, iv.hi
    s_lo = P.eval_at(sf, lo)
    # Sturm guarantees exactly one root; endpoints are non-roots by isolation.
    while hi - lo > rel * (1 + abs(lo) + abs(hi)):
        mid = (lo + hi) / 2
        v = P.eval_at(sf, mid)
        if v == 0:
            return RootInterval(mid, mid, True)
        if (v > 0) == (s_lo > 0):
            lo, s_lo = mid, v
        else:
            hi = mid
    return RootInterval(lo, hi, False)


def interval_sign(p: Poly, iv: RootInterval) -> int | None:
    """Certified sign of p on the interval: requires no roots of p inside.

    Returns +1/-1 (EXACT), or None if p vanishes inside (caller must refine
    the interval or declare degeneracy).
    """
    if iv.exact:
        v = P.eval_at(p, iv.lo)
        return (v > 0) - (v < 0)
    if count_roots(p, iv.lo, iv.hi) != 0:
        return None
    v = P.eval_at(p, iv.mid)
    if v == 0:   # can only happen at a root of p — excluded above for p != 0
        return None
    return 1 if v > 0 else -1


def _simple_root_derivative_sign(p: Poly, iv: RootInterval) -> int:
    """Sign of p' at the unique simple root isolated by iv.

    For a non-exact isolating interval with one simple root, the sign of p
    immediately to the right of the root is the sign of p' at the root.  The
    right endpoint is root-free by construction/refinement.
    """
    if iv.exact:
        v = P.eval_at(P.deriv(p), iv.lo)
    else:
        v = P.eval_at(p, iv.hi)
    return (v > 0) - (v < 0)


# --------------------------------------------------------------------- #
# Critical-point enumeration for a Model                                 #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CriticalPoint:
    b: float                     # float midpoint (cosmetic; RESIDUAL)
    a: float
    kind: str                    # 'min' | 'saddle' | 'degenerate'
    source: str                  # 'N' | 'B' | 'both'
    interval: RootInterval       # EXACT isolating interval for b
    u2_sign: int                 # sign of u'' at the root (0 if degenerate)


@dataclass(frozen=True)
class Enumeration:
    points: tuple[CriticalPoint, ...]
    psi_positive: bool           # EXACT
    morse: bool                  # EXACT: B·N squarefree and no common roots
    alternates: bool             # EXACT (trivially true if not Morse: skipped)

    @property
    def minima(self):
        return tuple(p for p in self.points if p.kind == "min")

    @property
    def saddles(self):
        return tuple(p for p in self.points if p.kind == "saddle")


def enumerate_critical_points(m: Model) -> Enumeration:
    """The certified critical-point set of L.  See module docstring."""
    psi_ok = is_positive(m.alpha)

    B, N = m.beta, m.N
    common = P.gcd_poly(B, N)
    has_common = P.degree(common) > 0
    B_squarefree = is_squarefree(B)
    N_squarefree = is_squarefree(N)
    morse = (not has_common) and B_squarefree and N_squarefree \
        and P.degree(B) >= 0

    pts: list[CriticalPoint] = []

    def classify(iv: RootInterval, source: str, signpoly: Poly):
        # refine until the sign polynomial is certified on the interval
        cur = refine(P.mul(B, N) if source == "both" else
                     (N if source == "N" else B), iv)
        s = None
        if source == "N" and not has_common and N_squarefree:
            # At a simple N-root, sign(u'') = sign(B) * sign(N').  This is
            # the same theorem used by BN', but avoids constructing and
            # Sturm-counting the much higher-degree product polynomial.
            s_B = interval_sign(B, cur)
            s_Np = _simple_root_derivative_sign(N, cur)
            if s_B is not None and s_Np != 0:
                s = s_B * s_Np
        elif source == "B" and not has_common and B_squarefree:
            # At a simple B-root, N(b0) = -2B'(b0)A(b0), hence
            # u'' = B'N/A^2 = -2B'^2/A < 0.  The exact product sign path
            # remains below for non-Morse/degenerate cases.
            s = -1
        if s is None:
            s = interval_sign(signpoly, cur)
            for _ in range(64):
                if s is not None:
                    break
                cur = refine(N if source == "N" else B, cur,
                             rel=(cur.hi - cur.lo) / (1 + abs(cur.mid)) / 4
                             if not cur.exact else Fraction(1))
                s = interval_sign(signpoly, cur)
        bf = float(cur.mid)
        af = float(P.eval_at(m.beta, cur.mid) / P.eval_at(m.alpha, cur.mid))
        if s is None or source == "both":
            kind, s2 = "degenerate", 0
        elif source == "B":
            # At a simple B-root, N(b0) = -2B'A, so u'' = B'N/A^2
            # = -2B'^2/A < 0 AUTOMATICALLY: every simple B-root is a
            # saddle by the same universal identity det H = 2A u''
            # (= -4B'^2 here).  Forcing 'saddle' is a simplification of
            # the sign evaluation, not an exception to the theorem.
            kind, s2 = "saddle", s
        else:
            kind, s2 = ("min" if s > 0 else "saddle"), s
        pts.append(CriticalPoint(bf, af, kind, source, cur, s2))

    BNp = P.mul(B, P.deriv(N))       # sign of u'' at N-roots
    BpN = P.mul(P.deriv(B), N)       # sign of u'' at B-roots

    if has_common:
        for iv in isolate_roots(common):
            classify(iv, "both", P.mul(B, N))

    for iv in isolate_roots(N):
        if has_common and count_roots(common, iv.lo, iv.hi) > 0:
            continue                     # already reported as degenerate
        if iv.exact and P.eval_at(common, iv.lo) == 0:
            continue
        classify(iv, "N", BNp)

    for iv in isolate_roots(B):
        if has_common and count_roots(common, iv.lo, iv.hi) > 0:
            continue
        if iv.exact and P.eval_at(common, iv.lo) == 0:
            continue
        classify(iv, "B", BpN)

    pts.sort(key=lambda p: (p.interval.lo, p.interval.hi))

    # Alternation invariant (Theorem 2, corrected): the 1D Morse
    # alternation is a statement about u — the SIGNS of u'' alternate
    # along b.  L-types follow u at N-roots; B-roots are always 2D
    # saddles, so consecutive L-saddles can legitimately occur.  EXACT.
    signs = [p.u2_sign for p in pts if p.kind != "degenerate"]
    alternates = all(signs[i] * signs[i + 1] < 0
                     for i in range(len(signs) - 1))

    return Enumeration(tuple(pts), psi_ok, morse, alternates)
