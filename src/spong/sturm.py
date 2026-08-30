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

from dataclasses import dataclass, replace
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


# Cache sizes.  512 was NOT thrashing -- measured on linear-target-d17-thrash,
# _native_sturm_plan sees 6823 hits against 151 misses with a working set of
# 151, and raising the bound to 4096 changed the audit by 4s in 718s.  The
# cost is the 151 misses themselves: constructing a GMP Sturm chain for the
# degree-98 and degree-136 test polynomials the far-field funnel and the
# sublevel inventory generate runs to seconds apiece.  The larger bound is
# kept because it is free and these are pure functions, but it is not a fix.
_CACHE = 4096


@lru_cache(maxsize=_CACHE)
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


@lru_cache(maxsize=_CACHE)
def _native_sturm_plan(integers: tuple[int, ...]):
    """Persistent frontend-independent exact plan, or None without C core."""
    try:
        from . import _native
    except ImportError:
        return None
    return _native.SturmPlan(integers)


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


def _count_roots_python(p: Poly, lo: Fraction | None = None,
                        hi: Fraction | None = None) -> int:
    """Fraction/bigint oracle for exact distinct-real-root counting."""
    sf = squarefree_part(p)
    ch = sturm_chain(sf)
    if not ch:
        return 0
    v_lo = variations_at(ch, lo, positive_inf=False)
    v_hi = variations_at(ch, hi, positive_inf=True)
    return v_lo - v_hi


def count_roots(p: Poly, lo: Fraction | None = None,
                hi: Fraction | None = None) -> int:
    """EXACT number of distinct real roots of p in (lo, hi].

    Production counting uses the frontend-independent GMP C core, with one
    persistent squarefree Sturm chain per primitive integer polynomial.
    The Fraction implementation remains the independent qualification oracle.
    """
    integers = P.int_primitive(P.trim(p))
    if not integers:
        return 0
    plan = _native_sturm_plan(integers)
    if plan is not None:
        if lo is None and hi is None:
            # The analysis already computed this count while certifying
            # squarefreeness/repeated roots; avoid endpoint evaluation.
            result = plan.stats()
            if result["status"] != 0:
                raise ArithmeticError(
                    f"native exact Sturm analysis refused with status "
                    f"{result['status']}")
            return int(result["distinct_real_roots"])
        return int(plan.count(lo, hi))
    return _count_roots_python(p, lo, hi)


@lru_cache(maxsize=_CACHE)
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


def _isolate_roots_python(
        p: Poly, stats: dict | None = None) -> list[RootInterval]:
    """Disjoint intervals, one distinct real root each.  EXACT.

    Rational roots hit by a bisection point are returned exactly.
    If ``stats`` is supplied, algorithmic work counters are accumulated in it.
    """
    sf = squarefree_part(p)
    ch = sturm_chain(sf)
    if not ch:
        return []
    bound = cauchy_bound(sf)
    out: list[RootInterval] = []
    if stats is not None:
        stats["chain_length"] = len(ch)
        stats["chain_coefficients"] = sum(len(q) for q in ch)
        stats["chain_peak_coefficient_bits"] = max(
            (abs(c).bit_length() for q in ch for c in q), default=0)
        stats.setdefault("variation_evaluations", 0)
        stats.setdefault("variation_signs", 0)
        stats.setdefault("subdivision_nodes", 0)
        stats.setdefault("polynomial_evaluations", 0)
        stats.setdefault("puncture_halvings", 0)
        stats.setdefault("max_subdivision_depth", 0)
        stats.setdefault("max_endpoint_bits", 0)

    def var(x: Fraction) -> int:
        if stats is not None:
            stats["variation_evaluations"] += 1
            stats["variation_signs"] += len(ch)
        return variations_at(ch, x)

    def rec(lo: Fraction, hi: Fraction, v_lo: int, v_hi: int, depth: int):
        if stats is not None:
            stats["subdivision_nodes"] += 1
            stats["max_subdivision_depth"] = max(
                stats["max_subdivision_depth"], depth)
            stats["max_endpoint_bits"] = max(
                stats["max_endpoint_bits"],
                abs(lo.numerator).bit_length(), lo.denominator.bit_length(),
                abs(hi.numerator).bit_length(), hi.denominator.bit_length())
        n = v_lo - v_hi
        if n == 0:
            return
        if n == 1:
            out.append(RootInterval(lo, hi, False))
            return
        mid = (lo + hi) / 2
        if stats is not None:
            stats["polynomial_evaluations"] += 1
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
                if stats is not None:
                    stats["puncture_halvings"] += 1
                    stats["polynomial_evaluations"] += 2
                eps /= 2
                left, right = mid - eps, mid + eps
            rec(lo, left, v_lo, var(left), depth+1)
            rec(right, hi, var(right), v_hi, depth+1)
        else:
            v_mid = var(mid)
            rec(lo, mid, v_lo, v_mid, depth+1)
            rec(mid, hi, v_mid, v_hi, depth+1)

    rec(-bound, bound, var(-bound), var(bound), 0)
    out.sort(key=lambda r: r.lo)
    if stats is not None:
        stats["isolated_roots"] = len(out)
        stats["exact_roots"] = sum(iv.exact for iv in out)
    return out


@lru_cache(maxsize=4096)
def _isolate_cached(integers: tuple[int, ...]) -> tuple[RootInterval, ...]:
    """Memoized isolation, keyed on the primitive integer polynomial.

    Isolation is a pure function of the polynomial, and the same polynomial
    recurs constantly during an audit: measured on tricky-d11, 82 isolate
    calls against roughly a dozen distinct polynomials, at ~89ms each.  The
    Sturm chain behind them is already cached; the isolation was not.
    """
    plan = _native_sturm_plan(integers)
    result = plan.isolate()
    if result["status"] != 0:
        raise ArithmeticError(
            f"native exact root isolation refused with status "
            f"{result['status']}")
    return tuple(
        RootInterval(
            Fraction(int(lo_num), int(lo_den)),
            Fraction(int(hi_num), int(hi_den)),
            bool(exact))
        for lo_num, lo_den, hi_num, hi_den, exact in result["intervals"])


def isolate_roots(p: Poly, stats: dict | None = None) -> list[RootInterval]:
    """Disjoint exact rational intervals, one distinct real root each.

    Production isolation runs in the reusable GMP C core.  The Fraction
    implementation remains available as ``_isolate_roots_python`` for
    differential qualification.
    """
    integers = P.int_primitive(P.trim(p))
    if not integers:
        return []
    plan = _native_sturm_plan(integers)
    if plan is None:
        return _isolate_roots_python(p, stats)
    if stats is None:
        # The work counters are only meaningful for the call that actually
        # did the work, so a stats request bypasses the cache rather than
        # reporting someone else's bisections as its own.
        return list(_isolate_cached(integers))
    result = plan.isolate()
    if result["status"] != 0:
        raise ArithmeticError(
            f"native exact root isolation refused with status "
            f"{result['status']}")
    out = [
        RootInterval(
            Fraction(int(lo_num), int(lo_den)),
            Fraction(int(hi_num), int(hi_den)),
            bool(exact))
        for lo_num, lo_den, hi_num, hi_den, exact in result["intervals"]
    ]
    if stats is not None:
        plan_stats = plan.stats()
        stats["chain_length"] = plan_stats["sturm_chain_length"]
        stats["chain_coefficients"] = \
            plan_stats["sturm_chain_coefficients"]
        # The construction peak includes intermediate PRS coefficients and
        # is therefore a conservative chain-coefficient bound.
        stats["chain_peak_coefficient_bits"] = \
            plan_stats["peak_coefficient_bits"]
        for key in (
                "variation_evaluations", "subdivision_nodes",
                "polynomial_evaluations", "puncture_halvings",
                "max_subdivision_depth", "max_endpoint_bits"):
            stats[key] = result[key]
        stats["variation_signs"] = \
            result["variation_evaluations"] * stats["chain_length"]
        stats["isolated_roots"] = len(out)
        stats["exact_roots"] = sum(iv.exact for iv in out)
    return out


def _refine_python(p: Poly, iv: RootInterval,
                   rel: Fraction = Fraction(1, 2**48),
                   stats: dict | None = None) -> RootInterval:
    """Bisect an isolating interval to relative width `rel`.  EXACT."""
    if iv.exact:
        return iv
    sf = squarefree_part(p)
    lo, hi = iv.lo, iv.hi
    s_lo = P.eval_at(sf, lo)
    # Sturm guarantees exactly one root; endpoints are non-roots by isolation.
    while hi - lo > rel * (1 + abs(lo) + abs(hi)):
        if stats is not None:
            stats["refinement_bisections"] = \
                stats.get("refinement_bisections", 0)+1
        mid = (lo + hi) / 2
        v = P.eval_at(sf, mid)
        if v == 0:
            return RootInterval(mid, mid, True)
        if (v > 0) == (s_lo > 0):
            lo, s_lo = mid, v
        else:
            hi = mid
    if stats is not None:
        stats["max_refined_endpoint_bits"] = max(
            stats.get("max_refined_endpoint_bits", 0),
            abs(lo.numerator).bit_length(), lo.denominator.bit_length(),
            abs(hi.numerator).bit_length(), hi.denominator.bit_length())
    return RootInterval(lo, hi, False)


@lru_cache(maxsize=8192)
def _refine_cached(integers: tuple[int, ...], lo: Fraction, hi: Fraction,
                   rel: Fraction) -> RootInterval:
    """Memoized refinement of one isolating interval.

    Refinement is deterministic in (polynomial, interval, target width) and is
    repeated heavily: 194 calls at ~50ms on tricky-d11, against roughly a
    dozen roots.  Each call bisects from scratch, so the repetition is pure
    waste -- the narrowed interval for a given root is the same object every
    time it is asked for.
    """
    plan = _native_sturm_plan(integers)
    result = plan.refine(lo, hi, rel)
    if result["status"] != 0:
        raise ArithmeticError(
            f"native exact root refinement refused with status "
            f"{result['status']}")
    lo_num, lo_den, hi_num, hi_den, exact = result["interval"]
    return RootInterval(
        Fraction(int(lo_num), int(lo_den)),
        Fraction(int(hi_num), int(hi_den)),
        bool(exact))


def refine(p: Poly, iv: RootInterval,
           rel: Fraction = Fraction(1, 2**48),
           stats: dict | None = None) -> RootInterval:
    """Refine an isolating interval using the persistent exact C plan."""
    if iv.exact:
        return iv
    integers = P.int_primitive(P.trim(p))
    plan = _native_sturm_plan(integers) if integers else None
    if plan is None:
        return _refine_python(p, iv, rel, stats)
    if stats is None:
        return _refine_cached(integers, iv.lo, iv.hi, rel)
    result = plan.refine(iv.lo, iv.hi, rel)
    if result["status"] != 0:
        raise ArithmeticError(
            f"native exact root refinement refused with status "
            f"{result['status']}")
    lo_num, lo_den, hi_num, hi_den, exact = result["interval"]
    refined = RootInterval(
        Fraction(int(lo_num), int(lo_den)),
        Fraction(int(hi_num), int(hi_den)),
        bool(exact))
    if stats is not None:
        stats["refinement_bisections"] = \
            stats.get("refinement_bisections", 0) + result["bisections"]
        stats["max_refined_endpoint_bits"] = max(
            stats.get("max_refined_endpoint_bits", 0),
            result["max_endpoint_bits"])
    return refined


def interval_sign(p: Poly, iv: RootInterval) -> int | None:
    """Certified sign of p on the interval: requires no roots of p inside.

    Returns +1/-1 (EXACT), or None if p vanishes inside (caller must refine
    the interval or declare degeneracy).
    """
    integers = P.int_primitive(P.trim(p))
    if not integers:
        return 0 if iv.exact else None
    plan = _native_sturm_plan(integers)
    if plan is None:
        if iv.exact:
            v = P.eval_at(p, iv.lo)
            return (v > 0) - (v < 0)
        if _count_roots_python(p, iv.lo, iv.hi) != 0:
            return None
        v = P.eval_at(p, iv.mid)
        return None if v == 0 else (1 if v > 0 else -1)
    if iv.exact:
        return int(plan.sign_at(iv.lo))
    if plan.count(iv.lo, iv.hi) != 0:
        return None
    sign = int(plan.sign_at(iv.mid))
    return None if sign == 0 else sign


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
    source: str                  # 'N' | 'B' | 'H' (reduced u' numerator)
    interval: RootInterval       # EXACT isolating interval for b
    u2_sign: int                 # sign of u'' at the root (0 if degenerate)
    local: object | None = None  # conditioned finite jet (post-enumeration)
    stubs: tuple = ()             # certified physical invariant-manifold stubs


@dataclass(frozen=True)
class Enumeration:
    points: tuple[CriticalPoint, ...]
    psi_positive: bool           # EXACT
    morse: bool                  # EXACT: u' has no multiple REAL root
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
    H = m.critical_reduced
    repeated_H = P.gcd_poly(H, P.deriv(H))
    has_repeated_real = (
        P.degree(repeated_H) > 0 and count_roots(repeated_H) > 0)
    morse = bool(H) and not has_repeated_real
    # B and N are the cheapest exact factorization in the generic case.
    # Algebraically repeated/common COMPLEX factors do not affect real Morse
    # structure; when present, enumerate the reduced numerator of u' instead.
    use_factorized = not has_common and B_squarefree and N_squarefree

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

    if use_factorized:
        BNp = P.mul(B, P.deriv(N))       # sign of u'' at N-roots
        BpN = P.mul(P.deriv(B), N)       # sign of u'' at B-roots
        for iv in isolate_roots(N):
            classify(iv, "N", BNp)
        for iv in isolate_roots(B):
            classify(iv, "B", BpN)
    else:
        Hp = P.deriv(H)
        for iv in isolate_roots(H):
            cur = refine(H, iv)
            repeated_here = (
                P.degree(repeated_H) > 0
                and ((cur.exact and P.eval_at(repeated_H, cur.lo) == 0)
                     or (not cur.exact
                         and count_roots(
                             repeated_H, cur.lo, cur.hi) > 0)))
            s = None if repeated_here else interval_sign(Hp, cur)
            for _ in range(64):
                if s is not None or repeated_here or cur.exact:
                    break
                cur = refine(
                    H, cur,
                    rel=(cur.hi-cur.lo)/(1+abs(cur.mid))/4)
                s = interval_sign(Hp, cur)
            bf = float(cur.mid)
            af = float(
                P.eval_at(B, cur.mid)/P.eval_at(m.alpha, cur.mid))
            if repeated_here or s is None:
                kind, s2 = "degenerate", 0
            else:
                kind, s2 = ("min" if s > 0 else "saddle"), s
            pts.append(CriticalPoint(
                bf, af, kind, "H", cur, s2))

    pts.sort(key=lambda p: (p.interval.lo, p.interval.hi))

    # Only after the exact zero-dimensional skeleton is complete do we
    # elaborate each point numerically.  Neighbor distances used to size
    # local normal-form candidates are therefore known and immutable.
    from .local import build_local_jet
    centers = [p.b for p in pts]
    elaborated = []
    for i, p in enumerate(pts):
        if p.kind == "degenerate":
            elaborated.append(p)
            continue
        local = build_local_jet(
            m, p.interval, p.source,
            tuple(x for j, x in enumerate(centers) if j != i),
            root_poly=H if p.source == "H" else None)
        elaborated.append(replace(p, b=local.b, a=local.a, local=local))
    pts = elaborated

    # Alternation invariant (Theorem 2): the signs of u'' alternate along
    # the complete ordered critical set of the one-dimensional Morse
    # function u.  Planar L-types have the same signs.  B-roots are always
    # saddles, so their finite critical neighbors, when present, are minima.
    # EXACT.
    signs = [p.u2_sign for p in pts if p.kind != "degenerate"]
    alternates = all(signs[i] * signs[i + 1] < 0
                     for i in range(len(signs) - 1))

    return Enumeration(tuple(pts), psi_ok, morse, alternates)


def materialize_stubs(m: Model, e: Enumeration) -> Enumeration:
    """Elaborate saddles into four certified, reusable physical stubs."""
    from .local import build_stubs
    points = tuple(
        replace(p, stubs=build_stubs(m, p, e.minima))
        if p.kind == "saddle" else p
        for p in e.points)
    return replace(e, points=points)


def materialize_validated_launches(m: Model, e: Enumeration) -> Enumeration:
    """Attach exact-rational local launch boxes to every saddle stub.

    This is currently the independent Python/Fraction oracle and is kept
    separate from :func:`materialize_stubs` so ordinary portrait timing still
    measures the established floating graph transform.  The later GMP C
    kernel will make this certificate cheap enough to join the default path.
    """
    from .local_certificate import certify_poincare_launch

    if any(point.kind == "saddle" and not point.stubs for point in e.points):
        e = materialize_stubs(m, e)
    points = []
    for point in e.points:
        if point.kind != "saddle":
            points.append(point)
            continue
        charts = {chart.manifold: chart for chart in point.local.poincare}
        launches = []
        for stub in point.stubs:
            chart = charts[stub.manifold]
            launch = certify_poincare_launch(
                m, point, chart, stub.orientation)
            launches.append(replace(stub, validated_launch=launch))
        points.append(replace(point, stubs=tuple(launches)))
    return replace(e, points=tuple(points))
