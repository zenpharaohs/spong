"""Poincaré compactification: rim structure, index bookkeeping, box contract.

SPONG_FOUNDING Part II, sections 8 and 8b.  Under the genericity
conditions (effective degree d_eff = deg g as realized; leading moment
positive), the leading form of −∇L places the equatorial equilibria at
the axes and the four diagonals b = ±√d_eff·a; the diagonals are the
asymptote directions of every separatrix.  The degenerate backbone poles
carry the local model ḃ ≈ −C_inf/b², with C_inf an EXACT rational
coefficient ratio (the leading term of N cancels — the Wronskian
cancellation — so deg(B·N) = deg(A²) − 2 and the limit exists).

Global completeness cross-check (Poincaré–Hopf in practice): the winding
number of ∇L around a circle enclosing all finite critical points equals
n_min − n_saddle.  A mismatch proves the enumeration or the trace wrong —
the instrument says so itself.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import _poly as P
from .gauss import richardson3
from .model import Model


# --------------------------------------------------------------------- #
# genericity and rim structure                                           #
# --------------------------------------------------------------------- #


def effective_degree(m: Model) -> int:
    """deg g as realized (EXACT: trailing zero coefficients dropped)."""
    return P.degree(m.g)


def genericity(m: Model) -> dict:
    """§8 genericity conditions, certified EXACT.

    Requires mu_{2 d_eff} > 0 (automatic for a nondegenerate Hamburger
    sequence) so the leading coefficient alpha_{2 d_eff} = g_d² mu_{2d}
    is positive.
    """
    d_eff = effective_degree(m)
    mu_lead = m.mu[2 * d_eff]
    alpha_lead = m.alpha[-1] if m.alpha else Fraction(0)
    return {
        "d_eff": d_eff,
        "mu_lead_positive": mu_lead > 0,
        "alpha_lead_positive": alpha_lead > 0,
        "generic": mu_lead > 0 and alpha_lead > 0,
    }


def rim_directions(m: Model) -> dict:
    """Equatorial equilibria of the leading form (generic case).

    Tangential field ∝ a·b^(2d−1)·(d·a² − b²): zeros at the axes and at
    the four diagonals b = ±√d_eff·a.  The diagonals are hyperbolic-type
    (separatrix asymptotes); the b-poles are degenerate (backbone exit,
    local model ḃ ≈ −C_inf/b²); the a-axis pair resolves at next order.
    """
    d_eff = effective_degree(m)
    s = float(np.sqrt(d_eff))
    return {
        "diagonal_slopes": (s, -s),        # b/a of the separatrix asymptotes
        "b_poles_degenerate": True,
        "C_inf": C_inf(m),
    }


def C_inf(m: Model) -> Fraction:
    """EXACT: the backbone-pole constant, ḃ ≈ −C_inf/b² as |b| → ∞.

    u' = B·N/A²; the leading coefficient of N cancels identically
    (Wronskian cancellation), so deg(B·N) ≤ deg(A²) − 2 and
    C_inf = [B·N]_(deg A² − 2) / [A²]_(deg A²) as exact rationals.
    """
    BN = P.mul(m.beta, m.N)
    A2 = P.mul(m.alpha, m.alpha)
    k = P.degree(A2) - 2
    num = BN[k] if 0 <= k < len(BN) else Fraction(0)
    return num / A2[-1]


def asymptote_certificate(m: Model, Y: np.ndarray) -> dict:
    """Separatrix exit vs the diagonal asymptote: RESIDUAL certificate.

    Takes three geometrically spaced radius samples from the tail of the
    polyline, Aitken-extrapolates the slope |b/a| in the house style
    (`richardson3`), and compares with √d_eff.
    """
    d_eff = effective_degree(m)
    target = float(np.sqrt(d_eff))
    r = np.hypot(Y[:, 0], Y[:, 1])
    r_max = r[-1]
    idx = [int(np.argmin(np.abs(r - r_max / 4))),
           int(np.argmin(np.abs(r - r_max / 2))),
           len(Y) - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = [float(abs(Y[i, 1] / Y[i, 0])) for i in idx]
    extrap = float(richardson3(np.array([slopes[2]]), np.array([slopes[1]]),
                               np.array([slopes[0]]))[0])
    return {
        "slope_samples": slopes,
        "slope_extrapolated": extrap,
        "target": target,
        "residual": abs(extrap - target) / target,
        "radii": [float(r[i]) for i in idx],
    }


# --------------------------------------------------------------------- #
# index bookkeeping (global completeness cross-check)                    #
# --------------------------------------------------------------------- #


def winding_number(m: Model, radius: float, tol_angle: float = 1.0) -> int:
    """Winding of ∇L around the circle of given radius.

    Argument-principle adaptive bisection: the direction field rotates
    through narrow transition bands at large radius (the leading form
    vanishes to order 2d−1 at the axes), so uniform sampling cannot see
    it — segments are split exactly where the principal angle jump
    exceeds tol_angle, at logarithmic cost.  Sum of indices of the
    enclosed equilibria = n_min − n_saddle for a gradient field.
    """
    def angle(t):
        a = radius * np.cos(t)
        b = radius * np.sin(t)
        ga = 2.0 * (a * m.A(b) - m.B(b))
        gb = -2.0 * a * m.Bp(b) + a**2 * m.Ap(b)
        return float(np.arctan2(gb, ga))

    def wrap(d):
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    n0 = 64
    ts = np.linspace(0.0, 2.0 * np.pi, n0 + 1)
    total = 0.0
    stack = [(ts[i], angle(ts[i]), ts[i + 1], angle(ts[i + 1]))
             for i in range(n0)]
    while stack:
        t1, th1, t2, th2 = stack.pop()
        d = wrap(th2 - th1)
        if (t2 - t1) < 1e-13:
            total += d
            continue
        # midpoint-verified acceptance: a wrapped difference near zero can
        # hide a full 2π turn; accept only if the two half-segments sum
        # consistently to the whole
        tm = 0.5 * (t1 + t2)
        thm = angle(tm)
        d_halves = wrap(thm - th1) + wrap(th2 - thm)
        if abs(d) < tol_angle and abs(d_halves - d) < 1e-9:
            total += d
            continue
        stack.append((t1, th1, tm, thm))
        stack.append((tm, thm, t2, th2))
    return int(np.round(total / (2.0 * np.pi)))




def winding_number_exact(m: Model, a_bound: Fraction, b_bound: Fraction) -> int:
    """EXACT winding of ∇L around the rectangle [±a_bound]×[±b_bound]."""
    return winding_number_exact_box(m, -a_bound, a_bound, -b_bound, b_bound)


def winding_number_exact_box(m: Model, a_lo: Fraction, a_hi: Fraction,
                             b_lo: Fraction, b_hi: Fraction) -> int:
    """EXACT winding of ∇L around an arbitrary rational rectangle.

    Kronecker index by axis crossings: winding = ½ Σ sign(Δg_b)·sign(g_a)
    over the boundary zeros of g_b, traversed CCW.  On each edge both
    components are univariate polynomials with rational coefficients, so
    every zero is Sturm-isolated and every sign is certified.  Winding is
    additive over box subdivisions — the localization tool for hunting
    bookkeeping discrepancies.
    """
    from . import sturm as st

    alpha, beta = m.alpha, m.beta
    alpha_p, beta_p = P.deriv(alpha), P.deriv(beta)

    def gb_corner(a0, b0):
        return -2 * a0 * P.eval_at(beta_p, b0) \
            + a0 * a0 * P.eval_at(alpha_p, b0)

    guard = 0
    while any(gb_corner(a0, b0) == 0
              for a0 in (a_lo, a_hi) for b0 in (b_lo, b_hi)):
        a_lo -= Fraction(1, 7); a_hi += Fraction(1, 7)
        b_lo -= Fraction(1, 13); b_hi += Fraction(1, 13)
        guard += 1
        if guard > 50:
            raise RuntimeError("could not clear corners")

    total2 = 0

    def scan_edge(gb_poly, ga_poly, lo, hi, orient):
        nonlocal total2
        if P.degree(gb_poly) < 1:
            return
        for iv in st.isolate_roots(gb_poly):
            r = st.refine(gb_poly, iv)
            pos = r.lo if r.exact else r.mid
            if not (lo < pos < hi):
                continue
            # Near the b-poles the zeros of g_a and g_b cluster at scales
            # like 1e-15 and below (the degenerate rim structure casting
            # its shadow at finite radius): every probe window must be
            # CERTIFIED by a Sturm count, and sign refinement must be
            # allowed to bisect as deep as the exact layer can go.
            cur = r
            s_ga = st.interval_sign(ga_poly, cur)
            for _ in range(512):
                if s_ga is not None or cur.exact:
                    break
                w_cur = cur.hi - cur.lo
                cur = st.refine(gb_poly, cur,
                                rel=w_cur / (2 * (1 + abs(cur.mid))))
                s_ga = st.interval_sign(ga_poly, cur)
            if s_ga is None and cur.exact:
                v = P.eval_at(ga_poly, cur.lo)
                s_ga = (v > 0) - (v < 0)
            if not s_ga:
                raise RuntimeError("g_a sign undetermined at a g_b zero "
                                   "(critical point on the boundary?)")
            if cur.exact:
                eps = Fraction(1, 2**40) * (1 + abs(cur.lo))
                while (st.count_roots(gb_poly, cur.lo - eps,
                                      cur.lo + eps) != 1
                       or P.eval_at(gb_poly, cur.lo - eps) == 0
                       or P.eval_at(gb_poly, cur.lo + eps) == 0):
                    eps /= 4
                s_lo = P.eval_at(gb_poly, cur.lo - eps)
                s_hi = P.eval_at(gb_poly, cur.lo + eps)
            else:
                s_lo = P.eval_at(gb_poly, cur.lo)
                s_hi = P.eval_at(gb_poly, cur.hi)
            d = ((s_hi > 0) - (s_hi < 0)) - ((s_lo > 0) - (s_lo < 0))
            if d == 0:
                continue
            total2 += orient * (1 if d > 0 else -1) * s_ga

    two = Fraction(2)

    def vertical(a0, orient):
        gb = P.add(P.scale(beta_p, -two * a0), P.scale(alpha_p, a0 * a0))
        ga = P.sub(P.scale(alpha, two * a0), P.scale(beta, two))
        scan_edge(gb, ga, b_lo, b_hi, orient)

    def horizontal(b0, orient):
        bp = P.eval_at(beta_p, b0)
        ap_ = P.eval_at(alpha_p, b0)
        Av, Bv = P.eval_at(alpha, b0), P.eval_at(beta, b0)
        gb = P.poly([0, -2 * bp, ap_])
        ga = P.poly([-2 * Bv, 2 * Av])
        scan_edge(gb, ga, a_lo, a_hi, orient)

    vertical(a_hi, +1)      # right edge, b increasing
    horizontal(b_hi, -1)    # top edge, a decreasing
    vertical(a_lo, -1)      # left edge, b decreasing
    horizontal(b_lo, +1)    # bottom edge, a increasing

    if total2 % 2 != 0:
        raise RuntimeError("odd crossing sum: boundary handling error")
    return total2 // 2


def index_balance(m: Model, enumeration) -> dict:
    """Poincaré–Hopf in practice: winding == n_min − n_saddle.

    The circle radius is chosen beyond every finite critical point
    (enumeration intervals are certified, so this is sound).  A mismatch
    is a self-diagnosis: the enumeration missed something or an index is
    misclassified.
    """
    pts = enumeration.points
    a_max = max((abs(p.a) for p in pts), default=1.0)
    b_max = max((abs(p.b) for p in pts), default=1.0)
    a_bound = Fraction(int(np.ceil(2.0 * a_max + 1.0)))
    b_bound = Fraction(int(np.ceil(2.0 * b_max + 1.0)))
    w = winding_number_exact(m, a_bound, b_bound)
    n_min = len(enumeration.minima)
    n_sad = len(enumeration.saddles)
    return {
        "winding": w,
        "n_min": n_min,
        "n_saddle": n_sad,
        "expected": n_min - n_sad,
        "balanced": w == n_min - n_sad,
        "a_bound": float(a_bound),
        "b_bound": float(b_bound),
    }


# --------------------------------------------------------------------- #
# the box contract (§8b)                                                 #
# --------------------------------------------------------------------- #


def legal_max_b(m: Model) -> float:
    """Half-width in b of the legal maximum compute box: beyond the Cauchy
    bounds of N and B, every finite critical point is enclosed and the
    far field owns the dynamics."""
    from . import sturm
    bounds = [sturm.cauchy_bound(p) for p in (m.N, m.beta) if P.degree(p) > 0]
    return 1.5 * float(max(bounds, default=Fraction(10)))


def compute_box(m: Model, enumeration, view=None, margin: float = 0.2):
    """§8b: view box ⊆ compute box ⊆ legal max; all finite nondegenerate
    critical points inside by default.  Returns (a_min, a_max, b_min,
    b_max) for the compute box."""
    pts = [p for p in enumeration.points if p.kind != "degenerate"]
    a_vals = [p.a for p in pts] or [0.0]
    b_vals = [p.b for p in pts] or [0.0]
    a_lo, a_hi = min(a_vals), max(a_vals)
    b_lo, b_hi = min(b_vals), max(b_vals)
    da = max(margin * (a_hi - a_lo), 0.5)
    db = max(margin * (b_hi - b_lo), 0.5)
    box = [a_lo - da, a_hi + da, b_lo - db, b_hi + db]
    if view is not None:
        box = [min(box[0], view[0] - da), max(box[1], view[1] + da),
               min(box[2], view[2] - db), max(box[3], view[3] + db)]
    bmax = legal_max_b(m)
    box[2] = max(box[2], -bmax)
    box[3] = min(box[3], bmax)
    return tuple(box)
