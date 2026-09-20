"""Certified bounds on the constant-loss field's Lipschitz constant.

WHAT THIS IS FOR.  Ruling out a saddle connection is a NEGATIVE statement:
the source's unstable branch and the target's stable branch must be shown
not to meet.  Both are shot to a common regular loss level, where their
oriented separation D is a signed scalar; the connection is excluded when
|D| exceeds the enclosure width of the two shots.  Propagating a stub
rectangle to that level is a Gronwall argument, and its amplification is
exp(INT ||J|| dL) for the Jacobian J of the constant-loss field

    dz/dL = grad(L) / |grad(L)|^2 .

So a certified UPPER bound on ||J|| over a region is what converts traced
floating-point shots into a refusal that means something.

WHY A UNIFORM BOUND SUFFICES.  Separation is a sufficient condition: any
valid bound proves it, and a loose one costs only the occasional case that
could have been certified with more work.  Measured on near-slide-d2 at
Lambda = 1, the two stub grid errors are 1.36e-9 and 1.74e-10 against a
separation of 1.60e-1 -- eight orders of headroom.  Treating the SAMPLED
maximum ||J|| (35.4 and 32.6) as if it held uniformly over the whole loss
span of 0.38 gives amplifications of 7.4e5 and 2.6e5, hence an enclosure of
about 1.05e-3: still a factor of 150 inside the separation.  The honest
path integrals are 3.64 and 2.87, i.e. factors of 38 and 17.6, so the crude
bound wastes three orders and still clears the case comfortably.

When the margin is tight the refinement is piecewise, not sharper analysis:
bound ||J|| uniformly on each loss sub-slab and sum the exponents.  Every
term stays a bound and the method degrades by subdividing rather than by
failing.

THE ALGEBRA.  With L = C - 2aB + a^2 A,

    L_aa = 2A,   L_ab = 2(a A' - B'),   L_bb = a^2 A'' - 2a B'',

so every Hessian entry is POLYNOMIAL in (a, b) -- no backbone, no rational
functions, nothing that can blow up inside a box.  And since
||g (Hg)^T|| = |g| |Hg| <= ||H|| |g|^2,

    ||J|| = || H/|g|^2 - 2 g (Hg)^T / |g|^4 ||  <=  3 ||H|| / |g|^2 .

Bounding ||H|| above on a box is interval evaluation of three polynomials.
The work is all in the denominator:

    |grad L|^2 = 4(aA - B)^2 + a^2 (a A' - 2 B')^2

is a sum of two squares vanishing exactly on the critical set, so on a box
containing no critical point it is positive -- but naive interval arithmetic
returns zero for a square whose base interval straddles zero.  Hence
``grad2_lower`` subdivides: the two terms cannot vanish together except at a
critical point, so splitting separates them, and the recursion either
certifies a positive lower bound or reports that it could not.

EXACT THROUGHOUT.  Intervals are pairs of Fractions and every operation is
exact rational arithmetic, so there is no directed-rounding question and no
tolerance anywhere in this module.  Boxes are given as rational corners;
callers holding floats convert exactly, since an IEEE float IS a dyadic
rational.
"""

from __future__ import annotations

from fractions import Fraction

from . import _poly as P

Interval = tuple[Fraction, Fraction]


def _iv(lo, hi) -> Interval:
    lo, hi = P.as_fraction(lo), P.as_fraction(hi)
    return (lo, hi) if lo <= hi else (hi, lo)


def _add(x: Interval, y: Interval) -> Interval:
    return (x[0] + y[0], x[1] + y[1])


def _sub(x: Interval, y: Interval) -> Interval:
    return (x[0] - y[1], x[1] - y[0])


def _mul(x: Interval, y: Interval) -> Interval:
    products = (x[0]*y[0], x[0]*y[1], x[1]*y[0], x[1]*y[1])
    return (min(products), max(products))


def _scale(x: Interval, c: Fraction) -> Interval:
    a, b = x[0]*c, x[1]*c
    return (a, b) if a <= b else (b, a)


def _square(x: Interval) -> Interval:
    """[m^2, M^2] where m is the distance to zero -- ZERO when x straddles.

    This is the exact range of t -> t^2 on the interval, which is why a
    straddling base gives a useless lower bound and why grad2_lower has to
    subdivide rather than evaluate once.
    """
    lo, hi = x
    if lo <= 0 <= hi:
        return (Fraction(0), max(lo*lo, hi*hi))
    m, M = min(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    return (m*m, M*M)


def _abs_max(x: Interval) -> Fraction:
    return max(abs(x[0]), abs(x[1]))


def poly_interval(p: P.Poly, x: Interval) -> Interval:
    """Range enclosure of a univariate polynomial by interval Horner.

    Horner rather than summing monomials: it is the same evaluation the
    exact code uses elsewhere, and it overestimates less than term-by-term
    because each coefficient is added before the next multiplication.
    """
    acc: Interval = (Fraction(0), Fraction(0))
    for c in reversed(p):
        acc = _add(_mul(acc, x), (c, c))
    return acc


def hessian_box(m, a: Interval, b: Interval):
    """(L_aa, L_ab, L_bb) enclosures on the box a x b."""
    A = poly_interval(m.alpha, b)
    Ap = poly_interval(P.deriv(m.alpha), b)
    App = poly_interval(P.deriv(P.deriv(m.alpha)), b)
    Bp = poly_interval(P.deriv(m.beta), b)
    Bpp = poly_interval(P.deriv(P.deriv(m.beta)), b)
    two = Fraction(2)
    l_aa = _scale(A, two)
    l_ab = _scale(_sub(_mul(a, Ap), Bp), two)
    l_bb = _sub(_mul(_square(a), App), _scale(_mul(a, Bpp), two))
    return l_aa, l_ab, l_bb


def hess_norm_upper(m, a: Interval, b: Interval) -> Fraction:
    """Upper bound on the spectral norm of the Hessian over the box.

    For a symmetric matrix the spectral norm is at most the maximum absolute
    row sum, which needs no eigenvalue computation and stays exact.
    """
    l_aa, l_ab, l_bb = hessian_box(m, a, b)
    row_a = _abs_max(l_aa) + _abs_max(l_ab)
    row_b = _abs_max(l_ab) + _abs_max(l_bb)
    return max(row_a, row_b)


def _grad2_interval(m, a: Interval, b: Interval) -> Interval:
    """Enclosure of |grad L|^2 = 4(aA - B)^2 + a^2 (a A' - 2 B')^2."""
    A = poly_interval(m.alpha, b)
    B = poly_interval(m.beta, b)
    Ap = poly_interval(P.deriv(m.alpha), b)
    Bp = poly_interval(P.deriv(m.beta), b)
    first = _square(_sub(_mul(a, A), B))
    second = _mul(_square(a),
                  _square(_sub(_mul(a, Ap), _scale(Bp, Fraction(2)))))
    return _add(_scale(first, Fraction(4)), second)


def grad2_lower(m, a: Interval, b: Interval, max_depth: int = 12):
    """Certified positive lower bound on |grad L|^2 over the box, or None.

    Subdivides because the enclosure of a square is zero whenever its base
    straddles zero: the two terms of |grad L|^2 vanish SIMULTANEOUSLY only
    at a critical point, so splitting the box separates them everywhere
    else.  Returns None when max_depth is reached with a sub-box still
    straddling -- which means either a critical point is in the box or the
    bound needs more subdivision, and the caller must not treat the two
    cases as the same.
    """
    def recurse(ai: Interval, bi: Interval, depth: int):
        enclosure = _grad2_interval(m, ai, bi)
        if enclosure[0] > 0:
            return enclosure[0]
        if depth >= max_depth:
            return None
        awidth, bwidth = ai[1] - ai[0], bi[1] - bi[0]
        if awidth <= 0 and bwidth <= 0:
            return None                       # degenerate box on the zero set
        if awidth >= bwidth:
            mid = (ai[0] + ai[1])/2
            parts = (((ai[0], mid), bi), ((mid, ai[1]), bi))
        else:
            mid = (bi[0] + bi[1])/2
            parts = ((ai, (bi[0], mid)), (ai, (mid, bi[1])))
        best = None
        for sub_a, sub_b in parts:
            value = recurse(sub_a, sub_b, depth + 1)
            if value is None:
                return None
            best = value if best is None else min(best, value)
        return best

    return recurse(_iv(*a), _iv(*b), 0)


def compose_affine(p: P.Poly, c0: Fraction, c1: Fraction) -> P.Poly:
    """p(c0 + c1 t) as an exact polynomial in t, by Horner on polynomials."""
    arg = P.poly((c0, c1))
    acc: P.Poly = P.ZERO
    for c in reversed(p):
        acc = P.add(P.mul(acc, arg), P.poly((c,)))
    return acc


def _tube_value(p: P.Poly, p_along_t: P.Poly, window: Interval,
                b_tube: Interval, halo: Interval) -> Interval:
    """Enclosure of p(b(t) + v) for t in ``window`` and |v| <= radius.

    p(b(t) + v) = p(b(t)) + v p'(xi) for some xi in the tube, so composing
    along t and adding halo * p'(b_tube) is an enclosure that KEEPS the
    t-correlation in the main term and pays for the transverse extent only
    through the derivative.
    """
    main = poly_interval(p_along_t, window)
    slope = poly_interval(P.deriv(p), b_tube)
    return _add(main, _mul(halo, slope))


def segment_bound(m, z0, z1, radius, max_depth: int = 12):
    """||J|| bound on a tube of half-width ``radius`` about the chord z0-z1.

    ALONG THE CHORD, NOT OVER ITS BOUNDING BOX.  Both coordinates are affine
    in the chord parameter t, so every scalar the bound needs composes to a
    univariate polynomial in t:

        a(t) = a0 + (a1-a0) t,      b(t) = b0 + (b1-b0) t,
        aA - B  and  a A' - 2 B'  are then polynomials in t alone.

    That is the whole point.  Treating a and b as independent over the
    chord's bounding box throws away their correlation, and |grad L|^2 is a
    sum of squares that is small ON the curve and much smaller somewhere in
    the box: measured on near-slide-d2, per-step boxes gave 9.7 and 13.7
    against true 3.6 and 2.9, and adaptive bisection barely moved them
    (9.6 and 10.0) because halving a chord halves both box extents and buys
    only a factor of two per level -- a shape problem, not a resolution one.

    ``radius`` must cover the arc's sagitta away from its chord AND the
    enclosure being propagated, or the Gronwall hypothesis does not hold
    where it is being applied.  The caller supplies it and owns that
    bootstrap.
    """
    pad = P.as_fraction(radius)
    a0, a1 = P.as_fraction(z0[0]), P.as_fraction(z1[0])
    b0, b1 = P.as_fraction(z0[1]), P.as_fraction(z1[1])
    unit = (Fraction(0), Fraction(1))
    halo = (-pad, pad)
    a_line = P.poly((a0, a1 - a0))
    b_tube = _add(_iv(min(b0, b1), max(b0, b1)), halo)

    along = {}
    for name, p in (("A", m.alpha), ("B", m.beta),
                    ("Ap", P.deriv(m.alpha)), ("Bp", P.deriv(m.beta)),
                    ("App", P.deriv(P.deriv(m.alpha))),
                    ("Bpp", P.deriv(P.deriv(m.beta)))):
        along[name] = (p, compose_affine(p, b0, b1 - b0))

    def value(name, window):
        p, p_t = along[name]
        return _tube_value(p, p_t, window, b_tube, halo)

    def a_at(window):
        return _add(poly_interval(a_line, window), halo)

    def norm_bound(window):
        """Upper bound on ||J|| over this t-window, or None.

        TWO BOUNDS, WHICHEVER IS SMALLER.  Both are separately valid, so
        their minimum is valid, and neither dominates:

          entrywise   enclose J = H/|g|^2 - 2 g (Hg)^T/|g|^4 entry by entry.
                      Sharp where |g|^2 is well determined, because it keeps
                      the cancellation the triangle inequality discards --
                      J ghat = (I - 2 ghat ghat^T)(H ghat)/|g|^2 is a
                      reflection, not a doubling.
          crude       ||J|| <= 3||H||/|g|^2.  Measured EXACTLY 3x the true
                      norm at the worst sampled point, so it is the looser
                      form in the interior -- but it divides by |g|^2 ONCE,
                      where the entrywise form divides four separate
                      products by a wide reciprocal interval and widens each
                      independently.

        Next to a saddle, where |g|^2 varies by orders within a single step,
        the crude form therefore wins: on near-slide-d2's up shot the
        entrywise bound on segment 0 was 6.68 against the crude 2.59, while
        over the whole down shot the entrywise form gave 4.43 against a true
        3.64 where the crude gave 9.71.
        """
        a_w = a_at(window)
        A_w, B_w = value("A", window), value("B", window)
        Ap_w, Bp_w = value("Ap", window), value("Bp", window)
        App_w, Bpp_w = value("App", window), value("Bpp", window)

        first = _sub(_mul(a_w, A_w), B_w)                  # aA - B
        second = _sub(_mul(a_w, Ap_w), _scale(Bp_w, Fraction(2)))
        g1 = _scale(first, Fraction(2))                    # L_a
        g2 = _mul(a_w, second)                             # L_b
        n2 = _add(_scale(_square(first), Fraction(4)),
                  _mul(_square(a_w), _square(second)))
        if n2[0] <= 0:
            return None

        h11 = _scale(A_w, Fraction(2))
        h12 = _scale(_sub(_mul(a_w, Ap_w), Bp_w), Fraction(2))
        h22 = _sub(_mul(_square(a_w), App_w),
                   _scale(_mul(a_w, Bpp_w), Fraction(2)))

        crude = 3*max(_abs_max(h11) + _abs_max(h12),
                      _abs_max(h12) + _abs_max(h22))/n2[0]

        hg1 = _add(_mul(h11, g1), _mul(h12, g2))
        hg2 = _add(_mul(h12, g1), _mul(h22, g2))
        inv = (Fraction(1)/n2[1], Fraction(1)/n2[0])
        inv2 = _mul(inv, inv)
        minus_two = Fraction(-2)
        j11 = _add(_mul(h11, inv),
                   _scale(_mul(_mul(g1, hg1), inv2), minus_two))
        j12 = _add(_mul(h12, inv),
                   _scale(_mul(_mul(g1, hg2), inv2), minus_two))
        j21 = _add(_mul(h12, inv),
                   _scale(_mul(_mul(g2, hg1), inv2), minus_two))
        j22 = _add(_mul(h22, inv),
                   _scale(_mul(_mul(g2, hg2), inv2), minus_two))
        # For a general (non-symmetric) 2x2, ||J||_2 <= sqrt(||J||_1 ||J||_inf)
        # <= max(||J||_1, ||J||_inf); the latter needs no square root and so
        # stays exact over the rationals.
        entrywise = max(_abs_max(j11) + _abs_max(j12),
                        _abs_max(j21) + _abs_max(j22),
                        _abs_max(j11) + _abs_max(j21),
                        _abs_max(j12) + _abs_max(j22))
        return min(crude, entrywise)

    def bound_on(lo: Fraction, hi: Fraction, depth: int):
        window = (lo, hi)
        value_here = norm_bound(window)
        if value_here is not None:
            return value_here
        if depth >= max_depth:
            return None
        mid = (lo + hi)/2
        left = bound_on(lo, mid, depth + 1)
        if left is None:
            return None
        right = bound_on(mid, hi, depth + 1)
        if right is None:
            return None
        return max(left, right)

    return bound_on(Fraction(0), Fraction(1), 0)


def coefficient_derivatives(m):
    """[(label, dA, dB)] for every coefficient of f and g.

    A and B are bilinear in the coefficients with the moments fixed:

        A(b) = sum_ij g_i g_j mu_{i+j} b^{i+j}   ->  dA/dg_k = 2 sum_j g_j mu_{k+j} b^{k+j}
        B(b) = sum_j g_j (sum_i f_i mu_{i+j}) b^j
                                                 ->  dB/dg_k = (sum_i f_i mu_{i+k}) b^k
                                                     dB/df_i = sum_j g_j mu_{i+j} b^j

    C = <f,f> depends on f too, but C does not enter grad L, so it plays no
    part in the field's sensitivity -- only in the level VALUES, which the
    meeting sections handle separately.
    """
    f, g, mu = m.f, m.g, m.mu
    df, dg = P.degree(f), P.degree(g)
    out = []
    for i in range(df + 1):
        dB = [Fraction(0)]*(dg + 1)
        for j in range(dg + 1):
            dB[j] = g[j]*mu[i + j]
        out.append((f"f{i}", P.ZERO, P.trim(tuple(dB))))
    for k in range(dg + 1):
        dA = [Fraction(0)]*(2*dg + 1)
        for j in range(dg + 1):
            dA[k + j] += 2*g[j]*mu[k + j]
        dB = [Fraction(0)]*(dg + 1)
        dB[k] = sum((f[i]*mu[i + k] for i in range(df + 1)), Fraction(0))
        out.append((f"g{k}", P.trim(tuple(dA)), P.trim(tuple(dB))))
    return out


def field_sensitivity(m, dA: P.Poly, dB: P.Poly, z0, z1, radius,
                      max_depth: int = 12):
    """Bound on || d_theta (grad L / |grad L|^2) || over a chord's tube.

    THE OTHER HALF OF THE SMALE CERTIFICATE.  Separation says this model has
    no connection; the useful statement is that no model within a BALL of it
    does.  The meeting-level discrepancy D is smooth in the coefficients, so

        dist(theta, non-Smale)  >=  |D| / sup ||dD/dtheta|| ,

    and dD/dtheta solves the variational equation along the same shot

        dz_theta/dL = J z_theta + d_theta(field),

    bounded by exp(INT ||J|| dL) times the integrated forcing -- the SAME
    amplification the separation certificate already computes.  One
    amplification, three uses: propagating the stub enclosure, propagating a
    coefficient perturbation, and later converting a discrepancy enclosure
    into a Lambda enclosure through dD/dLambda.

    THE FORCING IS EXPLICIT.  With L = C - 2aB + a^2 A and the moments
    fixed, A and B are bilinear in the coefficients, so d_theta A and
    d_theta B are polynomials in b of the same shape (supplied by the
    caller as ``dA`` and ``dB``).  Then

        d_theta grad L = ( 2(a dA - dB),  a(a dA' - 2 dB') )
        d_theta field  = d_theta g / |g|^2 - 2 g (g . d_theta g) / |g|^4 .

    WHAT IT COSTS IN SHARPNESS, recorded because it was measured rather
    than guessed: on near-slide-d2 the true sensitivities are 0.6 to 9.8 per
    unit coefficient (finite differences), giving a ball radius of 1.6e-2,
    while exp(INT ||J||) alone is 84 and 2965 on the two shots.  The
    Gronwall form is therefore pessimistic by roughly the amplification
    itself, because the variational solution largely follows the trajectory
    instead of exploring the worst direction of J at every step.  Expect a
    certified radius nearer 1e-5 than 1e-2 -- still ten orders above
    binary64 representation error, which is the comparison that matters.
    """
    pad = P.as_fraction(radius)
    a0, a1 = P.as_fraction(z0[0]), P.as_fraction(z1[0])
    b0, b1 = P.as_fraction(z0[1]), P.as_fraction(z1[1])
    unit = (Fraction(0), Fraction(1))
    halo = (-pad, pad)
    a_line = P.poly((a0, a1 - a0))
    b_tube = _add(_iv(min(b0, b1), max(b0, b1)), halo)
    a_w = _add(poly_interval(a_line, unit), halo)

    def along(p):
        return _tube_value(p, compose_affine(p, b0, b1 - b0), unit,
                           b_tube, halo)

    A_w, B_w = along(m.alpha), along(m.beta)
    Ap_w, Bp_w = along(P.deriv(m.alpha)), along(P.deriv(m.beta))
    first = _sub(_mul(a_w, A_w), B_w)
    second = _sub(_mul(a_w, Ap_w), _scale(Bp_w, Fraction(2)))
    g1 = _scale(first, Fraction(2))
    g2 = _mul(a_w, second)
    n2 = _add(_scale(_square(first), Fraction(4)),
              _mul(_square(a_w), _square(second)))
    if n2[0] <= 0:
        return None

    dA_w, dB_w = along(dA), along(dB)
    dAp_w, dBp_w = along(P.deriv(dA)), along(P.deriv(dB))
    d1 = _scale(_sub(_mul(a_w, dA_w), dB_w), Fraction(2))
    d2 = _mul(a_w, _sub(_mul(a_w, dAp_w), _scale(dBp_w, Fraction(2))))

    inv = (Fraction(1)/n2[1], Fraction(1)/n2[0])
    inv2 = _mul(inv, inv)
    dot = _add(_mul(g1, d1), _mul(g2, d2))
    minus_two = Fraction(-2)
    s1 = _add(_mul(d1, inv), _scale(_mul(_mul(g1, dot), inv2), minus_two))
    s2 = _add(_mul(d2, inv), _scale(_mul(_mul(g2, dot), inv2), minus_two))
    # |(s1, s2)| <= |s1| + |s2|, exact over the rationals (no square root).
    return _abs_max(s1) + _abs_max(s2)


def path_integral_upper(m, points, levels, radius, *, splits: int = 1,
                        max_depth: int = 12):
    """Certified upper bound on INT ||J|| dL along a traced path.

    ``points`` and ``levels`` are the shot's own accepted step endpoints and
    their losses, so the decomposition comes free with the trace rather than
    costing extra meeting levels: the earlier uniform-slab experiment used
    127 sampled levels where the shot had already computed about 2000 steps,
    and its bound was still 17.6 against a true 3.6.

    THE TWO ENDS ARE NOT ALIKE.  Each shot has exactly one end adjacent to a
    saddle, where |grad L| -> 0 and ||J|| genuinely diverges -- 35.4 at the
    source end and 32.6 at the target end on near-slide-d2, against a path
    integral of only 3.6, because the spike is narrow.  That end is the
    materialized stub's business and the integration starts at its far edge;
    the first segment past it still carries much of the integral, which is
    why the per-segment contributions are returned and not only their sum.

    Returns (total, contributions); total is None when a segment could not
    be certified, with the contributions computed so far so the caller can
    see WHERE it failed rather than only that it did.
    """
    contributions = []
    total = Fraction(0)
    for i in range(len(points) - 1):
        a0, b0 = P.as_fraction(points[i][0]), P.as_fraction(points[i][1])
        a1, b1 = P.as_fraction(points[i+1][0]), P.as_fraction(points[i+1][1])
        span = abs(P.as_fraction(levels[i+1]) - P.as_fraction(levels[i]))
        if span == 0:
            continue
        piece = Fraction(0)
        for k in range(splits):
            t0, t1 = Fraction(k, splits), Fraction(k+1, splits)
            p0 = (a0 + (a1-a0)*t0, b0 + (b1-b0)*t0)
            p1 = (a0 + (a1-a0)*t1, b0 + (b1-b0)*t1)
            bound = segment_bound(m, p0, p1, radius, max_depth)
            if bound is None:
                contributions.append(None)
                return None, contributions
            piece += bound*span/splits
        contributions.append(float(piece))
        total += piece
    return total, contributions


def path_integral_adaptive(m, points, levels, radius, *, max_depth: int = 12,
                           split_depth: int = 8, gain: float = 1.1):
    """INT ||J|| dL along a path, subdividing only the segments that pay.

    Uniform refinement is the wrong instrument here because the excess is not
    spread evenly: measured on near-slide-d2 with one box per accepted step,
    the bounds were 9.7 and 13.7 against true 3.6 and 2.9, and segment 0 --
    the first step past the materialized stub, where |grad L| is smallest --
    carried 1.05 of 9.7 and 4.14 of 13.7 on its own, while the last segments
    contributed about 0.011.  Splitting everything to fix that would cost a
    hundredfold for nothing; splitting where it helps costs almost nothing.

    The policy is local and self-limiting: bisect a chord while doing so
    reduces its own bound by at least ``gain``, and stop otherwise or at
    ``split_depth``.  A segment whose bound is already sharp is left alone
    after one extra evaluation, and a segment at the singular end refines
    until the geometry stops rewarding it.

    THE LAUNCH END IS NOT THE SADDLE.  Integration starts at the far edge of
    the certified Frobenius/graph stub, whose own grid error is recorded
    separately (1.36e-9 and 1.74e-10 on this case), so the singular
    neighbourhood is covered by that certificate rather than by this bound.
    What remains here is the first step OUTSIDE it, which is why that
    segment dominates and why extending the stub's reach -- a separate
    accuracy-versus-cost question -- would attack the same excess from the
    other side.

    Returns (total, contributions, splits_used); total is None when a piece
    could not be certified.
    """
    def refine(p0, p1, span, depth):
        """(bound*span, evaluations) for one chord, bisecting while it pays."""
        whole = segment_bound(m, p0, p1, radius, max_depth)
        if whole is None:
            return None, 1
        if depth >= split_depth:
            return whole*span, 1
        mid = ((p0[0] + p1[0])/2, (p0[1] + p1[1])/2)
        left = segment_bound(m, p0, mid, radius, max_depth)
        right = segment_bound(m, mid, p1, radius, max_depth)
        if left is None or right is None:
            return whole*span, 3
        halved = (left + right)*span/2
        if halved*P.as_fraction(gain) >= whole*span:
            return whole*span, 3       # bisecting did not pay; keep the chord
        a, na = refine(p0, mid, span/2, depth + 1)
        b, nb = refine(mid, p1, span/2, depth + 1)
        if a is None or b is None:
            return whole*span, 3 + na + nb
        return a + b, 3 + na + nb

    contributions = []
    evaluations = 0
    total = Fraction(0)
    for i in range(len(points) - 1):
        a0, b0 = P.as_fraction(points[i][0]), P.as_fraction(points[i][1])
        a1, b1 = P.as_fraction(points[i+1][0]), P.as_fraction(points[i+1][1])
        span = abs(P.as_fraction(levels[i+1]) - P.as_fraction(levels[i]))
        if span == 0:
            continue
        piece, used = refine((a0, b0), (a1, b1), span, 0)
        evaluations += used
        if piece is None:
            contributions.append(None)
            return None, contributions, evaluations
        contributions.append(float(piece))
        total += piece
    return total, contributions, evaluations


def lipschitz_upper(m, a, b, max_depth: int = 12):
    """Certified upper bound on ||J|| over the box, or None.

    ||J|| <= 3 ||H|| / |grad L|^2, with the numerator an interval evaluation
    of polynomials and the denominator a subdivided positivity bound.  None
    means the denominator could not be certified positive at this depth, NOT
    that the field is unbounded -- the caller decides whether to subdivide
    the slab further or to report that the region reaches a critical point.
    """
    a_iv, b_iv = _iv(*a), _iv(*b)
    floor = grad2_lower(m, a_iv, b_iv, max_depth)
    if floor is None or floor <= 0:
        return None
    return 3*hess_norm_upper(m, a_iv, b_iv)/floor
