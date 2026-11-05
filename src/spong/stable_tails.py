"""The stable separatrices beyond the certified level bar: drawing only.

A certified stable branch stops at the level bar, just above the highest saddle,
because escape is already certain there.  A portrait still has to show where the
branch goes.  Each one is grown by the tracer until a measurement on the branch
licenses a closed-form tail: a level set of the invariant J along a diagonal, or
the b-nullcline toward an attracting root of A'.  See docs/stable_escape.md.

Nothing here enters a certificate.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from . import atlas, sturm
from .charts import Branch
from .portrait import Portrait

EPS = np.finfo(float).eps


def _horner(coef_desc, x):
    s = 0.0
    for c in coef_desc:
        s = s*x + c
    return s


def _abs_horner(coef_desc, x):
    """Sum of |c_k x^k| -- the running-error scale of a Horner evaluation."""
    ax = abs(x)
    s = 0.0
    for c in coef_desc:
        s = s*ax + abs(c)
    return s


def _root_resolution(Ap, App, rho):
    """How closely binary64 can place a simple root rho of A'.

    Horner's first-order error bound n*eps*sum|c_k x^k| (coefficient rounding
    charges one more eps), divided by the slope |A''| at the root.  Distances
    from rho below this are not measurements.
    """
    slope = abs(np.polyval(App, rho))
    if slope == 0.0:
        return math.inf
    return len(Ap)*EPS*_abs_horner(Ap, rho)/slope


def _traps(m):
    """The critical points at infinity, EXACTLY classified.

    They are the real roots of A' (Sturm-isolated, refined to 2^-40), each
    with the certified sign of A'' on its isolating interval: A'' < 0 is a
    local maximum of A and ATTRACTING under ascent (a stable branch can end
    pinned there, b -> rho with |a| -> inf); A'' > 0 is repelling.  Returns
    a sorted list of (rho, attracting).
    """
    from spong import _poly as P
    Ap = P.deriv(m.alpha)
    App = P.deriv(Ap)
    out = []
    for iv in sturm.isolate_roots(Ap):
        iv = sturm.refine(Ap, iv, Fraction(1, 2**40))
        s = sturm.interval_sign(App, iv)
        out.append((float(iv.mid), s is not None and s < 0))
    return sorted(out)


def _residue(A_desc, App_desc, z):
    """A(z)/A''(z) at a root of A', without overflowing either evaluation.

    Direct polyval overflows on the cases this viewer is most wanted for: at
    degree 2d with large coefficients and |z| well above 1, both numerator
    and denominator reach inf and the quotient is NaN --

        RuntimeWarning: invalid value encountered in scalar divide

    which is not cosmetic.  NaN residues make every J value NaN, the
    diagonal licence test np.all(np.isfinite(J)) then fails, and the branch
    silently gets no drawn tail at all.

    Same remedy as model._rational_product: for |z| > 1 evaluate through the
    REVERSED polynomials in 1/z, so only the net power z^(deg A - deg A'')
    is ever formed.  That power is 2 for the ratio at hand, so nothing large
    appears even when each polynomial separately would overflow.
    """
    if abs(z) <= 1.0:
        denominator = np.polyval(App_desc, z)
        return (np.polyval(A_desc, z)/denominator if denominator != 0
                else complex("nan"))
    inv = 1.0/z
    numerator = np.polyval(A_desc[::-1], inv)
    denominator = np.polyval(App_desc[::-1], inv)
    if denominator == 0:
        return complex("nan")
    power = (len(A_desc) - 1) - (len(App_desc) - 1)
    return (z**power)*numerator/denominator


def _j_invariant(m, d_eff: float):
    """The exact invariant of the B-truncated flow, in closed form.

    Dropping B, db/da = (a/2)A'(b)/A(b) is SEPARABLE, so

        J(a, b) = a^2/2 - INT^b 2A/A' ds

    is conserved exactly.  A/A' is rational with deg A = 2d and
    deg A' = 2d-1, so its polynomial part is linear, b/(2d) + c1, and
    partial fractions give the closed form

        J = a^2/2 - b^2/(2d) - 2 c1 b - 2 Re SUM_j r_j log(b - rho_j),
        r_j = A(rho_j)/A''(rho_j),   rho_j the roots of A'.

    ONE object covers BOTH tails, which is why this supersedes any sector
    dispatcher: near a root rho_j the log term dominates and the level set
    is the horizontal (regime-2) tail b -> rho_j; away from all of them the
    b^2/2d term dominates and it is the diagonal.  The log residues ARE the
    critical points at infinity -- the same rho_j that trap stable branches.

    dJ/db = -2A/A' exactly, so J is strictly monotone in b between
    consecutive real roots of A', which makes the level set a well-posed
    scalar root-find on that interval and bounds the tail by the traps on
    either side automatically.

    Returns (c1, rhos, residues).  The REAL roots, exactly classified, come
    from _traps; np.roots here supplies the residues only.
    """
    A = np.asarray(m._fa, dtype=float)              # ascending powers
    Ap = np.polyder(A[::-1])                        # numpy: descending
    App = np.polyder(Ap)
    q, _r = np.polydiv(A[::-1], Ap)                 # q = b/(2d) + c1
    c1 = float(q[-1]) if len(q) else 0.0
    rhos = np.roots(Ap)
    res = np.array([_residue(A[::-1], App, z) for z in rhos], dtype=complex)
    return c1, rhos, res


def _j_value(a, b, d_eff, c1, rhos, res):
    v = 0.5*a*a - b*b/(2.0*d_eff) - 2.0*c1*b
    v -= 2.0*float(np.real(np.sum(res*np.log(complex(b) - rhos))))
    return v


def _j_floor(a, b, d_eff, c1, rhos, res):
    """Running-error bound of _j_value: eps times the sum of |terms|."""
    logs = np.log(np.abs(complex(b) - rhos) + 1e-300)
    s = (0.5*a*a + b*b/(2.0*d_eff) + 2.0*abs(c1*b)
         + 2.0*float(np.sum(np.abs(res)*np.abs(logs))))
    return 8.0*EPS*max(s, 1.0)


def _j_root(aa, b, J0, d_eff, c1, rhos, res, A, Ap, lo, hi):
    """b with J(aa, b) = J0 on the trap interval (lo, hi), or None.

    BRACKETED, RESIDUAL-ACCEPTED.  J is strictly monotone on (lo, hi), so
    there is at most one root and a sign change brackets it.  Newton from
    the previous b proposes; the bracket disposes -- a proposal outside it
    is replaced by bisection.  A root is ACCEPTED only when the residual is
    below the evaluation floor of J itself (or the bracket has closed to a
    few ulps of b), never on a step-size test: the log terms make J
    steep near a trap, where a small Newton step is not a small residual.
    Returns None when no bracket exists at this aa (the level set has no
    real b here) or the residual cannot be driven to the floor.
    """
    def f(x):
        return _j_value(aa, x, d_eff, c1, rhos, res) - J0

    span = max(abs(b), 1.0)
    bl = b - span; bh = b + span
    if lo > -math.inf:
        bl = max(bl, lo + max(abs(lo), 1.0)*1e-15)
    if hi < math.inf:
        bh = min(bh, hi - max(abs(hi), 1.0)*1e-15)
    fl, fh = f(bl), f(bh)
    # Widen away from the traps until a sign change exists.
    k = 0
    while fl*fh > 0 and k < 200:
        k += 1
        if lo == -math.inf and hi == math.inf:
            bl = b - 2.0*(b - bl); bh = b + 2.0*(bh - b)
            fl, fh = f(bl), f(bh)
        elif lo == -math.inf:
            bl = b - 2.0*(b - bl); fl = f(bl)
        elif hi == math.inf:
            bh = b + 2.0*(bh - b); fh = f(bh)
        else:
            return None                          # bounded interval, no root
        if not (math.isfinite(fl) and math.isfinite(fh)):
            return None
    if fl*fh > 0:
        return None
    if fl == 0.0:
        return bl
    if fh == 0.0:
        return bh
    x = min(max(b, bl), bh)
    fx = f(x)
    for _ in range(120):
        floor = _j_floor(aa, x, d_eff, c1, rhos, res)
        if abs(fx) <= floor:
            return x
        if fl*fx < 0:
            bh, fh = x, fx
        else:
            bl, fl = x, fx
        if bh - bl <= 4.0*EPS*max(abs(bl), abs(bh)):
            return 0.5*(bl + bh)
        Av = float(np.polyval(A[::-1], x)); Apv = float(np.polyval(Ap, x))
        xn = x - fx/(-2.0*Av/Apv) if Apv != 0.0 else None
        if xn is None or not (bl < xn < bh):
            xn = 0.5*(bl + bh)
        x = xn
        fx = f(x)
        if not math.isfinite(fx):
            return None
    return None


def _nullcline_b(a, b, Ap, App, Bp, Bpp, lo, hi):
    """b on the b-nullcline a A'(b) = 2 B'(b) nearest the guess, or None.

    Newton with residual acceptance against the running-error floor of
    a A' - 2 B'; iterates are kept inside the trap interval (lo, hi).
    """
    # Plain floats and the same Horner order as np.polyval, so the same values
    # at a fraction of the cost: this runs for every point of every pinned tail.
    a = float(a); b = float(b)
    Ap, App, Bp, Bpp = (p.tolist() if hasattr(p, "tolist") else p
                        for p in (Ap, App, Bp, Bpp))
    for _ in range(80):
        F = a*_horner(Ap, b) - 2.0*_horner(Bp, b)
        floor = 8.0*EPS*(abs(a)*_abs_horner(Ap, b) + 2.0*_abs_horner(Bp, b))
        if abs(F) <= max(floor, 1e-300):
            return b
        Fp = a*_horner(App, b) - 2.0*_horner(Bpp, b)
        if Fp == 0.0:
            return None
        bn = b - F/Fp
        if not (lo < bn < hi):
            bn = b - 0.5*(b - (lo if bn <= lo else hi))
        if abs(bn - b) <= EPS*max(1.0, abs(b)):
            # Stagnation at an interval endpoint is not a root.  In
            # particular, after a finite nullcline fold, repeated bisection
            # toward a trap can reach machine zero with a large residual.
            residual = a*_horner(Ap, bn) - 2.0*_horner(Bp, bn)
            error = 8.0*EPS*(abs(a)*_abs_horner(Ap, bn)
                             + 2.0*_abs_horner(Bp, bn))
            return bn if abs(residual) <= max(error, 1e-300) else None
        b = bn
    return None


def _grow_stable(m, start, ds, r_max, critical, max_steps=200000,
                 ds0=None):
    """Continue a stable branch outward from ``start`` by the SAME two-tier
    machinery that traced it: the constant-potential-rate ascent, and, where
    that step-fails short of the box, the chart continuation engine from the
    point it reached.  Bounded by the box |a|,|b| <= r_max and by steps.
    Returns the new points (excluding start) and the terminal label.

    ``ds0`` IS NOT OPTIONAL IN PRACTICE.  charts.trace_stable hands the
    continuation ds0=launch_scale -- the stub's physical reach, a tiny first
    chord that the engine then ramps up from.  Passing the full chord as ds0
    instead makes the engine attempt its FIRST step at that size, and in a
    stiff canyon that step lands on a different branch of the flow: measured
    on inverse.separated_linear_case(4, 30), the wall from the B-saddle at
    b = 1 with ds = 7.2 goes to the trap at b = -14.892264 when ds0 = 7.2 and
    to the correct trap at b = +597.255444 when ds0 is the branch's own last
    chord (0.0022) or smaller.  It is not a step-size accuracy question --
    the destination changes, not its precision -- and the drawn curve was
    confidently wrong with a sector label and a named trap attached.  Default
    to something small rather than to ds.
    """
    from spong import charts
    box = (-r_max, r_max, -r_max, r_max)
    # The chord is derived from the box, as the tracer derives it: a chord
    # sized to the legal box (285 on d17-thrash) cannot take a step inside
    # a box of radius ten.
    ds = min(ds, 4.0*r_max/30000.0)
    def resolved_prefix(previous, candidate):
        # Potential-rate ascent emits at geometric_ds = 4*ds; chart
        # continuation emits at ds.  A chord beyond twice the larger budget
        # is an unresolved solver jump.  A short chord can still leave the
        # integral curve at a chart seam, so compare it with the ascent field
        # whenever the gradient direction carries three reliable digits.
        # Evaluate in chunks to keep million-vertex display growth bounded.
        for lo in range(0, len(candidate), 8192):
            z1 = np.asarray(candidate[lo:lo+8192], dtype=float)
            z0 = np.concatenate((
                np.asarray([previous if lo == 0 else candidate[lo-1]],
                           dtype=float), z1[:-1]), axis=0)
            d = z1-z0
            length = np.hypot(d[:, 0], d[:, 1])
            bad = ~np.isfinite(length) | (length == 0.0) | (length > 8.0*ds)
            if not np.any(bad):
                with np.errstate(over="ignore", divide="ignore",
                                 invalid="ignore"):
                    a, b = z0[:, 0], z0[:, 1]
                    grad = np.asarray(m.gradL(a, b)).T
                    norm = np.hypot(grad[:, 0], grad[:, 1])
                    scale_a = 2.0*(np.abs(a)*m.A(b)+np.abs(m.B(b)))
                    scale_b = (2.0*np.abs(a)*np.abs(m.Bp(b))
                               + a*a*np.abs(m.Ap(b)))
                    floor = (16.0*np.finfo(float).eps
                             * np.hypot(scale_a, scale_b))
                    cosine = np.einsum("ij,ij->i", grad, d)/(norm*length)
                resolved = norm >= np.maximum(1e-12, 1e3*floor)
                bad |= ~np.isfinite(norm)
                bad |= resolved & (cosine < math.cos(math.radians(15.0)))
            if np.any(bad):
                return candidate[:lo+int(np.flatnonzero(bad)[0])], False
        return candidate, True

    pts, term = charts._potential_rate_box_exit(
        m, tuple(start), box, ds, {}, max_steps=min(max_steps, 50000),
        critical=critical)
    out, valid = resolved_prefix(tuple(start),
        [(float(q[0]), float(q[1])) for q in pts[1:]])
    if not valid:
        return out, "growth_geometry_failure"
    if term != "box_exit":
        last = pts[-1] if len(pts) else start
        b0 = float(last[1]); w0 = float(last[0] - m.a_star(b0))
        # The fallback's continuation floor belongs to this handoff. The
        # traced branch's last chord can be thousands of times larger than
        # the potential phase's last accepted chord after a stiff fold;
        # reusing it makes the chart refuse at an artificial floor.
        handoff_ds0 = ds0
        if len(pts) > 1:
            local_chord = math.hypot(float(pts[-1][0]-pts[-2][0]),
                                     float(pts[-1][1]-pts[-2][1]))
            if local_chord > 0.0:
                handoff_ds0 = (local_chord if handoff_ds0 is None else
                               min(handoff_ds0, local_chord))
        try:
            more, term, _sw, _ = charts._continue_curve(
                m, b0, w0, -1, [], box, ds,
                ds0=(handoff_ds0 if handoff_ds0 is not None
                     else min(ds, 1e-6)),
                engine_diag={}, max_steps=max_steps)
            grown, valid = resolved_prefix(tuple(last),
                [(float(q[0]), float(q[1])) for q in more[1:]])
            out.extend(grown)
            if not valid:
                term = "growth_geometry_failure"
        except Exception:                            # display only
            term = "growth_failure"
    return out, term


def _j_tails(m, p, d_eff: float, enumeration=None, tol: float = 1e-4,
             n_pts: int = 1200, reach: float = 3.0e3, max_doublings: int = 8):
    """Closed-form tails for the stable branches, in TWO SECTORS, each
    licensed by a measurement on the branch itself.

    Every stable branch escapes to infinity in one of two ways, and both
    limits are ends of the far field's degeneracy (radiation conditions,
    docs/stable_escape.md):

      DIAGONAL  |b| -> inf along b ~ +-sqrt(d) a.  The tail is the level set
                J = J0 of the exact invariant of the B-truncated flow.
      PINNED    b -> rho, |a| -> inf, rho an ATTRACTING root of A' (a
                critical point at infinity; enumerated EXACTLY by _traps).
                The tail is the b-NULLCLINE a A'(b) = 2 B'(b): the curve on
                which the b-velocity vanishes.  It is not the branch and not
                an invariant manifold -- it is the slaved equilibrium the
                branch approaches; every critical point with a != 0 lies on
                it, and the branch's lag behind it is measured, not assumed
                (minimal-quartet br2: 5e-6 of r at a = 18, 8e-14 at 3000).

    LICENSING is by measurement over a RADIUS OCTAVE, so that the estimate
    is not vacuous:

      pinned:   over the last octave of r on the polyline the lag
                |b - b_null(a)|/r never exceeds ``tol`` and is no larger at
                the end than at the start of the octave.
      diagonal: at the switchover k the REMAINING DRIFT of J along the
                polyline, converted to a displacement in b relative to r,
                  D_k = max_{j>k} |J_j - J_k| * |A'/(2A)|_j / r_j,
                is at most ``tol``, and the polyline continues to r >= 2 r_k
                beyond k.  D predicts the realised tail error to within a
                factor of ~1 on the cases measured (scripts/jtail_probe.py).

    Where the certified terminal licenses neither -- the level bar stops a
    branch where certification is decided, which has nothing to do with
    where B stops mattering -- the branch is GROWN by the same machinery
    that traced it, one radius octave at a time, until one sector licenses
    or ``max_doublings`` is spent, and the growth is reported for drawing.
    The half-plane margin |a|/max(|a*|, 2|B'/A'|) is no longer a criterion:
    on a pinned branch it is identically 1 by construction (the branch IS
    on the b-nullcline), and on diagonal ones the tail error scales as
    c*margin^-p with c and p varying by orders of magnitude across models.

    ``tol`` is a display accuracy -- a displacement relative to the radius,
    i.e. a fraction of the view at that scale -- and is the only knob.
    """
    if d_eff <= 0:
        return []
    traps = _traps(m)
    real_rhos = [r for r, _att in traps]
    attracting_rhos = {r for r, att in traps if att}
    c1, rhos, res = _j_invariant(m, d_eff)
    A = np.asarray(m._fa, dtype=float)
    Ap = np.polyder(A[::-1]); App = np.polyder(Ap)
    Bc = np.asarray(m._fb, dtype=float)
    Bp = np.polyder(Bc[::-1]); Bpp = np.polyder(Bp)
    rho_floor = {rho: _root_resolution(Ap, App, rho) for rho in real_rhos}
    critical = (np.array([[float(q.a), float(q.b)]
                          for q in enumeration.points], dtype=float)
                if enumeration is not None else None)

    def interval(b0):
        lo = max([z for z in real_rhos if z < b0], default=-math.inf)
        hi = min([z for z in real_rhos if z > b0], default=math.inf)
        return lo, hi

    def newton_interval(lo, hi):
        # A nullcline root may sit on the far side of a stored rho by up to
        # that rho's resolution; a hard wall there would reject it.
        return (lo-rho_floor[lo] if lo in rho_floor else lo,
                hi+rho_floor[hi] if hi in rho_floor else hi)

    def license(pts):
        a = np.array([q[0] for q in pts]); b = np.array([q[1] for q in pts])
        r = np.hypot(a, b); n = len(pts)
        if n < 3:
            return None
        # pinned: nullcline lag over the last radius octave (sampled)
        k0 = next((k for k in range(n) if r[k] >= 0.5*r[-1]), n-1)
        if k0 < n-1 and abs(a[-1]) > abs(a[k0]):
            lo, hi = interval(b[-1])
            targets = [rho for rho in (lo, hi) if rho in attracting_rhos
                       and (abs(b[-1]-rho) < abs(b[k0]-rho)
                            or abs(b[-1]-rho) <= rho_floor[rho])]
            if targets:
                rho = min(targets, key=lambda z: abs(b[-1]-z))
                ks = np.unique(np.linspace(k0, n-1, min(400, n-k0)).astype(int))
                lag = []
                wide = newton_interval(lo, hi)
                for k in ks:
                    bn = _nullcline_b(a[k], b[k], Ap, App, Bp, Bpp, *wide)
                    lag.append(math.inf if bn is None else abs(b[k]-bn)/r[k])
                if max(lag) <= tol and lag[-1] <= lag[0]:
                    # A small lag alone does not license an infinite tail:
                    # the nullcline can fold at a finite radius and Newton
                    # can jump to another root in this same trap interval.
                    # Those comparisons mean nothing once they fall below
                    # the resolution of rho itself: on a tall ridge the
                    # nullcline's offset from rho is smaller than that long
                    # before |a| reaches the probe's far end.
                    floor = rho_floor[rho]
                    probe_b = float(b[-1])
                    safe = True
                    for aa in np.geomspace(abs(a[-1]), max(10*abs(a[-1]), reach), 65)[1:]:
                        bn = _nullcline_b(math.copysign(aa, a[-1]), probe_b,
                                          Ap, App, Bp, Bpp, *wide)
                        if (bn is None
                                or abs(bn-rho) > max(abs(probe_b-rho)*(1+1e-8), floor)
                                or abs(bn-probe_b) > max(0.25*abs(probe_b-rho), 2.0*floor)):
                            safe = False
                            break
                        probe_b = bn
                    if safe:
                        return {"sector": "pinned", "index": n-1,
                                "lag": max(lag), "lag_end": lag[-1],
                                "trap": rho}
        # A branch pinned within the resolution of an attracting root is not
        # on a diagonal, whatever the J drift says: the weight |A'/2A|/r in
        # that drift nearly vanishes on the ridge, so the test would pass and
        # hand back a level set with no real b beyond the branch.
        lo, hi = interval(b[-1])
        if any(rho in attracting_rhos and abs(b[-1]-rho) <= rho_floor[rho]
               for rho in (lo, hi)):
            return None
        # diagonal: remaining J drift after k, attested over an octave
        # Symmetric normal cases have a real A' root at b=0.  A sampled
        # branch may hit that trap exactly, where J is singular; the finite
        # check below rejects diagonal licensing for that branch.
        with np.errstate(divide="ignore", invalid="ignore"):
            J = (0.5*a*a - b*b/(2.0*d_eff) - 2.0*c1*b
                 - 2.0*np.real(np.log(b[:, None].astype(complex) - rhos[None, :])
                               @ res))
        if np.all(np.isfinite(J)):
            w = np.abs(np.polyval(Ap, b)/(2.0*np.polyval(A[::-1], b)))/r
            # The max over j > k is taken on a sampled j-set, dense enough
            # for a smooth drift.
            js = np.unique(np.linspace(0, n-1, min(3000, n)).astype(int))
            Jw, wj = J[js], w[js]
            for k in js[:-1]:
                if r[-1] < 2.0*r[k]:
                    break
                sel = js > k
                D = float(np.max(np.abs(Jw[sel]-J[k])*wj[sel]))
                if D <= tol:
                    return {"sector": "diagonal", "index": int(k),
                            "drift": D, "J0": float(J[k])}
        return None

    r_crit = max([math.hypot(float(q.a), float(q.b))
                  for q in enumeration.points] or [1.0]) \
        if enumeration is not None else 1.0
    out = []
    for i, br in enumerate(p.branches):
        if br.kind != "stable" or len(br.Y) < 2:
            continue
        curve = [(float(q[0]), float(q[1])) for q in br.Y]
        n_traced = len(curve)
        trace_failed = br.term not in ("box_exit", "level_bar")
        lic = None if trace_failed else license(curve)
        n_grown = 0
        grow_term = None
        growth_failed = False
        if lic is None and critical is not None \
                and br.term in ("box_exit", "level_bar"):
            ascent = br.diag.get("potential_rate_ascent") or {}
            box = p.box
            ds = (float(ascent["geometric_ds"])/4.0 if "geometric_ds" in ascent
                  else math.hypot(box[1]-box[0], box[3]-box[2])/30000.0)
            for _ in range(max_doublings):
                r_now = math.hypot(*curve[-1])
                r_max = 2.0*max(r_now, r_crit)
                # The engine ramps from ds0; give it the branch's own last
                # chord, as trace_stable gives it the stub's reach.
                tail_chord = math.hypot(curve[-1][0] - curve[-2][0],
                                        curve[-1][1] - curve[-2][1])
                more, grow_term = _grow_stable(
                    m, curve[-1], ds, r_max, critical,
                    ds0=(tail_chord if tail_chord > 0.0 else None))
                if not more:
                    growth_failed = grow_term != "box_exit"
                    break
                curve.extend(more)
                n_grown = len(curve) - n_traced
                if grow_term != "box_exit":
                    growth_failed = True
                    break
                lic = license(curve)
                if lic is not None:
                    break
        rec = {"branch": i, "kind": br.kind, "n_traced": n_traced,
               "n_grown": n_grown, "grow_term": grow_term,
               "grown": [[float(x), float(y)] for x, y in curve[n_traced-1:]],
               "points": [], "sector": None, "tol": tol}
        if lic is None or trace_failed or growth_failed:
            rec["reason"] = (
                f"stable trace stopped at {br.term}"
                if trace_failed else
                f"stable growth stopped at {grow_term}"
                if growth_failed else
                "neither sector licensed within the traced and grown branch")
            out.append(rec)
            continue
        rec.update(lic)
        a0, b0 = curve[-1]
        lo, hi = interval(b0)
        # direction of travel in a from the last chord, not sign(a0): J
        # depends on a^2 (two mirror arms) and the terminal point alone
        # cannot choose between them.
        da = a0 - curve[max(0, len(curve)-2)][0]
        s = 1.0 if da > 0 else -1.0 if da < 0 else (1.0 if a0 >= 0 else -1.0)
        amax = max(abs(a0)*10.0, reach)
        step0 = max(abs(a0)*1e-3, 1e-9)
        pts = [[a0, b0]]
        b = b0
        if lic["sector"] == "pinned":
            for k in range(1, n_pts+1):
                aa = a0 + s*step0*(amax/step0)**(k/n_pts)
                bn = _nullcline_b(aa, b, Ap, App, Bp, Bpp, *newton_interval(lo, hi))
                if bn is None:
                    rec["reason"] = "nullcline root lost"
                    break
                b = bn; pts.append([aa, b])
            rec["trap"] = lic["trap"]
        else:
            J0 = lic["J0"]
            for k in range(1, n_pts+1):
                aa = a0 + s*step0*(amax/step0)**(k/n_pts)
                bn = _j_root(aa, b, J0, d_eff, c1, rhos, res, A, Ap, lo, hi)
                if bn is None:
                    rec["reason"] = "level set has no real b beyond here"
                    break
                b = bn
                # Approaching an attracting trap the level set pins
                # Gaussian-fast to rho while the branch rides the nullcline
                # at 2B'/(A'' a) from it; once the two agree to tol the
                # nullcline is the more accurate curve and takes over.
                # The trap test goes first: far from a trap this Newton has no
                # root to find and runs to its iteration limit.
                near_trap = min(abs(b-lo), abs(b-hi)) < 1e-3*max(1.0, abs(b))
                bnull = _nullcline_b(aa, b, Ap, App, Bp, Bpp, lo, hi) if near_trap else None
                if bnull is not None and abs(bnull-b) <= tol*math.hypot(aa, b):
                    rec["handoff_to_nullcline"] = [aa, b]
                    for kk in range(k+1, n_pts+1):
                        aa = a0 + s*step0*(amax/step0)**(kk/n_pts)
                        bnull = _nullcline_b(aa, bnull, Ap, App, Bp, Bpp,
                                             lo, hi)
                        if bnull is None:
                            break
                        pts.append([aa, bnull])
                    break
                pts.append([aa, b])
        rec["points"] = pts
        rec["trap_lo"] = None if lo == -math.inf else lo
        rec["trap_hi"] = None if hi == math.inf else hi
        out.append(rec)
    return out


def with_tails(p: Portrait, enumeration=None) -> Portrait:
    """The portrait with each stable branch continued by its growth and tail."""
    enumeration = p.enumeration if enumeration is None else enumeration
    branches = list(p.branches)
    for tail in _j_tails(p.model, p, float(atlas.effective_degree(p.model)),
                         enumeration=enumeration):
        br = branches[tail["branch"]]
        parts = [np.asarray(br.Y, dtype=float)]
        parts += [np.asarray(x, dtype=float)[1:]
                  for x in (tail.get("grown"), tail.get("points"))
                  if x is not None and len(x) > 1]
        branches[tail["branch"]] = Branch(
            br.kind, np.vstack(parts), br.term, certs=br.certs, diag=br.diag)
    return Portrait(p.model, p.enumeration, branches, p.box, p.view, p.ledger)
