#!/usr/bin/env python3
"""Backend for the interactive one-neuron loss explorer.

    python demos/explorer/serve.py        # http://127.0.0.1:8710

Every change of f, g, or the moments in the browser posts to /portrait and
gets back a freshly computed CERTIFIED portrait.  The browser does no
analysis: it evaluates L(a,b) = C - 2a*B(b) + a^2*A(b) for the heatmap and
draws what this endpoint sends.  Critical points, their classification, the
separatrices and the ledger all come from portrait.certified_compute.

The demo endpoints share that exact cached model: /trace continues one
normalized-gradient orbit, while /allocator creates equal-versus-Thompson
optimizer traces from a common design in the visible window.

Nothing here re-implements spong.  Model.alpha, Model.beta and Model.C are
the exact A, B and C; Model.backbone_num / backbone_den are the reduced
B^2/A the constructor already forms; each CriticalPoint carries b, a, kind,
source and the exact sign of u''.  This file only serializes them.

Lives in demos/ because the descent traces it drives are descent methods,
which the founding document keeps out of the library.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Default to the accelerated path.  The explorer is a viewer, not the
# certifying CLI, and latency is what it trades in; both settings are
# overridable from the environment.  Must precede the spong import, since
# engine reads SPONG_ENGINE at import time.
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "8")

try:
    from spong import atlas, inverse, model, portrait, sturm, wall_shoot, zoo
except ImportError:                                  # running from a checkout
    sys.path.insert(0, str(REPO / "src"))
    from spong import atlas, inverse, model, portrait, sturm, wall_shoot, zoo

# The allocator experiment remains demo code, but the viewer needs to run it
# against the exact model behind the active portrait.  A script launched as
# ``python demos/explorer/serve.py`` does not otherwise put the repository
# root (and hence the ``demos`` package) on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demos import optimizer_moustaches as optimizer_gallery
from demos import saddle_connection_triptych as saddle_wall
from demos import thompson_moustaches as allocator_demo

try:                                                 # exact C core
    from spong import _native
    NATIVE = {
        "loaded": True,
        "abi": getattr(_native, "SPONG_ABI_VERSION", None),
        "types": [n for n in ("Kernel", "LocalKernel",
                              "SturmPlan", "ContactScan")
                  if hasattr(_native, n)],
    }
except ImportError as _exc:                          # ABI or interpreter skew
    NATIVE = {"loaded": False, "error": str(_exc), "types": []}

PAGE = HERE / "index.html"
PORT = 8710

# Recomputing the same (f, g, mu, view) is common -- flipping between presets,
# re-focusing a coefficient field -- and a degree-11 enumeration plus branch
# tracing is not cheap.  Small LRU on the exact request.
_CACHE: dict = {}
_CACHE_MAX = 24

# Enumeration and materialized stubs are shared between the preview and the
# final stage of the same model: they are upstream of the geometry ladder and
# cost seconds on the hard cases, so recomputing them per stage would be pure
# waste.
_ENUM: dict = {}
_ENUM_MAX = 8

# Models are cached too, so /trace can integrate against the same object the
# portrait was built from without rebuilding it.
_MODELS: dict = {}
_MODELS_MAX = 8

# One Brent solve per wall family (about three seconds at ds = 5e-4),
# computed on the first wall-limit request and kept for the process.
_WALL_ROOTS: dict = {}
WALL_SHOOT_DS = 5e-4


def _wall_root(family):
    """The family's Brent root, or None if shooting is unavailable (no C
    core) or refuses (no sign change on the bracket) -- callers then fall
    back to the stored wall_parameter and the older construction."""
    if family.name in _WALL_ROOTS:
        return _WALL_ROOTS[family.name]
    try:
        root = wall_shoot.find_wall(family, ds=WALL_SHOOT_DS)
    except (ValueError, ArithmeticError, RuntimeError):
        root = None
    _WALL_ROOTS[family.name] = root
    return root


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------

def moment_vector(spec: dict, n: int):
    """Exact moments for the requested input distribution.

    'empirical' takes literal sample points from the client and forms the
    exact rational empirical moments, so a batch portrait is a genuine SPONG
    model rather than an approximation of one.
    """
    kind = spec.get("kind", "uniform01")
    if kind == "uniform01":
        return model.moments_uniform01(n)
    if kind == "normal01":
        return model.moments_normal01(n)
    if kind == "empirical":
        xs = [Fraction(float(x)) for x in spec.get("samples", [])]
        if not xs:
            raise ValueError("empirical moments need at least one sample")
        inv = Fraction(1, len(xs))
        return [inv * sum((x ** k for x in xs), Fraction(0)) for k in range(n)]
    raise ValueError(f"unknown moment kind {kind!r}")


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------

def _field_coeffs(m):
    """A(b), B(b), C as ascending float coefficient lists, plus the reduced
    backbone numerator and denominator the model already carries."""
    return {
        "A": [float(c) for c in m.alpha],
        "B": [float(c) for c in m.beta],
        "C": float(m.C),
        "backbone_num": [float(c) for c in m.backbone_num],
        "backbone_den": [float(c) for c in m.backbone_den],
    }


def _critical_points(e, m):
    """The certified inventory, verbatim from sturm.CriticalPoint.

    u2_sign is the EXACT sign of u''; no magnitude is reported, because the
    enumeration certifies the sign rather than a value.  source records which
    polynomial produced the root -- 'B' roots are the a* = 0 saddles that are
    saddles by the identity det H = -4 B'^2.

    'global' follows render.py's convention: minima within a relative
    tolerance of the least loss get an open circle; other minima get the
    equal-size filled version.
    """
    pts = sorted(e.points, key=lambda p: p.b)
    losses = {id(p): float(m.L(p.a, p.b)) for p in pts}
    mins = [losses[id(p)] for p in pts if p.kind == "min"]
    best = min(mins) if mins else None
    out = []
    for p in pts:
        L = losses[id(p)]
        is_global = (p.kind == "min" and best is not None
                     and L <= best + 1e-9 * (1.0 + abs(best)))
        out.append({"b": float(p.b), "a": float(p.a), "kind": p.kind,
                    "source": p.source, "u2_sign": int(p.u2_sign),
                    "loss": L, "global": bool(is_global)})
    return out


def _max_chord(Y) -> float:
    """Longest chord on a traced branch, in model units.

    Interpolating between vertices is licensed by the branch residuals, so the
    chord length is the scale below which the drawn curve stops being a
    certified rendering and becomes a magnified straight line.  The viewer
    reports it rather than silently zooming past it.
    """
    worst = 0.0
    for i in range(1, len(Y)):
        da = float(Y[i][0]) - float(Y[i - 1][0])
        db = float(Y[i][1]) - float(Y[i - 1][1])
        d = (da * da + db * db) ** 0.5
        if d > worst:
            worst = d
    return worst


EPS = np.finfo(float).eps


def _abs_horner(coef_desc, x):
    """Sum of |c_k x^k| -- the running-error scale of a Horner evaluation."""
    ax = abs(x)
    s = 0.0
    for c in coef_desc:
        s = s*ax + abs(c)
    return s


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
    for _ in range(80):
        F = a*np.polyval(Ap, b) - 2.0*np.polyval(Bp, b)
        floor = 8.0*EPS*(abs(a)*_abs_horner(Ap, b) + 2.0*_abs_horner(Bp, b))
        if abs(F) <= max(floor, 1e-300):
            return b
        Fp = a*np.polyval(App, b) - 2.0*np.polyval(Bpp, b)
        if Fp == 0.0:
            return None
        bn = b - F/Fp
        if not (lo < bn < hi):
            bn = b - 0.5*(b - (lo if bn <= lo else hi))
        if abs(bn - b) <= EPS*max(1.0, abs(b)):
            return bn
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
    pts, term = charts._potential_rate_box_exit(
        m, tuple(start), box, ds, {}, max_steps=min(max_steps, 50000),
        critical=critical)
    out = [(float(q[0]), float(q[1])) for q in pts[1:]]
    if term != "box_exit":
        last = pts[-1] if len(pts) else start
        b0 = float(last[1]); w0 = float(last[0] - m.a_star(b0))
        try:
            more, term, _sw, _ = charts._continue_curve(
                m, b0, w0, -1, [], box, ds,
                ds0=(ds0 if ds0 is not None else min(ds, 1e-6)),
                engine_diag={}, max_steps=max_steps)
            out.extend((float(q[0]), float(q[1])) for q in more[1:])
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
    c1, rhos, res = _j_invariant(m, d_eff)
    A = np.asarray(m._fa, dtype=float)
    Ap = np.polyder(A[::-1]); App = np.polyder(Ap)
    Bc = np.asarray(m._fb, dtype=float)
    Bp = np.polyder(Bc[::-1]); Bpp = np.polyder(Bp)
    critical = (np.array([[float(q.a), float(q.b)]
                          for q in enumeration.points], dtype=float)
                if enumeration is not None else None)

    def interval(b0):
        lo = max([z for z in real_rhos if z < b0], default=-math.inf)
        hi = min([z for z in real_rhos if z > b0], default=math.inf)
        return lo, hi

    def license(pts):
        a = np.array([q[0] for q in pts]); b = np.array([q[1] for q in pts])
        r = np.hypot(a, b); n = len(pts)
        if n < 3:
            return None
        # pinned: nullcline lag over the last radius octave (sampled)
        k0 = next((k for k in range(n) if r[k] >= 0.5*r[-1]), n-1)
        if k0 < n-1 and abs(a[-1]) > abs(a[k0]):
            lo, hi = interval(b[-1])
            ks = np.unique(np.linspace(k0, n-1, min(400, n-k0)).astype(int))
            lag = []
            for k in ks:
                bn = _nullcline_b(a[k], b[k], Ap, App, Bp, Bpp, lo, hi)
                lag.append(math.inf if bn is None else abs(b[k]-bn)/r[k])
            if max(lag) <= tol and lag[-1] <= lag[0]:
                return {"sector": "pinned", "index": n-1, "lag": max(lag),
                        "lag_end": lag[-1]}
        # diagonal: remaining J drift after k, attested over an octave
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
        lic = license(curve)
        n_grown = 0
        grow_term = None
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
                    break
                curve.extend(more)
                n_grown = len(curve) - n_traced
                lic = license(curve)
                if lic is not None or grow_term == "growth_failure":
                    break
        rec = {"branch": i, "kind": br.kind, "n_traced": n_traced,
               "n_grown": n_grown, "grow_term": grow_term,
               "grown": [[float(x), float(y)] for x, y in curve[n_traced-1:]],
               "points": [], "sector": None, "tol": tol}
        if lic is None:
            rec["reason"] = ("neither sector licensed within the traced and "
                             "grown branch")
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
                bn = _nullcline_b(aa, b, Ap, App, Bp, Bpp, lo, hi)
                if bn is None:
                    rec["reason"] = "nullcline root lost"
                    break
                b = bn; pts.append([aa, b])
            # the trap it approaches: the interval end the nullcline tends to
            rec["trap"] = (lo if abs(b-lo) < abs(b-hi) else hi) \
                if math.isfinite(lo) or math.isfinite(hi) else None
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
                bnull = _nullcline_b(aa, b, Ap, App, Bp, Bpp, lo, hi)
                if bnull is not None and abs(bnull-b) <= tol*math.hypot(aa, b) \
                        and min(abs(b-lo), abs(b-hi)) < 1e-3*max(1.0, abs(b)):
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


def _branches(p, max_pts: int = 200000):
    """Branch polylines at full traced resolution.

    The tracing is deliberately fine so that the viewer can zoom without
    recomputing -- thinning here would throw that away.  The client decimates
    at draw time against the current pixel scale instead, so the data stays
    complete and only the drawing adapts.  max_pts is a safety valve, not a
    display budget.
    """
    out = []
    for br in p.branches:
        Y = br.Y
        n = len(Y)
        step = max(1, n // max_pts)
        pts = [[float(Y[i][0]), float(Y[i][1])] for i in range(0, n, step)]
        if step > 1 and n:
            pts.append([float(Y[n - 1][0]), float(Y[n - 1][1])])
        # The asymptote certificate travels with the branch.  Interpolation
        # between traced vertices is already certified by the angle and turn
        # residuals, so zooming IN never needs a recompute; extrapolation past
        # the traced box is the only open case, and under genericity it has a
        # certificate of its own -- every separatrix leaves along a diagonal
        # b = +-sqrt(d_eff)*a, and this residual says how close the tail is.
        cert = br.certs.get("asymptote")
        out.append({"kind": br.kind, "term": br.term,
                    "saddle_b": br.diag.get("saddle_b"),
                    "direction": br.diag.get("unstable_direction"),
                    "stable_sign": br.diag.get("stable_sign"),
                    "target": br.diag.get("target"),
                    "connection_ok": bool(br.certs.get("connection_ok", False)),
                    "n_traced": n, "stride": step,
                    "chord_max": _max_chord(Y),
                    "asymptote": (None if cert is None else {
                        "slope_extrapolated": float(
                            cert["slope_extrapolated"]),
                        "target": float(cert["target"]),
                        "residual": float(cert["residual"]),
                    }),
                    "points": pts})
    return out


def _resolve(payload: dict):
    """(f, g, view, moment spec, cache key) from a request."""
    wall = payload.get("wall")
    name = payload.get("zoo")
    separated = payload.get("separated_linear")
    if wall:
        # A rheostat member of a wall family, at an arbitrary Lambda.
        # zoo.rheostat_member materializes only the three named members; the
        # scaling f/sqrt(L), g*sqrt(L) is the same one, opened up so the
        # viewer can move through the family continuously.
        #
        # The wall COORDINATE is not the citable object -- wall_bracket is,
        # an interval whose endpoints have verified opposite landing fates.
        # Inside the bracket the fate is launch-protocol-dependent, so the
        # page must show the bracket rather than imply that any particular
        # Lambda in it is "the" wall.
        w = zoo.get_wall_family(wall)
        base = zoo.get(w.base_case)
        # ``wall_limit`` names the wall itself, not a nearby slider
        # coordinate.  The coordinate used is the Brent root of the
        # two-sided shooting separation (spong.wall_shoot), which is the
        # binary64 wall to the integrator's accuracy; the family's stored
        # ``wall_parameter`` is the older fate-bisection value and can sit
        # a few 1e-9 away from it.  Ordinary slider positions use their
        # literal Lambda.
        wall_root = _wall_root(w) if payload.get("wall_limit") else None
        lam = (wall_root.lam if wall_root is not None else
               w.wall_parameter if payload.get("wall_limit") else
               float(payload.get("lam", w.wall_parameter)))
        scale = math.sqrt(lam)
        f = [float(x) / scale for x in base.f]
        g = [float(x) * scale for x in base.g]
        view = tuple(float(x) for x in w.default_view)
        spec = {"kind": base.moment_dist}
    elif name:
        z = zoo.get(name)
        f = [float(x) for x in z.f]
        g = [float(x) for x in z.g]
        view = tuple(float(x) for x in z.default_view) \
            if z.default_view else None
        spec = {"kind": z.moment_dist}
    elif separated:
        degree = int(separated.get("degree", 4))
        if not 1 <= degree <= 8:
            raise ValueError(
                "inspector separated-linear degree must lie in [1, 8]")
        raw_separation = separated.get("separation", "30")
        try:
            scale = Fraction(str(raw_separation))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("separation must be a rational number") from exc
        if not Fraction(1) < scale <= Fraction(1000):
            raise ValueError("separation must lie in (1, 1000]")
        family = inverse.separated_linear_case(degree, scale)
        # Keep the rational coefficients through model construction.  They
        # are converted only when serialized for the browser.
        f = list(family.f)
        g = list(family.g)
        view = None
        spec = {"kind": "uniform01"}
    else:
        f = [float(x) for x in payload["f"]]
        g = [float(x) for x in payload["g"]]
        view = payload.get("view")
        if view is not None:
            view = tuple(float(x) for x in view)   # (a_lo, a_hi, b_lo, b_hi)
        spec = payload.get("moments", {})
    key = (tuple(f), tuple(g), view, spec.get("kind", "uniform01"),
           tuple(float(x) for x in spec.get("samples", ())))
    return f, g, view, spec, key


def _wall_pair(payload: dict, p):
    """The two continuations a wall family is about, at ANY Lambda.

    The source saddle's unstable branch in the family's direction and the
    target saddle's stable branch nearest the source, with the closest
    approach of each to the other's critical point.  At the wall these
    coincide in the limit; away from it they miss by an amount the viewer
    should print rather than leave to the eye, since a near-connection at
    ordinary zoom and a clean miss at deep zoom are the same picture.

    Returns None outside a wall family or when the branches cannot be
    identified (a refused portrait may lack them).
    """
    name = payload.get("wall")
    if not name:
        return None
    family = zoo.get_wall_family(name)
    try:
        source = saddle_wall.critical_near(p, family.source_b)
        target = saddle_wall.critical_near(p, family.target_b)
    except (LookupError, ValueError):
        return None
    unstable = [
        (index, branch) for index, branch in enumerate(p.branches)
        if branch.kind == "unstable"
        and abs(branch.diag.get("saddle_b", math.inf)-source.b) < 1e-7
        and branch.diag.get("unstable_direction") ==
        family.unstable_direction]
    stable = [
        (index, branch) for index, branch in enumerate(p.branches)
        if branch.kind == "stable"
        and abs(branch.diag.get("saddle_b", math.inf)-target.b) < 1e-7]
    if not unstable or not stable:
        return None
    su_index, su = unstable[0]
    distance_to_target = ((su.Y[:, 0]-target.a)**2
                          + (su.Y[:, 1]-target.b)**2)
    target_index = int(distance_to_target.argmin())
    closest_target = math.sqrt(float(distance_to_target[target_index]))

    def source_distance(branch):
        return math.sqrt(float(np.min(
            (branch.Y[:, 0]-source.a)**2 + (branch.Y[:, 1]-source.b)**2)))
    ts_index, ts = min(stable, key=lambda item: source_distance(item[1]))
    return {
        "source_b": float(source.b), "target_b": float(target.b),
        "source_a": float(source.a), "target_a": float(target.a),
        "source_unstable": su_index, "target_stable": ts_index,
        "unstable_to_target": closest_target,
        "stable_to_source": source_distance(ts),
        "closest_index": target_index,
        "unstable_direction": family.unstable_direction,
    }


def _geometric_wall_limit(payload: dict, p, root=None):
    """Dedicated display geometry for the selected saddle-connection wall.

    An ordinary portrait at a non-Morse-Smale wall must refuse: its unstable
    and stable numerical continuations are not independently meaningful once
    they coincide.  Remove those two continuations and insert the connection
    candidate.

    With ``root`` (a :class:`wall_shoot.WallRoot`) the candidate is the
    glued two-sided shot at the Brent root: source saddle down to the
    midlevel, target saddle up to it, meeting there to |delta| at binary64
    resolution.  Without it -- the older construction, kept for callers
    that have no root -- the house continuation is truncated at closest
    approach and its last vertex moved onto the target, and the miss is
    reported.
    """
    name = payload.get("wall")
    if not name or not payload.get("wall_limit"):
        return p, [], None

    family = zoo.get_wall_family(name)
    pair = _wall_pair(payload, p)
    if pair is None:
        raise ValueError("wall limit: the source unstable or target stable "
                         "continuation is missing from this portrait")
    if root is not None:
        connection = np.asarray(root.shot.candidate, dtype=float)
        method = "two-sided shooting, Brent root"
        gap = abs(root.delta)
        parameter = root.lam
    else:
        source_unstable = p.branches[pair["source_unstable"]]
        target_index = pair["closest_index"]
        connection = source_unstable.Y[:target_index+1].copy()
        # The house continuation gets within its event/chord tolerance and
        # then continues into one of the adjacent chambers.  At the
        # geometric wall limit the orbit ends at the exact target critical
        # point.
        connection[-1] = (pair["target_a"], pair["target_b"])
        method = "paired house continuations"
        gap = pair["unstable_to_target"]
        parameter = float(family.wall_parameter)

    display, surgery = saddle_wall.wall_limit_portrait(
        p, family, connection)
    points = [[float(q[0]), float(q[1])] for q in connection]
    record = {
        "kind": "stable_unstable",
        "label": "saddle connection (Wu = Ws)",
        "source_b": pair["source_b"],
        "target_b": pair["target_b"],
        "n_traced": len(points),
        "points": points,
        # For the shooting candidate: |delta| at the midlevel where the two
        # shots meet.  For the older construction: how far the numerical
        # continuation missed the target before its last vertex was snapped
        # onto it.  Either way, this number is what binary64 actually did.
        "numerical_miss": float(gap),
    }
    diagnostics = {
        "geometry_method": "geometric wall limit",
        "parameter": parameter,
        "trace": {
            "method": method,
            "source_unstable_closest_to_target": pair["unstable_to_target"],
            "target_stable_closest_to_source": pair["stable_to_source"],
        },
        "surgery": surgery,
    }
    return display, [record], diagnostics


def _model_for(key, f, g, spec):
    m = _MODELS.get(key)
    if m is None:
        n_moments = 2 * max(len(f), len(g)) - 1
        m = model.build(f, g, moment_vector(spec, n_moments))
        _MODELS[key] = m
        if len(_MODELS) > _MODELS_MAX:
            _MODELS.pop(next(iter(_MODELS)))
    return m


def trace(payload: dict) -> dict:
    """Arclength continuation of the gradient field from one point.

    Uses Kernel.normalized_step -- the C core's 2-D normalized-gradient
    integrator at GEOMETRIC_IRK_PRIMARY order, the same one charts falls back
    to when both graph parameterizations go singular.  An explicit method has
    no business on this field: the stiffness that forced the whole certified
    machinery is exactly what makes a descent trajectory crawl.

    Unit speed is the point.  True gradient time never arrives on a stiff
    valley; arclength travels the SAME curve at constant speed, so a bounded
    number of steps answers "where does this initial condition go" instead of
    "how far does it get before you lose patience".
    """
    f, g, _view, spec, key = _resolve(payload)
    m = _model_for(key, f, g, spec)
    kernel = getattr(m, "_native_kernel", None)
    if kernel is None or not hasattr(kernel, "normalized_step"):
        return {"error": "normalized_step unavailable (no C core)"}

    a = float(payload["a"])
    b = float(payload["b"])
    flow = 1 if int(payload.get("flow", 1)) > 0 else -1
    ds = float(payload.get("ds", 0.01))
    steps = max(1, min(200000, int(payload.get("steps", 4000))))
    order = int(payload.get("order", 8))
    box = payload.get("box")
    box = [float(x) for x in box] if box else None

    pts = [a, b]
    term = "steps"
    for _ in range(steps):
        try:
            a_new, b_new = kernel.normalized_step(a, b, -flow * ds, order)
        except (ArithmeticError, ValueError, OverflowError,
                ZeroDivisionError):
            term = "step_failure"
            break
        if not (a_new == a_new and b_new == b_new):      # NaN
            term = "nonfinite"
            break
        if abs(a_new - a) + abs(b_new - b) < 1e-14 * ds:
            a, b = a_new, b_new
            term = "stationary"
            break
        a, b = float(a_new), float(b_new)
        pts.extend((a, b))
        if box and not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
            term = "box_exit"
            break
    return {"points": pts, "term": term, "order": order,
            "arclength": ds * (len(pts) // 2 - 1)}


def _bounded_int(payload, name, default, lower, upper):
    value = int(payload.get(name, default))
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _allocator_policy(result, color, max_points=350, *,
                      width_base=0.55, width_gain=1.2,
                      opacity_base=0.08, opacity_gain=0.72,
                      mark_start=False, mark_end=False):
    """Serialize one allocation policy as weighted viewer traces."""
    counts = result.allocations
    largest = max(int(max(counts)), 1)
    traces = []
    for arm, (trajectory, pulls) in enumerate(
            zip(result.trajectories, counts)):
        raw_points = optimizer_gallery._thin(
            trajectory, max_points=max_points).tolist()
        # json.dumps would otherwise emit non-standard NaN/Infinity tokens
        # when a deliberately ill-scaled optimizer escapes.  Stop a drawn
        # trajectory at its last finite point, as the standalone gallery does
        # visually, while the allocation and final observation still record
        # the failed continuation.
        points = []
        for point in raw_points:
            if len(point) != 2 or not all(math.isfinite(float(x))
                                          for x in point):
                break
            points.append([float(point[0]), float(point[1])])
        share = float(pulls) / largest
        traces.append({
            "arm": arm,
            "pulls": int(pulls),
            "executed_steps": int(result.executed_steps[arm]),
            "termination": result.termination_reasons[arm],
            "points": points,
            "color": color,
            "width": width_base + width_gain * share,
            "opacity": opacity_base + opacity_gain * math.sqrt(share),
            "mark_start": bool(mark_start),
            "mark_end": bool(mark_end),
        })
    return {
        **allocator_demo._allocation_summary(result),
        "traces": traces,
        "allocations": [int(x) for x in counts],
        "final_observations": [
            (float(values[-1]) if values else None)
            for values in result.observations],
    }


def allocator(payload: dict) -> dict:
    """Equal-versus-Thompson traces for the active portrait model.

    The request uses the same model selector as ``/portrait`` and ``/trace``.
    ``allocation_view`` is deliberately separate: it is the visible window
    from which the browser asks us to place the common initialization design,
    not a request to rebuild or recertify the portrait.
    """
    f, g, resolved_view, spec, key = _resolve(payload)
    m = _model_for(key, f, g, spec)

    raw_view = payload.get("allocation_view", resolved_view)
    if raw_view is None or len(raw_view) != 4:
        raise ValueError("allocation_view must contain [a0, a1, b0, b1]")
    view = tuple(float(x) for x in raw_view)
    if not all(math.isfinite(x) for x in view):
        raise ValueError("allocation_view must be finite")
    if not (view[0] < view[1] and view[2] < view[3]):
        raise ValueError("allocation_view must have increasing bounds")

    starts = _bounded_int(payload, "starts", 32, 1, 512)
    rounds = _bounded_int(payload, "rounds", 3200, starts, 250000)
    chunk_steps = _bounded_int(payload, "chunk_steps", 10, 1, 1000)
    batch_size = _bounded_int(payload, "batch_size", 32, 1, 100000)
    seed = _bounded_int(payload, "seed", 1729, 0, 2**32 - 1)
    if rounds * chunk_steps > 2_000_000:
        raise ValueError("rounds * chunk_steps must not exceed 2,000,000")

    method = str(payload.get("method", "adam"))
    schedule = str(payload.get("schedule", "inverse-sqrt"))
    design = str(payload.get("design", "low-discrepancy"))
    # "auto" calibrates each arm from its own exact gradient at its own
    # start, so the opening displacement is a fraction of the visible window
    # rather than a fixed multiple of whatever gradient the arm landed on.
    # SGD needs this: it moves lr*|grad L| per step with nothing normalizing
    # it, so an arm starting high on a quartic wall takes an enormous first
    # jump.  A numeric rate is still honoured verbatim.
    raw_lr = payload.get(
        "learning_rate", optimizer_gallery.DEFAULT_LR.get(method, 1e-3))
    if isinstance(raw_lr, str) and raw_lr.strip().lower() == "auto":
        learning_rate = "auto"
    else:
        learning_rate = float(raw_lr)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
    time_limit_sec = float(payload.get("time_limit_sec", 20.0))
    if not math.isfinite(time_limit_sec) or not 1.0 <= time_limit_sec <= 300.0:
        raise ValueError("time_limit_sec must be finite and in [1, 300]")

    distribution = spec.get("kind", "uniform01")
    samples = spec.get("samples") if distribution == "empirical" else None
    cb_library = allocator_demo.cb_sampler.resolve_library(auto_build=True)
    t0 = time.perf_counter()
    deadline = t0 + time_limit_sec
    comparison = allocator_demo.compare_allocators(
        m, f, g, view, starts=starts, rounds=rounds,
        chunk_steps=chunk_steps, batch_size=batch_size, method=method,
        schedule=schedule, design=design, seed=seed,
        distribution=distribution, samples=samples,
        learning_rate=learning_rate, cb_library=cb_library,
        should_stop=lambda: time.perf_counter() >= deadline)
    elapsed = time.perf_counter() - t0

    return {
        "format": "spong-equal-thompson-traces-v1",
        "observation": "L/(1+L)",
        "selection": "minimum exact continuous-Bernoulli posterior draw",
        "capture_used_for_allocation": False,
        "allocation_view": list(view),
        "elapsed_sec": elapsed,
        "work_loss_histogram": allocator_demo.work_loss_histogram(
            comparison["equal"], comparison["thompson"]),
        "configuration": {
            "starts": starts,
            "rounds": rounds,
            "chunk_steps": chunk_steps,
            "total_optimizer_steps_per_policy": rounds * chunk_steps,
            "batch_size": batch_size,
            "method": method,
            "schedule": schedule,
            "warmup_steps": comparison["warmup_steps"],
            # A list when calibrated per arm; a scalar when fixed.
            "learning_rate": comparison["learning_rate"],
            "learning_rate_mode": ("auto" if learning_rate == "auto"
                                   else "fixed"),
            "time_limit_sec": time_limit_sec,
            "design": design,
            "seed": seed,
            "distribution": distribution,
            "cb_library": comparison["library_path"],
        },
        # Equal is deliberately a broad, half-transparent underlay.  The
        # narrower saturated Thompson trace then remains legible while the
        # common initial portion still reads as an overlap.
        "equal": _allocator_policy(
            comparison["equal"], "#66727b",
            width_base=0.85, width_gain=1.30,
            opacity_base=0.14, opacity_gain=0.28),
        "thompson": _allocator_policy(
            comparison["thompson"], "#5b21b6",
            opacity_base=0.10, opacity_gain=0.80,
            mark_start=True, mark_end=True),
    }


def isolated_allocator(payload: dict) -> dict:
    """Run the interactive allocator beyond a killable process boundary.

    The cooperative deadline inside :func:`allocator` keeps ordinary runs
    cheap to stop.  This outer deadline also covers a native sampler call
    that never returns to Python.
    """
    raw_limit = payload.get("time_limit_sec", 20.0)
    try:
        requested_limit = float(raw_limit)
    except (TypeError, ValueError):
        requested_limit = 20.0
    if not math.isfinite(requested_limit):
        requested_limit = 20.0
    hard_limit = min(300.0, max(1.0, requested_limit)) + 2.0

    command = [sys.executable, str(Path(__file__).resolve()),
               "--allocator-worker"]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload), timeout=hard_limit)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError(
            "allocation stopped by hard interactive time limit") from None

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        detail = stderr.strip() or stdout.strip() or "allocator worker failed"
        raise RuntimeError(detail) from exc
    if envelope.get("ok"):
        return envelope["result"]
    message = envelope.get("error", "allocator worker failed")
    if envelope.get("kind") == "allocation_timeout":
        raise TimeoutError(message)
    if envelope.get("kind") == "value_error":
        raise ValueError(message)
    raise RuntimeError(message)


def allocator_worker_main() -> int:
    """Private JSON transport used by :func:`isolated_allocator`."""
    try:
        payload = json.load(sys.stdin)
        envelope = {"ok": True, "result": allocator(payload)}
    except TimeoutError as exc:
        envelope = {"ok": False, "kind": "allocation_timeout",
                    "error": str(exc)}
    except ValueError as exc:
        envelope = {"ok": False, "kind": "value_error",
                    "error": str(exc)}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        envelope = {"ok": False, "kind": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}"}
    json.dump(envelope, sys.stdout)
    return 0


def compute(payload: dict) -> dict:
    # A zoo case supplies its own f, g, moment distribution and default_view.
    # Using them is not a convenience: default_view is tuned per case, and
    # _trace_box widens whatever view it is handed, so an invented box costs
    # real tracing time.  This is the cli.zoo_phase_portrait path.
    f, g, view, spec, key = _resolve(payload)

    # Two stages.  'preview' is geometry_level 0 only -- the picture, with no
    # escalation ladder behind it; 'final' runs certified_compute to a verdict.
    # On linear-target-d17-thrash level 0 is about 95s of an 808s portrait, and
    # levels 1 and 2 refine the VERDICT rather than the curves, so blocking the
    # display on them makes the viewer unusable on exactly the cases it is most
    # wanted for.
    stage = payload.get("stage", "final")

    display_mode = "wall_limit" if payload.get("wall_limit") else "ordinary"
    # A response computed before the family's Brent root existed lacks the
    # root and the candidate overlay; once the root exists, recompute.
    if _WALL_ROOTS.get(payload.get("wall")) is not None:
        display_mode += "+root"
    hit = _CACHE.get((key, stage, display_mode))
    if hit is not None:
        return dict(hit, cached=True)

    t0 = time.perf_counter()
    n_moments = 2 * max(len(f), len(g)) - 1     # exactly mu_0..mu_2D, as cli
    mu = moment_vector(spec, n_moments)
    t1 = time.perf_counter()

    m = _model_for(key, f, g, spec)
    # The box contract is view subset of compute box subset of legal max, and
    # the repo policy is a compute box as big as is sensible.
    # atlas.legal_max_b is that bound: past the Cauchy bounds of N and B every
    # finite critical point is enclosed and the far field owns the dynamics.
    # But compute_box only CLAMPS to it -- it builds the skeleton's bounding
    # box plus a margin, unioned with whatever view it is handed -- so an
    # interactive request sized to the current view gets a box sized to the
    # view, and stable separatrices are cut off long before their diagonal
    # asymptotes take hold.
    #
    # Taking the legal box costs nothing in steps: ds = span/30000 is derived
    # from the box, so the count to cross it is fixed whatever its size.  A
    # larger box buys reach at proportionally coarser chords, and 30000 chords
    # across the legal box is still thousands across a typical view.  The view
    # the client asked for travels separately, as frame_view, for framing.
    bmax = atlas.legal_max_b(m)
    amax = bmax / max(1.0, math.sqrt(max(1, atlas.effective_degree(m))))
    trace_view = (-amax, amax, -bmax, bmax)
    enumeration = _ENUM.get(key)
    if enumeration is None:
        enumeration = sturm.materialize_stubs(
            m, sturm.enumerate_critical_points(m))
        _ENUM[key] = enumeration
        if len(_ENUM) > _ENUM_MAX:
            _ENUM.pop(next(iter(_ENUM)))
    t2 = time.perf_counter()

    if stage == "preview":
        # Geometry only, no audit.  The audit is the cost -- 714s of an 808s
        # level-0 portrait on linear-target-d17-thrash -- and the viewer draws
        # curves, not certificates.  The status comes back `not_audited` and
        # the page must say so rather than imply a verdict.
        display_view = atlas.compute_box(m, enumeration, view=trace_view)
        p = portrait.compute(
            m, view=trace_view, geometry_level=0, _enumeration=enumeration,
            _display_view=display_view, _genericity=atlas.genericity(m),
            _skip_audit=True)
    else:
        p = portrait.certified_compute(m, view=trace_view,
                                       _enumeration=enumeration)
    # The pair a wall family is about, identified on the ORDINARY portrait
    # before any wall-limit surgery removes one of them.
    wall_pair = _wall_pair(payload, p)
    wall_name = payload.get("wall")
    family = zoo.get_wall_family(wall_name) if wall_name else None
    root = (_wall_root(family)
             if family is not None and payload.get("wall_limit") else
             _WALL_ROOTS.get(wall_name) if wall_name else None)
    # The two-sided shot at THIS Lambda: source unstable down and target
    # stable up to the common midlevel, with the signed gap.  Smooth in
    # Lambda and zero exactly at the wall -- the honest "how close do they
    # come" number, and what the Brent root is a root of.
    wall_shot = None
    if family is not None:
        shot = wall_shoot.shoot(
            m, family.source_b, family.unstable_direction, family.target_b,
            root.target_direction if root is not None else None,
            ds=WALL_SHOOT_DS, enumeration=enumeration)
        wall_shot = shot.as_dict() if shot is not None else None
    p, wall_connections, wall_limit = _geometric_wall_limit(payload, p, root)
    if wall_limit:
        # After surgery the branch indices no longer apply; the connection
        # record carries the distances instead.
        wall_pair = None
    wall_root = None if root is None else {
        "lam": root.lam, "delta": root.delta,
        "evaluations": root.evaluations, "bracket": list(root.bracket),
        "stored_wall": float(family.wall_parameter),
        "stored_bracket": (list(family.wall_bracket)
                           if family.wall_bracket else None),
        "ds": WALL_SHOOT_DS,
        "level": root.shot.level,
        "candidate_points": root.shot.candidate.tolist(),
    }
    t3 = time.perf_counter()
    elapsed = t3 - t0
    timing = {"moments": t1 - t0, "build": t2 - t1, "portrait": t3 - t2}

    mean = sum((Fraction(f[i]) * mu[i] for i in range(len(f))), Fraction(0))
    csq = sum((Fraction(f[i]) * Fraction(f[j]) * mu[i + j]
               for i in range(len(f)) for j in range(len(f))), Fraction(0))
    varf = float(csq - mean * mean)

    e = p.enumeration
    out = {
        "f": [float(x) for x in f], "g": [float(x) for x in g],
        "separated_linear": ({
            "degree": int(payload["separated_linear"].get("degree", 4)),
            "separation": str(payload["separated_linear"].get(
                "separation", "30")),
            "algebraic_ceiling": 4 * int(
                payload["separated_linear"].get("degree", 4)) - 2,
        } if payload.get("separated_linear") else None),
        "stage": stage,
        "varf": varf if varf > 0 else 1.0,
        "elapsed_sec": elapsed,
        "timing": timing,
        "n_branch_points": sum(len(br.Y) for br in p.branches),
        "n_branches": len(p.branches),
        "chord_max": max((_max_chord(br.Y) for br in p.branches), default=0.0),
        "native": NATIVE,
        "cached": False,
        "field": _field_coeffs(m),
        "critical": _critical_points(e, m),
        "branches": _branches(p),
        # Closed-form stable tails (J level set or b-nullcline), each with
        # the growth that licensed it, for drawing only.
        "tails": _j_tails(m, p, float(atlas.effective_degree(m)),
                          enumeration=enumeration),
        "wall_connections": wall_connections,
        "wall_limit": wall_limit,
        "wall_pair": wall_pair,
        "wall_shot": wall_shot,
        "wall_root": wall_root,
        "box": [float(x) for x in p.box],
        "view": None if p.view is None else [float(x) for x in p.view],
        "enumeration": {
            "n_critical": len(e.points),
            "n_min": len(e.minima),
            "n_saddle": len(e.saddles),
            "psi_positive": bool(e.psi_positive),
            "morse": bool(e.morse),
            "alternates": bool(e.alternates),
        },
        "ledger_summary": (p.ledger or {}).get("summary", {}),
        "status": ("geometric_wall_limit" if wall_limit else
                   (p.ledger or {}).get("topology", {}).get("status")),
        "ordinary_status": (p.ledger or {}).get(
            "topology", {}).get("status") if wall_limit else None,
        "d_eff": atlas.effective_degree(m),
        # What the client asked to LOOK at, distinct from what was traced.
        "frame_view": (list(view) if view is not None else None),
        "legal_box": list(trace_view),
        "reason": ("saddle_connection_wall" if wall_limit else
                   (p.ledger or {}).get(
                       "topology", {}).get("resolution_reason")),
        # The library measures itself: enumeration / stub / geometry split,
        # and one entry per geometry escalation with its status and reason.
        "ledger_timing": (p.ledger or {}).get("timing", {}),
        "attempts": (p.ledger or {}).get("topology", {}).get("attempts", []),
        "geometry_level": (p.ledger or {}).get(
            "topology", {}).get("geometry_level"),
    }
    _CACHE[(key, stage, display_mode)] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    return out


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # the page reloaded or abandoned the request while we were
            # still computing; nothing to report
            pass

    def do_GET(self):
        if self.path == "/walls":
            out = []
            for nm in zoo.wall_family_names():
                w = zoo.get_wall_family(nm)
                out.append({
                    "name": nm, "base_case": w.base_case,
                    "parameter_name": w.parameter_name,
                    "below": w.below_parameter, "wall": w.wall_parameter,
                    "above": w.above_parameter,
                    "bracket": (list(w.wall_bracket) if w.wall_bracket
                                else None),
                    "bracket_protocol": w.bracket_protocol,
                    "description": w.description,
                })
            self._send(200, json.dumps(out), "application/json")
            return
        if self.path == "/zoo":
            cases = []
            for nm in zoo.names():
                z = zoo.get(nm)
                cases.append({"name": nm,
                              "description": getattr(z, "description", ""),
                              "moment_dist": getattr(z, "moment_dist", ""),
                              "deg_f": len(z.f) - 1, "deg_g": len(z.g) - 1})
            self._send(200, json.dumps(cases), "application/json")
            return
        if self.path in ("/", "/index.html"):
            if not PAGE.exists():
                self._send(404, f"missing {PAGE.name}", "text/plain")
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/portrait", "/trace", "/allocator"):
            self._send(404, "not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            result = (trace(payload) if self.path == "/trace"
                      else isolated_allocator(payload)
                      if self.path == "/allocator"
                      else compute(payload))
            self._send(200, json.dumps(result), "application/json")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError as exc:
            self._send(408, json.dumps({
                "error": str(exc), "kind": "allocation_timeout",
            }), "application/json")
        except Exception as exc:
            traceback.print_exc()
            self._send(400, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}), "application/json")


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    core = (f"C core abi={NATIVE['abi']} "
            f"[{', '.join(NATIVE['types'])}]" if NATIVE["loaded"]
            else f"NO C CORE -- {NATIVE['error']} "
                 f"(pure-Python fallback: expect this to be slow)")
    print(f"spong explorer on http://127.0.0.1:{port}\n  {core}\n"
          f"  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(allocator_worker_main()
                     if "--allocator-worker" in sys.argv else main())
