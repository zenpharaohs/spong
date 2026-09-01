"""One invariance equation, several charts, one dispatcher.

SPONG_FOUNDING Part II, sections 6-7.  In the deviation chart
(b, w = a - a*(b)) the descent flow is

    v_b = -P,   v_w = -2Aw + a*'·P,    P = u' + A'w² - 2Aw·a*'

and an invariant manifold is a solution of the parametrization-free
invariance equation, served by whichever chart is well-posed:

  * SLOW GRAPH w(b) — shallow water (sounding κ = 2A/|u''| ≥ κ_hi):
    Hadamard fixed point  w ← P·(a*' + w')/(2A), contraction rate ~1/κ,
    first iterate w₁ = a*'u'/(2A).  The multiplicative form is REQUIRED
    here (the divided ODE form is catastrophic cancellation in this zone).
  * SLOW GRAPH ODE  dw/db = 2Aw/P - a*' — deep water, marched with Gauss.
  * FAST GRAPH ODE  db/dw = P/(2Aw - a*'P) — steep segments (separatrix
    launches, post-fold arcs), marched with Gauss.

The CONTINUATION ENGINE walks a physical trajectory through chart pieces:
it integrates the active chart until the velocity ratio says the other
chart is better conditioned (a fold in the active one), switches, and
continues — "there is no magic tracer; there is one invariance equation
and a dispatcher."  Certificates per branch: angle-energy, seam residual
(fixed point vs ODE in the overlap channel), Richardson endpoint
agreement, capture/exit data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gauss
from .model import Model

KAPPA_HI = 1e4          # shallow-water threshold (fixed point owns κ ≥ this)
KAPPA_LO = 1e2          # channel floor for the seam certificate
KAPPA_EXIT = 1e3        # shallow-zone exit (hysteresis: enter at HI, leave
                        # below EXIT, so the zone loop cannot ping-pong)
R_SWITCH = 20.0         # velocity-ratio chart handoff threshold
MAX_SWITCHES = 12
TURN_MAX = float(np.cos(np.radians(0.75)))   # engine turn budget (cosine)
UNSTABLE_LAUNCH_REL = 1e-6
STABLE_LAUNCH_DELTA = 1e-4
GEOMETRIC_IRK_PRIMARY = 8

# Step-length cap near critical points for the potential-rate phases.  Those
# phases step in LOSS with the field grad(L)/|grad(L)|^2, which is singular
# at every critical point: near one, |grad L| ~ lambda*r, so a loss step h
# is an arclength h/|grad L| and a modest h crosses the whole lingering
# region in one step.  Observed (2026-08-30): a stable branch 2.6e-9 from a
# wall stepped straight through the source saddle onto the opposite
# unstable ray, and every downstream check -- Richardson, Hermite, the
# turn budget -- passed, because the overshooting step is itself straight.
# The flow's curvature radius near a critical point is ~r, so an arclength
# step of tan(theta)*r turns by about theta.  Nominal theta = 3.75 degrees,
# a quarter of the audit's 15-degree attestation angle: the loss step is
# sized with |grad L| at the step's START, and |grad L| falls toward the
# closest approach, so the realised arclength runs 2-2.5x the nominal
# (measured 15-20 degrees per vertex at tan(7.5 degrees) on a passage
# 1e-7 from a saddle).  At this setting the realised turn stays under
# ~10 degrees, inside the audit's bound with margin, at ~36 vertices per
# e-fold of approach -- a passage in to 1e-5 and back is a few hundred
# vertices, not a picture.
CRITICAL_STEP_FRACTION = float(np.tan(np.radians(3.75)))

# Minimum significant digits the DIRECTION of grad L must carry for a vertex to
# contribute to angle_energy.  R = |grad L| / g_floor; below this the vertex
# reports evaluation noise as geometry.  Measured choice -- see
# angle_energy_detail.
ANGLE_DIGIT_BUDGET = 1e3

# --------------------------------------------------------------------- #
# soundings, velocities, launch data                                     #
# --------------------------------------------------------------------- #


def _sym2_eigh(H: np.ndarray):
    """Closed-form eigenpairs for a real symmetric 2x2 matrix.

    Returns eigenvalues in ascending order and a 2x2 matrix whose columns are
    the corresponding unit eigenvectors, matching the small slice of
    np.linalg.eigh that saddle_frame needs without invoking LAPACK.
    """
    a = float(H[0, 0])
    b = float(0.5 * (H[0, 1] + H[1, 0]))
    d = float(H[1, 1])
    mid = 0.5 * (a + d)
    rad = float(np.hypot(0.5 * (a - d), b))
    lo, hi = mid-rad, mid+rad
    determinant = a*d-b*b
    # Recover the smaller-magnitude eigenvalue from the determinant instead
    # of subtracting two nearly equal binary64 numbers.
    if abs(hi) >= abs(lo) and hi != 0.0:
        lo = determinant/hi
    elif lo != 0.0:
        hi = determinant/lo
    lam = np.array([lo, hi], dtype=float)

    if abs(b) <= 1e-300:
        if a <= d:
            V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        else:
            V = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        return lam, V

    cols = []
    for l in lam:
        if abs(b) > abs(l - a):
            v = np.array([l - d, b], dtype=float)
        else:
            v = np.array([b, l - a], dtype=float)
        cols.append(v / np.hypot(v[0], v[1]))
    return lam, np.column_stack(cols)


def sounding(m: Model, b):
    """Spectral ratio κ = 2A/|u''| — DIAGNOSTIC only (lies at u-inflections;
    kept for reporting).  The dispatcher's depth gauge is depth_gauge()."""
    return 2.0 * m.A(b) / np.maximum(np.abs(m.u_pp(b)), 1e-300)


def depth_gauge(m: Model, b, w):
    """THE depth gauge: the slow-RHS cancellation ratio.

        cond = (|2Aw| + |a*'·P|) / |2Aw − a*'·P|

    (the RHS-term ratio, multiplied through by |P|).  This measures what
    actually fails in shallow water — evaluation cancellation of
    dw/db = 2Aw/P − a*', whose two terms balance exactly on the slaved
    floor — rather than a spectral proxy.  On the floor it ≈ 2κ
    generically, so thresholds carry over from the spectral gauge; at
    u-inflections (where κ spikes falsely) the higher-order terms keep it
    finite.  See docs/kahan_riccati.md, gauge slate item 4.
    """
    A = m.A(b)
    asp = m.a_star_p(b)
    Pv = m.u_p(b) + m.Ap(b) * w**2 - 2.0 * A * w * asp
    t1 = np.abs(2.0 * A * w)
    t2 = np.abs(asp * Pv)
    num = t1 + t2
    den = np.abs(2.0 * A * w - asp * Pv)
    return num / np.maximum(den, 1e-16 * num + 1e-300)


def depth_gauge_floor(m: Model, b):
    """Depth gauge ON the slaved floor, in closed form:  2|a*'| / |w₁'|.

    On the floor the slow-RHS denominator is the physical slaved slope, so
    the cancellation ratio reduces to this analytic expression — which is
    also what the pointwise depth_gauge reads at the CONVERGED floor.
    (Evaluating depth_gauge at the first iterate w₁ instead systematically
    underreads and desynchronizes the zone extents from the engine
    trigger — measured as instant zone ping-pong.)  0/0 at critical
    points themselves; callers fill those entries from neighbors.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        A = m.A(b)
        asp, aspp = m.a_star_p(b), m.a_star_pp(b)
        up, upp = m.u_p(b), m.u_pp(b)
        w1p = (aspp * up + asp * upp) / (2.0 * A) \
            - asp * up * m.Ap(b) / (2.0 * A**2)
        gauge = 2.0 * np.abs(asp) / np.maximum(
            np.abs(w1p), 1e-16 * np.abs(asp) + 1e-300)
    return np.where(np.isfinite(gauge), gauge, np.inf)


def velocities(m: Model, b, w):
    """Descent velocities (v_b, v_w) in the deviation chart."""
    Pv = m.P_of(b, w)
    return -Pv, -2.0 * m.A(b) * w + m.a_star_p(b) * Pv


def saddle_frame(m: Model, b_s: float, a_s: float, local=None):
    """Closed-form 2x2 Hessian launch data at a saddle, in chart components.

    Returns dict with unstable/stable eigenvectors as (d_w, d_b) pairs:
    d_w = v1 - a*'·v2, d_b = v2 for eigvec (v1, v2) of the Hessian.
    Chart selection at launch uses these components (never the stiff-limit
    formulas): the eigvecs are orthogonal, so at least one chart is
    well-posed for each manifold at every saddle.
    """
    H = (np.asarray(local.hessian, dtype=float) if local is not None
         else m.hessL(a_s, b_s))
    if local is not None:
        lam = np.asarray(local.spectral.eigenvalues)
        V = np.asarray(local.spectral.frame)
    else:
        lam, V = _sym2_eigh(H)
    # At a critical point H_ab = -H_aa a*'.  This extracts the chart
    # derivative from the already-conditioned local jet.
    asp = (-H[0, 1] / H[0, 0] if local is not None
           else m.a_star_p(b_s))
    out = {}
    for name, idx in (("unstable", int(np.argmin(lam))),
                      ("stable", int(np.argmax(lam)))):
        v = V[:, idx]
        out[name] = {"lam": lam[idx],
                     "d_w": v[0] - asp * v[1],
                     "d_b": v[1]}
    return out


def _critical_chart_curve(local, manifold: str, sign: int,
                          component: str, extent: float,
                          order: int = 6) -> tuple[np.ndarray, dict]:
    """Continue a saddle branch in its centered finite-jet chart.

    The launch radius is selected by an expected-linear-flow inequality, not
    by the magnitude of the global coordinates.  Every IRK stage thereafter
    is evaluated by the native centered kernel.
    """
    H = np.asarray(local.hessian, dtype=float)
    lam = np.asarray(local.spectral.eigenvalues)
    V = np.asarray(local.spectral.frame)
    idx = int(np.argmin(lam) if manifold == "unstable" else np.argmax(lam))
    v = V[:, idx]
    asp = -H[0, 1] / H[0, 0]
    chart_v = v[1] if component == "b" else v[0] - asp * v[1]
    if sign * chart_v < 0:
        v = -v
        chart_v = -chart_v
    if abs(chart_v) < 1e-14:
        raise ValueError("critical chart is singular in requested component")

    # Find a point at which the nonlinear remainder is at most 1e-10 of the
    # exact linear term.  This is the launch analogue of expected descent.
    r = abs(extent / chart_v)
    ratio = np.inf
    for _ in range(60):
        d = r * v
        g = np.asarray(local.gradient(float(d[0]), float(d[1])))
        linear = H @ d
        ratio = float(np.hypot(*(g-linear))
                      / max(np.hypot(*linear), 1e-300))
        if ratio <= 1e-10:
            break
        r *= 0.5
    if ratio > 1e-8:
        raise ArithmeticError("unable to resolve a linear local launch")

    z = r * v
    flow_sign = -1.0 if manifold == "unstable" else 1.0
    chord = max(abs(extent) / 64.0, np.finfo(float).tiny)
    cur = min(chord, max(r, np.finfo(float).tiny))
    points = [np.array([0.0, 0.0]), z.copy()]
    for _ in range(4096):
        value = z[1] if component == "b" else z[0] - asp * z[1]
        remaining = abs(extent) - abs(value)
        if remaining <= 2e-12 * abs(extent):
            break
        hmag = min(cur, remaining)
        zn = None
        for _retry in range(16):
            trial = np.asarray(local.normalized_step(
                float(z[0]), float(z[1]), float(flow_sign * hmag), order))
            if np.all(np.isfinite(trial)):
                zn = trial
                break
            hmag *= 0.5
        if zn is None:
            raise ArithmeticError("native critical-chart IRK step failed")
        z = zn
        points.append(z.copy())
        cur = min(chord, 1.5 * hmag)
    else:
        raise ArithmeticError("critical chart exceeded step budget")

    Y = np.asarray(points)
    Y[:, 0] += local.a
    Y[:, 1] += local.b
    # Several centered steps may map to the same binary64 global point when
    # the critical coordinate is large.  They are valid local states but not
    # distinct polyline vertices; retaining them creates zero chords.
    if len(Y) > 1:
        keep = np.r_[True, np.any(Y[1:] != Y[:-1], axis=1)]
        Y = Y[keep]
    return Y, {"critical_chart": True, "critical_order": order,
               "critical_launch_radius": float(r),
               "critical_launch_nonlinearity": ratio,
               "critical_steps": len(points) - 1}


# --------------------------------------------------------------------- #
# slow-graph fixed point (shallow water)                                 #
# --------------------------------------------------------------------- #


def _dw_db(w: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
    """w' for the graph transform: 4th-order on a uniform grid, else 2nd.

    The transform's ONLY error is this derivative, and it is damped by the
    contraction factor 1/κ — so the stencil order sets the whole method's
    order.  Measured over 84 stretches, going 2nd → 4th costs no extra
    iterations at all and buys 7e4× at κ ∈ [1e4,1e6), reaching machine-exact
    from 1e6 up.  6th order adds ~2.6× over 4th and is not worth the width.

    The production caller (`trace_unstable`) always passes `np.linspace`, but
    this stays correct for any grid: non-uniform or short input falls back.
    """
    n = w.size
    if n < 5:
        return np.gradient(w, b_grid, edge_order=2)
    h = float(b_grid[1] - b_grid[0])
    d = np.diff(b_grid)
    if not np.all(np.abs(d - h) <= 1e-12 * abs(h)):     # non-uniform
        return np.gradient(w, b_grid, edge_order=2)
    out = np.empty_like(w)
    out[2:-2] = (w[:-4] - 8.0 * w[1:-3] + 8.0 * w[3:-1] - w[4:]) / (12.0 * h)
    out[0] = (-25.0*w[0] + 48.0*w[1] - 36.0*w[2] + 16.0*w[3] - 3.0*w[4]) / (12.0*h)
    out[1] = (-3.0*w[0] - 10.0*w[1] + 18.0*w[2] - 6.0*w[3] + w[4]) / (12.0*h)
    out[-1] = (25.0*w[-1] - 48.0*w[-2] + 36.0*w[-3] - 16.0*w[-4] + 3.0*w[-5]) / (12.0*h)
    out[-2] = (3.0*w[-1] + 10.0*w[-2] - 18.0*w[-3] + 6.0*w[-4] - w[-5]) / (12.0*h)
    return out


def _slow_fixed_point_python(m: Model, b_grid: np.ndarray, tol: float = 1e-13,
                             max_iter: int = 40):
    """Hadamard graph transform on a grid: w ← P·(a*' + w')/(2A).

    Returns (w, iterations, final relative change).  Converges at rate
    ~1/κ; caller guarantees κ ≥ KAPPA_LO on the grid.
    """
    A, Ap = m.A(b_grid), m.Ap(b_grid)
    asp = m.a_star_p(b_grid)
    up = m.u_p(b_grid)

    w = asp * up / (2.0 * A)                    # w₁, closed form
    rel = np.inf
    for it in range(1, max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            wp = _dw_db(w, b_grid)
            Pv = up + Ap * w**2 - 2.0 * A * w * asp
            w_new = Pv * (asp + wp) / (2.0 * A)
        if not np.all(np.isfinite(w_new)):
            return w, it, np.inf        # diverged: caller must reject
        rel = float(np.max(np.abs(w_new - w))
                    / max(float(np.max(np.abs(w_new))), 1e-300))
        w = w_new
        if rel < tol:
            break
    return w, it, rel


def slow_fixed_point(m: Model, b_grid: np.ndarray, tol: float = 1e-13,
                     max_iter: int = 40):
    """Production Hadamard graph transform; Python remains the parity oracle."""
    native = getattr(m, "_native_kernel", None)
    if native is not None and len(b_grid) >= 5:
        h = float(b_grid[1] - b_grid[0])
        if np.all(np.abs(np.diff(b_grid) - h) <= 1e-12 * abs(h)):
            w, it, rel = native.slow_fixed_point(b_grid, tol, max_iter)
            return np.asarray(w, dtype=float), it, rel
    return _slow_fixed_point_python(m, b_grid, tol, max_iter)


# --------------------------------------------------------------------- #
# chart ODEs (right-hand sides + analytic Jacobians for Gauss stages)    #
# --------------------------------------------------------------------- #


def _P_and_derivs(m: Model, b, w):
    A, Ap, App = m.A(b), m.Ap(b), m.App(b)
    asp, aspp = m.a_star_p(b), m.a_star_pp(b)
    Pv = m.u_p(b) + Ap * w**2 - 2.0 * A * w * asp
    P_w = 2.0 * Ap * w - 2.0 * A * asp
    P_b = m.u_pp(b) + App * w**2 - 2.0 * w * (Ap * asp + A * aspp)
    return A, Ap, asp, aspp, Pv, P_w, P_b


def slow_rhs(m: Model):
    def F(b, w):
        A, _, asp, _, Pv, _, _ = _P_and_derivs(m, b, float(w[0]))
        return np.array([2.0 * A * w[0] / Pv - asp])

    def J(b, w):
        A, _, asp, _, Pv, P_w, _ = _P_and_derivs(m, b, float(w[0]))
        return np.array([[2.0 * A / Pv - 2.0 * A * w[0] * P_w / Pv**2]])

    return F, J


def fast_rhs(m: Model):
    def F(w, b):
        A, _, asp, _, Pv, _, _ = _P_and_derivs(m, float(b[0]), w)
        return np.array([Pv / (2.0 * A * w - asp * Pv)])

    def J(w, b):
        A, Ap, asp, aspp, Pv, P_w, P_b = _P_and_derivs(m, float(b[0]), w)
        D = 2.0 * A * w - asp * Pv
        D_b = 2.0 * Ap * w - aspp * Pv - asp * P_b
        return np.array([[(P_b * D - Pv * D_b) / D**2]])

    return F, J


# ---- Tier-0 scalar fast path (pure floats; the engine loop lives here) ---

def _s_P(m: Model, b: float, w: float) -> float:
    return (m.s_u_p(b) + m.sAp(b) * w * w
            - 2.0 * m.sA(b) * w * m.s_a_star_p(b))


def _s_velocities(m: Model, b: float, w: float):
    Pv = _s_P(m, b, w)
    return -Pv, -2.0 * m.sA(b) * w + m.s_a_star_p(b) * Pv


def slow_rhs_s(m: Model):
    cache = [None]

    def vals(b: float, w: float):
        key = (b, w)
        if cache[0] is not None and cache[0][0] == key:
            return cache[0][1]
        A, Ap = m.sA(b), m.sAp(b)
        B, Bp, Nv = m.sB(b), m.sBp(b), m.sN(b)
        A2 = A * A
        asp = Bp / A - B * Ap / A2
        Pv = B * Nv / A2 + Ap * w * w - 2.0 * A * w * asp
        P_w = 2.0 * Ap * w - 2.0 * A * asp
        out = (A, asp, Pv, P_w)
        cache[0] = (key, out)
        return out

    def f(b, w):
        A, asp, Pv, _ = vals(b, w)
        return 2.0 * A * w / Pv - asp

    def j(b, w):
        A, _, Pv, P_w = vals(b, w)
        return 2.0 * A / Pv - 2.0 * A * w * P_w / (Pv * Pv)

    return f, j


def fast_rhs_s(m: Model):
    cache = [None]

    def vals(w: float, b: float):
        key = (w, b)
        if cache[0] is not None and cache[0][0] == key:
            return cache[0][1]
        A, Ap, App = m.sA(b), m.sAp(b), m.sApp(b)
        B, Bp, Bpp = m.sB(b), m.sBp(b), m.sBpp(b)
        Nv, Np = m.sN(b), m.sNp(b)
        A2, A3 = A * A, A * A * A
        asp = Bp / A - B * Ap / A2
        aspp = ((Bpp * A - B * App) / A2
                - 2.0 * Ap * (Bp * A - B * Ap) / A3)
        up = B * Nv / A2
        upp = ((Bp * Nv + B * Np) / A2
               - 2.0 * B * Nv * Ap / A3)
        Pv = up + Ap * w * w - 2.0 * A * w * asp
        P_b = upp + App * w * w - 2.0 * w * (Ap * asp + A * aspp)
        D = 2.0 * A * w - asp * Pv
        D_b = 2.0 * Ap * w - aspp * Pv - asp * P_b
        out = (Pv, D, P_b, D_b)
        cache[0] = (key, out)
        return out

    def f(w, b):
        Pv, D, _, _ = vals(w, b)
        return Pv / D

    def j(w, b):
        Pv, D, P_b, D_b = vals(w, b)
        return (P_b * D - Pv * D_b) / (D * D)

    return f, j


def _s_depth_gauge(m: Model, b: float, w: float) -> float:
    A = m.sA(b)
    asp = m.s_a_star_p(b)
    Pv = m.s_u_p(b) + m.sAp(b) * w * w - 2.0 * A * w * asp
    t1 = abs(2.0 * A * w)
    t2 = abs(asp * Pv)
    num = t1 + t2
    den = abs(2.0 * A * w - asp * Pv)
    lo = 1e-16 * num + 1e-300
    return num / (den if den > lo else lo)


def _s_depth_gauge_floor(m: Model, b: float) -> float:
    A = m.sA(b)
    asp, aspp = m.s_a_star_p(b), m.s_a_star_pp(b)
    up, upp = m.s_u_p(b), m.s_u_pp(b)
    w1p = (aspp * up + asp * upp) / (2.0 * A) \
        - asp * up * m.sAp(b) / (2.0 * A * A)
    lo = 1e-16 * abs(asp) + 1e-300
    d = abs(w1p)
    return 2.0 * abs(asp) / (d if d > lo else lo)


# --------------------------------------------------------------------- #
# the continuation engine                                                #
# --------------------------------------------------------------------- #


@dataclass
class Branch:
    kind: str                    # 'unstable' | 'stable'
    Y: np.ndarray                # polyline, shape (n, 2): columns (a, b)
    term: str                    # 'capture' | 'box_exit' | 'abort_*'
    certs: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)


def _segment_capture(a0, b0, a1, b1, at, bt, radius):
    """Whether a resolved chord enters a target neighbourhood."""
    da, db = a1-a0, b1-b0
    denom = da*da+db*db
    if denom == 0.0:
        return (a1-at)**2+(b1-bt)**2 < radius*radius
    t = ((at-a0)*da+(bt-b0)*db)/denom
    t = min(1.0, max(0.0, t))
    return ((a0+t*da-at)**2+(b0+t*db-bt)**2
            < radius*radius)


def _append_resolved_point(points, point, resolution=0.0):
    """Append a geometric vertex only when binary64 resolves a new point."""
    q = tuple(map(float, point))
    if points:
        p = points[-1]
        scale = 1.0+max(abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1]))
        floor = max(64.0*np.finfo(float).eps*scale, float(resolution))
        if np.hypot(q[0]-p[0], q[1]-p[1]) <= floor:
            points[-1] = q
            return False
    points.append(q)
    return True


def _full_and_two_half(step, z, h, order):
    """Independent same-order compositions used by the C IRK fallbacks."""
    full = np.asarray(step(float(z[0]), float(z[1]), h, order))
    midpoint = np.asarray(step(float(z[0]), float(z[1]), 0.5*h, order))
    endpoint = midpoint
    if np.all(np.isfinite(midpoint)):
        endpoint = np.asarray(step(
            float(midpoint[0]), float(midpoint[1]), 0.5*h, order))
    if not (np.all(np.isfinite(full))
            and np.all(np.isfinite(midpoint))
            and np.all(np.isfinite(endpoint))):
        return None
    return full, midpoint, endpoint


def _geometric_orders():
    return ((8, 6) if GEOMETRIC_IRK_PRIMARY == 8 else (6, 8))


def _cubic_hermite(z0, z1, f0, f1, h, s):
    """Cubic dense output in the flow parameter, 0 <= s <= 1."""
    s2, s3 = s*s, s*s*s
    return ((2*s3-3*s2+1)*z0 + (s3-2*s2+s)*h*f0
            + (-2*s3+3*s2)*z1 + (s3-s2)*h*f1)


def _critical_array(critical_points):
    """The enumeration's critical points as an (n, 2) float array, or None."""
    if critical_points is None:
        return None
    array = np.asarray(critical_points, dtype=float).reshape(-1, 2)
    return array if len(array) else None


def _critical_arclength_cap(z, critical):
    """Proximity guard: a quarter of the distance to the nearest critical
    point, or +inf without a list.  This only prevents a step from JUMPING
    a corner the curvature bound has not yet seen; the resolution of the
    corner itself is the curvature term in :func:`_step_arclength_cap`."""
    if critical is None or len(critical) == 0:
        return np.inf
    d = np.hypot(critical[:, 0]-float(z[0]), critical[:, 1]-float(z[1]))
    return 0.25*float(np.min(d))


def _shared_field(m: Model):
    """Scalar (L, grad, hess) through the library's Horner kernels.

    The C segment entry point (spong/spong_potential.h) evaluates every
    quantity with plain Horner on the coefficient arrays.  The model's
    evaluators switch to range-guarded rational products for |b| > 32, so a
    loop driven by them can differ from the C by an ulp in the far field —
    and an ulp flips an accept/reject.  Bit parity must compare LOOP LOGIC,
    not evaluators: the oracle loops below evaluate through the kernel
    (Kernel.loss / gradient / hessian — the same arithmetic the segment
    uses) and fall back to the model only when there is no kernel at all,
    in which case there is no native segment to compare against either.

    grad returns a (g_a, g_b) float pair and hess an (H11, H12, H22) float
    triple: the loops consume them SCALAR by scalar, matching the C
    statement for statement (numpy's small matmul may round differently
    from written-out scalar expressions).
    """
    native = getattr(m, "_native_kernel", None)
    if native is not None and hasattr(native, "loss"):
        C = float(m.C)
        return (lambda a, b: native.loss(float(a), float(b), C),
                lambda a, b: native.gradient(float(a), float(b)),
                lambda a, b: native.hessian(float(a), float(b)))

    def _L(a, b):
        return float(m.L(float(a), float(b)))

    def _grad(a, b):
        g = m.gradL(float(a), float(b))
        return float(g[0]), float(g[1])

    def _hess(a, b):
        H = m.hessL(float(a), float(b))
        return float(H[0][0]), float(H[0][1]), float(H[1][1])

    return _L, _grad, _hess


def _step_arclength_cap(m, z, critical, ng: float | None = None,
                        grad=None, hess=None):
    """Arclength a potential-rate step may take from z.

    The unit-speed field is t = grad L/|grad L|, and along its flow line
    dt/ds = (I - t t^T) H t / |grad L|, so the line's curvature is

        kappa = |(I - t t^T) H t| / |grad L|

    -- one Hessian evaluation, exact pointwise.  A step of
    CRITICAL_STEP_FRACTION/kappa turns by about the nominal angle wherever
    the line actually turns, and costs nothing where it runs straight.
    Measured: a proximity-only cap (step ~ distance to the nearest critical
    point) held every branch of linear-target-d17-thrash to millimetre
    steps along straight valleys because its skeleton is dense, and cost
    75% on that case; curvature is what the step should follow.  The
    proximity guard remains, loosely, so a step cannot leap a corner that
    kappa at the step's start has not yet seen.
    """
    a, b = float(z[0]), float(z[1])
    if grad is None or hess is None:
        _L_, grad, hess = _shared_field(m)
    g0, g1 = grad(a, b)
    if ng is None:
        ng = float(np.hypot(g0, g1))
    cap = _critical_arclength_cap(z, critical)
    if not (np.isfinite(ng) and ng > 0.0):
        return cap
    t0, t1 = g0/ng, g1/ng
    h11, h12, h22 = hess(a, b)
    w0 = h11*t0 + h12*t1
    w1 = h12*t0 + h22*t1
    wt = w0*t0 + w1*t1
    p0, p1 = w0-wt*t0, w1-wt*t1
    kappa = float(np.hypot(p0, p1))/ng
    if np.isfinite(kappa) and kappa > 0.0:
        cap = min(cap, CRITICAL_STEP_FRACTION/kappa)
    return cap


def _arclength_step(native, m, z, arclength: float, flow_sign: float,
                    L=None):
    """One unit-speed step where a loss step could not be resolved.

    Constant-potential-rate stepping has a floor: a loss difference below
    ~4096*eps*(1+|L|) is unrepresentable, and near a critical point that
    floor is an arclength of floor/|grad L| -- at 1e-7 from a saddle, a
    hundred times the passage distance.  Arclength has no such floor, since
    the direction field grad L/|grad L| is resolvable down to where grad L
    itself drowns in evaluation noise.  Same order-8 stepper, same
    full/two-half Richardson check; the loss must move the right way
    within roundoff.  Returns (endpoint, midpoint) or None.
    """
    if not hasattr(native, "normalized_step"):
        return None
    if L is None:
        L = _shared_field(m)[0]
    L0 = L(float(z[0]), float(z[1]))
    # Roundoff floor for the two-composition agreement: the endpoints sit at
    # coordinates of size |z|, and steps of a few tens of ulps cannot agree
    # to a relative 1e-6 of their own length.
    noise = 64.0*np.finfo(float).eps*(1.0+abs(float(z[0]))+abs(float(z[1])))
    for order in _geometric_orders():
        trial = _full_and_two_half(
            native.normalized_step, z, flow_sign*arclength, order)
        if trial is None:
            continue
        full, midpoint, half = trial
        chord = float(np.hypot(*(half-z)))
        if chord == 0.0:
            continue
        richardson = float(np.hypot(*(full-half)))
        if richardson > max(1e-6*chord, noise):
            continue
        L1 = L(float(half[0]), float(half[1]))
        slack = 64.0*np.finfo(float).eps*(1.0+abs(L0))
        if not np.isfinite(L1) or flow_sign*(L1-L0) < -slack:
            continue
        return half, midpoint
    return None


def _potential_rate_prefix_python(m: Model, a0: float, b0: float, target,
                                  box, cap_r: float, engine_diag: dict,
                                  n_levels: int = 12000, critical=None):
    """Trace a resolved anisotropic connection by constant loss decrease.

    THE EXECUTABLE SPECIFICATION for spong_potential_rate_segment's prefix
    mode: the corpus test (scripts/potential_corpus.py) demands bit-identical
    vertices and counters between this loop and the C entry point.

    This owns only the regular prefix.  The vector field
    ``-grad(L)/|grad(L)|^2`` follows the same unparameterized integral curve
    and satisfies dL/dt=-1, so a long narrow valley is sampled by loss events
    instead of hundreds of thousands of microscopic arclength chords.
    Arrival remains with the ordinary chart/capture engine because this
    normalization is singular at the minimum.
    """
    native = getattr(m, "_native_kernel", None)
    if native is None or not hasattr(native, "potential_step"):
        return [(a0, b0)], b0, float(a0-m.s_a_star(b0)), "unavailable"
    L, grad, hess = _shared_field(m)
    at, bt = map(float, target)
    target_level = L(at, bt)
    start_level = L(a0, b0)
    gap0 = start_level-target_level
    if not (np.isfinite(gap0) and gap0 > 0.0):
        return [(a0, b0)], b0, float(a0-m.s_a_star(b0)), "unavailable"
    base = gap0/max(int(n_levels), 1)
    near_gap = base/16384.0
    cur_level_step = base
    z = np.array([a0, b0], dtype=float)
    pts = [tuple(z)]
    geometry_floor = 128.0*np.finfo(float).eps*(
        1.0+max(abs(float(x)) for x in box))
    accepted = rejected = 0
    gl8_attempted = gl8_accepted = 0
    max_richardson = 0.0
    critical_capped = 0
    arclength_steps = 0
    term = None
    # Capped and arclength steps are extra RESOLUTION, not progress toward
    # the target; they must not consume the level budget (measured: 888
    # capped steps exhausted this loop before the target neighbourhood, and
    # the centered arrival then finished the haul on coarse raw-gradient
    # steps whose angle energy was 800x the founding parity gate).
    iteration = 0
    while (iteration < n_levels+1024+critical_capped+arclength_steps
           and iteration < 4*(n_levels+1024)):
        iteration += 1
        level = L(float(z[0]), float(z[1]))
        gap = level-target_level
        if gap <= near_gap:
            term = "near_target"
            break
        h = -min(cur_level_step, 0.2*gap)
        # Arclength per unit loss is 1/|grad L|: cap the loss step so the
        # step does not outrun the flow's curvature near a critical point.
        # Where that cap falls below the loss floor, the loss cannot
        # resolve the step at all: take it in arclength instead.
        g0, g1 = grad(float(z[0]), float(z[1]))
        ng = float(np.hypot(g0, g1))
        arc_cap = _step_arclength_cap(m, z, critical, ng,
                                      grad=grad, hess=hess)
        cap = arc_cap*ng
        loss_floor = 4096*np.finfo(float).eps*(1.0+abs(level))
        if np.isfinite(cap) and cap > 0.0 and abs(h) > cap:
            critical_capped += 1
            if cap < loss_floor:
                arc = _arclength_step(native, m, z, arc_cap, -1.0, L=L)
                if arc is not None:
                    half, midpoint = arc
                    arclength_steps += 1
                    previous = z
                    z = half
                    accepted += 1
                    exited = False
                    for sample in (midpoint, z):
                        if not (box[0] <= sample[0] <= box[1]
                                and box[2] <= sample[1] <= box[3]):
                            _append_resolved_point(
                                pts, sample, geometry_floor)
                            z = sample
                            term = "box_exit"
                            exited = True
                            break
                        if _segment_capture(
                                float(previous[0]), float(previous[1]),
                                float(sample[0]), float(sample[1]),
                                at, bt, cap_r):
                            _append_resolved_point(
                                pts, sample, geometry_floor)
                            _append_resolved_point(
                                pts, (at, bt), geometry_floor)
                            z = np.array([at, bt])
                            term = "capture"
                            exited = True
                            break
                        _append_resolved_point(pts, sample, geometry_floor)
                        previous = sample
                    if exited:
                        break
                    continue
            h = -max(cap, loss_floor)
        zn = None
        accepted_midpoint = None
        for _retry in range(12):
            for order in _geometric_orders():
                if order == 8:
                    gl8_attempted += 1
                trial = _full_and_two_half(
                    native.potential_step, z, h, order)
                if trial is None:
                    continue
                full, midpoint, half = trial
                chord = float(np.hypot(*(half-z)))
                richardson = float(np.hypot(*(full-half)))
                new_level = L(float(half[0]), float(half[1]))
                loss_error = abs((new_level-level)-h)
                # Independent curve and parameterization checks.  The
                # accepted point is the two-half-step value.
                if (new_level >= level
                        or richardson > 1e-6*max(chord, 1e-8)
                        or loss_error > 2e-5*max(abs(h), 1e-12)):
                    continue
                zn = half
                accepted_midpoint = midpoint
                max_richardson = max(max_richardson, richardson)
                if order == 8:
                    gl8_accepted += 1
                break
            if zn is not None:
                break
            h *= 0.5
            rejected += 1
        if zn is None:
            term = "step_failure"
            break
        previous = z
        z = zn
        accepted += 1
        cur_level_step = min(base, 1.5*abs(h))
        for sample in (accepted_midpoint, z):
            if not (box[0] <= sample[0] <= box[1]
                    and box[2] <= sample[1] <= box[3]):
                _append_resolved_point(pts, sample, geometry_floor)
                z = sample
                term = "box_exit"
                break
            if _segment_capture(
                    float(previous[0]), float(previous[1]),
                    float(sample[0]), float(sample[1]), at, bt, cap_r):
                _append_resolved_point(pts, sample, geometry_floor)
                _append_resolved_point(pts, (at, bt), geometry_floor)
                z = np.array([at, bt])
                term = "capture"
                break
            _append_resolved_point(pts, sample, geometry_floor)
            previous = sample
        if term in {"box_exit", "capture"}:
            break
    else:
        term = "budget"
    engine_diag["potential_rate"] = {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "critical_capped_steps": critical_capped,
        "arclength_steps": arclength_steps,
        "gl8_attempted": gl8_attempted,
        "gl8_accepted": gl8_accepted,
        "level_step": float(base),
        "max_richardson": float(max_richardson),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return pts, float(z[1]), float(z[0]-m.s_a_star(z[1])), term


def _potential_rate_level_event_python(m: Model, a0: float, b0: float,
                                       targets, box, cap_r: float,
                                       engine_diag: dict,
                                       n_levels: int = 2048, critical=None):
    """Continue to the next minimum-level event without choosing a basin.

    THE EXECUTABLE SPECIFICATION for spong_potential_rate_segment's
    level-event mode; see _potential_rate_prefix_python.
    """
    native = getattr(m, "_native_kernel", None)
    if native is None or not hasattr(native, "potential_step"):
        return [(a0, b0)], "unavailable", None
    L, grad, hess = _shared_field(m)
    z = np.array((a0, b0), dtype=float)
    level0 = L(float(z[0]), float(z[1]))
    geometry_floor = 128.0*np.finfo(float).eps*(
        1.0+max(abs(float(x)) for x in box))
    target_levels = [
        (L(float(a), float(b)), (float(a), float(b)))
        for a, b in targets]
    slack0 = 1024*np.finfo(float).eps*(1.0+abs(level0))
    lower = [(v, q) for v, q in target_levels if v < level0-slack0]
    if not lower:
        return [tuple(z)], "unavailable", None
    event_level = max(v for v, _q in lower)
    gap0 = level0-event_level
    base = gap0/max(int(n_levels), 1)
    crossing_floor = max(
        base/1024.0,
        4096*np.finfo(float).eps*(1.0+abs(event_level)))
    cur_level_step = base
    pts = [tuple(z)]
    accepted = rejected = 0
    gl8_attempted = gl8_accepted = 0
    max_richardson = 0.0
    critical_capped = 0
    arclength_steps = 0
    captured = None
    term = "budget"
    iteration = 0
    while (iteration < 4*n_levels+critical_capped+arclength_steps
           and iteration < 16*n_levels):
        iteration += 1
        level = L(float(z[0]), float(z[1]))
        gap = level-event_level
        if gap <= -crossing_floor:
            term = "level_event"
            break
        # Away from the event use uniform potential samples.  Once its
        # enclosure is reached, deliberately step across it: merely
        # approaching from above cannot eliminate the higher minimum.
        requested = (max(2.0*gap, crossing_floor)
                     if gap <= crossing_floor else
                     min(cur_level_step, 0.5*gap))
        g0, g1 = grad(float(z[0]), float(z[1]))
        ng = float(np.hypot(g0, g1))
        arc_cap = _step_arclength_cap(m, z, critical, ng,
                                      grad=grad, hess=hess)
        cap = arc_cap*ng
        if (np.isfinite(cap) and cap > 0.0 and requested > cap
                and gap > crossing_floor):
            critical_capped += 1
            if cap < crossing_floor:
                arc = _arclength_step(native, m, z, arc_cap, -1.0, L=L)
                if arc is not None:
                    half, midpoint = arc
                    arclength_steps += 1
                    previous = z
                    z = half
                    accepted += 1
                    _append_resolved_point(pts, midpoint, geometry_floor)
                    _append_resolved_point(pts, z, geometry_floor)
                    if not (box[0] <= z[0] <= box[1]
                            and box[2] <= z[1] <= box[3]):
                        term = "box_exit"
                        break
                    for at, bt in targets:
                        if _segment_capture(
                                float(previous[0]), float(previous[1]),
                                float(z[0]), float(z[1]), at, bt, cap_r):
                            pts.append((at, bt))
                            captured = (at, bt)
                            term = "capture"
                            break
                    if captured is not None:
                        break
                    continue
            requested = max(cap, crossing_floor)
        h = -requested
        zn = None
        accepted_midpoint = None
        for _retry in range(14):
            for order in _geometric_orders():
                if order == 8:
                    gl8_attempted += 1
                trial = _full_and_two_half(
                    native.potential_step, z, h, order)
                if trial is None:
                    continue
                full, midpoint, half = trial
                chord = float(np.hypot(*(half-z)))
                richardson = float(np.hypot(*(full-half)))
                new_level = L(float(half[0]), float(half[1]))
                loss_error = abs((new_level-level)-h)
                if (new_level >= level
                        or richardson > 1e-6*max(chord, 1e-8)
                        or loss_error > 2e-5*max(abs(h), 1e-12)):
                    continue
                zn = half
                accepted_midpoint = midpoint
                max_richardson = max(max_richardson, richardson)
                if order == 8:
                    gl8_accepted += 1
                break
            if zn is not None:
                break
            h *= 0.5
            rejected += 1
        if zn is None:
            term = "step_failure"
            break
        previous = z
        z = zn
        accepted += 1
        cur_level_step = min(base, 1.5*abs(h))
        _append_resolved_point(pts, accepted_midpoint, geometry_floor)
        _append_resolved_point(pts, z, geometry_floor)
        if not (box[0] <= z[0] <= box[1]
                and box[2] <= z[1] <= box[3]):
            term = "box_exit"
            break
        for at, bt in targets:
            if _segment_capture(
                    float(previous[0]), float(previous[1]),
                    float(z[0]), float(z[1]), at, bt, cap_r):
                pts.append((at, bt))
                captured = (at, bt)
                term = "capture"
                break
        if captured is not None:
            break
    engine_diag.setdefault("candidate_level_events", []).append({
        "event_level": float(event_level),
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "critical_capped_steps": critical_capped,
        "arclength_steps": arclength_steps,
        "gl8_attempted": gl8_attempted,
        "gl8_accepted": gl8_accepted,
        "max_richardson": float(max_richardson),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    })
    return pts, term, captured


def _centered_raw_arrival_python(start, target, arrival_local,
                                 cap_r: float, engine_diag: dict,
                                 max_steps: int = 4096):
    """Finish a known connection with the regular target-centered flow.

    Constant-potential-rate and arclength parameterizations are singular at
    a minimum.  The unnormalized gradient field is regular there.  Its full
    translated polynomial is already stored in the zero-dimensional Morse
    data, so this phase neither re-expands the model nor evaluates a
    cancellation-prone global gradient.

    THE EXECUTABLE SPECIFICATION for spong_centered_arrival (C); see
    _potential_rate_prefix_python for the doctrine.  The jet's potential
    and raw steps are evaluated through the native LocalKernel -- the same
    arithmetic the C arrival uses -- and the turn cosine is written out
    scalar by scalar rather than as a NumPy dot product, which may fuse.
    """
    if (arrival_local is None or arrival_local.native is None
            or not hasattr(arrival_local.native, "raw_step")):
        return [tuple(map(float, start))], "unavailable"
    at, bt = map(float, target)
    finish_r = max(
        cap_r/64.0,
        4096*np.finfo(float).eps*(1.0+np.hypot(at, bt)))
    geometry_floor = 128.0*np.finfo(float).eps*(
        1.0+max(abs(at), abs(bt), abs(float(start[0])),
                abs(float(start[1]))))
    center = np.array((float(arrival_local.a), float(arrival_local.b)))
    z = np.asarray(start, dtype=float)-center
    lam = np.asarray(arrival_local.spectral.eigenvalues, dtype=float)
    if not (np.all(np.isfinite(lam)) and np.min(lam) > 0.0):
        return [tuple(map(float, start))], "unavailable"

    slow, fast = float(np.min(lam)), float(np.max(lam))
    dt = 0.25/fast
    dt_cap = 4.0/slow
    # Turn control.  The raw flow's direction rotates at rate ~||H|| and dt
    # = 0.25/fast turns it by ~14 degrees per step near a saddle -- this
    # phase can pass one on the way to its minimum (observed: a near-wall
    # branch whose target minimum lies just past the saddle it nearly
    # connects to).  Reject a step that turns more than twice the nominal
    # CRITICAL_STEP_FRACTION angle from the last accepted direction and
    # halve dt; the ramp reopens along the slow approach, where the
    # direction does not turn and dt_cap is what makes stiff arrivals
    # affordable.
    turn_reject = float(np.cos(2.0*np.arctan(CRITICAL_STEP_FRACTION)))
    last_direction = None
    turn_rejected = 0
    pts = [tuple(map(float, start))]
    accepted = rejected = 0
    gl8_attempted = gl8_accepted = 0
    max_richardson = 0.0
    term = "budget"
    for _ in range(max_steps):
        physical = z+center
        if np.hypot(physical[0]-at, physical[1]-bt) < finish_r:
            pts.append((at, bt))
            term = "capture"
            break
        value = float(arrival_local.potential(float(z[0]), float(z[1])))
        if not (np.isfinite(value) and value > 0.0):
            term = "invalid_potential"
            break
        zn = None
        accepted_midpoint = None
        trial_dt = dt
        for _retry in range(16):
            h = -trial_dt
            for order in _geometric_orders():
                if order == 8:
                    gl8_attempted += 1
                trial = _full_and_two_half(
                    arrival_local.raw_step, z, h, order)
                if trial is None:
                    continue
                full, midpoint, half = trial
                chord = float(np.hypot(*(half-z)))
                richardson = float(np.hypot(*(full-half)))
                next_value = float(arrival_local.potential(
                    float(half[0]), float(half[1])))
                # The two-half-step curve must descend the exact centered
                # potential, and the independent compositions must agree.
                tolerance = 2e-7*max(chord, 0.05*finish_r, 1e-13)
                if (not np.isfinite(next_value) or next_value >= value
                        or richardson > tolerance):
                    continue
                direction = half-z
                if last_direction is not None and chord > 0.0:
                    cosine = float(direction[0]*last_direction[0]
                                   + direction[1]*last_direction[1])/(
                        chord*float(np.hypot(*last_direction)))
                    if cosine < turn_reject:
                        turn_rejected += 1
                        continue
                zn = half
                accepted_midpoint = midpoint
                max_richardson = max(max_richardson, richardson)
                if order == 8:
                    gl8_accepted += 1
                break
            if zn is not None:
                break
            trial_dt *= 0.5
            rejected += 1
        if zn is None:
            term = "step_failure"
            break
        previous = z
        z = zn
        last_direction = z-previous
        accepted += 1
        dt = min(dt_cap, 1.5*trial_dt)
        p0 = previous+center
        for sample in (accepted_midpoint, z):
            p1 = sample+center
            _append_resolved_point(pts, p1, geometry_floor)
            if _segment_capture(
                    float(p0[0]), float(p0[1]),
                    float(p1[0]), float(p1[1]), at, bt, finish_r):
                pts.append((at, bt))
                term = "capture"
                break
            p0 = p1
        if term == "capture":
            break
    engine_diag["centered_arrival"] = {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "turn_rejected_steps": turn_rejected,
        "gl8_attempted": gl8_attempted,
        "gl8_accepted": gl8_accepted,
        "max_richardson": float(max_richardson),
        "spectral_ratio": fast/slow,
        "finish_radius": float(finish_r),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return pts, term


_ARRIVAL_TERM = {0: "capture", 1: "invalid_potential", 2: "step_failure",
                 3: "budget", 4: "unavailable"}


def _arrival_native(arrival_local):
    """The native module when the C arrival should run; see _potential_native."""
    from . import engine
    if engine.active_name() != "native":
        return None
    if (arrival_local is None or arrival_local.native is None
            or not hasattr(arrival_local.native, "potential")):
        return None
    try:
        from . import _native as native
    except ImportError:
        return None
    if not hasattr(native, "centered_arrival"):
        return None
    return native


def _centered_raw_arrival(start, target, arrival_local, cap_r: float,
                          engine_diag: dict, max_steps: int = 4096):
    """Dispatch the centered raw arrival; see _centered_raw_arrival_python."""
    native = _arrival_native(arrival_local)
    if native is None:
        return _centered_raw_arrival_python(
            start, target, arrival_local, cap_r, engine_diag,
            max_steps=max_steps)
    at, bt = map(float, target)
    lam = np.asarray(arrival_local.spectral.eigenvalues, dtype=float)
    if not (np.all(np.isfinite(lam)) and np.min(lam) > 0.0):
        return [tuple(map(float, start))], "unavailable"
    slow, fast = float(np.min(lam)), float(np.max(lam))
    turn_reject = float(np.cos(2.0*np.arctan(CRITICAL_STEP_FRACTION)))
    (term_code, _a_end, _b_end, accepted, rejected, turn_rejected,
     gl8_attempted, gl8_accepted, max_richardson, finish_r,
     spectral_ratio, blob) = native.centered_arrival(
        arrival_local.native, float(start[0]), float(start[1]), at, bt,
        float(arrival_local.a), float(arrival_local.b), slow, fast,
        float(cap_r), int(max_steps), turn_reject, GEOMETRIC_IRK_PRIMARY)
    term = _ARRIVAL_TERM[term_code]
    pts = [tuple(p) for p in
           np.frombuffer(blob, dtype=float).reshape(-1, 2).tolist()]
    engine_diag["centered_arrival"] = {
        "accepted_steps": int(accepted),
        "rejected_steps": int(rejected),
        "turn_rejected_steps": int(turn_rejected),
        "gl8_attempted": int(gl8_attempted),
        "gl8_accepted": int(gl8_accepted),
        "max_richardson": float(max_richardson),
        "spectral_ratio": float(spectral_ratio),
        "finish_radius": float(finish_r),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return pts, term


def _potential_rate_box_exit_python(m: Model, start, box, ds: float,
                                    engine_diag: dict,
                                    max_steps: int = 100000,
                                    critical=None):
    """Trace a stable branch outward with constant-potential-rate ascent.

    THE EXECUTABLE SPECIFICATION for spong_potential_rate_segment's ascent
    mode; see _potential_rate_prefix_python.
    """
    native = getattr(m, "_native_kernel", None)
    if native is None or not hasattr(native, "potential_step"):
        return [tuple(map(float, start))], "unavailable"
    L, grad, hess = _shared_field(m)
    z = np.asarray(start, dtype=float)
    pts = [tuple(z)]
    geometry_floor = 128.0*np.finfo(float).eps*(
        1.0+max(abs(float(x)) for x in box))
    accepted = rejected = 0
    gl8_attempted = gl8_accepted = 0
    max_richardson = 0.0
    max_interpolation_error = 0.0
    # The full/two-half composition supplies a stronger local curve check
    # than the legacy scalar-chart sampler.  Four legacy chords retain ample
    # topology resolution while avoiding million-segment stable tails.
    geometric_ds = 4.0*ds
    critical_capped = 0
    arclength_steps = 0
    # Step memory.  The nominal arclength is 16 legacy chords, but a step
    # rejected by the curvature cap or the Richardson/Hermite checks says
    # the flow is not that tame here; restarting from the full nominal on
    # the next step and halving down again cost three implicit solves per
    # rejection (measured 3:1 rejected:accepted on tricky-d11 and the
    # near-wall case).  Ramp from the last accepted chord instead, 1.5x per
    # step, as the prefix phase already does.
    last_arc = None
    term = "budget"
    iteration = 0
    while (iteration < max_steps+critical_capped+arclength_steps
           and iteration < 4*max_steps):
        iteration += 1
        level = L(float(z[0]), float(z[1]))
        g0, g1 = grad(float(z[0]), float(z[1]))
        ng = float(np.hypot(g0, g1))
        if not (np.isfinite(level) and np.isfinite(ng) and ng > 0.0):
            term = "unresolved_field"
            break
        nominal_arc = 16.0*geometric_ds
        if last_arc is not None:
            nominal_arc = min(nominal_arc, 1.5*last_arc)
        h = max(
            nominal_arc*ng,
            4096*np.finfo(float).eps*(1.0+abs(level)))
        # Near a critical point the loss step is an arclength h/|grad L|;
        # bound it by the flow's local curvature radius (see
        # CRITICAL_STEP_FRACTION) so the step cannot cross the saddle.
        # Where the cap falls below the loss floor, step in arclength.
        arc_cap = _step_arclength_cap(m, z, critical, ng,
                                      grad=grad, hess=hess)
        cap = arc_cap*ng
        loss_floor = 4096*np.finfo(float).eps*(1.0+abs(level))
        if np.isfinite(cap) and cap > 0.0 and h > cap:
            critical_capped += 1
            if cap < loss_floor:
                arc = _arclength_step(native, m, z, arc_cap, +1.0, L=L)
                if arc is not None:
                    half, midpoint = arc
                    arclength_steps += 1
                    last_arc = float(np.hypot(*(half-z)))
                    z = half
                    accepted += 1
                    exited = False
                    for sample in (midpoint, z):
                        _append_resolved_point(pts, sample, geometry_floor)
                        if not (box[0] <= sample[0] <= box[1]
                                and box[2] <= sample[1] <= box[3]):
                            z = sample
                            term = "box_exit"
                            exited = True
                            break
                    if exited:
                        break
                    continue
            h = max(cap, loss_floor)
        zn = None
        dense_data = None
        for _retry in range(14):
            for order in _geometric_orders():
                if order == 8:
                    gl8_attempted += 1
                trial = _full_and_two_half(
                    native.potential_step, z, h, order)
                if trial is None:
                    continue
                full, midpoint, half = trial
                chord = float(np.hypot(*(half-z)))
                richardson = float(np.hypot(*(full-half)))
                new_level = L(float(half[0]), float(half[1]))
                loss_error = abs((new_level-level)-h)
                e0, e1 = grad(float(half[0]), float(half[1]))
                q1 = e0*e0+e1*e1
                if not (q1 > 0.0 and np.isfinite(q1)):
                    continue
                f0 = np.array([g0/(ng*ng), g1/(ng*ng)])
                f1 = np.array([e0/q1, e1/q1])
                hermite_mid = _cubic_hermite(
                    z, half, f0, f1, h, 0.5)
                interpolation_error = float(
                    np.hypot(*(hermite_mid-midpoint)))
                curve_tol = 2e-6*max(chord, geometric_ds, 1e-8)
                if (new_level <= level
                        or richardson > 1e-6*max(chord, 1e-8)
                        or interpolation_error > curve_tol
                        or loss_error > 2e-5*max(abs(h), 1e-12)):
                    continue
                zn = half
                dense_data = (f0, f1, h, chord, interpolation_error)
                max_richardson = max(max_richardson, richardson)
                if order == 8:
                    gl8_accepted += 1
                break
            if zn is not None:
                break
            h *= 0.5
            rejected += 1
        if zn is None:
            term = "step_failure"
            break
        z0 = z
        z = zn
        f0, f1, h_used, chord, interpolation_error = dense_data
        last_arc = chord
        max_interpolation_error = max(
            max_interpolation_error, interpolation_error)
        subdivisions = max(1, int(np.ceil(chord/geometric_ds)))
        exited = False
        for j in range(1, subdivisions+1):
            p = _cubic_hermite(
                z0, z, f0, f1, h_used, j/subdivisions)
            _append_resolved_point(pts, p, geometry_floor)
            if not (box[0] <= p[0] <= box[1]
                    and box[2] <= p[1] <= box[3]):
                z = p
                term = "box_exit"
                exited = True
                break
        accepted += 1
        if exited:
            break
    engine_diag["potential_rate_ascent"] = {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "critical_capped_steps": critical_capped,
        "arclength_steps": arclength_steps,
        "gl8_attempted": gl8_attempted,
        "gl8_accepted": gl8_accepted,
        "max_richardson": float(max_richardson),
        "max_interpolation_error": float(max_interpolation_error),
        "geometric_ds": float(geometric_ds),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return pts, term


_POTENTIAL_TERM = {
    0: "near_target", 1: "capture", 2: "box_exit", 3: "level_event",
    4: "step_failure", 5: "budget", 6: "unresolved_field", 7: "unavailable",
}
_POTENTIAL_MODE = {"prefix": 0, "level_event": 1, "ascent": 2}


def _potential_native(m: Model):
    """(native module, kernel) when the C segment entry point should run.

    Same contract as _continue_curve's dispatch: SPONG_ENGINE=native selects
    the C port, which holds no Python objects and releases the GIL for the
    whole segment; anything else, or a missing binding, runs the Python
    loops, which remain the executable specification and the parity
    oracle (scripts/potential_corpus.py).  Unlike continue_curve there is
    no DELEGATE path: the three loops carry no corpus-invisible rescues,
    so the port answers every request it accepts.
    """
    from . import engine
    if engine.active_name() != "native":
        return None, None
    kernel = getattr(m, "_native_kernel", None)
    if kernel is None:
        return None, None
    try:
        from . import _native as native
    except ImportError:
        return None, None
    if not hasattr(native, "potential_rate_segment"):
        return None, None
    return native, kernel


def _potential_segment(native, kernel, m: Model, mode: str, a0: float,
                       b0: float, targets, cap_r: float, box, ds: float,
                       n_levels: int, max_steps: int, critical):
    """One native constant-potential-rate segment, unpacked."""
    flat_targets = [float(c) for t in targets for c in t]
    flat_critical = ([] if critical is None
                     else [float(c) for c in np.asarray(critical).reshape(-1)])
    (term_code, a_end, b_end, captured, captured_a, captured_b,
     event_level, level_step, accepted, rejected, critical_capped,
     arclength_steps, gl8_attempted, gl8_accepted,
     max_richardson, max_interpolation_error, blob) = \
        native.potential_rate_segment(
            kernel, float(m.C), _POTENTIAL_MODE[mode], float(a0), float(b0),
            flat_targets, float(cap_r), [float(x) for x in box], float(ds),
            int(n_levels), int(max_steps), flat_critical,
            CRITICAL_STEP_FRACTION, GEOMETRIC_IRK_PRIMARY)
    pts = [tuple(p) for p in
           np.frombuffer(blob, dtype=float).reshape(-1, 2).tolist()]
    return (_POTENTIAL_TERM[term_code], pts,
            ((captured_a, captured_b) if captured else None),
            {"a_end": float(a_end), "b_end": float(b_end),
             "event_level": float(event_level),
             "level_step": float(level_step),
             "accepted": int(accepted), "rejected": int(rejected),
             "critical_capped": int(critical_capped),
             "arclength_steps": int(arclength_steps),
             "gl8_attempted": int(gl8_attempted),
             "gl8_accepted": int(gl8_accepted),
             "max_richardson": float(max_richardson),
             "max_interpolation_error": float(max_interpolation_error)})


def _potential_rate_prefix(m: Model, a0: float, b0: float, target,
                           box, cap_r: float, engine_diag: dict,
                           n_levels: int = 12000, critical=None):
    """Dispatch one prefix segment; see _potential_rate_prefix_python."""
    native, kernel = _potential_native(m)
    if native is None:
        return _potential_rate_prefix_python(
            m, a0, b0, target, box, cap_r, engine_diag,
            n_levels=n_levels, critical=critical)
    term, pts, _captured, r = _potential_segment(
        native, kernel, m, "prefix", a0, b0,
        [tuple(map(float, target))], cap_r, box, 0.0, n_levels, 0, critical)
    if term == "unavailable":
        return [(a0, b0)], b0, float(a0-m.s_a_star(b0)), "unavailable"
    engine_diag["potential_rate"] = {
        "accepted_steps": r["accepted"],
        "rejected_steps": r["rejected"],
        "critical_capped_steps": r["critical_capped"],
        "arclength_steps": r["arclength_steps"],
        "gl8_attempted": r["gl8_attempted"],
        "gl8_accepted": r["gl8_accepted"],
        "level_step": r["level_step"],
        "max_richardson": r["max_richardson"],
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return (pts, r["b_end"],
            float(r["a_end"]-m.s_a_star(r["b_end"])), term)


def _potential_rate_level_event(m: Model, a0: float, b0: float, targets,
                                box, cap_r: float, engine_diag: dict,
                                n_levels: int = 2048, critical=None):
    """Dispatch one level-event segment; see the _python specification."""
    native, kernel = _potential_native(m)
    if native is None:
        return _potential_rate_level_event_python(
            m, a0, b0, targets, box, cap_r, engine_diag,
            n_levels=n_levels, critical=critical)
    term, pts, captured, r = _potential_segment(
        native, kernel, m, "level_event", a0, b0,
        [tuple(map(float, t)) for t in targets], cap_r, box, 0.0,
        n_levels, 0, critical)
    if term == "unavailable":
        return [(float(a0), float(b0))], "unavailable", None
    engine_diag.setdefault("candidate_level_events", []).append({
        "event_level": r["event_level"],
        "accepted_steps": r["accepted"],
        "rejected_steps": r["rejected"],
        "critical_capped_steps": r["critical_capped"],
        "arclength_steps": r["arclength_steps"],
        "gl8_attempted": r["gl8_attempted"],
        "gl8_accepted": r["gl8_accepted"],
        "max_richardson": r["max_richardson"],
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    })
    return pts, term, captured


def _potential_rate_box_exit(m: Model, start, box, ds: float,
                             engine_diag: dict, max_steps: int = 100000,
                             critical=None):
    """Dispatch one ascent segment; see the _python specification."""
    native, kernel = _potential_native(m)
    if native is None:
        return _potential_rate_box_exit_python(
            m, start, box, ds, engine_diag,
            max_steps=max_steps, critical=critical)
    a0, b0 = map(float, start)
    term, pts, _captured, r = _potential_segment(
        native, kernel, m, "ascent", a0, b0, [], 0.0, box, ds, 0,
        max_steps, critical)
    engine_diag["potential_rate_ascent"] = {
        "accepted_steps": r["accepted"],
        "rejected_steps": r["rejected"],
        "critical_capped_steps": r["critical_capped"],
        "arclength_steps": r["arclength_steps"],
        "gl8_attempted": r["gl8_attempted"],
        "gl8_accepted": r["gl8_accepted"],
        "max_richardson": r["max_richardson"],
        "max_interpolation_error": r["max_interpolation_error"],
        "geometric_ds": float(4.0*ds),
        "term": term,
        "primary_order": GEOMETRIC_IRK_PRIMARY,
    }
    return pts, term


def _continue_curve(m: Model, b0: float, w0: float, flow: int,
                    targets, box, ds: float, max_steps: int | None = None,
                    cap_r: float = 2e-3, ds0: float | None = None,
                    shallow_gate=None, engine_diag: dict | None = None,
                    centered_local=None):
    """Dispatch one engine segment to the selected engine.

    SPONG_ENGINE=native runs the C port, which holds no Python objects and
    releases the GIL for the whole segment -- the property that makes branch
    tracing concurrent.  It reproduces _continue_curve_python bit for bit on
    every segment of tests/corpus/continue_curve.json.

    Paths the corpus cannot judge -- the floor-fallback ladder, the
    normalized-arclength rescue, the centered rescue, the stall trim -- are not
    reimplemented in C.  The port returns DELEGATE and the WHOLE segment is
    re-run here, from the original arguments.  Resuming mid-segment would have
    to carry the chart, the ramped chord, the floor fixed at launch and the
    stall window across the boundary; re-running is identical by construction
    and costs nothing in a case that has never yet occurred.
    """
    from . import engine
    if max_steps is None:
        # TWO bounds, and they answer different questions.
        #
        # The arclength budget inside the engines is the SEMANTIC one: how
        # much curve the branch may draw.  It is invariant under the halving
        # the engine does, which a step count is not.
        #
        # This step count is the WORK bound, and it must stay where it was.
        # Raising it to 128*diagonal/ds alongside the arclength budget was a
        # bad regression: a stiff branch whose chord has collapsed to ds/128
        # needs 1024*diagonal/ds steps to travel 8 diagonals, so the two
        # loosenings compounded and a 14-minute qualification run passed four
        # and a half hours of CPU.  A branch still marching after this many
        # accepted steps is not going to arrive.
        #
        # Net effect of the pair: the engine can now stop EARLIER than before
        # (on distance) but never later (on work).
        diagonal = float(np.hypot(box[1]-box[0], box[3]-box[2]))
        max_steps = int(max(200000.0, 8.0*diagonal/max(ds, 1e-300)))
    if engine.active_name() != "native":
        return _continue_curve_python(
            m, b0, w0, flow, targets, box, ds, max_steps=max_steps,
            cap_r=cap_r, ds0=ds0, shallow_gate=shallow_gate,
            engine_diag=engine_diag, centered_local=centered_local)

    kernel = getattr(m, "_native_kernel", None)
    native = None
    if kernel is not None:
        try:
            from . import _native as native
        except ImportError:
            native = None
    if native is None or not hasattr(native, "continue_curve"):
        return _continue_curve_python(
            m, b0, w0, flow, targets, box, ds, max_steps=max_steps,
            cap_r=cap_r, ds0=ds0, shallow_gate=shallow_gate,
            engine_diag=engine_diag, centered_local=centered_local)

    flat = [float(c) for t in targets for c in t]
    term_code, reason, switches, b_end, w_end, taken, rejected, blob = \
        native.continue_curve(
            kernel, float(m.C), float(b0), float(w0), int(flow),
            flat, float(cap_r), [float(x) for x in box],
            float(ds), -1.0 if ds0 is None else float(ds0),
            None if shallow_gate is None else [float(x) for x in shallow_gate],
            int(max_steps))

    if term_code == _NATIVE_DELEGATE:
        if engine_diag is not None:
            key = f"native_delegate_{_DELEGATE_REASON.get(reason, reason)}"
            engine_diag[key] = engine_diag.get(key, 0) + 1
        return _continue_curve_python(
            m, b0, w0, flow, targets, box, ds, max_steps=max_steps,
            cap_r=cap_r, ds0=ds0, shallow_gate=shallow_gate,
            engine_diag=engine_diag, centered_local=centered_local)

    pts = np.frombuffer(blob, dtype=float).reshape(-1, 2).tolist()
    if engine_diag is not None:
        engine_diag["native_steps"] = (
            engine_diag.get("native_steps", 0) + int(taken))
        engine_diag["native_rejected"] = (
            engine_diag.get("native_rejected", 0) + int(rejected))
    return pts, _NATIVE_TERM[term_code], int(switches), (b_end, w_end)


_NATIVE_TERM = {
    0: "capture", 1: "box_exit", 2: "enter_shallow",
    3: "abort_stationary", 4: "abort_switch_limit", 5: "abort_nonfinite",
    6: "abort_step_failure", 7: "abort_max_steps",
}
_NATIVE_DELEGATE = 100
_DELEGATE_REASON = {
    1: "floor_ladder", 2: "stall_trim", 3: "centered_chart",
}


def _continue_curve_python(m: Model, b0: float, w0: float, flow: int,
                           targets, box, ds: float, max_steps: int = 200000,
                           cap_r: float = 2e-3, ds0: float | None = None,
                           shallow_gate=None, engine_diag: dict | None = None,
                           centered_local=None):
    """Walk the trajectory through chart pieces.

    flow: +1 descent (unstable branches), -1 ascent (separatrices).
    targets: list of candidate (a, b) minimum capture points or [].
    Returns (points, term_reason, n_switches, (b, w) final state).

    Dispatcher contract, third handoff: on DESCENT, if the sounding rises
    above KAPPA_HI while the trajectory is slaved to the valley floor
    (|w - w1| small), the engine exits with 'enter_shallow' — the slow ODE
    is the forbidden cancellation form there and the Hadamard fixed point
    owns the regime.  Ascent never triggers this (a separatrix crossing
    the backbone is not slaved).
    """
    sf, sj = slow_rhs_s(m)
    ff, fj = fast_rhs_s(m)
    native = getattr(m, "_native_kernel", None)
    b, w = float(b0), float(w0)
    pts = [(m.s_a_star(b) + w, b)]
    # Budget the DISTANCE TRAVELLED, not the step count.  A step budget is not
    # invariant under the halving below: the turn budget and the
    # descent-realization test drive `cur` down to continuation_floor =
    # cur0/128, so a branch crossing stiff country takes up to 128x more steps
    # than diagonal/ds predicts, and a count derived from ds is optimistic by
    # exactly that factor.  Measured across 20 directed models,
    # abort_max_steps accounted for 9 of 15 unfinished branches -- 5 unstable
    # and 4 stable -- while the launch-class refusal it was confused with
    # accounted for 2.  Arclength is what the budget was always trying to say,
    # and halving does not change it.  max_steps survives as a runaway guard.
    arc_budget = 8.0*float(np.hypot(box[1]-box[0], box[3]-box[2]))
    travelled = 0.0

    vb, vw = _s_velocities(m, b, w)
    vb, vw = flow * vb, flow * vw
    chart = "slow" if abs(vw) <= R_SWITCH * abs(vb) else "fast"
    switches = 0
    recent_b: list[float] = []          # stall detector window
    # geometric launch ramp: begin at the launch scale, grow into ds, so
    # polyline spacing is smooth and the angle-energy carries no launch kink
    cur = ds if ds0 is None else min(ds, max(ds0, 1e-12))
    # A fixed resolution floor belongs to this handoff, not to the possibly
    # enormous eventual portrait spacing and not to the moving current step.
    # The former can forbid all seam adjustment; the latter can shrink
    # forever.  Seven halvings of the certified incoming chord is the finite
    # local retry budget.
    continuation_floor = cur/128.0

    def centered_trial(a_start, b_start, chord):
        if centered_local is None or centered_local.native is None:
            return None
        z = np.array([
            a_start-centered_local.a,
            b_start-centered_local.b])
        glocal = np.asarray(
            centered_local.gradient(float(z[0]), float(z[1])))
        preferred = int(abs(glocal[1]) > abs(glocal[0]))
        for independent in (preferred, 1-preferred):
            dependent = 1-independent
            if abs(glocal[independent]) < 1e-300:
                continue
            h = chord / np.hypot(
                1.0, glocal[dependent]/glocal[independent])
            h *= 1.0 if -flow*glocal[independent] > 0.0 else -1.0
            for order in (6, 4):
                try:
                    y_try = centered_local.native.curve_step(
                        float(z[independent]), float(z[dependent]), float(h),
                        independent, order)
                except (ArithmeticError, ValueError,
                        FloatingPointError, OverflowError):
                    continue
                z_try = z.copy()
                z_try[independent] += h
                z_try[dependent] = y_try
                if not np.all(np.isfinite(z_try)):
                    continue
                delta = z_try-z
                actual_chord = float(np.hypot(*delta))
                if actual_chord > 2.0*chord or actual_chord == 0.0:
                    continue
                expected = flow*float(glocal @ delta)
                p0 = centered_local.potential(float(z[0]), float(z[1]))
                p1 = centered_local.potential(
                    float(z_try[0]), float(z_try[1]))
                actual = flow*float(p1-p0)
                slack = 64.0*np.finfo(float).eps*(1.0+abs(p0))
                if expected >= 0.0 or actual > 1e-4*expected+slack:
                    continue
                a_try = centered_local.a+z_try[0]
                b_try = centered_local.b+z_try[1]
                if not (np.isfinite(a_try) and np.isfinite(b_try)):
                    continue
                key = (
                    f"centered_{'a' if independent == 0 else 'b'}_"
                    f"gl{order}")
                return float(a_try), float(b_try), key
        return None

    for _ in range(max_steps):
        vb, vw = _s_velocities(m, b, w)
        vb, vw = flow * vb, flow * vw
        speed = max(abs(vb), abs(vw))
        if speed < 1e-300:
            return pts, "abort_stationary", switches, (b, w)

        if chart == "slow" and abs(vw) > R_SWITCH * abs(vb):
            chart, switches = "fast", switches + 1
        elif chart == "fast" and abs(vb) > R_SWITCH * abs(vw):
            chart, switches = "slow", switches + 1
        if switches > MAX_SWITCHES:
            return pts, "abort_switch_limit", switches, (b, w)

        if flow > 0 and (
                shallow_gate is None
                or (b - shallow_gate[0]) * shallow_gate[1] >= 0):
            # chart-agnostic: the fast chart can bang-bang across the
            # floor (w overshooting ±ds) just as the slow chart can
            # two-cycle — both are hover stalls of a non-L-stable
            # discretization over shallow water
            # Shallow-water handoff (floor gauge = the cancellation ratio
            # ON the floor, closed form — honest at u-inflections).  Two
            # ways in, both mandatory because Gauss methods are A-stable
            # but NOT L-stable (R(-inf) = -1 for IMM, +1 for GL2): once
            # |h·lambda| is huge the discrete flow stops contracting onto
            # the floor.
            #   (a) healthy approach: trajectory slaved within 5%;
            #   (b) STALL: the discrete two-cycle/freeze at hover
            #       altitude — detected as no net b-progress over a
            #       window; the stalled tail is a discrete artifact and
            #       is TRIMMED, its hover altitude becoming the seam
            #       residual.
            if _s_depth_gauge_floor(m, b) >= KAPPA_HI:
                w1 = m.s_a_star_p(b) * m.s_u_p(b) / (2.0 * m.sA(b))
                if abs(w - w1) <= 0.05 * abs(w1) + 1e-9 * (1 + abs(w)):
                    return pts, "enter_shallow", switches, (b, w)
                recent_b.append(b)
                if len(recent_b) > 12:
                    recent_b.pop(0)
                    if abs(b - recent_b[0]) < 1.0 * ds:
                        # stalled: trim the hover tail
                        n_trim = min(12, len(pts) - 1)
                        del pts[-n_trim:]
                        a_bk, b_bk = pts[-1]
                        w_bk = a_bk - m.s_a_star(b_bk)
                        return pts, "enter_shallow", switches, (b_bk, w_bk)
            else:
                recent_b.clear()

        # curvature-adaptive chord: take the step; if the polyline turns
        # more than the turn budget at the new vertex, revert, halve,
        # retry — the zoom-proof guarantee enforced at trace time
        b_prev, w_prev = b, w
        retry_floor = continuation_floor
        # Initial attempt plus seven halvings to ``continuation_floor``.
        # Seven total attempts fell out of the loop immediately after the
        # seventh halve, bypassing the floor fallback with the last NaN stage
        # value still live.
        for _retry in range(8):
            h = np.nan
            try:
                if chart == "slow":
                    h = cur / (1.0 + (vw / vb) ** 2) ** 0.5 * (
                        1.0 if vb > 0 else -1.0)
                    w_new = native.slow_step(b_prev, w_prev, h) \
                        if native is not None else \
                        gauss.gl6_scalar(sf, sj, b_prev, w_prev, h)
                    b_new = b_prev + h
                else:
                    h = cur / (1.0 + (vb / vw) ** 2) ** 0.5 * (
                        1.0 if vw > 0 else -1.0)
                    b_new = native.fast_step(w_prev, b_prev, h) \
                        if native is not None else \
                        gauss.gl6_scalar(ff, fj, w_prev, b_prev, h)
                    w_new = w_prev + h
                step_failed = False
            except (ZeroDivisionError, FloatingPointError, OverflowError):
                step_failed = True
            # A nonlinear Gauss stage system can have several roots.  Newton
            # (especially its Armijo restart) must select the root connected
            # to the local flow, not merely a small-residual reversible root.
            # The gradient supplies the branch-independent discriminator:
            # descent/ascent must realize a fixed fraction of its first-order
            # expected change.  Rejection is a step-size signal.
            if (not step_failed
                    and np.isfinite(b_new) and np.isfinite(w_new)):
                a_prev = m.s_a_star(b_prev) + w_prev
                a_new_test = m.s_a_star(b_new) + w_new
                da, db = a_new_test-a_prev, b_new-b_prev
                expected = flow * float(m.gradL(a_prev, b_prev) @
                                        np.array([da, db]))
                actual = flow * float(
                    m.L(a_new_test, b_new) - m.L(a_prev, b_prev))
                descent_slack = 64.0*np.finfo(float).eps * (
                    1.0 + abs(float(m.L(a_prev, b_prev))))
                if (not all(np.isfinite(x) for x in (
                        a_prev, a_new_test, expected, actual,
                        descent_slack))
                        or expected >= 0.0
                        or actual > 1e-4*expected + descent_slack):
                    step_failed = True
            # A failed stage solve is a STEP-SIZE signal, not a fatal error.
            # The stage matrix is M = I − h·diag(J_i)·A, which is diagonally
            # dominant — hence trivially solvable — for small enough h, so
            # halving is the correct response and aborting throws away a
            # branch the engine can still trace.  Measured: GL6's 3-stage
            # solve is tighter than GL4's 2-stage one, so it signals here
            # where GL4 silently proceeds; without halving that showed up out
            # of sample as abort_nonfinite on branches GL4 captured.
            if step_failed or not (np.isfinite(b_new) and np.isfinite(w_new)):
                if cur > retry_floor:
                    cur *= 0.5
                    continue
                # At the spatial resolution floor, further halving moves into
                # cancellation.  The two graph charts describe the same curve,
                # and GL4 has a different stage system, so exhaust those
                # equivalent representations before declaring the state
                # numerically undefined.  GL6 remains the normal engine; this
                # ladder is reached only after its retry budget is exhausted.
                rescued = False
                alternatives = [
                    (chart, "gl4"),
                    ("fast" if chart == "slow" else "slow", "gl6"),
                    ("fast" if chart == "slow" else "slow", "gl4"),
                ]
                for alt_chart, method in alternatives:
                    try:
                        if alt_chart == "slow":
                            h = cur / (1.0 + (vw / vb) ** 2) ** 0.5 * (
                                1.0 if vb > 0 else -1.0)
                            if native is not None:
                                step = native.slow_step if method == "gl6" \
                                    else native.slow_step_gl4
                                w_new = step(b_prev, w_prev, h)
                            else:
                                step = gauss.gl6_scalar if method == "gl6" \
                                    else gauss.gl4_scalar
                                w_new = step(sf, sj, b_prev, w_prev, h)
                            b_new = b_prev + h
                        else:
                            h = cur / (1.0 + (vb / vw) ** 2) ** 0.5 * (
                                1.0 if vw > 0 else -1.0)
                            if native is not None:
                                step = native.fast_step if method == "gl6" \
                                    else native.fast_step_gl4
                                b_new = step(w_prev, b_prev, h)
                            else:
                                step = gauss.gl6_scalar if method == "gl6" \
                                    else gauss.gl4_scalar
                                b_new = step(ff, fj, w_prev, b_prev, h)
                            w_new = w_prev + h
                    except (ZeroDivisionError, FloatingPointError, OverflowError):
                        continue
                    if np.isfinite(b_new) and np.isfinite(w_new):
                        a_prev = m.s_a_star(b_prev) + w_prev
                        a_new_test = m.s_a_star(b_new) + w_new
                        da, db = a_new_test-a_prev, b_new-b_prev
                        expected = flow * float(
                            m.gradL(a_prev, b_prev) @ np.array([da, db]))
                        actual = flow * float(
                            m.L(a_new_test, b_new) - m.L(a_prev, b_prev))
                        descent_slack = 64.0*np.finfo(float).eps * (
                            1.0 + abs(float(m.L(a_prev, b_prev))))
                        if (expected >= 0.0
                                or actual > 1e-4*expected + descent_slack):
                            continue
                        if engine_diag is not None:
                            key = f"floor_fallback_{alt_chart}_{method}"
                            engine_diag[key] = engine_diag.get(key, 0) + 1
                        rescued = True
                        break
                # Both graph parameterizations can become singular at the
                # same geometric point (most commonly on arrival).  The curve
                # itself is still regular there.  Continue it by arclength in
                # the native 2-D normalized-gradient field before declaring
                # FP64 defeat; this is representation-independent and its
                # Armijo expected-change check selects the local stage root.
                if not rescued and native is not None:
                    for order in (*_geometric_orders(), 4):
                        a_prev = m.s_a_star(b_prev) + w_prev
                        try:
                            a_try, b_try = native.normalized_step(
                                a_prev, b_prev, -flow*cur, order)
                        except (ArithmeticError, ValueError,
                                FloatingPointError, OverflowError):
                            continue
                        if not (np.isfinite(a_try) and np.isfinite(b_try)):
                            continue
                        da, db = a_try-a_prev, b_try-b_prev
                        expected = flow * float(
                            m.gradL(a_prev, b_prev) @ np.array([da, db]))
                        actual = flow * float(
                            m.L(a_try, b_try) - m.L(a_prev, b_prev))
                        descent_slack = 64.0*np.finfo(float).eps * (
                            1.0 + abs(float(m.L(a_prev, b_prev))))
                        if (expected >= 0.0
                                or actual > 1e-4*expected + descent_slack):
                            continue
                        b_new, w_new = float(b_try), float(
                            a_try-m.s_a_star(b_try))
                        if engine_diag is not None:
                            key = f"floor_fallback_normalized_gl{order}"
                            engine_diag[key] = engine_diag.get(key, 0) + 1
                        rescued = True
                        break
                if not rescued:
                    a_prev = m.s_a_star(b_prev)+w_prev
                    centered = centered_trial(a_prev, b_prev, cur)
                    if centered is not None:
                        a_try, b_new, centered_key = centered
                        w_new = a_try-m.s_a_star(b_new)
                        if engine_diag is not None:
                            key = f"floor_fallback_{centered_key}"
                            engine_diag[key] = engine_diag.get(key, 0)+1
                        rescued = True
                if rescued:
                    step_failed = False
                else:
                    if engine_diag is not None:
                        engine_diag["step_failure"] = {
                            "b": float(b_prev), "w": float(w_prev),
                            "cur": float(cur), "ds": float(ds), "chart": chart,
                            "h": float(h), "vb": float(vb), "vw": float(vw),
                            "retry": _retry,
                        }
                    return pts, "abort_step_failure", switches, (b_prev, w_prev)
            if len(pts) >= 2:
                a_new = m.s_a_star(b_new) + w_new
                p1, p0 = pts[-1], pts[-2]
                d1a, d1b = p1[0] - p0[0], p1[1] - p0[1]
                d2a, d2b = a_new - p1[0], b_new - p1[1]
                n1 = (d1a * d1a + d1b * d1b) ** 0.5
                n2 = (d2a * d2a + d2b * d2b) ** 0.5
                if (n1 > 1e-14 and n2 > 1e-14
                        and (d1a * d2a + d1b * d2b) / (n1 * n2) < TURN_MAX
                        and cur > retry_floor):
                    cur *= 0.5
                    continue
            break
        b, w = b_new, w_new
        a = m.s_a_star(b) + w
        cur = min(ds, cur * 1.06)   # gentle ramp: the angle-energy
        # functional is a symmetric difference — 2nd-order on uniform
        # spacing, 1st-order under spacing jumps
        travelled += float(np.hypot(a - a_prev, b - b_prev))
        if travelled > arc_budget:
            return pts, "abort_max_steps", switches, (b, w)

        if not (np.isfinite(b) and np.isfinite(w)):
            return pts, "abort_nonfinite", switches, (b, w)

        for (at, bt) in targets:
            captured = _segment_capture(
                a_prev, b_prev, a, b, at, bt, cap_r)
            if captured:
                target_level = float(m.L(at, bt))
                current_level = float(m.L(a_prev, b_prev))
                level_slack = 128.0*np.finfo(float).eps*(
                    1.0+abs(current_level))
                if flow > 0 and target_level > current_level+level_slack:
                    continue
                pts.append((a, b))
                if (a - at) ** 2 + (b - bt) ** 2 > 1e-24:
                    pts.append((at, bt))
                return pts, "capture", switches, (b, w)
        pts.append((a, b))

        if not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
            return pts, "box_exit", switches, (b, w)

    return pts, "abort_max_steps", switches, (b, w)


# --------------------------------------------------------------------- #
# branch tracers                                                         #
# --------------------------------------------------------------------- #


def _native_curve_diagnostics(m, Y, digits, start):
    kernel = getattr(m, "_native_kernel", None)
    if kernel is None or not hasattr(kernel, "curve_diagnostics"):
        return None
    if np.asarray(Y).shape[0] < 2:
        # No chord, so no angle and no residual to measure.  The native
        # entry point rejects such an array outright; the Python oracle
        # returns zeros from an empty loop, which is the right answer.
        return None
    K = ANGLE_DIGIT_BUDGET if digits is None else digits
    return kernel.curve_diagnostics(
        np.ascontiguousarray(Y, dtype=np.float64), float(K), int(start))


def _angle_energy_detail_python(m: Model, Y: np.ndarray,
                                digits: float = None, start: int = 1):
    """(E, n_resolved, n_unresolved) — angle energy over resolved vertices.

    E = Σ ½‖d_⊥‖² is the discrete integral-curve certificate (E = 0 ⟺ the
    chords are everywhere parallel to ∇L).  Each gradient component is a
    cancellation whose magnitude sets an evaluation floor ~eps·(term scale),
    so R = ‖∇L‖ / g_floor is how many significant digits the DIRECTION of ∇L
    carries.  A vertex is skipped when R < `digits`.

    The guard used to fire only at R < 1, a cliff that never triggers: the
    direction degrades CONTINUOUSLY (measured 7.7 digits at b = −3 falling to
    0.3 at b = −11 on the tricky portrait's escaping branch), so a certificate
    reading R ≈ 6 reports its own evaluation noise as curve geometry.  That is
    what broke the founding parity gate when the compute box was widened —
    99.7% of the reported 2.007e-08 came from b < −9, and 0 of 3999 vertices
    were skipped.

    `ANGLE_DIGIT_BUDGET = 1e3` (3 digits) chosen by measurement, not taste:
    it reproduces the historical passing gate almost exactly (E = 3.414e-14
    cutting at b = −6.931, against 3.704e-14 for the pre-inflation box edge at
    b = −7.069).  The principle explains the accident — that box happened to
    stop right where the direction still had ~3 digits.

    n_unresolved MUST be reported: a budget strict enough to silence all noise
    would skip every vertex and return E = 0, which is a vacuous pass, not a
    clean one.  Beyond the resolved region the branch is certified
    ALGEBRAICALLY instead — out there it IS the backbone a*(b), with
    |w_s/a*| ~ 3.4e-25.  The two certificates run in opposite directions
    (geometric decays outward, algebraic improves outward) and overlap around
    b ≈ −4.5, where both are strong.
    """
    K = ANGLE_DIGIT_BUDGET if digits is None else digits
    E = 0.0
    used = 0
    skipped = 0
    eps = np.finfo(float).eps
    for k in range(max(1, start), len(Y) - 1):
        a, b = Y[k, 0], Y[k, 1]
        d = Y[k + 1] - Y[k - 1]
        g = m.gradL(a, b)
        ng = float(np.hypot(g[0], g[1]))
        nd = float(np.hypot(d[0], d[1]))
        scale_a = 2.0 * (abs(a) * m.A(b) + abs(m.B(b)))
        scale_b = 2.0 * abs(a) * abs(m.Bp(b)) + a * a * abs(m.Ap(b))
        g_floor = 16.0 * eps * float(np.hypot(scale_a, scale_b))
        if ng < max(1e-12, K * g_floor) or nd < 1e-14:
            skipped += 1
            continue
        gh = g / ng
        dp = d - (gh @ d) * gh
        E += 0.5 * float(dp @ dp)
        used += 1
    return E, used, skipped


def angle_energy_detail(m: Model, Y: np.ndarray, digits: float = None,
                        start: int = 1):
    """Native angle certificate; Python implementation retained as oracle."""
    result = _native_curve_diagnostics(m, Y, digits, start)
    if result is not None:
        return result[:3]
    return _angle_energy_detail_python(m, Y, digits=digits, start=start)


def angle_energy(m: Model, Y: np.ndarray) -> float:
    """E over the resolved vertices; see `angle_energy_detail` for the counts
    (which callers need — E alone cannot distinguish clean from vacuous)."""
    return angle_energy_detail(m, Y)[0]


def _backbone_residual_python(m: Model, Y: np.ndarray, digits: float = None,
                              start: int = 1) -> float:
    """max |w| / |a*| over the vertices `angle_energy` could NOT resolve.

    The ALGEBRAIC certificate.  Where the geometric one runs out of digits the
    branch has become the backbone a* = B/A, an exact rational function, so
    "is this the invariant manifold" is answerable without any chord geometry —
    and it IMPROVES outward, exactly where `angle_energy` decays.

    Scoped to the UNRESOLVED vertices on purpose.  Measured over the whole
    polyline instead it reports where the branch is legitimately far from the
    backbone (a genuine curve, or a*≈0 making the ratio unbounded), and then
    fails on branches it was never asked about: out of sample that read as
    37 of 75 branches "uncertified" when in fact their unresolved tails were
    fine.  A certificate must be measured where it is relied upon.

    Returns 0.0 when nothing was unresolved — there is no claim to make.
    """
    K = ANGLE_DIGIT_BUDGET if digits is None else digits
    eps = np.finfo(float).eps
    worst = 0.0
    for k in range(max(1, start), len(Y) - 1):
        a, b = float(Y[k, 0]), float(Y[k, 1])
        g = m.gradL(a, b)
        ng = float(np.hypot(g[0], g[1]))
        scale_a = 2.0 * (abs(a) * m.A(b) + abs(m.B(b)))
        scale_b = 2.0 * abs(a) * abs(m.Bp(b)) + a * a * abs(m.Ap(b))
        g_floor = 16.0 * eps * float(np.hypot(scale_a, scale_b))
        if ng >= max(1e-12, K * g_floor):
            continue                      # resolved: angle_energy covers it
        astar = float(m.a_star(b))
        if abs(astar) < 1e-300:
            continue
        worst = max(worst, abs(a - astar) / abs(astar))
    return worst


def backbone_residual(m: Model, Y: np.ndarray, digits: float = None,
                      start: int = 1) -> float:
    """Native algebraic tail certificate; Python retained as oracle."""
    result = _native_curve_diagnostics(m, Y, digits, start)
    if result is not None:
        return result[3]
    return _backbone_residual_python(m, Y, digits=digits, start=start)


def trace_unstable(m: Model, b_saddle: float, target: tuple[float, float],
                   box=(-50.0, 50.0, -50.0, 50.0), n_grid: int = 4001,
                   ds: float | None = None,
                   cap_r: float | None = None,
                   _launch_rel: float | None = None,
                   critical_local=None, critical_stub=None,
                   capture_targets=None, arrival_local=None,
                   candidate_minima=None, candidate_enumeration=None,
                   critical_points=None) -> Branch:
    """Candidate-directed unstable continuation to capture, exit, or failure.

    Zone loop per the dispatcher contract: whenever the sounding says
    shallow water and the trajectory is slaved, the Hadamard fixed point
    owns the stretch; elsewhere the continuation engine owns it.  A branch
    may alternate zones several times (a mild saddle whose journey passes
    through stiff country and back).  Every junction records a seam
    residual (fixed point vs engine at the handoff point).
    """
    a_t, b_t = target
    targets = ([tuple(map(float, target))] if capture_targets is None else
               [tuple(map(float, q)) for q in capture_targets])
    critical = _critical_array(critical_points)
    span = b_t - b_saddle
    sgn = 1.0 if span > 0 else -1.0
    if ds is None:
        ds = abs(span) / 4000.0
    if cap_r is None:
        cap_r = 4.0 * ds             # capture connector ~ a few chords

    bg = np.linspace(b_saddle, b_t, n_grid)
    kap = depth_gauge_floor(m, bg)
    # 0/0 at the saddle and target (critical points): fill from neighbors
    kap[0], kap[-1] = kap[1], kap[-2]
    kap = np.where(np.isfinite(kap), kap, np.inf)
    hstep = abs(span) / (n_grid - 1)

    pts: list[tuple[float, float]] = []
    certs: dict = {"seam_residuals": []}
    diag: dict = {"kappa_saddle": float(kap[0]), "zones": [],
                  "switches": 0}

    def grid_index(b):
        return int(round((b - b_saddle) / (b_t - b_saddle) * (n_grid - 1)))

    # initial state
    b_cur = float(b_saddle)
    w_cur = 0.0
    launch_scale = None

    diag["kappa_spectral_saddle"] = float(sounding(m, b_saddle))
    launch_rel = UNSTABLE_LAUNCH_REL if _launch_rel is None else _launch_rel
    db0 = sgn*min(
        abs(launch_rel*span), 1e-3*(1.0+abs(b_saddle)))
    saddle_mild = diag["kappa_spectral_saddle"] < KAPPA_HI
    if critical_stub is not None:
        local_Y = np.asarray(critical_stub.curve)
        pts.extend(map(tuple, local_Y))
        a_cur, b_cur = local_Y[-1]
        w_cur = float(a_cur - m.a_star(b_cur))
        sc = dict(critical_stub.certificates)
        physical_reach = max(
            float(np.hypot(*(local_Y[-1]-local_Y[0]))),
            np.finfo(float).tiny)
        # The first global step must continue the *spacing* at the handoff,
        # not jump by the total physical reach of a finely sampled stub.
        # Confusing the two made a 0.05-reach/512-chord stub request a first
        # global step of 0.05 and could overflow an otherwise resolved GL6
        # launch.  The gentle 1.06 ramp below grows from the last certified
        # chord instead.
        launch_scale = max(
            float(np.hypot(*(local_Y[-1]-local_Y[-2])))
            if len(local_Y) > 1 else physical_reach,
            np.finfo(float).tiny)
        diag.update({
            "materialized_stub": True,
            "stub_reach": float(sc["reach"]),
            "stub_physical_reach": physical_reach,
            "stub_handoff_chord": launch_scale,
            "stub_global_field_ready": bool(sc["global_field_ready"]),
            "stub_endpoint_evaluator": (
                "global" if sc["global_field_ready"] else "centered"),
            "critical_chart": True,
            "critical_order": 6,
            "critical_steps": len(local_Y) - 1,
        })
        if not sc["global_field_ready"]:
            diag["conditioning_refusal"] = {
                "global_resolution_margin":
                    sc.get("global_resolution_margin"),
                "field_absolute_error": sc.get("field_absolute_error"),
                "global_roundoff_floor": sc.get("global_roundoff_floor"),
                "injectivity_margin": sc.get("injectivity_margin"),
                "spectral_resolution_margin":
                    sc.get("spectral_resolution_margin"),
                "fp64_spectral_resolved":
                    sc.get("fp64_spectral_resolved"),
            }
            if saddle_mild or len(local_Y) < 2:
                # A mild saddle whose stub cannot reach the global field has
                # no second owner -- the sounding says deep water, where the
                # continuation engine is the only candidate and it is exactly
                # the field that failed to condition.  The refusal stands.
                #
                # A single-vertex stub is refused for a different reason:
                # there is no chord, hence no launch scale and nothing for the
                # fixed point to continue from.  Falling through on one
                # produced a one-vertex branch whose certificate call then
                # failed outright (seed 2149547, b* = -16384).
                return Branch(
                    "unstable", local_Y, "abort_conditioning_handoff",
                    {"handoff_certified": False}, diag)
            # Otherwise the sounding already says SHALLOW, and the Hadamard
            # fixed point in the zone loop below is the other downstream owner
            # this chart was always meant to have (see the comment on the
            # branch immediately following).  global_field_ready is a
            # statement about the stiff ODE field at the stub endpoint; the
            # slaved graph does not evaluate that field at all, only a* and
            # u', which are rational functions of the exact coefficients.
            #
            # Refusing here is what made the dead-neuron saddles unreachable:
            # at b = 50.7 with a* = -5e-6, A(b) ~ 1e10, so the global handoff
            # cannot condition and BOTH unstable branches aborted at launch
            # with 514 vertices, taking the whole portrait to branch_abort.
            # The manifold is perfectly well behaved there -- it is the
            # backbone to machine precision, which is precisely the regime the
            # fixed point owns.
            certs["handoff_certified"] = False
            diag["shallow_launch"] = True
    elif (saddle_mild and critical_local is not None
            and critical_local.native is not None):
        # The centered critical chart precedes either downstream owner:
        # continuation in deep water or Hadamard graph transform in shallow.
        try:
            local_Y, local_diag = _critical_chart_curve(
                critical_local, "unstable", 1 if db0 > 0 else -1,
                "b", abs(db0))
        except (ArithmeticError, ValueError) as exc:
            diag["critical_chart_rejected"] = str(exc)
            fr = saddle_frame(m, b_saddle, critical_local.a,
                              critical_local)["unstable"]
            d_b = fr["d_b"] if fr["d_b"] * span >= 0 else -fr["d_b"]
            d_w = fr["d_w"] * (1 if fr["d_b"] * span >= 0 else -1)
            pts.append((critical_local.a, b_saddle))
            b_cur, w_cur = b_saddle + db0, (d_w / d_b) * db0
        else:
            pts.extend(map(tuple, local_Y))
            a_cur, b_cur = local_Y[-1]
            w_cur = float(a_cur - m.a_star(b_cur))
            diag.update(local_diag)
        launch_scale = abs(db0)
    elif saddle_mild:
        # Compatibility path when the native local chart is unavailable.
        a_s = (critical_local.a if critical_local is not None
               else float(m.a_star(b_saddle)))
        fr = saddle_frame(m, b_saddle, a_s, critical_local)["unstable"]
        d_b = fr["d_b"] if fr["d_b"] * span >= 0 else -fr["d_b"]
        d_w = fr["d_w"] * (1 if fr["d_b"] * span >= 0 else -1)
        if abs(d_b) < 1e-12:
            return Branch("unstable", np.array([[a_s, b_saddle]]),
                          "abort_not_graph", certs, diag)
        pts.append((a_s, b_saddle))
        b_cur, w_cur = b_saddle + db0, (d_w / d_b) * db0
        launch_scale = abs(db0)

    term = None
    live_minima = list(candidate_minima or ())
    if (critical_stub is not None and len(live_minima) > 1):
        sc = dict(critical_stub.certificates)
        if (sc.get("global_field_ready", 0.0)
                and sc.get("global_resolution_margin", 0.0) >= 1024.0):
            from . import topology
            for _event in range(len(live_minima)):
                event_curve, event_term, captured = (
                    _potential_rate_level_event(
                        m, float(m.s_a_star(b_cur)+w_cur), float(b_cur),
                        [(q.a, q.b) for q in live_minima],
                        box, cap_r, diag, critical=critical))
                pts.extend(event_curve[1:])
                if event_term == "capture":
                    term = "capture"
                    b_cur = float(captured[1])
                    w_cur = float(captured[0]-m.s_a_star(b_cur))
                    break
                if event_term == "box_exit":
                    term = "box_exit"
                    break
                if len(event_curve) > 1:
                    endpoint = event_curve[-1]
                    b_cur = float(endpoint[1])
                    w_cur = float(endpoint[0]-m.s_a_star(b_cur))
                    launch_scale = float(np.hypot(*(
                        np.asarray(event_curve[-1])
                        - np.asarray(event_curve[-2]))))
                if event_term == "step_failure":
                    event_level = diag["candidate_level_events"][-1][
                        "event_level"]
                    level_slack = 2048*np.finfo(float).eps*(
                        1.0+abs(event_level))
                    event_minima = [
                        q for q in live_minima
                        if abs(float(m.L(q.a, q.b))-event_level)
                        <= level_slack]
                    if len(event_minima) == 1:
                        q = event_minima[0]
                        arrival, arrival_term = _centered_raw_arrival(
                            event_curve[-1], (q.a, q.b), q.local,
                            cap_r, diag)
                        pts.extend(arrival[1:])
                        if arrival_term == "capture":
                            term = "capture"
                            b_cur = float(q.b)
                            w_cur = float(q.a-m.s_a_star(b_cur))
                    break
                if event_term != "level_event":
                    break
                endpoint = event_curve[-1]
                feasible_ids = None
                if candidate_enumeration is not None:
                    feasible_ids = {
                        id(q) for q in topology.sublevel_component_minima(
                            m, candidate_enumeration, endpoint)}
                # Combine the newly measured exact sublevel component with
                # the loss event.  In particular, a minimum strictly above
                # the current loss cannot terminate a descent orbit.
                current_level = float(m.L(*endpoint))
                level_slack = 2048*np.finfo(float).eps*(
                    1.0+abs(current_level))
                reduced = [
                    q for q in live_minima
                    if (feasible_ids is None or id(q) in feasible_ids)
                    and float(m.L(q.a, q.b)) <= current_level+level_slack]
                if len(reduced) >= len(live_minima):
                    break
                live_minima = reduced
                targets = [(float(q.a), float(q.b)) for q in live_minima]
                diag.setdefault("candidate_domain_sizes", []).append(
                    len(live_minima))
                if len(live_minima) <= 1:
                    break
            if term is None and len(live_minima) == 1:
                chosen = live_minima[0]
                a_t, b_t = float(chosen.a), float(chosen.b)
                target = (a_t, b_t)
                arrival_local = chosen.local
                span = b_t-b_saddle
                bg = np.linspace(b_saddle, b_t, n_grid)
                kap = depth_gauge_floor(m, bg)
                kap[0], kap[-1] = kap[1], kap[-2]
                kap = np.where(np.isfinite(kap), kap, np.inf)
                hstep = abs(span)/(n_grid-1)

    if (critical_stub is not None and arrival_local is not None
            and critical_local is not None):
        # `target` is the independently discovered destination used to
        # parameterize this refinement.  `targets` deliberately remains the
        # complete live capture set: numerical refinement must not turn a
        # coarse topological label into an assumption.
        at_unique, bt_unique = map(float, target)
        anisotropy = (
            abs(at_unique-critical_local.a)
            / max(abs(bt_unique-b_saddle), 1e-300))
        diag["target_anisotropy"] = float(anisotropy)
        sc = dict(critical_stub.certificates)
        if (sc.get("global_field_ready", 0.0)
                and sc.get("global_resolution_margin", 0.0) >= 1024.0):
            prefix, b_cur, w_cur, prefix_term = _potential_rate_prefix(
                m, float(m.s_a_star(b_cur)+w_cur), float(b_cur),
                (at_unique, bt_unique), box, cap_r, diag,
                critical=critical)
            pts.extend(prefix[1:])
            if prefix_term == "capture":
                term = "capture"
            elif prefix_term == "box_exit":
                term = "box_exit"
            elif prefix_term in {"near_target", "step_failure", "budget"}:
                arrival, arrival_term = _centered_raw_arrival(
                    prefix[-1], (at_unique, bt_unique),
                    arrival_local, cap_r, diag)
                pts.extend(arrival[1:])
                if arrival_term == "capture":
                    term = "capture"
                    b_cur = bt_unique
                    w_cur = float(at_unique-m.s_a_star(b_cur))
                elif len(arrival) > 1:
                    b_cur = float(arrival[-1][1])
                    w_cur = float(arrival[-1][0]-m.s_a_star(b_cur))
            launch_scale = (
                float(np.hypot(*(np.asarray(prefix[-1])
                                 - np.asarray(prefix[-2]))))
                if len(prefix) > 1 else launch_scale)

    for _zone in range(0 if term is not None else 32):
        i_cur = grid_index(b_cur)
        # Continuation may legitimately overshoot the nominal saddle/target
        # interval before capture is discovered.  The sounding grid is only a
        # lookup table on that interval: never use its unbounded extrapolated
        # index for NumPy access or to size a graph-transform zone.
        i_grid = min(max(i_cur, 0), n_grid - 1)
        # Ownership is a pointwise decision.  On very large target intervals
        # the nearest sounding node can be tens of thousands of units away
        # and may lie in a completely different conditioning regime.
        g_cur = _s_depth_gauge_floor(m, b_cur)
        if (not pts and abs(b_cur-b_saddle)
                <= 16.0*np.finfo(float).eps*(1.0+abs(b_saddle))):
            # The slow-depth expression is 0/0 at the critical point itself.
            # Its Hessian spectral ratio is the exact limiting ownership
            # datum.  This compatibility path is used only when no
            # materialized local stub has already moved the state away.
            g_cur = max(g_cur, diag["kappa_spectral_saddle"])
        if g_cur >= KAPPA_HI:
            # NB: the junction is placed by this VALIDITY threshold, not by
            # minimising the two representations' disagreement.  That was
            # tried and reverted — see docs/inverse_construction.md: the
            # disagreement is a calibrated error ESTIMATE, so minimising it
            # seeks its own zeros (sign crossings), where it reads ~0 while
            # both errors are finite.
            # ---- shallow water: Hadamard fixed point owns it ---------- #
            # grid index always INCREASES toward the target (bg runs
            # saddle -> target regardless of the sign of the span)
            j = i_grid
            while j < n_grid and kap[j] >= KAPPA_EXIT:
                j += 1
            j = min(max(j, 0), n_grid - 1)
            # grid_index rounds to NEAREST, so bg[i_cur] can lie BEHIND b_cur.
            # When the while loop does not advance (that node's gauge is
            # already below KAPPA_EXIT) the zone would run backward: measured
            # at g4/b*=20480, entering at b=4.674 and ending at b=3.381, after
            # which the engine re-runs the same ground and hands off again with
            # an O(1) seam.  The zone must end strictly ahead of where it
            # started.
            while j < n_grid - 1 and (bg[j] - b_cur) * sgn <= 0.0:
                j += 1
            # The sounding grid locates a candidate zone; it does not set the
            # graph's geometric resolution.  In a 2^17 targeted case one
            # sounding cell spanned about 58,000 units and the old minimum of
            # eight samples necessarily rejected an otherwise smooth slow
            # graph.  Resolve it at the continuation chord scale.
            # The gauge proposes the zone's end, but the gauge knows only
            # stiffness: on a bottom saddle's descent the shallow stretch can
            # contain interior critical points of u (measured: a proposed
            # zone from b = -4.46 to +0.72 across two minima and two
            # saddles at d_g = 13, kappa_spectral ~ 1e19).  The slaved fixed
            # point rightly refuses to self-certify across u' roots, and the
            # rejection fallback then pushes the stiff ODE through the whole
            # channel -- abort_step_failure was this.  Clamp the zone at the
            # first candidate ahead: the fixed point certifies on a
            # monotone stretch and the zone's own capture check fires there.
            b_end = float(bg[j])
            ahead = [bt for (_, bt) in targets
                     if (bt - b_cur) * sgn > 1e-12 * (1.0 + abs(b_cur))]
            if ahead:
                b_first = min(ahead, key=lambda bt: (bt - b_cur) * sgn)
                if (b_end - b_first) * sgn > 0.0:
                    b_end = float(b_first)
            zone_span = abs(b_end-float(b_cur))
            n_pts = max(
                abs(j-i_grid)+1, 8,
                int(np.ceil(zone_span/max(ds, np.finfo(float).tiny)))+1)
            j = min(j, n_grid - 1)
            b_zone = np.linspace(b_cur, b_end, n_pts)
            w_zone, iters, rel = slow_fixed_point(m, b_zone)
            if rel > 1e-10 or not np.all(np.isfinite(w_zone)):
                # SELF-CERTIFICATION FAILED: the sounding gauge 2A/|u''|
                # spikes falsely at inflections of u (u'' = 0), where no
                # real slaving exists.  Reject the zone and push the
                # engine through the spike with the trigger gated off
                # until past it.
                diag["zones"].append(("shallow_rejected", b_cur,
                                      b_end, iters))
                gate_end = b_end
                if (gate_end-b_cur)*sgn <= 0.0:
                    # Continuation has overshot the nominal sounding
                    # interval.  Re-entering with a gate behind the current
                    # state immediately triggers the same rejected zone and
                    # eventually aborts at the zone-count limit.  The finite
                    # sounding grid has no forward shallow-water ownership
                    # information here, so let the certified continuation
                    # engine own the remainder to the trace-box boundary.
                    gate_end = float(box[3] if sgn > 0.0 else box[2])
                    diag["sounding_interval_exhausted"] = (
                        diag.get("sounding_interval_exhausted", 0)+1)
                shallow_gate = (gate_end, sgn)
                tail, term_e, sw, (b_cur, w_cur) = _continue_curve(
                    m, b_cur, w_cur, +1, targets, box, ds,
                    cap_r=cap_r, ds0=launch_scale,
                    shallow_gate=shallow_gate, engine_diag=diag,
                    centered_local=critical_local)
                pts.extend(tail if not pts else tail[1:])
                diag["switches"] += sw
                launch_scale = ds
                if term_e == "enter_shallow":
                    continue
                term = term_e
                break
            diag["zones"].append(("shallow", b_cur, b_end, iters))
            # seam: fixed point vs incoming state at the junction
            certs["seam_residuals"].append(abs(float(w_zone[0]) - w_cur))
            a_zone = m.a_star(b_zone) + w_zone
            start = 0
            if pts and abs(pts[-1][1] - float(b_zone[0])) < 1e-12 * (
                    1 + abs(b_cur)):
                start = 1                    # grid re-includes current point
            previous = (np.asarray(pts[-1]) if pts else
                        np.array([a_zone[0], b_zone[0]]))
            pts.extend(zip(a_zone.tolist()[start:], b_zone.tolist()[start:]))
            captured_at = None
            for k, (az, bz) in enumerate(zip(a_zone[start:], b_zone[start:]),
                                         start=start):
                for at, bt in targets:
                    if _segment_capture(
                            float(previous[0]), float(previous[1]),
                            float(az), float(bz), at, bt, cap_r):
                        target_level = float(m.L(at, bt))
                        current_level = float(m.L(
                            float(previous[0]), float(previous[1])))
                        level_slack = 128.0*np.finfo(float).eps*(
                            1.0+abs(current_level))
                        if target_level > current_level+level_slack:
                            continue
                        captured_at = (k, at, bt)
                        break
                if captured_at is not None:
                    break
                previous = np.array([az, bz])
            if captured_at is not None:
                k, at, bt = captured_at
                keep = len(pts) - (len(b_zone)-k)
                pts = pts[:max(keep, 0)]
                pts.append((at, bt))
                b_cur, w_cur = bt, float(at-m.a_star(bt))
                term = "capture"
                break
            b_cur, w_cur = float(b_zone[-1]), float(w_zone[-1])
            launch_scale = hstep
            if abs(b_cur - b_t) <= hstep * 1.5:
                term = "box_exit"
                break
        else:
            # ---- deep water / steep: the continuation engine ---------- #
            tail, term_e, sw, (b_cur, w_cur) = _continue_curve(
                m, b_cur, w_cur, +1, targets, box, ds,
                cap_r=cap_r, ds0=launch_scale, engine_diag=diag,
                centered_local=critical_local)
            pts.extend(tail if not pts else tail[1:])
            diag["switches"] += sw
            diag["zones"].append(("engine", tail[0][1], b_cur, term_e))
            launch_scale = ds
            if term_e == "enter_shallow":
                continue
            term = term_e
            break
    else:
        if term is None:
            term = "abort_zone_limit"

    Y = np.array(pts)
    certificate_start = int(diag.get("critical_steps", 0)) + 1
    _E, _used, _skip = angle_energy_detail(
        m, Y, start=certificate_start)
    certs["angle_energy"] = _E
    certs["angle_resolved"] = _used
    certs["angle_unresolved"] = _skip
    certs["backbone_residual"] = backbone_residual(
        m, Y, start=certificate_start)
    certs["endpoint"] = tuple(Y[-1])
    if certs["seam_residuals"]:
        certs["seam_residual"] = max(certs["seam_residuals"])
    diag["stiff_frac"] = float(np.mean(kap >= KAPPA_HI))
    diag["final_state_bw"] = (float(b_cur), float(w_cur))
    return Branch("unstable", Y, term, certs, diag)


def _slaved_valley_points(m: Model, b0: float, b1: float,
                          n_grid: int) -> tuple[np.ndarray, str]:
    b = np.linspace(b0, b1, n_grid)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        A = m.A(b)
        w = m.a_star_p(b) * m.u_p(b) / (2.0 * A)
        a = m.a_star(b) + w
    Y = np.column_stack([a, b])
    finite = np.isfinite(Y[:, 0]) & np.isfinite(Y[:, 1])
    if not np.all(finite):
        last = int(np.argmax(~finite))
        return Y[:max(last, 1)], "abort_nonfinite"
    return Y, "box_exit"


def trace_valley_exit(m: Model, b_saddle: float, b_exit: float,
                      box=(-50.0, 50.0, -50.0, 50.0),
                      n_grid: int = 4001,
                      local_until: float | None = None,
                      critical_local=None, critical_stub=None) -> Branch:
    """Unstable branch with no finite minimum on that side: valley → box edge.

    This is the pseudo-target case from portrait assembly.  The branch needs
    an honest eigenvector/ODE launch in the metrological window, then becomes
    asymptotic to the shallow slaved valley.  The far tail uses the slaved
    graph to avoid evaluating second-derivative gauges across huge high-degree
    intervals where only the compute-box exit is being certified.
    """
    direction = 1.0 if b_exit > b_saddle else -1.0
    zones = []
    seam_residuals = []
    switches = 0

    b_tail0 = float(b_saddle)
    prefix = None
    if local_until is None and critical_stub is not None:
        local_until = b_saddle + direction*min(
            10.0, abs(b_exit-b_saddle))
    if local_until is not None:
        b_local = float(local_until)
        if direction * (b_local - b_saddle) > 0:
            if direction * (b_local - b_exit) > 0:
                b_local = float(b_exit)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                A = m.A(b_local)
                a_local = float(m.a_star(b_local)
                                + m.a_star_p(b_local) * m.u_p(b_local)
                                / (2.0 * A))
            if np.isfinite(a_local):
                local = trace_unstable(
                    m, b_saddle, (a_local, b_local), box=box,
                    ds=abs(b_local - b_saddle) / 4000.0,
                    critical_local=critical_local,
                    critical_stub=critical_stub)
                if local.term in ("capture", "box_exit"):
                    prefix = local.Y
                    b_tail0 = float(prefix[-1, 1])
                    switches = int(local.diag.get("switches", 0))
                    zones.extend(local.diag.get("zones", []))
                    if direction * (b_exit - b_tail0) > 0:
                        tail, term = _slaved_valley_points(
                            m, b_tail0, b_exit, n_grid)
                        seam_residuals.append(float(np.hypot(
                            tail[0, 0] - prefix[-1, 0],
                            tail[0, 1] - prefix[-1, 1])))
                        Y = np.vstack([prefix, tail[1:]])
                    else:
                        Y, term = prefix, "box_exit"
                else:
                    prefix = None

    if prefix is None:
        Y, term = _slaved_valley_points(m, b_saddle, b_exit, n_grid)

    _E, _used, _skip = angle_energy_detail(m, Y)
    certs = {"angle_energy": _E, "angle_resolved": _used,
             "angle_unresolved": _skip,
             "backbone_residual": backbone_residual(m, Y),
             "endpoint": tuple(Y[-1])}
    if seam_residuals:
        certs["seam_residuals"] = seam_residuals
        certs["seam_residual"] = max(seam_residuals)
    diag = {"saddle_b": b_saddle, "target": None, "switches": 0,
            "zones": zones + [("slaved_valley_exit", float(b_tail0),
                               float(Y[-1, 1]), len(Y))],
            "stiff_frac": 1.0}
    diag["switches"] = switches
    return Branch("unstable", Y, term, certs, diag)


def trace_stable(m: Model, b_saddle: float, sign: int,
                 box=(-25.0, 25.0, -12.0, 16.0), ds: float | None = None,
                 delta: float | None = None, critical_local=None,
                 critical_stub=None, critical_points=None) -> Branch:
    """Stable branch (separatrix): saddle → box exit, ascent flow.

    Fast-graph launch from the exact eigenvector jet; the continuation
    engine handles any folds back to the slow chart.
    """
    a_s = (critical_local.a if critical_local is not None
           else float(m.a_star(b_saddle)))
    fr = saddle_frame(m, b_saddle, a_s, critical_local)["stable"]
    d_w = fr["d_w"] if sign * fr["d_w"] >= 0 else -fr["d_w"]
    d_b = fr["d_b"] * (1 if sign * fr["d_w"] >= 0 else -1)
    if abs(d_w) < 1e-12:
        return Branch("stable", np.array([[a_s, b_saddle]]),
                      "abort_not_graph")
    if ds is None:
        ds = (abs(box[1] - box[0]) + abs(box[3] - box[2])) / 30000.0
    if delta is None:
        delta = STABLE_LAUNCH_DELTA

    diag = {"switches": 0}
    prefix = []
    launch_scale = abs(delta)
    if critical_stub is not None:
        local_Y = np.asarray(critical_stub.curve)
        prefix = list(map(tuple, local_Y[1:]))
        a0, b0 = local_Y[-1]
        w0 = float(a0 - m.a_star(b0))
        sc = dict(critical_stub.certificates)
        launch_scale = max(
            float(np.hypot(*(local_Y[-1]-local_Y[0]))),
            np.finfo(float).tiny)
        diag.update({
            "materialized_stub": True,
            "stub_reach": float(sc["reach"]),
            "stub_physical_reach": launch_scale,
            "stub_global_field_ready": bool(sc["global_field_ready"]),
            "stub_endpoint_evaluator": (
                "global" if sc["global_field_ready"] else "centered"),
            "critical_chart": True,
            "critical_order": 6,
            "critical_steps": len(local_Y) - 1,
        })
        if not sc["global_field_ready"]:
            diag["conditioning_refusal"] = {
                "global_resolution_margin":
                    sc.get("global_resolution_margin"),
                "field_absolute_error": sc.get("field_absolute_error"),
                "global_roundoff_floor": sc.get("global_roundoff_floor"),
                "injectivity_margin": sc.get("injectivity_margin"),
                "spectral_resolution_margin":
                    sc.get("spectral_resolution_margin"),
                "fp64_spectral_resolved":
                    sc.get("fp64_spectral_resolved"),
            }
            return Branch(
                "stable", local_Y, "abort_conditioning_handoff",
                {"handoff_certified": False}, diag)
    elif critical_local is not None and critical_local.native is not None:
        try:
            local_Y, local_diag = _critical_chart_curve(
                critical_local, "stable", sign, "w", abs(delta))
        except (ArithmeticError, ValueError) as exc:
            diag["critical_chart_rejected"] = str(exc)
            w0 = sign * delta
            b0 = b_saddle + (d_b / d_w) * w0
        else:
            prefix = list(map(tuple, local_Y[1:]))
            a0, b0 = local_Y[-1]
            w0 = float(a0 - m.a_star(b0))
            diag.update(local_diag)
    else:
        w0 = sign * delta
        b0 = b_saddle + (d_b / d_w) * w0
    potential_term = None
    if (critical_stub is not None
            and dict(critical_stub.certificates).get(
                "global_field_ready", 0.0)):
        potential_pts, potential_term = _potential_rate_box_exit(
            m, (m.a_star(b0)+w0, b0), box, ds, diag,
            critical=_critical_array(critical_points))
        if prefix:
            prefix.extend(potential_pts[1:])
        else:
            prefix = potential_pts
        b0 = float(potential_pts[-1][1])
        w0 = float(potential_pts[-1][0]-m.a_star(b0))
    if potential_term == "box_exit":
        engine_pts, term, sw = [prefix[-1]], "box_exit", 0
    else:
        engine_pts, term, sw, _ = _continue_curve(
            m, b0, w0, -1, [], box, ds, ds0=launch_scale,
            engine_diag=diag, centered_local=critical_local)
    pts = prefix + engine_pts[1:] if prefix else engine_pts
    Y = np.array([(a_s, b_saddle)] + pts)
    certificate_start = int(diag.get("critical_steps", 0)) + 1
    _E, _used, _skip = angle_energy_detail(
        m, Y, start=certificate_start)
    certs = {"angle_energy": _E, "angle_resolved": _used,
             "angle_unresolved": _skip,
             "backbone_residual": backbone_residual(
                 m, Y, start=certificate_start),
             "endpoint": tuple(Y[-1])}
    diag["switches"] = sw
    return Branch("stable", Y, term, certs, diag)
