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
    lam = np.array([mid - rad, mid + rad], dtype=float)

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


def saddle_frame(m: Model, b_s: float, a_s: float):
    """Closed-form 2x2 Hessian launch data at a saddle, in chart components.

    Returns dict with unstable/stable eigenvectors as (d_w, d_b) pairs:
    d_w = v1 - a*'·v2, d_b = v2 for eigvec (v1, v2) of the Hessian.
    Chart selection at launch uses these components (never the stiff-limit
    formulas): the eigvecs are orthogonal, so at least one chart is
    well-posed for each manifold at every saddle.
    """
    H = m.hessL(a_s, b_s)
    lam, V = _sym2_eigh(H)
    asp = m.a_star_p(b_s)
    out = {}
    for name, idx in (("unstable", int(np.argmin(lam))),
                      ("stable", int(np.argmax(lam)))):
        v = V[:, idx]
        out[name] = {"lam": lam[idx],
                     "d_w": v[0] - asp * v[1],
                     "d_b": v[1]}
    return out


# --------------------------------------------------------------------- #
# slow-graph fixed point (shallow water)                                 #
# --------------------------------------------------------------------- #


def slow_fixed_point(m: Model, b_grid: np.ndarray, tol: float = 1e-13,
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
            wp = np.gradient(w, b_grid)
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


def _continue_curve(m: Model, b0: float, w0: float, flow: int,
                    targets, box, ds: float, max_steps: int = 200000,
                    cap_r: float = 2e-3, ds0: float | None = None,
                    shallow_gate=None):
    """Walk the trajectory through chart pieces.

    flow: +1 descent (unstable branches), -1 ascent (separatrices).
    targets: list of (a, b) capture points (adjacent minima) or [].
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

    vb, vw = _s_velocities(m, b, w)
    vb, vw = flow * vb, flow * vw
    chart = "slow" if abs(vw) <= R_SWITCH * abs(vb) else "fast"
    switches = 0
    recent_b: list[float] = []          # stall detector window
    # geometric launch ramp: begin at the launch scale, grow into ds, so
    # polyline spacing is smooth and the angle-energy carries no launch kink
    cur = ds if ds0 is None else min(ds, max(ds0, 1e-12))

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
        for _retry in range(7):
            try:
                if chart == "slow":
                    h = cur / (1.0 + (vw / vb) ** 2) ** 0.5 * (
                        1.0 if vb > 0 else -1.0)
                    w_new = native.slow_step(b_prev, w_prev, h) \
                        if native is not None else \
                        gauss.gl4_scalar(sf, sj, b_prev, w_prev, h)
                    b_new = b_prev + h
                else:
                    h = cur / (1.0 + (vb / vw) ** 2) ** 0.5 * (
                        1.0 if vw > 0 else -1.0)
                    b_new = native.fast_step(w_prev, b_prev, h) \
                        if native is not None else \
                        gauss.gl4_scalar(ff, fj, w_prev, b_prev, h)
                    w_new = w_prev + h
            except (ZeroDivisionError, FloatingPointError, OverflowError):
                return pts, "abort_step_failure", switches, (b_prev, w_prev)
            if len(pts) >= 2 and np.isfinite(b_new) and np.isfinite(w_new):
                a_new = m.s_a_star(b_new) + w_new
                p1, p0 = pts[-1], pts[-2]
                d1a, d1b = p1[0] - p0[0], p1[1] - p0[1]
                d2a, d2b = a_new - p1[0], b_new - p1[1]
                n1 = (d1a * d1a + d1b * d1b) ** 0.5
                n2 = (d2a * d2a + d2b * d2b) ** 0.5
                if (n1 > 1e-14 and n2 > 1e-14
                        and (d1a * d2a + d1b * d2b) / (n1 * n2) < TURN_MAX
                        and cur > ds / 128.0):
                    cur *= 0.5
                    continue
            break
        b, w = b_new, w_new
        cur = min(ds, cur * 1.06)   # gentle ramp: the angle-energy
        # functional is a symmetric difference — 2nd-order on uniform
        # spacing, 1st-order under spacing jumps

        if not (np.isfinite(b) and np.isfinite(w)):
            return pts, "abort_nonfinite", switches, (b, w)

        a = m.s_a_star(b) + w
        pts.append((a, b))

        for (at, bt) in targets:
            if (a - at) ** 2 + (b - bt) ** 2 < cap_r**2:
                if (a - at) ** 2 + (b - bt) ** 2 > 1e-24:
                    pts.append((at, bt))
                return pts, "capture", switches, (b, w)

        if not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
            return pts, "box_exit", switches, (b, w)

    return pts, "abort_max_steps", switches, (b, w)


# --------------------------------------------------------------------- #
# branch tracers                                                         #
# --------------------------------------------------------------------- #


def angle_energy(m: Model, Y: np.ndarray) -> float:
    """E = Σ ½‖d_⊥‖²: the discrete integral-curve certificate (E = 0 ⟺
    the polyline chords are everywhere parallel to ∇L).

    Noise-floor aware (§4 doctrine): each gradient component is a
    cancellation of terms whose magnitudes set an evaluation floor
    ~eps·(term scale); where ‖∇L‖ sits below that floor its DIRECTION is
    numerically meaningless and the vertex is skipped — otherwise the
    certificate measures its own evaluation noise, not the curve (seen
    on far valley stretches where |∇L| ~ C_inf/b²).
    """
    E = 0.0
    eps = np.finfo(float).eps
    for k in range(1, len(Y) - 1):
        a, b = Y[k, 0], Y[k, 1]
        d = Y[k + 1] - Y[k - 1]
        g = m.gradL(a, b)
        ng = float(np.hypot(g[0], g[1]))
        nd = float(np.hypot(d[0], d[1]))
        scale_a = 2.0 * (abs(a) * m.A(b) + abs(m.B(b)))
        scale_b = 2.0 * abs(a) * abs(m.Bp(b)) + a * a * abs(m.Ap(b))
        g_floor = 16.0 * eps * float(np.hypot(scale_a, scale_b))
        if ng < max(1e-12, g_floor) or nd < 1e-14:
            continue
        gh = g / ng
        dp = d - (gh @ d) * gh
        E += 0.5 * float(dp @ dp)
    return E


def trace_unstable(m: Model, b_saddle: float, target: tuple[float, float],
                   box=(-50.0, 50.0, -50.0, 50.0), n_grid: int = 4001,
                   ds: float | None = None,
                   cap_r: float | None = None) -> Branch:
    """Unstable branch: saddle → adjacent minimum (or box exit).

    Zone loop per the dispatcher contract: whenever the sounding says
    shallow water and the trajectory is slaved, the Hadamard fixed point
    owns the stretch; elsewhere the continuation engine owns it.  A branch
    may alternate zones several times (a mild saddle whose journey passes
    through stiff country and back).  Every junction records a seam
    residual (fixed point vs engine at the handoff point).
    """
    a_t, b_t = target
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
    if kap[0] < KAPPA_HI:
        # mild saddle: exact-eigenvector jet launch into the engine
        a_s = float(m.a_star(b_saddle))
        fr = saddle_frame(m, b_saddle, a_s)["unstable"]
        d_b = fr["d_b"] if fr["d_b"] * span >= 0 else -fr["d_b"]
        d_w = fr["d_w"] * (1 if fr["d_b"] * span >= 0 else -1)
        if abs(d_b) < 1e-12:
            return Branch("unstable", np.array([[a_s, b_saddle]]),
                          "abort_not_graph", certs, diag)
        db0 = 1e-6 * span
        pts.append((a_s, b_saddle))
        b_cur, w_cur = b_saddle + db0, (d_w / d_b) * db0
        launch_scale = abs(db0)

    term = None
    for _zone in range(32):
        i_cur = grid_index(b_cur)
        g_cur = kap[min(max(i_cur, 0), n_grid - 1)]
        # The handoff point can lie just across KAPPA_HI from its nearest
        # precomputed grid node.  Use the exact scalar gauge too, otherwise
        # the engine can immediately return enter_shallow while the outer
        # dispatcher keeps sending it back into the engine.
        g_cur = max(float(g_cur), _s_depth_gauge_floor(m, b_cur))
        if g_cur >= KAPPA_HI:
            # ---- shallow water: Hadamard fixed point owns it ---------- #
            # grid index always INCREASES toward the target (bg runs
            # saddle -> target regardless of the sign of the span)
            j = i_cur
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
            n_pts = max(abs(j - i_cur) + 1, 8)
            j = min(j, n_grid - 1)
            b_zone = np.linspace(b_cur, bg[j], n_pts)
            w_zone, iters, rel = slow_fixed_point(m, b_zone)
            if rel > 1e-10 or not np.all(np.isfinite(w_zone)):
                # SELF-CERTIFICATION FAILED: the sounding gauge 2A/|u''|
                # spikes falsely at inflections of u (u'' = 0), where no
                # real slaving exists.  Reject the zone and push the
                # engine through the spike with the trigger gated off
                # until past it.
                diag["zones"].append(("shallow_rejected", b_cur,
                                      float(bg[j]), iters))
                shallow_gate = (float(bg[min(j, n_grid - 1)]), sgn)
                tail, term_e, sw, (b_cur, w_cur) = _continue_curve(
                    m, b_cur, w_cur, +1, [(a_t, b_t)], box, ds,
                    cap_r=cap_r, ds0=launch_scale,
                    shallow_gate=shallow_gate)
                pts.extend(tail if not pts else tail[1:])
                diag["switches"] += sw
                launch_scale = ds
                if term_e == "enter_shallow":
                    continue
                term = term_e
                break
            diag["zones"].append(("shallow", b_cur, float(bg[j]), iters))
            # seam: fixed point vs incoming state at the junction
            certs["seam_residuals"].append(abs(float(w_zone[0]) - w_cur))
            a_zone = m.a_star(b_zone) + w_zone
            start = 0
            if pts and abs(pts[-1][1] - float(b_zone[0])) < 1e-12 * (
                    1 + abs(b_cur)):
                start = 1                    # grid re-includes current point
            pts.extend(zip(a_zone.tolist()[start:], b_zone.tolist()[start:]))
            b_cur, w_cur = float(b_zone[-1]), float(w_zone[-1])
            launch_scale = hstep
            if abs(b_cur - b_t) <= hstep * 1.5:
                if (pts[-1][0] - a_t) ** 2 + (pts[-1][1] - b_t) ** 2 > 1e-24:
                    pts.append((a_t, b_t))
                term = "capture"
                break
        else:
            # ---- deep water / steep: the continuation engine ---------- #
            tail, term_e, sw, (b_cur, w_cur) = _continue_curve(
                m, b_cur, w_cur, +1, [(a_t, b_t)], box, ds,
                cap_r=cap_r, ds0=launch_scale)
            pts.extend(tail if not pts else tail[1:])
            diag["switches"] += sw
            diag["zones"].append(("engine", tail[0][1], b_cur, term_e))
            launch_scale = ds
            if term_e == "enter_shallow":
                continue
            term = term_e
            break
    else:
        term = "abort_zone_limit"

    Y = np.array(pts)
    certs["angle_energy"] = angle_energy(m, Y)
    certs["endpoint"] = tuple(Y[-1])
    if certs["seam_residuals"]:
        certs["seam_residual"] = max(certs["seam_residuals"])
    diag["stiff_frac"] = float(np.mean(kap >= KAPPA_HI))
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
                      local_until: float | None = None) -> Branch:
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
                    ds=abs(b_local - b_saddle) / 4000.0)
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

    certs = {"angle_energy": angle_energy(m, Y), "endpoint": tuple(Y[-1])}
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
                 delta: float = 1e-4) -> Branch:
    """Stable branch (separatrix): saddle → box exit, ascent flow.

    Fast-graph launch from the exact eigenvector jet; the continuation
    engine handles any folds back to the slow chart.
    """
    a_s = float(m.a_star(b_saddle))
    fr = saddle_frame(m, b_saddle, a_s)["stable"]
    d_w = fr["d_w"] if sign * fr["d_w"] >= 0 else -fr["d_w"]
    d_b = fr["d_b"] * (1 if sign * fr["d_w"] >= 0 else -1)
    if abs(d_w) < 1e-12:
        return Branch("stable", np.array([[a_s, b_saddle]]),
                      "abort_not_graph")
    if ds is None:
        ds = (abs(box[1] - box[0]) + abs(box[3] - box[2])) / 30000.0

    w0 = sign * delta
    b0 = b_saddle + (d_b / d_w) * w0

    pts, term, sw, _ = _continue_curve(m, b0, w0, -1, [], box, ds,
                                       ds0=abs(delta))
    Y = np.array([(a_s, b_saddle)] + pts)
    certs = {"angle_energy": angle_energy(m, Y), "endpoint": tuple(Y[-1])}
    return Branch("stable", Y, term, certs, {"switches": sw})
