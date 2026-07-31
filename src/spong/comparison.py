"""Uncertified textbook baselines for comparison with SPONG's portraitist.

These algorithms are intentionally ordinary.  They make the familiar
"why not just use ...?" choices concrete without allowing them into the
certified production path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import atlas, sturm
from .charts import Branch
from .model import Model
from .portrait import Portrait


GEOMETRY_METHODS = (
    "forward-euler",
    "backward-euler",
    "explicit-midpoint",
    "implicit-midpoint",
    "rkf45",
    "ros2",
)


@dataclass(frozen=True)
class CasualCriticalPoint:
    a: float
    b: float
    kind: str
    source: str = "grid-newton"


@dataclass(frozen=True)
class CasualEnumeration:
    points: tuple[CasualCriticalPoint, ...]
    psi_positive: bool = False
    morse: bool = False
    alternates: bool = False

    @property
    def minima(self):
        return tuple(q for q in self.points if q.kind == "min")

    @property
    def saddles(self):
        return tuple(q for q in self.points if q.kind == "saddle")


def _solve2(A, b):
    a, c = float(A[0, 0]), float(A[0, 1])
    d, e = float(A[1, 0]), float(A[1, 1])
    det = a*e-c*d
    scale = max(abs(a), abs(c), abs(d), abs(e), 1.0)
    if not math.isfinite(det) or abs(det) <= 32*np.finfo(float).eps*scale**2:
        return None
    return np.array([(e*b[0]-c*b[1])/det,
                     (-d*b[0]+a*b[1])/det], dtype=float)


def _finite_difference_jacobian(m: Model, y):
    J = np.empty((2, 2), dtype=float)
    eps = math.sqrt(np.finfo(float).eps)
    for j in range(2):
        h = eps*(1.0+abs(float(y[j])))
        yp, ym = np.array(y, dtype=float), np.array(y, dtype=float)
        yp[j] += h
        ym[j] -= h
        J[:, j] = (m.gradL(*yp)-m.gradL(*ym))/(2*h)
    return J


def _classify_hessian(H):
    h11, h12, h22 = float(H[0, 0]), float(H[0, 1]), float(H[1, 1])
    trace = h11+h22
    disc = math.hypot(h11-h22, 2*h12)
    lo, hi = 0.5*(trace-disc), 0.5*(trace+disc)
    tol = 128*np.finfo(float).eps*max(abs(lo), abs(hi), 1.0)
    if lo > tol:
        return "min"
    if hi < -tol:
        return "max"
    if lo < -tol and hi > tol:
        return "saddle"
    return "degenerate"


def grid_newton_critical_points(
        m: Model, box, grid: int = 17, residual_tol: float = 1e-9,
        dedup_rel: float = 1e-6, max_iterations: int = 30):
    """Multistart finite-difference Newton, a typical casual equilibrium scan.

    No completeness claim is made: roots outside Newton basins are missed,
    close roots may be merged, and a nearly singular Jacobian may be refused.
    """
    a0, a1, b0, b1 = map(float, box)
    scale = max(a1-a0, b1-b0, 1.0)
    roots = []
    attempted = converged = singular = 0
    for a in np.linspace(a0, a1, grid):
        for b in np.linspace(b0, b1, grid):
            attempted += 1
            y = np.array([a, b], dtype=float)
            ok = False
            for _ in range(max_iterations):
                f = np.asarray(m.gradL(*y), dtype=float)
                fn = float(np.hypot(*f))
                if not np.all(np.isfinite(f)):
                    break
                if fn <= residual_tol:
                    ok = True
                    break
                J = _finite_difference_jacobian(m, y)
                step = _solve2(J, -f)
                if step is None:
                    singular += 1
                    break
                accepted = False
                damping = 1.0
                for _ in range(10):
                    candidate = y+damping*step
                    if not np.all(np.isfinite(candidate)):
                        damping *= 0.5
                        continue
                    candidate_norm = float(np.hypot(*m.gradL(*candidate)))
                    if candidate_norm < fn:
                        y = candidate
                        accepted = True
                        break
                    damping *= 0.5
                if not accepted:
                    break
            if not ok or not (a0 <= y[0] <= a1 and b0 <= y[1] <= b1):
                continue
            if any(np.hypot(*(y-q)) <= dedup_rel*(1+np.hypot(*y))
                   for q in roots):
                continue
            roots.append(y)
            converged += 1
    roots.sort(key=lambda y: (float(y[1]), float(y[0])))
    points = tuple(
        CasualCriticalPoint(
            float(y[0]), float(y[1]),
            _classify_hessian(_finite_difference_jacobian(m, y)))
        for y in roots
    )
    return CasualEnumeration(points), {
        "attempted_seeds": attempted,
        "distinct_converged": converged,
        "singular_iterations": singular,
        "grid": grid,
        "residual_tol": residual_tol,
        "dedup_rel": dedup_rel,
    }


def _rhs(m, y, time_direction):
    return -float(time_direction)*np.asarray(m.gradL(*y), dtype=float)


def _implicit_step(m, y, h, time_direction, midpoint):
    z = y+h*_rhs(m, y, time_direction)
    for _ in range(12):
        evaluation = 0.5*(y+z) if midpoint else z
        f = _rhs(m, evaluation, time_direction)
        residual = z-y-h*f
        if np.hypot(*residual) <= 1e-12*(1+np.hypot(*z)):
            return z, True
        H = np.asarray(m.hessL(*evaluation), dtype=float)
        factor = 0.5 if midpoint else 1.0
        jacobian = np.eye(2)+h*factor*float(time_direction)*H
        correction = _solve2(jacobian, -residual)
        if correction is None:
            return z, False
        z = z+correction
        if not np.all(np.isfinite(z)):
            return z, False
    return z, False


def _fixed_step(m, method, y, h, time_direction):
    f0 = _rhs(m, y, time_direction)
    if method == "forward-euler":
        return y+h*f0, True
    if method == "explicit-midpoint":
        return y+h*_rhs(m, y+0.5*h*f0, time_direction), True
    if method == "backward-euler":
        return _implicit_step(m, y, h, time_direction, midpoint=False)
    if method == "implicit-midpoint":
        return _implicit_step(m, y, h, time_direction, midpoint=True)
    raise ValueError(f"not a fixed-step method: {method}")


def _rkf45_trial(m, y, h, time_direction):
    """Classical Fehlberg 4(5) embedded pair."""
    f = lambda z: _rhs(m, z, time_direction)
    k1 = f(y)
    k2 = f(y+h*(k1/4))
    k3 = f(y+h*(3*k1/32+9*k2/32))
    k4 = f(y+h*(1932*k1/2197-7200*k2/2197+7296*k3/2197))
    k5 = f(y+h*(439*k1/216-8*k2+3680*k3/513-845*k4/4104))
    k6 = f(y+h*(-8*k1/27+2*k2-3544*k3/2565
                +1859*k4/4104-11*k5/40))
    fourth = y+h*(25*k1/216+1408*k3/2565
                  +2197*k4/4104-k5/5)
    fifth = y+h*(16*k1/135+6656*k3/12825
                 +28561*k4/56430-9*k5/50+2*k6/55)
    return fifth, fifth-fourth


def _ros2_trial(m, y, h, time_direction):
    """L-stable two-stage Rosenbrock method of order 2.

    This is an open textbook analogue of MATLAB's modified Rosenbrock
    order-2 ode23s algorithm, not a claim to reproduce proprietary details.
    """
    gamma = 1.0+1.0/math.sqrt(2.0)
    a21 = 1.0/gamma
    c21 = -2.0/gamma
    m1, m2 = 3.0/(2.0*gamma), 1.0/(2.0*gamma)
    f0 = _rhs(m, y, time_direction)
    H = np.asarray(m.hessL(*y), dtype=float)
    matrix = np.eye(2)/(gamma*h)+float(time_direction)*H
    k1 = _solve2(matrix, f0)
    if k1 is None:
        return y.copy(), np.full(2, np.inf), False
    f1 = _rhs(m, y+a21*k1, time_direction)
    k2 = _solve2(matrix, f1+(c21/h)*k1)
    if k2 is None:
        return y.copy(), np.full(2, np.inf), False
    second = y+m1*k1+m2*k2
    first = y+k1/gamma
    return second, second-first, True


def _symmetric_eigenvector(H, lower):
    h11, h12, h22 = float(H[0, 0]), float(H[0, 1]), float(H[1, 1])
    trace = h11+h22
    disc = math.hypot(h11-h22, 2*h12)
    eigenvalue = 0.5*(trace-disc if lower else trace+disc)
    if h12 == 0.0:
        vector = (np.array([1.0, 0.0])
                  if abs(eigenvalue-h11) <= abs(eigenvalue-h22)
                  else np.array([0.0, 1.0]))
    elif abs(h12) > abs(eigenvalue-h11):
        vector = np.array([h12, eigenvalue-h11], dtype=float)
    else:
        vector = np.array([eigenvalue-h22, h12], dtype=float)
    norm = float(np.hypot(*vector))
    if norm != 0:
        vector /= norm
    return eigenvalue, vector


def _outside(y, box):
    return not (box[0] <= y[0] <= box[1]
                and box[2] <= y[1] <= box[3])


def _trace(m, start, method, box, critical_points, origin, time_direction,
           step_size, max_steps, time_horizon, rtol, atol):
    y = np.asarray(start, dtype=float)
    points = [y.copy()]
    h = float(step_size)
    accepted = rejected = nonlinear_failures = 0
    elapsed_time = 0.0
    min_accepted_step = math.inf
    max_accepted_step = 0.0
    span = max(box[1]-box[0], box[3]-box[2], 1.0)
    capture = 2e-4*span
    display_resolution = span/5000.0
    term = "max_steps"
    for iteration in range(max_steps):
        remaining = time_horizon-elapsed_time
        if remaining <= 16*np.finfo(float).eps*max(1.0, time_horizon):
            term = "time_horizon"
            break
        trial_h = min(h, remaining)
        if method in ("rkf45", "ros2"):
            for _ in range(32):
                with np.errstate(over="ignore", invalid="ignore",
                                 divide="ignore"):
                    if method == "rkf45":
                        candidate, error = _rkf45_trial(
                            m, y, trial_h, time_direction)
                        solved = True
                        accept_power, reject_power = -0.2, -0.25
                    else:
                        candidate, error, solved = _ros2_trial(
                            m, y, trial_h, time_direction)
                        accept_power = reject_power = -0.5
                    scale = atol+rtol*np.maximum(
                        np.abs(y), np.abs(candidate))
                    err = float(np.max(np.abs(error)/scale))
                if solved and np.all(np.isfinite(candidate)) and err <= 1.0:
                    accepted += 1
                    factor = 5.0 if err == 0 else min(
                        5.0, max(0.2, 0.9*err**accept_power))
                    next_h = trial_h*factor
                    break
                rejected += 1
                trial_h *= max(0.1, 0.9*err**reject_power) \
                    if math.isfinite(err) and err > 0 else 0.1
                if trial_h < np.finfo(float).eps:
                    term = "step_underflow"
                    return points, term, {
                        "accepted": accepted, "rejected": rejected,
                        "nonlinear_failures": nonlinear_failures,
                        "elapsed_time": elapsed_time,
                        "min_accepted_step": (
                            None if math.isinf(min_accepted_step)
                            else min_accepted_step),
                        "max_accepted_step": max_accepted_step,
                    }
            else:
                term = "step_failure"
                break
            ynew = candidate
            h = next_h
            h_used = trial_h
        else:
            ynew, ok = _fixed_step(
                m, method, y, trial_h, time_direction)
            if not ok:
                nonlinear_failures += 1
                term = "nonlinear_failure"
                break
            accepted += 1
            h_used = trial_h
        elapsed_time += h_used
        min_accepted_step = min(min_accepted_step, h_used)
        max_accepted_step = max(max_accepted_step, h_used)
        if not np.all(np.isfinite(ynew)):
            term = "nonfinite"
            break
        y = ynew
        if np.hypot(*(y-points[-1])) >= display_resolution:
            points.append(y.copy())
        if _outside(y, box):
            term = "box_exit"
            break
        if iteration >= 3:
            distances = [
                np.hypot(y[0]-q.a, y[1]-q.b) for q in critical_points
                if np.hypot(q.a-origin[0], q.b-origin[1]) > capture]
            if distances and min(distances) < capture:
                term = "capture"
                break
        if elapsed_time >= time_horizon:
            term = "time_horizon"
            break
    if np.hypot(*(y-points[-1])) > 0:
        points.append(y.copy())
    if (term == "max_steps"
            and time_horizon-elapsed_time
            <= 1e-12*max(1.0, time_horizon)):
        term = "time_horizon"
    return points, term, {
        "accepted": accepted, "rejected": rejected,
        "nonlinear_failures": nonlinear_failures,
        "elapsed_time": elapsed_time,
        "min_accepted_step": (
            None if math.isinf(min_accepted_step) else min_accepted_step),
        "max_accepted_step": max_accepted_step,
    }


def _resample_polyline(Y, spacing):
    Y = np.asarray(Y, dtype=float)
    if len(Y) < 2:
        return Y
    lengths = np.hypot(*(np.diff(Y, axis=0).T))
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if not math.isfinite(total) or total == 0:
        return Y[:1]
    count = min(20000, max(2, int(math.ceil(total/spacing))+1))
    samples = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(samples, cumulative, Y[:, 0]),
        np.interp(samples, cumulative, Y[:, 1]),
    ))


def integral_curve_diagnostics(m: Model, Y, spacing):
    """Common-resolution tangent/gradient alignment diagnostics."""
    Y = _resample_polyline(Y, spacing)
    with np.errstate(over="ignore", invalid="ignore"):
        gradient_norms = np.asarray([
            np.hypot(*np.asarray(m.gradL(*point), dtype=float))
            for point in Y
        ])
    finite_gradient_norms = gradient_norms[np.isfinite(gradient_norms)]
    relative_gradient_floor = (
        1e-3*float(np.max(finite_gradient_norms))
        if len(finite_gradient_norms) else math.inf)
    energy = sum_perp2 = sum_chord2 = 0.0
    used = skipped = 0
    max_angle = 0.0
    eps = np.finfo(float).eps
    for k in range(1, len(Y)-1):
        a, b = Y[k]
        d = Y[k+1]-Y[k-1]
        with np.errstate(over="ignore", invalid="ignore"):
            g = np.asarray(m.gradL(a, b), dtype=float)
            ng = float(np.hypot(*g))
            scale_a = 2.0*(abs(a)*m.A(b)+abs(m.B(b)))
            scale_b = 2.0*abs(a)*abs(m.Bp(b))+a*a*abs(m.Ap(b))
            floor = 16.0*eps*float(np.hypot(scale_a, scale_b))
        nd = float(np.hypot(*d))
        if (not math.isfinite(ng) or not math.isfinite(floor)
                or ng < max(1e-12, relative_gradient_floor, 1e3*floor)
                or nd < 1e-14):
            skipped += 1
            continue
        gh = g/ng
        perpendicular = d-(gh@d)*gh
        perpendicular2 = float(perpendicular@perpendicular)
        chord2 = float(d@d)
        energy += 0.5*perpendicular2
        sum_perp2 += perpendicular2
        sum_chord2 += chord2
        sine = min(1.0, math.sqrt(perpendicular2/chord2))
        max_angle = max(max_angle, math.degrees(math.asin(sine)))
        used += 1
    rms_sine = math.sqrt(sum_perp2/sum_chord2) if sum_chord2 else None
    return {
        "angle_energy_common": energy,
        "angle_rms_deg": (
            math.degrees(math.asin(min(1.0, rms_sine)))
            if rms_sine is not None else None),
        "angle_max_deg": max_angle if used else None,
        "angle_resolved": used,
        "angle_unresolved": skipped,
        "resampled_points": len(Y),
    }


def casual_portrait(
        m: Model, method: str, *, critical_method: str = "certified",
        reference_enumeration=None, view=None, step_size: float = 0.01,
        max_steps: int = 20000, time_horizon: float | None = None,
        rtol: float = 1e-3, atol: float = 1e-6, critical_grid: int = 17):
    """Trace saddle manifolds with an explicitly uncertified baseline."""
    if method not in GEOMETRY_METHODS:
        raise ValueError(f"unknown comparison geometry method {method!r}")
    certified = (reference_enumeration if reference_enumeration is not None
                 else sturm.enumerate_critical_points(m))
    display_box = atlas.compute_box(m, certified, view=view)
    if critical_method == "certified":
        enumeration = certified
        critical_diag = {"method": "certified-reference"}
    elif critical_method == "grid-newton":
        enumeration, critical_diag = grid_newton_critical_points(
            m, display_box, grid=critical_grid)
    else:
        raise ValueError(f"unknown critical method {critical_method!r}")

    span = max(display_box[1]-display_box[0],
               display_box[3]-display_box[2], 1.0)
    if time_horizon is None:
        time_horizon = step_size*max_steps
    launch = 1e-4*span
    branches = []
    totals = {"accepted": 0, "rejected": 0, "nonlinear_failures": 0}
    for saddle in enumeration.saddles:
        H = np.asarray(m.hessL(saddle.a, saddle.b), dtype=float)
        _, unstable_vector = _symmetric_eigenvector(H, lower=True)
        _, stable_vector = _symmetric_eigenvector(H, lower=False)
        for kind, vector, time_direction in (
                ("unstable", unstable_vector, +1),
                ("stable", stable_vector, -1)):
            for sign in (-1, +1):
                start = np.array([saddle.a, saddle.b])+sign*launch*vector
                points, term, diag = _trace(
                    m, start, method, display_box, enumeration.points,
                    (saddle.a, saddle.b),
                    time_direction, step_size, max_steps, time_horizon,
                    rtol, atol)
                for key in totals:
                    totals[key] += diag[key]
                Y = np.asarray(points)
                geometry = integral_curve_diagnostics(
                    m, Y, spacing=span/5000.0)
                branches.append(Branch(
                    kind, Y, term,
                    certs=geometry, diag={
                        **diag, "saddle_b": saddle.b,
                        "uncertified": True,
                    }))
    ledger = {
        "comparison": {
            "uncertified": True,
            "geometry_method": method,
            "critical_method": critical_method,
            "step_size": step_size,
            "time_horizon": time_horizon,
            "rtol": rtol,
            "atol": atol,
            "critical_diagnostics": critical_diag,
            "integration_totals": totals,
        },
        "summary": {},
    }
    return Portrait(
        m, enumeration, branches, display_box,
        view if view is not None else display_box, ledger)
