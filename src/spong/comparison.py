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
    "stork2",
    "stork4",
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


def _stork2_b(j):
    """RKG2 coefficient used by the published STORK-2 recurrence."""
    if j == 0:
        return 1.0
    if j == 1:
        return 1.0/3.0
    return (4.0*(j-1)*(j+4)
            / (3.0*j*(j+1)*(j+2)*(j+3)))


def _stork2_step(m, y, h, time_direction, previous_velocity,
                 previous_h, stages=20):
    """One autonomous-flow STORK-2 macro step with first-order Taylor NFEs.

    STORK replaces the internal RKG2 model evaluations by a time-Taylor
    approximation from macro-step velocity history.  For an autonomous SPONG
    flow, this is the total velocity derivative along the computed orbit.  The
    first macro step is Forward Euler, matching the published startup.
    """
    velocity = _rhs(m, y, time_direction)
    if previous_velocity is None or previous_h is None:
        return y+h*velocity, velocity
    if stages < 2:
        raise ValueError("STORK-2 requires at least two stabilized stages")
    velocity_derivative = (velocity-previous_velocity)/previous_h
    base = np.asarray(y, dtype=float)
    y_jm2 = base.copy()
    y_jm1 = base.copy()
    denominator = stages*stages+stages-2
    for j in range(1, stages+1):
        if j == 1:
            mu_tilde = 6.0/((stages+4)*(stages-1))
            y_j = y_jm1+h*mu_tilde*velocity
        else:
            fraction = (4.0/(3.0*denominator) if j == 2 else
                        ((j-1)**2+(j-1)-2.0)/denominator)
            virtual_velocity = velocity+fraction*h*velocity_derivative
            bj = _stork2_b(j)
            mu = (2*j+1)*bj/(j*_stork2_b(j-1))
            nu = -(j+1)*bj/(j*_stork2_b(j-2))
            mu_tilde = mu*6.0/((stages+4)*(stages-1))
            gamma_tilde = -mu_tilde*(
                1.0-j*(j+1)*_stork2_b(j-1)/2.0)
            y_j = (mu*y_jm1+nu*y_jm2+(1.0-mu-nu)*base
                   +h*mu_tilde*virtual_velocity+h*gamma_tilde*velocity)
        y_jm2, y_jm1 = y_jm1, y_j
    return y_jm1, velocity


_STORK4_COEFFICIENTS = {
    # Published ROCK4 tables used by the STORK reference implementation.
    # The recurrence degree is followed by the four-stage composition.
    9: (
        (0.01862250741526137, 0.02250861576538788,
         0.006766202022280821, 0.02586805882924489,
         0.0317004234603281, 0.02877953008086195,
         0.06376714023863055, 0.03131867762450776,
         0.09801783524123062, 0.0335586405620185,
         0.1322218594416999, 0.035606882312253,
         0.1658724905539629, 0.0377737167357727,
         0.2016177039517425, 0.04119698781191684,
         0.2539572842035754),
        (-0.148972804305139, 0.475296959544537,
         -0.245899260153289, 0.01272030101911,
         -0.0357508839759137, 0.481825981739298),
        (0.678662206121957, -0.263538563990534,
         -0.347971227180309, 0.621040683222628)),
    20: (
        (0.005278040811520425, 0.006402421198476983,
         0.007026749310824704, 0.007381444032562888,
         0.03309125386700502, 0.0082346855199241,
         0.06683990648107069, 0.00898122293954147,
         0.1030531116670419, 0.009637773923336658,
         0.1392177117208699, 0.01021841796444464,
         0.1741238287384007, 0.01073483180895483,
         0.2072226942408981, 0.01119665624742548,
         0.238310549894101, 0.01161186412382382,
         0.2673653866792523, 0.01198709958412187,
         0.294460708386312, 0.01232800940556657,
         0.3197217653056301, 0.01263963481803662,
         0.3433101656487968, 0.01292700918588335,
         0.3654364911202174, 0.01319625157476125,
         0.3864151426812009, 0.01345671217793918,
         0.4067976665874104, 0.01372517143690529,
         0.4276578990766819, 0.0140336859283097,
         0.451161671749244, 0.01444284729184994,
         0.481626462022525, 0.01505946839641039,
         0.5272614695163532),
        (-0.126496410155342, 0.495458691659965,
         -0.269513232370108, -0.0103575442717928,
         -0.017638711552781, 0.479887174404288),
        (0.731832397913825, -0.309659464712225,
         -0.358881456195461, 0.614544900863432)),
}


def _stork4_step(m, y, h, time_direction, previous_velocity,
                 previous_h, stages=20):
    """Published STORK-4 recurrence with first-order Taylor virtual NFEs."""
    velocity = _rhs(m, y, time_direction)
    if previous_velocity is None or previous_h is None:
        return y+h*velocity, velocity
    if stages not in _STORK4_COEFFICIENTS:
        raise ValueError(
            "comparison STORK-4 tables are available for 9 or 20 stages")
    recurrence, finishing_a, finishing_b = _STORK4_COEFFICIENTS[stages]
    velocity_derivative = (velocity-previous_velocity)/previous_h
    base = np.asarray(y, dtype=float)

    # Orthogonal-polynomial recurrence.  The released scheduler leaves Y_j
    # stale after constructing Y_1; the paper's Algorithm 2 explicitly uses
    # this Y_1, which is the mathematically consistent initialization here.
    y_jm2 = base.copy()
    y_jm1 = base+h*recurrence[0]*velocity
    c_jm2 = 0.0
    c_jm1 = h*recurrence[0]
    for j in range(2, stages+1):
        mu = recurrence[2*(j-2)+1]
        previous_weight = 1.0+recurrence[2*(j-2)+2]
        older_weight = -recurrence[2*(j-2)+2]
        virtual_velocity = velocity+c_jm1*velocity_derivative
        y_j = (h*mu*virtual_velocity+previous_weight*y_jm1
               +older_weight*y_jm2)
        c_j = h*mu+previous_weight*c_jm1+older_weight*c_jm2
        y_jm2, y_jm1 = y_jm1, y_j
        c_jm2, c_jm1 = c_jm1, c_j

    # Four-stage order-four composition, retaining Taylor virtual NFEs.
    v1 = velocity+c_jm1*velocity_derivative
    c2 = c_jm1+h*finishing_a[0]
    v2 = velocity+c2*velocity_derivative
    c3 = c_jm1+h*(finishing_a[1]+finishing_a[2])
    v3 = velocity+c3*velocity_derivative
    c4 = c_jm1+h*sum(finishing_a[3:6])
    v4 = velocity+c4*velocity_derivative
    result = y_jm1+h*(finishing_b[0]*v1+finishing_b[1]*v2
                      +finishing_b[2]*v3+finishing_b[3]*v4)
    return result, velocity


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
           step_size, max_steps, time_horizon, rtol, atol,
           capture_kinds=None, stork_stages=20):
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
    stork_previous_velocity = None
    stork_previous_h = None
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
        elif method in ("stork2", "stork4"):
            with np.errstate(over="ignore", invalid="ignore",
                             divide="ignore"):
                stepper = (_stork2_step if method == "stork2"
                           else _stork4_step)
                ynew, current_velocity = stepper(
                    m, y, trial_h, time_direction,
                    stork_previous_velocity, stork_previous_h,
                    stages=stork_stages)
            stork_previous_velocity = current_velocity
            stork_previous_h = trial_h
            accepted += 1
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
                if capture_kinds is None or q.kind in capture_kinds
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


def arc_forward_euler(m: Model, branches, box, stride: int = 32):
    """Controlled low-order replacement for the manifold proposer.

    The exact gradient flow and its unit-speed reparametrization have the
    same integral curves.  Starting at each supplied launch point, march

        y[n+1] = y[n] + sigma*h[n]*grad(L)(y[n])/|grad(L)(y[n])|

    by Forward Euler.  ``h[n]`` is the arc length of ``stride`` consecutive
    reference chords.  Thus launch points, trace box, and physical arc budget
    remain comparable while the production continuation method is removed.
    Terminal and local-graph certificates are deliberately not inherited.
    """
    if stride < 1:
        raise ValueError("Euler stride must be positive")
    result = []
    for branch in branches:
        reference = np.asarray(branch.Y, dtype=float)
        if len(reference) < 2:
            result.append(Branch(
                branch.kind, reference.copy(), "max_steps",
                certs={}, diag={**branch.diag, "uncertified": True,
                                "arc_forward_euler_stride": stride,
                                "critical_steps": 0}))
            continue
        initial_gradient = np.asarray(
            m.gradL(float(reference[0, 0]), float(reference[0, 1])),
            dtype=float)
        initial_chord = reference[1]-reference[0]
        orientation = (1.0 if float(initial_chord@initial_gradient) >= 0.0
                       else -1.0)
        chord = np.hypot(
            np.diff(reference[:, 0]), np.diff(reference[:, 1]))
        points = [reference[0].copy()]
        term = "arc_budget"
        for k in range(0, len(chord), stride):
            step = float(np.sum(chord[k:k+stride]))
            current = points[-1]
            gradient = np.asarray(
                m.gradL(float(current[0]), float(current[1])), dtype=float)
            norm = float(np.hypot(gradient[0], gradient[1]))
            if (not math.isfinite(norm) or norm == 0.0
                    or not math.isfinite(step)):
                term = "nonfinite"
                break
            candidate = current+orientation*step*gradient/norm
            if not np.all(np.isfinite(candidate)):
                term = "nonfinite"
                break
            points.append(candidate)
            if not (box[0] <= candidate[0] <= box[1]
                    and box[2] <= candidate[1] <= box[3]):
                term = "box_exit"
                break
        result.append(Branch(
            branch.kind, np.asarray(points), term, certs={},
            diag={**branch.diag, "uncertified": True,
                  "arc_forward_euler_stride": stride,
                  # Only the launch segment gets the common-saddle
                  # exclusion.  The reference local-graph certificate does
                  # not apply to the Euler continuation constructed here.
                  "critical_steps": min(1, len(points)-1)}))
    return result


def manifold_contact_diagnostics(
        m: Model, reference_enumeration, branches, box, *,
        threshold: float = 4.0, candidate_limit: int = 50000):
    """Apply production contact semantics to an uncertified branch proposal.

    The comparison methods inherit no terminal, local-graph, or endpoint
    certificates.  Consequently this is a diagnostic, not a route by which a
    textbook integrator can certify a portrait.  Resolved pair roots and
    transverse self-crossings are faults; contacts without adequate witnesses
    remain unresolved.
    """
    from . import order_sweep, topology

    if candidate_limit < 1:
        raise ValueError("candidate limit must be positive")
    scale = max(1.0, *(abs(float(value)) for value in box))
    predicate_tolerance = 128*np.finfo(float).eps*scale
    contacts, pair_limited = order_sweep.pair_contact_candidates(
        branches, reference_enumeration.points, box,
        predicate_tolerance, limit=candidate_limit)
    pair_result = order_sweep.classify_contacts(
        m, reference_enumeration, branches, contacts,
        predicate_tolerance, threshold=threshold)

    # Self-intersections require two parameter sheets of one trace and cannot
    # be represented by the pairwise single-valued loss profile.  Retain the
    # established chord predicate here, but separate transverse crossings
    # from ambiguous contacts in the report.
    remaining = max(0, candidate_limit-len(contacts))
    self_crosses = self_ambiguous = 0
    self_examples = []
    self_limited = False
    if not pair_limited and remaining:
        native = topology._native_contact_available()
        critical = np.asarray([
            (point.a, point.b) for point in reference_enumeration.points],
            dtype=float)
        allowed_radius = max(1024*np.finfo(float).eps*scale, 1e-11)
        for branch_index, branch in enumerate(branches):
            Y = np.asarray(branch.Y, dtype=float)
            if len(Y) < 2:
                continue
            root = None if native else topology._tree(Y)
            sagitta = topology._sagitta_bounds(Y)
            for si, sj, kind, point in topology._self_contact_events(
                    Y, root, predicate_tolerance, sagitta):
                location = np.asarray(point, dtype=float)
                if len(critical) and np.min(topology._row_norm2(
                        critical-location)) <= allowed_radius:
                    continue
                if kind == "cross":
                    self_crosses += 1
                else:
                    self_ambiguous += 1
                if len(self_examples) < 32:
                    self_examples.append({
                        "branch": branch_index,
                        "segments": (int(si), int(sj)),
                        "kind": kind,
                        "point": tuple(float(value) for value in point),
                    })
                remaining -= 1
                if remaining == 0:
                    self_limited = True
                    break
            if self_limited:
                break

    complete = not pair_limited and not self_limited
    decision = (
        "fault" if pair_result["roots"] or self_crosses else
        "unresolved" if (not complete
                         or pair_result["decision"] == "unresolved"
                         or self_ambiguous) else
        "accepted")
    compact_pairs = []
    for pair in pair_result["pairs"]:
        compact_pairs.append({
            key: pair.get(key) for key in (
                "branches", "candidate_count", "root_count", "roots",
                "same_order_count", "terminal_count",
                "critical_transition_count", "unresolved_count",
                "contact_cluster_count", "clusters", "profile_levels",
                "first_dropped_nonmonotone",
                "second_dropped_nonmonotone", "profile_valid_fraction",
                "minimum_transversality")
        })
    return {
        "method": "loss_level_order_sweep",
        "decision": decision,
        "complete": complete,
        "threshold": float(threshold),
        "predicate_tolerance": float(predicate_tolerance),
        "candidate_limit": int(candidate_limit),
        "limit_hit": not complete,
        "pair_order_sweep": {
            key: pair_result[key] for key in (
                "decision", "candidates", "roots", "same_order",
                "terminal", "critical_transition", "unresolved")
        } | {"pairs": compact_pairs},
        "self_contacts": {
            "crosses": self_crosses,
            "ambiguous": self_ambiguous,
            "examples": self_examples,
        },
    }


def casual_portrait(
        m: Model, method: str, *, critical_method: str = "certified",
        reference_enumeration=None, view=None, step_size: float = 0.01,
        max_steps: int = 20000, time_horizon: float | None = None,
        rtol: float = 1e-3, atol: float = 1e-6, critical_grid: int = 17,
        capture_saddles: bool = True, stork_stages: int = 20):
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
                    rtol, atol,
                    # A naive finite-radius saddle capture can make a
                    # structurally unstable wall look accidentally exact.
                    # Comparison demos may instead continue unstable traces
                    # until a minimum so truncation and launch errors reveal
                    # which chamber the numerical pseudo-orbit selected.
                    ({"min"} if kind == "unstable" and not capture_saddles
                     else None), stork_stages=stork_stages)
                for key in totals:
                    totals[key] += diag[key]
                Y = np.asarray(points)
                geometry = integral_curve_diagnostics(
                    m, Y, spacing=span/5000.0)
                branches.append(Branch(
                    kind, Y, term,
                    certs=geometry, diag={
                        **diag, "saddle_b": saddle.b,
                        f"{kind}_direction": (
                            1 if start[1] > saddle.b else -1),
                        "launch_eigenvector_sign": sign,
                        "uncertified": True,
                    }))
    ledger = {
        "comparison": {
            "uncertified": True,
            "geometry_method": method,
            "critical_method": critical_method,
            "capture_saddles": capture_saddles,
            "step_size": step_size,
            "time_horizon": time_horizon,
            "rtol": rtol,
            "atol": atol,
            "stork_stages": (
                stork_stages if method in ("stork2", "stork4") else None),
            "critical_diagnostics": critical_diag,
            "integration_totals": totals,
        },
        "summary": {},
    }
    return Portrait(
        m, enumeration, branches, display_box,
        view if view is not None else display_box, ledger)
