"""Numerical oracle for the Dereich--Jentzen--Kassing Adam vector field.

At every fixed parameter point, independent minibatch innovations drive the
stationary recursions

    m <- alpha*m + (1-alpha)*X
    v <- beta*v + (1-beta)*X^2,

and the limiting field is E[m/(sqrt(v)+eps)].  A common innovation history is
used at every grid point, producing a smooth random-function approximation
rather than unrelated Monte Carlo noise from pixel to pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def _polyval(coefficients, x):
    value = np.zeros_like(x, dtype=float)
    for coefficient in reversed(coefficients):
        value = value*x+float(coefficient)
    return value


def empirical_uniform_grid(n):
    """Exact midpoint empirical distribution on [0,1]."""
    if n <= 0:
        raise ValueError("sample size must be positive")
    rationals = tuple(Fraction(2*i+1, 2*n) for i in range(n))
    return np.asarray(list(map(float, rationals))), rationals


def empirical_moments(points, count):
    """Exact moments of rational empirical support points."""
    n = len(points)
    return tuple(
        sum((point**degree for point in points), Fraction(0))/n
        for degree in range(count))


def negative_sample_gradient(f, g, a, b, x):
    """Per-example negative gradient of SPONG's unhalved square loss."""
    gp = [degree*float(g[degree]) for degree in range(1, len(g))]
    bx = b[..., None]*x
    gv = _polyval(g, bx)
    residual = _polyval(f, x)-a[..., None]*gv
    xa = 2.0*residual*gv
    xb = 2.0*residual*a[..., None]*_polyval(gp, bx)*x
    return np.stack((xa, xb), axis=-1)


@dataclass
class AdamFieldEstimate:
    a: np.ndarray
    b: np.ndarray
    field: np.ndarray
    chains: tuple[np.ndarray, ...]
    diagnostics: dict


def estimate_grid(
        f, g, inputs, view, *, grid=35, batch_size=32,
        alpha=0.9, beta=0.999, epsilon=1e-8,
        burn_in=8000, samples=4000, chains=2, seed=314159):
    """Estimate the autonomous Adam field on a regular parameter grid."""
    if not (0 <= alpha < np.sqrt(beta) < 1):
        raise ValueError("need 0 <= alpha < sqrt(beta) < 1")
    if grid < 3 or batch_size <= 0 or burn_in < 0 or samples <= 0:
        raise ValueError("invalid grid, batch, burn-in, or sample count")
    if chains < 1:
        raise ValueError("at least one chain is required")
    a_axis = np.linspace(view[0], view[1], grid)
    b_axis = np.linspace(view[2], view[3], grid)
    aa, bb = np.meshgrid(a_axis, b_axis)
    flat_a, flat_b = aa.ravel(), bb.ravel()
    n_points = flat_a.size
    chain_fields = []
    for chain in range(chains):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, chain, grid, batch_size]))
        first = np.zeros((n_points, 2))
        second = np.zeros((n_points, 2))
        for step in range(burn_in+samples):
            x = inputs[rng.integers(0, len(inputs), size=batch_size)]
            innovation = np.mean(
                negative_sample_gradient(
                    f, g, flat_a, flat_b, x),
                axis=1)
            first = alpha*first+(1-alpha)*innovation
            second = beta*second+(1-beta)*innovation*innovation
            if step == burn_in:
                accumulated = np.zeros_like(first)
            if step >= burn_in:
                accumulated += first/(np.sqrt(second)+epsilon)
        chain_fields.append(
            (accumulated/samples).reshape(grid, grid, 2))

    field = np.mean(chain_fields, axis=0)
    if chains >= 2:
        delta = np.linalg.norm(chain_fields[0]-chain_fields[1], axis=2)
        speed = np.linalg.norm(field, axis=2)
        resolved = speed > max(1e-12, 0.02*float(np.max(speed)))
        relative = delta[resolved]/speed[resolved] if np.any(resolved) else delta
        dot = np.sum(chain_fields[0]*chain_fields[1], axis=2)
        denom = (np.linalg.norm(chain_fields[0], axis=2)
                 * np.linalg.norm(chain_fields[1], axis=2))
        cosine = np.clip(dot[resolved]/np.maximum(denom[resolved], 1e-300),
                         -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        diagnostics = {
            "chain_rms_difference": float(np.sqrt(np.mean(delta*delta))),
            "resolved_grid_fraction": float(np.mean(resolved)),
            "resolved_relative_difference_median": (
                float(np.median(relative)) if relative.size else None),
            "resolved_angle_degrees_median": (
                float(np.median(angles)) if angles.size else None),
            "resolved_angle_degrees_95pct": (
                float(np.quantile(angles, .95)) if angles.size else None),
        }
    else:
        diagnostics = {"chain_rms_difference": None}
    diagnostics.update({
        "alpha": alpha,
        "beta": beta,
        "epsilon": epsilon,
        "burn_in": burn_in,
        "samples": samples,
        "chains": chains,
        "batch_size": batch_size,
        "grid": grid,
        "beta_burn_residual": float(beta**burn_in),
    })
    return AdamFieldEstimate(
        a_axis, b_axis, field, tuple(chain_fields), diagnostics)


def sign_change_zero_candidates(estimate):
    """Bilinear roots in sign-changing cells, clustered across cell edges.

    A componentwise range test alone produces long false bands when the two
    components cross zero at different places in the same coarse cell.  Solve
    the two bilinear equations and retain only roots lying inside their cell.
    """
    candidates = []
    field_scale = max(float(np.max(np.linalg.norm(
        estimate.field, axis=2))), 1e-300)
    for j in range(len(estimate.b)-1):
        for i in range(len(estimate.a)-1):
            corners = np.asarray((
                estimate.field[j, i],
                estimate.field[j, i+1],
                estimate.field[j+1, i],
                estimate.field[j+1, i+1]))
            if not all(np.min(corners[:, k]) <= 0 <= np.max(corners[:, k])
                       for k in (0, 1)):
                continue
            f00, f10, f01, f11 = corners
            linear_t = f10-f00
            linear_s = f01-f00
            mixed = f11-f10-f01+f00
            t = s = .5
            accepted = False
            for _ in range(20):
                value = f00+t*linear_t+s*linear_s+t*s*mixed
                dt = linear_t+s*mixed
                ds = linear_s+t*mixed
                determinant = dt[0]*ds[1]-dt[1]*ds[0]
                if abs(determinant) <= 1e-14*field_scale*field_scale:
                    break
                # Exact 2-by-2 solve J delta = -value.
                delta_t = (-value[0]*ds[1]+value[1]*ds[0])/determinant
                delta_s = (-dt[0]*value[1]+dt[1]*value[0])/determinant
                t += delta_t
                s += delta_s
                if max(abs(delta_t), abs(delta_s)) < 1e-11:
                    accepted = True
                    break
            residual = np.linalg.norm(
                f00+t*linear_t+s*linear_s+t*s*mixed)
            if (accepted and -1e-8 <= t <= 1+1e-8
                    and -1e-8 <= s <= 1+1e-8
                    and residual <= 1e-8*field_scale):
                candidates.append(np.array((
                    estimate.a[i]+t*(estimate.a[i+1]-estimate.a[i]),
                    estimate.b[j]+s*(estimate.b[j+1]-estimate.b[j]))))
    if not candidates:
        return np.empty((0, 2))
    span = np.array((
        estimate.a[-1]-estimate.a[0],
        estimate.b[-1]-estimate.b[0]))
    cell = np.array((
        (estimate.a[1]-estimate.a[0])/span[0],
        (estimate.b[1]-estimate.b[0])/span[1]))
    clustered = []
    for candidate in candidates:
        match = next((
            group for group in clustered
            if np.linalg.norm((candidate-np.mean(group, axis=0))/span)
            <= .6*np.linalg.norm(cell)), None)
        if match is None:
            clustered.append([candidate])
        else:
            match.append(candidate)
    return np.asarray([np.mean(group, axis=0) for group in clustered])


def _interpolate(estimate, point):
    a, b = point
    if not (estimate.a[0] <= a <= estimate.a[-1]
            and estimate.b[0] <= b <= estimate.b[-1]):
        return None
    i = min(np.searchsorted(estimate.a, a)-1, len(estimate.a)-2)
    j = min(np.searchsorted(estimate.b, b)-1, len(estimate.b)-2)
    i, j = max(i, 0), max(j, 0)
    ta = (a-estimate.a[i])/(estimate.a[i+1]-estimate.a[i])
    tb = (b-estimate.b[j])/(estimate.b[j+1]-estimate.b[j])
    f = estimate.field
    return ((1-ta)*(1-tb)*f[j, i]+ta*(1-tb)*f[j, i+1]
            +(1-ta)*tb*f[j+1, i]+ta*tb*f[j+1, i+1])


def streamlines(estimate, n_a=15, n_b=12, step=0.004, max_steps=1200):
    """Trace the geometry of the interpolated field in box-scaled coordinates."""
    span = np.array([
        estimate.a[-1]-estimate.a[0],
        estimate.b[-1]-estimate.b[0]])
    seeds_a = np.linspace(estimate.a[0], estimate.a[-1], n_a+2)[1:-1]
    seeds_b = np.linspace(estimate.b[0], estimate.b[-1], n_b+2)[1:-1]

    def direction(point, sign):
        value = _interpolate(estimate, point)
        if value is None:
            return None
        scaled = value/span
        norm = float(np.hypot(*scaled))
        if norm < 1e-10:
            return None
        return sign*scaled/norm*span

    def half(seed, sign):
        points = [np.asarray(seed, dtype=float)]
        for _ in range(max_steps):
            p = points[-1]
            k1 = direction(p, sign)
            if k1 is None:
                break
            k2 = direction(p+.5*step*k1, sign)
            k3 = direction(p+.5*step*(k2 if k2 is not None else k1), sign)
            k4 = direction(p+step*(k3 if k3 is not None else k1), sign)
            if k2 is None or k3 is None or k4 is None:
                break
            q = p+(step/6)*(k1+2*k2+2*k3+k4)
            if _interpolate(estimate, q) is None:
                break
            points.append(q)
        return np.asarray(points)

    curves = []
    for b in seeds_b:
        for a in seeds_a:
            seed = np.array([a, b])
            backward = half(seed, -1)
            forward = half(seed, 1)
            curve = np.vstack((backward[:0:-1], forward))
            if len(curve) >= 3:
                curves.append(curve)
    return curves
