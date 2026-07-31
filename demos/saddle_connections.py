"""Oracle search utilities for exceptional SPONG saddle connections.

An exact connection p -> q is one unstable half-manifold of p coinciding
with one stable half-manifold of q.  It is a codimension-one event, so random
portraits are used only as starts.  SPSA then moves the f/g coefficients to
reduce a coarse, normalized stable--unstable separation.  Once a pair is
close, a signed level-section shooting residual should replace this discovery
objective for the final bracket and certificate.

This module intentionally remains in ``demos``: the production portrait is
the oracle being consumed, not modified by the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from spong import model, portrait, sturm


@dataclass(frozen=True)
class NearPair:
    distance: float
    normalized_distance: float
    source_b: float
    target_b: float
    source_level: float
    target_level: float
    unstable_branch: int
    stable_branch: int
    unstable_vertex: int
    stable_vertex: int
    unstable_direction: int | None = None
    stable_sign: int | None = None
    section_level: float | None = None
    shooting_mismatch: float | None = None
    normalized_shooting: float | None = None


@dataclass(frozen=True)
class Evaluation:
    score: float
    valid: bool
    reason: str
    pair: NearPair | None
    n_saddles: int
    n_minima: int
    topology_status: str | None
    objective_kind: str = "nearest"


def quantized_unit_coefficients(values, bits=20):
    """Remove common scaling and return bounded-denominator dyadics.

    Multiplying f and g together merely rescales the loss and time.  Without
    normalization an optimizer can exploit that null direction, while raw
    FP64 updates make exact Sturm inputs needlessly enormous.
    """
    z = np.asarray(values, dtype=float)
    scale = float(np.hypot.reduce(z)) if len(z) else 0.0
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("coefficient vector must be finite and nonzero")
    denominator = 1 << int(bits)
    q = np.rint(z / scale * denominator).astype(np.int64)
    if not np.any(q):
        q[int(np.argmax(np.abs(z)))] = 1
    return tuple(Fraction(int(x), denominator) for x in q)


def affine_coefficients(left, right, parameter):
    """Exact point on a continuous rational coefficient segment.

    Final connection bracketing must not requantize each trial point:
    independently rounded probes do not define a continuous parameter
    family, so an opposite pair of shooting signs would not support an
    intermediate-value argument.
    """
    if len(left) != len(right):
        raise ValueError("coefficient vectors must have equal length")
    t = (parameter if isinstance(parameter, Fraction)
         else Fraction(parameter))
    return tuple(x + t*(y-x) for x, y in zip(left, right))


def random_coefficients(rng, degree, bits=20):
    z = rng.normal(size=int(degree) + 1)
    # Do not permit an accidental degree drop after dyadic quantization.
    if abs(z[-1]) < 0.2:
        z[-1] = np.copysign(0.2, z[-1] if z[-1] else 1.0)
    return quantized_unit_coefficients(z, bits)


def central_strip_width(degree, constant=16.0):
    """The user's high-degree search scale: |b| ~ constant**(1/m)."""
    degree = int(degree)
    if degree <= 0 or constant <= 0:
        raise ValueError("positive degree and constant required")
    return float(constant) ** (1.0 / degree)


def _sample_indices(count, maximum):
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(
        np.linspace(0, count - 1, maximum)).astype(np.int64))


def point_cloud_nearness(first, second, maximum=768):
    """Blocked closest-vertex distance without a black-box spatial solver."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    ix = _sample_indices(len(x), maximum)
    iy = _sample_indices(len(y), maximum)
    xs, ys = x[ix], y[iy]
    best = np.inf
    best_i = best_j = -1
    block = 128
    for start in range(0, len(xs), block):
        xb = xs[start:start + block]
        delta = xb[:, None, :] - ys[None, :, :]
        d2 = delta[..., 0] * delta[..., 0] \
            + delta[..., 1] * delta[..., 1]
        flat = int(np.argmin(d2))
        i, j = np.unravel_index(flat, d2.shape)
        value = float(d2[i, j])
        if value < best:
            best = value
            best_i = int(ix[start + i])
            best_j = int(iy[j])
    return float(np.sqrt(best)), best_i, best_j


def level_crossing(m, Y, level):
    """First certified-direction polyline crossing of a regular loss level.

    The chord parameter is refined against the essentially exact polynomial
    loss rather than accepted from linear interpolation of endpoint values.
    """
    Y = np.asarray(Y, dtype=float)
    values = np.asarray([m.L(a, b) - level for a, b in Y], dtype=float)
    for i in range(len(Y) - 1):
        v0, v1 = values[i], values[i + 1]
        if v0 == 0.0:
            return Y[i].copy()
        if v0 * v1 > 0.0:
            continue
        lo, hi = 0.0, 1.0
        slo = -1 if v0 < 0 else 1
        chord = Y[i + 1] - Y[i]
        for _ in range(56):
            mid = 0.5 * (lo + hi)
            point = Y[i] + mid * chord
            value = float(m.L(point[0], point[1]) - level)
            if value == 0.0:
                return point
            if (-1 if value < 0 else 1) == slo:
                lo = mid
            else:
                hi = mid
        return Y[i] + 0.5 * (lo + hi) * chord
    return None


def shooting_residual(m, unstable_Y, stable_Y, source_level, target_level,
                      scale):
    """Signed separation on the midpoint regular level.

    At a saddle connection both branches cross the level at the same point.
    The tangent J grad(L) supplies a coordinate-free sign on the level curve.
    """
    if not source_level > target_level:
        return None
    level = 0.5 * (source_level + target_level)
    xu = level_crossing(m, unstable_Y, level)
    xs = level_crossing(m, stable_Y, level)
    if xu is None or xs is None:
        return None
    displacement = xs-xu
    distance = float(np.hypot(displacement[0], displacement[1]))
    # A tangent coordinate is a shooting coordinate only in one local chart
    # of one level-set component.  Without this gate, distant points on the
    # same loss level can have almost identical tangent projections and create
    # a false sign change.  The critical-separation scale keeps the chart local;
    # the normal-defect gate enforces the implicit-function geometry.
    if not np.isfinite(distance) or distance > float(scale):
        return None
    gradient = np.asarray(m.gradL(xu[0], xu[1]), dtype=float)
    norm = float(np.hypot(gradient[0], gradient[1]))
    if not np.isfinite(norm) or norm <= 1e-14:
        return None
    normal = gradient/norm
    normal_defect = abs(float(np.dot(displacement, normal)))
    if distance > 1e-12 and normal_defect > 0.5*distance:
        return None
    tangent = np.array([-gradient[1], gradient[0]]) / norm
    mismatch = float(np.dot(displacement, tangent))
    return level, mismatch, mismatch / max(float(scale), 1e-12)


def common_level_separation(m, unstable_Y, stable_Y, source_level,
                            target_level, fraction=0.5):
    """Physical separation where both branches meet one regular loss level.

    Raw closest-point distance is not a dynamical nearness measurement:
    two vertices can be spatially close while carrying very different loss,
    and therefore cannot converge to the same noncritical orbit point.
    """
    if not source_level > target_level or not 0.0 < fraction < 1.0:
        return None
    level = target_level + float(fraction) * (source_level-target_level)
    xu = level_crossing(m, unstable_Y, level)
    xs = level_crossing(m, stable_Y, level)
    if xu is None or xs is None:
        return None
    displacement = xs-xu
    distance = float(np.hypot(displacement[0], displacement[1]))
    if not np.isfinite(distance):
        return None
    return level, xu, xs, distance


def _trim_critical_neighborhood(Y, criticals, radius):
    """Drop vertices close to any finite critical point.

    Nearness at a shared saddle or terminal minimum is not evidence of an
    interior saddle connection.  Keep at least two points so callers can
    report an invalid candidate cleanly.
    """
    Y = np.asarray(Y, dtype=float)
    if len(Y) <= 2 or not criticals:
        return Y, np.arange(len(Y), dtype=np.int64)
    keep = np.ones(len(Y), dtype=bool)
    r2 = radius * radius
    for q in criticals:
        da, db = Y[:, 0] - q.a, Y[:, 1] - q.b
        keep &= da * da + db * db > r2
    index = np.flatnonzero(keep)
    return Y[index], index


def nearest_stable_unstable(p, strip_factor=3.0, maximum=768, tracked=None):
    """Coarse discovery objective over energy-feasible distinct saddles."""
    e = p.enumeration
    saddles = list(e.saddles)
    if len(saddles) < 2:
        return None
    degree = max(len(p.model.f), len(p.model.g)) - 1
    strip = strip_factor * central_strip_width(max(1, degree))
    if tracked is None:
        central = [s for s in saddles if abs(s.b) <= strip]
        if len(central) >= 2:
            saddles = central
    else:
        source = min(saddles, key=lambda s: abs(
            s.b - float(tracked["source_b"])))
        remaining = [s for s in saddles if s is not source]
        if not remaining:
            return None
        target = min(remaining, key=lambda s: abs(
            s.b - float(tracked["target_b"])))
        saddles = [source, target]

    criticals = list(e.points)
    separation = min(
        (float(np.hypot(x.a-y.a, x.b-y.b))
         for i, x in enumerate(criticals) for y in criticals[i + 1:]),
        default=1.0)
    trim_radius = max(1e-8, min(0.03 * separation, 0.01 * strip))
    scale = max(separation, 0.1 * strip, 1e-8)

    stable = [(i, br) for i, br in enumerate(p.branches)
              if br.kind == "stable" and len(br.Y) > 2]
    unstable = [(i, br) for i, br in enumerate(p.branches)
                if br.kind == "unstable" and len(br.Y) > 2]
    by_b = {round(s.b, 11): s for s in saddles}
    best = None
    for ui, ub in unstable:
        source = by_b.get(round(float(ub.diag.get("saddle_b", np.nan)), 11))
        if source is None:
            continue
        if tracked is not None:
            if abs(source.b-saddles[0].b) > 1e-8 * (1+abs(source.b)):
                continue
            direction = ub.diag.get("unstable_direction")
            wanted = tracked.get("unstable_direction")
            if wanted is not None and direction != wanted:
                continue
        source_level = float(p.model.u(source.b))
        U, u_index = _trim_critical_neighborhood(
            ub.Y, criticals, trim_radius)
        if len(U) < 2:
            continue
        for si, sb in stable:
            target = by_b.get(round(
                float(sb.diag.get("saddle_b", np.nan)), 11))
            if target is None or target is source:
                continue
            if tracked is not None:
                if abs(target.b-saddles[1].b) > 1e-8 * (1+abs(target.b)):
                    continue
                sign = sb.diag.get("stable_sign")
                wanted = tracked.get("stable_sign")
                if wanted is not None and sign != wanted:
                    continue
            target_level = float(p.model.u(target.b))
            # A forward descent connection can only run high -> low.
            if not source_level > target_level + 1e-10 * (
                    1.0 + abs(source_level) + abs(target_level)):
                continue
            S, s_index = _trim_critical_neighborhood(
                sb.Y, criticals, trim_radius)
            if len(S) < 2:
                continue
            section = common_level_separation(
                p.model, ub.Y, sb.Y, source_level, target_level)
            if section is None:
                continue
            _, xu, xs, distance = section
            # Retain representative vertex indices for diagnostics/rendering,
            # but do not use their unconstrained distance as the objective.
            iu = int(np.argmin(np.sum((U-xu)**2, axis=1)))
            js = int(np.argmin(np.sum((S-xs)**2, axis=1)))
            candidate = NearPair(
                distance=distance,
                normalized_distance=distance / scale,
                source_b=float(source.b),
                target_b=float(target.b),
                source_level=source_level,
                target_level=target_level,
                unstable_branch=ui,
                stable_branch=si,
                unstable_vertex=int(u_index[iu]),
                stable_vertex=int(s_index[js]),
                unstable_direction=ub.diag.get("unstable_direction"),
                stable_sign=sb.diag.get("stable_sign"))
            shot = shooting_residual(
                p.model, ub.Y, sb.Y, source_level, target_level, scale)
            if shot is not None:
                candidate = NearPair(
                    **{**candidate.__dict__,
                       "section_level": shot[0],
                       "shooting_mismatch": shot[1],
                       "normalized_shooting": shot[2]})
            if best is None or candidate.normalized_distance \
                    < best.normalized_distance:
                best = candidate
    return best


def evaluate(f, g, moment_dist="uniform01", geometry_level=0,
             strip_factor=3.0, maximum=768, shooting_switch=0.5,
             tracked=None):
    """Build and screen one exact-dyadic SPONG candidate."""
    degree = max(len(f), len(g)) - 1
    moments_fn = (model.moments_uniform01 if moment_dist == "uniform01"
                  else model.moments_normal01)
    try:
        m = model.build(f, g, moments_fn(2 * degree + 1))
        e0 = sturm.enumerate_critical_points(m)
        if not e0.psi_positive:
            return Evaluation(np.inf, False, "A_not_positive", None,
                              len(e0.saddles), len(e0.minima), None, "invalid")
        if not e0.morse:
            return Evaluation(np.inf, False, "non_morse", None,
                              len(e0.saddles), len(e0.minima), None, "invalid")
        if len(e0.saddles) < 2:
            return Evaluation(np.inf, False, "fewer_than_two_saddles", None,
                              len(e0.saddles), len(e0.minima), None, "invalid")
        levels = np.asarray([m.u(s.b) for s in e0.saddles], dtype=float)
        level_scale = max(1.0, float(np.max(np.abs(levels))))
        if float(np.max(levels) - np.min(levels)) \
                <= 1e-10 * level_scale:
            # Every simple B-root saddle has exactly the common level C.
            # With no lower N-root saddle, strict loss descent rules out a
            # saddle-to-saddle orbit before any manifold is constructed.
            return Evaluation(
                np.inf, False, "no_unequal_saddle_levels", None,
                len(e0.saddles), len(e0.minima), None, "invalid")
        # Consume the same certified critical charts/stubs as production.
        # Skipping this stage made a fast oracle, but could turn an otherwise
        # certified candidate into an artificial FP64 handoff failure.
        e = sturm.materialize_stubs(m, e0)
        p = portrait.compute(
            m, geometry_level=int(geometry_level), _enumeration=e)
        pair = nearest_stable_unstable(
            p, strip_factor=strip_factor, maximum=maximum, tracked=tracked)
        status = p.ledger.get("topology", {}).get("status")
        if pair is None:
            return Evaluation(np.inf, False, "no_energy_feasible_pair", None,
                              len(e0.saddles), len(e0.minima), status,
                              "invalid")
        use_shooting = (
            (tracked is not None or
             pair.normalized_distance <= shooting_switch)
            and pair.normalized_shooting is not None)
        score = ((pair.normalized_shooting**2) if use_shooting
                 else (pair.normalized_distance**2))
        return Evaluation(score, True, "ok", pair,
                          len(e0.saddles), len(e0.minima), status,
                          "shooting" if use_shooting else "nearest")
    except (ArithmeticError, FloatingPointError, OverflowError, ValueError,
            ZeroDivisionError) as exc:
        return Evaluation(np.inf, False, type(exc).__name__, None, 0, 0, None)


def perturb_spsa(f, g, rng, epsilon, bits=20):
    """Symmetric coefficient-space probe used by one bandit pull."""
    f0 = np.asarray([float(x) for x in f])
    g0 = np.asarray([float(x) for x in g])
    delta_f = rng.choice((-1.0, 1.0), size=len(f0))
    delta_g = rng.choice((-1.0, 1.0), size=len(g0))
    plus = (
        quantized_unit_coefficients(f0 + epsilon * delta_f, bits),
        quantized_unit_coefficients(g0 + epsilon * delta_g, bits))
    minus = (
        quantized_unit_coefficients(f0 - epsilon * delta_f, bits),
        quantized_unit_coefficients(g0 - epsilon * delta_g, bits))
    return plus, minus, delta_f, delta_g


def spsa_update(f, g, plus_score, minus_score, delta_f, delta_g,
                epsilon, learning_rate, bits=20, max_step=5e-3):
    difference = (float(plus_score) - float(minus_score)) / (2.0 * epsilon)
    coefficient_step = learning_rate * difference
    dimension = len(delta_f) + len(delta_g)
    step_norm = abs(coefficient_step) * np.sqrt(dimension)
    if max_step is not None and step_norm > max_step:
        coefficient_step *= max_step / step_norm
    # Since delta entries are ±1, division by delta equals multiplication.
    f1 = np.asarray([float(x) for x in f]) \
        - coefficient_step * delta_f
    g1 = np.asarray([float(x) for x in g]) \
        - coefficient_step * delta_g
    return (quantized_unit_coefficients(f1, bits),
            quantized_unit_coefficients(g1, bits))


def shooting_secant_update(f, g, current_mismatch, plus_mismatch,
                           minus_mismatch, delta_f, delta_g, epsilon,
                           bits=20, max_step=5e-3):
    """Trust-region secant step for the scalar shooting equation S=0."""
    derivative = (
        float(plus_mismatch) - float(minus_mismatch)) / (2.0 * epsilon)
    if not np.isfinite(derivative) or abs(derivative) < 1e-14:
        return None
    parameter_step = -float(current_mismatch) / derivative
    dimension = len(delta_f) + len(delta_g)
    norm = abs(parameter_step) * np.sqrt(dimension)
    if norm > max_step:
        parameter_step *= max_step / norm
    f1 = np.asarray([float(x) for x in f]) + parameter_step * delta_f
    g1 = np.asarray([float(x) for x in g]) + parameter_step * delta_g
    return (quantized_unit_coefficients(f1, bits),
            quantized_unit_coefficients(g1, bits))


def tracked_pair(pair):
    """Serializable identity of one continued saddle/branch pair."""
    return {
        "source_b": pair.source_b,
        "target_b": pair.target_b,
        "unstable_direction": pair.unstable_direction,
        "stable_sign": pair.stable_sign,
    }
