"""Independent scrutiny of a proposed trajectory object.

No tracer imports and no solver calls. The exact polynomial model is the
authority; constructor residuals and stored tangents are not trusted.
This is an evidence/coverage check, NOT a portrait certification gate.
"""
from fractions import Fraction as Q

import numpy as np


KNOWN_PIECES = frozenset({
    "precision_bridge",
    "manifold_jet", "poincare_grid", "centered_grid",
    "centered_grid_extension", "centered_launch", "linear_launch",
    "backbone_launch", "potential_prefix", "potential_level_event",
    "potential_ascent", "centered_arrival", "hadamard_graph",
    "chart_continuation", "continuation_after_shallow_rejection",
    "handoff", "capture_connector", "minimum_arrival",
})


def inventory(points, pieces):
    """Refuse unknown kinds, holes, overlaps, and out-of-range ownership."""
    try:
        Y = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "invalid_points"}
    if Y.ndim != 2 or Y.shape[1] != 2 or not len(Y):
        return {"status": "invalid", "reason": "invalid_points"}
    if not np.all(np.isfinite(Y)):
        return {"status": "invalid", "reason": "nonfinite_points"}
    cursor = 0
    if pieces is None:
        return {"status": "invalid", "reason": "missing_inventory"}
    for i, piece in enumerate(pieces):
        if getattr(piece, "kind", None) not in KNOWN_PIECES:
            return {"status": "unresolved", "reason": "unknown_piece", "piece": i}
        if not isinstance(piece.evidence, dict):
            return {"status": "invalid", "reason": "invalid_evidence", "piece": i}
        if (type(piece.start) is not int or type(piece.stop) is not int
                or piece.start != cursor or not cursor < piece.stop < len(Y)):
            return {"status": "invalid", "reason": "invalid_partition", "piece": i}
        cursor = piece.stop
    if cursor != len(Y)-1:
        return {"status": "invalid", "reason": "uncovered_chords"}
    return {"status": "complete", "chords": cursor, "pieces": len(pieces),
            "repeated_chords": int(np.sum(np.all(Y[1:] == Y[:-1], axis=1)))}


def _eval(coefficients, x):
    value = Q(0)
    for c in reversed(coefficients):
        value = value*x+c
    return value


def _derivative(coefficients):
    return tuple(i*c for i, c in enumerate(coefficients) if i)


def _horner_bound(coefficients, x):
    # Standard binary64 Horner forward-error bound, plus an absolute allowance
    # for subnormal operations. Used only to bind the sampled display to the
    # supplied polynomial, never as an invariance or coefficient-error bound.
    operations = 2*len(coefficients)
    unit = Q(1, 2**53)
    gamma = operations*unit/(1-operations*unit)
    magnitude = _eval(tuple(abs(c) for c in coefficients), abs(x))
    # Each underflow rounding error can be amplified by remaining powers of x.
    underflow = Q(1, 2**1074)*sum(
        (abs(x)**k for k in range(len(coefficients))), Q(0))*2
    return gamma*magnitude+underflow/(1-operations*unit)


def measure_jet(m, points, evidence, direction):
    """Re-derive smooth K and K' over Q; measure vertices AND midparameters.

    The supplied floating coefficients define the proposed polynomial. This
    does not prove their recurrence or convergence to an invariant manifold.
    Evaluating center + K in Q retains offsets lost in global coordinates.
    """
    try:
        if direction not in (-1, 1) or isinstance(direction, bool):
            raise ValueError("invalid flow direction")
        if evidence.get("representation") != "centered_polynomial":
            raise ValueError("missing polynomial")
        center = tuple(Q(float(v)) for v in evidence["center"])
        coefficients = tuple(tuple(Q(float(v)) for v in c)
                             for c in evidence["coefficients"])
        parameters = tuple(Q(float(v)) for v in evidence["parameters"])
        if (len(center) != 2 or len(coefficients) != 2
                or not all(coefficients) or len(parameters) < len(points)
                or parameters[0] != 0
                or any(b < a for a, b in zip(parameters, parameters[1:]))):
            raise ValueError("invalid polynomial data")
        parameters = parameters[:len(points)]
        if not parameters or parameters[-1] <= parameters[0]:
            raise ValueError("empty parameter interval")
        for point, t in zip(points, parameters):
            for j in (0, 1):
                value = center[j]+_eval(coefficients[j], t)
                bound = _horner_bound(coefficients[j], t)
                bound += (abs(value)+bound)*Q(1, 2**53)+Q(1, 2**1074)
                if abs(Q(float(point[j]))-value) > bound:
                    return {"status": "invalid", "reason": "geometry_mismatch"}
        A, B = tuple(map(Q, m.alpha)), tuple(map(Q, m.beta))
        Ap, Bp = _derivative(A), _derivative(B)
        derivatives = tuple(_derivative(c) for c in coefficients)
        samples = sorted(set(parameters) | {
            (a+b)/2 for a, b in zip(parameters, parameters[1:]) if b > a})
        worst = Q(0)
        opposed = stationary = zero_tangent = orthogonal = 0
        anchor = None
        for t in samples:
            a, b = (center[j]+_eval(coefficients[j], t) for j in (0, 1))
            da, db = (_eval(c, t) for c in derivatives)
            ga = 2*(a*_eval(A, b)-_eval(B, b))
            gb = a*a*_eval(Ap, b)-2*a*_eval(Bp, b)
            ng, nt = ga*ga+gb*gb, da*da+db*db
            if t == 0:
                # The stored center is a rational proxy for an algebraic
                # critical point. Its tiny nonzero gradient has no meaningful
                # launch direction. Retain this measurement separately; the
                # attachment obligation is not discharged by a tangent test.
                anchor = {"status": "requires_attachment_check",
                          "sin_squared": (float((da*gb-db*ga)**2/(nt*ng))
                                          if nt and ng else None)}
                continue
            if not ng:
                stationary += 1
                continue
            if not nt:
                zero_tangent += 1
                continue
            dot = direction*(da*ga+db*gb)
            opposed += dot < 0
            orthogonal += dot == 0
            worst = max(worst, (da*gb-db*ga)**2/(nt*ng))
        return {"status": "measured", "samples": len(samples),
                "stationary_samples": stationary, "zero_tangents": zero_tangent,
                "opposed_flow": opposed, "orthogonal_flow": orthogonal,
                "max_sin_squared": float(worst), "cross_nonzero": bool(worst),
                "anchor": anchor,
                "scope": "sampled_smooth_polynomial"}
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return {"status": "invalid", "reason": "invalid_jet_evidence"}


def inspect_trajectory(m, branch, *, measure_jets=False, measure_native=False,
                       sample_budget=250000):
    """Report evidence coverage without claiming the four portrait obligations.

    Exact jet measurements are opt-in because their cost depends on degree,
    order and coefficient size. Unsupported smooth representations are always
    unresolved, even if the old topology verdict is certified.
    """
    pieces = branch.pieces or ()
    coverage = inventory(branch.Y, pieces)
    reports = []
    if branch.kind not in ("stable", "unstable"):
        coverage = {"status": "invalid", "reason": "unknown_flow"}
    if coverage["status"] == "complete":
        for piece in pieces:
            result = {"status": "unresolved", "reason": "smooth_evidence_unavailable"}
            if piece.kind == "manifold_jet":
                result = (measure_jet(m, branch.Y[piece.start:piece.stop+1],
                                      piece.evidence,
                                      1 if branch.kind == "stable" else -1)
                          if measure_jets else
                          {"status": "unresolved", "reason": "measurement_not_requested"})
            elif piece.kind == "minimum_arrival":
                if measure_native or measure_jets:
                    from .arrival_check import measure_arrival
                    result = measure_arrival(m, branch.Y[piece.start:piece.stop+1],
                                             piece.evidence,
                                             1 if branch.kind == "stable" else -1,
                                             sample_budget=sample_budget)
                else:
                    result = {"status": "unresolved", "reason": "measurement_not_requested"}
            elif piece.kind == "precision_bridge":
                if branch.kind != "unstable":
                    result = {"status": "invalid", "reason": "precision_bridge_requires_descent"}
                elif measure_native:
                    from .precision_bridge_check import measure_precision_bridge
                    result = measure_precision_bridge(m, branch.Y, piece, sample_budget=sample_budget)
                else:
                    result = {"status": "unresolved", "reason": "measurement_not_requested"}
            elif piece.kind in ("capture_connector", "handoff"):
                result = {"status": "unresolved", "reason": "requires_attachment_check"}
            elif piece.evidence.get("representation") == "native_dense_trace":
                if measure_native:
                    from .native_trace import measure_native as measure
                    result = measure(m, branch.Y, piece,
                                     1 if branch.kind == "stable" else -1,
                                     sample_budget=sample_budget)
                else:
                    result = {"status": "unresolved", "reason": "measurement_not_requested"}
            reports.append({**piece.summary(), **result})
    return {"status": "invalid" if coverage["status"] == "invalid"
            or any(r["status"] == "invalid" for r in reports) else "unresolved",
            "coverage": coverage, "pieces": reports,
            "termination": getattr(branch, "term", None),
            "certified": False, "scope": "trajectory_evidence_only"}
