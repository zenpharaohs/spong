"""Native analytic arrival proposals and bounded per-session chart reuse.

This adapter does no chart arithmetic. The C core consumes exact rational
jet coefficients and high-precision spectral data; arrival_check remains an
independent admission check against the original model.
"""
from functools import lru_cache
from decimal import Decimal

import numpy as np


@lru_cache(maxsize=128)
def _chart(strings, order, precision):
    from . import _native
    return _native.arrival_chart(strings, order, precision)


def clear_cache():
    _chart.cache_clear()


def _rational_text(value):
    """Exact decimal transport without changing Python's global digit limit.

    High-degree translated jets can exceed int.__str__'s default 4300 digits.
    Decimal(int) is exact regardless of the active arithmetic precision;
    fixed-point formatting avoids a lossy float or a process-wide exemption.
    """
    try:
        return str(value)
    except ValueError:
        return f'{Decimal(value.numerator):f}/{Decimal(value.denominator):f}'


def propose(local, start, target, center, *, order=8, precision=80, exact_start=None):
    from . import _native
    if type(order) is not int or not 2 <= order <= 12:
        raise ValueError('arrival_order')
    if type(precision) is not int or not 50 <= precision <= local.spectral.decimal_precision:
        raise ValueError('arrival_precision')
    def coefficient(d, i, j):
        c = local.exact_grad[d]
        return _rational_text(c[i][j]) if i < len(c) and j < len(c[i]) else '0'
    strings = tuple(coefficient(d, i, j) for d in range(2)
                    for i in range(order+1) for j in range(order+1))
    strings += tuple(_rational_text(x) for row in local.exact_hessian for x in row)
    strings += tuple(str(x) for row in local.spectral.decimal_frame for x in row)
    strings += tuple(map(str, local.spectral.decimal_eigenvalues))
    chart = _chart(strings, order, precision)
    extra = () if exact_start is None else tuple(_rational_text(x) for x in exact_start)
    raw = _native.arrival_match(chart, *map(str, center), *map(float, start), *map(float, target), *extra)
    points = np.asarray(raw.pop('points'))
    evidence = {'representation': 'poincare_frobenius_arrival', 'schema': 1,
                'center': tuple(map(str, center)), 'order': order, 'precision': precision,
                'terminal': 'analytic_limit', 'backend': 'native_gmp', **raw}
    if exact_start is not None:
        evidence['entry_coordinates']=extra
        evidence['entry_mode']='exact_rational'
    return points, evidence


def checked_proposal(m, minimum, start, target, *, order=8, precision=80,
                     check_precision=90, sample_budget=10000, exact_start=None):
    """Return only independently admitted local geometry, never a certificate."""
    from . import _poly as P
    from .arrival_check import measure_arrival, check_target
    b = minimum.local.center_interval.mid
    center = (P.eval_at(m.beta, b)/P.eval_at(m.alpha, b), b)
    points, evidence = propose(minimum.local, start, target, center, order=order, precision=precision, exact_start=exact_start)
    geometry = measure_arrival(m, points, evidence, precision=check_precision,
                               sample_budget=sample_budget)
    attachment = check_target(m, minimum, evidence, precision=check_precision)
    if (geometry.get('status') != 'measured'
            or geometry.get('max_sin_squared', 1) > 1e-6
            or geometry.get('binding_eps', float('inf')) > 64
            or any(geometry.get(k, 0) for k in ('opposed_flow', 'zero_tangents', 'stationary_samples'))
            or attachment.get('status') != 'measured'
            or not all(attachment.get(k) for k in ('center_bound', 'positive_hessian', 'inward'))
            or attachment.get('sin_squared', 1) > 1e-12):
        raise ValueError('arrival_independent_check_failed')
    return points, evidence, {'geometry': geometry, 'attachment': attachment,
                              'scope': 'sampled_local_arrival', 'certified': False}



def _prefix_admitted(g):
    return (g.get('status') == 'measured' and g.get('max_sin_squared', 1) <= 1e-6
            and g.get('binding_eps', float('inf')) <= 64
            and g.get('join_eps', float('inf')) <= 64
            and not any(g.get(k, 0) for k in ('opposed_flow', 'zero_tangents',
                                             'stationary_samples', 'unvalidated_steps')))


def _entry_trace(minimum, start, target):
    """Explicit native call: no mutation of global engine selection."""
    from . import _native
    from .charts import CRITICAL_STEP_FRACTION, GEOMETRIC_IRK_PRIMARY
    local = minimum.local
    rates = np.asarray(local.spectral.eigenvalues, dtype=float)
    if local.native is None or not np.all(np.isfinite(rates)) or np.min(rates) <= 0:
        raise ValueError('arrival_handoff_trace_unavailable')
    raw = _native.centered_arrival(
        local.native, *map(float, start), *map(float, target),
        float(local.a), float(local.b), float(np.min(rates)), float(np.max(rates)),
        0., 512, float(np.cos(2*np.arctan(CRITICAL_STEP_FRACTION))),
        GEOMETRIC_IRK_PRIMARY, 0, 0, float('nan'), float('nan'), 1, 1)
    bridge = raw[-1]
    raw = raw[:-1]
    points = np.frombuffer(raw[-2], dtype=float).reshape(-1, 2).copy()
    if not len(points) or not np.array_equal(points[0], start):
        raise ValueError('arrival_handoff_incoming_anchor_lost')
    evidence = {'representation': 'native_dense_trace', 'schema': 1,
                'trace': raw[-1], 'branch_offset': 0, 'source_points': len(points)}
    return points, evidence, {'term_code': raw[0], 'accepted_steps': raw[3],
                              'rejected_steps': raw[4], 'finish_radius': raw[9],
                              'bridge': bridge}


def checked_path(m, minimum, start, target, *, order=8, precision=80,
                 check_precision=90, sample_budget=10000):
    """Admit an analytic tail or a retained checked native-prefix/tail pair.

    Bounded radius-shell search schedules inverse attempts; passing distance
    alone never accepts an entry. Original points and all owned chords stay
    represented. Failure returns no partial replacement and remains explicit.
    """
    from .trajectory import TrajectoryPiece
    from .native_trace import measure_native
    from .arrival_check import measure_handoff
    kwargs = dict(order=order, precision=precision,
                  check_precision=check_precision, sample_budget=sample_budget)
    try:
        points, evidence, checks = checked_proposal(m, minimum, start, target, **kwargs)
        return points, (TrajectoryPiece('minimum_arrival', 0, len(points)-1, evidence),), checks
    except ValueError as exc:
        reason = str(exc)
        if reason not in ('arrival_matching_not_converged', 'arrival_independent_check_failed'):
            raise
        original = {'reason': reason, 'matching': getattr(exc, 'matching', None)}
    attempts = []
    diagnostics = {'policy': 'checked-native-handoff-v2', 'original': original,
                   'attempts': attempts, 'step_budget': 512, 'shell_budget': 12}
    def refuse(reason):
        error = ValueError(reason)
        error.handoff = diagnostics
        raise error
    try:
        raw, trace, tracing = _entry_trace(minimum, start, target)
    except ValueError as exc:
        diagnostics['trace_error'] = str(exc)
        refuse('arrival_handoff_trace_unavailable')
    diagnostics['trace'] = tracing
    distances = np.hypot(*(raw-np.asarray(target, dtype=float)).T)
    candidates = []
    # Exclude the raw tracer's terminal connector and any rounded target.
    for shell in range(1, 13):
        indices = np.flatnonzero((distances <= distances[0]/(2**shell))
                                 & (distances > 0) & (np.arange(len(raw)) > 0))
        if len(indices) and int(indices[0]) not in candidates:
            candidates.append(int(indices[0]))
    for index in candidates:
        trial = {'index': index, 'distance_ratio': float(distances[index]/distances[0])}
        attempts.append(trial)
        try:
            tail, evidence, checks = checked_proposal(m, minimum, raw[index], target, **kwargs)
        except ValueError as exc:
            trial['reason'] = str(exc)
            trial['matching'] = getattr(exc, 'matching', None)
            continue
        prefix = TrajectoryPiece('centered_arrival', 0, index, trace)
        g = measure_native(m, raw, prefix, -1, sample_budget=sample_budget)
        trial['prefix'] = g
        if not _prefix_admitted(g):
            # Later candidates contain this same rejected prefix. Do not
            # spend further checker budget trying to hide it under a tail.
            refuse('arrival_handoff_prefix_rejected')
        points = np.vstack((raw[:index+1], tail[1:]))
        arrival = TrajectoryPiece('minimum_arrival', index, len(points)-1, evidence)
        join = measure_handoff(m, points, prefix, arrival)
        trial['join'] = join
        if (join.get('status') != 'measured' or join.get('join_eps', float('inf')) > 64
                or join.get('sin_squared', 1) > 1e-6 or not join.get('same_orientation')):
            trial['reason'] = 'arrival_handoff_join_rejected'
            continue
        trial['accepted'] = True
        diagnostics['selected_index'] = index
        checks = {**checks, 'prefix': g, 'join': join, 'handoff': diagnostics,
                  'scope': 'sampled_native_prefix_and_analytic_arrival', 'certified': False}
        return points, (prefix, arrival), checks
    refuse('arrival_handoff_no_checked_entry')

def refine_branches(m, enumeration, branches):
    """Checked portrait dispatch; failed arrivals remain explicit open prefixes."""
    from dataclasses import replace
    from .arrival_series import refine_capture
    result = []
    for branch in branches:
        if branch.term != 'capture':
            result.append(branch)
            continue
        try:
            result.append(refine_capture(m, enumeration, branch, backend='native'))
        except (ValueError, ArithmeticError) as exc:
            has_connector = bool(branch.pieces and branch.pieces[-1].kind == 'capture_connector')
            result.append(replace(
                branch, Y=branch.Y[:-1].copy() if has_connector else branch.Y.copy(),
                pieces=branch.pieces[:-1] if has_connector else branch.pieces,
                term='abort_analytic_arrival', certs={},
                diag={**branch.diag, 'target': None,
                      'arrival_refusal': {'reason': str(exc), 'backend': 'native_gmp',
                                          'proposed_target': branch.Y[-1].tolist(),
                                          'handoff': getattr(exc, 'handoff', None)}}))
    return result
