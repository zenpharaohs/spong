"""Independent decoder and rational scrutiny of native trace schema 1.

No constructor or integration code is imported. Stages define a proposed
collocation polynomial; their agreement with the field is remeasured here.
"""
from fractions import Fraction as Q

import numpy as np

from .trajectory_check import _eval, _derivative


def decode(blob):
    if not isinstance(blob, bytes) or len(blob) % (24*8):
        raise ValueError("malformed_trace")
    rows = np.frombuffer(blob, dtype=np.float64).reshape(-1, 24)
    if not np.all(np.isfinite(rows)):
        raise ValueError("nonfinite_trace")
    vertices, dense = {}, {}
    replaced = trimmed = 0
    for i, row in enumerate(rows):
        kind = row[0]
        if kind in (1, 2):
            if (row[1] not in (0, 1, 2, 3)
                    or row[21] not in (0, 1, 2, 3)
                    or (kind == 1 and row[2] not in (4, 6, 8))
                    or (kind == 2 and row[1] != 0)):
                raise ValueError("unknown_dense_representation")
            dense[i] = row
        elif kind == 3:
            index, source = int(row[20]), int(row[23])
            if index != row[20] or index < 0 or source != row[23]:
                raise ValueError("invalid_vertex_map")
            if source != -1 and source not in dense:
                raise ValueError("missing_dense_source")
            if not 0 <= row[22] <= 1:
                raise ValueError("invalid_parameter")
            replaced += index in vertices
            vertices[index] = (source, Q(float(row[22])), tuple(row[4:6]))
        elif kind == 4:
            count = int(row[20])
            if count != row[20] or count < 1 or count > len(vertices):
                raise ValueError("invalid_trim")
            trimmed += len(vertices)-count
            vertices = {j: v for j, v in vertices.items() if j < count}
        else:
            raise ValueError("unknown_trace_event")
    if sorted(vertices) != list(range(len(vertices))):
        raise ValueError("incomplete_vertex_map")
    return dense, vertices, {"replaced_vertices": replaced, "trimmed_vertices": trimmed}


def _mul(a, b):
    out = [Q(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def polynomial(row):
    r = tuple(Q(float(v)) for v in row)
    if r[0] == 2:
        out = []
        for d in (0, 1):
            a, b, f, g = r[4+d], r[6+d], r[3]*r[8+d], r[3]*r[10+d]
            out.append((a, f, -3*a+3*b-2*f-g, 2*a-2*b+f+g))
        return tuple(out)
    n = int(r[2])//2
    nodes = r[14:14+n]
    if len(set(nodes)) != n or not all(0 < x < 1 for x in nodes):
        raise ValueError("invalid_collocation_nodes")
    out = [[r[4+d]]+[Q(0)]*n for d in (0, 1)]
    for i in range(n):
        basis, den = [Q(1)], Q(1)
        for j in range(n):
            if i != j:
                basis = _mul(basis, [-nodes[j], Q(1)])
                den *= nodes[i]-nodes[j]
        for d in (0, 1):
            for k, c in enumerate(basis):
                out[d][k+1] += r[3]*r[6+2*i+d]*c/(den*(k+1))
    return tuple(tuple(c) for c in out)


def physical(m, row, coefficients, t):
    x, y = (_eval(c, t) for c in coefficients)
    dx, dy = (_eval(_derivative(c), t) for c in coefficients)
    chart = int(row[1])
    if chart in (1, 2):
        b, w, db, dw = (x, y, dx, dy) if chart == 1 else (y, x, dy, dx)
        A, B = _eval(m.alpha, b), _eval(m.beta, b)
        if not A:
            raise ValueError("chart_pole")
        Ap, Bp = _eval(_derivative(m.alpha), b), _eval(_derivative(m.beta), b)
        return (B/A+w, b), ((Bp*A-B*Ap)*db/(A*A)+dw, db)
    if chart == 3:
        x += Q(float(row[22])); y += Q(float(row[23]))
    return (x, y), (dx, dy)


def measure_native(m, points, piece, direction, *, sample_budget=250000):
    """Every active step at endpoints, quartiles, and all emitted parameters.

    Budget exhaustion is unresolved; measurements never silently subsample.
    Binding error is in eps*(1+|coordinate|) units, not an acceptance test.
    """
    try:
        if direction not in (-1, 1):
            raise ValueError("invalid_direction")
        e = piece.evidence
        if e.get("schema") != 1:
            raise ValueError("unknown_trace_schema")
        dense, vertices, report = decode(e["trace"])
        if len(vertices) != e["source_points"]:
            raise ValueError("source_count_mismatch")
        offset = e["branch_offset"]
        first, last = piece.start-offset, piece.stop-offset
        if not 0 <= first < last < len(vertices):
            raise ValueError("invalid_source_range")
        for j in range(first, last+1):
            if not np.array_equal(vertices[j][2], points[offset+j]):
                raise ValueError("geometry_mismatch")
            if j>first and vertices[j][0]<0:
                raise ValueError("unrepresented_vertex")
        referenced = [source for j, (source, _, _) in vertices.items()
                      if j <= last and source >= 0]
        if not referenced:
            raise ValueError("missing_dense_evidence")
        active = {i: row for i, row in dense.items() if i <= max(referenced)}
        # A deleted tail must never contribute evidence to the surviving path.
        report["discarded_steps"] = len(dense)-len(active)
        parameters = {i: {Q(0), Q(1,4), Q(1,2), Q(3,4), Q(1)} for i in active}
        samples_by_source = {i: [] for i in active}
        for j, (source, t, _) in vertices.items():
            if j <= last and source in parameters:
                parameters[source].add(t)
                if j >= first:
                    samples_by_source[source].append(vertices[j])
        # If a boundary cuts dense output before its endpoint, inspect only
        # the surviving prefix of that final step.
        final_source = max(referenced)
        final_t = max(t for j,(s,t,_) in vertices.items() if j <= last and s == final_source)
        parameters[final_source] = {t for t in parameters[final_source] if t <= final_t}
        required = sum(map(len, parameters.values()))
        if required > sample_budget:
            return {**report, "status": "unresolved", "reason": "measurement_budget",
                    "required_samples": required, "sample_budget": sample_budget}
        A, B = tuple(map(Q, m.alpha)), tuple(map(Q, m.beta))
        Ap, Bp = _derivative(A), _derivative(B)
        worst = Q(0); binding = Q(0); joins = Q(0)
        opposed = stationary = zero = unchecked = 0
        # Bind the incoming anchor as well as all emitted endpoints.
        # Otherwise a displaced first step could leave its first chord unchecked.
        previous = tuple(Q(float(x)) for x in vertices[0][2])
        for i, row in active.items():
            c = polynomial(row)
            unchecked += int(row[21] == 3)
            start, _ = physical(m, row, c, Q(0))
            if previous is not None:
                joins = max(joins, *(abs(x-y)/(Q(1,2**52)*(1+max(abs(x),abs(y))))
                                     for x,y in zip(start, previous)))
            end_parameter = final_t if i == final_source else Q(1)
            previous, _ = physical(m, row, c, end_parameter)
            for t in sorted(parameters[i]):
                (a,b), (da,db) = physical(m, row, c, t)
                ga=2*(a*_eval(A,b)-_eval(B,b))
                gb=a*a*_eval(Ap,b)-2*a*_eval(Bp,b)
                ng, nt = ga*ga+gb*gb, da*da+db*db
                if not ng: stationary += 1; continue
                if not nt: zero += 1; continue
                opposed += direction*(ga*da+gb*db) <= 0
                worst = max(worst,(da*gb-db*ga)**2/(ng*nt))
            for source,t,q in samples_by_source[i]:
                value, _ = physical(m,row,c,t)
                binding=max(binding,*(abs(x-Q(float(y)))/(Q(1,2**52)*(1+abs(x)))
                                      for x,y in zip(value,q)))
        return {**report,"status":"measured","steps":len(active),"samples":required,
                "max_sin_squared":float(worst),"cross_nonzero":bool(worst),
                "opposed_flow":opposed,"stationary_samples":stationary,"zero_tangents":zero,
                "binding_eps":float(binding),"join_eps":float(joins),
                "unvalidated_steps":unchecked,"scope":"sampled_native_dense_geometry"}
    except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return {"status":"invalid","reason":str(exc)}
