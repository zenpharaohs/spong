"""A posteriori topology certification for computed phase portraits.

The geometry machine is allowed to propose curves.  This module checks the
proposal without using the continuation dispatcher's internal decisions.
It uses bounding-volume hierarchies (the same output-sensitive role as a
Bentley--Ottmann sweep for these polylines) and robust orientation predicates
to find forbidden invariant-manifold intersections.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import hypot

import numpy as np

from . import _poly as P
from . import sturm


def _norm2(x):
    """Overflow-safe Euclidean norm for a planar vector."""
    x = np.asarray(x)
    return float(np.hypot(x[0], x[1]))


def _row_norm2(x):
    """Overflow-safe planar row norms."""
    x = np.asarray(x)
    return np.hypot(x[..., 0], x[..., 1])


@dataclass(frozen=True)
class _Node:
    lo: int
    hi: int
    box: tuple[float, float, float, float]
    left: object | None = None
    right: object | None = None


def _tree(Y: np.ndarray, lo: int = 0, hi: int | None = None) -> _Node:
    hi = len(Y)-1 if hi is None else hi
    points = Y[lo:hi+1]
    box = (float(np.min(points[:, 0])), float(np.max(points[:, 0])),
           float(np.min(points[:, 1])), float(np.max(points[:, 1])))
    if hi-lo <= 32:
        return _Node(lo, hi, box)
    mid = (lo+hi)//2
    return _Node(lo, hi, box, _tree(Y, lo, mid), _tree(Y, mid, hi))


def _boxes_overlap(a, b, tol):
    return not (a[1]+tol < b[0] or b[1]+tol < a[0]
                or a[3]+tol < b[2] or b[3]+tol < a[2])


def _orient(a, b, c):
    return ((b[0]-a[0])*(c[1]-a[1])
            - (b[1]-a[1])*(c[0]-a[0]))


def _point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    px, py = p[0]-a[0], p[1]-a[1]
    q = dx*dx+dy*dy
    if q == 0:
        return hypot(px, py)
    t = max(0.0, min(1.0, (px*dx+py*dy)/q))
    return hypot(px-t*dx, py-t*dy)


def _segment_event(a, b, c, d, tol):
    scale = max(hypot(b[0]-a[0], b[1]-a[1]),
                hypot(d[0]-c[0], d[1]-c[1]), 1.0)
    floor = tol*scale
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1*o2 < 0 and o3*o4 < 0:
        den = (b[0]-a[0])*(d[1]-c[1])-(b[1]-a[1])*(d[0]-c[0])
        t = ((c[0]-a[0])*(d[1]-c[1])
             - (c[1]-a[1])*(d[0]-c[0]))/den
        return "cross", a+t*(b-a)
    distance = min(_point_segment_distance(a, c, d),
                   _point_segment_distance(b, c, d),
                   _point_segment_distance(c, a, b),
                   _point_segment_distance(d, a, b))
    if (min(abs(o1), abs(o2), abs(o3), abs(o4)) <= floor
            and distance <= tol):
        # A predicate at its FP64 resolution floor cannot certify ordering.
        return "ambiguous", 0.25*(a+b+c+d)
    return None


def _leaf_events(Y1, n1, Y2, n2, tol):
    for i in range(n1.lo, n1.hi):
        a, b = Y1[i], Y1[i+1]
        ab = (min(a[0], b[0]), max(a[0], b[0]),
              min(a[1], b[1]), max(a[1], b[1]))
        for j in range(n2.lo, n2.hi):
            c, d = Y2[j], Y2[j+1]
            cd = (min(c[0], d[0]), max(c[0], d[0]),
                  min(c[1], d[1]), max(c[1], d[1]))
            if not _boxes_overlap(ab, cd, tol):
                continue
            event = _segment_event(a, b, c, d, tol)
            if event is not None:
                yield (i, j, event[0], tuple(map(float, event[1])))


def _pair_events(Y1, root1, Y2, root2, tol):
    stack = [(root1, root2)]
    while stack:
        a, b = stack.pop()
        if not _boxes_overlap(a.box, b.box, tol):
            continue
        if a.left is None and b.left is None:
            yield from _leaf_events(Y1, a, Y2, b, tol)
        elif b.left is None or (
                a.left is not None and a.hi-a.lo >= b.hi-b.lo):
            stack.extend(((a.left, b), (a.right, b)))
        else:
            stack.extend(((a, b.left), (a, b.right)))
def _self_events(Y, root, tol):
    def visit(node):
        if node.left is None:
            yield from (x for x in _leaf_events(Y, node, Y, node, tol)
                        if x[0] < x[1] and abs(x[0]-x[1]) > 1)
            return
        yield from visit(node.left)
        yield from visit(node.right)
        yield from (x for x in _pair_events(
            Y, node.left, Y, node.right, tol)
                    if abs(x[0]-x[1]) > 1)
    yield from visit(root)


def _backbone_crossings(m, Y):
    w = np.asarray([a-m.s_a_star(float(b)) for a, b in Y])
    crossings = []
    for i in np.flatnonzero(w[:-1]*w[1:] < 0):
        t = abs(w[i])/(abs(w[i])+abs(w[i+1]))
        p = Y[i]+t*(Y[i+1]-Y[i])
        crossings.append((int(i), float(p[0]), float(p[1])))
    return crossings


def _derivative_coefficients(p, order):
    c = [float(x) for x in p]
    for _ in range(order):
        c = [(i+1)*c[i+1] for i in range(len(c)-1)]
    return c


def _polynomial_abs_bound(p, order, radius):
    """Coefficient-majorant bound for |p^(order)(x)|, |x| <= radius."""
    c = _derivative_coefficients(p, order)
    power, total = 1.0, 0.0
    for value in c:
        total += abs(value)*power
        power *= radius
    return total


def _polyval(coefficients, x):
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value*x+coefficient
    return value


def _centered_combination_bound(terms, center, radius):
    """Taylor-majorant bound retaining cancellations at a nonzero center."""
    max_degree = max((len(p)-1-order for p, order, _ in terms), default=0)
    total, factorial, power = 0.0, 1.0, 1.0
    for n in range(max_degree+1):
        if n:
            factorial *= n
            power *= radius
        coefficient = 0.0
        for p, order, multiplier in terms:
            coefficient += multiplier*_polyval(
                _derivative_coefficients(p, order+n), center)/factorial
        total += abs(coefficient)*power
    return total*(1.0+64*np.finfo(float).eps)


def _minimum_basin_radii(m, enumeration):
    """Certified strictly-convex isolating balls around the minima.

    Weyl's inequality gives lambda_min(H(z)) >= lambda_min(H(q))-r M,
    where M bounds the Frobenius norm of the Hessian derivative throughout
    the radius-r box.  Coefficient majorants make M an outward bound for the
    polynomial Hessian.  The ball is also kept disjoint from every other
    critical point.  Once an unstable branch enters such a ball, topology
    only needs its terminal label; sub-ulp crossings between two sampled
    curves converging to that same minimum are immaterial.
    """
    points = np.asarray([(p.a, p.b) for p in enumeration.points], dtype=float)
    radii = {}
    for q in enumeration.minima:
        center = np.array([q.a, q.b], dtype=float)
        distances = _row_norm2(points-center)
        distances = distances[distances > 0.0]
        separation = float(np.min(distances)) if len(distances) else 2.0
        r = min(0.45*separation, max(1.0, _norm2(center)))
        # The basin certificate consumes the same exact critical-point
        # spectral data as the local charts.  Re-diagonalizing the globally
        # evaluated FP64 Hessian here can erase its smaller eigenvalue.
        lam = float(q.local.spectral.eigenvalues[0])
        certified = 0.0
        for _ in range(60):
            Ap = _centered_combination_bound(
                [(m.alpha, 1, 1.0)], q.b, r)
            App = _centered_combination_bound(
                [(m.alpha, 2, 1.0)], q.b, r)
            Appp = _centered_combination_bound(
                [(m.alpha, 3, 1.0)], q.b, r)
            q0 = _centered_combination_bound([
                (m.beta, 2, -2.0), (m.alpha, 2, 2.0*q.a)], q.b, r)
            qbound = q0 + 2.0*r*App
            r0 = _centered_combination_bound([
                (m.beta, 3, -2.0*q.a),
                (m.alpha, 3, q.a*q.a)], q.b, r)
            r1 = _centered_combination_bound([
                (m.beta, 3, -2.0),
                (m.alpha, 3, 2.0*q.a)], q.b, r)
            rbound = r0 + r*r1 + r*r*Appp
            Ha = np.sqrt(2.0*(2.0*Ap)**2 + qbound*qbound)
            Hb = np.sqrt((2.0*Ap)**2 + 2.0*qbound*qbound
                         + rbound*rbound)
            lipschitz = float(np.hypot(Ha, Hb))
            if lam > r*lipschitz*(1.0+64*np.finfo(float).eps):
                certified = r
                break
            r *= 0.5
        radii[(float(q.a), float(q.b))] = certified
    return radii


def sublevel_component_minima(m, enumeration, point):
    """Minima in the certified algebraic sublevel component containing point.

    Since ``L = A(b)(a-a*)^2 + u(b)`` and ``A>0``, components of ``{L<c}``
    project to components of

        A(b)(C-c) - B(b)^2 < 0.

    The level polynomial is built exactly from the binary64 upper enclosure
    of the measured level and its real roots are isolated by exact Sturm
    arithmetic.  Returning every minimum on the resulting b-interval is a
    safe topological filter; an ambiguity returns all minima.
    """
    a, b = map(float, point)
    A, B = float(m.A(b)), float(m.B(b))
    level = float(m.L(a, b))
    scale = abs(float(m.C))+2.0*abs(a*B)+a*a*abs(A)
    error = 1024.0*np.finfo(float).eps*(1.0+scale)
    c_upper = np.nextafter(level+error, np.inf)
    if not np.isfinite(c_upper):
        return list(enumeration.minima)
    c = Fraction.from_float(float(c_upper))
    level_poly = P.sub(
        P.scale(m.alpha, m.C-c), P.mul(m.beta, m.beta))
    try:
        roots = [
            sturm.refine(level_poly, iv, Fraction(1, 2**80))
            for iv in sturm.isolate_roots(level_poly)]
    except (ArithmeticError, OverflowError, ValueError):
        return list(enumeration.minima)
    bq = Fraction.from_float(b)
    value = P.eval_at(level_poly, bq)
    if value >= 0:
        # The point should lie strictly inside the enlarged sublevel.  If
        # global evaluation was too poorly conditioned to establish that,
        # declining the filter is safer than excluding a true destination.
        return list(enumeration.minima)
    for iv in roots:
        if iv.lo <= bq <= iv.hi:
            return list(enumeration.minima)
    left = max((iv.hi for iv in roots if iv.hi < bq), default=None)
    right = min((iv.lo for iv in roots if iv.lo > bq), default=None)
    feasible = [
        q for q in enumeration.minima
        if (left is None or Fraction.from_float(float(q.b)) > left)
        and (right is None or Fraction.from_float(float(q.b)) < right)]
    return feasible or list(enumeration.minima)


def _same_certified_minimum_tail(branch1, branch2, point, basin_radii,
                                 allowed_radius):
    if branch1.term != "capture" or branch2.term != "capture":
        return False
    t1, t2 = np.asarray(branch1.Y[-1]), np.asarray(branch2.Y[-1])
    if _norm2(t1-t2) > allowed_radius:
        return False
    radius = basin_radii.get((float(t1[0]), float(t1[1])), 0.0)
    return radius > 0.0 and _norm2(np.asarray(point)-t1) < radius


def _same_certified_infinity_tail(branch1, branch2, point):
    """Whether an apparent contact lies in a shared certified end at infinity."""
    a1, a2 = branch1.certs.get("asymptote"), branch2.certs.get("asymptote")
    if not a1 or not a2 or branch1.term != "box_exit" \
            or branch2.term != "box_exit":
        return False
    if not (np.isfinite(a1["residual"]) and a1["residual"] < 0.2
            and np.isfinite(a2["residual"]) and a2["residual"] < 0.2):
        return False
    u1 = np.asarray(branch1.Y[-1], dtype=float)
    u2 = np.asarray(branch2.Y[-1], dtype=float)
    u1 /= _norm2(u1)
    u2 /= _norm2(u2)
    # The compactified end includes the sign of the ray, not merely |b/a|.
    if float(u1@u2) < 0.95:
        return False
    entry = max(float(a1["radii"][0]), float(a2["radii"][0]))
    return _norm2(point) >= entry


def _exit_side(point, box):
    distances = (abs(point[0]-box[0]), abs(point[0]-box[1]),
                 abs(point[1]-box[2]), abs(point[1]-box[3]))
    return int(np.argmin(distances))


def _level_suffix(m, branch, threshold, below):
    """First index of the final strict sub/superlevel suffix, or None."""
    values = np.asarray([float(m.L(float(a), float(b)))
                         for a, b in branch.Y])
    slack = 512*np.finfo(float).eps*(1.0+abs(threshold))
    inside = values < threshold-slack if below else values > threshold+slack
    if not inside[-1]:
        return None
    outside = np.flatnonzero(~inside)
    return int(outside[-1]+1) if len(outside) else 0


def audit(m, enumeration, branches, box) -> dict:
    """Return an independent topology/FP64 certificate for a portrait."""
    scale = max(1.0, *(abs(x) for x in box))
    predicate_tol = 128*np.finfo(float).eps*scale
    segment_count = sum(max(0, len(br.Y)-1) for br in branches)
    segment_budget = 1000000
    aborted = [i for i, br in enumerate(branches)
               if br.term not in ("capture", "box_exit")]
    if aborted:
        # A partial invariant manifold cannot support a topology
        # certificate.  In particular, enlarging the trace box cannot repair
        # a failed local handoff or continuation step.  Do not spend minutes
        # constructing BVHs and enumerating thousands of intersections among
        # curves which are already known to be incomplete.
        stable_tails = [{
            "branch": i, "box_exit": br.term == "box_exit",
            "asymptote_residual": (
                None if br.certs.get("asymptote") is None
                else br.certs["asymptote"]["residual"]),
            "superlevel_start": None, "exit_side": None,
            "certified": False,
        } for i, br in enumerate(branches) if br.kind == "stable"]
        return {
            "status": "fp64_unresolved",
            "audit_complete": False,
            "resolution_reason": "branch_abort",
            "aborted_branches": aborted,
            "segment_count": segment_count,
            "segment_budget": segment_budget,
            "raw_event_count": 0,
            "raw_event_budget": 5000,
            "event_sample_limit": 256,
            "forbidden_count": 0,
            "ambiguous_count": 0,
            "predicate_tolerance": float(predicate_tol),
            "minimum_basin_radii": [],
            "terminal_suffixes": [],
            "forbidden_intersections": [],
            "ambiguous_contacts": [],
            "backbone_crossings": [],
            "unstable_candidates": [],
            "stable_tails": stable_tails,
        }
    budget_exceeded = segment_count > segment_budget
    trees = ([] if budget_exceeded else
             [_tree(np.asarray(br.Y)) for br in branches])
    critical = np.asarray([(p.a, p.b) for p in enumeration.points])
    allowed_radius = max(1024*np.finfo(float).eps*scale, 1e-11)
    basin_radii = _minimum_basin_radii(m, enumeration)
    saddle_levels = [float(m.L(p.a, p.b)) for p in enumeration.saddles]
    low_saddle = min(saddle_levels, default=np.inf)
    high_saddle = max(saddle_levels, default=-np.inf)
    terminal_suffixes = []
    for br in branches:
        if br.kind == "unstable" and br.term == "capture":
            start = _level_suffix(m, br, low_saddle, below=True)
            terminal_suffixes.append({
                "kind": "minimum_sublevel", "start": start,
                "terminal": tuple(map(float, br.Y[-1])),
            })
        elif br.kind == "stable" and br.term == "box_exit":
            start = _level_suffix(m, br, high_saddle, below=False)
            terminal_suffixes.append({
                "kind": "infinity_superlevel", "start": start,
                "side": _exit_side(br.Y[-1], box),
            })
        else:
            terminal_suffixes.append({"kind": None, "start": None})
    # Intersection output can itself be enormous when two failed traces run
    # nearly coincident.  Certification needs the existence and count of such
    # events, not millions of retained Python dictionaries.
    event_sample_limit = 256
    raw_event_budget = 5000
    raw_event_count = 0
    event_budget_exceeded = False
    forbidden, ambiguous = [], []
    forbidden_count = ambiguous_count = 0

    def record(item, kind):
        nonlocal forbidden_count, ambiguous_count
        sample = forbidden if kind == "cross" else ambiguous
        if kind == "cross":
            forbidden_count += 1
        else:
            ambiguous_count += 1
        if len(sample) < event_sample_limit:
            sample.append(item)

    for i, br in enumerate(branches if not budget_exceeded else ()):
        for si, sj, kind, point in _self_events(
                np.asarray(br.Y), trees[i], predicate_tol):
            if max(si, sj) < int(br.diag.get("critical_steps", 0)):
                # The materialized invariant graph has its own injectivity
                # and fixed-point certificate; logarithmic samples can be
                # closer than a physical-coordinate ulp near the saddle.
                continue
            p = np.asarray(point)
            if len(critical) and np.min(_row_norm2(
                    critical-p)) <= allowed_radius:
                continue
            raw_event_count += 1
            if raw_event_count > raw_event_budget:
                event_budget_exceeded = True
                break
            item = {"branches": (i, i), "segments": (si, sj),
                    "kind": kind, "point": point}
            record(item, kind)
        if event_budget_exceeded:
            break
    for i in range(0 if budget_exceeded or event_budget_exceeded
                   else len(branches)):
        for j in range(i+1, len(branches)):
            # Unstable curves ending at the same certified minimum are basin
            # interiors, not separatrices.  Their mutual polyline ordering is
            # topologically irrelevant, so do not generate a potentially
            # quadratic contact stream only to discard every event later.
            pair_same_captured_basin = (
                branches[i].kind == branches[j].kind == "unstable"
                and branches[i].term == branches[j].term == "capture"
                and _norm2(
                    branches[i].Y[-1]-branches[j].Y[-1])
                <= allowed_radius)
            if pair_same_captured_basin:
                continue
            events = _pair_events(
                np.asarray(branches[i].Y), trees[i],
                np.asarray(branches[j].Y), trees[j], predicate_tol)
            for si, sj, kind, point in events:
                same_source_stub_germ = (
                    branches[i].diag.get("saddle_b") is not None
                    and branches[j].diag.get("saddle_b") is not None
                    and abs(branches[i].diag["saddle_b"]
                            - branches[j].diag["saddle_b"])
                    <= allowed_radius
                    and si < int(branches[i].diag.get(
                        "critical_steps", 0))
                    and sj < int(branches[j].diag.get(
                        "critical_steps", 0)))
                if same_source_stub_germ:
                    # Stable and unstable local graphs meet at their common
                    # critical point.  In a very stiff chart many distinct
                    # centered samples map inside one global-coordinate
                    # predicate tolerance, but their fixed-point,
                    # injectivity, and spectral certificates already prove
                    # that the two germs are transverse and do not cross
                    # again.  Do not reinterpret that certified local chart
                    # as hundreds of ambiguous global contacts.
                    continue
                p = np.asarray(point)
                if len(critical) and np.min(_row_norm2(
                        critical-p)) <= allowed_radius:
                    continue
                ti, tj = terminal_suffixes[i], terminal_suffixes[j]
                same_sublevel_end = (
                    ti["kind"] == tj["kind"] == "minimum_sublevel"
                    and ti["start"] is not None and tj["start"] is not None
                    and si >= ti["start"] and sj >= tj["start"]
                    and _norm2(
                        np.asarray(ti["terminal"])
                        - np.asarray(tj["terminal"])) <= allowed_radius)
                # Unstable curves are basin interiors, not basin boundaries.
                # Once continuation has independently captured both at the
                # same certified critical point, their mutual polyline
                # ordering carries no Morse-complex information and may be
                # rerouted inside that basin.  Crossings with a STABLE
                # separatrix remain forbidden everywhere: those are the
                # topology-changing events this audit is designed to catch.
                same_superlevel_end = (
                    ti["kind"] == tj["kind"] == "infinity_superlevel"
                    and ti["start"] is not None and tj["start"] is not None
                    and si >= ti["start"] and sj >= tj["start"]
                    and ti["side"] == tj["side"])
                if (same_sublevel_end
                        or same_superlevel_end
                        or _same_certified_minimum_tail(
                        branches[i], branches[j], point, basin_radii,
                        allowed_radius)
                        or _same_certified_infinity_tail(
                            branches[i], branches[j], point)):
                    continue
                raw_event_count += 1
                if raw_event_count > raw_event_budget:
                    event_budget_exceeded = True
                    break
                item = {"branches": (i, j), "segments": (si, sj),
                        "kind": kind, "point": point}
                record(item, kind)
            if event_budget_exceeded:
                break
        if event_budget_exceeded:
            break

    backbone = [
        {"branch": i, "kind": br.kind,
         "crossings": _backbone_crossings(m, np.asarray(br.Y))}
        for i, br in enumerate(branches)
    ]
    candidates = []
    minimum_levels = [(p.b, float(m.L(p.a, p.b)))
                      for p in enumeration.minima]
    for i, br in enumerate(branches):
        if br.kind != "unstable":
            continue
        sampled_floor = min(float(m.L(float(a), float(b)))
                            for a, b in br.Y[::max(1, len(br.Y)//4096)])
        eligible = [b for b, level in minimum_levels
                    if level <= sampled_floor
                    + 128*np.finfo(float).eps*(1+abs(sampled_floor))]
        candidates.append({"branch": i, "sampled_loss_floor": sampled_floor,
                           "eligible_minimum_b": eligible})

    stable_tails = []
    for i, br in enumerate(branches):
        if br.kind != "stable":
            continue
        asym = br.certs.get("asymptote")
        level_tail = terminal_suffixes[i]
        superlevel_certified = (
            level_tail["kind"] == "infinity_superlevel"
            and level_tail["start"] is not None)
        stable_tails.append({
            "branch": i, "box_exit": br.term == "box_exit",
            "asymptote_residual": None if asym is None else asym["residual"],
            "superlevel_start": level_tail["start"],
            "exit_side": level_tail.get("side"),
            "certified": (br.term == "box_exit"
                          and (superlevel_certified
                               or (asym is not None
                                   and np.isfinite(asym["residual"])
                                   and asym["residual"] < 0.2))),
        })
    resolved = (not budget_exceeded and not event_budget_exceeded
                and forbidden_count == 0 and ambiguous_count == 0
                and all(x["certified"] for x in stable_tails)
                and all(br.term in ("capture", "box_exit") for br in branches))
    return {
        "status": "certified" if resolved else "fp64_unresolved",
        "audit_complete": not budget_exceeded and not event_budget_exceeded,
        "resolution_reason": (
            "certification_segment_budget" if budget_exceeded else
            "certification_event_budget" if event_budget_exceeded else None),
        "segment_count": segment_count,
        "segment_budget": segment_budget,
        "raw_event_count": raw_event_count,
        "raw_event_budget": raw_event_budget,
        "event_sample_limit": event_sample_limit,
        "forbidden_count": forbidden_count,
        "ambiguous_count": ambiguous_count,
        "predicate_tolerance": float(predicate_tol),
        "minimum_basin_radii": [
            {"minimum": key, "radius": float(value)}
            for key, value in basin_radii.items()],
        "terminal_suffixes": terminal_suffixes,
        "forbidden_intersections": forbidden,
        "ambiguous_contacts": ambiguous,
        "backbone_crossings": backbone,
        "unstable_candidates": candidates,
        "stable_tails": stable_tails,
    }
