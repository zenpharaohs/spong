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
from math import comb, hypot

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


def _native_contact_available():
    try:
        from . import _native
    except ImportError:
        return False
    return hasattr(_native, "ContactScan")


def _topology_decision_python(
        saddle_count, branch_count, stable_count, unstable_count,
        segment_count, segment_budget, raw_event_count, raw_event_budget,
        forbidden_count, ambiguous_count, uncertified_unstable_ends,
        uncertified_stable_tails, branch_aborted):
    """Independent oracle for the portable C topology state machine."""
    expected = 2*saddle_count
    inventory = (stable_count == expected
                 and unstable_count == expected
                 and branch_count == 2*expected)
    segment_ok = segment_count <= segment_budget
    event_ok = raw_event_count <= raw_event_budget
    certified = (not branch_aborted and segment_ok and event_ok and inventory
                 and forbidden_count == 0 and ambiguous_count == 0
                 and uncertified_unstable_ends == 0
                 and uncertified_stable_tails == 0)
    reason = (
        "branch_abort" if branch_aborted else
        "certification_segment_budget" if not segment_ok else
        "certification_event_budget" if not event_ok else
        "branch_inventory_incomplete" if not inventory else
        "topology_contact" if forbidden_count or ambiguous_count else
        "unstable_endpoint_unresolved"
        if uncertified_unstable_ends else
        "stable_escape_unresolved" if uncertified_stable_tails else None)
    return {
        "certified": certified,
        "audit_complete": not branch_aborted and segment_ok and event_ok,
        "branch_inventory_certified": inventory,
        "reason": reason,
        "expected_stable": expected,
        "expected_unstable": expected,
    }


def _topology_decision(*values):
    try:
        from . import _native
        decide = _native.topology_decide
    except (ImportError, AttributeError):
        return _topology_decision_python(*values)
    certified, complete, inventory, reason, stable, unstable = decide(*values)
    return {
        "certified": bool(certified),
        "audit_complete": bool(complete),
        "branch_inventory_certified": bool(inventory),
        "reason": None if reason == "none" else reason,
        "expected_stable": int(stable),
        "expected_unstable": int(unstable),
    }


def _pair_contact_events(Y1, root1, Y2, root2, tol):
    """Production contact stream, with the Python BVH retained as oracle."""
    try:
        from . import _native
        scanner = _native.ContactScan(
            np.ascontiguousarray(Y1, dtype=np.float64),
            np.ascontiguousarray(Y2, dtype=np.float64), tol, False)
    except (ImportError, AttributeError):
        yield from _pair_events(Y1, root1, Y2, root2, tol)
        return
    yield from scanner


def _self_contact_events(Y, root, tol):
    """Production self-contact stream, with the Python BVH as oracle."""
    try:
        from . import _native
        scanner = _native.ContactScan(
            np.ascontiguousarray(Y, dtype=np.float64), None, tol, True)
    except (ImportError, AttributeError):
        yield from _self_events(Y, root, tol)
        return
    yield from scanner


def _strictly_monotone_subarc(Y, first_segment, second_segment):
    """Whether one coordinate exactly orders the intervening polyline."""
    lo, hi = sorted((first_segment, second_segment))
    steps = np.diff(np.asarray(Y[lo:hi+2], dtype=float), axis=0)
    return any(np.all(steps[:, axis] > 0.0)
               or np.all(steps[:, axis] < 0.0) for axis in (0, 1))


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
    inventory = _sublevel_component_inventory(m, enumeration, point)
    return (list(inventory["minima"])
            if inventory["certified"] else list(enumeration.minima))


def _strict_level_at_float_point(m, point, above):
    """Exact level just above/below the loss at a binary64 point."""
    a, b = (Fraction.from_float(float(x)) for x in point)
    A = P.eval_at(m.alpha, b)
    B = P.eval_at(m.beta, b)
    level = m.C-2*a*B+a*a*A
    # The scale excludes the additive loss constant: level-set geometry and
    # every admission decision must be invariant under L -> L+constant.
    scale = abs(A)*(1+a*a)+2*abs(a*B)+abs(B)
    slack = scale/2**48
    return level+slack if above else level-slack


def _finite_float_or_none(value):
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if np.isfinite(result) else None


def _sublevel_component_inventory(m, enumeration, point):
    """Exact inventory of the strict sublevel component containing ``point``.

    The measured binary64 point is treated as an exact dyadic point.  We use
    an exact relative upper level, rather than a floating evaluation of L, so
    membership in ``{L<c}`` is strict by construction.  Because A>0, the
    component projects to one component of

        A(b) (C-c) - B(b)^2 < 0.

    A bounded projection gives a bounded tube.  A projection unbounded at
    exactly one end names a unique finite-plane end.  Critical isolating
    intervals must lie wholly inside or outside the component; an overlap
    makes the certificate decline rather than guess.
    """
    try:
        bq = Fraction.from_float(float(point[1]))
        c = _strict_level_at_float_point(m, point, above=True)
        level_poly = P.sub(
            P.scale(m.alpha, m.C-c), P.mul(m.beta, m.beta))
        roots = [
            sturm.refine(level_poly, iv, Fraction(1, 2**80))
            for iv in sturm.isolate_roots(level_poly)]
    except (ArithmeticError, OverflowError, ValueError):
        return {"certified": False, "reason": "exact_level_failure",
                "minima": (), "saddles": ()}
    value = P.eval_at(level_poly, bq)
    if value >= 0:
        return {"certified": False, "reason": "point_not_strictly_inside",
                "minima": (), "saddles": ()}
    for iv in roots:
        if iv.lo <= bq <= iv.hi:
            return {"certified": False, "reason": "level_root_overlap",
                    "minima": (), "saddles": ()}
    left_iv = max((iv for iv in roots if iv.hi < bq),
                  key=lambda iv: iv.hi, default=None)
    right_iv = min((iv for iv in roots if iv.lo > bq),
                   key=lambda iv: iv.lo, default=None)

    inside = []
    for q in enumeration.points:
        qiv = q.interval
        left_inside = left_iv is None or qiv.lo > left_iv.hi
        right_inside = right_iv is None or qiv.hi < right_iv.lo
        if left_inside and right_inside:
            inside.append(q)
            continue
        definitely_left = left_iv is not None and qiv.hi < left_iv.lo
        definitely_right = right_iv is not None and qiv.lo > right_iv.hi
        if definitely_left or definitely_right:
            continue
        return {"certified": False,
                "reason": "critical_level_boundary_overlap",
                "minima": (), "saddles": ()}

    unbounded_sides = []
    if left_iv is None:
        unbounded_sides.append("b_minus_infinity")
    if right_iv is None:
        unbounded_sides.append("b_plus_infinity")
    return {
        "certified": True,
        "reason": None,
        "level_upper": _finite_float_or_none(c),
        "bounded": not unbounded_sides,
        "unbounded_sides": tuple(unbounded_sides),
        "left_boundary": None if left_iv is None else (
            float(left_iv.lo), float(left_iv.hi)),
        "right_boundary": None if right_iv is None else (
            float(right_iv.lo), float(right_iv.hi)),
        "minima": tuple(q for q in inside if q.kind == "min"),
        "saddles": tuple(q for q in inside if q.kind == "saddle"),
    }


def _target_matches(point, target, tolerance):
    return target is not None and _norm2(
        np.asarray(point, dtype=float)-np.asarray(target, dtype=float)
    ) <= tolerance


def _earliest_monotone_certificate(last_index, certificate_at):
    """Find the first certified sample in a terminal monotone suffix.

    ``certificate_at`` must be truthy on a suffix of ``0..last_index``.
    Exponential search makes the cost logarithmic even for the very long
    capture and escape curves produced by stiff examples; bisection then
    places the topological completion at the earliest certified sample.
    """
    best = None
    failed_index = None
    offset = 0
    while True:
        index = max(0, last_index-offset)
        candidate = certificate_at(index)
        if candidate is not None:
            best = candidate
        elif best is not None:
            failed_index = index
            break
        if index == 0:
            break
        offset = 1 if offset == 0 else min(last_index, 2*offset)

    if best is not None and failed_index is not None:
        lo, hi = failed_index+1, best["entry_index"]
        while lo < hi:
            mid = (lo+hi)//2
            candidate = certificate_at(mid)
            if candidate is None:
                lo = mid+1
            else:
                best = candidate
                hi = mid
    return best


def _capture_certificate(m, enumeration, branch, allowed_radius,
                         basin_radii):
    """Find a pre-connector point in a one-minimum bounded sublevel tube."""
    target = branch.diag.get("target")
    if branch.term != "capture" or target is None or len(branch.Y) < 2:
        return {"certified": False, "reason": "missing_capture_target"}
    # Y[-1] is the exact critical endpoint appended after event detection.
    # It is useful for drawing but cannot prove that continuation entered the
    # basin.  Search only measured points preceding that connector.
    matching = [q for q in enumeration.minima if _target_matches(
        (q.a, q.b), target, allowed_radius)]
    minimum = matching[0] if len(matching) == 1 else None
    radius = (0.0 if minimum is None else basin_radii.get(
        (float(minimum.a), float(minimum.b)), 0.0))
    last = len(branch.Y)-2

    def first_true(predicate):
        if not predicate(last):
            return None
        lo, hi = 0, last
        while lo < hi:
            mid = (lo+hi)//2
            if predicate(mid):
                hi = mid
            else:
                lo = mid+1
        return lo

    def exact_tube_at(index):
        inventory = _sublevel_component_inventory(
            m, enumeration, branch.Y[index])
        if (inventory["certified"] and inventory["bounded"]
                and not inventory["saddles"]
                and len(inventory["minima"]) == 1):
            contained = inventory["minima"][0]
            if _target_matches(
                    (contained.a, contained.b), target, allowed_radius):
                return {
                    "certified": True,
                    "reason": None,
                    "method": "exact_level_tube",
                    "entry_index": index,
                    "minimum": (float(contained.a), float(contained.b)),
                    "level_upper": inventory["level_upper"],
                    "b_interval": (inventory["left_boundary"],
                                   inventory["right_boundary"]),
                }
        return None

    # Find the earliest measured point, after the independently certified
    # local graph, in the terminal one-minimum product.  Once descent has
    # entered a bounded sublevel component containing the target minimum and
    # no saddle, every later point remains in that product.  The lowest saddle
    # level above the minimum is only a cheap floating locator; every accepted
    # point is independently proved by ``exact_tube_at``.
    #
    # An older rule selected the halfway level between the minimum and the
    # lowest saddle above it.  That was safe for endpoint naming but much too
    # late for topology completion: independently traced unstable branches
    # entering the same basin can become closer than the polyline resolution
    # thousands of samples before the halfway level.  Their harmless terminal
    # chords then exhausted the contact-event budget.  Starting the exact
    # product at its first certified post-graph sample discharges only
    # unstable/unstable contacts inside the named basin; stable/unstable
    # contacts remain fully audited.  Starting at ``critical_steps`` also
    # avoids repeating exact level-set work inside the materialized graph,
    # whose injectivity is already certified separately.
    if minimum is not None:
        lower = min(last, max(0, int(branch.diag.get(
            "critical_steps", 0))))
        result = exact_tube_at(lower)
        if result is not None:
            return result

        minimum_level = float(m.L(minimum.a, minimum.b))
        saddle_levels = sorted(
            float(m.L(q.a, q.b)) for q in enumeration.saddles
            if float(m.L(q.a, q.b)) > minimum_level)
        if saddle_levels:
            threshold = np.nextafter(saddle_levels[0], -np.inf)

            def below_merging_level(index):
                return float(m.L(
                    float(branch.Y[index, 0]),
                    float(branch.Y[index, 1]))) < threshold

            if below_merging_level(last):
                lo, hi = lower, last
                while lo < hi:
                    mid = (lo+hi)//2
                    if below_merging_level(mid):
                        hi = mid
                    else:
                        lo = mid+1
                locator = lo
                # The strict dyadic upper level may still overlap the saddle
                # at the first floating locator.  Probe forward
                # exponentially until the exact one-minimum inventory closes.
                failed = lower
                offset = 0
                best = None
                while True:
                    index = min(last, locator+offset)
                    candidate = exact_tube_at(index)
                    if candidate is not None:
                        best = candidate
                        break
                    failed = index
                    if index == last:
                        break
                    offset = 1 if offset == 0 else min(
                        last-locator, 2*offset)
                if best is not None:
                    lo, hi = failed+1, best["entry_index"]
                    while lo < hi:
                        mid = (lo+hi)//2
                        candidate = exact_tube_at(mid)
                        if candidate is None:
                            lo = mid+1
                        else:
                            best = candidate
                            hi = mid
                    return best

    # A global minimum can share every positive sublevel component with an
    # end at infinity.  Its independently certified strong-convexity ball is
    # a forward-invariant substitute for the bounded level tube.  Its
    # Euclidean membership predicate is cheap enough for exact bisection.
    if minimum is not None and radius > 0.0:
        center = np.asarray((minimum.a, minimum.b))
        index = first_true(lambda i: _norm2(
            np.asarray(branch.Y[i], dtype=float)-center) < radius)
        if index is not None:
            return {
                "certified": True,
                "reason": None,
                "method": "strictly_convex_ball",
                "entry_index": index,
                "minimum": (float(minimum.a), float(minimum.b)),
                "radius": float(radius),
            }
    return {"certified": False,
            "reason": "no_level_tube_or_convex_capture_ball"}


def _strictly_positive_on_ray(polynomial, start, direction):
    """Exact positivity on [start,+inf) or (-inf,start]."""
    if not polynomial or P.eval_at(polynomial, start) <= 0:
        return False
    # Translate the ray to x>=0.  Nonnegative exact coefficients (with a
    # positive constant) prove positivity immediately and avoid constructing
    # an expensive global Sturm chain for the high-degree funnel polynomials.
    shifted = [
        sum(polynomial[i]*comb(i, j)*start**(i-j)
            for i in range(j, len(polynomial)))
        * (1 if direction > 0 or j % 2 == 0 else -1)
        for j in range(len(polynomial))]
    # This is deliberately the complete production policy for the funnel.
    # Constructing a fresh high-degree global Sturm plan here can dominate an
    # otherwise finished portrait.  Coefficient positivity is exact and
    # cheap; when it does not apply, the total engine declines this optional
    # funnel and returns unresolved rather than entering unbounded exact work.
    return all(coefficient >= 0 for coefficient in shifted)


def _unstable_far_field_funnel(m, branch, index=None):
    """Exact shrinking invariant corridor around the backbone on a b-ray.

    Put r=a/a*(b)-1, h=sign(b)b, and s=hr.  The corridor |s|<=k
    shrinks like 1/|b|, which is necessary because the leading terms of N
    cancel.  After clearing the positive denominator A^3 h^3,

      sign(A^3 h^3 sdot) = sign(S(s,b)),

    where S is the polynomial assembled below and D=A B'-B A'.  Four exact
    polynomial sign tests make b monotone outward and the two boundaries
    point inward on the complete ray.
    """
    if index is None:
        index = len(branch.Y)-1
    endpoint = branch.Y[index]
    try:
        aq, bq = (Fraction.from_float(float(x)) for x in endpoint)
        A, B = m.alpha, m.beta
        Ap, Bp = P.deriv(A), P.deriv(B)
        Aval, Bval = P.eval_at(A, bq), P.eval_at(B, bq)
        if Aval <= 0 or Bval == 0:
            return None
        ratio = aq*Aval/Bval-1
        direction = 1 if float(endpoint[1]) > float(
            branch.diag.get("saddle_b", endpoint[1])) else -1
        hq = Fraction(direction)*bq
        if hq <= 0:
            return None
        scaled_ratio = hq*ratio
        BAp = P.mul(B, Ap)
        D = P.sub(P.mul(A, Bp), BAp)
        A2 = P.mul(A, A)
        A4 = P.mul(A2, A2)
        h = (Fraction(0), Fraction(direction))
        hN = P.mul(h, m.N)
        outward = P.scale(P.mul(B, m.N), Fraction(-direction))

        def scaled_radial(s):
            h_plus_s = P.add(h, (s,))
            shifted = P.add(hN, P.scale(BAp, s))
            first = P.scale(
                P.mul(P.mul(P.mul(h_plus_s, A), B), shifted),
                -direction*s)
            h3 = P.mul(P.mul(h, h), h)
            second = P.scale(P.mul(h3, A4), -2*s)
            third = P.mul(
                P.mul(P.mul(h, P.mul(h_plus_s, h_plus_s)), D), shifted)
            return P.add(P.add(first, second), third)

        accepted_width = None
        for width in (Fraction(1, 2**power)
                      for power in range(48, -1, -1)):
            if abs(scaled_ratio) >= width or width >= hq:
                continue
            robust = P.sub(
                P.mul(hN, hN),
                P.scale(P.mul(BAp, BAp), width*width))
            inward_upper = P.scale(
                scaled_radial(width), Fraction(-1))
            inward_lower = scaled_radial(-width)
            tests = (robust, outward, inward_upper, inward_lower)
            # Endpoint signs are necessary and cheap.  Defer global Sturm
            # counts until a corridor width can at least work locally.
            if not all(P.eval_at(p, bq) > 0 for p in tests):
                continue
            if all(_strictly_positive_on_ray(p, bq, direction)
                   for p in tests):
                accepted_width = width
                break
        if accepted_width is None:
            return None
        width = accepted_width
    except (ArithmeticError, OverflowError, ValueError):
        return None
    return {
        "certified": True,
        "reason": None,
        "method": "exact_backbone_funnel",
        "entry_index": index,
        "end": ("b_plus_infinity" if direction > 0
                else "b_minus_infinity"),
        "scaled_relative_half_width": float(width),
        "endpoint_relative_half_width": float(width/hq),
    }


def _unstable_escape_certificate(m, enumeration, branch, box,
                                 boundary_tolerance):
    """Certify entry into a critical-free sublevel tube with one open end."""
    if branch.kind != "unstable" or branch.term != "box_exit":
        return {"certified": False, "reason": "not_unstable_box_exit"}
    if not _box_exit_crossing(branch.Y, box, boundary_tolerance):
        return {"certified": False, "reason": "no_box_boundary_crossing"}
    last = len(branch.Y)-1

    def certificate_at(index):
        inventory = _sublevel_component_inventory(
            m, enumeration, branch.Y[index])
        if not inventory["certified"]:
            return None
        if inventory["minima"] or inventory["saddles"]:
            return None
        if len(inventory["unbounded_sides"]) != 1:
            return None
        return {
            "certified": True,
            "reason": None,
            "method": "exact_sublevel_tube",
            "entry_index": index,
            "end": inventory["unbounded_sides"][0],
            "level_upper": inventory["level_upper"],
            "b_interval": (inventory["left_boundary"],
                           inventory["right_boundary"]),
        }

    # Begin no earlier than the separately certified local graph.  A global
    # ``below every critical value`` locator is too conservative here: an
    # unbounded component may already contain no critical point while a lower
    # minimum lives in a different component.  Exact inventories are cheap
    # enough to bracket the product directly.  Every retained candidate is
    # independently verified; the search ordering is only an accelerator.
    lower = min(last, max(0, int(branch.diag.get("critical_steps", 0))))
    tube = None
    failed_before_tube = None
    offset = 0
    while True:
        index = max(lower, last-offset)
        candidate = certificate_at(index)
        if candidate is not None:
            tube = candidate
        elif tube is not None:
            failed_before_tube = index
            break
        if index == lower:
            break
        offset = 1 if offset == 0 else min(last-lower, 2*offset)
    if tube is not None and failed_before_tube is not None:
        lo = failed_before_tube
        hi = tube["entry_index"]
        while hi-lo > 1:
            mid = (lo+hi)//2
            candidate = certificate_at(mid)
            if candidate is None:
                lo = mid
            else:
                hi = mid
                tube = candidate
    if tube is not None:
        return tube
    # The shrinking corridor is often already invariant shortly before the
    # trace reaches the box.  Certifying it only at the final sample leaves a
    # small band of numerically indistinguishable same-end chords in the
    # audited prefix.  Probe backwards geometrically and retain the earliest
    # independently verified corridor; no monotonicity assumption about the
    # floating trace is used.
    funnel = None
    failed_before_funnel = None
    offset = 0
    while True:
        index = max(lower, last-offset)
        candidate = _unstable_far_field_funnel(m, branch, index)
        if candidate is not None:
            funnel = candidate
        elif funnel is not None:
            failed_before_funnel = index
            break
        if index == lower:
            break
        offset = 1 if offset == 0 else min(last-lower, 2*offset)
    if funnel is not None and failed_before_funnel is not None:
        # The geometric probes deliberately leave a coarse bracket.  Refine
        # that bracket with independently proved candidates so the completed
        # product begins before the first asymptotic chord-contact band.  The
        # returned corridor is valid even if candidate success were not
        # monotone; monotonicity is used only to find a useful earlier sample.
        lo = failed_before_funnel
        hi = funnel["entry_index"]
        while hi-lo > 1:
            mid = (lo+hi)//2
            candidate = _unstable_far_field_funnel(m, branch, mid)
            if candidate is None:
                lo = mid
            else:
                hi = mid
                funnel = candidate
    if funnel is not None:
        return funnel
    return {"certified": False,
            "reason": "no_sublevel_tube_or_far_field_funnel"}


def _critical_root_polynomial(m, point):
    if point.source == "N":
        return m.N
    if point.source == "B":
        return m.beta
    return m.critical_reduced


def _polynomial_interval(polynomial, interval):
    """Exact Horner interval enclosure over a rational interval."""
    lo = hi = Fraction(0)
    for coefficient in reversed(polynomial):
        products = (lo*interval.lo, lo*interval.hi,
                    hi*interval.lo, hi*interval.hi)
        lo, hi = min(products)+coefficient, max(products)+coefficient
    return lo, hi


def _sign_at_critical_point_python(m, point, polynomial):
    """Certified sign of ``polynomial`` at an isolated critical b-value."""
    interval = point.interval
    root_polynomial = _critical_root_polynomial(m, point)
    for _ in range(80):
        if interval.exact:
            value = P.eval_at(polynomial, interval.lo)
            return (value > 0)-(value < 0)
        lo, hi = _polynomial_interval(polynomial, interval)
        if lo > 0:
            return 1
        if hi < 0:
            return -1
        width = interval.hi-interval.lo
        interval = sturm.refine(
            root_polynomial, interval,
            rel=width/(4*(1+abs(interval.mid))))
    return None


def _sign_at_critical_point(m, point, polynomial):
    """Certified sign at a critical b-value, using the exact C core."""
    integers = P.int_primitive(P.trim(polynomial))
    if not integers:
        return 0
    root_integers = P.int_primitive(P.trim(
        _critical_root_polynomial(m, point)))
    plan = sturm._native_sturm_plan(root_integers) if root_integers else None
    if plan is None:
        return _sign_at_critical_point_python(m, point, polynomial)
    interval = point.interval
    result = plan.sign_polynomial_at_root(
        integers, interval.lo, interval.hi, interval.exact,
        max_bisections=160)
    if result["status"] != 0:
        raise ArithmeticError(
            "native algebraic sign evaluation refused with status "
            f"{result['status']}")
    return result["sign"]


def _stable_escape_certificate(m, enumeration, branch, box,
                               boundary_tolerance):
    """Certify ascent beyond every finite critical level and out of the box."""
    if branch.kind != "stable" or branch.term != "box_exit":
        return {"certified": False, "reason": "not_stable_box_exit"}
    if not _box_exit_crossing(branch.Y, box, boundary_tolerance):
        return {"certified": False, "reason": "no_box_boundary_crossing"}
    cache = {}

    def certificate_at(index):
        if index in cache:
            return cache[index]
        try:
            lower = _strict_level_at_float_point(
                m, branch.Y[index], above=False)
            level_polynomial = P.sub(
                P.scale(m.alpha, m.C-lower), P.mul(m.beta, m.beta))
            signs = [_sign_at_critical_point(
                m, point, level_polynomial) for point in enumeration.points]
        except (ArithmeticError, OverflowError, ValueError):
            cache[index] = None
            return None
        # u(q) < lower exactly when A(q)(C-lower)-B(q)^2 < 0.
        if all(sign == -1 for sign in signs):
            result = {
                "certified": True,
                "reason": None,
                "method": "exact_superlevel_product",
                "entry_index": index,
                "level_lower": _finite_float_or_none(lower),
                "exit_side": _exit_side(branch.Y[-1], box),
            }
        else:
            result = None
        cache[index] = result
        return result

    # The old fixed 128-point lookback placed the certified exterior suffix
    # almost at the compute-box boundary.  Stable branches can contain tens of
    # thousands of samples, and asymptotically adjacent tails then become
    # indistinguishable before the suffix begins.  Search exponentially back
    # through the monotone-ascent trace, then bisect the transition to retain
    # the earliest sampled point whose loss is *exactly* above every finite
    # critical value.
    best = _earliest_monotone_certificate(
        len(branch.Y)-1, certificate_at)
    if best is not None:
        return best
    return {"certified": False,
            "reason": "no_exact_superlevel_escape_point"}


def _exit_side(point, box):
    distances = (abs(point[0]-box[0]), abs(point[0]-box[1]),
                 abs(point[1]-box[2]), abs(point[1]-box[3]))
    return int(np.argmin(distances))


def _on_box_boundary(point, box, tolerance):
    a, b = map(float, point)
    inside = (box[0]-tolerance <= a <= box[1]+tolerance
              and box[2]-tolerance <= b <= box[3]+tolerance)
    return inside and min(abs(a-box[0]), abs(a-box[1]),
                          abs(b-box[2]), abs(b-box[3])) <= tolerance


def _inside_box(point, box, tolerance):
    a, b = map(float, point)
    return (box[0]-tolerance <= a <= box[1]+tolerance
            and box[2]-tolerance <= b <= box[3]+tolerance)


def _box_exit_crossing(curve, box, tolerance):
    """Whether the last computed segment genuinely crosses the trace box."""
    if len(curve) == 0:
        return False
    if _on_box_boundary(curve[-1], box, tolerance):
        return len(curve) >= 2 and _inside_box(
            curve[-2], box, tolerance)
    if len(curve) < 2:
        return False
    p, q = np.asarray(curve[-2], dtype=float), np.asarray(curve[-1], dtype=float)
    if not _inside_box(p, box, tolerance):
        return False
    delta = q-p
    for axis, side, other_lo, other_hi in (
            (0, box[0], box[2], box[3]),
            (0, box[1], box[2], box[3]),
            (1, box[2], box[0], box[1]),
            (1, box[3], box[0], box[1])):
        if delta[axis] == 0.0:
            continue
        t = (side-p[axis])/delta[axis]
        if not -tolerance <= t <= 1.0+tolerance:
            continue
        other = p[1-axis]+t*delta[1-axis]
        if other_lo-tolerance <= other <= other_hi+tolerance:
            return True
    return False


def audit(m, enumeration, branches, box) -> dict:
    """Return an independent topology/FP64 certificate for a portrait."""
    scale = max(1.0, *(abs(x) for x in box))
    predicate_tol = 128*np.finfo(float).eps*scale
    segment_count = sum(max(0, len(br.Y)-1) for br in branches)
    segment_budget = 1000000
    raw_event_budget = 5000
    observed_stable = sum(br.kind == "stable" for br in branches)
    observed_unstable = sum(br.kind == "unstable" for br in branches)
    aborted = [i for i, br in enumerate(branches)
               if br.term not in ("capture", "box_exit")]
    initial_decision = _topology_decision(
        len(enumeration.saddles), len(branches),
        observed_stable, observed_unstable,
        segment_count, segment_budget, 0, raw_event_budget,
        0, 0, 0, 0, bool(aborted))
    branch_inventory = {
        "expected_stable": initial_decision["expected_stable"],
        "observed_stable": observed_stable,
        "expected_unstable": initial_decision["expected_unstable"],
        "observed_unstable": observed_unstable,
        "certified": initial_decision["branch_inventory_certified"],
    }
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
            "certified": False, "reason": "branch_set_incomplete",
        } for i, br in enumerate(branches) if br.kind == "stable"]
        unstable_ends = [{
            "branch": i, "kind": "incomplete", "certified": False,
            "reason": "branch_set_incomplete",
        } for i, br in enumerate(branches) if br.kind == "unstable"]
        return {
            "status": "fp64_unresolved",
            "audit_complete": initial_decision["audit_complete"],
            "resolution_reason": initial_decision["reason"],
            "aborted_branches": aborted,
            "segment_count": segment_count,
            "segment_budget": segment_budget,
            "branch_inventory": branch_inventory,
            "raw_event_count": 0,
            "raw_event_budget": raw_event_budget,
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
            "unstable_ends": unstable_ends,
            "stable_tails": stable_tails,
        }
    budget_exceeded = segment_count > segment_budget
    native_contacts = _native_contact_available()
    trees = ([] if budget_exceeded or native_contacts else
             [_tree(np.asarray(br.Y)) for br in branches])
    critical = np.asarray([(p.a, p.b) for p in enumeration.points])
    allowed_radius = max(1024*np.finfo(float).eps*scale, 1e-11)
    basin_radii = _minimum_basin_radii(m, enumeration)

    unstable_end_by_branch = {}
    stable_tail_by_branch = {}
    for i, branch in enumerate(branches):
        if branch.kind == "unstable":
            if branch.term == "capture":
                certificate = _capture_certificate(
                    m, enumeration, branch, allowed_radius, basin_radii)
                kind = "finite_capture"
            elif branch.term == "box_exit":
                certificate = _unstable_escape_certificate(
                    m, enumeration, branch, box, 16*predicate_tol)
                kind = "infinity_escape"
            else:
                certificate = {"certified": False,
                               "reason": "incomplete_unstable_branch"}
                kind = "incomplete"
            unstable_end_by_branch[i] = {
                "branch": i, "kind": kind, **certificate}
        elif branch.kind == "stable":
            certificate = _stable_escape_certificate(
                m, enumeration, branch, box, 16*predicate_tol)
            asymptote = branch.certs.get("asymptote")
            stable_tail_by_branch[i] = {
                "branch": i,
                "box_exit": branch.term == "box_exit",
                "asymptote_residual": (
                    None if asymptote is None else asymptote["residual"]),
                "superlevel_start": certificate.get("entry_index"),
                **certificate,
            }

    unstable_ends = list(unstable_end_by_branch.values())
    stable_tails = list(stable_tail_by_branch.values())
    terminal_suffixes = []
    for i, br in enumerate(branches):
        endpoint = unstable_end_by_branch.get(i, stable_tail_by_branch.get(i))
        if (br.kind == "unstable" and endpoint is not None
                and endpoint["kind"] == "finite_capture"
                and endpoint["certified"]):
            terminal_suffixes.append({
                "kind": "minimum_sublevel",
                "start": endpoint["entry_index"],
                "terminal": endpoint["minimum"],
            })
        elif (br.kind == "unstable" and endpoint is not None
              and endpoint["kind"] == "infinity_escape"
              and endpoint["certified"]):
            terminal_suffixes.append({
                "kind": "unstable_infinity",
                "start": endpoint["entry_index"],
                "end": endpoint["end"],
            })
        elif (br.kind == "stable" and endpoint is not None
              and endpoint["certified"]):
            terminal_suffixes.append({
                "kind": "stable_infinity",
                "start": endpoint["entry_index"],
                "side": endpoint["exit_side"],
            })
        else:
            terminal_suffixes.append({"kind": None, "start": None})
    # Intersection output can itself be enormous when two failed traces run
    # nearly coincident.  Certification needs the existence and count of such
    # events, not millions of retained Python dictionaries.
    event_sample_limit = 256
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
        root = None if native_contacts else trees[i]
        for si, sj, kind, point in _self_contact_events(
                np.asarray(br.Y), root, predicate_tol):
            suffix = terminal_suffixes[i]
            completed_stable_tail = (
                suffix["kind"] == "stable_infinity"
                and suffix["start"] is not None
                and min(si, sj) >= suffix["start"])
            if completed_stable_tail:
                # Above every finite critical value the gradient flow is a
                # product.  The certified stable suffix is represented by an
                # order-preserving arc to the compactification boundary, not
                # by numerically indistinguishable asymptotic chords.
                continue
            if (kind == "ambiguous" and _strictly_monotone_subarc(
                    br.Y, si, sj)):
                # Exact monotonicity of either polyline coordinate orders the
                # entire intervening parameter interval, so two nonadjacent
                # chords cannot meet.  This discharges dense-output chords
                # below the global predicate floor without an arbitrary
                # adjacency stencil.  Retain every transverse `cross` and
                # every ambiguity whose intervening subarc can turn back.
                continue
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
            events = _pair_contact_events(
                np.asarray(branches[i].Y),
                None if native_contacts else trees[i],
                np.asarray(branches[j].Y),
                None if native_contacts else trees[j], predicate_tol)
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
                same_completed_stable_exterior = (
                    ti["kind"] == tj["kind"] == "stable_infinity"
                    and ti["start"] is not None and tj["start"] is not None
                    and ti["side"] == tj["side"]
                    and si >= ti["start"] and sj >= tj["start"])
                same_completed_unstable_exterior = (
                    ti["kind"] == tj["kind"] == "unstable_infinity"
                    and ti["start"] is not None and tj["start"] is not None
                    and ti["end"] == tj["end"]
                    and si >= ti["start"] and sj >= tj["start"])
                # Unstable curves are basin interiors, not basin boundaries.
                # Once continuation has independently captured both at the
                # same certified critical point, their mutual polyline
                # ordering carries no Morse-complex information and may be
                # rerouted inside that basin.  Crossings with a STABLE
                # separatrix remain forbidden everywhere: those are the
                # topology-changing events this audit is designed to catch.
                # The time-reversed analogue applies to stable tails after
                # exact entry above every finite critical level.  That
                # critical-point-free exterior is a product, so the sampled
                # suffixes are replaced by disjoint, order-preserving arcs to
                # infinity.  All prefixes and every stable/unstable contact
                # remain subject to the scan.
                if (same_sublevel_end or same_completed_stable_exterior
                        or same_completed_unstable_exterior):
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

    decision = _topology_decision(
        len(enumeration.saddles), len(branches),
        observed_stable, observed_unstable,
        segment_count, segment_budget, raw_event_count, raw_event_budget,
        forbidden_count, ambiguous_count,
        sum(not x["certified"] for x in unstable_ends),
        sum(not x["certified"] for x in stable_tails), False)
    # The inventory in the returned ledger is the same native decision used
    # for the terminal status, not a separately reimplemented frontend rule.
    branch_inventory["certified"] = decision["branch_inventory_certified"]
    return {
        "status": ("certified" if decision["certified"]
                   else "fp64_unresolved"),
        "audit_complete": decision["audit_complete"],
        "resolution_reason": decision["reason"],
        "segment_count": segment_count,
        "segment_budget": segment_budget,
        "branch_inventory": branch_inventory,
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
        "unstable_ends": unstable_ends,
        "stable_tails": stable_tails,
    }
