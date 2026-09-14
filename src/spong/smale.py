"""Development orchestrator for fail-closed finite-plane certification.

For a Morse gradient field on the plane, a finite critical-point
nontransversality is a saddle--saddle connection.  In the SPONG family a
descending connection can end only at a lower-loss N-saddle: every B-saddle
has the common level C.  This module enumerates those incidences, transports
validated local stable and unstable launch boxes to one exact regular loss
fibre, and requires their terminal rectangles to be disjoint.

Floating branches are used only to propose tube centres.  The certificate is
carried by the exact-rational local launch and trapping-tube inequalities in
``local_certificate`` and ``hyperelliptic``.  A missing branch, unresolved
loss order, failed launch, failed tube, or overlapping terminal rectangle is
therefore an explicit refusal.

This is intentionally a *finite-plane* certificate.  Identifying and proving
the equilibria of a chosen compactification is a separate obligation; callers
must not promote this status to compactified Morse--Smale without it.

The ordinary local Frobenius launch, its analytic fixed-b section, the
fixed-sheet global tube, and the fixed-b global handoff are native.  The old
piecewise numerical-graph and two-coordinate Python/Fraction rescues are
development-only and disabled by default.  This Python orchestrator is not
yet a browser/mobile production frontend; production callers must wait for
``production_ready`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import time

from . import hyperelliptic, local_certificate, merge_tree, sturm


@dataclass(frozen=True)
class SeparatrixLabel:
    critical_index: int
    critical_source: str
    manifold: str
    orientation: int

    def as_dict(self) -> dict:
        return {
            "critical_index": self.critical_index,
            "critical_source": self.critical_source,
            "manifold": self.manifold,
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class PairExclusion:
    unstable: SeparatrixLabel
    stable: SeparatrixLabel
    level: Fraction
    status: str
    reason: str | None
    decision: object | None = None
    witness: str | None = None

    @property
    def connection_excluded(self) -> bool:
        if self.status != "connection_excluded":
            return False
        return (self.witness == "launch_sublevel_component"
                or (self.decision is not None
                    and self.decision.status == "connection_excluded"))

    def as_dict(self) -> dict:
        return {
            "unstable": self.unstable.as_dict(),
            "stable": self.stable.as_dict(),
            "level": float(self.level),
            "level_exact": (self.level.numerator, self.level.denominator),
            "status": self.status,
            "connection_excluded[VALIDATED]": self.connection_excluded,
            "witness": self.witness,
            "reason": self.reason,
            "decision": (None if self.decision is None
                         else self.decision.as_dict()),
        }


@dataclass(frozen=True)
class FinitePlaneSmaleCertificate:
    status: str
    exact_morse: bool
    model_hypothesis: bool
    records: tuple[PairExclusion, ...]
    ordering_refusals: tuple[tuple[int, int, str], ...]
    source_target_pairs: int
    loss_order_pruned: int
    target_kind_pruned: int
    elapsed_seconds: float

    @property
    def validated(self) -> bool:
        return self.status == "certified_finite_plane_morse_smale"

    @property
    def excluded_count(self) -> int:
        return sum(record.connection_excluded for record in self.records)

    @property
    def unresolved_count(self) -> int:
        return len(self.records)-self.excluded_count+len(self.ordering_refusals)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "engine_maturity": "development_oracle",
            "production_ready": False,
            "scope": "finite critical points in the real plane",
            "finite_plane_morse_smale[VALIDATED]": self.validated,
            "morse[EXACT]": self.exact_morse,
            "model_hypothesis[EXACT]": self.model_hypothesis,
            "candidate_half_branch_pairs": len(self.records),
            "connection_exclusions": self.excluded_count,
            "unresolved": self.unresolved_count,
            "source_target_pairs": self.source_target_pairs,
            "loss_order_pruned": self.loss_order_pruned,
            "target_kind_pruned": self.target_kind_pruned,
            "ordering_refusals": [
                {"source_index": source, "target_index": target,
                 "reason": reason}
                for source, target, reason in self.ordering_refusals],
            "pairs": [record.as_dict() for record in self.records],
            "elapsed_seconds[RESIDUAL]": self.elapsed_seconds,
            "compactified_ends": {
                "status": "not_part_of_this_certificate",
                "reason": ("a compactified Morse--Smale verdict additionally "
                           "requires validated end identification"),
            },
        }


def _label(index, point, stub) -> SeparatrixLabel:
    return SeparatrixLabel(index, point.source, stub.manifold,
                           int(stub.orientation))


def _branch_at_point(branch, point) -> bool:
    value = branch.diag.get("saddle_b")
    if value is None:
        return False
    scale = max(1.0, abs(float(point.b)), abs(float(value)))
    return abs(float(value)-float(point.b)) <= 32.0*math.ulp(scale)


def _launch_sheet(launch) -> int:
    interval = launch.y_interval
    return 1 if interval.lo > 0 else -1 if interval.hi < 0 else 0


def _matching_branch(portrait, point, stub):
    candidates = [
        branch for branch in portrait.branches
        if branch.kind == stub.manifold and _branch_at_point(branch, point)]
    if stub.manifold == "unstable":
        candidates = [branch for branch in candidates
                      if branch.diag.get("unstable_direction")
                      == stub.b_direction]
    else:
        sheet = _launch_sheet(stub.validated_launch)
        candidates = [branch for branch in candidates
                      if branch.diag.get("stable_sign") == sheet]
    return candidates[0] if len(candidates) == 1 else None


def _as_centre(point) -> hyperelliptic.HolonomyCentre:
    return hyperelliptic.HolonomyCentre(point.level, point.b, point.y)


def _decimate(points, limit: int):
    if len(points) <= limit:
        return tuple(points)
    if limit < 2:
        raise ValueError("centre limit must be at least two")
    # The validated launch is tiny and the level field is singular at the
    # saddle.  Retain quadratically more centres near that handoff; uniform
    # index thinning discards precisely the local prefix that lets the first
    # tube slab exclude the critical point.
    indices = []
    for k in range(limit):
        index = (k*k*(len(points)-1))//((limit-1)*(limit-1))
        if not indices or index != indices[-1]:
            indices.append(index)
    if indices[-1] != len(points)-1:
        indices.append(len(points)-1)
    return tuple(points[index] for index in indices)


def _proposal_to_level(model, branch, launch, target: Fraction,
                       centre_limit: int):
    """Lift a floating trace and end it at one exact requested level."""
    direction = int(launch.time_direction)
    if direction not in (-1, 1) or direction*(target-launch.section_level) <= 0:
        return None, "target level does not follow the launch direction"

    previous = hyperelliptic.HolonomyCentre(
        launch.section_level, launch.b_interval.midpoint,
        launch.y_interval.midpoint)
    accepted = []
    for a, b in branch.Y:
        try:
            lifted = hyperelliptic.lift_exact(
                model, Fraction(float(a)), Fraction(float(b)))
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if direction*(lifted.level-previous.level) <= 0:
            continue
        reached = direction*(lifted.level-target) >= 0
        if not reached:
            centre = _as_centre(lifted)
            accepted.append(centre)
            previous = centre
            continue

        if lifted.level == target:
            terminal = _as_centre(lifted)
        else:
            fraction = ((target-previous.level)
                        / (lifted.level-previous.level))
            terminal = hyperelliptic.HolonomyCentre(
                target,
                previous.b+fraction*(lifted.b-previous.b),
                previous.y+fraction*(lifted.y-previous.y))
        accepted.append(terminal)
        return _decimate(accepted, centre_limit), None
    return None, "floating proposal did not reach the common loss fibre"


def _poincare_proposal_prefix(model, point, chart, launch, target: Fraction,
                              *, multiplier: int = 1, nodes: int = 4097,
                              bridge_bits: int = 36):
    """Dense non-load-bearing centres between a graph launch and portrait.

    The validated graph tube proves the launch but its rational knots are far
    too sparse to be a good scalar-tube centreline in the final slab.  Sample
    the floating graph densely at the *same* certified reach.  These points
    carry no proof obligation: the fixed-sheet face tests below either enclose
    the exact orbit around them or refuse.  The section-box midpoint remains
    the sole launch anchor, so no crossing interpolated from this proposal is
    trusted.
    """
    if launch.reach is None:
        return ()
    graph_tube = getattr(launch, "graph_tube", None)
    if graph_tube is not None and graph_tube.validated:
        reach = float(graph_tube.knots[-1].t)
        try:
            physical, _ = chart.graph(
                point.local, int(launch.orientation), n=nodes, reach=reach)
        except (ArithmeticError, FloatingPointError, OverflowError,
                RuntimeError, ValueError):
            physical = [chart.physical(
                point.local, int(launch.orientation)*float(knot.t),
                float(knot.s)) for knot in graph_tube.knots]
    else:
        reach = min(float(chart.desired_reach),
                    multiplier*float(launch.reach))
        if reach < float(launch.reach):
            return ()
        try:
            physical, _ = chart.graph(
                point.local, int(launch.orientation), n=nodes, reach=reach)
        except (ArithmeticError, FloatingPointError, OverflowError,
                RuntimeError, ValueError):
            return ()
    direction = int(launch.time_direction)
    lifted_points = []
    previous_level = None
    for a, b in physical:
        try:
            lifted = hyperelliptic.lift_exact(
                model, Fraction(float(a)), Fraction(float(b)))
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if (previous_level is not None
                and direction*(lifted.level-previous_level) <= 0):
            continue
        lifted_points.append(lifted)
        previous_level = lifted.level
    first_after = next((index for index, lifted in enumerate(lifted_points)
                        if direction*(lifted.level-launch.section_level) > 0),
                       None)
    if first_after is None:
        return ()
    after = lifted_points[first_after]
    if graph_tube is not None and graph_tube.validated:
        crossing = hyperelliptic.HolonomyCentre(
            launch.section_level, launch.b_interval.midpoint,
            launch.y_interval.midpoint)
    elif first_after:
        before = lifted_points[first_after-1]
        fraction = ((launch.section_level-before.level)
                    /(after.level-before.level))
        crossing = hyperelliptic.HolonomyCentre(
            launch.section_level,
            before.b+fraction*(after.b-before.b),
            before.y+fraction*(after.y-before.y))
    else:
        crossing = hyperelliptic.HolonomyCentre(
            launch.section_level, launch.b_interval.midpoint,
            launch.y_interval.midpoint)
    out = []
    for exponent in range(bridge_bits, -1, -1):
        fraction = Fraction(1, 2**exponent)
        centre = hyperelliptic.HolonomyCentre(
            crossing.level+fraction*(after.level-crossing.level),
            crossing.b+fraction*(after.b-crossing.b),
            crossing.y+fraction*(after.y-crossing.y))
        if direction*(centre.level-target) >= 0:
            break
        out.append(centre)
    for lifted in lifted_points[first_after+1:]:
        if direction*(lifted.level-target) >= 0:
            break
        out.append(_as_centre(lifted))
    return tuple(out)


def _merge_proposal_centres(prefix, centres, direction: int):
    ordered = sorted((*prefix, *centres),
                     key=lambda point: direction*point.level)
    out = []
    for point in ordered:
        if out and direction*(point.level-out[-1].level) <= 0:
            continue
        out.append(point)
    return tuple(out)


def _b_parameter_proposal(model, branch, point, chart, launch, section,
                          target: Fraction, *, nodes: int = 4097):
    """Propose a monotone-b centreline extending strictly past ``target``."""
    graph = getattr(launch, "graph_tube", None)
    reach = (graph.knots[-1].t
             if graph is not None and graph.validated and graph.knots
             else launch.reach)
    if reach is None:
        return None, "local Poincare reach is unavailable"
    try:
        physical, _ = chart.graph(
            point.local, int(launch.orientation), n=nodes,
            reach=float(reach))
    except (ArithmeticError, FloatingPointError, OverflowError,
            RuntimeError, ValueError) as exc:
        return None, "b-parameter graph proposal failed: "+str(exc)
    candidates = []
    for a, b in (*physical, *branch.Y):
        try:
            lifted = hyperelliptic.lift_exact(
                model, Fraction(float(a)), Fraction(float(b)))
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if section.b_direction*(lifted.b-section.b) > 0:
            candidates.append(_as_centre(lifted))
    candidates.sort(key=lambda centre: section.b_direction*centre.b)
    out = []
    last_b = section.b
    for centre in candidates:
        if section.b_direction*(centre.b-last_b) <= 0:
            continue
        out.append(centre)
        last_b = centre.b
        if launch.time_direction*(centre.level-target) > 0:
            return tuple(out), None
    return None, "b-parameter proposal did not pass the target loss fibre"


def _certify_half_tube(model, portrait, point, stub, level,
                       centre_limit, tube_options):
    launch = stub.validated_launch
    if launch is None or not launch.validated:
        return None, "local separatrix launch is not validated"
    branch = _matching_branch(portrait, point, stub)
    if branch is None:
        return None, "no unique floating proposal matches the validated launch"
    centres, reason = _proposal_to_level(
        model, branch, launch, level, centre_limit)
    if centres is None:
        return None, reason
    try:
        tube = hyperelliptic.certify_flow_tube_from_launch(
            model, launch, centres, **tube_options)
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
        return None, "tube construction refused: "+str(exc)
    if tube.status != "validated":
        return tube, tube.reason or "holonomy tube did not validate"
    return tube, None


def _sheet_options(tube_options):
    allowed = {
        "max_radius", "max_inflations", "max_endpoint_bits",
        "radius_round_bits", "max_slab_bisections",
    }
    return {key: value for key, value in tube_options.items()
            if key in allowed}


def _certify_half_sheet_tube(model, portrait, point, stub, level,
                             centre_limit, tube_options,
                             allow_development_fallbacks):
    """Compose the Frobenius launch, an optional 2D handoff, and a sheet tube."""
    launch = stub.validated_launch
    if launch is None or not launch.validated:
        return None, "local separatrix launch is not validated"
    branch = _matching_branch(portrait, point, stub)
    if branch is None:
        return None, "no unique floating proposal matches the validated launch"
    # Retain the complete monotone proposal while choosing the handoff.  It is
    # thinned only after the proof coordinate has been selected.
    centres, reason = _proposal_to_level(
        model, branch, launch, level, max(2, len(branch.Y)+1))
    if centres is None:
        return None, reason
    options = _sheet_options(tube_options)
    failures = []

    def scalar_from(start, initial_b, initial_y, sheet, tail, handoff=None):
        proposed = (start, *_decimate(tail, centre_limit))
        try:
            return hyperelliptic.certify_sheet_flow_tube_native(
                model, proposed, initial_b_interval=initial_b,
                initial_y_interval=initial_y, sheet=sheet, handoff=handoff,
                **options)
        except (ArithmeticError, OverflowError, ValueError,
                ZeroDivisionError) as exc:
            return None, "fibre-constrained tube refused: "+str(exc)

    sheet = _launch_sheet(launch)
    start = hyperelliptic.HolonomyCentre(
        launch.section_level, launch.b_interval.midpoint,
        launch.y_interval.midpoint)
    if sheet:
        direct = scalar_from(
            start, launch.b_interval, launch.y_interval, sheet, centres)
        if not isinstance(direct, tuple):
            if direct.status == "validated":
                return direct, None
            failures.append(direct.reason or
                            "direct fibre-constrained tube did not validate")
        else:
            failures.append(direct[1])

    # Near y=0 the fixed-loss sheet has enormous curvature: a perfectly
    # accurate straight (level,b) chord may leave the real fibre between its
    # endpoints.  Cut the already validated local graph at fixed b instead,
    # propagate the exact scalar graph y=y(b), and project that trapping tube
    # back onto the requested exact loss fibre.  Floating graph/portrait
    # points remain proposals only; all load-bearing tests are rational.
    chart = next((candidate for candidate in point.local.poincare
                  if candidate.manifold == stub.manifold), None)
    if chart is not None:
        section = getattr(launch, "b_section", None)
        if section is not None and section.validated:
            b_centres, b_reason = _b_parameter_proposal(
                    model, branch, point, chart, launch, section, level)
            if b_centres is not None:
                start = hyperelliptic.HolonomyCentre(
                    section.level_interval.midpoint, section.b,
                    section.y_interval.midpoint)
                b_options = {key: value for key, value in options.items()
                             if key in {"max_radius", "max_inflations",
                                        "max_endpoint_bits",
                                        "radius_round_bits",
                                        "max_slab_bisections"}}
                try:
                    projected = \
                        hyperelliptic.certify_b_parameter_handoff_native(
                            model,
                            (start, *_decimate(
                                b_centres, max(256, centre_limit))),
                            initial_y_interval=section.y_interval,
                            target_level=level,
                            **b_options)
                except (ArithmeticError, OverflowError, ValueError,
                        ZeroDivisionError) as exc:
                    failures.append(
                        "b-parameter handoff refused: "+str(exc))
                else:
                    if projected.status == "validated":
                        return projected, None
                    failures.append(projected.reason or
                                    "b-parameter handoff did not validate")
            else:
                failures.append(b_reason)
        else:
            failures.append(
                "native Frobenius cone has no regular fixed-b section")

    if not allow_development_fallbacks:
        summary = next((failure for failure in reversed(failures) if failure),
                       "native fixed-sheet charts did not close")
        return None, summary

    # Development-only last-resort handoff in the unconstrained (b,y) chart.
    # Unlike the
    # experimental implicit y(level) chart, this path now proves regularity
    # on every complete tube slab as well as its lateral faces.
    last_handoff = len(centres)-2
    candidates = []
    index = 0
    while index <= last_handoff:
        candidates.append(index)
        index = 2*index+1
    if last_handoff >= 0 and (not candidates or candidates[-1] != last_handoff):
        candidates.append(last_handoff)
    for index in candidates:
        try:
            handoff = hyperelliptic.certify_flow_tube_from_launch(
                model, launch, centres[:index+1], **options)
        except (ArithmeticError, OverflowError, ValueError,
                ZeroDivisionError) as exc:
            failures.append("two-coordinate handoff refused: "+str(exc))
            continue
        if handoff.status != "validated" or not handoff.knots:
            failures.append(handoff.reason or
                            "two-coordinate handoff did not validate")
            continue
        knot = handoff.knots[-1]
        projection = hyperelliptic.certify_fibre_projection(
            model, knot.level, knot.b_interval, knot.y_interval)
        if not projection.validated:
            failures.append(projection.reason or
                            "handoff endpoint does not project to one strict sheet")
            continue
        start = hyperelliptic.HolonomyCentre(
            knot.level, projection.projected_b.midpoint, knot.y)
        scalar = scalar_from(
            start, projection.projected_b, knot.y_interval, projection.sheet,
            centres[index+1:], handoff)
        if not isinstance(scalar, tuple):
            if scalar.status == "validated":
                return scalar, None
            failures.append(scalar.reason or
                            "fibre-constrained tube did not validate")
        else:
            failures.append(scalar[1])
    summary = next((failure for failure in reversed(failures) if failure),
                   "no safe fixed-sheet handoff was found")
    return None, summary


def _common_level(sequence, model, source, target):
    forward, reverse = [], []
    for level in sequence.levels:
        source_sign = merge_tree.value_sign(model, source, level)
        target_sign = merge_tree.value_sign(model, target, level)
        if source_sign == 1 and target_sign == -1:
            forward.append(level)
        elif source_sign == -1 and target_sign == 1:
            reverse.append(level)
    if forward:
        midpoint = 0.5*(float(model.L(source.a, source.b))
                        + float(model.L(target.a, target.b)))
        return min(forward, key=lambda level: abs(float(level)-midpoint)), 1
    if reverse:
        return min(reverse), -1
    return None, 0


def _materialize_required_launches(model, enumeration, admissible):
    """Certify only the local germs occurring in admissible incidences."""
    if any(point.kind == "saddle" and not point.stubs
           for point in enumeration.points):
        enumeration = sturm.materialize_stubs(model, enumeration)
    required = {
        (source_index, "unstable")
        for source_index, _target_index, _level in admissible
    } | {
        (target_index, "stable")
        for _source_index, target_index, _level in admissible
    }
    points = []
    for index, point in enumerate(enumeration.points):
        if point.kind != "saddle":
            points.append(point)
            continue
        charts = {chart.manifold: chart for chart in point.local.poincare}
        stubs = []
        for stub in point.stubs:
            if ((index, stub.manifold) in required
                    and stub.validated_launch is None):
                launch = local_certificate.certify_poincare_launch_native(
                    model, point, charts[stub.manifold], stub.orientation)
                stub = replace(stub, validated_launch=launch)
            stubs.append(stub)
        points.append(replace(point, stubs=tuple(stubs)))
    return replace(enumeration, points=tuple(points))


def _certified_b_direction(point, stub, launch):
    chart = next((candidate for candidate in point.local.poincare
                  if candidate.manifold == "unstable"), None)
    if chart is None or launch.cone_slope is None:
        return 0
    v10 = Fraction(float(chart.frame[1][0]))
    v11 = Fraction(float(chart.frame[1][1]))
    slope = launch.cone_slope
    if getattr(launch, "cone_power", 1) == 2:
        tangent = getattr(launch, "tangent_slope_interval", None)
        if tangent is None:
            tangent = hyperelliptic.RationalInterval.point(0)
        transverse = tangent+hyperelliptic.RationalInterval(
            -slope*launch.reach, slope*launch.reach)
    else:
        transverse = hyperelliptic.RationalInterval(-slope, slope)
    coefficient = (hyperelliptic.RationalInterval.point(
        int(stub.orientation)*v10)+v11*transverse)
    return 1 if coefficient.lo > 0 else -1 if coefficient.hi < 0 else 0


def _launch_component(model, enumeration, point, stub):
    """Component entered just after a validated decreasing-loss launch."""
    launch = stub.validated_launch
    if (launch is None or not launch.validated
            or launch.time_direction != -1):
        return None, "unstable local launch is not validated"
    direction = _certified_b_direction(point, stub, launch)
    if direction == 0 or direction != stub.b_direction:
        return None, "local cone does not certify the unstable b-direction"
    level = launch.section_level
    try:
        saddle_gap = merge_tree.critical_gap_index(
            model, level, point)
        components = merge_tree.components_at(model, enumeration, level)
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
        return None, "launch component inventory refused: "+str(exc)
    gap = saddle_gap+direction
    component = next((candidate for candidate in components
                      if candidate.gap == gap), None)
    if component is None:
        return None, "launch sublevel gap has no component inventory"
    return component, None


def certify_finite_plane(model, portrait, *, enumeration=None,
                         centre_limit: int = 192,
                         tube_options: dict | None = None,
                         allow_development_fallbacks: bool = False
                         ) -> FinitePlaneSmaleCertificate:
    """Exclude every admissible finite saddle connection, or refuse.

    ``portrait`` supplies floating centre proposals only.  Its audit status is
    neither read nor trusted.  Pass a pre-materialized validated enumeration to
    amortize the exact local graph certificates across calls.
    """
    started = time.perf_counter()
    e = portrait.enumeration if enumeration is None else enumeration
    if not e.morse or not e.psi_positive:
        status = "certified_non_morse" if not e.morse else "unresolved"
        return FinitePlaneSmaleCertificate(
            status, bool(e.morse), bool(e.psi_positive), (), (), 0, 0, 0,
            time.perf_counter()-started)
    options = {
        "max_inflations": 48,
        "max_slab_bisections": 10,
        # The exact Frobenius-to-fixed-b handoff can transiently exceed the
        # older 16k Fraction-oracle ceiling during its first affine face.
        # This remains a hard, caller-overridable backend work bound.
        "max_endpoint_bits": 32768,
    }
    if tube_options:
        options.update(tube_options)
    sequence = merge_tree.separating_levels(model, e)
    indexed = list(enumerate(e.points))
    records = []
    ordering_refusals = []
    source_target_pairs = 0
    loss_order_pruned = 0
    target_kind_pruned = 0
    tube_cache = {}
    sheet_tube_cache = {}
    admissible = []

    for source_index, source in indexed:
        if source.kind != "saddle":
            continue
        for target_index, target in indexed:
            if target is source or target.kind != "saddle":
                continue
            if target.source != "N":
                target_kind_pruned += 1
                continue
            level, order = _common_level(sequence, model, source, target)
            if order < 0:
                loss_order_pruned += 1
                continue
            if order == 0 or level is None:
                ordering_refusals.append((
                    source_index, target_index,
                    "exact critical-loss order was not separated"))
                continue
            source_target_pairs += 1
            admissible.append((source_index, target_index, level))

    if not admissible:
        status = ("unresolved" if ordering_refusals else
                  "certified_finite_plane_morse_smale")
        return FinitePlaneSmaleCertificate(
            status, True, True, (), tuple(ordering_refusals),
            source_target_pairs, loss_order_pruned, target_kind_pruned,
            time.perf_counter()-started)

    try:
        e = _materialize_required_launches(model, e, admissible)
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        return FinitePlaneSmaleCertificate(
            "unresolved", True, True, (), tuple(ordering_refusals)+(
                (-1, -1,
                 "validated local launch materialization failed"),),
            source_target_pairs, loss_order_pruned, target_kind_pruned,
            time.perf_counter()-started)

    for source_index, target_index, level in admissible:
        source, target = e.points[source_index], e.points[target_index]
        unstable = [stub for stub in source.stubs
                    if stub.manifold == "unstable"]
        stable = [stub for stub in target.stubs
                  if stub.manifold == "stable"]
        if len(unstable) != 2 or len(stable) != 2:
            ordering_refusals.append((
                source_index, target_index,
                "complete stable/unstable launch inventory is unavailable"))
            continue
        for unstable_stub in unstable:
            component, component_reason = _launch_component(
                model, e, source, unstable_stub)
            for stable_stub in stable:
                ulabel = _label(source_index, source, unstable_stub)
                slabel = _label(target_index, target, stable_stub)
                if (component is not None
                        and target_index not in component.saddles):
                    records.append(PairExclusion(
                        ulabel, slabel, level, "connection_excluded",
                        ("target N-saddle is outside the exact sublevel "
                         "component entered by the unstable launch"),
                        witness="launch_sublevel_component"))
                    continue
                ukey = (source_index, "unstable",
                        unstable_stub.orientation, level)
                skey = (target_index, "stable",
                        stable_stub.orientation, level)
                if ukey not in sheet_tube_cache:
                    sheet_tube_cache[ukey] = _certify_half_sheet_tube(
                        model, portrait, source, unstable_stub, level,
                        centre_limit, options, allow_development_fallbacks)
                if skey not in sheet_tube_cache:
                    sheet_tube_cache[skey] = _certify_half_sheet_tube(
                        model, portrait, target, stable_stub, level,
                        centre_limit, options, allow_development_fallbacks)
                unstable_sheet, unstable_sheet_reason = \
                    sheet_tube_cache[ukey]
                stable_sheet, stable_sheet_reason = sheet_tube_cache[skey]
                if (unstable_sheet is not None and stable_sheet is not None
                        and unstable_sheet.status == "validated"
                        and stable_sheet.status == "validated"):
                    decision = \
                        hyperelliptic.certify_sheet_connection_exclusion(
                            unstable_sheet, stable_sheet)
                    if decision.status == "connection_excluded":
                        records.append(PairExclusion(
                            ulabel, slabel, level, decision.status,
                            decision.reason, decision,
                            witness="common_fibre_constrained_intervals"))
                        continue
                # The branch-point-regular two-coordinate chart remains a
                # complete fallback.  The scalar chart is deliberately strict
                # and may refuse a trajectory that crosses y=0 before the
                # common fibre.
                if ukey not in tube_cache:
                    tube_cache[ukey] = _certify_half_tube(
                        model, portrait, source, unstable_stub, level,
                        centre_limit, options)
                if skey not in tube_cache:
                    tube_cache[skey] = _certify_half_tube(
                        model, portrait, target, stable_stub, level,
                        centre_limit, options)
                unstable_tube, unstable_reason = tube_cache[ukey]
                stable_tube, stable_reason = tube_cache[skey]
                if (unstable_tube is None or stable_tube is None
                        or unstable_tube.status != "validated"
                        or stable_tube.status != "validated"):
                    reason = "; ".join(x for x in (
                        component_reason,
                        unstable_sheet_reason, stable_sheet_reason,
                        unstable_reason, stable_reason)
                        if x)
                    records.append(PairExclusion(
                        ulabel, slabel, level, "unresolved", reason))
                    continue
                decision = hyperelliptic.certify_connection_exclusion(
                    model, unstable_tube, stable_tube)
                records.append(PairExclusion(
                    ulabel, slabel, level, decision.status,
                    decision.reason, decision,
                    witness="common_fibre_rectangles"))

    unresolved = (bool(ordering_refusals)
                  or any(not record.connection_excluded for record in records))
    status = ("unresolved" if unresolved else
              "certified_finite_plane_morse_smale")
    return FinitePlaneSmaleCertificate(
        status, True, True, tuple(records), tuple(ordering_refusals),
        source_target_pairs, loss_order_pruned, target_kind_pruned,
        time.perf_counter()-started)
