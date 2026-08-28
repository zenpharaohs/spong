"""Loss-level order sweep for pairwise topology contacts.

The contact frontend asks whether sampled *chords* cross.  This module then
asks the invariant question: on regular loss levels, does the resolved
transverse order of two branches reverse?

For a regular value c, orient {L=c} by J grad(L).  A branch with monotone
loss meets that level once, so two branches have signed separation

    delta(c) = <J grad(L) / |grad(L)|, x_j(c) - x_i(c)>.

The order is resolved only when |delta| exceeds ``threshold`` times the
combined interpolation/predicate allowance.  A candidate contact is a
reportable root only when its nearest resolved witnesses on the two sides
have opposite signs inside one critical-free loss slab.  Same-sign witnesses
prove that the chord contact did not reverse the order.  A missing witness is
discharged only by a compatible certified terminal suffix; otherwise it
remains unresolved.

The established chord verdict remains available as an explicit comparison
policy, and shadow mode records both decisions without changing that verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LossTrace:
    """One branch represented as a single-valued function of loss."""

    levels: np.ndarray
    points: np.ndarray
    errors: np.ndarray
    transversality: np.ndarray
    scales: np.ndarray
    dropped_nonmonotone: int


@dataclass(frozen=True)
class OrderProfile:
    """Pairwise signed order and its local resolution scale."""

    levels: np.ndarray
    delta: np.ndarray
    error: np.ndarray
    ratio: np.ndarray
    valid: np.ndarray
    transversality: np.ndarray
    arc: np.ndarray
    scale: np.ndarray
    first_dropped_nonmonotone: int
    second_dropped_nonmonotone: int


def sagitta_allowance(Y) -> np.ndarray:
    """Local chord-to-arc display allowance used by the current audit."""
    points = np.asarray(Y, dtype=float)
    steps = np.diff(points, axis=0)
    count = len(steps)
    if count == 0:
        return np.zeros(0)
    chord = np.hypot(steps[:, 0], steps[:, 1])
    turn = np.zeros(count)
    if count >= 2:
        first, second = steps[:-1], steps[1:]
        cross = first[:, 0]*second[:, 1] - first[:, 1]*second[:, 0]
        dot = first[:, 0]*second[:, 0] + first[:, 1]*second[:, 1]
        angle = np.abs(np.arctan2(cross, dot))
        turn[:-1] = np.maximum(turn[:-1], angle)
        turn[1:] = np.maximum(turn[1:], angle)
    return chord*turn/8.0


def loss_trace(m, branch) -> LossTrace:
    """Put a sampled branch in strictly increasing loss order.

    Exact gradient trajectories have strictly monotone loss away from their
    critical endpoints.  Equal or reversed binary64 levels cannot serve as a
    coordinate, so the smallest strictly increasing subsequence is retained
    and the number dropped is reported as a diagnostic rather than hidden.
    """
    points = np.asarray(branch.Y, dtype=float)
    if len(points) == 0:
        return LossTrace(np.zeros(0), np.empty((0, 2)), np.zeros(0),
                         np.zeros(0), np.zeros(0), 0)
    levels = np.asarray(m.L(points[:, 0], points[:, 1]), dtype=float)
    sagitta = sagitta_allowance(points)
    errors = np.zeros(len(points))
    if len(sagitta):
        errors[:-1] = np.maximum(errors[:-1], sagitta)
        errors[1:] = np.maximum(errors[1:], sagitta)
    scales = np.zeros(len(points))
    chord = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    if len(chord):
        scales[:-1] = np.maximum(scales[:-1], chord)
        scales[1:] = np.maximum(scales[1:], chord)

    # Loss is a well-conditioned section coordinate only while the sampled
    # curve crosses its level sets transversely.  An exact gradient trajectory
    # has tangent parallel to grad(L), hence alignment one.  In an unresolved
    # asymptotic tail the samples can march almost *along* a level set; trying
    # to invert loss there amplifies tiny level errors into alternating order.
    tangent = np.zeros_like(points)
    if len(points) >= 2:
        tangent[0] = points[1]-points[0]
        tangent[-1] = points[-1]-points[-2]
    if len(points) >= 3:
        tangent[1:-1] = points[2:]-points[:-2]
    grad = np.asarray(m.gradL(points[:, 0], points[:, 1]), dtype=float).T
    denom = np.hypot(tangent[:, 0], tangent[:, 1]) * np.hypot(
        grad[:, 0], grad[:, 1])
    transversality = np.zeros(len(points))
    good = np.isfinite(denom) & (denom > 0.0)
    transversality[good] = np.abs(np.einsum(
        "ij,ij->i", tangent[good], grad[good]))/denom[good]

    if len(levels) >= 2 and levels[-1] < levels[0]:
        levels = levels[::-1]
        points = points[::-1]
        errors = errors[::-1]
        transversality = transversality[::-1]
        scales = scales[::-1]

    keep = np.ones(len(levels), dtype=bool)
    last = levels[0]
    for k in range(1, len(levels)):
        if not np.isfinite(levels[k]) or not levels[k] > last:
            keep[k] = False
        else:
            last = levels[k]
    return LossTrace(levels[keep], points[keep], errors[keep],
                     transversality[keep], scales[keep],
                     int(np.count_nonzero(~keep)))


def order_profile(m, first: LossTrace, second: LossTrace,
                  predicate_tolerance: float,
                  min_transversality: float = 0.25) -> OrderProfile:
    """Signed transverse order on the union of the two loss grids."""
    if not len(first.levels) or not len(second.levels):
        empty = np.zeros(0)
        return OrderProfile(empty, empty, empty, empty,
                            np.zeros(0, dtype=bool),
                            empty, empty, empty,
                            first.dropped_nonmonotone,
                            second.dropped_nonmonotone)
    low = max(first.levels[0], second.levels[0])
    high = min(first.levels[-1], second.levels[-1])
    if not high > low:
        empty = np.zeros(0)
        return OrderProfile(empty, empty, empty, empty,
                            np.zeros(0, dtype=bool),
                            empty, empty, empty,
                            first.dropped_nonmonotone,
                            second.dropped_nonmonotone)
    levels = np.unique(np.concatenate((
        first.levels[(first.levels >= low) & (first.levels <= high)],
        second.levels[(second.levels >= low) & (second.levels <= high)])))
    p = np.column_stack((
        np.interp(levels, first.levels, first.points[:, 0]),
        np.interp(levels, first.levels, first.points[:, 1])))
    q = np.column_stack((
        np.interp(levels, second.levels, second.points[:, 0]),
        np.interp(levels, second.levels, second.points[:, 1])))
    midpoint_curve = 0.5*(p+q)
    arc = np.zeros(len(levels))
    if len(levels) >= 2:
        arc[1:] = np.cumsum(np.hypot(
            np.diff(midpoint_curve[:, 0]), np.diff(midpoint_curve[:, 1])))
    scale = np.maximum(
        np.interp(levels, first.levels, first.scales),
        np.interp(levels, second.levels, second.scales))
    transversality = np.minimum(
        np.interp(levels, first.levels, first.transversality),
        np.interp(levels, second.levels, second.transversality))
    # Conditioning of level inversion deteriorates like 1/cos(angle) as a
    # chord turns toward the level-set tangent.  Below the explicit cutoff it
    # is not merely imprecise but the wrong coordinate, so mark it unresolved.
    condition = np.maximum(transversality, np.finfo(float).eps)
    error = (np.interp(levels, first.levels, first.errors)
             + np.interp(levels, second.levels, second.errors)
             + float(predicate_tolerance))/condition
    midpoint = 0.5*(p+q)
    grad = np.asarray(m.gradL(midpoint[:, 0], midpoint[:, 1]),
                      dtype=float).T
    norm = np.hypot(grad[:, 0], grad[:, 1])
    valid = (np.isfinite(norm) & (norm > 0.0)
             & np.isfinite(error) & (error > 0)
             & (transversality >= min_transversality))
    tangent = np.column_stack((-grad[:, 1], grad[:, 0]))
    tangent[valid] /= norm[valid, None]
    delta = np.full(len(levels), np.nan)
    delta[valid] = np.einsum("ij,ij->i", tangent[valid], (q-p)[valid])
    ratio = np.zeros(len(levels))
    ratio[valid] = np.abs(delta[valid])/error[valid]
    return OrderProfile(levels, delta, error, ratio, valid, transversality,
                        arc, scale,
                        first.dropped_nonmonotone,
                        second.dropped_nonmonotone)


def classify_signed_profile(levels, delta, error, event_levels,
                            *, threshold: float = 4.0,
                            critical_levels: Sequence[float] = (),
                            terminal_ok: Sequence[bool] | None = None,
                            terminal_sides: Sequence[str | None] | None = None,
                            coordinates=None, resolution_scale=None,
                            persistence_factor: float = 2.0,
                            event_intervals=None,
                            ) -> dict:
    """Classify candidate zeros from a precomputed signed-order profile.

    This pure helper is also the mathematical oracle for synthetic tests.
    Events sharing the same resolved witness pair represent one root cluster.
    """
    levels = np.asarray(levels, dtype=float)
    delta = np.asarray(delta, dtype=float)
    error = np.asarray(error, dtype=float)
    events = np.asarray(event_levels, dtype=float)
    intervals = ([(float(c), float(c)) for c in events]
                 if event_intervals is None else
                 [(float(lo), float(hi)) for lo, hi in event_intervals])
    if len(intervals) != len(events):
        raise ValueError("event_intervals must match event_levels")
    resolved, persistent = _persistent_order_masks(
        levels, delta, error, threshold=threshold,
        coordinates=coordinates, resolution_scale=resolution_scale,
        persistence_factor=persistence_factor)

    resolved_indices = np.flatnonzero(persistent)
    resolved_levels = levels[resolved_indices]

    terminal = ([False]*len(events) if terminal_ok is None
                else [bool(x) for x in terminal_ok])
    sides = ([None]*len(events) if terminal_sides is None
             else list(terminal_sides))
    roots: dict[tuple[int, int], dict] = {}
    counts = {"same_order": 0, "terminal": 0,
              "critical_transition": 0, "unresolved": 0}
    event_classes = []
    critical = sorted(float(c) for c in critical_levels if np.isfinite(c))
    for event_index, c in enumerate(events):
        event_low, event_high = intervals[event_index]
        if not len(levels) or not np.isfinite(c):
            kind = "terminal" if terminal[event_index] else "unresolved"
            counts[kind] += 1
            event_classes.append(kind)
            continue
        critical_floor = 32*np.finfo(float).eps*max(1.0, abs(c))
        if any(abs(c-value) <= critical_floor for value in critical):
            counts["critical_transition"] += 1
            event_classes.append("critical_transition")
            continue
        # Order preservation applies on one critical-free slab only.  Search
        # for witnesses inside the component of R \ critical_values that
        # contains c; a distant sign on the far side of a saddle is not a
        # witness for this candidate.
        critical_pos = int(np.searchsorted(critical, c))
        slab_low = (-np.inf if critical_pos == 0
                    else critical[critical_pos-1])
        slab_high = (np.inf if critical_pos == len(critical)
                     else critical[critical_pos])
        slab_begin = int(np.searchsorted(
            resolved_levels, slab_low, side="right"))
        slab_end = int(np.searchsorted(
            resolved_levels, slab_high, side="left"))
        # Strictly bracket the proposed root.  If c lies between two sampled
        # levels, the level immediately to its right is itself an eligible
        # witness; if c equals a grid level, that level is excluded from both
        # sides so a candidate can never witness its own sign.
        cut_left = int(np.searchsorted(
            resolved_levels, event_low, side="left"))
        cut_right = int(np.searchsorted(
            resolved_levels, event_high, side="right"))
        missing_left = cut_left <= slab_begin
        missing_right = cut_right >= slab_end
        if missing_left or missing_right:
            side = sides[event_index]
            terminal_side_missing = (
                terminal[event_index]
                or (side == "lower" and missing_left)
                or (side == "upper" and missing_right))
            kind = "terminal" if terminal_side_missing else "unresolved"
            counts[kind] += 1
            event_classes.append(kind)
            continue
        left = int(resolved_indices[cut_left-1])
        right = int(resolved_indices[cut_right])
        lo, hi = float(levels[left]), float(levels[right])
        if np.signbit(delta[left]) == np.signbit(delta[right]):
            counts["same_order"] += 1
            event_classes.append("same_order")
            continue

        key = (int(left), int(right))
        if key not in roots:
            span = hi-lo
            roots[key] = {
                "loss_bracket": (lo, hi),
                "delta_bracket": (float(delta[left]), float(delta[right])),
                "error_bracket": (float(error[left]), float(error[right])),
                "resolution_margin": float(min(
                    abs(delta[left])/error[left],
                    abs(delta[right])/error[right])),
                "slope": (None if span == 0.0 else
                          float((delta[right]-delta[left])/span)),
                "candidate_count": 0,
            }
        roots[key]["candidate_count"] += 1
        event_classes.append("resolved_root")
    return {
        "threshold": float(threshold),
        "persistence_factor": float(persistence_factor),
        "pointwise_resolved_count": int(np.count_nonzero(resolved)),
        "persistent_resolved_count": int(np.count_nonzero(persistent)),
        "candidate_count": int(len(events)),
        "root_count": int(len(roots)),
        "roots": list(roots.values()),
        "same_order_count": counts["same_order"],
        "terminal_count": counts["terminal"],
        "critical_transition_count": counts["critical_transition"],
        "unresolved_count": counts["unresolved"],
        "event_classes": event_classes,
    }


def _persistent_order_masks(levels, delta, error, *, threshold,
                            coordinates=None, resolution_scale=None,
                            persistence_factor=2.0):
    """Pointwise and scale-persistent resolved-order masks."""
    levels = np.asarray(levels, dtype=float)
    delta = np.asarray(delta, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = np.isfinite(delta) & np.isfinite(error) & (error > 0)
    resolved = valid & (np.abs(delta) >= threshold*error)
    persistent = resolved.copy()
    if coordinates is None or resolution_scale is None or not len(levels):
        return resolved, persistent
    coordinate = np.asarray(coordinates, dtype=float)
    scale = np.asarray(resolution_scale, dtype=float)
    if len(coordinate) != len(levels) or len(scale) != len(levels):
        raise ValueError("coordinates and resolution_scale must match levels")
    persistent[:] = False
    k = 0
    while k < len(levels):
        if not resolved[k]:
            k += 1
            continue
        sign = np.signbit(delta[k])
        end = k+1
        while (end < len(levels) and resolved[end]
               and np.signbit(delta[end]) == sign):
            end += 1
        left = (coordinate[k] if k == 0 else
                0.5*(coordinate[k-1]+coordinate[k]))
        right = (coordinate[end-1] if end == len(levels) else
                 0.5*(coordinate[end-1]+coordinate[end]))
        run_scale = float(np.max(scale[k:end]))
        if right-left >= persistence_factor*run_scale:
            persistent[k:end] = True
        k = end
    return resolved, persistent


def _terminal_side(suffixes, i: int, j: int,
    tolerance: float) -> str | None:
    if not suffixes or i >= len(suffixes) or j >= len(suffixes):
        return None
    first, second = suffixes[i], suffixes[j]
    if first.get("kind") != second.get("kind"):
        return None
    if first["kind"] == "minimum_sublevel":
        p = np.asarray(first.get("terminal"), dtype=float)
        q = np.asarray(second.get("terminal"), dtype=float)
        distance = float(np.hypot(*(p-q)))
        return "lower" if distance <= tolerance else None
    if first["kind"] == "stable_infinity":
        return ("upper" if first.get("side") == second.get("side") else None)
    if first["kind"] == "unstable_infinity":
        return ("lower" if first.get("end") == second.get("end") else None)
    return None


def _contact_clusters(items, adjacency: int = 2):
    """Connected contact bands in the two branch-parameter indices."""
    count = len(items)
    parent = list(range(count))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    segments = [tuple(int(x) for x in item["segments"]) for item in items]
    for a in range(count):
        si, sj = segments[a]
        for b in range(a+1, count):
            ti, tj = segments[b]
            if abs(si-ti) <= adjacency and abs(sj-tj) <= adjacency:
                union(a, b)
    groups = {}
    for k in range(count):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _merge_clusters_without_persistent_gap(clusters, event_levels,
                                           profile: OrderProfile,
                                           threshold: float):
    """Missing chord events do not split a band without resolved order."""
    if len(clusters) <= 1:
        return clusters
    _, persistent = _persistent_order_masks(
        profile.levels, profile.delta, profile.error, threshold=threshold,
        coordinates=profile.arc, resolution_scale=profile.scale)
    witness_levels = profile.levels[persistent]
    ordered = sorted(
        (list(cluster) for cluster in clusters),
        key=lambda group: min(event_levels[k] for k in group))
    merged = [ordered[0]]
    for group in ordered[1:]:
        previous = merged[-1]
        left = max(event_levels[k] for k in previous)
        right = min(event_levels[k] for k in group)
        if right < left:
            left, right = right, left
        begin = int(np.searchsorted(witness_levels, left, side="right"))
        end = int(np.searchsorted(witness_levels, right, side="left"))
        if begin == end:
            previous.extend(group)
        else:
            merged.append(group)
    return merged


def _cluster_extent(branches, i, j, items, indices):
    """Band diameter in units of its largest local input chord."""
    normalized = []
    details = []
    for branch_index, which in ((i, 0), (j, 1)):
        Y = np.asarray(branches[branch_index].Y, dtype=float)
        chord = np.hypot(np.diff(Y[:, 0]), np.diff(Y[:, 1]))
        segment_indices = [int(items[k]["segments"][which]) for k in indices]
        lo, hi = min(segment_indices), max(segment_indices)
        hi_vertex = min(len(Y)-1, hi+1)
        extent = float(np.sum(chord[lo:hi_vertex]))
        local = float(np.max(chord[lo:hi_vertex])) if hi_vertex > lo else 0.0
        ratio = (float("inf") if local == 0.0 and extent > 0.0 else
                 0.0 if local == 0.0 else extent/local)
        normalized.append(ratio)
        details.append({"segments": (lo, hi), "arc_extent": extent,
                        "local_chord": local})
    return max(normalized), details


def _cluster_touches_terminal(suffixes, i, j, items, indices,
                              branches, tolerance,
                              gap_factor: float) -> bool:
    if _terminal_side(suffixes, i, j, tolerance) is None:
        return False
    first, second = suffixes[i], suffixes[j]
    if first.get("start") is None or second.get("start") is None:
        return False
    for branch_index, which, suffix in (
            (i, 0, first), (j, 1, second)):
        Y = np.asarray(branches[branch_index].Y, dtype=float)
        chord = np.hypot(np.diff(Y[:, 0]), np.diff(Y[:, 1]))
        last_contact = max(int(items[k]["segments"][which]) for k in indices)
        start = int(suffix["start"])
        if start <= last_contact:
            continue
        lo, hi = last_contact+1, min(start, len(chord))
        gap = float(np.sum(chord[lo:hi]))
        neighborhood_lo = max(0, last_contact-2)
        neighborhood_hi = min(len(chord), start+2)
        local = (float(np.max(chord[neighborhood_lo:neighborhood_hi]))
                 if neighborhood_hi > neighborhood_lo else 0.0)
        if local == 0.0 or gap > gap_factor*local:
            return False
    return True


def pair_contact_candidates(branches, critical_points, box,
                            predicate_tolerance: float,
                            limit: int = 50000):
    """Collect preterminal pair contacts for diagnostics and qualification.

    This is the common frontend used by production-shadow probes and by the
    deliberately uncertified integrator comparison.  It applies only the
    representation-independent exclusions shared by both: certified local
    germs when present, and the allowed meeting at an exact critical point.
    Terminal suffixes are intentionally not trimmed here; ``classify_contacts``
    must account for every returned event.
    """
    if limit < 1:
        raise ValueError("contact limit must be positive")
    from . import topology

    tolerance = float(predicate_tolerance)
    sagittae = [topology._sagitta_bounds(branch.Y) for branch in branches]
    native = topology._native_contact_available()
    trees = ([] if native else
             [topology._tree(np.asarray(branch.Y)) for branch in branches])
    critical = np.asarray(
        [(point.a, point.b) for point in critical_points], dtype=float)
    scale = max(1.0, *(abs(float(value)) for value in box))
    allowed_radius = max(1024*np.finfo(float).eps*scale, 1e-11)
    contacts = []
    for i in range(len(branches)):
        if len(branches[i].Y) < 2:
            continue
        for j in range(i+1, len(branches)):
            if len(branches[j].Y) < 2:
                continue
            events = topology._pair_contact_events(
                np.asarray(branches[i].Y),
                None if native else trees[i],
                np.asarray(branches[j].Y),
                None if native else trees[j],
                tolerance, sagittae[i], sagittae[j])
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
                    continue
                location = np.asarray(point)
                if len(critical) and np.min(topology._row_norm2(
                        critical-location)) <= allowed_radius:
                    continue
                contacts.append({
                    "branches": (i, j), "segments": (si, sj),
                    "kind": kind, "point": point,
                })
                if len(contacts) >= limit:
                    return contacts, True
    return contacts, False


def classify_contacts(m, enumeration, branches, contacts: Iterable[dict],
                      predicate_tolerance: float, *, threshold: float = 4.0,
                      terminal_suffixes: Sequence[dict] = (),
                      terminal_tolerance: float = 1e-11,
                      isolation_factor: float = 8.0,
                      terminal_gap_factor: float = 16.0,
                      cluster_gap_segments: int = 16) -> dict:
    """Run the experimental order sweep on proposed contact events."""
    grouped: dict[tuple[int, int], list[dict]] = {}
    for item in contacts:
        pair = tuple(int(x) for x in item["branches"])
        grouped.setdefault(pair, []).append(item)
    traces = {}
    critical_levels = [float(m.L(p.a, p.b)) for p in enumeration.points]
    pairs = []
    totals = {"candidates": 0, "roots": 0, "same_order": 0,
              "terminal": 0, "critical_transition": 0, "unresolved": 0}
    for (i, j), items in sorted(grouped.items()):
        if i == j:
            # A self-contact requires two parameter sheets of one trace and
            # is not representable by the one-valued pair profile below.
            result = {
                "threshold": float(threshold),
                "candidate_count": len(items), "root_count": 0, "roots": [],
                "same_order_count": 0, "terminal_count": 0,
                "critical_transition_count": 0,
                "unresolved_count": len(items),
                "event_classes": ["unresolved"]*len(items),
                "pointwise_resolved_count": 0,
                "persistent_resolved_count": 0,
                "persistence_factor": 2.0,
            }
            profile = None
        else:
            traces.setdefault(i, loss_trace(m, branches[i]))
            traces.setdefault(j, loss_trace(m, branches[j]))
            profile = order_profile(
                m, traces[i], traces[j], predicate_tolerance)
            event_levels = [float(m.L(*map(float, item["point"])))
                            for item in items]
            side = _terminal_side(
                terminal_suffixes, i, j, terminal_tolerance)
            clusters = _contact_clusters(items, adjacency=cluster_gap_segments)
            clusters = _merge_clusters_without_persistent_gap(
                clusters, event_levels, profile, threshold)
            cluster_reports = []
            roots = []
            event_classes = [None]*len(items)
            same_order_count = terminal_count = 0
            critical_transition_count = unresolved_count = 0
            pointwise = persistent = 0
            for indices in clusters:
                values = [event_levels[k] for k in indices]
                interval = (min(values), max(values))
                probe = classify_signed_profile(
                    profile.levels, profile.delta, profile.error,
                    [0.5*(interval[0]+interval[1])],
                    threshold=threshold, critical_levels=critical_levels,
                    terminal_sides=[None], coordinates=profile.arc,
                    resolution_scale=profile.scale,
                    event_intervals=[interval])
                pointwise = probe["pointwise_resolved_count"]
                persistent = probe["persistent_resolved_count"]
                kind = probe["event_classes"][0]
                normalized_extent, extent_detail = _cluster_extent(
                    branches, i, j, items, indices)
                isolated = normalized_extent <= isolation_factor
                touches_terminal = _cluster_touches_terminal(
                    terminal_suffixes, i, j, items, indices, branches,
                    terminal_tolerance, terminal_gap_factor)
                if kind == "resolved_root" and not isolated:
                    kind = "terminal" if touches_terminal else "unresolved"
                elif (kind == "unresolved" and not isolated
                      and touches_terminal):
                    kind = "terminal"
                candidate_count = len(indices)
                if kind == "resolved_root":
                    root = dict(probe["roots"][0])
                    root["candidate_count"] = candidate_count
                    root["normalized_extent"] = float(normalized_extent)
                    roots.append(root)
                elif kind == "same_order":
                    same_order_count += candidate_count
                elif kind == "terminal":
                    terminal_count += candidate_count
                elif kind == "critical_transition":
                    critical_transition_count += candidate_count
                else:
                    unresolved_count += candidate_count
                for k in indices:
                    event_classes[k] = kind
                cluster_reports.append({
                    "kind": kind, "candidate_count": candidate_count,
                    "loss_interval": interval,
                    "normalized_extent": float(normalized_extent),
                    "isolated": bool(isolated),
                    "touches_terminal": bool(touches_terminal),
                    "extent": extent_detail,
                })
            result = {
                "threshold": float(threshold),
                "persistence_factor": 2.0,
                "pointwise_resolved_count": pointwise,
                "persistent_resolved_count": persistent,
                "candidate_count": len(items),
                "root_count": len(roots), "roots": roots,
                "same_order_count": same_order_count,
                "terminal_count": terminal_count,
                "critical_transition_count": critical_transition_count,
                "unresolved_count": unresolved_count,
                "event_classes": event_classes,
                "contact_cluster_count": len(clusters),
                "clusters": cluster_reports,
                "terminal_side": side,
            }
        entry = {
            "branches": (i, j),
            **result,
            "profile_levels": (0 if profile is None else len(profile.levels)),
            "first_dropped_nonmonotone": (
                None if profile is None else profile.first_dropped_nonmonotone),
            "second_dropped_nonmonotone": (
                None if profile is None else profile.second_dropped_nonmonotone),
            "profile_valid_fraction": (
                None if profile is None or not len(profile.valid) else
                float(np.count_nonzero(profile.valid)/len(profile.valid))),
            "minimum_transversality": (
                None if profile is None or not len(profile.transversality) else
                float(np.min(profile.transversality))),
        }
        pairs.append(entry)
        totals["candidates"] += result["candidate_count"]
        totals["roots"] += result["root_count"]
        totals["same_order"] += result["same_order_count"]
        totals["terminal"] += result["terminal_count"]
        totals["critical_transition"] += result["critical_transition_count"]
        totals["unresolved"] += result["unresolved_count"]
    # A critical-level event is not a crossing witness, but neither is it a
    # discharge: regular-level order is undefined across that transition.
    # Keep the audit conservative until a separate local critical-point or
    # level-component certificate accounts for the contact.
    decision = ("fault" if totals["roots"] else
                "unresolved" if (totals["unresolved"]
                                 or totals["critical_transition"])
                else "accepted")
    return {
        "method": "loss_level_order_sweep",
        "decision": decision,
        "threshold": float(threshold),
        **totals,
        "pairs": pairs,
    }
