"""Assembly: enumeration → branches → skeleton, plus the certificate ledger.

SPONG_FOUNDING Part II, section 11.  A Portrait is a measurement: every
drawn object carries residuals a skeptic can recompute without trusting
the code that produced it.

Unstable connections are discovered by continuation against every minimum;
backbone order does not determine their destinations because a stable
separatrix may cross the backbone away from a critical point.  Stable branches
(separatrices) run to compute-box exit and carry the √d_eff certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from . import atlas, charts, sturm, topology
from .model import Model


@dataclass
class Portrait:
    model: Model
    enumeration: sturm.Enumeration
    branches: list
    box: tuple                    # compute box (a_lo, a_hi, b_lo, b_hi)
    view: tuple | None
    ledger: dict = field(default_factory=dict)


def _trace_box(m: Model, display_box, scale: float = 1.35):
    """Larger integration box around the display view.

    The default view keeps the critical skeleton readable, but separatrices
    may leave that view and later pass through it again.  Trace against a
    larger box so rendering has those visible re-entry chords.
    """
    a0, a1, b0, b1 = display_box
    ac, bc = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    ah = 0.5 * scale * max(a1 - a0, 1.0)
    bh = 0.5 * scale * max(b1 - b0, 1.0)
    bmax = atlas.legal_max_b(m)
    return (ac - ah, ac + ah, max(bc - bh, -bmax), min(bc + bh, bmax))


def _unstable_stub(s, direction):
    return next((stub for stub in s.stubs
                 if stub.manifold == "unstable"
                 and stub.b_direction == direction), None)


def _stable_stub(s, sign, m):
    candidates = [stub for stub in s.stubs if stub.manifold == "stable"]
    if not candidates:
        return None
    def w_departure(stub):
        curve = np.asarray(stub.curve)
        if len(curve) < 2:
            return 0.0
        a, b = curve[-1]
        return float(a - m.a_star(b))
    return next((stub for stub in candidates
                 if sign*w_departure(stub) > 0), None)


def _capture_radius(e: sturm.Enumeration, ds: float) -> float:
    """An isolated endpoint neighbourhood, never merely four huge chords.

    The continuation engine historically used ``4*ds``.  On a connection
    thousands of units long that radius can contain a different critical
    point, so an all-minima discovery trace can be assigned to the wrong
    sink before it has entered any local sink neighbourhood.  Bound the
    radius by one eighth of the critical-point separation; the resulting
    balls are disjoint from one another and from every saddle.
    """
    points = e.points
    separation = min(
        (float(np.hypot(p.a-q.a, p.b-q.b))
         for i, p in enumerate(points) for q in points[i+1:]),
        default=np.inf)
    return float(min(4.0*ds, separation/8.0))


def compute(m: Model, view=None, geometry_level: int = 0,
            _enumeration=None, _display_view=None,
            _genericity=None) -> Portrait:
    """Compute the certified portrait inside the §8b box contract."""
    e = (_enumeration if _enumeration is not None else
         sturm.materialize_stubs(m, sturm.enumerate_critical_points(m)))
    display_view = (_display_view if _display_view is not None else
                    atlas.compute_box(m, e, view=view))
    box = _trace_box(m, display_view, scale=1.35*(2.0**geometry_level))
    gen = _genericity if _genericity is not None else atlas.genericity(m)

    branches = []
    # Stable separatrices are computed first for readability of this routine,
    # NOT because unstable branches consume them: the unstable loop below
    # reads only (m, e, box, display_view), and discovers destinations through
    # topology.sublevel_component_minima and capture against e.minima.  Every
    # branch here is independent of every other, which is what makes the two
    # loops parallelisable and the contact scan the only barrier.
    span_scale = max(display_view[3] - display_view[2],
                     display_view[1] - display_view[0])
    # Escalation must improve geometric resolution as well as enlarge the
    # trace box.  Merely extending the same sampled curves repeats identical
    # pre-terminal chord contacts at every level.  Halving the requested chord
    # scale lets dense collocation distinguish nearby invariant manifolds
    # before the exact terminal-product completion takes over.
    resolution_divisor = 2.0**geometry_level
    for s in e.saddles:
        for sign in (+1, -1):
            br = charts.trace_stable(
                m, s.b, sign, box=box,
                ds=span_scale/(30000.0*resolution_divisor),
                critical_local=s.local,
                critical_stub=_stable_stub(s, sign, m))
            br.diag["saddle_b"] = s.b
            br.diag["stable_sign"] = sign
            if br.term == "box_exit" and len(br.Y) > 50:
                br.certs["asymptote"] = atlas.asymptote_certificate(
                    m, br.Y)
            branches.append(br)

    discovery_ds = max(display_view[3]-display_view[2],
                       display_view[1]-display_view[0]) / (
                           30000.0*resolution_divisor)
    for s in e.saddles:
        # ---- unstable branches: discover capture, never prescribe it --- #
        for direction in (+1, -1):
            stub = _unstable_stub(s, direction)
            feasible = topology.sublevel_component_minima(
                m, e, stub.curve[-1]) if stub is not None else list(e.minima)
            same_side = [
                q for q in feasible if direction*(q.b-s.b) > 0.0]
            b_exit = box[3] if direction > 0 else box[2]
            a_exit = float(m.a_star(b_exit))
            direct = same_side[0] if len(same_side) == 1 else None
            nominal = ((direct.a, direct.b) if direct is not None
                       else (a_exit, b_exit))
            trace_ds = discovery_ds
            if direct is not None:
                db_direct = abs(direct.b-s.b)
                chord_direct = float(np.hypot(
                    direct.a-s.a, direct.b-s.b))
                trace_ds = max(
                    db_direct/(4000.0*resolution_divisor),
                    chord_direct/(8000.0*resolution_divisor))
            br = charts.trace_unstable(
                m, s.b, nominal, box=box,
                ds=trace_ds,
                cap_r=_capture_radius(e, trace_ds),
                critical_local=s.local,
                critical_stub=stub,
                arrival_local=(direct.local if direct is not None else None),
                candidate_minima=same_side,
                candidate_enumeration=e,
                capture_targets=[(q.a, q.b) for q in same_side])
            br.diag["saddle_b"] = s.b
            br.diag["unstable_direction"] = direction
            br.diag["sublevel_candidate_b"] = [
                float(q.b) for q in same_side]
            br.diag["sublevel_unique"] = direct is not None
            if br.term == "capture":
                destination = min(
                    e.minima,
                    key=lambda q: (br.Y[-1, 0]-q.a)**2
                    + (br.Y[-1, 1]-q.b)**2)
                # Discovery establishes topology; now retrace at a resolution
                # scaled to the observed finite connection.  This is a
                # numerical refinement, not a prior destination assumption.
                db = abs(destination.b-s.b)
                chord = float(np.hypot(destination.a-s.a, db))
                refine_ds = max(db/4000.0, chord/8000.0)
                refine_cap_r = _capture_radius(e, refine_ds)
                refined = charts.trace_unstable(
                    m, s.b, (destination.a, destination.b), box=box,
                    ds=refine_ds, cap_r=refine_cap_r,
                    critical_local=s.local,
                    critical_stub=_unstable_stub(s, direction),
                    arrival_local=destination.local,
                    # The coarse trace proposes a scale, not a topological
                    # label.  Keep every minimum live during refinement:
                    # otherwise a coarse separatrix crossing can nominate
                    # the wrong basin and the refined integral curve reaches
                    # its true minimum with capture artificially disabled.
                    capture_targets=[(q.a, q.b) for q in e.minima])
                if refined.term == "capture":
                    destination = min(
                        e.minima,
                        key=lambda q: (refined.Y[-1, 0]-q.a)**2
                        + (refined.Y[-1, 1]-q.b)**2)
                    br = refined
                    br.diag["connection_discovered"] = True
                    br.diag["target"] = (destination.a, destination.b)
                    br.certs["connection_ok"] = (
                        abs(br.Y[-1, 0]-destination.a) < 1e-9
                        and abs(br.Y[-1, 1]-destination.b) < 1e-9)
                else:
                    # A coarse discovery capture is only a candidate label.
                    # Never retain its long straight connector when the
                    # target-specific retrace cannot reproduce the capture.
                    br = refined
                    br.diag["candidate_target"] = (
                        destination.a, destination.b)
                    br.diag["target"] = None
                    br.certs["connection_ok"] = False
            else:
                br.diag["target"] = None
                br.certs["connection_ok"] = br.term == "box_exit"
            br.diag["saddle_b"] = s.b
            br.diag["unstable_direction"] = direction
            branches.append(br)

    p = Portrait(m, e, branches, box,
                 view if view is not None else display_view)
    p.ledger = build_ledger(p, gen)
    p.ledger["topology"] = topology.audit(m, e, branches, box)
    p.ledger["topology"]["geometry_level"] = geometry_level
    p.ledger["summary"]["topology_status"] = p.ledger["topology"]["status"]
    return p


def certified_compute(m: Model, view=None, max_geometry_level: int = 2,
                      _enumeration=None) -> Portrait:
    """Step up the portraitist until topology certifies or explicitly refuse."""
    # Geometry escalation must not redo the exact zero-dimensional Morse
    # skeleton, its conditioned critical jets, or its materialized stubs.
    # Those are properties of the model and are deliberately upstream of the
    # interactive geometry machine.
    started = time.perf_counter()
    enumeration = (_enumeration if _enumeration is not None else
                   sturm.enumerate_critical_points(m))
    enumerated = time.perf_counter()
    enumeration = sturm.materialize_stubs(m, enumeration)
    stubbed = time.perf_counter()
    display_view = atlas.compute_box(m, enumeration, view=view)
    genericity = atlas.genericity(m)
    attempts = []
    result = None
    for level in range(max_geometry_level+1):
        geometry_started = time.perf_counter()
        result = compute(
            m, view=view, geometry_level=level, _enumeration=enumeration,
            _display_view=display_view, _genericity=genericity)
        top = result.ledger["topology"]
        attempts.append({
            "geometry_level": level, "status": top["status"],
            "reason": top["resolution_reason"],
            "forbidden": top["forbidden_count"],
            "ambiguous": top["ambiguous_count"],
            "uncertified_ends": sum(
                not x["certified"] for x in top["unstable_ends"]),
            "uncertified_tails": sum(
                not x["certified"] for x in top["stable_tails"]),
            "elapsed_sec": time.perf_counter()-geometry_started,
        })
        if (top["status"] == "certified"
                or top["resolution_reason"] in (
                    "branch_abort", "branch_inventory_incomplete")):
            break
    result.ledger["topology"]["attempts"] = attempts
    result.ledger["timing"] = {
        "enumeration_sec": enumerated-started,
        "stub_materialization_sec": stubbed-enumerated,
        "geometry_sec": sum(x["elapsed_sec"] for x in attempts),
        "total_sec": time.perf_counter()-started,
    }
    return result


def _max_turn_deg(Y: np.ndarray) -> float:
    if len(Y) < 3:
        return 0.0
    d = np.diff(Y, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    ok = (seg[:-1] > 1e-13) & (seg[1:] > 1e-13)
    if not ok.any():
        return 0.0
    dot = d[:-1, 0] * d[1:, 0] + d[:-1, 1] * d[1:, 1]
    ct = dot[ok] / (seg[:-1][ok] * seg[1:][ok])
    return float(np.degrees(np.arccos(np.clip(ct, -1.0, 1.0)).max()))


def build_ledger(p: Portrait, gen: dict) -> dict:
    """The certificate ledger (SPONG_FOUNDING §11), machine-checkable."""
    e = p.enumeration
    balance = atlas.index_balance(p.model, e)
    led = {
        "enumeration": {
            "n_critical": len(e.points),
            "n_min": len(e.minima),
            "n_saddle": len(e.saddles),
            "psi_positive[EXACT]": e.psi_positive,
            "morse[EXACT]": e.morse,
            "u2_alternation[EXACT]": e.alternates,
        },
        "genericity[EXACT]": gen,
        "index_balance[EXACT]": balance,
        "branches": [],
    }
    for br in p.branches:
        entry = {
            "kind": br.kind,
            "saddle_b": br.diag.get("saddle_b"),
            "term": br.term,
            "n_points": int(len(br.Y)),
            "angle_energy[RESIDUAL]": br.certs.get("angle_energy"),
            "angle_resolved": br.certs.get("angle_resolved"),
            "angle_unresolved": br.certs.get("angle_unresolved"),
            "backbone_residual[RESIDUAL]": br.certs.get("backbone_residual"),
            "max_turn_deg[RESIDUAL]": _max_turn_deg(br.Y),
        }
        if "seam_residual" in br.certs:
            entry["seam_residual[RESIDUAL]"] = br.certs["seam_residual"]
        if "connection_ok" in br.certs:
            entry["connection[RESIDUAL]"] = br.certs["connection_ok"]
        if "asymptote" in br.certs:
            entry["asymptote_residual[RESIDUAL]"] = \
                br.certs["asymptote"]["residual"]
        led["branches"].append(entry)

    led["summary"] = {
        "all_branches_clean": all(
            b["term"] in ("capture", "box_exit") for b in led["branches"]),
        "worst_angle_energy": max(
            (b["angle_energy[RESIDUAL]"] or 0.0) for b in led["branches"])
        if led["branches"] else 0.0,
        "worst_max_turn_deg": max(
            b["max_turn_deg[RESIDUAL]"] for b in led["branches"])
        if led["branches"] else 0.0,
        "balanced": balance["balanced"],
    }
    return led
