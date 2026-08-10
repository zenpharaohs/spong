"""Fate discovery from level-set components -- the nonlinear Lehmer filter.

For a point p on ANY integral curve of the descent flow, and ANY exact
level c > L(p), the forward orbit stays in {L <= L(p)}, which lies inside
the component K_c(p) of {L < c} containing p.  So the possible fates of
the curve -- its omega-limits -- are contained in

    minima(K_c(p))  union  the unbounded ends of K_c(p).

This holds at EVERY certified rung of the slack ladder, and a deeper rung
(smaller c) gives a smaller component and hence a TIGHTER candidate set.
When some rung yields a bounded component with no saddle and exactly one
minimum, the fate is FORCED: the branch terminates at that minimum, with
no further reference to the traced curve.

Because L = u(b) + A(b)(a - a*(b))^2 with A > 0, every b-section of a
sublevel set is a single interval, so planar components are tubes over
components of {u < c}: the component hypergraph over all levels is the
merge tree of the one-variable function u, and every critical point of L
sits on the backbone.  L_aa = 2A > 0 also rules out local maxima, so a
bounded component is a disk and Euler characteristic sharpens the Conley
count to an equality: #minima = #saddles + 1.  A certified inventory
violating that is a soundness alarm, not a data point, and is reported as
such.

Superlevel components -- the time-reversed filter for stable branches --
are ALWAYS unbounded: from any point of {L > c} the vertical ray to
|a| -> infinity stays in the component, since L is coercive in a while u
is bounded above.  Stable-branch fates are therefore ends, never traps,
and the audit's exact_superlevel_product is already their correct
certificate.

This module is REPORTING ONLY: nothing here alters an audit verdict.
Audit integration -- a merge_tree_forced capture certificate, and the
candidate-consistency cross-check as an assertion -- is a separate,
deliberate step, because it interacts with branch_set_incomplete (a forced
completion could stand in for an unfinished trace, which today the audit
does not allow).
"""

from __future__ import annotations

from . import topology


def component_fates(m, enumeration, point, ladder=None):
    """Certified fate candidates for a descent orbit through ``point``.

    Walks the slack ladder shallow to deep, keeping the DEEPEST certified
    inventory (the tightest valid candidate set), and stops early once the
    fate is forced.  Every certified rung is independently a valid
    overapproximation of the fate set, so descending can only tighten,
    never guess.
    """
    ladder = topology._TUBE_SLACK_SHIFTS if ladder is None else ladder
    best = None
    for shift in ladder:
        inventory = topology._sublevel_component_inventory(
            m, enumeration, point, slack_shift=shift)
        if not inventory["certified"]:
            continue
        minima = tuple(inventory["minima"])
        saddles = tuple(inventory["saddles"])
        bounded = bool(inventory["bounded"])
        if bounded and len(minima) != len(saddles) + 1:
            return {"certified": False,
                    "reason": "euler_characteristic_violation",
                    "slack_shift": shift,
                    "n_minima": len(minima),
                    "n_saddles": len(saddles)}
        best = {
            "certified": True,
            "reason": None,
            "slack_shift": shift,
            "level_upper": inventory["level_upper"],
            "bounded": bounded,
            "b_interval": (inventory["left_boundary"],
                           inventory["right_boundary"]),
            "minima": tuple((float(q.a), float(q.b)) for q in minima),
            "saddles": tuple((float(q.a), float(q.b)) for q in saddles),
            "unbounded_ends": tuple(inventory["unbounded_sides"]),
            "forced": bounded and not saddles and len(minima) == 1,
        }
        if best["forced"]:
            break
    if best is None:
        return {"certified": False,
                "reason": "no_certified_inventory_on_ladder"}
    return best


def launch_fates(m, enumeration, branch):
    """Fate candidates at the first sample past the certified local graph.

    The materialized stub is separately certified (fixed point,
    injectivity, spectrum), so its far end is the earliest point already
    accepted as lying on the invariant curve.  The candidate set there is
    the launch filter; ``forced`` True means the branch's termination is
    decided with no reference to the traced curve beyond the stub.
    """
    last = len(branch.Y) - 1
    index = min(last, max(0, int(branch.diag.get("critical_steps", 0))))
    result = dict(component_fates(m, enumeration, branch.Y[index]))
    result["index"] = index
    result["point"] = (float(branch.Y[index, 0]), float(branch.Y[index, 1]))
    return result


def _matches(candidate, target, tolerance):
    return (abs(candidate[0] - float(target[0])) <= tolerance
            and abs(candidate[1] - float(target[1])) <= tolerance)


def fate_report(m, enumeration, branches, tolerance=1e-9):
    """Per-branch launch fates, for the ledger or the explorer.

    Unstable branches get the sublevel filter, plus a consistency flag:
    a recorded capture target OUTSIDE its own candidate set would indict
    either the filter or the trace, and must never pass silently.  Stable
    branches are listed with kind and termination only -- their
    time-reversed components are always unbounded (module docstring), so
    the filter adds nothing to the existing superlevel certificate.
    Capture branches additionally carry ``terminal_fates`` -- the same
    filter at the last measured sample, where launch-unforced captures
    typically become forced (the launch level can sit above u_infinity
    while the terminal level has descended into the forcing window).
    """
    report = []
    for i, branch in enumerate(branches):
        entry = {"branch": i, "kind": branch.kind, "term": branch.term,
                 "target": branch.diag.get("target")}
        if branch.kind == "unstable":
            entry.update(launch_fates(m, enumeration, branch))
            target = branch.diag.get("target")
            if target is not None and entry.get("certified"):
                entry["target_in_candidates"] = any(
                    _matches(q, target, tolerance)
                    for q in entry["minima"])
            if branch.term == "capture" and len(branch.Y) >= 2:
                # Y[-1] is the appended exact connector; the last MEASURED
                # sample is Y[-2], mirroring the audit's convention.
                terminal = component_fates(
                    m, enumeration, branch.Y[len(branch.Y) - 2])
                if target is not None and terminal.get("certified"):
                    terminal["target_in_candidates"] = any(
                        _matches(q, target, tolerance)
                        for q in terminal["minima"])
                entry["terminal_fates"] = terminal
        report.append(entry)
    return report
