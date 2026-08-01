"""Render the three portraits across the Theorem-4 saddle connection.

The two side panels are certified Morse-chamber SPONG portraits with clean
individual branches and robustly separated landing fates.  Exact sublevel
and superlevel product completions prevent coincident terminal samples from
being mistaken for changes to the finite Morse skeleton.  The center panel
is deliberately different: the ordinary portraitist must refuse a
non-Morse-Smale saddle connection, so we remove the two numerical branch
continuations involved in the slide and insert their common geometric wall
limit.  A high-accuracy Radau trace at the measured wall supplies that segment
and must enter a small neighborhood of the target saddle.

The horizontal scale is the physical ``a`` coordinate in every panel.  The
three full portraits are kept visually unobstructed; a second three-panel
strip shows the B-to-N connection/turning region at larger scale.

Usage:
  PYTHONPATH=src python3 demos/saddle_connection_triptych.py
  open out/saddle_connection_triptych/nonnearest-saddle-connection.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from spong import charts, model, portrait, render, zoo


def build_member(member: str):
    case = zoo.rheostat_member("nonnearest-saddle-connection", member)
    degree = len(case.g)-1
    if case.moment_dist != "uniform01":
        raise ValueError(f"unsupported wall-demo moments: {case.moment_dist}")
    return model.build(case.f, case.g, model.moments_uniform01(2*degree+1))


def critical_near(p, b_value: float, kind: str = "saddle"):
    candidates = [q for q in p.enumeration.points if q.kind == kind]
    return min(candidates, key=lambda q: abs(q.b-b_value))


def trace_wall_connection(m, source, target, radius: float = 1e-8):
    """Evidence-grade direct trace of the measured wall connection."""
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError(
            "the triptych demo requires scipy (use the system python or "
            "install the optional demo dependency)") from exc

    H = np.asarray(m.hessL(source.a, source.b), dtype=float)
    # Closed-form eigenvector of the smaller eigenvalue of a symmetric 2x2
    # Hessian; the flow's unstable direction is the Hessian's negative mode.
    h00, h01, h11 = H[0, 0], H[0, 1], H[1, 1]
    root = math.hypot(h00-h11, 2.0*h01)
    eigenvalue = 0.5*(h00+h11-root)
    vector = np.array((h01, eigenvalue-h00), dtype=float)
    if np.hypot(*vector) == 0.0:
        vector = np.array((eigenvalue-h11, h01), dtype=float)
    vector /= np.hypot(*vector)
    if vector[1] < 0.0:
        vector = -vector

    scale = 1.0+math.hypot(source.a, source.b)
    start = np.array((source.a, source.b))+1e-10*scale*vector

    def rhs(_time, state):
        return -m.gradL(float(state[0]), float(state[1]))

    def arrival(_time, state):
        return math.hypot(state[0]-target.a, state[1]-target.b)-radius

    arrival.terminal = True
    arrival.direction = -1
    solution = solve_ivp(
        rhs, (0.0, 1e6), start, method="Radau",
        rtol=2e-13, atol=2e-15*scale, max_step=10.0, events=arrival)
    curve = solution.y.T
    distances = np.hypot(curve[:, 0]-target.a, curve[:, 1]-target.b)
    closest = float(np.min(distances))
    if solution.status != 1 or closest > 1.05*radius:
        raise ArithmeticError(
            f"wall trace did not resolve the target saddle: {closest:.3e}")
    curve = np.vstack(((source.a, source.b), curve, (target.a, target.b)))
    return curve, {
        "integrator": "Radau",
        "rtol": 2e-13,
        "atol_scale": 2e-15,
        "arrival_radius": radius,
        "closest_approach": closest,
        "steps": int(len(solution.t)),
        "terminal_time": float(solution.t[-1]),
    }


def wall_limit_portrait(p, family, connection):
    """Remove the two continuations replaced by the wall connection."""
    source = critical_near(p, family.source_b)
    target = critical_near(p, family.target_b)
    source_unstable = next(
        i for i, branch in enumerate(p.branches)
        if branch.kind == "unstable"
        and abs(branch.diag.get("saddle_b", math.inf)-source.b) < 1e-7
        and branch.diag.get("unstable_direction") == family.unstable_direction)
    target_stable = [
        (float(np.min(np.hypot(
            branch.Y[:, 0]-source.a, branch.Y[:, 1]-source.b))), i)
        for i, branch in enumerate(p.branches)
        if branch.kind == "stable"
        and abs(branch.diag.get("saddle_b", math.inf)-target.b) < 1e-7]
    closest, target_stable_index = min(target_stable)
    branches = [
        charts.Branch(
            branch.kind, np.asarray(branch.Y), branch.term,
            dict(branch.certs), dict(branch.diag))
        for i, branch in enumerate(p.branches)
        if i not in {source_unstable, target_stable_index}]
    ledger = dict(p.ledger)
    ledger["comparison"] = {
        "geometry_method": "geometric wall limit",
        "critical_method": "exact Sturm",
    }
    wall = portrait.Portrait(
        p.model, p.enumeration, branches, p.box, p.view, ledger)
    return wall, {
        "removed_source_unstable": source_unstable,
        "removed_target_stable": target_stable_index,
        "removed_stable_closest_approach": closest,
        "connection_points": int(len(connection)),
    }


def decimate_portrait(p, max_points: int = 6000):
    """Screen-resolution copy; the underlying portrait remains untouched."""
    branches = []
    for branch in p.branches:
        Y = np.asarray(branch.Y)
        if len(Y) > max_points:
            indices = np.unique(np.linspace(
                0, len(Y)-1, max_points, dtype=int))
            Y = Y[indices]
        branches.append(charts.Branch(
            branch.kind, Y, branch.term,
            dict(branch.certs), dict(branch.diag)))
    return portrait.Portrait(
        p.model, p.enumeration, branches, p.box, p.view, dict(p.ledger))


def detail_view(p, family):
    source = critical_near(p, family.source_b)
    target = critical_near(p, family.target_b)
    a0, a1 = sorted((source.a, target.a))
    da = max(a1-a0, 0.25)
    db = target.b-source.b
    return (a0-0.22*da, a1+0.22*da,
            source.b-0.12*db, target.b+0.12*db)


def _svg_body(svg: str) -> str:
    return svg[svg.find(">")+1:svg.rfind("</svg>")]


def panel_svg(p, family, title, connection=None):
    overlays = None
    if connection is not None:
        overlays = [{
            "Y": connection,
            "color": "#7b3fb4",
            "width": 3.2,
            "mark_start": False,
            "mark_end": False,
            "label": "B→N saddle connection",
        }]
    main = render.plane_view(
        p, view=family.default_view, width=640, height=500,
        n_levels=32, n_grid=901, title=title, overlays=overlays)
    topology = topology_status(p)
    audit_note = ""
    if connection is None and topology != "certified":
        audit_note = (
            '<rect x="164" y="27" width="312" height="14" fill="white" '
            'fill-opacity="0.88"/>\n'
            '<text x="320" y="37" text-anchor="middle" font-size="9" '
            'fill="#a33">branches qualified; global contact audit: '
            f'{topology or "not run"}</text>\n')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="500" '
        'viewBox="0 0 640 500">\n'
        f'{_svg_body(main)}\n{audit_note}</svg>')


def detail_panel_svg(p, family, title, connection=None):
    overlays = None
    if connection is not None:
        overlays = [{
            "Y": connection,
            "color": "#7b3fb4",
            "width": 3.2,
            "mark_start": False,
            "mark_end": False,
            "label": "B→N saddle connection",
        }]
    return render.plane_view(
        p, view=detail_view(p, family), width=640, height=330,
        n_levels=18, n_grid=601, title=title, overlays=overlays)


def triptych_svg(panels, panel_height=500):
    width, height = 1920, panel_height
    nested = []
    for i, svg in enumerate(panels):
        nested.append(
            f'<svg x="{640*i}" y="0" width="640" height="{height}" '
            f'viewBox="0 0 640 {height}">{_svg_body(svg)}</svg>')
        if i:
            nested.append(
                f'<line x1="{640*i}" y1="0" x2="{640*i}" y2="{height}" '
                'stroke="#999" stroke-width="1"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(nested)+"\n</svg>")


def topology_status(p):
    return p.ledger.get("topology", {}).get("status")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the nonnearest-attachment saddle-connection wall")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("out/saddle_connection_triptych"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    family = zoo.get_wall_family("nonnearest-saddle-connection")
    portraits = {}
    for member in ("below", "wall", "above"):
        # Pass the requested view through the normal box contract so every
        # visible curve is continued beyond the rendering crop.  The
        # resulting trace box also contains the remote b≈7.42 minimum.
        portraits[member] = portrait.compute(
            build_member(member), view=family.default_view)

    for member in ("below", "above"):
        if not portraits[member].ledger.get("summary", {}).get(
                "all_branches_clean"):
            raise ArithmeticError(
                f"{member} wall chamber has an unclean branch")

    source = critical_near(portraits["wall"], family.source_b)
    target = critical_near(portraits["wall"], family.target_b)
    connection, trace = trace_wall_connection(
        portraits["wall"].model, source, target)
    wall, surgery = wall_limit_portrait(
        portraits["wall"], family, connection)
    portraits["wall"] = wall
    for key in portraits:
        portraits[key] = decimate_portrait(portraits[key])

    if family.wall_bracket is not None:
        lo, hi = family.wall_bracket
        wall_title = f"Wall: Λ* ∈ [{lo:.15g}, {hi:.15g}]"
    else:
        wall_title = f"Wall: Λ*≈{family.wall_parameter:.15g}"
    titles = {
        "below": f"Below wall: Λ={family.below_parameter:g} → far minimum",
        "wall": wall_title,
        "above": f"Above wall: Λ={family.above_parameter:g} → near minimum",
    }
    panels = {}
    detail_panels = {}
    for member in ("below", "wall", "above"):
        member_connection = connection if member == "wall" else None
        panels[member] = panel_svg(
            portraits[member], family, titles[member],
            member_connection)
        detail_panels[member] = detail_panel_svg(
            portraits[member], family,
            f'{titles[member]} | connection detail', member_connection)
        (args.output_dir/f"{family.name}-{member}.svg").write_text(
            panels[member])
    triptych = triptych_svg([panels[x] for x in ("below", "wall", "above")])
    triptych_name = f"{family.name}-triptych.svg"
    (args.output_dir/triptych_name).write_text(triptych)
    detail_triptych = triptych_svg(
        [detail_panels[x] for x in ("below", "wall", "above")],
        panel_height=330)
    detail_name = f"{family.name}-connection-detail.svg"
    (args.output_dir/detail_name).write_text(detail_triptych)

    if family.wall_bracket is not None:
        blo, bhi = family.wall_bracket
        bracket_html = (
            f"<p>The wall parameter is citable only through its bracket: "
            f"Λ* ∈ [<code>{blo!r}</code>, <code>{bhi!r}</code>]. "
            f"{family.bracket_protocol}</p>")
    else:
        bracket_html = ""
    html_name = f"{family.name}.html"
    (args.output_dir/html_name).write_text(f"""<!doctype html>
<meta charset="utf-8"><title>{family.name}</title>
<style>body{{font-family:system-ui;margin:24px;background:#fafafa}}
object{{width:100%;background:white;border:1px solid #bbb}}
code{{background:#eee;padding:2px 4px}}</style>
<h1>Saddle-connection handle slide</h1>
<p>{family.description}</p>
<object data="{triptych_name}" type="image/svg+xml"></object>
<h2>Connection detail</h2>
<object data="{detail_name}" type="image/svg+xml"></object>
<p>The side panels are ordinary Morse-chamber portraits with individually
qualified branches, certified global contact audits, and robustly different
landing fates. The center panel is the geometric wall limit: purple is the
common B→N invariant-manifold segment; the two off-wall continuations it
replaces are intentionally absent.</p>
{bracket_html}
""")
    report = {
        "format": "spong-saddle-connection-triptych-v1",
        "family": family.name,
        "parameters": {
            "below": family.below_parameter,
            "wall": family.wall_parameter,
            "above": family.above_parameter,
        },
        "wall_bracket": family.wall_bracket,
        "bracket_protocol": family.bracket_protocol,
        "tracked": {
            "source_b": family.source_b,
            "target_b": family.target_b,
            "unstable_direction": family.unstable_direction,
        },
        "side_topology": {
            key: topology_status(portraits[key])
            for key in ("below", "above")
        },
        "side_all_branches_clean": {
            key: bool(portraits[key].ledger.get("summary", {}).get(
                "all_branches_clean"))
            for key in ("below", "above")
        },
        "wall_trace": trace,
        "wall_surgery": surgery,
        "triptych": triptych_name,
        "connection_detail": detail_name,
        "html": html_name,
    }
    (args.output_dir/f"{family.name}.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(args.output_dir/html_name)
    return report


if __name__ == "__main__":
    main()
