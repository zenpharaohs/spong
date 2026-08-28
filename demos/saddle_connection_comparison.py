"""Compare ordinary ODE portraitists at the saddle-connection wall.

This is deliberately a geometry-only comparison.  Every method receives the
same exact Sturm critical-point inventory and the same wall member of the
Lambda-rheostat family.  The casual methods launch from a fixed eigenvector
offset and use ordinary IVP integration.  Their unstable traces are not
stopped merely for entering a finite-radius ball around a saddle: continuing
the discretized unstable branch exposes which outgoing branch its accumulated
error selects at the structurally unstable connection.  No optimizer or
trajectory overlay is drawn: each casual panel is just the phase portrait
produced by that discretization.

Usage:
  PYTHONPATH=src:. python3 demos/saddle_connection_comparison.py
  open out/saddle_connection_comparison/nonnearest-saddle-connection-casual-comparison.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

from spong import comparison, portrait, render, zoo

from demos.saddle_connections import (
    common_level_separation,
    level_crossing,
    shooting_residual,
)
from demos.saddle_connection_triptych import (
    build_member,
    critical_near,
    decimate_portrait,
    detail_view,
    trace_wall_connection,
    wall_limit_portrait,
)


def tracked_branch(p, family):
    return next(
        branch for branch in p.branches
        if branch.kind == "unstable"
        and abs(branch.diag.get("saddle_b", math.inf)-family.source_b) < 1e-7
        and branch.diag.get("unstable_direction")
        == family.unstable_direction)


def point_polyline_distance(point, curve):
    """Distance from a point to the piecewise-linear displayed trace."""
    point = np.asarray(point, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if len(curve) == 0:
        return math.inf
    if len(curve) == 1:
        return float(np.hypot(*(curve[0]-point)))
    starts, directions = curve[:-1], np.diff(curve, axis=0)
    denominators = np.sum(directions*directions, axis=1)
    parameters = np.zeros(len(directions))
    nonzero = denominators > 0.0
    parameters[nonzero] = np.sum(
        (point-starts[nonzero])*directions[nonzero], axis=1
    )/denominators[nonzero]
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts+parameters[:, None]*directions
    return float(np.min(np.hypot(
        closest[:, 0]-point[0], closest[:, 1]-point[1])))


def numerical_fate(enumeration, branch):
    endpoint = np.asarray(branch.Y[-1], dtype=float)
    destination = min(
        enumeration.minima,
        key=lambda q: (endpoint[0]-q.a)**2+(endpoint[1]-q.b)**2)
    distance = float(np.hypot(endpoint[0]-destination.a,
                              endpoint[1]-destination.b))
    return destination, distance


def connection_pair(p, family):
    """The independently traced W^u(B) and W^s(N) wall candidates."""
    source = critical_near(p, family.source_b)
    target = critical_near(p, family.target_b)
    unstable = tracked_branch(p, family)
    stable = min(
        (branch for branch in p.branches
         if branch.kind == "stable"
         and abs(branch.diag.get("saddle_b", math.inf)-target.b) < 1e-7),
        key=lambda branch: point_polyline_distance(
            (source.a, source.b), branch.Y))
    return source, target, unstable, stable


def two_sided_section_measurement(p, family):
    """Signed W^u(B)-versus-W^s(N) mismatch on one regular level.

    This is intentionally not called an anadromy defect: for an ablated
    portrait it includes both independent launch errors and continuation
    errors.  It is nevertheless the metrologically relevant test of whether
    two computations that must represent one wall orbit actually agree.
    """
    source, target, unstable, stable = connection_pair(p, family)
    source_level = float(p.model.L(source.a, source.b))
    target_level = float(p.model.L(target.a, target.b))
    scale = float(np.hypot(source.a-target.a, source.b-target.b))
    separation = common_level_separation(
        p.model, unstable.Y, stable.Y, source_level, target_level)
    shooting = shooting_residual(
        p.model, unstable.Y, stable.Y, source_level, target_level, scale)
    if separation is None or shooting is None:
        return {
            "resolved": False,
            "section_level": None,
            "physical_separation": None,
            "signed_mismatch": None,
            "normalized_mismatch": None,
        }
    return {
        "resolved": True,
        "section_level": float(shooting[0]),
        "physical_separation": float(separation[3]),
        "signed_mismatch": float(shooting[1]),
        "normalized_mismatch": float(shooting[2]),
    }


def transverse_section_samples(p, family, half_width=0.06, count=49):
    """Centerline-subtracted separation across regular loss sections.

    The ordinary corridor plot spends almost all of its resolution on motion
    along the connection.  Here each level is its own transverse section: the
    common centerline is subtracted, leaving W^u(B) at -d/2 and W^s(N) at
    +d/2, where d is their signed tangent-coordinate mismatch.
    """
    source, target, unstable, stable = connection_pair(p, family)
    upper = float(p.model.L(source.a, source.b))
    lower = float(p.model.L(target.a, target.b))
    if not upper > lower:
        return []
    samples = []
    for eta in np.linspace(-half_width, half_width, count):
        fraction = 0.5+float(eta)
        level = lower+fraction*(upper-lower)
        xu = level_crossing(p.model, unstable.Y, level)
        xs = level_crossing(p.model, stable.Y, level)
        if xu is None or xs is None:
            continue
        displacement = xs-xu
        midpoint = 0.5*(xu+xs)
        gradient = np.asarray(
            p.model.gradL(midpoint[0], midpoint[1]), dtype=float)
        norm = float(np.hypot(gradient[0], gradient[1]))
        if not np.isfinite(norm) or norm <= 1e-14:
            continue
        tangent = np.array([-gradient[1], gradient[0]])/norm
        mismatch = float(np.dot(displacement, tangent))
        if np.isfinite(mismatch):
            samples.append((float(eta), mismatch))
    return samples


def transverse_gallery_svg(panels, width=1040, panel_height=270):
    """Render transverse defects with a separate honest scale per panel."""
    columns = 2
    rows = (len(panels)+columns-1)//columns
    panel_width = width/columns
    height = rows*panel_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}'
        '.small{font-size:12px}.title{font-size:16px;font-weight:600}'
        '.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.wu{fill:none;stroke:#16803c;stroke-width:2.4}'
        '.ws{fill:none;stroke:#c62828;stroke-width:2.4}'
        '.wu-label{fill:#16803c}.ws-label{fill:#c62828}</style>',
    ]
    for index, (label, samples) in enumerate(panels):
        col, row = index % columns, index // columns
        ox, oy = col*panel_width, row*panel_height
        left, right, top, bottom = 88.0, 28.0, 42.0, 43.0
        pw = panel_width-left-right
        ph = panel_height-top-bottom
        defects = [d for _, d in samples]
        max_defect = max((abs(d) for d in defects), default=0.0)
        half_range = max(0.625*max_defect, 1e-12)
        x0, x1 = -half_range, half_range
        y0, y1 = -0.06, 0.06

        def sx(value):
            return ox+left+(value-x0)*pw/(x1-x0)

        def sy(value):
            return oy+top+(y1-value)*ph/(y1-y0)

        parts.extend([
            f'<text class="title" x="{ox+left:.2f}" y="{oy+23:.2f}">'
            f'{html.escape(label)}</text>',
            f'<line class="grid" x1="{sx(0):.2f}" y1="{sy(y0):.2f}" '
            f'x2="{sx(0):.2f}" y2="{sy(y1):.2f}"/>',
            f'<line class="axis" x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
            f'x2="{sx(x1):.2f}" y2="{sy(y0):.2f}"/>',
            f'<line class="axis" x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
            f'x2="{sx(x0):.2f}" y2="{sy(y1):.2f}"/>',
        ])
        if samples:
            wu = " ".join(
                f'{sx(-0.5*d):.2f},{sy(eta):.2f}' for eta, d in samples)
            ws = " ".join(
                f'{sx(+0.5*d):.2f},{sy(eta):.2f}' for eta, d in samples)
            parts.extend([
                f'<polyline class="wu" points="{wu}"/>',
                f'<polyline class="ws" points="{ws}"/>',
            ])
        parts.extend([
            f'<text class="small" x="{sx(x0):.2f}" y="{sy(y0)+19:.2f}">'
            f'{x0:.2e}</text>',
            f'<text class="small" text-anchor="end" x="{sx(x1):.2f}" '
            f'y="{sy(y0)+19:.2f}">{x1:.2e}</text>',
            f'<text class="small" text-anchor="middle" x="{sx(0):.2f}" '
            f'y="{sy(y0)+36:.2f}">signed transverse coordinate</text>',
            f'<text class="small" transform="translate({ox+20:.2f},'
            f'{oy+top+0.5*ph:.2f}) rotate(-90)" text-anchor="middle">'
            f'loss fraction - 1/2</text>',
            f'<text class="small wu-label" x="{ox+left+4:.2f}" '
            f'y="{oy+top+15:.2f}">W^u(B)</text>',
            f'<text class="small ws-label" x="{ox+left+70:.2f}" '
            f'y="{oy+top+15:.2f}">W^s(N)</text>',
            f'<text class="small" text-anchor="end" x="{ox+panel_width-right:.2f}" '
            f'y="{oy+top+15:.2f}">max |d|={max_defect:.3e}</text>',
        ])
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def _card(label, full_svg, detail_svg, note):
    return f"""<section>
<h2>{html.escape(label)}</h2>
<object data="{html.escape(full_svg)}" type="image/svg+xml"></object>
<h3>Connection corridor</h3>
<object class="detail" data="{html.escape(detail_svg)}"
        type="image/svg+xml"></object>
<p>{html.escape(note)}</p>
</section>"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare casual ODE geometry at a saddle connection")
    parser.add_argument(
        "--methods", nargs="+", choices=comparison.GEOMETRY_METHODS,
        default=list(comparison.GEOMETRY_METHODS))
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--time-horizon", type=float, default=200.0)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--stork-stages", type=int, default=20,
        help="stabilized stages for STORK-2/4 (STORK-4 supports 9 or 20)")
    parser.add_argument("--contact-threshold", type=float, default=4.0)
    parser.add_argument("--contact-limit", type=int, default=50000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("out/saddle_connection_comparison"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    family = zoo.get_wall_family("nonnearest-saddle-connection")
    m = build_member("wall")

    raw_wall = portrait.compute(m, view=family.default_view)
    reference_section = two_sided_section_measurement(raw_wall, family)
    source = critical_near(raw_wall, family.source_b)
    target = critical_near(raw_wall, family.target_b)
    connection, trace = trace_wall_connection(m, source, target)
    reference, surgery = wall_limit_portrait(
        raw_wall, family, connection)
    reference = decimate_portrait(reference)
    corridor = detail_view(reference, family)
    transverse_panels = [(
        "production conditioned stubs + GL8",
        transverse_section_samples(raw_wall, family))]

    reference_full = f"{family.name}-reference.svg"
    reference_detail = f"{family.name}-reference-detail.svg"
    reference_overlay = [{
        "Y": connection,
        "color": "#7b3fb4",
        "width": 3.2,
        "mark_start": False,
        "mark_end": False,
        "label": "resolved geometric B→N limit",
    }]
    render.save(render.plane_view(
        reference, view=family.default_view, width=800, height=600,
        n_levels=32, n_grid=801, title="Geometric wall limit",
        overlays=reference_overlay), str(args.output_dir/reference_full))
    render.save(render.plane_view(
        reference, view=corridor, width=800, height=430,
        n_levels=20, n_grid=601, title="Geometric wall limit: corridor",
        overlays=reference_overlay), str(args.output_dir/reference_detail))

    cards = [_card(
        "Geometric wall limit", reference_full, reference_detail,
        "The B unstable branch terminates at the N saddle. The purple curve "
        "is the independently resolved wall connection; no outgoing N branch "
        "is spuriously assigned to B. Before wall surgery, the production "
        "W^u(B)/W^s(N) midpoint mismatch is "
        f"{abs(reference_section['signed_mismatch']):.3e}.")]
    records = []
    for method in args.methods:
        candidate = comparison.casual_portrait(
            m, method,
            reference_enumeration=raw_wall.enumeration,
            view=family.default_view,
            step_size=args.step_size,
            max_steps=args.max_steps,
            time_horizon=args.time_horizon,
            rtol=args.rtol,
            atol=args.atol,
            stork_stages=args.stork_stages,
            capture_saddles=False)
        branch = tracked_branch(candidate, family)
        destination, destination_distance = numerical_fate(
            candidate.enumeration, branch)
        target_approach = point_polyline_distance(
            (target.a, target.b), branch.Y)
        section = two_sided_section_measurement(candidate, family)
        contact_diagnostics = comparison.manifold_contact_diagnostics(
            m, raw_wall.enumeration, candidate.branches, candidate.box,
            threshold=args.contact_threshold,
            candidate_limit=args.contact_limit)
        pair_contacts = contact_diagnostics["pair_order_sweep"]
        self_contacts = contact_diagnostics["self_contacts"]
        transverse_panels.append((
            method, transverse_section_samples(candidate, family)))
        destination_label = (
            f"minimum (a={destination.a:.6g}, b={destination.b:.6g})")
        stem = f"{family.name}-{method}"
        full_name, detail_name = f"{stem}.svg", f"{stem}-detail.svg"
        render.save(render.plane_view(
            candidate, view=family.default_view, width=800, height=600,
            n_levels=32, n_grid=801,
            title=f"{method}: discretized phase portrait"),
            str(args.output_dir/full_name))
        render.save(render.plane_view(
            candidate, view=corridor, width=800, height=430,
            n_levels=20, n_grid=601,
            title=f"{method}: connection corridor"),
            str(args.output_dir/detail_name))
        note = (
            f"The B-unstable branch passes within {target_approach:.3e} of N, "
            f"then selects {destination_label}; endpoint distance to that "
            f"minimum is {destination_distance:.3e}. The independent "
            f"W^u(B)/W^s(N) midpoint mismatch is "
            f"{abs(section['signed_mismatch']):.3e}. Contact diagnostic="
            f"{contact_diagnostics['decision']}: pair roots="
            f"{pair_contacts['roots']}, unresolved="
            f"{pair_contacts['unresolved'] + pair_contacts['critical_transition']}, "
            f"self-crossings={self_contacts['crosses']}. This is a numerical "
            "chamber choice, not the wall portrait.")
        cards.append(_card(method, full_name, detail_name, note))
        records.append({
            "method": method,
            "term": branch.term,
            "points": int(len(branch.Y)),
            "target_saddle_approach": target_approach,
            "two_sided_section": section,
            "contact_diagnostics": contact_diagnostics,
            "destination": {
                "a": float(destination.a),
                "b": float(destination.b),
                "endpoint_distance": destination_distance,
            },
            "elapsed_time": branch.diag.get("elapsed_time"),
            "accepted_steps": branch.diag.get("accepted"),
            "rejected_steps": branch.diag.get("rejected"),
            "full_svg": full_name,
            "detail_svg": detail_name,
        })

    transverse_name = f"{family.name}-transverse-defects.svg"
    (args.output_dir/transverse_name).write_text(
        transverse_gallery_svg(transverse_panels))

    report_name = f"{family.name}-casual-comparison.json"
    report = {
        "format": "spong-saddle-connection-comparison-v3",
        "family": family.name,
        "wall_parameter": family.wall_parameter,
        "critical_points": "exact Sturm inventory shared by all panels",
        "configuration": {
            "step_size": args.step_size,
            "time_horizon": args.time_horizon,
            "max_steps": args.max_steps,
            "rtol": args.rtol,
            "atol": args.atol,
            "stork_stages": args.stork_stages,
            "capture_saddles": False,
            "contact_threshold": args.contact_threshold,
            "contact_limit": args.contact_limit,
        },
        "reference": {
            "wall_trace": trace,
            "wall_surgery": surgery,
            "two_sided_section": reference_section,
            "full_svg": reference_full,
            "detail_svg": reference_detail,
        },
        "transverse_defects_svg": transverse_name,
        "comparisons": records,
    }
    (args.output_dir/report_name).write_text(
        json.dumps(report, indent=2)+"\n")

    html_name = f"{family.name}-casual-comparison.html"
    section_rows = [
        ("production conditioned stubs + GL8", reference_section)] + [
        (record["method"], record["two_sided_section"])
        for record in records]
    table_rows = "".join(
        "<tr><td>"+html.escape(label)+"</td><td>"+
        (f"{measurement['signed_mismatch']:+.6e}"
         if measurement["resolved"] else "unresolved")+
        "</td><td>"+
        (f"{measurement['physical_separation']:.6e}"
         if measurement["resolved"] else "unresolved")+
        "</td></tr>"
        for label, measurement in section_rows)
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG saddle-connection geometry comparison</title>
<style>
body {{ font-family: system-ui,sans-serif; margin: 24px; background: #fafafa; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(520px,1fr));
        gap: 20px; }}
section {{ background: white; border: 1px solid #ddd; padding: 12px; }}
object {{ width: 100%; aspect-ratio: 4 / 3; }}
object.detail {{ aspect-ratio: 80 / 43; }}
object.transverse {{ aspect-ratio: 104 / 108; max-width: 1040px; }}
h1,h2 {{ margin: 0 0 10px; }}
p {{ color: #444; }}
table {{ border-collapse: collapse; margin: 16px 0; }}
th,td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
th:first-child,td:first-child {{ text-align: left; }}
</style>
<h1>Ordinary ODE geometry at the saddle-connection wall</h1>
<p>Every panel uses the same wall model and exact critical points; only the
manifold discretization changes. Green and red are respectively the computed
unstable and stable branches. There are no optimizer or trajectory overlays.
Saddle capture is intentionally disabled for unstable branches: a finite-radius
stop would conceal the method's numerical choice of an outgoing N branch.</p>
<p>Fixed-step panels use <code>h={args.step_size:g}</code>. Adaptive panels use
<code>rtol={args.rtol:g}</code> and <code>atol={args.atol:g}</code>. These values
are recorded with the numerical results in the JSON report.</p>
<table><thead><tr><th>Geometry</th><th>signed midpoint mismatch</th>
<th>physical separation</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Centerline-subtracted transverse sections</h2>
<p>The common motion along the connection has been removed. Each panel has its
own printed transverse scale, so these are metrology views rather than a
shared-scale visual comparison. Green is the independently traced
<i>W</i><sup>u</sup>(B); red is <i>W</i><sup>s</sup>(N).</p>
<object class="transverse" data="{html.escape(transverse_name)}"
        type="image/svg+xml"></object>
<main>{''.join(cards)}</main>
"""
    (args.output_dir/html_name).write_text(document)
    print(args.output_dir/html_name)
    print(args.output_dir/report_name)
    return report


if __name__ == "__main__":
    main()
