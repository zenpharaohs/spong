"""Generate certified-versus-textbook phase-portrait comparison panels."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from . import comparison, model, portrait, render, zoo


def _zoo_model(case):
    degree = max(len(case.f)-1, len(case.g)-1)
    moments = (model.moments_uniform01
               if case.moment_dist == "uniform01"
               else model.moments_normal01)(2*degree+1)
    return model.build(case.f, case.g, moments)


def _card(label, filename, note, zoom=None):
    zoom_html = (
        f'<h3>Bottom-canyon zoom</h3>'
        f'<object data="{html.escape(zoom)}" type="image/svg+xml"></object>'
        if zoom else "")
    return f"""<section>
<h2>{html.escape(label)}</h2>
<object data="{html.escape(filename)}" type="image/svg+xml"></object>
{zoom_html}
<p>{html.escape(note)}</p>
</section>"""


def _bottom_zoom(reference):
    branches = [br for br in reference.branches
                if br.kind == "unstable" and len(br.Y) >= 2]
    if not branches:
        return None
    branch = min(branches, key=lambda br: float(np.min(br.Y[:, 1])))
    Y = np.asarray(branch.Y, dtype=float)
    finite = np.all(np.isfinite(Y), axis=1)
    Y = Y[finite]
    if len(Y) < 2:
        return None
    bmin, bmax = float(Y[:, 1].min()), float(Y[:, 1].max())
    low = Y[Y[:, 1] <= bmin+0.35*(bmax-bmin)]
    if len(low) < 2:
        low = Y
    amin, amax = float(low[:, 0].min()), float(low[:, 0].max())
    bmin, bmax = float(low[:, 1].min()), float(low[:, 1].max())
    da, db = max(0.08*(amax-amin), 1e-3), max(0.08*(bmax-bmin), 1e-3)
    return (amin-da, amax+da, bmin-db, bmax+db)


def _branch_record(m, branch, spacing):
    Y = np.asarray(branch.Y, dtype=float)
    finite = Y[np.all(np.isfinite(Y), axis=1)]
    geometry = comparison.integral_curve_diagnostics(m, finite, spacing)
    return {
        "kind": branch.kind,
        "term": branch.term,
        "saddle_b": branch.diag.get("saddle_b"),
        "points": len(Y),
        "a_range": (
            [float(finite[:, 0].min()), float(finite[:, 0].max())]
            if len(finite) else None),
        "b_range": (
            [float(finite[:, 1].min()), float(finite[:, 1].max())]
            if len(finite) else None),
        "elapsed_time": branch.diag.get("elapsed_time"),
        "min_accepted_step": branch.diag.get("min_accepted_step"),
        "max_accepted_step": branch.diag.get("max_accepted_step"),
        **geometry,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare SPONG with ordinary phase-portrait algorithms")
    parser.add_argument("--zoo", choices=zoo.names(),
                        default="quadratic-stiff")
    parser.add_argument(
        "--methods", nargs="+", choices=comparison.GEOMETRY_METHODS,
        default=list(comparison.GEOMETRY_METHODS))
    parser.add_argument("--critical-method",
                        choices=("certified", "grid-newton"),
                        default="certified")
    parser.add_argument("--critical-grid", type=int, default=17)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument(
        "--time-horizon", type=float,
        help="common physical final time; default is step-size*max-steps")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--no-bottom-zoom", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/comparisons"))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = zoo.get(args.zoo)
    m = _zoo_model(case)
    reference = portrait.certified_compute(m, view=case.default_view)
    view = reference.view
    span = max(view[1]-view[0], view[3]-view[2], 1.0)
    diagnostic_spacing = span/5000.0
    reference_branch_records = [
        _branch_record(m, branch, diagnostic_spacing)
        for branch in reference.branches
    ]
    zoom_view = None if args.no_bottom_zoom else _bottom_zoom(reference)
    cards = []
    reference_name = f"{case.name}_certified.svg"
    render.save(render.plane_view(
        reference, view=view, width=args.width, height=args.height,
        n_levels=32, n_grid=801,
        title=f"{case.name}: certified SPONG portrait"),
        str(args.output_dir/reference_name))
    reference_zoom = None
    if zoom_view is not None:
        reference_zoom = f"{case.name}_certified_zoom.svg"
        render.save(render.plane_view(
            reference, view=zoom_view, width=args.width, height=args.height,
            n_levels=24, n_grid=801,
            title=f"{case.name}: certified bottom-canyon zoom"),
            str(args.output_dir/reference_zoom))
    cards.append(_card(
        "Certified SPONG",
        reference_name,
        "Exact Morse skeleton, conditioned manifold stubs, implicit Gauss "
        "continuation, and a-posteriori separatrix audit.",
        reference_zoom))

    records = []
    for method in args.methods:
        candidate = comparison.casual_portrait(
            m, method, critical_method=args.critical_method,
            reference_enumeration=reference.enumeration, view=view,
            step_size=args.step_size, max_steps=args.max_steps,
            time_horizon=args.time_horizon,
            rtol=args.rtol, atol=args.atol,
            critical_grid=args.critical_grid)
        filename = (
            f"{case.name}_{args.critical_method}_{method}.svg")
        render.save(render.plane_view(
            candidate, view=view, width=args.width, height=args.height,
            n_levels=32, n_grid=801,
            title=f"{case.name}: {method}"),
            str(args.output_dir/filename))
        zoom_name = None
        if zoom_view is not None:
            zoom_name = (
                f"{case.name}_{args.critical_method}_{method}_zoom.svg")
            render.save(render.plane_view(
                candidate, view=zoom_view,
                width=args.width, height=args.height,
                n_levels=24, n_grid=801,
                title=f"{case.name}: {method} bottom-canyon zoom"),
                str(args.output_dir/zoom_name))
        terms = {}
        for branch in candidate.branches:
            terms[branch.term] = terms.get(branch.term, 0)+1
        branch_records = [
            _branch_record(m, branch, diagnostic_spacing)
            for branch in candidate.branches
        ]
        unstable_records = [
            branch for branch in branch_records
            if branch["kind"] == "unstable" and branch["b_range"] is not None
        ]
        bottom_branch = (
            min(unstable_records, key=lambda branch: branch["b_range"][0])
            if unstable_records else None)
        record = {
            **candidate.ledger["comparison"],
            "branches": len(candidate.branches),
            "terms": terms,
            "critical_points": len(candidate.enumeration.points),
            "saddles": len(candidate.enumeration.saddles),
            "worst_angle_energy_common": max(
                (br.certs["angle_energy_common"] for br in candidate.branches),
                default=0.0),
            "worst_angle_rms_deg": max(
                (br.certs["angle_rms_deg"] or 0.0
                 for br in candidate.branches), default=0.0),
            "worst_angle_max_deg": max(
                (br.certs["angle_max_deg"] or 0.0
                 for br in candidate.branches), default=0.0),
            "elapsed_time_range": [
                min((br.diag["elapsed_time"] for br in candidate.branches),
                    default=0.0),
                max((br.diag["elapsed_time"] for br in candidate.branches),
                    default=0.0),
            ],
            "bottom_unstable_branch": bottom_branch,
            "branch_diagnostics": branch_records,
            "svg": filename,
            "zoom_svg": zoom_name,
        }
        records.append(record)
        cards.append(_card(
            method,
            filename,
            f"Uncertified {args.critical_method} critical points; "
            f"initial dt={args.step_size:g}; common time horizon="
            f"{candidate.ledger['comparison']['time_horizon']:g}; "
            f"rtol={args.rtol:g}, atol={args.atol:g}; "
            f"worst RMS angle defect={record['worst_angle_rms_deg']:.3g}°; "
            f"branch terms={terms}.",
            zoom_name))

    report = {
        "format": "spong-portrait-comparison-v1",
        "zoo": case.name,
        "description": case.description,
        "reference": reference_name,
        "reference_diagnostics": {
            "worst_angle_rms_deg": max(
                (branch["angle_rms_deg"] or 0.0
                 for branch in reference_branch_records), default=0.0),
            "worst_angle_max_deg": max(
                (branch["angle_max_deg"] or 0.0
                 for branch in reference_branch_records), default=0.0),
            "branch_diagnostics": reference_branch_records,
        },
        "configuration": {
            "critical_method": args.critical_method,
            "critical_grid": args.critical_grid,
            "step_size": args.step_size,
            "max_steps": args.max_steps,
            "time_horizon": args.time_horizon,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "comparisons": records,
    }
    report_name = f"{case.name}_comparison.json"
    (args.output_dir/report_name).write_text(
        json.dumps(report, indent=2)+"\n")
    index_name = f"{case.name}_comparison.html"
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG comparison: {html.escape(case.name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(520px,1fr));
        gap: 20px; }}
section {{ background: white; border: 1px solid #ddd; padding: 12px; }}
object {{ width: 100%; aspect-ratio: 4 / 3; }}
h1,h2 {{ margin: 0 0 10px; }} p {{ color: #444; }}
</style>
<h1>SPONG portrait comparison: {html.escape(case.name)}</h1>
<p>{html.escape(case.description)}</p>
<main>{''.join(cards)}</main>
"""
    (args.output_dir/index_name).write_text(document)
    print(args.output_dir/index_name)
    print(args.output_dir/report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
