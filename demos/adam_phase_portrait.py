"""Compare a certified SPONG portrait with the limiting Adam semiflow."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from demos import adam_flow
from spong import model, portrait, render, zoo


def _empirical_model(case, sample_size):
    inputs, rational_inputs = adam_flow.empirical_uniform_grid(sample_size)
    degree = max(len(case.f)-1, len(case.g)-1)
    moments = adam_flow.empirical_moments(rational_inputs, 2*degree+1)
    return model.build(case.f, case.g, moments), inputs


def _path(points, mapper, color, opacity=.42, width=.7):
    pixels = [mapper(*point) for point in points]
    data = "M"+" L".join(f"{x:.6g},{y:.6g}" for x, y in pixels)
    return (f'<path d="{data}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            'stroke-linecap="round" vector-effect="non-scaling-stroke"/>')


def _arrow(points, mapper, color):
    if len(points) < 8:
        return ""
    k = int(.62*(len(points)-1))
    p0 = np.asarray(mapper(*points[max(0, k-2)]))
    p1 = np.asarray(mapper(*points[min(len(points)-1, k+2)]))
    tangent = p1-p0
    norm = float(np.hypot(*tangent))
    if norm == 0:
        return ""
    tangent /= norm
    normal = np.array([-tangent[1], tangent[0]])
    tip = np.asarray(mapper(*points[k]))+4*tangent
    left = tip-8*tangent+3.2*normal
    right = tip-8*tangent-3.2*normal
    coords = " ".join(
        f"{p[0]:.6g},{p[1]:.6g}" for p in (tip, left, right))
    return f'<polygon points="{coords}" fill="{color}" fill-opacity=".72"/>'


def _adam_svg(p, estimate, uphill, exact_zeros, width=800, height=600):
    empty = portrait.Portrait(
        p.model, p.enumeration, [], p.box, p.view, ledger={})
    base = render.plane_view(
        empty, view=p.view, width=width, height=height,
        n_levels=32, n_grid=801,
        title="Limiting stochastic Adam field (numerical oracle)")
    mapper = render._mapper(p.view, width, height, 42)
    shading = ['<g id="adam-uphill" fill="#d73027" fill-opacity=".16">']
    for j in range(len(estimate.b)-1):
        for i in range(len(estimate.a)-1):
            if not np.any(uphill[j:j+2, i:i+2]):
                continue
            x0, y0 = mapper(estimate.a[i], estimate.b[j])
            x1, y1 = mapper(estimate.a[i+1], estimate.b[j+1])
            shading.append(
                f'<rect x="{min(x0, x1):.6g}" y="{min(y0, y1):.6g}" '
                f'width="{abs(x1-x0):.6g}" height="{abs(y1-y0):.6g}"/>')
    shading.append("</g>")
    base_parts = base.splitlines()
    base_parts[2:2] = shading
    base = "\n".join(base_parts)
    extra = [
        '<g id="adam-field" aria-label="estimated Adam streamlines">']
    for curve in adam_flow.streamlines(estimate):
        extra.append(_path(curve, mapper, "#3155c6"))
        extra.append(_arrow(curve, mapper, "#3155c6"))
    for a, b in adam_flow.sign_change_zero_candidates(estimate):
        x, y = mapper(a, b)
        extra.append(
            f'<path d="M{x-4:.6g},{y-4:.6g} L{x+4:.6g},{y+4:.6g} '
            f'M{x-4:.6g},{y+4:.6g} L{x+4:.6g},{y-4:.6g}" '
            'stroke="#7a1fa2" stroke-width="1.5"/>')
    for zero in exact_zeros:
        x, y = mapper(zero.a, zero.b)
        extra.append(
            f'<circle cx="{x:.6g}" cy="{y:.6g}" r="6" fill="none" '
            'stroke="#0057b8" stroke-width="2.4"/>')
    extra.append("</g>")
    return base.replace("</svg>", "\n".join(extra)+"\n</svg>")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare certified steepest descent and limiting Adam flow")
    parser.add_argument("--zoo", choices=zoo.names(), default="quadratic-stiff")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid", type=int, default=35)
    parser.add_argument("--alpha", type=float, default=.9)
    parser.add_argument("--beta", type=float, default=.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--burn-in", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/adam_phase_portrait"))
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    case = zoo.get(args.zoo)
    if case.moment_dist != "uniform01":
        parser.error("the first Adam oracle supports uniform01 cases")
    m, inputs = _empirical_model(case, args.sample_size)
    p = portrait.certified_compute(m, view=case.default_view)
    estimate = adam_flow.estimate_grid(
        case.f, case.g, inputs, p.view, grid=args.grid,
        batch_size=args.batch_size, alpha=args.alpha, beta=args.beta,
        epsilon=args.epsilon, burn_in=args.burn_in, samples=args.samples,
        chains=args.chains, seed=args.seed)

    aa, bb = np.meshgrid(estimate.a, estimate.b)
    gradients = np.asarray([
        m.gradL(a, b) for a, b in zip(aa.ravel(), bb.ravel())
    ]).reshape(aa.shape+(2,))
    directional = np.sum(gradients*estimate.field, axis=2)
    uphill = directional > 0
    candidates = adam_flow.sign_change_zero_candidates(estimate)
    loss_scale = max(
        [abs(float(m.L(q.a, q.b))) for q in p.enumeration.points]+[1.0])
    zero_tolerance = 1e-12*loss_scale
    exact_zeros = [
        q for q in p.enumeration.points
        if abs(float(m.L(q.a, q.b))) <= zero_tolerance]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.name}_empirical_n{args.sample_size}"
    steepest_name = f"{stem}_steepest.svg"
    adam_name = f"{stem}_adam.svg"
    render.save(render.plane_view(
        p, view=p.view, width=800, height=600, n_levels=32, n_grid=801,
        title=f"{case.name}: certified steepest descent"),
        str(args.output_dir/steepest_name))
    render.save(
        _adam_svg(p, estimate, uphill, exact_zeros),
        str(args.output_dir/adam_name))

    span = np.array((p.view[1]-p.view[0], p.view[3]-p.view[2]))
    critical = list(p.enumeration.points)
    candidate_records = []
    for candidate in candidates:
        nearest = min(
            critical,
            key=lambda q: np.linalg.norm(
                (candidate-np.array((q.a, q.b)))/span))
        normalized_distance = float(np.linalg.norm(
            (candidate-np.array((nearest.a, nearest.b)))/span))
        boundary_distance = min(
            (candidate[0]-p.view[0])/span[0],
            (p.view[1]-candidate[0])/span[0],
            (candidate[1]-p.view[2])/span[1],
            (p.view[3]-candidate[1])/span[1])
        nearest_is_exact = (
            abs(float(m.L(nearest.a, nearest.b))) <= zero_tolerance)
        candidate_records.append({
            "a": float(candidate[0]),
            "b": float(candidate[1]),
            "near_view_boundary": bool(boundary_distance < 1/args.grid),
            "nearest_loss_critical": {
                "a": nearest.a, "b": nearest.b, "kind": nearest.kind},
            "normalized_displacement": normalized_distance,
            "nearest_is_exact_zero_loss": nearest_is_exact,
            "interpretation": (
                "coarse-grid interpolation error; the exact zero-loss point "
                "is analytically an Adam equilibrium"
                if nearest_is_exact else
                "possible Adam/loss equilibrium displacement; requires "
                "local refinement"),
        })

    report = {
        "format": "spong-adam-phase-portrait-v1",
        "status": "numerical-oracle-not-certified",
        "zoo": case.name,
        "empirical_sample_size": args.sample_size,
        "empirical_support": "exact midpoint grid on [0,1]",
        "adam": estimate.diagnostics,
        "uphill_grid_fraction": float(np.mean(uphill)),
        "exact_zero_loss_adam_equilibria": [
            {"a": q.a, "b": q.b} for q in exact_zeros],
        "adam_zero_candidates": candidate_records,
        "loss_critical_points": [
            {"a": q.a, "b": q.b, "kind": q.kind,
             "loss": float(m.L(q.a, q.b))}
            for q in p.enumeration.points],
        "steepest_svg": steepest_name,
        "adam_svg": adam_name,
    }
    report_name = f"{stem}.json"
    (args.output_dir/report_name).write_text(
        json.dumps(report, indent=2)+"\n")
    index = args.output_dir/f"{stem}.html"
    diag = estimate.diagnostics
    index.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG versus limiting Adam: {html.escape(case.name)}</title>
<style>
body {{ font-family:system-ui,sans-serif; margin:24px; background:#fafafa; }}
main {{ display:grid; grid-template-columns:repeat(2,minmax(520px,1fr)); gap:20px; }}
section {{ background:white; border:1px solid #ddd; padding:12px; }}
object {{ width:100%; aspect-ratio:4/3; }}
.warning {{ color:#8a4b00; }}
</style>
<h1>Steepest descent versus limiting Adam</h1>
<p>Both panels use the same finite empirical sample of {args.sample_size}
midpoints. The steepest portrait is SPONG-certified. The Adam field is a
stationary-history numerical oracle with {args.chains} independent chains,
{args.burn_in} burn-in updates, and {args.samples} averaging updates.</p>
<p class="warning">Adam field chain disagreement: median resolved angle
{diag.get('resolved_angle_degrees_median')}°, 95th percentile
{diag.get('resolved_angle_degrees_95pct')}°. Purple crosses are bilinear-grid
roots that only nominate, rather than certify, Adam zeros. Translucent red cells
mark sampled regions where the Adam field points uphill for the loss. Blue
rings mark exact zero-loss points, which are analytically exact Adam
equilibria; any nearby purple displacement is interpolation error.</p>
<main>
<section><h2>Certified steepest descent</h2>
<object data="{html.escape(steepest_name)}" type="image/svg+xml"></object>
</section>
<section><h2>Limiting stochastic Adam field</h2>
<object data="{html.escape(adam_name)}" type="image/svg+xml"></object>
</section>
</main>
""")
    print(index)
    print(args.output_dir/report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
