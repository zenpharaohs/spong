"""Many-start optimizer trajectories over certified SPONG portraits.

Run, for example:

    PYTHONPATH=src:. python demos/optimizer_moustaches.py \
        --zoo quadratic-stiff --starts 100

The certified portrait is the reference instrument.  Every optimizer overlay
is explicitly empirical and receives the same initialization design and,
within each start/batch-size pair, the same random-number stream seed.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from demos import initializers
from demos import optimizers as opt
from spong import model, portrait, render, zoo


METHODS = ("sgd", "sgd-momentum", "adam", "muon-adamw", "vector-muon")
DEFAULT_METHODS = ("sgd", "sgd-momentum", "adam")
DESIGNS = ("low-discrepancy", "blue-noise")
COLORS = {
    "sgd": "#2060e0",
    "sgd-momentum": "#e07020",
    "adam": "#9020c0",
    "muon-adamw": "#008c83",
    "vector-muon": "#008c83",
}
DEFAULT_LR = {
    "sgd": 1e-2,
    "sgd-momentum": 1e-2,
    "adam": 1e-3,
    "muon-adamw": 3e-4,
    "vector-muon": 2e-2,
}


def _zoo_model(case):
    degree = max(len(case.f)-1, len(case.g)-1)
    moments = (model.moments_uniform01
               if case.moment_dist == "uniform01"
               else model.moments_normal01)(2*degree+1)
    return model.build(case.f, case.g, moments)


def _schedule(name, base_lr, steps):
    if name == "constant":
        return float(base_lr)
    if name == "cosine":
        return opt.cosine_schedule(base_lr, steps)
    if name == "inverse-sqrt":
        return opt.inverse_sqrt_schedule(
            base_lr, warmup_steps=max(1, int(0.05*steps)))
    raise ValueError(name)


def _run(method, gradient, start, lr, steps, box):
    if method == "sgd":
        return opt.run_sgd(gradient, start, lr, steps, box=box)
    if method == "sgd-momentum":
        return opt.run_sgd(
            gradient, start, lr, steps, box=box,
            momentum=0.9, nesterov=True)
    if method == "adam":
        return opt.run_adam(gradient, start, lr, steps, box=box)
    if method == "muon-adamw":
        # Faithful parameter grouping: (a,b) is a vector, so a practical
        # Muon-with-auxiliary-Adam implementation sends it to AdamW.
        return opt.run_adamw(
            gradient, start, lr, steps, box=box,
            b1=0.9, b2=0.95, eps=1e-10, weight_decay=0.0)
    if method == "vector-muon":
        return opt.run_vector_muon(
            gradient, start, lr, steps, box=box,
            momentum=0.95, nesterov=True)
    raise ValueError(method)


def _thin(trajectory, max_points=350, *, bucket_axis="linear"):
    """LTTB display thinning in the (a, b) plane, bucketed by step order.

    Keep the first and last finite iterates.  In each step-order bucket,
    select the point making the largest triangle with the previous
    selection and the mean of the next bucket.  This preserves excursions and
    bends that uniform index sampling can skip entirely.  Log-step buckets
    reserve more of the same point budget for early iterates.
    """
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory must have shape (steps, 2)")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if bucket_axis not in {"linear", "log"}:
        raise ValueError("bucket_axis must be linear or log")
    finite = np.all(np.isfinite(points), axis=1)
    if not np.all(finite):
        points = points[:int(np.flatnonzero(~finite)[0])]
    n = len(points)
    if n <= max_points:
        return points

    # Normalize before taking cross products so even very large but finite
    # iterates cannot overflow the area calculation.  A fixed scale per axis
    # multiplies every triangle area by the same factor, preserving its rank.
    scale = np.max(np.abs(points), axis=0)
    scale[scale == 0.0] = 1.0
    xy = points / scale
    buckets = max_points - 2
    interior = n - 2
    indices = np.empty(max_points, dtype=np.intp)
    indices[0], indices[-1] = 0, n - 1
    if bucket_axis == "log":
        log_steps = np.log1p(np.arange(n, dtype=float))
        edges = [1]
        for i in range(1, buckets):
            boundary = log_steps[n-2] * i / buckets
            target = int(np.searchsorted(log_steps, boundary, side="left"))
            # Every bucket needs a point, including those in early decades
            # narrower than one optimizer step.
            edges.append(max(edges[-1] + 1,
                             min(target, n-1-(buckets-i))))
        edges.append(n-1)
    previous = 0
    for i in range(buckets):
        if bucket_axis == "log":
            start, end = edges[i:i+2]
            next_start = end
            next_end = edges[i+2] if i+1 < buckets else n
        else:
            start = 1 + i * interior // buckets
            end = 1 + (i + 1) * interior // buckets
            next_start = end
            next_end = (1 + (i + 2) * interior // buckets
                        if i + 1 < buckets else n)
        center = np.mean(xy[next_start:next_end], axis=0)
        candidates = xy[start:end]
        delta = candidates - xy[previous]
        toward_center = center - xy[previous]
        areas = np.abs(delta[:, 0] * toward_center[1]
                       - delta[:, 1] * toward_center[0])
        previous = start + int(np.argmax(areas))
        indices[i + 1] = previous
    return points[indices]


def _thin_comparison(trajectory, uniform_steps, prefix_points=350,
                     extension_points=250):
    """Give an extended arm its own budget without diluting its shared prefix.

    The prefix ends at the uniform arm's last iterate and uses the same LTTB
    selection as the gray trace.  Additional bandit steps receive a separate
    log-step budget, concentrating detail near the start of the extension
    instead of spending most segments on a long late noise ball.
    """
    points = np.asarray(trajectory, dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.all(finite):
        points = points[:int(np.flatnonzero(~finite)[0])]
    if len(points) == 0:
        return points
    split = min(len(points), max(1, int(uniform_steps) + 1))
    prefix = _thin(points[:split], max_points=prefix_points)
    extra_steps = len(points) - split
    if extra_steps == 0:
        return prefix
    extra_budget = max(2, min(extension_points, int(np.ceil(
        extension_points * extra_steps / (len(points)-1)))))
    extension = _thin(points[split-1:], max_points=extra_budget+1,
                      bucket_axis="log")
    return np.concatenate((prefix, extension[1:]))


def _loss(m, point):
    if not np.all(np.isfinite(point)):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(m.L(*point))
    return value


def _summary(m, p, trajectories, box):
    losses = np.asarray([_loss(m, tr[-1]) for tr in trajectories])
    finite = losses[np.isfinite(losses)]
    transformed = 1-1/(1+finite)
    escaped = sum(
        not np.all(np.isfinite(tr[-1]))
        or not (box[0] <= tr[-1, 0] <= box[1]
                and box[2] <= tr[-1, 1] <= box[3])
        for tr in trajectories)
    span = max(box[1]-box[0], box[3]-box[2], 1.0)
    captured = 0
    fates = {}
    for tr in trajectories:
        if not np.all(np.isfinite(tr[-1])):
            continue
        critical, distance = opt.nearest_critical(p.enumeration, tr[-1])
        if distance <= 2e-3*span:
            captured += 1
            key = f"{critical.kind}@b={critical.b:.8g}"
            fates[key] = fates.get(key, 0)+1
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {
        "runs": len(trajectories),
        "escaped": escaped,
        "captured": captured,
        "steps": [len(tr)-1 for tr in trajectories],
        "final_loss_quantiles": (
            dict(zip(map(str, quantiles), map(float, np.quantile(finite, quantiles))))
            if len(finite) else None),
        "transformed_loss_quantiles": (
            dict(zip(map(str, quantiles),
                     map(float, np.quantile(transformed, quantiles))))
            if len(transformed) else None),
        "critical_fates": fates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Overlay many-start descent trajectories on SPONG")
    parser.add_argument("--zoo", choices=zoo.names(),
                        default="quadratic-stiff")
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--designs", nargs="+", choices=DESIGNS,
                        default=list(DESIGNS))
    parser.add_argument("--schedule",
                        choices=("constant", "cosine", "inverse-sqrt"),
                        default="cosine")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/optimizer_moustaches"))
    args = parser.parse_args(argv)
    if args.starts <= 0 or args.steps <= 0:
        parser.error("--starts and --steps must be positive")
    if any(batch <= 0 for batch in args.batch_sizes):
        parser.error("batch sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = zoo.get(args.zoo)
    if case.moment_dist != "uniform01":
        parser.error(
            "the current raw-minibatch oracle supports uniform01 zoo cases")
    m = _zoo_model(case)
    p = portrait.certified_compute(m, view=case.default_view)
    view = p.view
    reference_name = f"{case.name}_certified.svg"
    render.save(render.plane_view(
        p, view=view, width=args.width, height=args.height,
        n_levels=32, n_grid=801,
        title=f"{case.name}: certified steepest-descent portrait"),
        str(args.output_dir/reference_name))

    design_points = {}
    for design in args.designs:
        design_points[design] = (
            initializers.low_discrepancy(args.starts, view)
            if design == "low-discrepancy"
            else initializers.blue_noise(args.starts, view, seed=args.seed))

    panels = []
    records = []
    for design in args.designs:
        starts = design_points[design]
        for batch_size in args.batch_sizes:
            for method in args.methods:
                trajectories = []
                overlays = []
                for index, start in enumerate(starts):
                    # Common random numbers across optimizer methods.
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [args.seed, index, batch_size,
                             0 if design == "low-discrepancy" else 1]))
                    gradient = opt.BatchGradient(
                        case.f, case.g, batch_size=batch_size, rng=rng)
                    lr = _schedule(
                        args.schedule, DEFAULT_LR[method], args.steps)
                    trajectory = _run(
                        method, gradient, start, lr, args.steps, p.box)
                    trajectories.append(trajectory)
                    overlays.append({
                        "Y": _thin(trajectory),
                        "color": COLORS[method],
                        "width": 0.65,
                        "opacity": 0.18,
                        "mark_start": True,
                        "mark_end": True,
                    })
                label = (
                    f"{method}; {design}; batch={batch_size}; "
                    f"{args.schedule}; n={args.starts}")
                overlays[0]["label"] = label
                filename = (
                    f"{case.name}_{design}_{method}_b{batch_size}.svg")
                render.save(render.plane_view(
                    p, view=view, width=args.width, height=args.height,
                    n_levels=32, n_grid=801, overlays=overlays,
                    title=f"{case.name}: {label}"),
                    str(args.output_dir/filename))
                summary = _summary(m, p, trajectories, p.box)
                median = (
                    f'{summary["final_loss_quantiles"]["0.5"]:.5g}'
                    if summary["final_loss_quantiles"] is not None
                    else "nonfinite")
                records.append({
                    "method": method,
                    "muon_semantics": (
                        ("2x1 normalized-momentum surrogate"
                         if method == "vector-muon"
                         else ("AdamW auxiliary parameter group; "
                               "weight_decay=0"
                               if method == "muon-adamw" else None))),
                    "design": design,
                    "batch_size": batch_size,
                    "schedule": args.schedule,
                    "base_learning_rate": DEFAULT_LR[method],
                    "svg": filename,
                    **summary,
                })
                panels.append(
                    f'<section><h2>{html.escape(label)}</h2>'
                    f'<object data="{html.escape(filename)}" '
                    f'type="image/svg+xml"></object>'
                    f'<p>captured={summary["captured"]}; '
                    f'escaped={summary["escaped"]}; '
                    f'median final loss={median}</p>'
                    f'</section>')

    report = {
        "format": "spong-optimizer-moustaches-v1",
        "zoo": case.name,
        "reference": reference_name,
        "configuration": {
            "starts": args.starts,
            "steps": args.steps,
            "batch_sizes": args.batch_sizes,
            "methods": args.methods,
            "designs": args.designs,
            "schedule": args.schedule,
            "seed": args.seed,
        },
        "muon_applicability": {
            "applicable": False,
            "reason": (
                "The trainable state (a,b) is a vector, while Muon is "
                "defined for matrix-valued hidden weights. Practical Muon "
                "routes this parameter group to auxiliary AdamW."
            ),
            "optional_diagnostics": {
                "muon-adamw": "auxiliary AdamW defaults",
                "vector-muon": "2x1 normalized-momentum surrogate",
            },
        },
        "comparisons": records,
    }
    report_name = f"{case.name}_optimizer_moustaches.json"
    (args.output_dir/report_name).write_text(
        json.dumps(report, indent=2)+"\n")
    index_name = f"{case.name}_optimizer_moustaches.html"
    (args.output_dir/index_name).write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG optimizer moustaches: {html.escape(case.name)}</title>
<style>
body {{ font-family: system-ui,sans-serif; margin:24px; background:#fafafa; }}
main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(620px,1fr));
        gap:20px; }}
section {{ background:white; border:1px solid #ddd; padding:12px; }}
object {{ width:100%; aspect-ratio:4/3; }}
h1,h2 {{ margin:0 0 10px; }} p {{ color:#444; }}
</style>
<h1>Optimizer moustaches on {html.escape(case.name)}</h1>
<p>The underlying portrait is certified; optimizer trajectories are empirical.
Muon is inapplicable to the vector parameter (a,b); practical Muon routes it
to AdamW. The duplicate fallback and a 2×1 normalized-momentum surrogate are
available only as optional diagnostics.</p>
<main>{''.join(panels)}</main>
""")
    print(args.output_dir/index_name)
    print(args.output_dir/report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
