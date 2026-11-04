#!/usr/bin/env python3
"""spong's certified portrait beside the standard SciPy/Matplotlib recipe.

SciPy has no phase-portrait function.  The recipe people actually use is
scipy.optimize.root from a grid of guesses, Matplotlib streamplot on a sampled
field, and solve_ivp launched along saddle eigenvectors.  This demo runs that
recipe on a spong model.  It is a demo consumer: SciPy stays out of src/spong.

    pip install -e ".[render]" scipy
    python demos/scipy_portrait_comparison.py --zoo quadratic-stiff
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
from pathlib import Path

# As the inspector does.  The engine is chosen when spong is first imported.
os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK45, Radau
from scipy.optimize import root

from spong import atlas, comparison, model, portrait, render, sturm, zoo
from spong.charts import Branch
from spong.portrait import Portrait

# The stable tails are the inspector's, so its server is the reference.
sys.path.insert(0, str(Path(__file__).resolve().parent/"explorer"))
import serve as inspector  # noqa: E402

WIDTH, HEIGHT, PAD = 1000, 640, 40   # render.plane_view's mapping

SOLVERS = {"RK45": RK45, "Radau": Radau}   # solve_ivp's default and stiff choice


def trace(m, start, solver, sign, box, minima, capture, horizon, rtol, atol,
          max_steps, max_seconds):
    """solve_ivp's stepper loop, with a budget solve_ivp itself does not have."""
    def fun(_t, y):
        return sign*-np.asarray(m.gradL(y[0], y[1]), dtype=float)

    stepper = SOLVERS[solver](fun, 0.0, np.asarray(start, float), horizon,
                              rtol=rtol, atol=atol)
    points, began, term = [np.array(start, float)], time.perf_counter(), "time_horizon"
    for count in range(max_steps):
        try:
            message = stepper.step()
        except (ValueError, FloatingPointError, OverflowError):
            term = "solver_error"
            break
        if message is not None or stepper.status == "failed":
            term = "step_failure"
            break
        y = stepper.y.copy()
        points.append(y)
        if not np.all(np.isfinite(y)):
            term = "nonfinite"
            break
        if not (box[0] <= y[0] <= box[1] and box[2] <= y[1] <= box[3]):
            term = "box_exit"
            break
        if sign > 0 and minima and min(
                np.hypot(y[0]-a, y[1]-b) for a, b in minima) < capture:
            term = "captured"
            break
        if stepper.status == "finished":
            break
        if time.perf_counter()-began > max_seconds:
            term = "wall_budget"
            break
    else:
        term = "step_budget"
    return np.array(points), term, len(points)-1, stepper.nfev


def for_display(p):
    """The same portrait with polylines thinned to a quarter pixel, for the SVG only.

    A tracer pinned against a step limit emits millions of vertices, and a
    panel that size will not render.  Diagnostics use the full polylines.
    """
    limit = .25*(p.view[1]-p.view[0])/(WIDTH-2*PAD)
    branches = []
    for br in p.branches:
        keep, last = [], None
        for i, y in enumerate(br.Y):
            if last is None or i == len(br.Y)-1 or not np.all(np.isfinite(y)) \
                    or np.hypot(y[0]-last[0], y[1]-last[1]) >= limit:
                keep.append(i)
                last = y
        branches.append(Branch(br.kind, br.Y[keep], br.term, certs=br.certs, diag=br.diag))
    return Portrait(p.model, p.enumeration, branches, p.box, p.view, p.ledger)


def fit_to_skeleton(enum):
    """The inspector's "fit to skeleton" followed by its "true angle"."""
    a = [p.a for p in enum.points]
    b = [p.b for p in enum.points]
    pad_a, pad_b = .35*(max(a)-min(a)+1), .35*(max(b)-min(b)+1)
    a0, a1, b0, b1 = min(a)-pad_a, max(a)+pad_a, min(b)-pad_b, max(b)+pad_b
    w, h = WIDTH-2*PAD, HEIGHT-2*PAD
    scale = max((a1-a0)/w, (b1-b0)/h)
    ac, bc = (a0+a1)/2, (b0+b1)/2
    return (ac-scale*w/2, ac+scale*w/2, bc-scale*h/2, bc+scale*h/2)


def inspector_portrait(m):
    """The inspector's final stage: legal trace box, certified portrait, tails."""
    bmax = atlas.legal_max_b(m)
    amax = bmax/max(1.0, math.sqrt(max(1, atlas.effective_degree(m))))
    enum = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    p = portrait.certified_compute(m, view=(-amax, amax, -bmax, bmax),
                                   _enumeration=enum)
    tails = inspector._j_tails(m, p, float(atlas.effective_degree(m)),
                               enumeration=enum)
    branches = list(p.branches)
    n_grown = n_tail = 0
    for tail in tails:    # one unbroken path, as the inspector's SVG export
        grown, closed = tail.get("grown") or [], tail.get("points") or []
        n_grown += max(len(grown)-1, 0)
        n_tail += max(len(closed)-1, 0)
        br = branches[tail["branch"]]
        parts = [br.Y]+[np.asarray(x, float)[1:] for x in (grown, closed) if len(x) > 1]
        branches[tail["branch"]] = Branch(br.kind, np.vstack(parts), br.term,
                                          certs=br.certs, diag=br.diag)
    return p, Portrait(m, p.enumeration, branches, p.box, fit_to_skeleton(p.enumeration),
                       p.ledger), len(tails)


def scipy_portrait(m, reference, solver, args, enum=None, critical_method="spong exact (given)"):
    """Trace from spong's exact saddles, or from `enum`, SciPy's own scan."""
    enum = reference.enumeration if enum is None else enum
    view = reference.view
    margin = .05*(view[1]-view[0])
    box = (view[0]-margin, view[1]+margin, view[2]-margin, view[3]+margin)
    gap = max(min((np.hypot(p.a-q.a, p.b-q.b) for i, p in enumerate(enum.points)
                   for q in enum.points[i+1:]), default=1.0), 1e-3)
    minima = [(p.a, p.b) for p in enum.points if p.kind == "min"]
    branches, terms, steps, nfev = [], {}, 0, 0
    for s in enum.saddles:
        values, vectors = np.linalg.eigh(np.asarray(m.hessL(s.a, s.b), float))
        for kind, vector, sign in (("unstable", vectors[:, 0], 1),
                                   ("stable", vectors[:, 1], -1)):
            for side in (-1, 1):
                start = np.array([s.a, s.b])+side*1e-3*gap*vector
                Y, term, n, f = trace(m, start, solver, sign, box, minima, 1e-2*gap,
                                      args.horizon, args.rtol, args.atol,
                                      args.max_steps, args.max_seconds)
                terms[term] = terms.get(term, 0)+1
                steps += n
                nfev += f
                branches.append(Branch(
                    kind, Y, term,
                    certs=comparison.integral_curve_diagnostics(
                        m, Y[np.all(np.isfinite(Y), axis=1)],
                        (view[1]-view[0])/5000.0),
                    diag={"saddle_b": s.b, "uncertified": True,
                          "launch_eigenvector_sign": side}))
    return (Portrait(m, enum, branches, box, view,
                     {"comparison": {"uncertified": True,
                                     "geometry_method": f"scipy {solver}",
                                     "critical_method": critical_method},
                      "summary": {}}),
            terms, steps, nfev)


def root_scan(m, box, grid):
    """Multistart scipy.optimize.root, classified by the numerical Hessian."""
    found = []
    for a0 in np.linspace(box[0], box[1], grid):
        for b0 in np.linspace(box[2], box[3], grid):
            sol = root(lambda z: np.asarray(m.gradL(z[0], z[1]), float),
                       [a0, b0], method="hybr")
            if not sol.success:
                continue
            a, b = sol.x
            if not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
                continue
            if all(np.hypot(a-p[0], b-p[1]) > 1e-6 for p in found):
                eig = np.linalg.eigvalsh(np.asarray(m.hessL(a, b), float))
                found.append((a, b, "min" if eig[0] > 0 else "saddle"))
    return found


def root_enumeration(found):
    """SciPy's scan in the form the tracer and renderer take; no claim of completeness."""
    return comparison.CasualEnumeration(tuple(
        comparison.CasualCriticalPoint(float(a), float(b), kind)
        for a, b, kind in sorted(found, key=lambda x: (x[1], x[0]))))


def streamplot_panel(m, view, found, path, title, n=256):
    A, B = np.meshgrid(np.linspace(view[0], view[1], n),
                       np.linspace(view[2], view[3], n))
    grad = np.array([np.asarray(m.gradL(a, b), float)
                     for a, b in zip(A.ravel(), B.ravel())])
    U, V = -grad[:, 0].reshape(A.shape), -grad[:, 1].reshape(A.shape)
    fig, ax = plt.subplots(figsize=(WIDTH/150, HEIGHT/150), dpi=150)
    ax.streamplot(A, B, U, V, density=2.0, linewidth=0.6, arrowsize=0.7,
                  color="#3a6ea5")
    for a, b, kind in found:
        ax.plot(a, b, "o" if kind == "min" else "D", ms=6, color="k")
    ax.set(xlim=view[:2], ylim=view[2:], xlabel="a", ylabel="b", title=title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--zoo", default="quadratic-stiff", choices=zoo.names())
    p.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=list(SOLVERS))
    p.add_argument("--rtol", type=float, default=1e-3, help="solve_ivp default")
    p.add_argument("--atol", type=float, default=1e-6, help="solve_ivp default")
    p.add_argument("--horizon", type=float, default=1000.0)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--max-seconds", type=float, default=10.0, help="per branch")
    p.add_argument("--root-grid", type=int, default=13)
    p.add_argument("--output-dir", type=Path, default=Path("out/scipy_comparison"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case = zoo.get(args.zoo)
    degree = max(len(case.f)-1, len(case.g)-1)
    m = model.build(case.f, case.g, zoo.moments_for(case, 2*degree+1))
    print(f"{args.zoo}: computing the certified portrait", file=sys.stderr, flush=True)
    began = time.perf_counter()
    certified, reference, n_tails = inspector_portrait(m)
    seconds = time.perf_counter()-began
    view = reference.view
    name = args.zoo
    panel = dict(view=view, width=WIDTH, height=HEIGHT, n_levels=44, n_grid=1201)
    render.save(render.plane_view(for_display(reference), title=f"{name}: certified spong", **panel),
                str(args.output_dir/f"{name}_spong.svg"))
    status = certified.ledger.get("topology", {}).get("status", "not reported")
    cards = [("Certified spong", f"{name}_spong.svg",
              f"Exact Morse skeleton: {len(reference.enumeration.points)} critical "
              f"points, {len(reference.branches)} separatrices with {n_tails} stable "
              f"tails; topology status {status}; {seconds:.1f}s.")]

    report = {"zoo": name, "exact_critical_points": len(reference.enumeration.points),
              "spong_topology": status, "scipy": {}}
    began = time.perf_counter()
    found = root_scan(m, view, args.root_grid)
    gap = min(np.hypot(p.a-q.a, p.b-q.b) for i, p in enumerate(reference.enumeration.points)
              for q in reference.enumeration.points[i+1:])
    true_hit = [any(np.hypot(p.a-x[0], p.b-x[1]) < 1e-2*gap for x in found)
                for p in reference.enumeration.points]
    spurious = sum(all(np.hypot(p.a-x[0], p.b-x[1]) >= 1e-2*gap
                       for p in reference.enumeration.points) for x in found)
    report["root_scan"] = {"grid": args.root_grid, "reported": len(found),
                           "true_found": sum(true_hit), "spurious": spurious,
                           "seconds": time.perf_counter()-began}
    stream = f"{name}_streamplot.svg"
    streamplot_panel(m, view, found, args.output_dir/stream,
                     f"{name}: scipy.optimize.root + matplotlib streamplot")
    cards.append(("root + streamplot", stream,
                  f"UNCERTIFIED. {args.root_grid}x{args.root_grid} root scan found "
                  f"{sum(true_hit)} of {len(reference.enumeration.points)} critical "
                  f"points, plus {spurious} that are not critical points; "
                  "streamlines of a sampled field, no separatrices."))

    for solver in args.solvers:
        began = time.perf_counter()
        candidate, terms, steps, nfev = scipy_portrait(m, reference, solver, args)
        elapsed = time.perf_counter()-began
        contacts = comparison.manifold_contact_diagnostics(
            m, reference.enumeration, candidate.branches, candidate.box)
        filename = f"{name}_{solver}.svg"
        render.save(render.plane_view(for_display(candidate), title=f"{name}: solve_ivp {solver} from spong's saddles (UNCERTIFIED)",
                                      **panel),
                    str(args.output_dir/filename))
        rms = max((b.certs["angle_rms_deg"] or 0.0 for b in candidate.branches), default=0.0)
        report["scipy"][solver] = {"terms": terms, "steps": steps, "nfev": nfev,
                                   "seconds": elapsed, "worst_rms_angle_deg": rms,
                                   "contact_decision": contacts["decision"]}
        cards.append((f"solve_ivp {solver}, given spong's saddles", filename,
                      f"UNCERTIFIED, launched at spong's exact saddles. rtol={args.rtol:g}, "
                      f"atol={args.atol:g}, {args.max_seconds:g} s per branch; {steps} steps, {nfev} field evaluations, "
                      f"{elapsed:.1f}s; endings {terms}; worst RMS angle defect "
                      f"{rms:.3g} deg; contact diagnostic {contacts['decision']}."))

    own = root_enumeration(found)
    report["scipy_unaided"] = {}
    for solver in args.solvers:
        began = time.perf_counter()
        candidate, terms, steps, nfev = scipy_portrait(
            m, reference, solver, args, enum=own, critical_method="scipy root (uncertified)")
        elapsed = time.perf_counter()-began
        contacts = comparison.manifold_contact_diagnostics(
            m, reference.enumeration, candidate.branches, candidate.box)
        filename = f"{name}_{solver}_unaided.svg"
        render.save(render.plane_view(
            for_display(candidate), title=f"{name}: solve_ivp {solver} from its own roots (UNCERTIFIED)",
            **panel), str(args.output_dir/filename))
        report["scipy_unaided"][solver] = {
            "saddles_launched": len(own.saddles), "minima_used": len(own.minima),
            "terms": terms, "steps": steps, "nfev": nfev, "seconds": elapsed,
            "contact_decision": contacts["decision"]}
        cards.append((f"solve_ivp {solver}, unaided", filename,
                      f"UNCERTIFIED, launched from SciPy's own {len(own.saddles)} 'saddles' "
                      f"with {len(own.minima)} 'minima'. {args.max_seconds:g} s per branch; {steps} steps, {nfev} field evaluations, "
                      f"{elapsed:.1f}s; endings {terms}; contact diagnostic {contacts['decision']}."))

    (args.output_dir/f"{name}_report.json").write_text(json.dumps(report, indent=2, default=str))
    body = "".join(f"<section><h2>{html.escape(t)}</h2><object data='{f}' type='image/svg+xml'>"
                   f"</object><p>{html.escape(n)}</p></section>" for t, f, n in cards)
    out = args.output_dir/f"{name}_comparison.html"
    out.write_text("<!doctype html><meta charset=utf-8><title>spong vs SciPy</title><style>"
                   "body{font:14px system-ui;margin:16px}main{display:grid;gap:16px;"
                   "grid-template-columns:repeat(auto-fit,minmax(480px,1fr))}object{width:100%}"
                   f"</style><h1>{html.escape(name)}: spong vs the SciPy recipe</h1><main>{body}</main>")
    print(json.dumps(report, indent=2, default=str))
    print(out)


if __name__ == "__main__":
    main()
