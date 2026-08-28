"""Three arms for the stall class: geometry, units, both.

Supersedes box_experiment.py, whose clustering floored a singleton's span at
1e-6 and therefore split EVERY gap: four singleton clusters, max-by-size took
the first (the remote minimum at b = -40960), no saddle selected, empty
table.  The Aug 26 run was null, not negative.

The two cheap hypotheses left for seed 953953598's family, after the
degeneracy and perturbation probes closed the model-side routes:

  ARM A (geometry): cut the box to the inner critical cluster and trace its
      saddle branches directly.  The remote minimum is outside the box, so
      this is NOT a certification -- it asks only whether the chord/box
      coupling (ds set by a skeleton 1e5 times wider than the inner
      geometry) is what stalls the tracer.

  ARM B (units): normalise f by ||f||_inf and g by ||g||_inf, exactly, and
      run the PRODUCTION pipeline (portrait.certified_compute) unchanged.
      L~(a,b) = L(tau a/sigma, b)/tau^2: a diffeomorphism in a composed with
      a positive scaling of the loss, so the level-set/merge-tree topology
      the certifier decides is invariant, while a*(0) falls from E[f]/g(0)
      (-4.4e7 here) to +-kappa (~58).  If this arm certifies, it IS a
      certification of the original model's topology, and normalisation is a
      preprocessing fix requiring no engine change.

  ARM C (both): the inner-cluster trace of arm A, on the normalised model.

Decision table: A clean only -> decouple tracing scale from the skeleton
box.  B clean -> preprocessing fix; sweep kappa over the branch_abort
population to predict conversions.  Only C clean -> both levers needed.
None clean -> the intrinsic range ~kappa also defeats the tracer, and the
Sundman/collocation route (mse-bundle) earns its slot.

Stalls terminate through the engine's internal step cap and report as
abort_max_steps; there is no external cap to pass.  Results are printed AND
written to out/box_arms_<seed>.json so nothing lives only in a transcript.

    python scripts/box_experiment_arms.py 953953598 --mode directed
    python scripts/box_experiment_arms.py 953953598 --arms A,C   # skip the
                                                    # (possibly long) arm B
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

from spong import atlas, charts, engine, model, portrait, sturm  # noqa: E402
from qualify import directed_model, random_model          # noqa: E402


# ------------------------------------------------------------------ #
# model measurement and exact normalisation
# ------------------------------------------------------------------ #

def measure(m):
    """The scaling_sweep diagnostics, inline: kappa and a*(0)."""
    fscale = max((abs(c) for c in m.f), default=Fraction(0))
    gscale = max((abs(c) for c in m.g), default=Fraction(0))
    g0 = m.g[0] if len(m.g) else Fraction(0)
    Ef = sum((c * mm for c, mm in zip(m.f, m.mu)), Fraction(0))
    astar0 = float(Ef / g0) if g0 else float("inf")
    kappa = (abs(astar0) * float(gscale) / float(fscale)
             if fscale else float("inf"))
    return {"astar0": astar0, "kappa": kappa,
            "fscale": float(fscale), "gscale": float(gscale)}


def normalised(m):
    """Rebuild with f/||f||_inf and g/||g||_inf, exactly.

    Fraction/Fraction is exact, and model.build reconstructs A, B, C, N and
    the reduced backbone from scratch, so every derived certificate object
    belongs to the normalised model in its own right.
    """
    tau = max(abs(c) for c in m.f)
    sigma = max(abs(c) for c in m.g)
    if tau == 0 or sigma == 0:
        raise ValueError("degenerate model: f or g identically zero")
    return model.build(tuple(c / tau for c in m.f),
                       tuple(c / sigma for c in m.g), m.mu)


# ------------------------------------------------------------------ #
# corrected clustering
# ------------------------------------------------------------------ #

def cluster_skeleton(pts, gap_factor):
    """Split at consecutive b-gaps far larger than the typical gap.

    The threshold is gap_factor times the MEDIAN inter-point gap -- a
    property of the whole skeleton, never of the (possibly singleton)
    running cluster, which is what broke box_experiment.py.
    """
    if len(pts) <= 1:
        return [list(pts)]
    gaps = [float(q.b) - float(p.b) for p, q in zip(pts, pts[1:])]
    threshold = gap_factor * statistics.median(gaps)
    groups, cur = [], [pts[0]]
    for gap, q in zip(gaps, pts[1:]):
        if gap > threshold:
            groups.append(cur)
            cur = [q]
        else:
            cur.append(q)
    groups.append(cur)
    return groups


def inner_cluster(groups):
    """The cluster to trace: most saddles, then most points, then nearest 0.

    Selecting on saddle count directly forecloses the failure mode of the
    original: a cluster with no saddle yields an empty tracing table.
    """
    def score(grp):
        saddles = sum(1 for q in grp if q.kind == "saddle")
        med_b = statistics.median(abs(float(q.b)) for q in grp)
        return (saddles, len(grp), -med_b)
    return max(groups, key=score)


def cluster_box(grp, pad):
    bs = [float(q.b) for q in grp]
    as_ = [float(q.a) for q in grp]
    bspan = max(max(bs) - min(bs), 1e-3)
    aspan = max(max(as_) - min(as_), 1e-3)
    bc, ac = 0.5 * (max(bs) + min(bs)), 0.5 * (max(as_) + min(as_))
    return (ac - pad * aspan, ac + pad * aspan,
            bc - pad * bspan, bc + pad * bspan)


# ------------------------------------------------------------------ #
# arms
# ------------------------------------------------------------------ #

def trace_arm(m, gap_factor, pad, label):
    """Arms A and C: direct saddle-branch traces in the inner-cluster box.

    Not a certification -- the remote cluster is outside the box.  The
    engine's internal step cap terminates stalls as abort_max_steps.
    """
    e = sturm.enumerate_critical_points(m)
    pts = sorted(e.points, key=lambda q: float(q.b))
    groups = cluster_skeleton(pts, gap_factor)
    inner = inner_cluster(groups)
    box = cluster_box(inner, pad)
    ds = (abs(box[1] - box[0]) + abs(box[3] - box[2])) / 30000.0
    full_ds = (2 * atlas.legal_max_b(m) * 2) / 30000.0

    out = {"label": label,
           "skeleton": [{"kind": q.kind, "b": float(q.b), "a": float(q.a)}
                        for q in pts],
           "clusters": [[float(q.b) for q in grp] for grp in groups],
           "inner_b": [float(q.b) for q in inner],
           "box": list(box), "ds": ds, "full_box_ds": full_ds,
           "branches": []}
    print(f"\n  [{label}] {len(groups)} cluster(s); inner holds "
          f"{sum(1 for q in inner if q.kind == 'saddle')} saddle(s), "
          f"{len(inner)} point(s)")
    print(f"  [{label}] box {tuple(round(x, 8) for x in box)}   "
          f"ds {ds:.6g}  (vs {full_ds:.6g} from legal_max_b)")

    inner_saddles = [q for q in inner if q.kind == "saddle"]
    minima = [p for p in pts if p.kind != "saddle"]
    print(f"  {'saddle b':>14}{'dir':>5}{'kind':>10}{'term':>26}{'n':>10}"
          f"{'secs':>9}{'b_end':>13}")
    print("  " + "-" * 87)

    def emit(row):
        out["branches"].append(row)
        print(f"  {row['saddle_b']:>14.6g}{row['dir']:>5}{row['kind']:>10}"
              f"{row['term']:>26}{row['n']:>10}{row['secs']:>9.1f}"
              f"{row['b_end']:>13.6g}")

    def run(row, fn, *fargs, **fkw):
        t0 = time.perf_counter()
        try:
            br = fn(*fargs, **fkw)
            row.update(term=br.term, n=len(br.Y),
                       secs=round(time.perf_counter() - t0, 2),
                       a_end=float(br.Y[-1][0]), b_end=float(br.Y[-1][1]))
        except Exception as exc:                       # noqa: BLE001
            row.update(term=f"ERROR {type(exc).__name__}", n=0,
                       secs=round(time.perf_counter() - t0, 2),
                       a_end=float("nan"), b_end=float("nan"))
        emit(row)

    for q in inner_saddles:
        b_s = float(q.b)
        for sign in (+1, -1):
            run({"saddle_b": b_s, "dir": sign, "kind": "stable"},
                engine.trace_stable, m, b_s, sign, box=box,
                critical_local=None, critical_stub=None)
        # Unstable branches are CANDIDATE-DIRECTED: trace_unstable's third
        # argument is a target point (a, b), not a sign -- the original
        # box_experiment.py call was invalid and raised TypeError before
        # tracing anything.  Per side: nearest minimum as the target; a
        # side with no minimum at all is the pseudo-target case, which
        # trace_valley_exit owns (valley -> box edge).
        for side in (-1, +1):
            # Only minima INSIDE the box are capturable targets.  Aiming at
            # the remote minimum (40960 units away, ds = span/4000) re-runs
            # the production zone-loop stall at minutes per row and can
            # only end at the box edge anyway.  A side whose minima all lie
            # outside the box is the pseudo-target case: slaved valley to
            # the box edge, ~n_grid evaluations, instant.
            cands = [p for p in minima
                     if (float(p.b) - b_s) * side > 0
                     and box[2] <= float(p.b) <= box[3]]
            row = {"saddle_b": b_s, "dir": side, "kind": "unstable"}
            if cands:
                t = min(cands, key=lambda p: abs(float(p.b) - b_s))
                row["target_b"] = float(t.b)
                # ds MUST be the box chord, same as the stable rows use.
                # trace_unstable's default is |b-span|/4000, which in
                # original units prices a 4.5e7-tall a-descent at a
                # 5e-5 chord: ~1e12 steps of pointless faithfulness
                # before the zone/step caps fire.  The box chord is the
                # controlled comparison.
                run(row, engine.trace_unstable, m, b_s,
                    (float(t.a), float(t.b)), box=box, ds=ds)
            else:
                row["target_b"] = None
                run(row, charts.trace_valley_exit, m, b_s,
                    box[3] if side > 0 else box[2], box=box)
    if not out["branches"]:
        print(f"  [{label}] EMPTY TABLE -- selection failed; inspect "
              "'clusters' in the JSON")
    return out


def production_arm(m, label):
    """Arm B: the unchanged production pipeline on the normalised model."""
    print(f"\n  [{label}] portrait.certified_compute on the normalised "
          "model (this is the real pipeline; a stall-class case may take "
          "a while and will report its abort terms)")
    t0 = time.perf_counter()
    p = portrait.certified_compute(m)
    wall = time.perf_counter() - t0
    top = p.ledger.get("topology", {})
    terms = {}
    for br in p.branches:
        key = f"{br.kind}/{br.term}"
        terms[key] = terms.get(key, 0) + 1
    print(f"  [{label}] {top.get('status')}   "
          f"{top.get('resolution_reason')}   "
          f"level {top.get('geometry_level')}   {wall:.1f}s")
    for key in sorted(terms):
        print(f"      {key:<40}{terms[key]}")
    return {"label": label, "status": top.get("status"),
            "resolution_reason": top.get("resolution_reason"),
            "geometry_level": top.get("geometry_level"),
            "seconds": round(wall, 2), "terms": terms}


# ------------------------------------------------------------------ #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--gap-factor", type=float, default=50.0,
                    help="split clusters at gaps this many times the median "
                         "inter-point gap")
    ap.add_argument("--pad", type=float, default=4.0)
    ap.add_argument("--arms", default="A,B,C",
                    help="comma subset of A,B,C")
    ap.add_argument("--out", default=None,
                    help="JSON path (default out/box_arms_<seed>.json)")
    args = ap.parse_args(argv)
    arms = {s.strip().upper() for s in args.arms.split(",") if s.strip()}

    generate = directed_model if args.mode == "directed" else random_model
    built = generate(random.Random(args.seed), args.degree)
    if built is None or built[0] is None:
        raise SystemExit("generator declined this seed")
    m, spec = built
    base = measure(m)
    print(f"seed {args.seed}   {spec}")
    print(f"  a*(0) = {base['astar0']:.6g}   kappa = {base['kappa']:.6g}   "
          f"legal_max_b = {atlas.legal_max_b(m):.6g}")

    results = {"seed": args.seed, "mode": args.mode, "degree": args.degree,
               "spec": str(spec), "measure": base,
               "gap_factor": args.gap_factor, "pad": args.pad, "arms": {}}

    mn = None
    if arms & {"B", "C"}:
        mn = normalised(m)
        after = measure(mn)
        results["measure_normalised"] = after
        print(f"  normalised: a*(0) = {after['astar0']:.6g}   "
              f"legal_max_b = {atlas.legal_max_b(mn):.6g}")

    if "A" in arms:
        results["arms"]["A"] = trace_arm(
            m, args.gap_factor, args.pad, "arm A: inner box, original units")
    if "B" in arms:
        results["arms"]["B"] = production_arm(
            mn, "arm B: full pipeline, normalised units")
    if "C" in arms:
        results["arms"]["C"] = trace_arm(
            mn, args.gap_factor, args.pad, "arm C: inner box, normalised")

    out_path = Path(args.out) if args.out else (
        REPO / "out" / f"box_arms_{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\n  results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
