"""Normalisation retry sweep over the uncertified ensemble cases.

Arm B of box_experiment_arms certified the flagship stall case (seed
953953598: 1685.6s branch_abort in original units -> 18.8s certified,
geometry level 0, no escalation).  This script asks whether that
generalises: for every non-certified case in an ensemble jsonl, rebuild
the model from its seed, normalise f and g exactly (f/||f||_inf,
g/||g||_inf -- a diffeomorphism plus a positive scaling of the loss, so
the merge-tree/contact topology the certifier decides is invariant), and
run the UNCHANGED production pipeline.

Predictions on record (2026-08-28), falsifiable by this sweep:
  * converts concentrate in the anisotropy/stall subclass (large
    |E[f]|/||f||, i.e. large kappa numerator);
  * abort_conditioning_handoff far-saddle cases do NOT convert -- b* is
    untouched by f,g scaling and their obstruction is fp64 spectral
    resolution, not units.
Each output row therefore carries kappa and the dead-neuron margin
|g(0)|/||g||_inf so the predictors are tested in the same pass.

Results stream to a jsonl (one line per case, flushed immediately) and
the run is RESUMABLE: seeds already present in the output are skipped,
so Ctrl-C costs nothing.  Cases run cheapest-first by their original
wall time (--order file for ensemble order).

    python scripts/normalize_retry.py                       # all 57
    python scripts/normalize_retry.py --reasons branch_abort
    python scripts/normalize_retry.py --seeds 202251424
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from box_experiment_arms import measure, normalised       # noqa: E402
from spong import portrait                                # noqa: E402
from qualify import directed_model, random_model          # noqa: E402


def retry_case(rec, mode, degree, pow2=False, identity=False):
    generate = directed_model if mode == "directed" else random_model
    built = generate(random.Random(int(rec["seed"])), degree)
    if built is None or built[0] is None:
        return {"error": "generator declined seed"}
    m, spec = built
    if str(spec) != rec.get("spec", str(spec)):
        # Reproducibility guard: the rebuilt model must be the ensemble's.
        return {"error": f"spec mismatch: rebuilt {spec!r}"}
    base = measure(m)
    g0_margin = (abs(float(m.g[0])) / base["gscale"]
                 if base["gscale"] else 0.0)
    mn = m if identity else normalised(m, pow2=pow2)
    t0 = time.perf_counter()
    try:
        p = portrait.certified_compute(mn)
    except Exception as exc:                               # noqa: BLE001
        return {"kappa": base["kappa"], "g0_margin": g0_margin,
                "after": {"status": f"ERROR {type(exc).__name__}",
                          "seconds": round(time.perf_counter() - t0, 2)}}
    wall = time.perf_counter() - t0
    top = p.ledger.get("topology", {})
    terms = {}
    for br in p.branches:
        key = f"{br.kind}/{br.term}"
        terms[key] = terms.get(key, 0) + 1
    return {"kappa": base["kappa"], "g0_margin": g0_margin,
            "astar0": base["astar0"],
            "after": {"status": top.get("status"),
                      "reason": top.get("resolution_reason"),
                      "geometry_level": top.get("geometry_level"),
                      "seconds": round(wall, 2), "terms": terms}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble",
                    default=str(REPO / "out" / "ensemble-directed-d5.jsonl"))
    ap.add_argument("--mode", choices=("directed", "random"), default=None,
                    help="default: inferred from the ensemble filename")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--reasons", default=None,
                    help="comma subset of original refusal reasons "
                         "(default: every non-certified case)")
    ap.add_argument("--all", action="store_true",
                    help="retry EVERY case, certified ones included -- "
                         "measures the speed effect on the healthy "
                         "population and catches regressions "
                         "(certified -> not certified)")
    ap.add_argument("--pow2", action="store_true",
                    help="normalise by powers of two: exact in FP64 and "
                         "denominator-free in the exact layers, at the "
                         "price of a factor <2 in the compression")
    ap.add_argument("--identity", action="store_true",
                    help="CONTROL: rerun the ORIGINAL model, no "
                         "normalisation at all -- isolates the harness/"
                         "environment component of any before-vs-after "
                         "timing difference")
    ap.add_argument("--seeds", default=None,
                    help="comma list of specific seeds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--order", choices=("cheap", "file"), default="cheap",
                    help="cheapest original wall time first (default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ens_path = Path(args.ensemble)
    mode = args.mode or ("random" if "random" in ens_path.stem
                         else "directed")
    out_path = Path(args.out) if args.out else (
        REPO / "out" / (f"normalize_retry-{ens_path.stem}"
                        + ("-identity" if args.identity else
                           "-pow2" if args.pow2 else "") + ".jsonl"))

    records = [json.loads(line) for line in
               ens_path.read_text().splitlines() if line.strip()]
    todo = (list(records) if args.all
            else [r for r in records if r.get("status") != "certified"])
    if args.reasons:
        wanted = {s.strip() for s in args.reasons.split(",")}
        todo = [r for r in todo if r.get("reason") in wanted]
    if args.seeds:
        wanted = {int(s) for s in args.seeds.split(",")}
        todo = [r for r in todo if int(r["seed"]) in wanted]
    if args.order == "cheap":
        todo.sort(key=lambda r: float(r.get("seconds", 0.0)))
    if args.limit:
        todo = todo[:args.limit]

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["seed"]))
    skip = [r for r in todo if int(r["seed"]) in done]
    todo = [r for r in todo if int(r["seed"]) not in done]
    print(f"{len(todo)} case(s) to retry ({len(skip)} already in "
          f"{out_path.name}); mode {mode}, degree {args.degree}")
    print(f"{'seed':>12} {'before':<24}{'bsecs':>8}{'kappa':>11}"
          f"{'g0marg':>9}  {'after':<34}{'asecs':>8}")
    print("-" * 108)

    tallies: dict[str, list[int]] = {}
    t_before = t_after = 0.0
    regressions = []
    with out_path.open("a") as fh:
        for rec in todo:
            row = {"case": rec.get("case"), "seed": int(rec["seed"]),
                   "spec": rec.get("spec"), "pow2": bool(args.pow2),
                   "before": {"status": rec.get("status"),
                              "reason": rec.get("reason"),
                              "seconds": rec.get("seconds"),
                              "terms": rec.get("terms")}}
            row.update(retry_case(rec, mode, args.degree, pow2=args.pow2,
                                  identity=args.identity))
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            after = row.get("after", {})
            a_status = after.get("status", row.get("error", "?"))
            a_txt = str(a_status) + (
                f":{after['reason']}" if after.get("reason") else "")
            before_txt = f"{rec.get('status')}:{rec.get('reason')}"
            tally = tallies.setdefault(str(rec.get("reason")), [0, 0])
            tally[1] += 1
            tally[0] += int(a_status == "certified")
            t_before += float(rec.get("seconds") or 0.0)
            t_after += float(after.get("seconds") or 0.0)
            if (rec.get("status") == "certified"
                    and a_status != "certified"):
                regressions.append(row["seed"])
            print(f"{row['seed']:>12} {before_txt:<24}"
                  f"{float(rec.get('seconds', 0)):>8.1f}"
                  f"{row.get('kappa', float('nan')):>11.4g}"
                  f"{row.get('g0_margin', float('nan')):>9.3g}  "
                  f"{a_txt:<34}{after.get('seconds', 0.0):>8.1f}")

    print("-" * 108)
    print("certified-after / total by original reason "
          "(reason None = originally certified):")
    for reason in sorted(tallies):
        got, tot = tallies[reason]
        print(f"  {reason:<40}{got:>4} / {tot}")
    print(f"wall time: before {t_before:.1f}s, after {t_after:.1f}s")
    if regressions:
        print(f"REGRESSIONS (certified -> not certified): {regressions}")
    print(f"results appended to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
