#!/usr/bin/env python3
"""Measure existing zoo portraits independently; does not change their verdicts.

python scripts/credibility_probe.py minimal-quartet tricky-d11 > evidence.jsonl
Every chord is measured. Regions are reported separately, not silently skipped.
"""
import collections
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from spong import credibility, model, portrait, zoo
from spong.trajectory_check import inspect_trajectory


def summarize(rows):
    measured = [r for r in rows if r["status"] == "measured"]
    return {"segments": len(rows),
            "statuses": dict(collections.Counter(r["status"] for r in rows)),
            "max_sin_squared": max((r["sin_squared"] for r in measured), default=None),
            "opposed_midpoint_flow": sum(r["flow_alignment"] == -1 for r in measured),
            "orthogonal_midpoint_flow": sum(r["flow_alignment"] == 0 for r in measured),
            "reversed_loss": sum(r["loss_direction"] == -1 for r in rows),
            "constant_loss": sum(r["loss_direction"] == 0 for r in rows)}


def probe(name, *, measure_jets=False, measure_native=False, policy=None, refine_arrivals=False):
    case = zoo.get(name)
    moments = model.moments_normal01 if case.moment_dist == "normal01" else model.moments_uniform01
    m = model.build(list(case.f), list(case.g), moments(2*max(len(case.f), len(case.g))-1))
    start = time.perf_counter()
    p = portrait.certified_compute(m, view=case.default_view)
    construction_seconds = time.perf_counter()-start
    refinement = []
    refinement_seconds = 0.0
    if refine_arrivals:
        refinement_start = time.perf_counter()
        from dataclasses import replace
        from spong.arrival_series import refine_capture
        from spong import topology
        candidates = []
        for i, branch in enumerate(p.branches):
            if branch.term == "capture":
                try:
                    branch = refine_capture(m, p.enumeration, branch)
                    refinement.append({"branch": i, "status": "proposed"})
                except ValueError as exc:
                    refinement.append({"branch": i, "status": "unresolved", "reason": str(exc)})
            candidates.append(branch)
        # No cached geometry certificate survives replacement of a curve.
        p = replace(p, branches=candidates, ledger={"topology": topology.audit(
            m, p.enumeration, candidates, p.box)})
        refinement_seconds = time.perf_counter()-refinement_start
    assessment = None
    assessment_seconds = 0.0
    if policy is not None:
        from spong.portrait_check import assess
        start = time.perf_counter()
        assessment = assess(p, policy)
        assessment_seconds = time.perf_counter()-start
    branches = []
    suffixes = p.ledger["topology"].get("terminal_suffixes", [])
    for i, br in enumerate(p.branches):
        start = time.perf_counter()
        rows = credibility.level_normal_reference(m, br.Y, 1 if br.kind == "stable" else -1)
        elapsed = time.perf_counter()-start
        germ = int(br.diag.get("critical_steps", 0))
        terminal = suffixes[i].get("start") if i < len(suffixes) else None
        terminal = len(rows) if terminal is None else max(germ, int(terminal))
        start = time.perf_counter()
        trajectory = (assessment["branches"][i]["trajectory"] if assessment is not None
                      and "branches" in assessment else inspect_trajectory(
                          m, br, measure_jets=measure_jets, measure_native=measure_native))
        trajectory_seconds = time.perf_counter()-start
        branches.append({"branch": i, "kind": br.kind, "term": br.term,
                         "trajectory": trajectory,
                         "trajectory_seconds": trajectory_seconds,
                         "seconds": elapsed, "all": summarize(rows),
                         "local_launch": summarize(rows[:germ]),
                         "continuation": summarize(rows[germ:terminal]),
                         "terminal_representation": summarize(rows[terminal:])})
    return {"case": name, "status": p.ledger["topology"]["status"],
            "construction_seconds": construction_seconds,
            "chord_reference_seconds": sum(b["seconds"] for b in branches),
            "trajectory_seconds": sum(b["trajectory_seconds"] for b in branches),
            "assessment_seconds": assessment_seconds,
            "reference_seconds": assessment_seconds+sum(
                b["seconds"]+b["trajectory_seconds"] for b in branches),
            "assessment": assessment, "arrival_refinement": refinement,
            "arrival_refinement_seconds": refinement_seconds,
            "branches": branches}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["minimal-quartet"])
    parser.add_argument("--refine-arrivals", action="store_true",
                        help="propose experimental two-mode analytic minimum entry charts")
    parser.add_argument("--measure-jets", action="store_true",
                        help="independently evaluate retained smooth jets over Q")
    parser.add_argument("--measure-native", action="store_true",
                        help="independently evaluate every retained native step over Q")
    parser.add_argument("--max-sin-squared", type=float,
                        help="explicit experimental four-obligation angular policy")
    parser.add_argument("--max-eigen-sin-squared", type=float, default=1e-12)
    parser.add_argument("--max-binding-eps", type=float, default=64.0)
    parser.add_argument("--sample-budget", type=int, default=250000)
    args = parser.parse_args()
    os.environ["SPONG_WORKERS"] = "1"
    from spong.portrait_check import CredibilityPolicy
    policy = (None if args.max_sin_squared is None else CredibilityPolicy(
        args.max_sin_squared, args.max_eigen_sin_squared, args.max_binding_eps,
        args.sample_budget))
    for name in args.names:
        print(json.dumps(probe(name, measure_jets=args.measure_jets,
                               measure_native=args.measure_native, policy=policy,
                               refine_arrivals=args.refine_arrivals)), flush=True)
