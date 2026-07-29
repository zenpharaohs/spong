#!/usr/bin/env python3
"""Parallel out-of-sample qualification for native SPONG portrait algorithms."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import signal
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spong import charts, inverse, model, portrait, qualification, zoo


MAX_GEOMETRY_LEVEL = 2


def _certified(m):
    return portrait.certified_compute(
        m, max_geometry_level=MAX_GEOMETRY_LEVEL)


def _summary(kind, seed, metadata, m, p, elapsed):
    if m._native_kernel is None:
        raise RuntimeError("native kernel unavailable")
    branches = p.ledger["branches"]
    seams = [b.get("seam_residual[RESIDUAL]") for b in branches]
    seams = [x for x in seams if x is not None]
    backbones = [b.get("backbone_residual[RESIDUAL]") for b in branches]
    backbones = [x for x in backbones if x is not None]
    aborts = []
    stub_certificates = [
        dict(stub.certificates)
        for point in p.enumeration.saddles for stub in point.stubs]
    for br in p.branches:
        if not br.term.startswith("abort"):
            continue
        last = br.Y[-1]
        target = br.diag.get("target")
        aborts.append({
            "kind": br.kind,
            "term": br.term,
            "n_points": int(len(br.Y)),
            "last": [float(last[0]), float(last[1])],
            "last_chord": float(np.hypot(*(br.Y[-1] - br.Y[-2])))
            if len(br.Y) > 1 else None,
            "saddle_b": br.diag.get("saddle_b"),
            "target": None if target is None
            else [float(target[0]), float(target[1])],
            "switches": br.diag.get("switches"),
            "kappa_saddle": br.diag.get("kappa_saddle"),
            "final_state_bw": br.diag.get("final_state_bw"),
            "step_failure": br.diag.get("step_failure"),
            "conditioning_refusal": br.diag.get("conditioning_refusal"),
        })
    return {
        "kind": kind,
        "seed": seed,
        **metadata,
        "arithmetic": qualification.arithmetic_profile(m),
        "skeleton_arithmetic": qualification.skeleton_profile(
            m, p.enumeration),
        "elapsed_sec": elapsed,
        "timing": p.ledger.get("timing", {}),
        "n_critical": len(p.enumeration.points),
        "n_branches": len(branches),
        "local_work": {
            "stub_count": len(stub_certificates),
            "conditioning_refusals": sum(
                not bool(x.get("graph_certified", 0.0))
                for x in stub_certificates),
            "max_graph_iterations": max(
                (x.get("graph_iterations_fine", 0.0)
                 for x in stub_certificates), default=0.0),
            "max_reach_halvings": max(
                (x.get("reach_halvings", 0.0)
                 for x in stub_certificates), default=0.0),
            "max_extension_steps": max(
                (x.get("extension_steps", 0.0)
                 for x in stub_certificates), default=0.0),
            "min_reach": min(
                (x.get("reach", np.inf) for x in stub_certificates),
                default=None),
            "not_continuation_ready": sum(
                not bool(x.get("continuation_ready", 0.0))
                for x in stub_certificates),
        },
        "continuation_work": {
            "total_points": sum(len(br.Y) for br in p.branches),
            "max_branch_points": max(
                (len(br.Y) for br in p.branches), default=0),
            "total_critical_points": sum(
                int(br.diag.get("critical_steps", 0))
                for br in p.branches),
            "total_switches": sum(
                int(br.diag.get("switches", 0) or 0)
                for br in p.branches),
        },
        "terms": {term: sum(b["term"] == term for b in branches)
                  for term in sorted({b["term"] for b in branches})},
        "clean": p.ledger["summary"]["all_branches_clean"],
        "balanced": p.ledger["summary"]["balanced"],
        "worst_angle_energy": p.ledger["summary"]["worst_angle_energy"],
        "worst_seam": max(seams, default=0.0),
        "worst_backbone": max(backbones, default=0.0),
        "topology_status": p.ledger.get("topology", {}).get("status"),
        "topology_audit_complete": p.ledger.get(
            "topology", {}).get("audit_complete"),
        "topology_resolution_reason": p.ledger.get(
            "topology", {}).get("resolution_reason"),
        "geometry_attempts": p.ledger.get(
            "topology", {}).get("attempts", []),
        "forbidden_intersections": p.ledger.get(
            "topology", {}).get("forbidden_count", 0),
        "ambiguous_contacts": p.ledger.get(
            "topology", {}).get("ambiguous_count", 0),
        "aborts": aborts,
    }


def _untargeted(seed):
    rng = np.random.default_rng(seed)
    df, dg = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    f = rng.normal(size=df + 1)
    g = f.copy() if rng.random() < 0.25 and df == dg else rng.normal(size=dg + 1)
    moments_name = "uniform01" if rng.random() < 0.75 else "normal01"
    moments = (model.moments_uniform01 if moments_name == "uniform01"
               else model.moments_normal01)(2 * max(df, dg) + 1)
    m = model.build(f, g, moments)
    t0 = time.perf_counter()
    p = _certified(m)
    return _summary("untargeted", seed, {
        "f_degree": df, "g_degree": dg, "moments": moments_name,
        "same": bool(np.array_equal(f, g)),
    }, m, p, time.perf_counter() - t0)


def _targeted(seed, exponent):
    rng = np.random.default_rng(seed)
    dg = int(rng.integers(3, 7))
    coeff = [int(x) for x in rng.integers(-4, 5, size=dg + 1)]
    if coeff[0] == 0:
        coeff[0] = int(rng.choice([-1, 1]))
    if coeff[-1] == 0:
        coeff[-1] = int(rng.choice([-1, 1]))
    radius = Fraction(int(rng.choice([-1, 1])) * (2 ** exponent))
    deg_f = dg + 1
    moments = model.moments_uniform01(2 * max(deg_f, dg) + 1)
    d = inverse.design([radius], coeff, moments, deg_f=deg_f)
    rep = inverse.report(d)
    if rep.missing:
        raise RuntimeError(f"prescribed critical point missing: {rep.missing}")
    t0 = time.perf_counter()
    p = _certified(d.model)
    return _summary("targeted", seed, {
        "f_degree": len(d.f) - 1, "g_degree": dg,
        "radius": float(radius), "exponent": exponent,
        "g": coeff, "gauge": rep.gauges[0],
    }, d.model, p, time.perf_counter() - t0)


def _near_morse(seed, exponent):
    """Exact nearby backbone critical points separated by 2**(-exponent)."""
    rng = np.random.default_rng(seed)
    dg = int(rng.integers(3, 7))
    coeff = [int(x) for x in rng.integers(-4, 5, size=dg + 1)]
    if coeff[0] == 0:
        coeff[0] = int(rng.choice([-1, 1]))
    if coeff[-1] == 0:
        coeff[-1] = int(rng.choice([-1, 1]))
    center = Fraction(int(rng.integers(-4, 5)), 2)
    separation = Fraction(1, 2**exponent)
    prescribed = [
        center-separation/2,
        center+separation/2,
    ]
    deg_f = dg + 2
    moments_name = "uniform01" if rng.random() < 0.75 else "normal01"
    moments = (model.moments_uniform01 if moments_name == "uniform01"
               else model.moments_normal01)(2 * max(deg_f, dg) + 1)
    d = inverse.design(prescribed, coeff, moments, deg_f=deg_f)
    rep = inverse.report(d)
    if rep.missing:
        raise RuntimeError(f"prescribed critical points missing: {rep.missing}")
    t0 = time.perf_counter()
    p = _certified(d.model)
    return _summary("near_morse", seed, {
        "f_degree": len(d.f) - 1,
        "g_degree": dg,
        "moments": moments_name,
        "center": float(center),
        "separation": float(separation),
        "separation_exponent": exponent,
        "prescribed": [float(x) for x in prescribed],
        "g": coeff,
        "gauges": rep.gauges,
    }, d.model, p, time.perf_counter() - t0)


def _zoo(name):
    z = zoo.get(name)
    moments = (model.moments_uniform01 if z.moment_dist == "uniform01"
               else model.moments_normal01)(2*max(len(z.f)-1, len(z.g)-1)+1)
    m = model.build(z.f, z.g, moments)
    t0 = time.perf_counter()
    p = _certified(m)
    observed = []
    for source_b, target_b in z.expected_connections:
        matches = [
            br for br in p.branches
            if br.kind == "unstable" and br.term == "capture"
            and abs(br.diag.get("saddle_b", np.inf)-source_b) < 1e-9
            and br.diag.get("target") is not None
            and abs(br.diag["target"][1]-target_b) < 1e-9
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"zoo expectation {source_b} -> {target_b} not observed")
        observed.append([source_b, target_b])
    return _summary("zoo", z.seed or 0, {
        "name": z.name, "description": z.description,
        "f_degree": len(z.f)-1, "g_degree": len(z.g)-1,
        "moments": z.moment_dist, "expected_connections": observed,
    }, m, p, time.perf_counter()-t0)


def _run(spec):
    kind, seed, parameter = spec
    try:
        if kind == "untargeted":
            return _untargeted(seed)
        if kind == "targeted":
            return _targeted(seed, parameter)
        if kind == "near_morse":
            return _near_morse(seed, parameter)
        if kind == "zoo":
            return _zoo(parameter)
        raise ValueError(f"unknown qualification kind {kind}")
    except Exception as exc:  # qualification must preserve every failure
        return {
            "kind": kind, "seed": seed, "parameter": parameter,
            "exception": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux and the BSDs exposed by Python report KiB.
    return int(rss if sys.platform == "darwin" else rss * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--untargeted", type=int, default=400)
    ap.add_argument("--targeted-per-radius", type=int, default=40)
    ap.add_argument("--near-morse-per-separation", type=int, default=0)
    ap.add_argument("--near-morse-exponents", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 24, 28, 32])
    ap.add_argument("--exponents", type=int, nargs="+",
                    default=[2, 5, 8, 11, 14, 17])
    ap.add_argument("--no-zoo", action="store_true",
                    help="omit the named deterministic regression zoo")
    ap.add_argument("--max-geometry-level", type=int, default=2,
                    help="trace-box escalations allowed for certification")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument(
        "--selection-shards", type=int, default=1,
        help="deterministically select one strided shard of the generated specs")
    ap.add_argument(
        "--selection-index", type=int, default=0,
        help="zero-based generated-spec shard selected by --selection-shards")
    ap.add_argument("--seed-file", type=Path,
                    help="JSON list of exact untargeted case seeds")
    ap.add_argument("--spec-file", type=Path,
                    help="JSON list of {kind, seed, exponent} exact cases")
    ap.add_argument(
        "--resume-part", type=Path, action="append", default=[],
        help="skip selected-spec indices checkpointed in a prior JSONL part")
    ap.add_argument("--unstable-launch-rel", type=float, default=1e-6)
    ap.add_argument("--stable-launch-delta", type=float, default=1e-4)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument(
        "--memory-limit-gib", type=float, default=8.0,
        help="per-worker address-space ceiling; 0 disables the ceiling")
    ap.add_argument(
        "--case-timeout-sec", type=float, default=600.0,
        help="wall-time ceiling per case; 0 disables the ceiling")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    global MAX_GEOMETRY_LEVEL
    MAX_GEOMETRY_LEVEL = args.max_geometry_level
    charts.UNSTABLE_LAUNCH_REL = args.unstable_launch_rel
    charts.STABLE_LAUNCH_DELTA = args.stable_launch_delta
    rng = np.random.default_rng(args.seed)
    if args.spec_file:
        raw_specs = json.loads(args.spec_file.read_text())
        specs = [(x["kind"], int(x.get("seed", 0)),
                  x.get("name", x.get("exponent")))
                 for x in raw_specs]
    elif args.seed_file:
        case_seeds = json.loads(args.seed_file.read_text())
        specs = [("untargeted", int(seed), None) for seed in case_seeds]
    else:
        case_seeds = [int(rng.integers(2**32)) for _ in range(args.untargeted)]
        specs = [("untargeted", int(seed), None) for seed in case_seeds]
        for exponent in args.exponents:
            specs.extend(("targeted", int(rng.integers(2**32)), exponent)
                         for _ in range(args.targeted_per_radius))
        for exponent in args.near_morse_exponents:
            specs.extend(("near_morse", int(rng.integers(2**32)), exponent)
                         for _ in range(args.near_morse_per_separation))
        if not args.no_zoo:
            specs.extend(("zoo", z.seed or 0, z.name)
                         for z in zoo.CASES.values())
    if args.selection_shards < 1:
        ap.error("--selection-shards must be positive")
    if not 0 <= args.selection_index < args.selection_shards:
        ap.error("--selection-index must be in [0, --selection-shards)")
    specs = specs[args.selection_index::args.selection_shards]
    selected_indexed_specs = list(enumerate(specs))
    resumed_results = {}
    for path in args.resume_part:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                resumed_results[item["_spec_index"]] = item
    indexed_specs = [
        (i, spec) for i, spec in selected_indexed_specs
        if i not in resumed_results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = list(resumed_results.values())
    t0 = time.perf_counter()
    jobs = min(args.jobs, len(indexed_specs)) if indexed_specs else 0
    chunks = [indexed_specs[i::jobs] for i in range(jobs)]
    part_paths = [args.output.with_suffix(args.output.suffix + f".part{i}")
                  for i in range(jobs)]
    pids = []
    for i, chunk in enumerate(chunks):
        pid = os.fork()
        if pid == 0:
            memory_limit = 0
            case_started = [time.monotonic()]
            if args.memory_limit_gib > 0.0:
                memory_limit = int(args.memory_limit_gib * (1024**3))
            if memory_limit or args.case_timeout_sec > 0.0:
                def resource_watchdog(_signum, _frame):
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    if memory_limit and _rss_bytes() >= memory_limit:
                        raise MemoryError(
                            f"qualification worker exceeded "
                            f"{args.memory_limit_gib:g} GiB RSS")
                    if (args.case_timeout_sec > 0.0
                            and time.monotonic()-case_started[0]
                            >= args.case_timeout_sec):
                        raise TimeoutError(
                            f"qualification case exceeded "
                            f"{args.case_timeout_sec:g} seconds")
                    signal.setitimer(signal.ITIMER_REAL, 1.0)
                signal.signal(signal.SIGALRM, resource_watchdog)
            # Checkpoint each case.  A pathological later case must not erase
            # the completed work from the same shard.
            with part_paths[i].open("w") as stream:
                for spec_index, spec in chunk:
                    case_started[0] = time.monotonic()
                    if memory_limit or args.case_timeout_sec > 0.0:
                        signal.setitimer(signal.ITIMER_REAL, 1.0)
                    result = _run(spec)
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    result["_spec_index"] = spec_index
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    gc.collect()
                    if memory_limit and _rss_bytes() >= memory_limit:
                        os._exit(75)
            os._exit(0)
        pids.append(pid)
    worker_failures = []
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        if status != 0:
            worker_failures.append({"pid": pid, "status": status})
    for path in part_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                results.append(json.loads(line))
    completed_indices = {r["_spec_index"] for r in results}
    missing_specs = [
        {"index": i, "kind": spec[0], "seed": spec[1],
         "parameter": spec[2]}
        for i, spec in selected_indexed_specs if i not in completed_indices
    ]
    failures = sum("exception" in r for r in results)
    unclean = sum(not r.get("clean", True) for r in results
                  if "exception" not in r)
    unresolved = sum(r.get("topology_status") != "certified" for r in results
                     if "exception" not in r)
    print(f"{len(results)}/{len(selected_indexed_specs)} failures={failures} "
          f"unclean={unclean} unresolved={unresolved} "
          f"missing={len(missing_specs)}", flush=True)

    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [jsonable(x) for x in value]
        return value
    config = {k: jsonable(v) for k, v in vars(args).items()}
    payload = {
        "config": config,
        "elapsed_sec": time.perf_counter() - t0,
        "worker_failures": worker_failures,
        "missing_specs": missing_specs,
        "results": sorted(results, key=lambda r: (r["kind"], r["seed"])),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}; failures={failures}, unclean={unclean}")
    return 1 if (failures or unclean or unresolved or missing_specs
                 or worker_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
