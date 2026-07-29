#!/usr/bin/env python3
"""Parallel out-of-sample qualification of critical-point stubs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spong import model, sturm


def run_case(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    df, dg = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    f = rng.normal(size=df + 1)
    g = f.copy() if rng.random() < .25 and df == dg \
        else rng.normal(size=dg + 1)
    moments_name = "uniform01" if rng.random() < .75 else "normal01"
    moments = (model.moments_uniform01 if moments_name == "uniform01"
               else model.moments_normal01)(2 * max(df, dg) + 1)
    try:
        m = model.build(f, g, moments)
        e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
        ss = [s for p in e.saddles for s in p.stubs]
        certs = [dict(s.certificates) for s in ss]
        return {
            "seed": seed, "f_degree": df, "g_degree": dg,
            "moments": moments_name, "saddles": len(e.saddles),
            "stubs": len(ss),
            "global_ready": sum(c["global_field_ready"] for c in certs),
            "max_reach_halvings": max(
                (c["reach_halvings"] for c in certs), default=0),
            "max_physical_map_grid_error": max(
                (c["physical_map_grid_error"] for c in certs), default=0),
        }
    except Exception as exc:
        return {
            "seed": seed, "f_degree": df, "g_degree": dg,
            "moments": moments_name, "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(2**32)) for _ in range(args.cases)]
    jobs = min(args.jobs, len(seeds))
    chunks = [seeds[i::jobs] for i in range(jobs)]
    parts = [args.output.with_suffix(args.output.suffix + f".part{i}")
             for i in range(jobs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pids = []
    for chunk, part in zip(chunks, parts):
        pid = os.fork()
        if pid == 0:
            part.write_text(json.dumps([run_case(seed) for seed in chunk]))
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        if status:
            raise RuntimeError(f"qualification worker {pid} exited {status}")
    results = []
    for part in parts:
        results.extend(json.loads(part.read_text()))
        part.unlink()
    failures = [r for r in results if "exception" in r]
    passed = [r for r in results if "exception" not in r]
    summary = {
        "cases": len(results), "passed": len(passed),
        "failed": len(failures),
        "saddles": sum(r["saddles"] for r in passed),
        "stubs": sum(r["stubs"] for r in passed),
        "global_ready": int(sum(r["global_ready"] for r in passed)),
        "max_reach_halvings": max(
            (r["max_reach_halvings"] for r in passed), default=0),
        "max_physical_map_grid_error": max(
            (r["max_physical_map_grid_error"] for r in passed), default=0),
        "elapsed_sec": time.perf_counter() - t0,
    }
    args.output.write_text(json.dumps(
        {"config": vars(args) | {"output": str(args.output)},
         "summary": summary, "results": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
