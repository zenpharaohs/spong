#!/usr/bin/env python3
"""Find the FP64 portrait boundary under N(0, sigma^2) moment growth."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spong import model, portrait, sturm


def normal_moments(n: int, sigma: Fraction):
    base = model.moments_normal01(n)
    return tuple(mu*sigma**k for k, mu in enumerate(base))


def run_case(seed: int, degree: int, exponent: int, compensated: bool):
    rng = np.random.default_rng(seed)
    f = [Fraction(int(x)) for x in rng.integers(-4, 5, size=degree+1)]
    g = [Fraction(int(x)) for x in rng.integers(-4, 5, size=degree+1)]
    f[0] = f[0] or Fraction(1)
    g[0] = g[0] or Fraction(-1)
    sigma = Fraction(2)**exponent
    if compensated:
        f = [c/sigma**k for k, c in enumerate(f)]
        g = [c/sigma**k for k, c in enumerate(g)]
    out = {"seed": seed, "degree": degree, "sigma_exponent": exponent,
           "sigma": float(sigma), "compensated": compensated}
    try:
        m = model.build(f, g, normal_moments(2*degree+1, sigma))
        e = sturm.enumerate_critical_points(m)
        out.update({"skeleton": "ok", "critical": len(e.points),
                    "morse": e.morse, "psi_positive": e.psi_positive})
        ee = sturm.materialize_stubs(m, e)
        out.update({"stubs": "ok",
                    "stub_count": sum(len(q.stubs) for q in ee.saddles)})
        p = portrait.compute(m)
        top = p.ledger["topology"]
        out.update({
            "geometry": "ok",
            "clean": p.ledger["summary"]["all_branches_clean"],
            "topology_status": top["status"],
            "audit_complete": top["audit_complete"],
            "resolution_reason": top["resolution_reason"],
            "segments": top["segment_count"],
            "forbidden": top["forbidden_count"],
            "ambiguous": top["ambiguous_count"],
            "uncertified_tails": sum(
                not x["certified"] for x in top["stable_tails"]),
        })
    except Exception as exc:
        out.update({"geometry": "exception",
                    "exception": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--exponents", type=int, nargs="+",
                    default=list(range(0, 17, 2)))
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(2**32)) for _ in range(args.cases)]
    specs = [(seed, args.degree, exponent, compensated)
             for seed in seeds for exponent in args.exponents
             for compensated in (False, True)]
    jobs = min(args.jobs, len(specs))
    chunks = [specs[i::jobs] for i in range(jobs)]
    parts = [args.output.with_suffix(args.output.suffix+f".part{i}")
             for i in range(jobs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pids = []
    for chunk, part in zip(chunks, parts):
        pid = os.fork()
        if pid == 0:
            part.write_text(json.dumps([run_case(*spec) for spec in chunk]))
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        if status:
            raise RuntimeError(f"worker {pid} exited {status}")
    results = []
    for part in parts:
        results.extend(json.loads(part.read_text()))
        part.unlink()
    results.sort(key=lambda x: (
        x["seed"], x["compensated"], x["sigma_exponent"]))
    payload = {"config": vars(args) | {"output": str(args.output)},
               "results": results}
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    for compensated in (False, True):
        group = [x for x in results if x["compensated"] == compensated]
        print("compensated" if compensated else "fixed",
              "exceptions", sum(x.get("geometry") == "exception" for x in group),
              "certified", sum(x.get("topology_status") == "certified"
                               for x in group), "/", len(group))


if __name__ == "__main__":
    main()
