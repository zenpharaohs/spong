#!/usr/bin/env python3
"""Compare factorized B,N and combined reduced-u' Morse root pathways.

This is a qualification experiment, not a replacement enumerator.  Both routes
use exact primitive Sturm arithmetic, refine the roots to the same width, and
must agree on every critical b-value and sign of u''.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spong import _poly as P
from spong import model, qualification, sturm


REFINE_REL = Fraction(1, 2**80)


def _clear_sturm_caches():
    sturm.sturm_chain.cache_clear()
    sturm.squarefree_part.cache_clear()


def _merge_stats(items):
    additive = {
        "variation_evaluations", "variation_signs", "subdivision_nodes",
        "polynomial_evaluations", "puncture_halvings",
        "isolated_roots", "exact_roots", "refinement_bisections",
        "chain_coefficients",
    }
    maximum = {
        "chain_length", "chain_peak_coefficient_bits",
        "max_subdivision_depth", "max_endpoint_bits",
        "max_refined_endpoint_bits",
    }
    out = {}
    for key in additive:
        out[key] = sum(x.get(key, 0) for x in items)
    for key in maximum:
        out[key] = max((x.get(key, 0) for x in items), default=0)
    return out


def _route(polynomials, hprime):
    _clear_sturm_caches()
    started = time.perf_counter()
    records = []
    per_poly = []
    for label, polynomial in polynomials:
        stats = {}
        intervals = sturm.isolate_roots(polynomial, stats=stats)
        for interval in intervals:
            refined = sturm.refine(
                polynomial, interval, rel=REFINE_REL, stats=stats)
            derivative_sign = sturm.interval_sign(hprime, refined)
            rel = REFINE_REL
            for _ in range(32):
                if derivative_sign is not None:
                    break
                rel /= 4
                refined = sturm.refine(
                    polynomial, refined, rel=rel, stats=stats)
                derivative_sign = sturm.interval_sign(hprime, refined)
            if derivative_sign is None:
                raise ArithmeticError(
                    "could not certify u'' sign on isolated root")
            records.append({
                "source": label,
                "b": float(refined.mid),
                "u2_sign": derivative_sign,
            })
        per_poly.append(stats)
    elapsed = time.perf_counter()-started
    records.sort(key=lambda x: x["b"])
    return {
        "elapsed_sec": elapsed,
        "roots": records,
        "work": _merge_stats(per_poly),
        "per_polynomial_work": per_poly,
    }


def _agree(left, right):
    if len(left) != len(right):
        return False, f"root count {len(left)} != {len(right)}"
    for i, (a, b) in enumerate(zip(left, right)):
        scale = 1.0+max(abs(a["b"]), abs(b["b"]))
        if abs(a["b"]-b["b"]) > 2.0**-48*scale:
            return False, f"root {i} differs: {a['b']} vs {b['b']}"
        if a["u2_sign"] != b["u2_sign"]:
            return False, f"root {i} sign differs"
    return True, None


def _run_case(spec):
    seed, df, dg, moments_name = spec
    rng = np.random.default_rng(seed)
    f = rng.normal(size=df+1)
    g = rng.normal(size=dg+1)
    moments = (model.moments_uniform01
               if moments_name == "uniform01"
               else model.moments_normal01)(2*max(df, dg)+1)
    m = model.build(f, g, moments)
    reduced = qualification.reduced_backbone_polynomials(m)
    h = reduced["derivative_squarefree"]
    hp = P.deriv(h)
    common = P.gcd_poly(m.beta, m.N)
    morse_factorization = (
        P.degree(common) <= 0
        and sturm.is_squarefree(m.beta)
        and sturm.is_squarefree(m.N))

    warm_started = time.perf_counter()
    warm = qualification.morse_preflight(m, include_sturm_chain=True)
    warm_elapsed = time.perf_counter()-warm_started
    chains = warm["sturm_chains"]
    predicted_factorized = (
        chains["B_roots"]["total_coefficient_bits"]
        + chains["N_roots"]["total_coefficient_bits"])
    predicted_combined = chains["reduced_u_prime"]["total_coefficient_bits"]
    predicted = ("factorized"
                 if predicted_factorized <= predicted_combined
                 else "combined")

    factorized = _route((("B", m.beta), ("N", m.N)), hp)
    combined = _route((("H", h),), hp)
    agrees, mismatch = _agree(factorized["roots"], combined["roots"])
    actual = ("factorized"
              if factorized["elapsed_sec"] <= combined["elapsed_sec"]
              else "combined")
    return {
        "seed": seed, "f_degree": df, "g_degree": dg,
        "moments": moments_name,
        "morse_factorization": morse_factorization,
        "preflight_elapsed_sec": warm_elapsed,
        "predicted_path": predicted,
        "actual_faster_path": actual,
        "prediction_correct": predicted == actual,
        "agrees": agrees,
        "mismatch": mismatch,
        "predicted_factorized_chain_bits": predicted_factorized,
        "predicted_combined_chain_bits": predicted_combined,
        "factorized": factorized,
        "combined": combined,
    }


def _safe_run(spec):
    try:
        return _run_case(spec)
    except Exception as exc:
        return {
            "seed": spec[0], "f_degree": spec[1],
            "g_degree": spec[2], "moments": spec[3],
            "exception": type(exc).__name__, "message": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--jobs", type=int,
                        default=max(1, min(8, (os.cpu_count() or 2)-1)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    specs = [
        (int(rng.integers(2**32)),
         int(rng.integers(args.min_degree, args.max_degree+1)),
         int(rng.integers(args.min_degree, args.max_degree+1)),
         "uniform01" if rng.random() < 0.75 else "normal01")
        for _ in range(args.cases)
    ]
    jobs = min(args.jobs, len(specs))
    chunks = [specs[i::jobs] for i in range(jobs)]
    parts = [
        args.output.with_suffix(args.output.suffix+f".part{i}")
        for i in range(jobs)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pids = []
    for chunk, part in zip(chunks, parts):
        pid = os.fork()
        if pid == 0:
            part.write_text(json.dumps([_safe_run(spec) for spec in chunk]))
            os._exit(0)
        pids.append(pid)
    worker_failures = []
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        if status:
            worker_failures.append({"pid": pid, "status": status})
    results = []
    for part in parts:
        if part.exists():
            results.extend(json.loads(part.read_text()))
            part.unlink()
    if worker_failures:
        results.append({
            "seed": -1, "f_degree": -1, "g_degree": -1,
            "moments": "worker", "exception": "WorkerFailure",
            "message": repr(worker_failures),
        })
    results.sort(key=lambda x: x["seed"])
    valid = [x for x in results if "exception" not in x]
    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "summary": {
            "cases": len(results),
            "exceptions": len(results)-len(valid),
            "route_agreements": sum(x["agrees"] for x in valid),
            "prediction_correct": sum(
                x["prediction_correct"] for x in valid),
            "predicted_factorized": sum(
                x["predicted_path"] == "factorized" for x in valid),
            "actual_factorized": sum(
                x["actual_faster_path"] == "factorized" for x in valid),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["summary"], indent=2))
    return 1 if payload["summary"]["exceptions"] \
        or payload["summary"]["route_agreements"] != len(valid) else 0


if __name__ == "__main__":
    raise SystemExit(main())
