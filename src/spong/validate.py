"""Install-time differential validation of the native exact arithmetic core."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

from . import _native
from . import _poly as P
from . import model, sturm, zoo


def _analysis_oracle(p):
    trimmed = P.trim(p)
    gcd = P.gcd_poly(trimmed, P.deriv(trimmed))
    squarefree = (trimmed if P.degree(gcd) <= 0
                  else P.divmod_exact(trimmed, gcd)[0])
    return {
        "distinct_real_roots": sturm._count_roots_python(trimmed),
        "repeated_real_roots": (
            sturm._count_roots_python(gcd) if P.degree(gcd) > 0 else 0),
        "input_degree": P.degree(trimmed),
        "squarefree_degree": P.degree(squarefree),
    }


def _native_intervals(plan):
    result = plan.isolate()
    if result["status"] != _native.SPONG_EXACT_OK:
        raise ArithmeticError(f"isolation status {result['status']}")
    return [
        sturm.RootInterval(
            Fraction(int(a), int(b)), Fraction(int(c), int(d)), bool(exact))
        for a, b, c, d, exact in result["intervals"]
    ], result


def _validate_polynomial(label, p, seed):
    integers = P.int_primitive(P.trim(p))
    if not integers:
        return {"label": label, "degree": -1, "skipped_zero": True}
    plan = _native.SturmPlan(integers)
    native_stats = plan.stats()
    expected_stats = _analysis_oracle(p)
    for key, expected in expected_stats.items():
        if native_stats[key] != expected:
            raise AssertionError(
                f"{label}: analysis {key}: {native_stats[key]} != {expected}")

    native_ivs, isolation = _native_intervals(plan)
    oracle_ivs = sturm._isolate_roots_python(p)
    if native_ivs != oracle_ivs:
        raise AssertionError(f"{label}: isolating intervals differ")

    refinement_bisections = 0
    rel = Fraction(1, 2**48)
    for native_iv, oracle_iv in zip(native_ivs, oracle_ivs):
        expected = sturm._refine_python(p, oracle_iv, rel)
        result = plan.refine(native_iv.lo, native_iv.hi, rel)
        if result["status"] != _native.SPONG_EXACT_OK:
            raise ArithmeticError(
                f"{label}: refinement status {result['status']}")
        a, b, c, d, exact = result["interval"]
        actual = sturm.RootInterval(
            Fraction(int(a), int(b)), Fraction(int(c), int(d)), bool(exact))
        if actual != expected:
            raise AssertionError(f"{label}: refined interval differs")
        refinement_bisections += result["bisections"]

    rng = random.Random(seed)
    for _ in range(8):
        lo, hi = sorted((
            Fraction(rng.randrange(-100, 101), rng.randrange(1, 40)),
            Fraction(rng.randrange(-100, 101), rng.randrange(1, 40))))
        expected = sturm._count_roots_python(p, lo, hi)
        actual = plan.count(lo, hi)
        if actual != expected:
            raise AssertionError(
                f"{label}: count ({lo}, {hi}] {actual} != {expected}")
    for _ in range(8):
        x = Fraction(rng.randrange(-100, 101), rng.randrange(1, 40))
        value = P.eval_at(p, x)
        expected = (value > 0)-(value < 0)
        actual = plan.sign_at(x)
        if actual != expected:
            raise AssertionError(
                f"{label}: sign at {x}: {actual} != {expected}")
    return {
        "label": label,
        "degree": len(integers)-1,
        "roots": len(native_ivs),
        "peak_coefficient_bits": native_stats["peak_coefficient_bits"],
        "subdivision_nodes": isolation["subdivision_nodes"],
        "puncture_halvings": isolation["puncture_halvings"],
        "refinement_bisections": refinement_bisections,
    }


def _model_polynomials(m):
    return (
        ("A", m.alpha),
        ("B", m.beta),
        ("N", m.N),
        ("reduced-u-prime", m.critical_reduced),
    )


def _mundane_case(seed):
    rng = random.Random(seed)
    degree = rng.randrange(1, 13)
    coefficients = [
        rng.randrange(-2**18, 2**18) for _ in range(degree+1)]
    if coefficients[-1] == 0:
        coefficients[-1] = rng.choice((-1, 1))
    p = tuple(Fraction(x) for x in coefficients)
    return {
        "kind": "mundane", "seed": seed, "degree": degree,
        "polynomials": [_validate_polynomial("random-integer", p, seed)],
    }


def _linear_factor(root):
    return (-root, Fraction(1))


def _targeted_case(seed, exponent, family):
    rng = random.Random(seed)
    center = Fraction(rng.randrange(-12, 13), rng.randrange(1, 8))
    delta = Fraction(1, 2**exponent)
    if family == "close":
        roots = [center, center+delta, Fraction(rng.randrange(-5, 6))]
        p = (Fraction(1),)
        for root in roots:
            p = P.mul(p, _linear_factor(root))
    elif family == "far":
        far = Fraction(rng.choice((-1, 1))*2**exponent)
        p = P.mul(_linear_factor(far), _linear_factor(center))
    elif family == "repeated-real":
        factor = _linear_factor(center)
        p = P.mul(P.mul(factor, factor),
                  _linear_factor(center+Fraction(rng.choice((-2, 2)))))
    elif family == "repeated-complex":
        complex_factor = (Fraction(1), Fraction(0), Fraction(1))
        p = P.mul(P.mul(complex_factor, complex_factor),
                  _linear_factor(center))
    else:
        raise ValueError(f"unknown targeted family {family}")
    return {
        "kind": "targeted", "family": family, "seed": seed,
        "exponent": exponent,
        "polynomials": [_validate_polynomial(
            f"{family}-e{exponent}", p, seed)],
    }


def _zoo_case(name):
    case = zoo.get(name)
    degree = max(len(case.f)-1, len(case.g)-1)
    moments_fn = (model.moments_uniform01
                  if case.moment_dist == "uniform01"
                  else model.moments_normal01)
    m = model.build(case.f, case.g, moments_fn(2*degree+1))
    seed = case.seed or 0
    return {
        "kind": "zoo", "name": name, "seed": seed,
        "polynomials": [
            _validate_polynomial(label, p, seed ^ (i << 20))
            for i, (label, p) in enumerate(_model_polynomials(m))
        ],
    }


def _run(spec):
    started = time.perf_counter()
    try:
        kind = spec["kind"]
        if kind == "mundane":
            result = _mundane_case(spec["seed"])
        elif kind == "targeted":
            result = _targeted_case(
                spec["seed"], spec["exponent"], spec["family"])
        elif kind == "zoo":
            result = _zoo_case(spec["name"])
        else:
            raise ValueError(f"unknown validation kind {kind}")
        result["ok"] = True
    except Exception as exc:
        result = {
            **spec, "ok": False, "exception": type(exc).__name__,
            "message": str(exc),
        }
    result["elapsed_sec"] = time.perf_counter()-started
    return result


def _specifications(seed, mundane, targeted_per_exponent, exponents,
                    include_zoo):
    rng = random.Random(seed)
    specs = [
        {"kind": "mundane", "seed": rng.randrange(2**32)}
        for _ in range(mundane)
    ]
    families = ("close", "far", "repeated-real", "repeated-complex")
    for exponent in exponents:
        for i in range(targeted_per_exponent):
            specs.append({
                "kind": "targeted", "seed": rng.randrange(2**32),
                "exponent": exponent, "family": families[i % len(families)],
            })
    if include_zoo:
        specs.extend({"kind": "zoo", "name": name}
                     for name in zoo.names())
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Differentially validate SPONG's installed C exact core")
    parser.add_argument("--mundane", type=int, default=64)
    parser.add_argument("--targeted-per-exponent", type=int, default=8)
    parser.add_argument("--exponents", type=int, nargs="+",
                        default=[8, 20, 40, 80])
    parser.add_argument("--no-zoo", action="store_true")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--jobs", type=int,
                        default=max(1, min(8, (os.cpu_count() or 2)-1)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    specs = _specifications(
        args.seed, args.mundane, args.targeted_per_exponent,
        args.exponents, not args.no_zoo)
    started = time.perf_counter()
    parallel_fallback = None
    if args.jobs == 1:
        results = [_run(spec) for spec in specs]
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.jobs) as pool:
                results = list(pool.map(_run, specs))
        except (OSError, PermissionError, NotImplementedError) as exc:
            parallel_fallback = f"{type(exc).__name__}: {exc}"
            print(
                f"parallel validation unavailable; running serially "
                f"({parallel_fallback})", file=sys.stderr)
            results = [_run(spec) for spec in specs]
    failures = [result for result in results if not result["ok"]]
    polynomials = [
        polynomial
        for result in results if result["ok"]
        for polynomial in result["polynomials"]
    ]
    summary = {
        "ok": not failures,
        "cases": len(results),
        "passed": len(results)-len(failures),
        "failed": len(failures),
        "polynomials": len(polynomials),
        "roots": sum(x.get("roots", 0) for x in polynomials),
        "max_peak_coefficient_bits": max(
            (x.get("peak_coefficient_bits", 0) for x in polynomials),
            default=0),
        "elapsed_sec": time.perf_counter()-started,
        "seed": args.seed,
    }
    payload = {
        "format": "spong-native-validation-v1",
        "python": sys.version,
        "summary": summary,
        "config": {
            "mundane": args.mundane,
            "targeted_per_exponent": args.targeted_per_exponent,
            "exponents": args.exponents,
            "zoo": not args.no_zoo,
            "jobs": args.jobs,
            "parallel_fallback": parallel_fallback,
        },
        "failures": failures,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    if failures:
        for failure in failures:
            print(json.dumps(failure), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
