#!/usr/bin/env python3
"""Signed continuation and exact-affine bracketing of a saddle connection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demos import saddle_connections as sc


def _coefficients(record, key):
    return tuple(Fraction(n, d) for n, d in record[key])


def _candidate(path):
    data = json.loads(Path(path).read_text())
    if "arms" in data:
        if not data["arms"]:
            raise ValueError(f"{path} contains no arms")
        return min(data["arms"], key=lambda x: x["evaluation"]["score"])
    if "f" in data and "g" in data:
        return data
    raise ValueError(f"{path} is not a candidate or search checkpoint")


def _residual(evaluation):
    if not evaluation.valid or evaluation.pair is None:
        return None
    return evaluation.pair.shooting_mismatch


def _record(parameter, f, g, evaluation):
    return {
        "parameter": [parameter.numerator, parameter.denominator],
        "f": [[x.numerator, x.denominator] for x in f],
        "g": [[x.numerator, x.denominator] for x in g],
        "evaluation": asdict(evaluation),
    }


def _evaluate_at(parameter, f0, f1, g0, g1, kwargs):
    f = sc.affine_coefficients(f0, f1, parameter)
    g = sc.affine_coefficients(g0, g1, parameter)
    evaluation = sc.evaluate(f, g, **kwargs)
    return _record(parameter, f, g, evaluation), evaluation


def _same_pair(evaluation, tracked):
    pair = evaluation.pair
    return (
        evaluation.valid
        and pair is not None
        and pair.unstable_direction == tracked["unstable_direction"]
        and pair.stable_sign == tracked["stable_sign"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Continue a SPONG shooting residual to a sign bracket")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--extension-steps", type=int, default=8)
    parser.add_argument("--bisections", type=int, default=8)
    parser.add_argument("--maximum-vertices", type=int, default=768)
    parser.add_argument("--geometry-level", type=int, default=0)
    parser.add_argument("--moment-dist", choices=("uniform01", "normal01"),
                        default="uniform01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    left = _candidate(args.left)
    right = _candidate(args.right)
    f0, f1 = _coefficients(left, "f"), _coefficients(right, "f")
    g0, g1 = _coefficients(left, "g"), _coefficients(right, "g")
    right_pair = right["evaluation"]["pair"]
    tracked = {
        key: right_pair[key] for key in (
            "source_b", "target_b", "unstable_direction", "stable_sign")
    }
    kwargs = {
        "moment_dist": args.moment_dist,
        "geometry_level": args.geometry_level,
        "maximum": args.maximum_vertices,
        "shooting_switch": float("inf"),
        "tracked": tracked,
    }

    points = []
    r0, e0 = _evaluate_at(Fraction(0), f0, f1, g0, g1, kwargs)
    r1, e1 = _evaluate_at(Fraction(1), f0, f1, g0, g1, kwargs)
    points.extend((r0, r1))
    s0, s1 = _residual(e0), _residual(e1)
    if s0 is None or s1 is None:
        raise RuntimeError("initial candidates lack signed shooting residuals")
    if not (_same_pair(e0, tracked) and _same_pair(e1, tracked)):
        raise RuntimeError("initial candidates do not track the same pair")

    t0, t1 = Fraction(0), Fraction(1)
    for _ in range(args.extension_steps):
        if s0*s1 <= 0:
            break
        denominator = s1-s0
        if denominator == 0:
            t2 = t1 + (t1-t0)
        else:
            predicted = float(t1) - s1*float(t1-t0)/denominator
            span = abs(float(t1-t0))
            predicted = min(max(
                predicted, float(t1)-2.0*span), float(t1)+2.0*span)
            t2 = Fraction(predicted).limit_denominator(1 << 24)
        r2, e2 = _evaluate_at(t2, f0, f1, g0, g1, kwargs)
        points.append(r2)
        s2 = _residual(e2)
        if s2 is None or not _same_pair(e2, tracked):
            raise RuntimeError("pair tracking failed during continuation")
        t0, s0, t1, s1 = t1, s1, t2, s2
    if s0*s1 > 0:
        raise RuntimeError("continuation did not find a sign bracket")

    if s0 <= 0:
        t0, t1, s0, s1 = t1, t0, s1, s0
    for _ in range(args.bisections):
        tm = (t0+t1)/2
        rm, em = _evaluate_at(tm, f0, f1, g0, g1, kwargs)
        points.append(rm)
        sm = _residual(em)
        if sm is None or not _same_pair(em, tracked):
            raise RuntimeError("pair tracking failed during bisection")
        if sm > 0:
            t0, s0 = tm, sm
        else:
            t1, s1 = tm, sm

    output = {
        "source": {"left": str(args.left), "right": str(args.right)},
        "tracked": tracked,
        "positive_parameter": [t0.numerator, t0.denominator],
        "negative_parameter": [t1.numerator, t1.denominator],
        "positive_residual": s0,
        "negative_residual": s1,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "positive_parameter": float(t0),
        "negative_parameter": float(t1),
        "positive_residual": s0,
        "negative_residual": s1,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
