#!/usr/bin/env python3
"""Construct a clean saddle-connection search family through an inverse wall.

The exact inverse construction is linear only for fixed ``g``.  That slice can
place a remote saddle-node only after crossing other bifurcations.  This demo
therefore solves the same inverse equations ``N(r)=N'(r)=0`` in the full
``(f,g,r)`` space with a row-scaled, minimum-norm modified-Gram--Schmidt
corrector.  Dyadic models just beyond the wall are then screened by exact
Sturm enumeration and the production portrait.

Use ``continue_saddle_connection.py`` on two emitted candidates whose signed
shooting residuals have opposite signs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demos import saddle_connections as sc
from spong import model


def _candidate(path):
    data = json.loads(Path(path).read_text())
    if "arms" in data:
        return min(data["arms"], key=lambda x: x["evaluation"]["score"])
    if "f" in data and "g" in data:
        return data
    raise ValueError(f"{path} is not a candidate or search checkpoint")


def _critical_polynomial(f, g, moments):
    A = np.convolve(g, g)*moments[:2*len(g)-1]
    B = g*np.asarray([
        sum(f[i]*moments[i+j] for i in range(len(f)))
        for j in range(len(g))])
    Ap = np.arange(1, len(A))*A[1:]
    Bp = np.arange(1, len(B))*B[1:]
    return np.convolve(Ap, B)-2.0*np.convolve(Bp, A)


def _polyval(coefficients, x):
    value = 0.0
    for coefficient in coefficients[::-1]:
        value = value*x+coefficient
    return value


def wall_residual(state, degree, moments):
    n = degree+1
    N = _critical_polynomial(state[:n], state[n:2*n], moments)
    Np = np.arange(1, len(N))*N[1:]
    r = state[-1]
    return np.asarray([_polyval(N, r), _polyval(Np, r)])


def minimum_norm_rows(rows, rhs):
    """Solve J x=rhs in row space by MGS QR of J.T."""
    rows = np.asarray(rows, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    count = len(rhs)
    Q = []
    R = np.zeros((count, count))
    for i, row in enumerate(rows):
        v = row.copy()
        for j, q in enumerate(Q):
            R[j, i] = np.dot(q, v)
            v -= R[j, i]*q
        R[i, i] = np.hypot.reduce(v)
        if R[i, i] <= 1e-13:
            return None
        Q.append(v/R[i, i])
    y = np.zeros(count)
    for i in range(count):
        y[i] = (rhs[i]-sum(R[j, i]*y[j] for j in range(i)))/R[i, i]
    return sum((y[i]*Q[i] for i in range(count)),
               np.zeros(rows.shape[1]))


def inverse_wall(f, g, moments, collision_b, iterations=12):
    """Nearest full-space numerical solution of N(r)=N'(r)=0."""
    degree = max(len(f), len(g))-1
    n = degree+1
    state = np.r_[f, g, float(collision_b)]
    for _ in range(iterations):
        residual = wall_residual(state, degree, moments)
        jacobian = np.empty((2, 2*n+1))
        for j in range(2*n+1):
            h = 1e-7*max(1.0, abs(state[j]))
            plus, minus = state.copy(), state.copy()
            plus[j] += h
            minus[j] -= h
            jacobian[:, j] = (
                wall_residual(plus, degree, moments)
                - wall_residual(minus, degree, moments))/(2.0*h)
        scales = np.maximum(np.max(np.abs(jacobian), axis=1),
                            np.abs(residual))
        step = minimum_norm_rows(
            jacobian/scales[:, None], -residual/scales)
        if step is None:
            raise ArithmeticError("rank-deficient inverse-wall corrector")
        baseline = np.hypot.reduce(residual/scales)
        accepted = False
        for exponent in range(14):
            rate = 2.0**(-exponent)
            trial = state+rate*step
            if np.hypot.reduce(
                    wall_residual(trial, degree, moments)/scales) < baseline:
                state = trial
                accepted = True
                break
        if baseline < 1e-11:
            break
        if not accepted:
            raise ArithmeticError("inverse-wall Armijo corrector stalled")
    return state


def _dyadic(values, bits):
    denominator = 1 << int(bits)
    return tuple(Fraction(round(x*denominator), denominator) for x in values)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross a remote saddle-node and emit clean candidates")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--collision-b", type=float, required=True)
    parser.add_argument("--factors", type=float, nargs="+",
                        default=(1.02, 1.05, 1.10))
    parser.add_argument("--coefficient-bits", type=int, default=28)
    parser.add_argument("--maximum-vertices", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = _candidate(args.candidate)
    f0 = np.asarray([n/d for n, d in source["f"]], dtype=float)
    g0 = np.asarray([n/d for n, d in source["g"]], dtype=float)
    if len(f0) != len(g0):
        parser.error("hybrid wall construction currently requires deg(f)=deg(g)")
    degree = max(len(f0), len(g0))-1
    exact_moments = model.moments_uniform01(2*degree+1)
    moments = np.asarray(list(map(float, exact_moments)))
    wall = inverse_wall(f0, g0, moments, args.collision_b)
    delta_f, delta_g = wall[:len(f0)]-f0, wall[len(f0):-1]-g0

    candidates = []
    for factor in args.factors:
        f = _dyadic(f0+factor*delta_f, args.coefficient_bits)
        g = _dyadic(g0+factor*delta_g, args.coefficient_bits)
        evaluation = sc.evaluate(
            f, g, maximum=args.maximum_vertices, shooting_switch=10.0)
        candidates.append({
            "factor": factor,
            "f": [[x.numerator, x.denominator] for x in f],
            "g": [[x.numerator, x.denominator] for x in g],
            "evaluation": asdict(evaluation),
        })

    output = {
        "source": str(args.candidate),
        "collision_b": args.collision_b,
        "wall": wall.tolist(),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps([
        {
            "factor": x["factor"],
            "valid": x["evaluation"]["valid"],
            "saddles": x["evaluation"]["n_saddles"],
            "minima": x["evaluation"]["n_minima"],
            "shooting_mismatch": (
                x["evaluation"]["pair"]["shooting_mismatch"]
                if x["evaluation"]["pair"] else None),
        }
        for x in candidates
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
