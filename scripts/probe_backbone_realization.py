"""Degree-five Chebyshev-backbone pilot under two fixed continuous laws.

First prescribe the cross-moment polynomial B, not the full loss
u=C-B**2/A. For f(x)=x**k, its b**j coefficient is g[j]*mu[k+j],
so nonzero moment weights give exact rational g coefficients. Also
measure one small-integer fit to a feasible-tail full-loss target.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb

import numpy as np

from spong import filter_family, model, portrait, sturm


def chebyshev_target(law: str) -> tuple[Fraction, ...]:
    p = filter_family.chebyshev_coefficients(5)
    if law == "normal01":
        return p
    # T_5(2b-1): the five intended saddle positions lie in (0,1).
    return tuple(sum((p[k] * comb(k, j) * 2**j * (-1)**(k-j)
                      for k in range(j, 6)), Fraction(0))
                 for j in range(6))


def candidate(law: str, target_degree: int
              ) -> tuple[model.Model, tuple[Fraction, ...]]:
    if law not in ("uniform01", "normal01"):
        raise ValueError("law must be uniform01 or normal01")
    if target_degree not in (1, 5):
        raise ValueError("target_degree must be 1 or 5")
    mu = (model.moments_uniform01 if law == "uniform01" else
          model.moments_normal01)(11)
    p = chebyshev_target(law)
    g = [p[j] / mu[target_degree + j]
         if mu[target_degree + j] else Fraction(0) for j in range(6)]
    if law == "normal01":
        # E[f]=0 makes B(0)=0 regardless of g[0]; a nonzero constant
        # keeps A(0)>0 and avoids a degenerate neuron at b=0.
        g[0] = Fraction(1, 100)
    f = (0,) * target_degree + (1,)
    m = model.build(f, g, mu)
    assert m.beta == p
    return m, p


def summarize(m: model.Model, p: tuple[Fraction, ...], law: str,
              target_tail: Fraction, design: str, certify: bool) -> dict:
    e = sturm.enumerate_critical_points(m)
    grid = (np.linspace(0, 1, 1001) if law == "uniform01"
            else np.linspace(-1, 1, 1001))
    p_on_grid = np.polynomial.polynomial.polyval(grid, np.asarray(p, float))
    target = float(target_tail) + (1 - float(target_tail)) / (1 + p_on_grid**2)
    realized = m.u(grid) / float(m.C)
    degree = len(m.g) - 1
    cross = sum((coefficient * m.mu[i + degree]
                 for i, coefficient in enumerate(m.f)), Fraction(0))
    tail = 1 - cross**2 / (m.C * m.mu[2*degree])
    out = {
        "law": law,
        "design": design,
        "f": list(map(str, m.f)),
        "g": list(map(str, m.g)),
        "B": list(map(str, m.beta)),
        "C": str(m.C),
        "tail_over_C": float(tail),
        "target_tail_over_C": float(target_tail),
        "rms_target_error": float(np.sqrt(np.mean((realized - target)**2))),
        "saddle_b": [q.b for q in e.saddles],
        "minimum_b": [q.b for q in e.minima],
        "minimum_loss_over_C": [float(m.u(q.b) / m.C) for q in e.minima],
        "morse": e.morse,
        "A_positive": e.psi_positive,
    }
    if certify:
        out["portrait_status"] = portrait.compute(m).ledger["summary"]["topology_status"]
    return out


def measure(law: str, target_degree: int, certify: bool) -> dict:
    m, p = candidate(law, target_degree)
    return summarize(m, p, law, Fraction(0), "B-prescribed", certify)


def integer_normal_fit(certify: bool) -> dict:
    """Small-integer degree-five g fitted to an attainable-tail ripple."""
    m = model.build((0, 1), (3, 6, -2, -8, -1, 1),
                    model.moments_normal01(11))
    p = chebyshev_target("normal01")
    return summarize(m, p, "normal01", Fraction(16, 21),
                     "full-loss integer fit", certify)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portrait", action="store_true",
                        help="run the full portrait audit for each case")
    args = parser.parse_args()
    for law in ("uniform01", "normal01"):
        for degree in (1, 5):
            print(json.dumps(measure(law, degree, args.portrait)), flush=True)
    print(json.dumps(integer_normal_fit(args.portrait)), flush=True)


if __name__ == "__main__":
    main()
