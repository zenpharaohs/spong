#!/usr/bin/env python3
"""Signed-measure models: indefinite moment functional, A(b) > 0, Morse.

WHAT THESE ARE.  Any finite real sequence mu_0..mu_2D is the moment
sequence of SOME signed measure, so the SPONG problem statement never
becomes meaningless when the Hankel matrix loses positive definiteness --
it becomes least squares in a KREIN space.  The bilinear form has a
signature, "squared lengths" can be negative, and L = C - 2aB + a^2 A is
still exactly the same quadratic in a.  Every identity the certification
rests on -- completing the square, u = C - B^2/A, grad L, det H = 2A u'' --
is algebra in the coefficients and does not consult the signature.

This is not exotic.  Newton-Cotes has negative weights from order 9, many
simplex cubatures do, and so do extrapolation and control-variate schemes:
anyone computing "expectations" that way and handing over the moments has
produced exactly this input.  It is also the same phenomenon the Krein
generalization of RKHS interpolation handles on the ACTIVATION side, where
tanh, softplus and the sigmoids are not positive-definite kernels at all
and split as K_+ - K_- by Schwartz's theorem.

THE HYPOTHESIS THAT STILL MATTERS is A(b) > 0 for all real b -- spong's
psi_positive, decided exactly by Sturm.  A(b) = <g(b.), g(b.)> is the form
evaluated on the one-dimensional span of g(b.), so A > 0 says the form is
positive on exactly the subspace where a is optimised, which is all that
minimising over a requires.  An indefinite Hankel can satisfy it: the
negative cone simply misses the Veronese curve (g_0, g_1 b, ..., g_d b^d).
That is the regime this script generates.

WHAT IS LOST is L >= 0 and the reading of L as an approximation error.  C
can be negative, u can be negative, and a minimum can sit where the
residual has negative square.  The landscape spong certifies is correct;
the inference from "global minimum" to "best approximation" is not
available.

    python scripts/signed_measure.py --count 6
    python scripts/signed_measure.py --newton-cotes 9
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("SPONG_ENGINE", "native")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spong import model, resolution, sturm                       # noqa: E402


def signed_moments(atoms, weights, count):
    """mu_k = sum_i w_i x_i^k, exactly.  Normalised so mu_0 = 1."""
    mu = []
    for k in range(count):
        mu.append(sum((w * x**k for x, w in zip(atoms, weights)),
                      Fraction(0)))
    if mu[0] == 0:
        raise ValueError("total mass is zero")
    return tuple(m / mu[0] for m in mu)


def hankel_inertia(mu, size):
    """(n_positive, n_negative) of the size x size Hankel matrix, EXACTLY.

    By Jacobi's rule the number of negative eigenvalues is the number of
    sign changes in the sequence of leading principal minors, provided none
    of them vanishes.  Returns None on a zero minor: the signature is then
    not decidable this way and the case is skipped rather than guessed.
    """
    minors = [Fraction(1)]
    for n in range(1, size + 1):
        rows = [[mu[i + j] for j in range(n)] for i in range(n)]
        det = _det_exact(rows)
        if det == 0:
            return None
        minors.append(det)
    neg = sum(1 for i in range(size)
              if (minors[i] > 0) != (minors[i + 1] > 0))
    return size - neg, neg


def _det_exact(rows):
    """Exact determinant by Fraction Gaussian elimination."""
    a = [row[:] for row in rows]
    n = len(a)
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        det *= a[col][col]
        inv = Fraction(1) / a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] * inv
            if factor:
                for c in range(col, n):
                    a[r][c] -= factor * a[col][c]
    return det


def examine(tag, f, g, mu, verbose=True):
    """Build, classify, and report one candidate."""
    degree = max(len(f), len(g)) - 1
    size = degree + 1
    inertia = hankel_inertia(mu, size)
    if inertia is None:
        return None
    positive, negative = inertia
    m = model.build(f, g, mu)
    e = sturm.enumerate_critical_points(m)
    kinds = {}
    for q in e.points:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    record = {
        "tag": tag, "f": list(f), "g": list(g),
        "hankel_signature": (positive, negative),
        "indefinite": negative > 0,
        "A_positive": e.psi_positive,
        "morse": e.morse,
        "kinds": kinds,
        "C": m.C,
    }
    if verbose:
        print(f"  {tag}: Hankel signature ({positive}+, {negative}-)  "
              f"A>0 {e.psi_positive}  Morse {e.morse}  {kinds}  "
              f"C = {float(m.C):.6g}")
    return record


def random_signed_case(rng, degree, n_atoms, span):
    """Atoms and weights over small rationals, at least one weight < 0."""
    atoms = []
    while len(atoms) < n_atoms:
        x = Fraction(rng.randrange(-span, span + 1), rng.randrange(1, 4))
        if x not in atoms:
            atoms.append(x)
    weights = [Fraction(rng.randrange(1, 6), rng.randrange(1, 4))
               for _ in atoms]
    # Make one or two weights negative: that is what breaks definiteness.
    for _ in range(rng.choice((1, 1, 2))):
        weights[rng.randrange(len(weights))] *= -1
    f = [Fraction(rng.randrange(-3, 4)) for _ in range(2)]
    g = [Fraction(rng.randrange(-3, 4)) for _ in range(degree + 1)]
    if g[-1] == 0:
        g[-1] = Fraction(1)
    if all(x == 0 for x in f):
        f[1] = Fraction(1)
    return atoms, weights, f, g


def newton_cotes_weights(n):
    """Closed Newton-Cotes weights on [0, 1] with n+1 nodes, EXACT.

    Negative weights appear from n = 8 (order 9); that is the textbook
    signed measure, and its Hankel matrix is indefinite for that reason.
    """
    nodes = [Fraction(i, n) for i in range(n + 1)]
    weights = []
    for i in range(n + 1):
        # integral over [0,1] of the i-th Lagrange basis polynomial
        num = [Fraction(1)]
        den = Fraction(1)
        for j, xj in enumerate(nodes):
            if j == i:
                continue
            num = _poly_mul(num, [-xj, Fraction(1)])
            den *= nodes[i] - xj
        integral = sum((c / (k + 1) for k, c in enumerate(num)), Fraction(0))
        weights.append(integral / den)
    return nodes, weights


def _poly_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8,
                    help="how many indefinite Morse cases to find")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--atoms", type=int, default=5)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--tries", type=int, default=2000)
    ap.add_argument("--newton-cotes", type=int, default=0,
                    help="also report closed Newton-Cotes of this order")
    ap.add_argument("--certify", action="store_true",
                    help="run resolution.resolve on each keeper")
    args = ap.parse_args(argv)

    if args.newton_cotes:
        n = args.newton_cotes
        nodes, weights = newton_cotes_weights(n)
        neg = [float(w) for w in weights if w < 0]
        print(f"closed Newton-Cotes, {n+1} nodes: "
              f"{len(neg)} negative weight(s) {neg[:4]}")
        mu = signed_moments(nodes, weights, 2*args.degree + 1)
        rng = random.Random(args.seed)
        for i in range(4):
            _a, _w, f, g = random_signed_case(
                rng, args.degree, args.atoms, args.span)
            examine(f"newton-cotes-{n} case {i}", f, g, mu)
        print()

    rng = random.Random(args.seed)
    keepers = []
    for attempt in range(args.tries):
        atoms, weights, f, g = random_signed_case(
            rng, args.degree, args.atoms, args.span)
        try:
            mu = signed_moments(atoms, weights, 2*args.degree + 1)
        except ValueError:
            continue
        rec = examine("", f, g, mu, verbose=False)
        if rec is None:
            continue
        if rec["indefinite"] and rec["A_positive"] and rec["morse"]:
            rec["atoms"] = atoms
            rec["weights"] = weights
            rec["mu"] = mu
            keepers.append(rec)
            print(f"[{len(keepers)}] after {attempt+1} tries: "
                  f"signature {rec['hankel_signature']}  {rec['kinds']}  "
                  f"C = {float(rec['C']):.6g}")
            print(f"    f = {[str(x) for x in f]}")
            print(f"    g = {[str(x) for x in g]}")
            print(f"    atoms   = {[str(x) for x in atoms]}")
            print(f"    weights = {[str(x) for x in weights]}")
            if args.certify:
                m = model.build(f, g, mu)
                r = resolution.resolve(m)
                print(f"    resolve: {r.status.value} / {r.reason.value}")
            if len(keepers) >= args.count:
                break
    if not keepers:
        print("no indefinite/A>0/Morse case found; widen --tries or --span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
