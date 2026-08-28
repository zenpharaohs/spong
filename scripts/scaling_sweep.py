"""Does poor SCALING explain the uncertified cases?

Seed 953953598 turned out not to be near any degeneracy: |g(0)|/||g|| = 0.017,
psi margin 2.7e-4, every root gap O(1).  What it has instead is E[f] = 1.34e6,
so the backbone a* = E[f]/g(0) sits at -4.4e7 while the saddles are at
a ~ 1e-42.  That is a units problem, not a shape problem -- and units are
exactly what the three-parameter scaling group can fix:

    f -> f/tau   with   a -> a/tau      (loss scales by tau^2)
    g -> g/sigma with   a -> a*sigma
    x -> x/lambda with  b -> b*lambda

The scale-INVARIANT question is whether the portrait's shape is wide-ranged
or merely its coordinates are.  The diagnostic below is

    kappa = |a*(0)| * ||g||_inf / ||f||_inf

which is unchanged by the first two scalings.  kappa ~ 1 means the model is
merely badly normalised and a rescale would fix it outright; kappa >> 1 means
the dynamic range is intrinsic and rescaling cannot help.

Reads the ensemble JSONL, rebuilds each model from its recorded seed, and
groups by status/reason.  No tracing, so the whole run is seconds.

    python scripts/scaling_sweep.py out/ensemble-directed-d5.jsonl --mode directed
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SPONG_ENGINE", "native")

from qualify import directed_model, random_model        # noqa: E402


def measure(m):
    f = [float(c) for c in m.f]
    g = [float(c) for c in m.g]
    mu = [float(c) for c in m.mu]
    A = np.asarray([float(c) for c in m.alpha])[::-1]
    fscale = max(abs(c) for c in f) or 1.0
    gscale = max(abs(c) for c in g) or 1.0
    g0 = g[0]
    Ef = sum(c*mm for c, mm in zip(f, mu))
    C = float(m.C)
    astar0 = (Ef/g0) if g0 else float("inf")
    kappa = abs(astar0) * gscale / fscale if fscale else float("inf")

    # psi margin: how close A comes to zero, relative to its own scale
    Ap = np.polyder(A)
    rr = np.roots(Ap) if len(Ap) > 1 else np.array([])
    rr = rr[np.abs(rr.imag) < 1e-9*np.maximum(1.0, np.abs(rr))].real
    vals = [float(np.polyval(A, r)) for r in rr] + [float(np.polyval(A, 0.0))]
    Ascale = max(max(abs(v) for v in vals), abs(float(A[0])), 1.0)
    psi = min(vals)/Ascale if vals else float("nan")
    return {"Ef": Ef, "C": C, "fscale": fscale, "gscale": gscale,
            "g0_margin": abs(g0)/gscale, "astar0": astar0,
            "kappa": kappa, "psi": psi}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--mode", choices=("random", "directed"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    args = ap.parse_args(argv)

    generate = directed_model if args.mode == "directed" else random_model
    rows = []
    for line in Path(args.jsonl).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        built = generate(random.Random(rec["seed"]), args.degree)
        if built is None or built[0] is None:
            continue
        m, _spec = built
        try:
            d = measure(m)
        except Exception:                               # noqa: BLE001
            continue
        d["key"] = (rec.get("reason") or "certified")
        d["seconds"] = rec.get("seconds", 0.0)
        d["seed"] = rec["seed"]
        rows.append(d)

    print(f"{len(rows)} cases from {args.jsonl}\n")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["key"]].append(r)

    def med(vals):
        return statistics.median(vals) if vals else float("nan")

    hdr = (f"{'group':<32}{'n':>5}{'med E[f]':>12}{'med sqrt(C)':>13}"
           f"{'med |a*(0)|':>13}{'med kappa':>11}{'med psi':>11}"
           f"{'med g0marg':>12}")
    print(hdr); print("-"*len(hdr))
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        gr = groups[key]
        print(f"{key:<32}{len(gr):>5}"
              f"{med([r['Ef'] for r in gr]):>12.4g}"
              f"{med([abs(r['C'])**0.5 for r in gr]):>13.4g}"
              f"{med([abs(r['astar0']) for r in gr]):>13.4g}"
              f"{med([r['kappa'] for r in gr]):>11.4g}"
              f"{med([r['psi'] for r in gr]):>11.4g}"
              f"{med([r['g0_margin'] for r in gr]):>12.4g}")

    # Does kappa track COST?  The expensive cases are the ones that matter.
    slow = sorted(rows, key=lambda r: -r["seconds"])[:10]
    print(f"\n  ten slowest cases:")
    print(f"    {'seconds':>9}{'seed':>13}{'E[f]':>12}{'|a*(0)|':>12}"
          f"{'kappa':>11}{'psi':>11}{'g0marg':>10}  reason")
    for r in slow:
        print(f"    {r['seconds']:>9.1f}{r['seed']:>13}{r['Ef']:>12.4g}"
              f"{abs(r['astar0']):>12.4g}{r['kappa']:>11.4g}"
              f"{r['psi']:>11.4g}{r['g0_margin']:>10.4g}  {r['key']}")

    fast = sorted(rows, key=lambda r: r["seconds"])[:10]
    print(f"\n  ten fastest cases (for contrast):")
    for r in fast:
        print(f"    {r['seconds']:>9.1f}{r['seed']:>13}{r['Ef']:>12.4g}"
              f"{abs(r['astar0']):>12.4g}{r['kappa']:>11.4g}"
              f"{r['psi']:>11.4g}{r['g0_margin']:>10.4g}  {r['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
