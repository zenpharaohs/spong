"""Does outcome entropy rise logarithmically as a rheostat nears its wall?

The claim under test.  At a saddle-connection wall the basin boundary passes
exactly through a saddle, so the deterministic flow carries NO information
about which minimum a trajectory reaches -- an arbitrarily small perturbation
flips it.  Meanwhile the lambda-lemma makes the transit time near the
intervening saddle diverge like -ln(distance)/lambda_u.  So batch noise gets a
long window in which to decide, exactly when the landscape has stopped
deciding.  Prediction:

    H(basin | start)  ->  maximum   as   Lambda -> Lambda*
    and the rise is LOGARITHMIC in |Lambda - Lambda*|

THE CONTROL THAT MATTERS.  Near the wall the DETERMINISTIC trajectory also
takes longer, so a fixed step budget truncates more runs before they resolve.
Truncation looks like entropy and is not.  Two guards, both reported:

  * every run is classified resolved / unresolved, and the headline entropy
    counts RESOLVED runs only;
  * the step budget scales with -ln|Lambda - Lambda*|, so every Lambda gets
    comparable opportunity to resolve rather than comparable arithmetic.

`H_all` is printed beside `H_resolved` precisely so the difference is visible.
If they diverge, the budget is doing the talking, not the geometry.

    python wall_entropy.py --family <name> --seeds 24 --starts 32
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from spong import model, sturm, zoo                       # noqa: E402
from demos import initializers                            # noqa: E402
from demos import optimizers as opt                       # noqa: E402


def member_at(family, lam):
    """Coefficients at rheostat coordinate lam, matching zoo.rheostat_member."""
    base = zoo.get(family.base_case)
    root = math.sqrt(lam)
    f = [float(x) / root for x in base.f]
    g = [float(x) * root for x in base.g]
    return f, g, base.moment_dist


def minima_of(m):
    e = sturm.enumerate_critical_points(m)
    return e, [(float(p.a), float(p.b)) for p in e.points if p.kind == "min"]


def entropy(counts):
    n = sum(counts)
    if n <= 0:
        return float("nan")
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def exact_grad(m):
    """Full-batch gradient, for the deterministic polish."""
    A = np.asarray([float(x) for x in m.alpha])[::-1]
    B = np.asarray([float(x) for x in m.beta])[::-1]
    Ap, Bp = np.polyder(A), np.polyder(B)

    def g(a, b):
        return np.array([2.0 * (a * np.polyval(A, b) - np.polyval(B, b)),
                         a * (a * np.polyval(Ap, b)
                              - 2.0 * np.polyval(Bp, b))])
    return g


def polish(m, z, steps=4000, ds=1e-3):
    """Deterministic normalized descent, so classification is not fighting
    the batch-noise floor.  The stochastic run decides WHICH basin; this
    only settles the point inside it."""
    g = exact_grad(m)
    z = np.asarray(z, dtype=float).copy()
    for _ in range(steps):
        v = g(z[0], z[1])
        n = float(np.hypot(v[0], v[1]))
        if not np.isfinite(n) or n < 1e-14:
            break
        z -= ds * v / n
    return z


def separatrix_starts(m, fam, n, jitter=2e-2, seed=0):
    """Starts concentrated where the outcome is actually ambiguous: near the
    source saddle's unstable launch, not spread over the whole view.  The
    coin flip is a property of a neighbourhood of the separatrix."""
    e = sturm.enumerate_critical_points(m)
    sad = [p for p in e.points if p.kind == "saddle"]
    if not sad:
        return None
    src = min(sad, key=lambda p: abs(float(p.b) - fam.source_b))
    a0, b0 = float(src.a), float(src.b)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = jitter * (0.25 + 0.75 * (i + 1) / n)
        th = 2.0 * math.pi * rng.random()
        out.append((a0 + r * math.cos(th), b0 + r * math.sin(th)))
    return np.asarray(out)


def run_one(f, g, z0, seed, steps, lr, batch, box):
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    grad = opt.BatchGradient(f, g, batch_size=batch, rng=rng)
    st = opt.AdamState(np.asarray(z0, dtype=float), lr)
    # run_state returns the whole trajectory; the endpoint is what classifies.
    return np.asarray(opt.run_state(st, grad, steps, box=box))[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None,
                    help="wall family name (default: first available)")
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--base-steps", type=int, default=4000)
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="distance to a minimum counting as resolved")
    ap.add_argument("--fixed-budget", action="store_true",
                    help="disable the -ln|dLambda| budget scaling (the "
                         "control run that should show truncation)")
    args = ap.parse_args(argv)

    fams = zoo.wall_family_names()
    if not fams:
        raise SystemExit("no wall families in this checkout")
    fam = zoo.get_wall_family(args.family or fams[0])
    lam_star = fam.wall_parameter
    print(f"family {fam.name}   base {fam.base_case}   "
          f"{fam.parameter_name}* = {lam_star:.12g}")
    if fam.wall_bracket:
        print(f"  bracket [{fam.wall_bracket[0]:.10g}, "
              f"{fam.wall_bracket[1]:.10g}]")
    print(f"  starts={args.starts} seeds={args.seeds} batch={args.batch}"
          f"  budget={'fixed' if args.fixed_budget else 'scaled by -ln|dL|'}")
    print()

    span = fam.above_parameter - fam.below_parameter
    offsets = [span * r for r in (0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)]

    hdr = (f"{'|dLambda|':>12}{'Lambda':>16}{'steps':>8}{'resolved':>10}"
           f"{'H_resolved':>12}{'H_all':>9}{'H_max':>8}")
    print(hdr); print("-" * len(hdr))

    for off in offsets:
        lam = lam_star - off          # approach from ONE side, as agreed
        f, g, dist = member_at(fam, lam)
        if dist != "uniform01":
            raise SystemExit("this demo oracle assumes uniform01")
        D = max(len(f) - 1, len(g) - 1)
        m = model.build(f, g, model.moments_uniform01(2 * D + 1))
        e, mins = minima_of(m)
        if not mins:
            print(f"{off:>12.3e}{lam:>16.10g}   (no minima)"); continue

        # Budget scales with the predicted lingering time unless disabled.
        steps = args.base_steps
        if not args.fixed_budget:
            steps = int(args.base_steps * max(1.0, math.log(span / off)))

        view = fam.default_view
        starts = separatrix_starts(m, fam, args.starts)
        if starts is None:
            starts = initializers.low_discrepancy(args.starts, view)
        box = (view[0] - 4, view[1] + 4, view[2] - 4, view[3] + 4)
        lr = args.lr if args.lr is not None else 1e-2

        per_start_H, per_start_H_all, nres, ntot = [], [], 0, 0
        for z0 in starts:
            counts = [0] * (len(mins) + 1)      # last slot = unresolved
            cres = [0] * len(mins)
            for s in range(args.seeds):
                z = run_one(f, g, z0, s, steps, lr, args.batch, box)
                if np.all(np.isfinite(z)):
                    z = polish(m, z)
                ntot += 1
                dists = [math.hypot(z[0] - ma, z[1] - mb) for ma, mb in mins]
                k = int(np.argmin(dists))
                if dists[k] <= args.tol:
                    counts[k] += 1; cres[k] += 1; nres += 1
                else:
                    counts[-1] += 1
            if sum(cres) >= 2:
                per_start_H.append(entropy(cres))
            per_start_H_all.append(entropy(counts))
        Hr = float(np.mean(per_start_H)) if per_start_H else float("nan")
        Ha = float(np.mean(per_start_H_all)) if per_start_H_all else float("nan")
        print(f"{off:>12.3e}{lam:>16.10g}{steps:>8}"
              f"{100.0*nres/max(ntot,1):>9.1f}%{Hr:>12.4f}{Ha:>9.4f}"
              f"{math.log2(len(mins)):>8.4f}")

    print()
    print("H_resolved is the headline: entropy over runs that actually")
    print("reached a minimum.  If H_all climbs while H_resolved does not,")
    print("the step budget is being measured, not the geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
