#!/usr/bin/env python3
"""Does the realisation ratio separate TRUE stage roots from SPURIOUS ones?

stage_root_sweep established that the stage solvers sometimes converge --
cleanly, |R| ~ 1e-16 in three iterations -- on a configuration that is not
the flow, and that no residual test can see it (GL8 at h = 1.5 realises
0.6521 of the requested chord).  This probe measures the candidate test
against ground truth, over a population rather than at one state.

THE RATIO.  |sum b_i K_i| / sum b_i |K_i|, with the weights positive and
summing to 1.  It is 1 exactly when every stage derivative points the same
way with the same magnitude -- a straight, uniformly-traversed arc -- and
falls below 1 by the amount the stages disagree.  Dividing by sum b_i |K_i|
rather than by |h| is what makes it dimensionless on ANY field: the
normalized field has |f| = 1 so the two agree there, but the potential-rate
field has |f| ~ 9e-4 and the |h| form reads 0.0000 for every root, true and
spurious alike -- vacuous exactly where task (3)'s failures live.
Palindromic weights make the ratio invariant under c -> 1-c, so gating on
it cannot cost the reversal.

WHAT THE SHORT VALUES ARE.  In a hemstitch the stages are all +-f, so the
realisable ratios are |sum eps_i b_i| over sign patterns: GL4 {1, 0}, GL6
{1, 4/9, 1/9}, GL8 {1, .6521, .3479, .3043, 0}.  Every short ratio observed
in stage_root_sweep is in that set.  It is discrete and fixed by the
tableau, so the gap below 1 is a property of the method rather than a
number anyone chose.

GROUND TRUTH.  Each converged root's endpoint is compared against the same
field integrated over the same h by Radau at rtol 1e-12, cross-checked
against rtol 1e-10; the error is reported relative to the arclength
travelled.  A STIFF solver is not optional here: the normalized field's
Jacobian carries |grad L| in the denominator, so in the canyon rho(J)
reaches 4e6, and a 512-substep RK4 has its own halving error of 5e-5
relative -- larger than several of the effects being measured.  Rows where
the reference itself did not finish are dropped rather than counted.

WHAT IT FOUND (99912409 at b = -1.0798824489, w = 2.471).  Three classes,
and the ratio addresses exactly one:

  * sound steps          ratio 1 - 1e-9, endpoint error 1e-6 of the arc;
  * SHORT REALISATION    ratio exactly a tableau short value (GL6 0.4444 at
                         h = 1.98, GL8 0.6521 at h = 1.50), endpoint error
                         0.46 and 0.29 of the arc.  1 - ratio tracks that
                         error to within ~1.2x, so the deficit is itself an
                         error estimate and not merely a flag;
  * FULL-LENGTH BUT WRONG  ratio 1 - 1e-9, stages parallel to 1e-4, residual
                         1e-19, endpoint error 0.97 to 1.44 of the arc
                         (GL4 h = 3.4, GL6 h = 4.5 and 6).  The true arc
                         turns 1.87 rad while the field direction at the
                         chord's endpoint sits at 0.295 rad -- the same
                         value it takes on every sound row -- so neither the
                         stage spread nor an endpoint field angle sees it.

The third class fails at a step size where the computation does not make
sense (h rho(J) ~ 2e7) and fails LOUDLY: halve h and the answer moves.  The
second fails at a sensible step size and fails SILENTLY.  That is why the
admissibility test in spong_irk2_step is scoped to arc realisation and the
third class is left to step-size policy.

    python scripts/chord_realisation_probe.py --case directed:99912409 \\
        --state=-1.0798824489,2.471 --h0 1 --h1 6 --n 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from stage_root_sweep import GL, context, make_field, stage_newton  # noqa

# Where a step is called wrong: endpoint off by this fraction of the arc.
# Every measured population sits either below 1e-5 or above 0.26, so nothing
# in the findings turns on the exact value.
WRONG = 0.05


def sign_patterns(order):
    """The ratios a hemstitched configuration can realise."""
    b = GL[order][2]
    s = len(b)
    out = {}
    for mask in range(1 << s):
        eps = np.array([1.0 if (mask >> i) & 1 == 0 else -1.0
                        for i in range(s)])
        out.setdefault(round(abs(float(eps @ b)), 6), []).append(
            "".join("+" if e > 0 else "-" for e in eps))
    return dict(sorted(out.items(), reverse=True))


def realise_min(order):
    """Midway across the tableau's own gap: (1 + largest short value)/2."""
    vals = [v for v in sign_patterns(order) if v < 1.0 - 1e-9]
    return 0.5*(1.0 + (max(vals) if vals else 0.0))


def reference(fj, z, h, rtol=1e-12, atol=1e-14):
    """Endpoint, arclength, ok, step count -- by Radau (see the header)."""
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        return fj(y)[0]

    def jac(_t, y):
        return fj(y)[1]

    sol = solve_ivp(rhs, (0.0, h), z, method="Radau", jac=jac,
                    rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] < 2:
        return z, 0.0, False, 0
    ys = sol.y.T
    arc = float(np.sum(np.hypot(np.diff(ys[:, 0]), np.diff(ys[:, 1]))))
    return ys[-1], arc, True, sol.y.shape[1]


def ratio(order, K):
    b = GL[order][2]
    num = float(np.hypot(*(b @ K)))
    den = float(b @ np.hypot(K[:, 0], K[:, 1]))
    return num/den if den > 0 else float("nan")


def angle(u, v):
    nu = max(float(np.hypot(*u)), 1e-300)
    nv = max(float(np.hypot(*v)), 1e-300)
    return float(np.arccos(np.clip(float(u @ v)/(nu*nv), -1.0, 1.0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="directed:99912409")
    ap.add_argument("--state", default="-1.0798824489,2.471")
    ap.add_argument("--field", default="normalized",
                    choices=("normalized", "potential"))
    ap.add_argument("--orders", default="4,6,8")
    ap.add_argument("--h0", type=float, default=0.05)
    ap.add_argument("--h1", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--flow", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    m, _e, _z = context(args.case)
    b, w = (float(s) for s in args.state.split(","))
    z = np.array([m.s_a_star(b) + w, b])
    fj = make_field(m, args.field)
    f0 = fj(z)[0]
    print(f"# {args.case} at b={b:.10g} w={w:g} ({args.field}), "
          f"|f|={float(np.hypot(*f0)):.4g}")

    rows = []
    for order in [int(x) for x in args.orders.split(",")]:
        print(f"# GL{order} realisable short ratios: "
              + ", ".join(f"{v:.4f}" for v in sign_patterns(order))
              + f"   -> gate at {realise_min(order):.6f}")
        s = len(GL[order][0])
        for hv in np.geomspace(args.h0, args.h1, args.n):
            h = args.flow*float(hv)
            ref, arc, ok_ref, _n = reference(fj, z, h)
            if not ok_ref:
                continue
            ref2, _a, ok2, _n2 = reference(fj, z, h, rtol=1e-10)
            refined = (float(np.hypot(*(ref2 - ref)))/max(arc, 1e-300)
                       if ok2 else float("nan"))
            starts = {"true": np.array([f0]*s)}
            alt = np.array([f0]*s)
            alt[s//2 if s % 2 else 1] = -f0
            starts["alt"] = alt
            for name, K0 in starts.items():
                K, _it, res, ok = stage_newton(fj, z, h, order, K0)
                if not ok:
                    continue
                end = z + h*(GL[order][2] @ K)
                err = float(np.hypot(*(end - ref)))/max(arc, 1e-300)
                u = K/np.hypot(K[:, 0], K[:, 1])[:, None]
                t_stage = float(np.arccos(np.clip(np.min(u @ u.T), -1, 1)))
                rows.append((order, h, name, ratio(order, K), err, res,
                             refined, t_stage, angle(f0, fj(end)[0]),
                             angle(f0, fj(ref)[0])))

    if not args.quiet:
        print(f"{'GL':>3} {'h':>9} {'start':>5} {'ratio':>8} {'1-ratio':>10} "
              f"{'end err/arc':>12} {'turn(K)':>8} {'turn(end)':>9} "
              f"{'turn(ref)':>9} {'|R|':>9} {'ref halve':>10}")
        for o, h, nm, R, err, res, refd, tst, tend, tref in rows:
            flag = "  <-- WRONG" if err > WRONG else ""
            print(f"{o:>3} {h:>9.4g} {nm:>5} {R:>8.4f} {1-R:>10.2e} "
                  f"{err:>12.3e} {tst:>8.4f} {tend:>9.4f} {tref:>9.4f} "
                  f"{res:>9.1e} {refd:>10.1e}{flag}")

    print("\n# classes (converged roots only)")
    for o in sorted({r[0] for r in rows}):
        gate = realise_min(o)
        mine = [r for r in rows if r[0] == o]
        sound = [r for r in mine if r[4] <= WRONG]
        short = [r for r in mine if r[4] > WRONG and r[3] < gate]
        wrong = [r for r in mine if r[4] > WRONG and r[3] >= gate]

        def stat(v):
            if not v:
                return "n=0"
            return (f"n={len(v)} ratio {min(x[3] for x in v):.4f}"
                    f"..{max(x[3] for x in v):.4f} err "
                    f"{min(x[4] for x in v):.1e}..{max(x[4] for x in v):.1e}")
        print(f"  GL{o} gate {gate:.6f}")
        print(f"    sound:            {stat(sound)}")
        print(f"    short realisation:{stat(short)}   (gate REJECTS these)")
        print(f"    full-length wrong:{stat(wrong)}   (gate cannot see these)")
        rejected = [r for r in sound if r[3] < gate]
        print(f"    sound rows the gate would reject: {len(rejected)}"
              + (f"  <-- FALSE POSITIVE at h="
                 + ",".join(f"{r[1]:.4g}" for r in rejected[:5])
                 if rejected else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
