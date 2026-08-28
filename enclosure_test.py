"""Does the enclosure E(a) actually bound the composite's error?

The claim under test.  Along a stable (ascent) branch in the diagonal regime,
the composite tail is

    b_asym(a) = sqrt(d) a - A_{2d-1}/(2 d A_{2d}) + K0 / (2 sqrt(d) a)

with K0 matched at the handoff.  Phi = b^2 - d a^2 + a1 a / sqrt(d) drifts
logarithmically, so Phi has no limit -- but the error that matters is in b,
and delta_b = delta_Phi / 2b, so the REMAINING deviation from the handoff at
a_p out to infinity is

    E(a) = int_a^inf |dPhi/da'| / (2 |b(a')|) da'

which converges like c / (2 sqrt(d) a).  If that is a genuine enclosure then
the measured |b_true - b_asym| must sit under E everywhere past the handoff.
If it pokes above, the bound is wrong.

dPhi/da is exact, no series: with P = b A' - 2 d A = A_{2d} b^(2d+1) alpha'
and Q = b B' - d B,

    dPhi/da = ( 2a[a P - 2 Q] / (2(aA - B)) ) + a1 / sqrt(d)

The true orbit is integrated by the C core's normalized-gradient stepper in a
box far wider than the skeleton, since the diagonal regime is only reachable
there at all.

    python enclosure_test.py tricky-d11
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from spong import atlas, model, portrait, sturm, zoo    # noqa: E402


def polys(m):
    Ac = np.asarray([float(x) for x in m.alpha])[::-1]
    Bc = np.asarray([float(x) for x in m.beta])[::-1]
    return Ac, Bc, np.polyder(Ac), np.polyder(Bc)


def dPhi_da(m, a, b, d, a1, Ac, Bc, Ap, Bp):
    """Exact drift of Phi along the ascent orbit, per unit a."""
    A = np.polyval(Ac, b); B = np.polyval(Bc, b)
    P = b * np.polyval(Ap, b) - (len(Ac) - 1) * A
    Q = b * np.polyval(Bp, b) - (len(Bc) - 1) * B
    ga = 2.0 * (a * A - B)
    if ga == 0.0:
        return float("inf")
    dK = 2.0 * a * (a * P - 2.0 * Q)          # dK/dt
    return dK / ga + a1 / math.sqrt(d)        # dPhi/da


def ascend(m, a0, b0, ds, steps, box, order=8):
    """Integrate the ASCENT orbit with the C core's normalized stepper."""
    k = getattr(m, "_native_kernel", None)
    if k is None or not hasattr(k, "normalized_step"):
        raise SystemExit("no C core: normalized_step unavailable")
    a, b = float(a0), float(b0)
    out = [(a, b)]
    for _ in range(steps):
        try:
            a, b = k.normalized_step(a, b, +ds, order)   # +ds = ascent
        except (ArithmeticError, ValueError, OverflowError, ZeroDivisionError):
            break
        a, b = float(a), float(b)
        if not (a == a and b == b):
            break
        out.append((a, b))
        if not (box[0] <= a <= box[1] and box[2] <= b <= box[3]):
            break
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="tricky-d11")
    ap.add_argument("--reach", type=float, default=1e4,
                    help="how far out in |b| to integrate the true orbit")
    args = ap.parse_args(argv)

    case = zoo.get(args.case)
    D = max(len(case.f) - 1, len(case.g) - 1)
    mu = (model.moments_uniform01 if case.moment_dist == "uniform01"
          else model.moments_normal01)(2 * D + 1)
    m = model.build(case.f, case.g, mu)
    Ac, Bc, Ap, Bp = polys(m)
    d = atlas.effective_degree(m)
    a1 = Ac[1] / Ac[0]
    c0 = -a1 / (2.0 * d)
    print(f"{case.name}: d_eff={d}  a1={a1:.6g}  offset c0={c0:.6g}")

    p = portrait.certified_compute(m, view=case.default_view)
    stable = [br for br in p.branches if br.kind == "stable" and len(br.Y) > 20]
    if not stable:
        raise SystemExit("no stable branches")

    reach = args.reach
    box = (-reach, reach, -reach, reach)
    print(f"integrating ascent out to |.| = {reach:g}\n")

    hdr = (f"{'br':>3} {'a_hand':>10} {'b_hand':>10} {'a_far':>11}"
           f" {'|b_err| max':>13} {'E(a) bound':>13} {'ratio':>8}  verdict")
    print(hdr); print("-" * len(hdr))

    for i, br in enumerate(stable):
        Y = np.asarray(br.Y, dtype=float)
        a_s, b_s = Y[-1]
        # Continue the TRUE orbit far past the compute box.
        span = max(abs(a_s), abs(b_s), 1.0)
        path = ascend(m, a_s, b_s, ds=span * 1e-3, steps=400000, box=box)
        if len(path) < 50:
            print(f"{i:>3}  (orbit did not extend)")
            continue
        A_, B_ = path[:, 0], path[:, 1]
        # Handoff where |b| has gone well past A's outermost root modulus.
        rA = float(np.max(np.abs(np.roots(Ac))))
        far = np.flatnonzero(np.abs(B_) > 30.0 * rA)
        if far.size == 0:
            print(f"{i:>3}  regime 2 (strip): max|b|={np.max(np.abs(B_)):.3g} "
                  f"vs rootA={rA:.3g}, but |a| ran to "
                  f"{np.max(np.abs(A_)):.4g} -- vertical tail, not diagonal")
            continue
        h = int(far[0])
        a_p, b_p = A_[h], B_[h]
        K0 = b_p * b_p - d * a_p * a_p + a1 * a_p / math.sqrt(d)
        sgn = math.copysign(1.0, b_p)

        # Composite vs true, past the handoff.
        idx = np.arange(h, len(A_))
        err = []
        for j in idx:
            aa, bb = A_[j], B_[j]
            disc = d * aa * aa - a1 * aa / math.sqrt(d) + K0
            if disc < 0:
                continue
            err.append(abs(math.copysign(math.sqrt(disc), sgn) - bb))
        if not err:
            continue
        emax = max(err)

        # E(a_p): remaining deviation bound, integrated along the true orbit.
        E = 0.0
        for j in range(h, len(A_) - 1):
            aa, bb = A_[j], B_[j]
            da = A_[j + 1] - aa
            dphi = dPhi_da(m, aa, bb, d, a1, Ac, Bc, Ap, Bp)
            if math.isfinite(dphi) and abs(bb) > 0:
                E += abs(dphi) * abs(da) / (2.0 * abs(bb))
        ok = "OK" if emax <= E else "VIOLATED"
        print(f"{i:>3} {a_p:>10.3f} {b_p:>10.3f} {A_[-1]:>11.4g}"
              f" {emax:>13.4e} {E:>13.4e} {emax/E if E else float('inf'):>8.3f}"
              f"  {ok}")

    print()
    print("verdict compares the measured composite error against the bound;")
    print("E is integrated over the SAME finite arc, so it is the honest")
    print("comparison -- a genuine enclosure needs ratio <= 1 throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
