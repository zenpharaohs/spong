"""Can an unbounded sublevel component still trap the orbit?

    python scripts/escape_probe.py

_capture_certificate demands a BOUNDED one-minimum sublevel component.  On the
9 refusing directed models the component is certified, holds the right
minimum, and is unbounded -- because at a dead-neuron minimum u(b) = C - B^2/A
flattens to a finite asymptote at or below the minimum's level, so the
component runs off along the backbone.

But boundedness is SUFFICIENT for the trap, not necessary.  Descent cannot
leave a sublevel set, so the orbit converges to the unique minimum unless it
escapes along an unbounded end -- and it cannot escape an end that is uphill
outward.  Along the backbone the loss is u(b), so the test is a sign condition
on u' = B*N/A^2, hence on B*N alone since A > 0:

    right end (b -> +inf) blocked iff  B*N > 0 on the ray
    left  end (b -> -inf) blocked iff  B*N < 0 on the ray

That is degree 18 on these models against A^4's 40, and _strictly_positive_on
_ray already certifies exactly this kind of claim -- the funnel's `outward`
test is the same polynomial.

Prints, per failing branch: which ends are unbounded, and whether B*N has the
blocking sign on each, exactly.
"""
from __future__ import annotations

import os
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait, topology, _poly as P           # noqa: E402
from qualify import directed_model                          # noqa: E402


def largest_real_root(coeffs):
    c = [float(x) for x in coeffs]
    while c and c[-1] == 0.0:
        c.pop()
    if len(c) < 2:
        return 0.0
    r = [abs(z.real) for z in np.roots(c[::-1])
         if abs(z.imag) < 1e-9 * max(1.0, abs(z))]
    return max(r, default=0.0)


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    rng = random.Random(20260806)
    blocked = open_end = 0

    for case in range(20):
        seed = rng.randrange(2 ** 31)
        sub = random.Random(seed)
        m, spec = directed_model(sub, 5)
        if m is None:
            continue
        p = portrait.certified_compute(m)
        top = p.ledger["topology"]
        if top.get("resolution_reason") != "unstable_endpoint_unresolved":
            continue
        bad = [e for e in top.get("unstable_ends", []) if not e.get("certified")]
        if not bad:
            continue

        BN = P.mul(m.beta, m.N)
        root = largest_real_root(BN)
        print(f"\n[{case}] seed {seed}  {spec}")
        print(f"    deg(B*N) = {P.degree(BN)}   largest |root| {root:.6g}")

        for end in bad:
            i = end["branch"]
            br = p.branches[i]
            idx = min(len(br.Y) - 2,
                      max(0, int(br.diag.get("critical_steps", 0))))
            inv = topology._sublevel_component_inventory(
                m, p.enumeration, br.Y[idx])
            left, right = inv.get("left_boundary"), inv.get("right_boundary")
            print(f"    br{i}  minima={len(inv.get('minima', []))} "
                  f"saddles={len(inv.get('saddles', []))} "
                  f"left={'unbounded' if left is None else 'closed'} "
                  f"right={'unbounded' if right is None else 'closed'}")

            b0 = Fraction.from_float(
                float(max(root, abs(float(br.Y[idx, 1])))) * 1.0625 + 1.0)
            for side, is_open in (("right", left is None and False),
                                  ("right", right is None),
                                  ("left", left is None)):
                if not is_open:
                    continue
                if side == "right":
                    ok = topology._strictly_positive_on_ray(BN, b0, 1)
                    want = "B*N > 0 on [b0, +inf)"
                else:
                    ok = topology._strictly_positive_on_ray(
                        P.scale(BN, Fraction(-1)), -b0, -1)
                    want = "B*N < 0 on (-inf, -b0]"
                blocked += ok
                open_end += (not ok)
                print(f"        {side:<5s} end: {want:<26s} "
                      + ("BLOCKED -- no escape" if ok
                         else "not established"))

    print(f"\nunbounded ends provably uphill outward: {blocked}")
    print(f"unbounded ends not established:         {open_end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
