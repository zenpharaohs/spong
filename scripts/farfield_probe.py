"""Why does _unstable_far_field_funnel refuse?

    python scripts/farfield_probe.py

The funnel needs four polynomials strictly positive on the COMPLETE ray to
infinity, and one of them is outward = -direction * B * N.  A real root of B
or N beyond the branch endpoint defeats that test however well behaved the
trajectory is.

atlas.legal_max_b is 1.5x the Cauchy bound of N and B -- beyond it no such
root exists.  But atlas.compute_box only CLAMPS to that bound; with view=None
it builds the skeleton's bounding box plus 20%, which for a directed model can
sit far inside.

So the two outcomes mean different work.  Endpoints INSIDE the last root: the
refusal is a REACH problem, the branch stops before the far field is provably
clean, and tracing to legal_max_b fixes it.  Endpoints PAST all roots: reach
is not the issue and the failure is in the three corridor tests, where the
A^4 degree lives and where a dK/dt certificate at degree 2d+2 would be the
real work.

Reruns the 20 directed models of the qualification suite at the same seed.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import atlas, model, portrait, _poly as P     # noqa: E402
from qualify import directed_model                        # noqa: E402


def largest_real_root(coeffs):
    c = [float(x) for x in coeffs]
    while c and c[-1] == 0.0:
        c.pop()
    if len(c) < 2:
        return None
    r = [z.real for z in np.roots(c[::-1])
         if abs(z.imag) < 1e-9 * max(1.0, abs(z))]
    return max((abs(x) for x in r), default=None)


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    rng = random.Random(20260806)

    inside_n = past_n = 0
    for case in range(20):
        seed = rng.randrange(2 ** 31)
        sub = random.Random(seed)
        try:
            m, spec = directed_model(sub, 5)
        except Exception as exc:
            print(f"[{case}] model rejected: {exc}")
            continue
        if m is None:
            continue
        try:
            p = portrait.certified_compute(m)
        except Exception as exc:
            print(f"[{case}] failed: {type(exc).__name__}: {exc}")
            continue
        top = p.ledger["topology"]
        if top.get("resolution_reason") != "unstable_endpoint_unresolved":
            continue

        bmax = float(atlas.legal_max_b(m))
        rootBN = largest_real_root(P.mul(m.beta, m.N))
        print(f"\n[{case}] seed {seed}  {spec}")
        print(f"      legal_max_b {bmax:12.4g}   largest |root| of B*N "
              + ("none" if rootBN is None else f"{rootBN:12.4g}"))
        for i, br in enumerate(p.branches):
            if br.kind != "unstable" or br.term != "box_exit":
                continue
            a_e, b_e = float(br.Y[-1, 0]), float(br.Y[-1, 1])
            A = float(m.A(b_e))
            B = float(m.B(b_e))
            astar = (B / A) if A else float("nan")
            ratio = (a_e / astar - 1.0) if astar else float("inf")
            inside = rootBN is not None and abs(b_e) < rootBN
            inside_n += inside
            past_n += (not inside)
            print(f"      br{i:<3d} end=({a_e:11.4g},{b_e:11.4g}) "
                  f"a*={astar:11.4g} a/a*-1={ratio:10.2e} "
                  f"h*ratio={abs(b_e) * ratio:10.2e} "
                  + ("INSIDE the last root" if inside else "past all roots"))

    print(f"\nendpoints inside the last B*N root: {inside_n}")
    print(f"endpoints past all roots:           {past_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
