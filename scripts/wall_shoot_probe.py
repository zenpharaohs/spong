#!/usr/bin/env python3
"""Shoot a wall family: delta(Lambda) at the three stored members, then
Brent's method to the binary64 root, against the family's bracket.

    python scripts/wall_shoot_probe.py                       # all families
    python scripts/wall_shoot_probe.py nonnearest-saddle-connection --ds 1e-3

Prints, per family: delta at below / wall / above, the Brent root Lambda*
with delta there, the evaluation count and history, whether Lambda* lies
inside the family's citable wall_bracket, and how far the two shots at
Lambda* end from the saddles they were aimed at.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("SPONG_ENGINE", "native")
try:
    from spong import wall_shoot, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import wall_shoot, zoo


def probe(name: str, ds: float) -> None:
    family = zoo.get_wall_family(name)
    print(f"\n{name}  ({family.parameter_name}: below {family.below_parameter!r}"
          f"  wall {family.wall_parameter!r}  above {family.above_parameter!r})")
    for label, lam in (("below", family.below_parameter),
                       ("wall", family.wall_parameter),
                       ("above", family.above_parameter)):
        t0 = time.perf_counter()
        m = wall_shoot.rheostat_model(family, lam)
        shot = wall_shoot.shoot(m, family.source_b, family.unstable_direction,
                                family.target_b, ds=ds)
        dt = time.perf_counter() - t0
        if shot is None:
            print(f"  {label:6s} {lam!r}: shot failed  ({dt:.2f}s)")
            continue
        print(f"  {label:6s} {lam!r}: delta {shot.delta:+.6e}  "
              f"level {shot.level:.6f}  stable {shot.target_direction:+d}  "
              f"steps {shot.steps}  ({dt:.2f}s)")
    t0 = time.perf_counter()
    try:
        root = wall_shoot.find_wall(family, ds=ds)
    except ValueError as exc:
        print(f"  Brent refused: {exc}")
        return
    dt = time.perf_counter() - t0
    inside = (family.wall_bracket is not None
              and family.wall_bracket[0] < root.lam < family.wall_bracket[1])
    print(f"  Brent: {family.parameter_name}* = {root.lam!r}  "
          f"delta* = {root.delta:+.3e}  evaluations {root.evaluations}  "
          f"({dt:.1f}s)")
    print(f"         stored wall {family.wall_parameter!r}  "
          f"offset {root.lam - family.wall_parameter:+.3e}")
    if family.wall_bracket is not None:
        print(f"         citable bracket {family.wall_bracket}  "
              f"root {'INSIDE' if inside else 'OUTSIDE'}")
    print(f"         final Brent bracket {root.bracket}")
    cand = root.shot.candidate
    print(f"         candidate: {len(cand)} points, "
          f"from ({cand[0][0]:.6f}, {cand[0][1]:.6f}) "
          f"to ({cand[-1][0]:.6f}, {cand[-1][1]:.6f}); "
          f"gap at level {root.shot.level:.6f}: "
          f"{abs(root.shot.unstable_crossing[1]-root.shot.stable_crossing[1]):.3e} in b")
    print("         history:")
    for lam, delta in root.history:
        print(f"           {lam!r:24s} {delta:+.6e}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("families", nargs="*")
    parser.add_argument("--ds", type=float, default=2e-3)
    args = parser.parse_args(argv)
    names = args.families or list(zoo.wall_family_names())
    for name in names:
        probe(name, args.ds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
