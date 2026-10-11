#!/usr/bin/env python3
"""Rank small-integer normal degree-5 ripple cases.

    python scripts/search_normal_flat_d5.py --height 4

Fix f=x+x^3+x^5 and even g coefficients at -3. Search the negative,
positive, negative odd-coefficient sign convention; the opposite convention
is its reflection across a=0. The root prefilter is exact. Critical-point
counts come from exact Sturm enumeration; loss gaps are numerical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spong import model, sturm  # noqa: E402


def search(height: int):
    f = (0, 1, 0, 1, 0, 1)
    moments = model.moments_normal01(11)
    rows = []
    for g1 in range(-height, 0):
        for g3 in range(1, height + 1):
            for g5 in range(-height, 0):
                # B(b)/b = 19*g1 + 123*g3*t + 1065*g5*t^2,
                # t=b^2. Two positive, distinct roots give five B roots.
                if (123 * g3) ** 2 <= 4 * 19 * 1065 * g1 * g5:
                    continue
                g = (-3, g1, -3, g3, -3, g5)
                m = model.build(f, g, moments)
                points = sturm.enumerate_critical_points(m)
                if not points.morse or not points.psi_positive:
                    continue
                if (len(points.saddles), len(points.minima)) != (5, 4):
                    continue
                gap = max((float(m.C) - m.u(q.b)) / float(m.C)
                          for q in points.minima)
                rows.append((gap, (g1, g3, g5)))
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=4)
    args = parser.parse_args()
    if args.height < 1:
        parser.error("--height must be positive")
    for gap, odd in search(args.height):
        print(f"g odd={odd}, max relative gap={100 * gap:.6f}%")


if __name__ == "__main__":
    main()
