#!/usr/bin/env python3
"""Probe a one-parameter recurrence for symmetric small-integer B roots.

    python scripts/probe_symmetric_escape_recurrence.py --max-degree 23

The recurrence is experimental.  The exact Sturm count verifies each row
that is printed, but existence of a suitable integer at every degree and
full portrait certification are separate questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spong import model, sturm  # noqa: E402


def next_row(previous, current, c):
    """H_m=(1+t)H_(m-1)+c*t*H_(m-2), ascending coefficients."""
    result = [0] * (len(current) + 1)
    for j, value in enumerate(current):
        result[j] += value
        result[j + 1] += value
    for j, value in enumerate(previous):
        result[j + 1] += c * value
    return tuple(result)


def b_root_count(degree, row):
    f = tuple(-3 if j % 2 else 0 for j in range(degree + 1))
    g = tuple((-1) ** ((j - 1) // 2 + 1) * row[(j - 1) // 2]
              if j % 2 else -3 for j in range(degree + 1))
    m = model.build(f, g, model.moments_normal01(2 * degree + 1))
    return sturm.count_roots(m.beta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=23)
    parser.add_argument("--max-c", type=int, default=1000)
    args = parser.parse_args()
    if args.max_degree < 7 or args.max_degree % 2 != 1:
        parser.error("--max-degree must be odd and at least 7")
    if args.max_c < 0:
        parser.error("--max-c must be nonnegative")
    previous, current = (1, 4), (1, 4, 1)  # original d=3; curated d=5
    print("degree 3: H=(1, 4) (original case)")
    print("degree 5: H=(1, 4, 1) (curated case)")
    for degree in range(7, args.max_degree + 1, 2):
        for c in range(args.max_c + 1):
            candidate = next_row(previous, current, c)
            if b_root_count(degree, candidate) == degree:
                print(f"degree {degree}: c={c}, H={candidate}", flush=True)
                previous, current = current, candidate
                break
        else:
            raise RuntimeError(
                f"no degree-{degree} row found with c <= {args.max_c}")


if __name__ == "__main__":
    main()
