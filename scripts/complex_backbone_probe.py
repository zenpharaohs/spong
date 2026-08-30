#!/usr/bin/env python3
"""Cost and completeness of the validated complex backbone, per zoo case.

    python scripts/complex_backbone_probe.py                # all zoo cases
    python scripts/complex_backbone_probe.py tricky-d11 linear-target-d17-thrash

`portrait.build_ledger` now calls `complex_structure.certify_backbone` on
every portrait.  This probe measures that call in isolation (cold cache) so
the decision to keep it on the default path, or to make it opt-in like
`materialize_validated_launches`, rests on numbers rather than on degree
arithmetic.  It also reports whether each of the four divisors was fully
isolated, since an incomplete divisor silently downgrades the ledger to
"partial".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

try:
    from spong import complex_structure, model, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import complex_structure, model, zoo


def _moments(name: str, n: int):
    if name == "normal01":
        return model.moments_normal01(n)
    return model.moments_uniform01(n)


def probe(name: str) -> None:
    z = zoo.get(name)
    f, g = list(z.f), list(z.g)
    m = model.build(f, g, _moments(z.moment_dist, 2*max(len(f), len(g))-1))
    complex_structure.certify_polynomial_roots.cache_clear()
    t0 = time.perf_counter()
    certificate = complex_structure.certify_backbone(m)
    elapsed = time.perf_counter()-t0
    divisors = (
        ("transverse", certificate.transverse),
        ("valley", certificate.valley_denominator),
        ("backbone", certificate.denominator),
        ("critical", certificate.critical),
    )
    print(f"{name:32s} {elapsed:8.3f}s  "
          f"{'complete' if certificate.complete else 'PARTIAL '}  "
          + "  ".join(
              f"{label}:d{divisor.degree}"
              f"{'' if divisor.complete else '!'}"
              f"{'' if not divisor.reason else '(' + divisor.reason + ')'}"
              for label, divisor in divisors))


def main(argv: list[str]) -> int:
    names = argv or list(zoo.names())
    for name in names:
        probe(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
