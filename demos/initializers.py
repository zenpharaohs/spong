"""Deterministic initialization designs for optimizer-overlay experiments."""

from __future__ import annotations

import numpy as np


def _radical_inverse(index, base):
    value = 0.0
    factor = 1.0/base
    while index:
        index, digit = divmod(index, base)
        value += digit*factor
        factor /= base
    return value


def low_discrepancy(n, box, skip=17):
    """Two-dimensional Halton design in bases 2 and 3."""
    unit = np.asarray([
        (_radical_inverse(k, 2), _radical_inverse(k, 3))
        for k in range(skip, skip+n)
    ])
    a0, a1, b0, b1 = map(float, box)
    return np.column_stack((
        a0+(a1-a0)*unit[:, 0],
        b0+(b1-b0)*unit[:, 1],
    ))


def blue_noise(n, box, seed=0, candidates=64):
    """Deterministic best-candidate blue-noise design.

    At each insertion, draw several candidates and retain the one farthest
    from the existing design in box-normalized coordinates.  For the intended
    n≈100 this is simpler and more reproducible than boundary-sensitive
    Poisson-disc rejection.
    """
    rng = np.random.default_rng(seed)
    unit = [rng.random(2)]
    while len(unit) < n:
        proposed = rng.random((candidates, 2))
        existing = np.asarray(unit)
        distance2 = np.sum(
            (proposed[:, None, :]-existing[None, :, :])**2, axis=2)
        unit.append(proposed[int(np.argmax(np.min(distance2, axis=1)))])
    unit = np.asarray(unit)
    a0, a1, b0, b1 = map(float, box)
    return np.column_stack((
        a0+(a1-a0)*unit[:, 0],
        b0+(b1-b0)*unit[:, 1],
    ))
