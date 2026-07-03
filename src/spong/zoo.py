"""Named phase-portrait zoo cases.

These are not random tests; they are memorable counterexamples and regression
fixtures that make specific mathematical or numerical points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZooCase:
    name: str
    f: tuple[float, ...]
    g: tuple[float, ...]
    moment_dist: str
    description: str
    seed: int | None = None
    default_view: tuple[float, float, float, float] | None = None


QUADRATIC_STIFF = ZooCase(
    name="quadratic-stiff",
    seed=2735729614,
    f=(-0.27126925828072923,
       -0.7363598557165663,
       0.7989868625933855),
    g=(-0.27126925828072923,
       -0.7363598557165663,
       0.7989868625933855),
    moment_dist="uniform01",
    default_view=None,
    description=(
        "Degree-2 f=g case with 3 minima and 3 saddles.  The bounded "
        "unstable branch from the lowest saddle to the far lower finite "
        "minimum is stiff and exercises the Hadamard handoff."
    ),
)


LINEAR_TARGET_D17_THRASH = ZooCase(
    name="linear-target-d17-thrash",
    seed=1158725111,
    f=(-0.9514652373581963,
       -1.7945943562259494),
    g=(-0.2302392536989173,
       2.4650232077321212,
       -1.3228162550152864,
       3.120814010750335,
       -1.4209087818998132,
       -2.0452998402729996,
       -0.10981879692405587,
       -1.5932352722307273,
       -0.6365486378841433,
       0.19535772631955794,
       0.26391606930746814,
       -0.7697423065439618,
       -0.7428291736196287,
       -0.21651008677854963,
       1.7960186839315102,
       0.4286765825415307,
       0.8227526313695055,
       0.15929379398579482),
    moment_dist="uniform01",
    default_view=None,
    description=(
        "Linear target data fit by a degree-17 model.  The portrait has a "
        "rich Morse skeleton with many local minima; one nearly horizontal "
        "finite unstable branch is a regression for chord-based branch "
        "spacing."
    ),
)


CASES = {
    QUADRATIC_STIFF.name: QUADRATIC_STIFF,
    LINEAR_TARGET_D17_THRASH.name: LINEAR_TARGET_D17_THRASH,
}


def names() -> tuple[str, ...]:
    return tuple(sorted(CASES))


def get(name: str) -> ZooCase:
    return CASES[name]
