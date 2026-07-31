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
    expected_connections: tuple[tuple[float, float], ...] = ()


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


TRICKY_D11 = ZooCase(
    name="tricky-d11",
    f=(1.12873645202828,
       -0.289963040800028,
       1.26155071814115,
       0.475424811707271,
       1.17411675149371,
       0.126947068043646,
       -0.656815928948082,
       -1.48139907157878,
       0.155488995903894,
       0.818551368521001,
       -0.292588130834394,
       -0.540786416488526),
    g=(1.12873645202828,
       -0.289963040800028,
       1.26155071814115,
       0.475424811707271,
       1.17411675149371,
       0.126947068043646,
       -0.656815928948082,
       -1.48139907157878,
       0.155488995903894,
       0.818551368521001,
       -0.292588130834394,
       -0.540786416488526),
    moment_dist="uniform01",
    default_view=(-1.5, 2.5, -4.0, 3.0),
    expected_connections=((-2.738230515199397, -0.7895860210707522),),
    description=(
        "Canonical degree-11 hard case with saddle stiffness ratio about "
        "8.5e8.  Generic adaptive tracers exhaust their budgets or follow "
        "the nearby backbone; the Hadamard graph transform resolves the "
        "unstable branch from b=-2.738230... to b=-0.789586...."
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


NONNEAREST_ATTACHMENT = ZooCase(
    name="nonnearest-attachment",
    seed=1802198452,
    f=(1.256187626797893,
       0.9685879451467192,
       -1.060859505649057),
    g=(0.20863543789521677,
       0.8523899399870873,
       0.005344187300176611,
       0.29270298657261434,
       -0.3638988208301866,
       -0.827925335821442),
    moment_dist="uniform01",
    default_view=None,
    expected_connections=((-0.4770682827686173, 0.9668071440250788),),
    description=(
        "Counterexample to nearest-minimum attachment.  The positive-b "
        "unstable branch from the saddle b=-0.477068... terminates at the "
        "nonadjacent minimum b=0.966807... because a stable separatrix "
        "crosses the backbone away from a critical point."
    ),
)


MINIMAL_QUARTET = ZooCase(
    name="minimal-quartet",
    f=(48.45917507921044,
       -256.81856167890794,
       246.60346150619446),
    g=(0.3263637061220092,
       -1.9874116808556541,
       1.212138011885117),
    moment_dist="uniform01",
    default_view=(-1.0, 42.0, -1.1, 2.3),
    expected_connections=((-0.6247727737374041, 1.8472900358594468),),
    description=(
        "Conjecturally minimal wall-capable portrait (theorems.md Theorem 8): "
        "four critical points S m S m at degree 2, B positive definite so no "
        "B-saddles.  Criticals b = -0.624773 (high saddle), 0.318562 (m1), "
        "0.639595 (low saddle S'), 1.847290 (m2).  At Lambda = 1 (as stored) "
        "the +b branch of the high saddle skips m1 and S', landing at m2 -- "
        "the minimal nonadjacent attachment.  Under the Lambda-rheostat "
        "(f/sqrt(L), sqrt(L)g) the landing flips at a wall "
        "Lambda* ~= 7.651823524762; the wall's saddle-connection type "
        "(S -> S' rather than rim) is empirical -- hug-scaling evidence, "
        "Theorem 8.5 -- and wall coordinates at this precision are "
        "launch-protocol-sensitive (Theorem 8.4)."
    ),
)


CASES = {
    QUADRATIC_STIFF.name: QUADRATIC_STIFF,
    MINIMAL_QUARTET.name: MINIMAL_QUARTET,
    TRICKY_D11.name: TRICKY_D11,
    LINEAR_TARGET_D17_THRASH.name: LINEAR_TARGET_D17_THRASH,
    NONNEAREST_ATTACHMENT.name: NONNEAREST_ATTACHMENT,
}


def names() -> tuple[str, ...]:
    return tuple(sorted(CASES))


def get(name: str) -> ZooCase:
    return CASES[name]
