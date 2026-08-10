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


@dataclass(frozen=True)
class WallFamily:
    """A parameterized non-Morse-Smale wall and two ordinary chambers.

    Wall families are deliberately separate from ``CASES``: the ordinary
    portrait command promises one certified Morse-Smale portrait, whereas a
    wall member has a saddle connection and must be depicted by a dedicated
    limiting construction.

    ``wall_parameter`` is a representative center for display and member
    materialization only.  The citable object is ``wall_bracket``: an
    interval whose endpoints have verified OPPOSITE landing fates under the
    protocol recorded in ``bracket_protocol``.  Wall coordinates tighter
    than the bracket are launch-protocol-dependent (theorems.md Theorem 8.4)
    and must not be quoted without their protocol.
    """

    name: str
    base_case: str
    parameter_name: str
    below_parameter: float
    wall_parameter: float
    above_parameter: float
    source_b: float
    target_b: float
    unstable_direction: int
    default_view: tuple[float, float, float, float]
    description: str
    wall_bracket: tuple[float, float] | None = None
    bracket_protocol: str = ""

    def __post_init__(self):
        if self.wall_bracket is None:
            return
        lo, hi = self.wall_bracket
        if not (self.below_parameter < lo < self.wall_parameter
                < hi < self.above_parameter):
            raise ValueError(
                "wall_bracket must satisfy below < lo < wall < hi < above")
        if not self.bracket_protocol:
            raise ValueError(
                "a wall_bracket requires a nonempty bracket_protocol")


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


NONNEAREST_SADDLE_CONNECTION = WallFamily(
    name="nonnearest-saddle-connection",
    base_case=NONNEAREST_ATTACHMENT.name,
    parameter_name="Lambda",
    below_parameter=2.0,
    wall_parameter=2.177709563954844,
    above_parameter=4.0,
    source_b=-0.4770682827686173,
    target_b=0.6402740918269282,
    unstable_direction=1,
    default_view=(-2.5, 4.5, -1.8, 1.8),
    description=(
        "Three-state Lambda-rheostat family through the B-to-N saddle "
        "connection forced in Theorem 4.  Lambda=2 and Lambda=4 lie in the "
        "two Morse-Smale chambers and have clean, robustly different branch "
        "landings; at the wall the positive-b unstable branch of the B "
        "saddle connects to the N saddle.  Lambda* is quoted through its "
        "bracket; digits inside the bracket are protocol-dependent.  The "
        "center portrait is a geometric wall limit, not an ordinary "
        "certified portrait."
    ),
    wall_bracket=(2.177709563952666, 2.1777095639570216),
    bracket_protocol=(
        "Landing fates at both endpoints verified by two independent "
        "integrators (Radau rtol 1e-12 atol 1e-14; DOP853 rtol 1e-13 "
        "atol 1e-15), jet-eigenvector launch offset 1e-8 from the "
        "Newton-polished saddle at b=-0.4770682827686173: far minimum at "
        "the lower endpoint, near minimum at the upper endpoint, all four "
        "runs agreeing.  Numerical-oracle grade, not a signed-shooting "
        "certificate."
    ),
)


DEAD_NEURON_FAR_SADDLE = ZooCase(
    name="dead-neuron-far-saddle-d3",
    f=(-0.866287, 0.481148, -0.507756, 0.839429),
    g=(0.803763, 0.88267, -0.140028, 0.896359),
    moment_dist="uniform01",
    # The far saddle at b = 50.729 is the point of this case, so the default
    # view has to contain it -- framing on the near cluster alone hides what
    # the entry is for.
    default_view=(-1.8, 0.8, -8.0, 56.0),
    description=(
        "Cheapest fixture for the SHALLOW LAUNCH: a degree-3 case whose far "
        "saddle is a dead neuron.  Two of its four critical points sit "
        "essentially on the a=0 axis -- a* is 5.3e-3 at the b=-4.908 "
        "minimum and -5e-6 at the b=50.729 saddle, with u'' of 4e-4 and ~0 "
        "-- so the backbone is nearly flat out there and A(b) ~ 1e10.  The "
        "materialized stub at that saddle cannot condition a handoff to the "
        "global field, and trace_unstable used to refuse outright: both "
        "unstable branches returned abort_conditioning_handoff after 514 "
        "vertices and the portrait went to branch_abort.  The manifold is "
        "perfectly well behaved there -- it is the backbone to machine "
        "precision -- so the Hadamard fixed point owns it, and routing the "
        "launch to that second owner certifies the case.  Keep it: it is "
        "seconds to run and it is the only fixture that exercises that "
        "route.  Found by random search in the interactive explorer.  Note "
        "the near-coincident levels, 0.336849 at b=-4.908 against 0.339436 "
        "at b=50.729, under 1% apart against a 0.39 range."
    ),
)


NEAR_SLIDE_D2 = ZooCase(
    name="near-slide-d2",
    f=(-0.891598, -0.077792, 0.101212),
    g=(0.199189, 0.515701, -0.228274),
    moment_dist="uniform01",
    default_view=None,
    expected_connections=((-0.651232686, 1.953667699),),
    description=(
        "Degree-2/degree-2 nonadjacent attachment sitting CLOSE TO ITS "
        "SLIDE.  Six critical points alternating m S m S m S along the "
        "backbone: minima b = -5.120861, -0.009938, 1.953668 and saddles "
        "b = -0.651233, 0.984296, 4.053796.  The two outer saddles are "
        "B-saddles -- roots of B, so a* = 0 and u = C exactly at both, and "
        "the level set at c = C contains the whole line a = 0 together "
        "with the curve a = 2a*(b).  The +b branch of the B-saddle "
        "b = -0.651233 skips the adjacent minimum b = -0.009938 and lands "
        "at b = 1.953668, passing the N-saddle b = 0.984296 at a distance "
        "of about 0.11: the same B-to-N slide as the nonnearest-saddle-"
        "connection family, at degree 2 in BOTH f and g.  What earns it a "
        "slot is the margin.  Under the Lambda rheostat (f/sqrt(L), "
        "sqrt(L)g) -- which leaves every critical b and the whole backbone "
        "topology fixed, since A -> L*A, B -> B, C -> C/L and a* -> a*/L, "
        "and moves only the planar geometry the flow sees -- the landing "
        "flips at Lambda* in [1.0798784758546, 1.0798784758605].  The "
        "stored case is Lambda = 1, i.e. EIGHT PERCENT from its wall, "
        "against 118% for nonnearest-saddle-connection and 665% for "
        "minimal-quartet.  That makes it the closest-to-wall stored case "
        "and the natural fixture for a reported Morse-Smale margin.  "
        "Bracket grade: endpoints verified to have opposite landings by "
        "Radau (rtol 1e-12) and DOP853 (rtol 1e-13) on an INDEPENDENT "
        "reconstruction of the model, not by spong's own certified "
        "machinery -- reconfirm before promoting this to a WallFamily.  "
        "Found by random search in the interactive explorer."
    ),
)


CASES = {
    QUADRATIC_STIFF.name: QUADRATIC_STIFF,
    NEAR_SLIDE_D2.name: NEAR_SLIDE_D2,
    MINIMAL_QUARTET.name: MINIMAL_QUARTET,
    TRICKY_D11.name: TRICKY_D11,
    LINEAR_TARGET_D17_THRASH.name: LINEAR_TARGET_D17_THRASH,
    NONNEAREST_ATTACHMENT.name: NONNEAREST_ATTACHMENT,
    DEAD_NEURON_FAR_SADDLE.name: DEAD_NEURON_FAR_SADDLE,
}


WALL_FAMILIES = {
    NONNEAREST_SADDLE_CONNECTION.name: NONNEAREST_SADDLE_CONNECTION,
}


def names() -> tuple[str, ...]:
    return tuple(sorted(CASES))


def get(name: str) -> ZooCase:
    return CASES[name]


def wall_family_names() -> tuple[str, ...]:
    return tuple(sorted(WALL_FAMILIES))


def get_wall_family(name: str) -> WallFamily:
    return WALL_FAMILIES[name]


def rheostat_member(family: str | WallFamily, member: str) -> ZooCase:
    """Materialize one Lambda-rheostat member as ordinary ``(f,g,mu)`` data.

    The wall member is returned as coefficient data but remains outside
    ``CASES`` because its geometry is intentionally non-Morse-Smale.
    """
    import math

    wall = get_wall_family(family) if isinstance(family, str) else family
    parameters = {
        "below": wall.below_parameter,
        "wall": wall.wall_parameter,
        "above": wall.above_parameter,
    }
    if member not in parameters:
        raise KeyError(member)
    lam = parameters[member]
    base = get(wall.base_case)
    root = math.sqrt(lam)
    return ZooCase(
        name=f"{wall.name}-{member}",
        f=tuple(value/root for value in base.f),
        g=tuple(root*value for value in base.g),
        moment_dist=base.moment_dist,
        default_view=wall.default_view,
        description=(
            f"{member.capitalize()} member of {wall.name} at Lambda={lam:.16g}. "
            + wall.description),
    )
