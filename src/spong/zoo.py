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
    wall_parameter=2.17770956082838,
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
    wall_bracket=(2.1777095608283794, 2.1777095608283807),
    bracket_protocol=(
        "Landing-fate bisection with spong's own tracer: the watched branch "
        "captures (1.168364, 0.966807) below the wall and "
        "(3.153984, -0.015763) above it, a discrete property needing no "
        "separation function.  Bisected to the binary64 floor at IRK-GL4 and "
        "IRK-GL6 independently; both return the identical one-ulp bracket "
        "quoted here (23 portraits each, scripts/wall_bisect.py).  IRK-GL8 "
        "returns a bracket equally sharp but offset +3.0e-12 -- at this "
        "chord its truncation error is already below the evaluation floor, "
        "so the extra stages contribute rounding rather than accuracy, and "
        "the GL8 value is recorded as the calibration of where order stops "
        "helping, not as a competing answer.  wall_shoot.py's Brent root "
        "(2.1777095613653787) inherits the same effect: its _level_crossing "
        "steps at a hard-coded order 8.  Numerical-oracle grade, not a "
        "signed-shooting certificate.\n\n"
        "SUPERSEDES a bracket (2.177709563952666, 2.1777095639570216) whose "
        "protocol cited scipy's Radau and DOP853.  Neither is anadromic and "
        "scipy is not a dependency of this project, so that protocol was "
        "unreproducible here; measured 2026-09-07, both of its endpoints AND "
        "the wall_parameter it carried give the same landing under the "
        "current tracer, so it did not straddle the flip at all."
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


@dataclass(frozen=True)
class SignedMeasureCase:
    """A model whose moment functional is INDEFINITE: a signed measure.

    Deliberately outside ``CASES``, like ``WallFamily``, because ``ZooCase``
    carries ``moment_dist`` as a two-valued string and every consumer
    branches on it with an ``else`` meaning normal01; a third value there
    would silently mis-build this case everywhere.

    WHY THE REGIME EXISTS.  Any finite real sequence is the moment sequence
    of some signed measure, so an indefinite Hankel does not make the SPONG
    problem meaningless -- it makes it least squares in a KREIN space.  The
    form has a signature, squared lengths can be negative, and C = <f,f> can
    be negative.  Signed quadrature is ordinary numerical analysis: closed
    Newton-Cotes has negative weights from order 9.

    WHAT STILL HOLDS, and why this case is a regression rather than a
    curiosity: none of the backbone theory used L >= 0.  It used H11 = 2A,
    and A(b) > 0 is enforced exactly, so the Hessian is never negative
    definite and there are STILL no local maxima in the plane.  Critical
    points still lie on the backbone, det H = 2A u'' still classifies them,
    u'' still alternates, and L is still bounded below because u -> u_inf.
    Only the reading of L as an approximation error is lost.

    The case is stored as exact rational atoms and weights -- the signed
    quadrature rule itself -- rather than as a moment vector, so its
    provenance is inspectable and mu is rebuilt exactly at any order.
    Coefficients are decimal strings parsed by ``Fraction``.
    """

    name: str
    f: tuple[str, ...]
    g: tuple[str, ...]
    atoms: tuple[str, ...]
    weights: tuple[str, ...]
    hankel_signature: tuple[int, int]     # (positive, negative) EXACT
    description: str

    def moments(self, count: int):
        """mu_0..mu_{count-1} of the signed measure, exactly, mu_0 = 1."""
        from fractions import Fraction
        atoms = [Fraction(x) for x in self.atoms]
        weights = [Fraction(w) for w in self.weights]
        raw = [sum((w * x**k for x, w in zip(atoms, weights)), Fraction(0))
               for k in range(count)]
        if raw[0] == 0:
            raise ValueError(f"{self.name}: total mass is zero")
        return tuple(value / raw[0] for value in raw)

    def build(self):
        """The Model, with exact rational coefficients and moments."""
        from fractions import Fraction
        from . import model as _model
        f = [Fraction(x) for x in self.f]
        g = [Fraction(x) for x in self.g]
        need = 2*max(len(f), len(g)) - 1
        return _model.build(f, g, self.moments(need))


KREIN_UNIT_TARGET = SignedMeasureCase(
    name="krein-unit-target-d4",
    f=("1", "0"),
    g=("-3", "3", "-2", "3", "2"),
    atoms=("2", "1", "-1/2", "-2", "3"),
    weights=("4", "4", "-5/3", "2/3", "1"),
    hankel_signature=(4, 1),
    description=(
        "Five-atom signed quadrature whose Hankel matrix has signature "
        "(4+, 1-) -- one genuinely negative direction, certified exactly by "
        "Jacobi's rule on the leading principal minors, no eigenvalues and "
        "no tolerance.  A(b) > 0 nonetheless holds for all real b: the "
        "negative cone misses the Veronese curve (g_0, g_1 b, ..., g_d b^d) "
        "that the model actually evaluates the form on, which is the whole "
        "content of psi_positive here.  Morse, 2 saddles and 2 minima, and "
        "resolution.resolve returns a certified portrait.\n\n"
        "f = (1, 0) gives C = <f,f> = 1 exactly, so this case does NOT "
        "exhibit a negative target norm; it isolates the indefiniteness of "
        "the ambient form from that separate effect.  Found by "
        "scripts/signed_measure.py, which generates the regime at large."
    ),
)


SIGNED_MEASURE_CASES = {
    KREIN_UNIT_TARGET.name: KREIN_UNIT_TARGET,
}


def signed_measure_names() -> tuple[str, ...]:
    return tuple(sorted(SIGNED_MEASURE_CASES))


def get_signed_measure(name: str) -> SignedMeasureCase:
    return SIGNED_MEASURE_CASES[name]


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


@dataclass(frozen=True)
class NumericalConnectionCase:
    """Exact rational model near a numerically located connection intersection.

    Kept outside CASES: neither the wall parameter nor its topology is
    certified. The custom positive measure must never default to normal01.
    """
    name: str = "two-independent-connections"
    parameter: str = "1.307179719334000599261222676631751034570168532337215797867979"
    connection_pairs: tuple = ((0.4729142401695701, 1.1142465931034908, 1),
                               (-0.4729142401695701, -1.1142465931034908, -1))
    moment_dist: str = "absolute_x"
    default_view: tuple = (-2.5, 1.5, -1.4, 1.4)
    description: str = (
        "Two simultaneous numerical saddle-connection candidates, related by "
        "b reflection, with independent splitting under Lambda and the odd "
        "deformation G(z)+eta*z. Positive measure |x| dx on [-1,1]. "
        "Exact Morse skeleton; connection existence is not certified. "
        "Inspector shows the full portrait and two blue candidate connections at the stored Lambda.")
    base_f: tuple = (
        "-223518530295946063188153523711403413281097319/285478393996286056500989623222688818581858816",
        "1", "1", "0", "1", "0", "1", "0", "1", "0", "1")
    base_g: tuple = ("460239/1000000", "0", "-1774853/1000000", "0",
                     "-426253/500000", "0", "-117097/1000000", "0", "-524251/500000")

    def coefficients(self, parameter=None):
        from decimal import Decimal, localcontext
        from fractions import Fraction
        lam = Decimal(self.parameter if parameter is None else str(parameter))
        if not lam.is_finite() or not Decimal("0.1") <= lam <= Decimal("10"):
            raise ValueError("candidate rheostat Lambda must lie in [0.1, 10]")
        with localcontext() as ctx:
            ctx.prec = 80
            scale = Fraction(lam.sqrt())
        return (tuple(Fraction(x) / scale for x in self.base_f),
                tuple(Fraction(x) * scale for x in self.base_g))

    @property
    def f(self):
        return self.coefficients()[0]

    @property
    def g(self):
        return self.coefficients()[1]

    @staticmethod
    def moments(count):
        from .model import moments_absolute_x
        return moments_absolute_x(count)

    def build(self, parameter=None):
        from . import model
        f, g = self.coefficients(parameter)
        return model.build(f, g, self.moments(2 * max(len(f), len(g)) - 1))

    def trace_pairs(self, m, ds=0.001, parameter=None):
        """Fresh native GMP germs and GL8 shots for this rheostat member.

        Launch points are recomputed for Lambda. Each pair has its own
        intermediate loss. No mirroring, snapping, or topology verdict.
        """
        from decimal import Decimal, localcontext
        from . import wall_shoot
        lam = Decimal(self.parameter if parameter is None else str(parameter))
        result = []
        for side, sign, source_b, target_b, continuation in _connection_contexts(self):
            germs = continuation.launch(lam)
            with localcontext() as ctx:
                ctx.prec = 80
                level = float(continuation.level / lam)
                starts = [(float(c.point[0] / lam), float(c.point[1]))
                          for c in (germs.unstable, germs.stable)]
            paths = [wall_shoot._level_crossing(
                m, m._native_kernel, *start, bool(k), level, ds, 10000)
                for k, start in enumerate(starts)]
            if any(path is None for path in paths):
                raise RuntimeError("candidate trace did not reach its loss section")
            result.append(dict(side=side, sign=sign, level=level,
                               source_b=source_b, target_b=target_b,
                               unstable=paths[0], stable=paths[1]))
        return result


# Thread-local: native continuation owns a mutable GMP arena. The threaded
# inspector must never run two launch operations on the same context.
from threading import local as _thread_local
_connection_local = _thread_local()


def _connection_contexts(case):
    from fractions import Fraction
    from . import model, sturm
    from .wall_native import NativeSectionContinuation
    cache = getattr(_connection_local, "cache", None)
    if cache is None:
        cache = _connection_local.cache = {}
    if case not in cache:
        m = model.build(tuple(map(Fraction, case.base_f)),
                        tuple(map(Fraction, case.base_g)),
                        case.moments(2 * max(len(case.base_f), len(case.base_g)) - 1))
        e = sturm.enumerate_critical_points(m)
        # Track named saddle pairs, not inventory offsets: stronger
        # deformations can introduce unrelated far-field critical points.
        saddles = [(i, point) for i, point in enumerate(e.points)
                   if point.kind == "saddle"]
        contexts = []
        for sb, tb, sign in case.connection_pairs:
            si, source = min(saddles, key=lambda item: abs(item[1].b - sb))
            ti, target = min(saddles, key=lambda item: abs(item[1].b - tb))
            if si == ti:
                raise ValueError("tracked connection saddles are no longer distinct")
            continuation = NativeSectionContinuation(m, source_index=si,
                target_index=ti, source_direction=sign, target_direction=-sign)
            contexts.append(("right" if sign > 0 else "left", sign,
                             source.b, target.b, continuation))
        cache[case] = tuple(contexts)
    return cache[case]


NUMERICAL_CONNECTION_CASES = {
    "two-independent-connections": NumericalConnectionCase(),
    "two-asymmetric-connections": NumericalConnectionCase(
        name="two-asymmetric-connections",
        parameter="1.307164649460474225609172398694194647622801706460234661528546",
        base_g=("460239/1000000", "-19544449395600534/1000000000000000000",
                "-1774853/1000000", "1/20", "-426253/500000", "0",
                "-117097/1000000", "0", "-524251/500000"),
        description=("Two numerical saddle connections without b-reflection symmetry. "
            "G has added eta*z + 0.05*z^3, eta=-0.019544449395600534. "
            "Lambda and eta were tuned independently. Positive measure |x| dx. "
            "Exact Morse skeleton; numerical connection evidence only. "
            "Inspector shows the full portrait at the selected Lambda.")),
    "two-strongly-asymmetric-connections": NumericalConnectionCase(
        name="two-strongly-asymmetric-connections",
        parameter="1.3136390836357023026894204981250735694446259302946882157339205",
        default_view=(-3.1, 1.65, -1.55, 1.35),
        connection_pairs=((0.4429110973152095, 1.0907299677754436, 1),
                          (-0.4995110906681631, -1.127018123990298, -1)),
        base_g=("460239/1000000", "-283652549200640338/1000000000000000000",
                "-1774853/1000000", "3/4", "-426253/500000", "0",
                "-117097/1000000", "0", "-524251/500000"),
        description=("Strongly asymmetric numerical double connection: "
            "G has added eta*z + 0.75*z^3, fifteen times the earlier cubic term. "
            "Four minima and five saddles, including an additional negative-b pair. "
            "The local view emphasizes both connections; fit to skeleton shows "
            "the distant saddle. Exact Morse skeleton; connection existence "
            "remains numerical evidence, not a certificate.")),
}


def numerical_connection_names():
    return tuple(NUMERICAL_CONNECTION_CASES)


def get_numerical_connection(name):
    return NUMERICAL_CONNECTION_CASES[name]
