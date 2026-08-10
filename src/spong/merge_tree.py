"""Exact merge tree of the backbone loss -- the level-set topology engine.

WHY THIS EXISTS

The audit used to answer "where does this branch go?" per branch, per index,
against a level MANUFACTURED from a floating orbit point.  Manufacturing the
level is what forced the strictness slack (scale/2**48), and on a flat valley
that slack exceeded the whole window between the orbit's loss and the
backbone's ceiling -- the measured cause of every unstable_endpoint_unresolved
refusal.  Here the levels come first, as RATIONALS chosen once per model, and
the orbit is merely COMPARED against them.  No slack, no ladder, and the
comparison is exact arithmetic on dyadic data.

THE STRUCTURE IT EXPLOITS

L = u(b) + A(b)(a - a*(b))^2 with A > 0 depends on a only through
(a - a*)^2, so a -> 2a*(b) - a is an exact symmetry of L: the backbone is the
AXIS of every level curve, not a curve that merely passes nearby.  Hence

  * every vertical line meets {L = c} in at most two points, which coincide
    only where u(b) = c, so a bounded level component meets the backbone
    exactly TWICE, at its two interval endpoints;
  * components of {L < c} are TUBES over components of {u < c}, so the planar
    component lattice is the merge tree of the one-variable u;
  * every critical point lies on the backbone (d_a L = 2A(a - a*)), so the
    critical set enclosed by a component is a backbone-CONTIGUOUS run -- no
    non-contiguous enclosure, and no nesting of same-level components (they
    are tubes over disjoint b-intervals, hence disjoint vertical strips);
  * L_aa = 2A > 0 forbids local maxima, so a bounded component is a disk and
    Euler gives #minima = #saddles + 1 EXACTLY.

Since u < c iff (C - c)A - B^2 < 0 (A > 0), every question above is a
question about the sign of ONE polynomial of degree max(deg A, 2 deg B)
~ 2 deg g -- against 4 deg A for the far-field funnel.

FLOATS PROPOSE, EXACT DISPOSES

Separating levels are GUESSED from the cosmetic float critical values and
then VERIFIED by exact sign tests of (C - c)A - B^2 at each critical point's
isolating interval.  A verified level is a certificate of the ordering it
induces, however it was guessed.  Critical values are therefore never
compared to each other directly -- which matters, because coincident
critical values are GENERIC here: u = C - B^2/A <= C with equality exactly at
the roots of B, so every B-saddle sits at the single level C, the global
maximum of the backbone loss and the roof of the merge tree.  Values that no
level separates are reported as one class rather than guessed apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from . import _poly as P
from . import sturm


# --------------------------------------------------------------------- #
# level polynomial and exact sign tests                                  #
# --------------------------------------------------------------------- #

def level_polynomial(m, c: Fraction):
    """(C - c)A - B^2, whose sign is the sign of u - c since A > 0."""
    return P.sub(P.scale(m.alpha, Fraction(m.C) - c), P.mul(m.beta, m.beta))


def _root_poly(m, point):
    """The polynomial whose root this critical point isolates."""
    if point.source == "N":
        return m.N
    if point.source == "B":
        return m.beta
    return m.critical_reduced


def value_sign(m, point, c: Fraction, budget: int = 64):
    """Exact sign of u(q) - c at the critical point q, or None if undecided.

    None means the isolating interval could not be refined enough to
    certify the sign within ``budget`` steps -- i.e. c is at (or
    indistinguishably near) this critical value.  Never guessed.
    """
    R = level_polynomial(m, c)
    iv = point.interval
    root = _root_poly(m, point)
    for _ in range(budget):
        s = sturm.interval_sign(R, iv)
        if s is not None:
            return s
        if iv.exact:
            v = P.eval_at(R, iv.lo)
            return (v > 0) - (v < 0)
        iv = sturm.refine(
            root, iv, rel=(iv.hi - iv.lo) / (1 + abs(iv.mid)) / 4)
    return None


def _float_value(m, point):
    """Cosmetic u at a critical point -- used ONLY to propose levels."""
    b = Fraction(point.interval.mid)
    A = P.eval_at(m.alpha, b)
    B = P.eval_at(m.beta, b)
    return float(Fraction(m.C) - B * B / A)


# --------------------------------------------------------------------- #
# separating levels                                                      #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class LevelSequence:
    levels: tuple[Fraction, ...]          # ascending, each verified
    classes: tuple[tuple[int, ...], ...]  # critical indices per value class
    unseparated: tuple[tuple[int, ...], ...]   # classes with >1 member


def backbone_level_at_infinity(m):
    """u at b -> +-infinity, exactly rational, or None if u is unbounded below.

    u = C - B^2/A, and the limit is the SAME at both ends because only the
    leading powers survive.  This value is a node of the merge tree even
    though no critical point realizes it: crossing c = u_inf is exactly
    where the outer components stop being bounded.  A level sequence built
    from critical values alone therefore misses the window in which an
    outermost minimum's basin is still enclosed -- which is not a corner
    case but the normal situation for the outermost minimum on each side.
    """
    A, B = m.alpha, m.beta
    dA, dB = P.degree(A), P.degree(B)
    if dA > 2 * dB:
        return Fraction(m.C)
    if dA < 2 * dB:
        return None
    return Fraction(m.C) - B[-1] * B[-1] / A[-1]


def separating_levels(m, e, attempts: int = 40) -> LevelSequence:
    """Rational levels separating the DISTINCT values of the merge tree.

    The values are the critical values AND u(infinity): a level is accepted
    when every critical point's sign against it is decided, and it is kept
    only if it induces a NEW signature -- the sign vector together with its
    side of u_inf.  Signature dedup replaces any attempt to compare
    critical values with each other, so coincident values (every B-saddle
    sits at u = C) simply share a class instead of being guessed apart.
    """
    points = [p for p in e.points if p.kind != "degenerate"]
    u_inf = backbone_level_at_infinity(m)
    marks = sorted({_float_value(m, p) for p in points}
                   | ({float(u_inf)} if u_inf is not None else set()))

    accepted: list[Fraction] = []
    seen: set[tuple] = set()
    for rank in range(len(marks) - 1):
        lo, hi = marks[rank], marks[rank + 1]
        if not hi > lo:
            continue
        # Place the level as HIGH in the gap as can be verified.  Every
        # level in a gap gives the SAME component structure, so this does
        # not change any topology -- but a higher level is crossed EARLIER
        # by a descending orbit, so the certified sublevel suffix starts
        # sooner.  Measured: the midpoint left a 24583-sample branch's
        # suffix starting at 24219, four segments after the contact events
        # it had to discharge, and the portrait refused on 120 contacts
        # that the per-index level of the old ladder had discharged.
        # Verification is what makes any of these sound: a proposed level
        # is accepted only when every critical sign against it is decided
        # and nonzero, so a level too close to the upper critical value is
        # rejected rather than trusted.
        for fraction in (0.99, 0.9, 0.75, 0.5, 0.25):
            c = Fraction(lo + fraction*(hi - lo))
            if u_inf is not None and c == u_inf:
                continue
            signs = [value_sign(m, p, c) for p in points]
            if any(s is None or s == 0 for s in signs):
                continue
            signature = (tuple(signs),
                         None if u_inf is None else c > u_inf)
            if signature not in seen:
                seen.add(signature)
                accepted.append(c)
            break

    levels = tuple(sorted(accepted))
    classes: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(points):
        key = tuple(1 if (value_sign(m, p, c) or 0) > 0 else 0 for c in levels)
        classes.setdefault(key, []).append(i)
    grouped = tuple(tuple(v) for _, v in sorted(
        classes.items(), key=lambda kv: sum(kv[0]), reverse=True))
    return LevelSequence(
        levels=levels,
        classes=grouped,
        unseparated=tuple(g for g in grouped if len(g) > 1))


# --------------------------------------------------------------------- #
# components of a sublevel set                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Component:
    level: Fraction
    index: int
    gap: int                            # roots of the level poly to its left
    lo: Fraction | None                 # None = unbounded to -infinity
    hi: Fraction | None
    sample: Fraction                    # a rational b strictly inside
    minima: tuple[int, ...]             # indices into enumeration.points
    saddles: tuple[int, ...]

    @property
    def bounded(self) -> bool:
        return self.lo is not None and self.hi is not None

    # ``lo``/``hi`` come from the ISOLATING intervals of the bounding roots
    # (left root's lower bound, right root's upper bound), so [lo, hi] is a
    # certified OUTER enclosure of the component, not its exact span.  It is
    # for reporting and plotting.  Every containment question -- which
    # component holds a point, which component nests in which -- must go
    # through ``gap`` and an exact root count instead, because two
    # enclosures at different levels can overlap in the slack even when the
    # true intervals nest strictly.


def _sign_at_infinity(R, positive: bool) -> int:
    if not R:
        return 0
    s = (R[-1] > 0) - (R[-1] < 0)
    return s if positive or (len(R) - 1) % 2 == 0 else -s


def components_at(m, e, c: Fraction) -> tuple[Component, ...]:
    """The components of {u < c}, as b-intervals with their critical sets.

    Every root of (C - c)A - B^2 is SIMPLE when c is not a critical value
    (a double root means u' = 0 there, i.e. c IS a critical value), so the
    sign alternates across consecutive roots.  That alternation is asserted:
    two consecutive gaps with the same sign means a missed root.
    """
    R = level_polynomial(m, c)
    roots = sorted(sturm.isolate_roots(R), key=lambda iv: (iv.lo, iv.hi))
    edges = [iv.hi if not iv.exact else iv.lo for iv in roots]
    lows = [iv.lo for iv in roots]

    samples: list[Fraction] = []
    if roots:
        samples.append(lows[0] - 1)
        for k in range(len(roots) - 1):
            samples.append((edges[k] + lows[k + 1]) / 2)
        samples.append(edges[-1] + 1)
    else:
        samples.append(Fraction(0))

    signs = []
    for x in samples:
        v = P.eval_at(R, x)
        signs.append((v > 0) - (v < 0))
    if roots:
        if signs[0] != _sign_at_infinity(R, positive=False):
            raise ValueError("level polynomial: sign disagrees with -infinity")
        if signs[-1] != _sign_at_infinity(R, positive=True):
            raise ValueError("level polynomial: sign disagrees with +infinity")
        for k in range(len(signs) - 1):
            if signs[k] == signs[k + 1]:
                raise ValueError(
                    "level polynomial: consecutive gaps share a sign at "
                    f"root {k} -- a root was missed or c is a critical value")

    points = list(e.points)
    # Hoisted: value_sign and the gap index are per critical point, not per
    # component, and both are exact refinement loops.
    inside: dict[int, int] = {}
    for i, p in enumerate(points):
        if p.kind == "degenerate":
            continue
        if value_sign(m, p, c) != -1:
            continue
        inside[i] = _gap_index(m, R, p)

    out: list[Component] = []
    for k, s in enumerate(signs):
        if s >= 0:
            continue
        lo = None if k == 0 else lows[k - 1]
        hi = None if k == len(signs) - 1 else edges[k]
        minima = tuple(i for i, g in inside.items()
                       if g == k and points[i].kind == "min")
        saddles = tuple(i for i, g in inside.items()
                        if g == k and points[i].kind == "saddle")
        out.append(Component(c, len(out), k, lo, hi, samples[k],
                             minima, saddles))
    placed = sum(len(comp.minima) + len(comp.saddles) for comp in out)
    if placed != len(inside):
        raise ValueError(
            "a critical point below the level landed in no component")
    for comp in out:
        if comp.bounded and len(comp.minima) != len(comp.saddles) + 1:
            raise ValueError(
                f"Euler violation in bounded component at level {float(c)}: "
                f"{len(comp.minima)} minima, {len(comp.saddles)} saddles")
    return tuple(out)


def _gap_index(m, R, point, budget: int = 64) -> int:
    """Which gap between the roots of R the critical point lies in.

    The isolating interval must be shrunk with the critical point's OWN
    root polynomial.  Refining it against R instead would bisect toward a
    root of R -- a level crossing near the critical point -- and walk the
    interval off the critical point entirely, after which the root count
    below its lower bound is a count for the wrong location.
    """
    iv = point.interval
    root = _root_poly(m, point)
    for _ in range(budget):
        if iv.exact or sturm.count_roots(R, iv.lo, iv.hi) == 0:
            break
        iv = sturm.refine(
            root, iv, rel=(iv.hi - iv.lo) / (1 + abs(iv.mid)) / 4)
    return sturm.count_roots(R, None, iv.lo)


# --------------------------------------------------------------------- #
# locating a traced point                                                #
# --------------------------------------------------------------------- #

def exact_loss(m, a: float, b: float) -> Fraction:
    """L at a binary64 point -- exactly rational, no slack anywhere."""
    aq, bq = Fraction(a), Fraction(b)
    A = P.eval_at(m.alpha, bq)
    B = P.eval_at(m.beta, bq)
    return Fraction(m.C) - 2 * aq * B + aq * aq * A


def locate(m, e, c: Fraction, a: float, b: float):
    """The component of {L < c} containing the binary64 point, or None.

    Two exact tests and nothing else.  L(p) < c already places the point
    transversally, because L(p) = u(b) + A(a - a*)^2 >= u(b): a point below
    the level is inside the tube over its own b-gap.
    """
    if exact_loss(m, a, b) >= c:
        return None
    R = level_polynomial(m, c)
    bq = Fraction(b)
    if P.eval_at(R, bq) == 0:
        return None                      # exactly on a boundary: undecided
    k = sturm.count_roots(R, None, bq)
    for comp in components_at(m, e, c):
        if comp.gap == k:
            return comp
    return None


# --------------------------------------------------------------------- #
# the tree, and fates                                                    #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class MergeTree:
    sequence: LevelSequence
    levels: tuple[Fraction, ...]
    components: tuple[tuple[Component, ...], ...]   # parallel to levels
    parents: tuple[tuple[int | None, ...], ...]     # index into next level up


def build(m, e) -> MergeTree:
    """The exact merge tree: components at every separating level, nested.

    Cost is one root isolation per level -- (#distinct critical values - 1)
    of them, at degree ~2 deg g -- shared by every branch of the portrait.
    """
    seq = separating_levels(m, e)
    per_level = tuple(components_at(m, e, c) for c in seq.levels)
    parents: list[tuple[int | None, ...]] = []
    for k, comps in enumerate(per_level):
        if k + 1 >= len(per_level):
            parents.append(tuple(None for _ in comps))
            continue
        above = per_level[k + 1]
        R_above = level_polynomial(m, seq.levels[k + 1])
        row: list[int | None] = []
        for comp in comps:
            # Exact: locate the child's interior sample among the PARENT
            # level's roots.  Comparing the enclosures' endpoints instead
            # is wrong -- the isolation slack can invert the comparison
            # even though the true intervals nest.
            if P.eval_at(R_above, comp.sample) == 0:
                raise ValueError(
                    "component sample lies on a higher level's boundary; "
                    "choose a different interior sample")
            kp = sturm.count_roots(R_above, None, comp.sample)
            row.append(next((j for j, up in enumerate(above)
                             if up.gap == kp), None))
        parents.append(tuple(row))
    return MergeTree(seq, seq.levels, per_level, tuple(parents))


def widest_forcing_component(m, e, tree, a: float, b: float):
    """Highest level whose component through the point still forces capture.

    Returns ``(level, component)`` or None.

    ``fate_from_tree`` answers "what is the fate?" and for that the TIGHTEST
    (lowest) containing level is right.  This answers a different question:
    "how early can the orbit's suffix be certified?"  -- and for that the
    WIDEST level is right, because the orbit crosses a higher level sooner.
    Components only grow with level, so scan ascending and keep the last one
    that is bounded, saddle-free, and holds the same single minimum.

    Soundness: at the first sample below that level the orbit is inside some
    component, forward invariance keeps it there, and the terminal sample is
    in this one -- so the whole suffix lies in a bounded region containing
    that minimum and no saddle.

    Using the tightest level here instead was measured to start the
    certified suffix within a few samples of the END of a 24000-sample
    branch, leaving the entire approach undischarged and turning contact
    events into a topology_contact refusal.
    """
    best = None
    for c in tree.levels:
        comp = locate(m, e, c, a, b)
        if comp is None:
            continue
        forcing = (comp.bounded and not comp.saddles
                   and len(comp.minima) == 1)
        if forcing and (best is None or comp.minima == best[1].minima):
            best = (c, comp)
        elif best is not None:
            break
    return best


def fate_from_tree(m, e, tree: MergeTree, a: float, b: float):
    """Candidate fates of the orbit through a binary64 point.

    Walks the levels upward and keeps the LOWEST level whose component
    contains the point: that is the tightest certified candidate set.
    ``forced`` means the fate is decided outright -- one minimum and no
    saddle in a bounded component (capture), or no critical point at all in
    a component with a single open end (escape).
    """
    for k, c in enumerate(tree.levels):
        comp = locate(m, e, c, a, b)
        if comp is None:
            continue
        forced_capture = (comp.bounded and not comp.saddles
                          and len(comp.minima) == 1)
        forced_escape = (not comp.bounded and not comp.saddles
                         and not comp.minima)
        return {
            "level_index": k,
            "level": c,
            "bounded": comp.bounded,
            "b_interval": (comp.lo, comp.hi),
            "minima": comp.minima,
            "saddles": comp.saddles,
            "forced": forced_capture or forced_escape,
            "fate": ("capture" if forced_capture else
                     "escape" if forced_escape else "undecided"),
            "splits_remaining": len(comp.saddles),
        }
    return None


def escape_eligible(m, e, tree: MergeTree) -> tuple[int, ...]:
    """Saddles that could possibly have an unbounded unstable branch.

    A saddle enclosed in a BOUNDED component has both unstable branches
    trapped there by forward invariance, and a bounded component with k
    saddles holds k+1 minima -- so all 2k branch ends land inside.  Escape
    is therefore possible only from saddles that no bounded component
    encloses.  Necessary, not sufficient.
    """
    trapped = set()
    for comps in tree.components:
        for comp in comps:
            if comp.bounded:
                trapped.update(comp.saddles)
    return tuple(i for i, p in enumerate(e.points)
                 if p.kind == "saddle" and i not in trapped)
