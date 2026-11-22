"""Which way can an orbit go?  Gates, the uphill lemma, and the climb test.

STATUS: reporting only.  Nothing here certifies anything yet; it predicts
which attachments are IMPOSSIBLE, and the census checks every prediction
against certified portraits.  See claude/uphill-lemma.md in the Project for
the statement and proof.

THE STRUCTURE

    L = C - 2aB(b) + a^2 A(b),        da/dt = 2A(a* - a),
    db/dt = a(2B' - aA') = -aA'(a - a_n),     a_n = 2B'/A'.

An orbit in {L < C} cannot reach a = 0 (where L = C) and cannot cross a
root of B in b (where u = C), so it keeps the sign of a and stays between
two consecutive roots of B.  Every unstable branch is such an orbit.  On
the backbone db/dt = -u', so the backbone always drifts downhill in u.

GATES.  At a real root of B' (db/dt = -a^2 A') or of A' (db/dt = 2aB')
the sign of db/dt is the same for every such orbit, and it is always the
DOWNHILL direction of u.  No orbit crosses one going uphill.

THE UPHILL LEMMA.  On a stretch of b free of roots of B, N = 2AB' - A'B,
A' and B', the set where b moves uphill in u lies beyond the b-nullcline,
on the side away from the backbone, and there

    |d(a^2)/db| >= 4A/|A'|,

with a moving toward the backbone.  The set is left only through the
nullcline and not re-entered through it.  So an uphill run from b0 to b1
has  W = INT 4A/|A'| db <= |a1^2 - a0^2|.  W is twice the change of the
b-part of the invariant J of stable_tails, whose closed form is used here.

THE CLIMB TEST.  To get from a minimum m over the next saddle s2 an orbit
at level below c needs no gate strictly between them, and

    type 1 (A'u' > 0, uphill set 0 < a < a_n < a*):   W <= a*(s2)^2
    type 2 (A'u' < 0, uphill set a > a_n > a*):       W <= a_max^2 - a*(s2)^2

with a_max = a*(m) + sqrt((c - u(m))/A(m)).  Necessary, not sufficient: a
failed test excludes everything beyond s2 AND the connection to s2; a
passed test decides nothing.  Beyond an outermost minimum W is infinite,
so a backbone end is reachable only past an outermost saddle.

THE RIGOROUS FORM.  Nothing in the test needs the critical points
themselves.  Take ANY rational nodes x_0, ..., x_n inside the stretch, in
climbing order.  An orbit that gets over the saddle crosses them all in one
sojourn in the uphill set, so for every i < j

    type 1:  INT[x_i, x_j] 4A/|A'|  <=  a(x_j)^2 - a(x_i)^2
                                    <=  a_n(x_j)^2 - a_min(x_i)^2
    type 2:  INT[x_i, x_j] 4A/|A'|  <=  a(x_i)^2 - a(x_j)^2
                                    <=  a_max(x_i)^2 - a_n(x_j)^2

where a_min, a_max = a* -+ sqrt((c - u)/A) are the ends of the sublevel
fibre {L < c} and c is any rational at or above the source saddle's
level.  At rational nodes a_n, a*, u and A are exact rationals; the square
root is rounded outward; and the integral is bounded BELOW by a sum of

    4 (A(mid) - (h/2) D) / D,     D = |A'(mid)| + (h/2) sup|A''|,

over the cells, with sup|A''| an exact interval-Horner enclosure per cell.  One
violated inequality, in exact arithmetic, excludes the climb.  The gates
are the limiting cases: a_n = 0 at a root of B', and a divergent integral
at a root of A'.  So is an empty fibre of the uphill set, a_n beyond
a_max or inside a_min, which is a gate that depends on the level.

FLOATS PROPOSE, EXACT DISPOSES.  ``Climb.blocked`` is the binary64 verdict
from J's closed form on the whole stretch -- the proposal.  ``Climb.proved``
is the exact verdict, refined until it is decided or a node budget runs
out.  Only ``proved`` (and the exact gate count) may be used as a
certificate; an unproved block is an open branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from . import _poly as P
from . import lipschitz, merge_tree, signature, stable_tails, sturm

SLACK = 1e-9


@dataclass(frozen=True)
class Climb:
    minimum: int
    saddle: int
    blocked: bool
    reason: str | None        # 'gate' | 'work' | None
    kind: str | None          # 'type1' | 'type2' | None (gate)
    work: float | None
    bound: float | None
    proved: bool = False      # exact: the climb is impossible
    proof: str | None = None  # 'gate' | 'work' | 'empty_fibre' | None
    nodes: int = 0            # rational nodes of the deciding refinement
    alarm: str | None = None  # a hypothesis of the lemma failed at a node

    def as_dict(self) -> dict:
        return {"minimum": self.minimum, "saddle": self.saddle,
                "blocked": self.blocked, "reason": self.reason,
                "kind": self.kind, "work": self.work, "bound": self.bound,
                "proved": self.proved, "proof": self.proof,
                "nodes": self.nodes, "alarm": self.alarm}


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def _sqrt_upper(q: Fraction, bits: int = 64) -> Fraction:
    """A rational at or above sqrt(q), within 2^-bits of it; q >= 0."""
    scaled = (q.numerator << (2*bits))//q.denominator
    return Fraction(math.isqrt(scaled) + 1, 1 << bits)


class Analysis:
    """Gates and climb tests for one model."""

    def __init__(self, m, e, bb: signature.Backbone | None = None):
        self.m, self.e = m, e
        self.tree = merge_tree.build(m, e)
        self.bb = signature.backbone(m, e, self.tree) if bb is None else bb
        self.A, self.B = m.alpha, m.beta
        self.Ap, self.Bp = P.deriv(self.A), P.deriv(self.B)
        self._App = P.deriv(self.Ap)
        self._c1, self._rhos, self._res = stable_tails._j_invariant(
            m, float(P.degree(self.A))/2.0)
        self._d = float(P.degree(self.A))/2.0
        self._climbs: dict = {}

    # -- exact pieces --------------------------------------------------- #

    def _inner(self, i: int, j: int):
        """A rational interval strictly between critical points i and j."""
        lo, hi = sorted((i, j))
        left = Fraction(self.e.points[lo].interval.hi)
        right = Fraction(self.e.points[hi].interval.lo)
        return (left, right) if left < right else None

    def gate_between(self, i: int, j: int) -> bool:
        inner = self._inner(i, j)
        if inner is None:
            return False
        lo, hi = inner
        for poly in (self.Ap, self.Bp):
            if P.degree(poly) >= 1 and sturm.count_roots(poly, lo, hi) > 0:
                return True
        return False

    # -- binary64 pieces ------------------------------------------------ #

    def _value(self, poly, b: Fraction) -> float:
        return float(P.eval_at(poly, b))

    def a_star(self, i: int) -> float:
        b = Fraction(self.e.points[i].interval.mid)
        return abs(float(P.eval_at(self.B, b)/P.eval_at(self.A, b)))

    def u(self, i: int) -> float:
        b = Fraction(self.e.points[i].interval.mid)
        A, B = P.eval_at(self.A, b), P.eval_at(self.B, b)
        return float(Fraction(self.m.C) - B*B/A)

    def _G(self, b: float) -> float:
        """INT^b 2A/A' : the b-part of J, closed form."""
        logs = np.log(complex(b) - self._rhos)
        return (b*b/(2.0*self._d) + 2.0*self._c1*b
                + 2.0*float(np.real(np.sum(self._res*logs))))

    def work(self, b0: float, b1: float) -> float:
        """W = INT over [b0, b1] of 4A/|A'|, for a stretch with no root of A'.

        Real roots of A' contribute a constant imaginary part to each
        logarithm on such a stretch, and complex ones stay in one half
        plane, so the difference of real parts is the integral.
        """
        return 2.0*abs(self._G(b1) - self._G(b0))

    def work_floor(self, b0: float, b1: float) -> float:
        return 2.0*(stable_tails._j_floor(0.0, b0, self._d, self._c1,
                                          self._rhos, self._res)
                    + stable_tails._j_floor(0.0, b1, self._d, self._c1,
                                            self._rhos, self._res))

    # -- the exact test ------------------------------------------------- #

    def level_above(self, s: int) -> Fraction:
        """A rational at or above u at critical point s.

        The lowest verified separating level above it, or C: u <= C always,
        with equality exactly at the roof saddles.
        """
        h = self.bb.height[s]
        levels = self.tree.levels
        return Fraction(levels[h]) if h < len(levels) else Fraction(self.m.C)

    @staticmethod
    def _cell_lower(A_mid: Fraction, slope_mid: Fraction, h: Fraction,
                    K2: Fraction) -> Fraction:
        """A dyadic lower bound on INT 4A/|A'| over one cell of width h.

        On the cell |A'| <= D = |A'(mid)| + (h/2) sup|A''| and
        A >= A(mid) - (h/2) D, so the integrand is at least
        4 (A(mid) - (h/2) D)/D.  Rounded DOWN to 2^-128 ABSOLUTE; a cell
        smaller than that contributes nothing, which is still a bound.
        """
        D = abs(slope_mid) + h/2*K2
        cell = 4*(A_mid - h/2*D)/D*h
        if cell <= 0:
            return Fraction(0)
        return Fraction((cell.numerator << 128)//cell.denominator, 1 << 128)

    def _curvature(self, left: Fraction, right: Fraction) -> Fraction:
        """sup|A''| on one cell: an exact interval-Horner enclosure.

        Per cell, not per stretch: on a stretch reaching out to a far
        saddle A'' varies by many orders, and one global bound would zero
        every cell near the small-|b| end.
        """
        return lipschitz._abs_max(lipschitz.poly_interval(
            self._App, lipschitz._iv(left, right)))

    def work_lower(self, lo: Fraction, hi: Fraction, cells: int) -> Fraction:
        """Exact lower bound on W over the rational interval [lo, hi]."""
        lo, hi = Fraction(lo), Fraction(hi)
        h = (hi - lo)/cells
        total = Fraction(0)
        for k in range(cells):
            left = lo + k*h
            c = left + h/2
            total += self._cell_lower(
                P.eval_at(self.A, c), P.eval_at(self.Ap, c), h,
                self._curvature(left, left + h))
        return total

    def prove(self, i_min: int, i_sad: int, level: Fraction,
              budget=(32, 128, 512, 2048)) -> tuple:
        """Exact attempt to exclude the climb.  (proved, proof, nodes, alarm).

        Refines a dyadic grid inside the stretch until one of the chain
        inequalities of the module docstring is violated in exact
        arithmetic, or a midpoint estimate says none will be, or the
        budget is spent.  Not proved means nothing.
        """
        inner = self._inner(i_min, i_sad)
        if inner is None:
            return False, None, 0, None
        lo, hi = inner
        A, B, Ap, Bp = self.A, self.B, self.Ap, self.Bp
        mid = (lo + hi)/2
        direction = 1 if i_sad > i_min else -1
        type1 = _sign(P.eval_at(Ap, mid))*direction > 0
        sB = _sign(P.eval_at(B, mid))
        if sB == 0:
            return False, None, 0, "B vanishes inside the stretch"
        C = Fraction(self.m.C)
        for n in budget:
            # Step 2^-k, with k NEGATIVE on a long stretch: a far saddle
            # can sit at |b| ~ 10^5, and a unit step there is 10^5 nodes.
            k = math.ceil(math.log2(n/float(hi - lo)))
            while True:
                h = Fraction(1, 1 << k) if k >= 0 else Fraction(1 << -k)
                j0 = -((-lo)//h)                                    # ceil
                j1 = hi//h                                          # floor
                if j1 - j0 >= 4:
                    break
                k += 1
            xs = [j*h for j in range(j0, j1 + 1)]
            if direction < 0:
                xs.reverse()
            # The lower sum is kept DYADIC: each cell's contribution is
            # rounded down to 2^-128, which only lowers a lower bound and
            # stops the denominators of a long exact sum from compounding.
            P_low, P_est = Fraction(0), 0.0
            best = best_est = None        # running minima over earlier nodes
            proved = likely = False
            for index, x in enumerate(xs):
                if index:
                    c = (xs[index - 1] + x)/2
                    Ac, Dc = P.eval_at(A, c), abs(P.eval_at(Ap, c))
                    P_low += self._cell_lower(
                        Ac, Dc, h, self._curvature(xs[index - 1], x))
                    if Dc:
                        P_est += float(4*Ac/Dc*h)
                Ax, Bx = P.eval_at(A, x), P.eval_at(B, x)
                Apx, Bpx = P.eval_at(Ap, x), P.eval_at(Bp, x)
                if Apx == 0:
                    return True, "gate", len(xs), None
                star = sB*Bx/Ax                       # a*, normalised > 0
                null = sB*2*Bpx/Apx                   # the b-nullcline
                q = (level - (C - Bx*Bx/Ax))/Ax
                if q < 0:
                    return True, "empty_fibre", len(xs), None
                rho = _sqrt_upper(q)
                if type1:
                    if not null < star:
                        return False, None, len(xs), "a_n >= a* in type 1"
                    low = max(Fraction(0), star - rho)
                    if null <= low:                   # uphill fibre empty
                        return True, "empty_fibre", len(xs), None
                    here, key = P_low - null*null, P_low - low*low
                    here_est = P_est - float(null*null)
                    key_est = P_est - float(low*low)
                else:
                    if not null > star:
                        return False, None, len(xs), "a_n <= a* in type 2"
                    high = star + rho
                    if null >= high:
                        return True, "empty_fibre", len(xs), None
                    here, key = P_low + null*null, P_low + high*high
                    here_est = P_est + float(null*null)
                    key_est = P_est + float(high*high)
                if best is not None and here > best:
                    proved = True
                    break
                # Would a 5% larger integral break a chain?  If not even
                # that, refining the lower sum cannot.
                if best_est is not None and \
                        here_est + P_est/20 > best_est:
                    likely = True
                best = key if best is None else min(best, key)
                best_est = key_est if best_est is None \
                    else min(best_est, key_est)
            if proved:
                return True, "work", len(xs), None
            if not likely:
                return False, None, len(xs), None
        return False, None, len(xs), None

    # -- the test ------------------------------------------------------- #

    def climb(self, i_min: int, i_sad: int, level: float,
              exact_level: Fraction | None = None) -> Climb:
        """Can an orbit below ``level`` get from minimum i_min over i_sad?

        ``level`` is the binary64 level for the proposal; ``exact_level``,
        a rational at or above the source's level, asks for the proof too.
        """
        if self.gate_between(i_min, i_sad):
            return Climb(i_min, i_sad, True, "gate", None, None, None,
                         proved=True, proof="gate")
        pts = self.e.points
        b_min, b_sad = float(pts[i_min].b), float(pts[i_sad].b)
        inner = self._inner(i_min, i_sad)
        mid = (Fraction(pts[i_min].interval.mid)
               + Fraction(pts[i_sad].interval.mid))/2 if inner is None \
            else (inner[0] + inner[1])/2
        uphill = 1 if b_sad > b_min else -1          # u rises toward i_sad
        slope_A = P.eval_at(self.Ap, mid)
        kind = "type1" if (slope_A > 0) == (uphill > 0) else "type2"
        W = self.work(b_min, b_sad)
        top = self.a_star(i_sad)
        if kind == "type1":
            bound = top*top
        else:
            b = Fraction(pts[i_min].interval.mid)
            A = float(P.eval_at(self.A, b))
            a_max = self.a_star(i_min) + math.sqrt(
                max(level - self.u(i_min), 0.0)/A)
            bound = a_max*a_max - top*top
        floor = self.work_floor(b_min, b_sad)
        blocked = W*(1.0 - SLACK) - floor > bound*(1.0 + SLACK)
        proved, proof, nodes, alarm = (
            self.prove(i_min, i_sad, exact_level)
            if exact_level is not None else (False, None, 0, None))
        return Climb(i_min, i_sad, bool(blocked),
                     "work" if blocked else None, kind, float(W),
                     float(bound), proved=proved, proof=proof, nodes=nodes,
                     alarm=alarm)

    def reachable(self, s: int, direction: int, rigorous: bool = False):
        """Candidates of the branch (s, direction) that survive the test.

        Returns (minima, saddles, end, climbs): the merge tree's candidates
        with everything beyond a blocked climb removed, and the backbone
        end kept only past an outermost saddle.  With ``rigorous`` only an
        exact gate or a proved climb blocks; otherwise the binary64
        proposal blocks as well.
        """
        bb = self.bb
        n = len(bb.letters)
        tree_minima, tree_saddles, _tree_end = signature.candidates(
            bb, s, direction)
        allowed = set(tree_minima) | set(tree_saddles)
        level = self.u(s)
        exact_level = self.level_above(s)
        minima, saddles, climbs, end = [], [], [], None
        i = s + direction
        while True:
            if not 0 <= i < n:
                end = (signature.END_PLUS if direction > 0
                       else signature.END_MINUS)
                break
            if i not in allowed:
                break                 # a saddle at least as high: the wall
            minima.append(i)
            j = i + direction
            if not 0 <= j < n or j not in allowed:
                break                 # outermost minimum, or the wall
            step = self._climbs.get((i, j, exact_level))
            if step is None:
                step = self.climb(i, j, level, exact_level)
                self._climbs[(i, j, exact_level)] = step
            climbs.append(step)
            if step.proved or (step.blocked and not rigorous):
                break
            saddles.append(j)
            i = j + direction
        return tuple(minima), tuple(saddles), end, tuple(climbs)


def report(m, e, attachments=None, bb=None) -> dict:
    """Pruning statistics for one model, checked against certified ends.

    ``attachments`` is signature.attachments(p): per saddle in b order, the
    (toward -b, toward +b) codes.  A certified end the test excludes is a
    counterexample to the lemma or a defect here, and is listed in full.
    """
    an = Analysis(m, e, bb)
    bb = an.bb
    saddles = [i for i in range(len(bb.letters)) if bb.is_saddle(i)]
    out = {"branches": 0, "open_tree": 0, "open_climb": 0, "open_proved": 0,
           "connections_tree": 0, "connections_climb": 0,
           "connections_proved": 0,
           "climbs": 0, "gate": 0, "type1_blocked": 0, "type1_pass": 0,
           "type2_blocked": 0, "type2_pass": 0, "checked": 0,
           "proved_work": 0, "proved_empty_fibre": 0, "proved_gate_node": 0,
           "blocked_unproved": 0, "proved_beyond_proposal": 0,
           "alarms": 0, "max_nodes": 0,
           "tightest_pass": None, "contradicted": []}
    for k, s in enumerate(saddles):
        for slot, d in ((0, -1), (1, +1)):
            t_min, t_sad, t_end = signature.candidates(bb, s, d)
            r_min, r_sad, r_end, climbs = an.reachable(s, d)
            x_min, x_sad, x_end, x_climbs = an.reachable(s, d, rigorous=True)
            out["open_proved"] += (len(x_min) + (x_end is not None) > 1
                                   or bool(x_sad))
            out["connections_proved"] += len(x_sad)
            for c in x_climbs:
                if c.reason == "gate":
                    continue
                out["max_nodes"] = max(out["max_nodes"], c.nodes)
                out["alarms"] += c.alarm is not None
                if c.proved:
                    out["proved_" + ("gate_node" if c.proof == "gate"
                                     else c.proof)] += 1
                    out["proved_beyond_proposal"] += not c.blocked
                elif c.blocked:
                    out["blocked_unproved"] += 1
            out["branches"] += 1
            out["open_tree"] += (len(t_min) + (t_end is not None) > 1
                                 or bool(t_sad))
            out["open_climb"] += (len(r_min) + (r_end is not None) > 1
                                  or bool(r_sad))
            out["connections_tree"] += len(t_sad)
            out["connections_climb"] += len(r_sad)
            for c in climbs:
                out["climbs"] += 1
                if c.reason == "gate":
                    out["gate"] += 1
                else:
                    out[f"{c.kind}_{'blocked' if c.blocked else 'pass'}"] += 1
                    if not c.blocked and c.bound and c.bound > 0:
                        ratio = c.work/c.bound
                        if (out["tightest_pass"] is None
                                or ratio > out["tightest_pass"]):
                            out["tightest_pass"] = ratio
            if attachments is None:
                continue
            got = attachments[k][slot]
            if got == signature.UNKNOWN:
                continue
            out["checked"] += 1
            ok = (r_end == got if isinstance(got, str)
                  else (s + got) in r_min)
            ok_exact = (x_end == got if isinstance(got, str)
                        else (s + got) in x_min)
            if not (ok and ok_exact):
                out["contradicted"].append({
                    "saddle": s, "direction": d, "certified": got,
                    "by": "proof" if not ok_exact else "proposal",
                    "tree_minima": list(t_min), "reachable_minima":
                    list(r_min), "reachable_end": r_end,
                    "climbs": [c.as_dict() for c in climbs]})
    return out


def uphill_chords(m, branch, stride: int = 20, skip: int = 600) -> dict:
    """The lemma's differential inequality, read off a traced branch.

    For every sampled chord of an unstable branch that moves uphill in u,
    the ratio |d(a^2)| / (4A/|A'| |db|), which the lemma says is >= 1.
    A polyline is not the orbit, so the ratio is reported, not asserted;
    the stub's logarithmic samples and the terminal connector are skipped.

    WHICH chords are uphill is decided EXACTLY, by comparing the rational
    values of u at the chord's two binary64 endpoints.  The sign of
    u' = -B(2AB' - A'B)/A^2 in binary64 is not good enough: when
    deg B = deg g the leading terms of 2AB' and A'B cancel, and at a far
    minimum (|b| = 524288 on a directed case) the float difference is zero
    or of the wrong sign.  That mislabelled one downhill chord as uphill
    in 74,043 and reported a ratio of 0 for it.
    """
    Y = np.asarray(branch.Y, dtype=float)[skip:-2:stride]
    if len(Y) < 3:
        return {"chords": 0, "min_ratio": None, "below": 0}
    desc = lambda p: np.asarray([float(c) for c in p][::-1])  # noqa: E731
    A = desc(m.alpha)
    Ap = np.polyder(A)
    a0, b0, a1, b1 = Y[:-1, 0], Y[:-1, 1], Y[1:, 0], Y[1:, 1]
    bm, db = 0.5*(b0 + b1), b1 - b0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fA, fAp = np.polyval(A, bm), np.polyval(Ap, bm)
        sel = (np.abs(db) > 1e-9) & (fAp != 0.0)
        # A chord that straddles a root of A' is not a stretch of the lemma.
        sel &= np.polyval(Ap, b0)*np.polyval(Ap, b1) > 0.0
        ratio = np.abs(a1*a1 - a0*a0)/(4.0*fA/np.abs(fAp)*np.abs(db))
    sel &= np.isfinite(ratio)
    C = Fraction(m.C)

    def u_exact(b: float) -> Fraction:
        x = Fraction(float(b))
        B = P.eval_at(m.beta, x)
        return C - B*B/P.eval_at(m.alpha, x)

    values = {}
    keep = []
    for index in np.flatnonzero(sel):
        for b in (b0[index], b1[index]):
            if b not in values:
                values[b] = u_exact(b)
        if values[b1[index]] > values[b0[index]]:
            keep.append(index)
    ratio = ratio[keep]
    if len(ratio) == 0:
        return {"chords": 0, "min_ratio": None, "below": 0}
    return {"chords": int(len(ratio)), "min_ratio": float(ratio.min()),
            "below": int(np.count_nonzero(ratio < 0.98))}
