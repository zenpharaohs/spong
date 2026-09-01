"""Real roots of the level polynomial without a Sturm chain of it.

The level polynomial R_c = (C - c)A - B^2 has, when A > 0, exactly the real
roots of u = c: R_c = A (u - c).  Between consecutive critical points --
the real roots of u' = -BN/A^2, which the exact enumeration holds with
isolating intervals -- u is strictly monotone, so u - c changes sign at
most once on each open piece, and does so iff its signs at the two ends
differ.  The signs at the ends are exact signs of R_c at algebraic points,
which the fused C kernel answers on the ROOT polynomial's persistent plan
(merge_tree.value_sign).  So the whole real-root structure of R_c -- how
many, in which pieces, on which side of every critical point -- follows
from n + 2 signs, with no Sturm chain of R_c at all.  On d17-thrash the
certifier built 29 such chains (degree 34, 300-bit coefficients, 7 s of
48) to learn what these signs say.

CONTRACT.  Alternative implementation of the three questions the merge
tree and the sublevel inventory ask of R_c:

    roots(c)              sorted isolating intervals, one per real root
    count(c, lo, hi)      number of real roots in (lo, hi], None = infinity
    gap_index(c, point)   number of real roots of R_c below the critical
                          point (which gap between roots it lies in)

Isolating intervals differ from sturm.isolate_roots' (those come from
bisection of Cauchy bounds; these from the critical intervals), so the
rationals are not comparable -- the COUNTS, ORDER and MEMBERSHIPS are.

REFUSALS.  A critical point whose sign is undecided (c is, or is within
the refinement budget of, its critical value) raises LevelTie: the level
is a critical value, the sign alternation fails, and the caller must
treat c exactly as components_at does today (ValueError).  A model that
is not psi-nice (A has a real root: u has poles, R_c = A(u - c) no longer
tracks u - c) is refused at construction; those cases keep the Sturm path.

This module is validated against the Sturm path by
scripts/level_roots_probe.py and is not yet wired into production.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from . import _poly as P
from . import merge_tree, sturm
from .sturm import RootInterval


class LevelTie(ValueError):
    """The level is (or is undecidably near) a critical value."""


def _sign(v: Fraction) -> int:
    return (v > 0) - (v < 0)


def _cauchy_bound(R) -> Fraction:
    """M with every real root of R in (-M, M)."""
    lead = R[-1]
    return 1 + max((abs(Fraction(a) / lead) for a in R[:-1]), default=Fraction(0))


@dataclass
class LevelRoots:
    m: object
    e: object

    def __post_init__(self):
        if not self.e.psi_positive:
            raise ValueError("level_roots requires a psi-nice model (A > 0)")
        pts = [p for p in self.e.points if p.kind != "degenerate"]
        pts.sort(key=lambda p: (p.interval.lo, p.interval.hi))
        self.points = pts
        self.index = {id(p): i for i, p in enumerate(pts)}
        self._signs: dict = {}
        self._roots: dict = {}

    # -- the n + 2 signs -------------------------------------------------

    def signs(self, c: Fraction):
        """(sign at -inf, [sign at each critical point], sign at +inf)."""
        R = merge_tree.level_polynomial(self.m, c)
        left = merge_tree._sign_at_infinity(R, positive=False)
        right = merge_tree._sign_at_infinity(R, positive=True)
        at = []
        for p in self.points:
            s = merge_tree.value_sign(self.m, p, c)
            if s is None or s == 0:
                raise LevelTie(
                    f"level {float(c)} is a critical value at b ~ {p.b}")
            at.append(s)
        return R, left, at, right

    # -- pieces ----------------------------------------------------------

    def _piece_signs(self, c):
        if c not in self._signs:
            R, left, at, right = self.signs(c)
            self._signs[c] = (R, [left, *at, right])
        return self._signs[c]

    def _tight(self, R, i: int, want: int, side: str) -> Fraction:
        """A rational endpoint of critical point i's interval, on the given
        side, at which R already has the point's sign ``want``.

        Refined with the point's OWN root polynomial (never with R, which
        would bisect toward a level crossing and walk off the point).
        """
        p = self.points[i]
        iv = p.interval
        root = merge_tree._root_poly(self.m, p)
        for _ in range(200):
            x = iv.lo if side == "lo" else iv.hi
            if iv.exact or _sign(P.eval_at(R, x)) == want:
                return x
            iv = sturm.refine(
                root, iv, rel=(iv.hi - iv.lo) / (1 + abs(iv.mid)) / 4)
        raise ArithmeticError("critical interval would not separate from "
                              "the level crossing")

    def roots(self, c: Fraction, width: Fraction | None = None
              ) -> list[RootInterval]:
        """Sorted isolating intervals of the real roots of R_c."""
        if width is None and c in self._roots:
            return self._roots[c]
        R, seq = self._piece_signs(c)
        n = len(self.points)
        out: list[RootInterval] = []
        for k in range(n + 1):           # piece k: between seq[k], seq[k+1]
            if seq[k] == seq[k + 1]:
                continue
            if k == 0:
                lo = -_cauchy_bound(R)
            else:
                lo = self._tight(R, k - 1, seq[k], "hi")
            if k == n:
                hi = _cauchy_bound(R)
            else:
                hi = self._tight(R, k, seq[k + 1], "lo")
            out.append(self._bisect(R, lo, hi, seq[k], seq[k + 1], width))
        if width is None:
            self._roots[c] = out
        return out

    @staticmethod
    def _bisect(R, lo, hi, s_lo, s_hi, width) -> RootInterval:
        """Shrink [lo, hi], on which R has exactly one sign change, to the
        requested width by exact sign evaluation; an exactly hit rational
        root is returned exact."""
        if width is None:
            return RootInterval(lo, hi, False)
        while hi - lo > width * (1 + abs(lo) + abs(hi)):
            mid = (lo + hi) / 2
            s = _sign(P.eval_at(R, mid))
            if s == 0:
                return RootInterval(mid, mid, True)
            if s == s_lo:
                lo = mid
            else:
                hi = mid
        return RootInterval(lo, hi, False)

    # -- counts -----------------------------------------------------------

    def count(self, c: Fraction, lo: Fraction | None, hi: Fraction | None
              ) -> int:
        """Number of real roots of R_c in (lo, hi]."""
        R, seq = self._piece_signs(c)
        total = 0
        for iv in self.roots(c):
            total += self._in(R, iv, lo, hi)
        return total

    @staticmethod
    def _in(R, iv: RootInterval, lo, hi) -> int:
        """Whether the single root in iv lies in (lo, hi], deciding by
        bisection on the exact sign of R when an endpoint falls inside."""
        a, b = iv.lo, iv.hi
        if iv.exact:
            r = a
            return int((lo is None or lo < r) and (hi is None or r <= hi))
        # R is nonzero at both ends by construction: a < root < b.
        s_a = _sign(P.eval_at(R, a))
        for _ in range(4096):
            left_ok = lo is None or lo <= a
            left_out = lo is not None and b <= lo
            right_ok = hi is None or b <= hi
            right_out = hi is not None and a >= hi
            if left_ok and right_ok:
                return 1
            if left_out or right_out:
                return 0
            # an endpoint (lo or hi) lies strictly inside (a, b): split there
            cut = lo if (lo is not None and a < lo < b) else hi
            s = _sign(P.eval_at(R, cut))
            if s == 0:                  # the root IS the endpoint
                return int(cut == hi)   # (lo, hi]: counts iff it is hi
            if s == s_a:
                a = cut
            else:
                b = cut
        raise ArithmeticError("root interval would not resolve against "
                              "the query endpoints")

    def gap_index(self, c: Fraction, point) -> int:
        """Number of real roots of R_c strictly below the critical point."""
        R, seq = self._piece_signs(c)
        i = self.index[id(point)]
        return sum(1 for k in range(i + 1) if seq[k] != seq[k + 1])
