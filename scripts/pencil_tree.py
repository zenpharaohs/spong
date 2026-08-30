"""The merge tree from the level pencil: exact, box-free, no tracing. v3.

SCOPE.  This is the Morse/Reeb tree, not the Smale attaching map.  The real
branch points of ``y^2=S_ell(b)`` determine sublevel components, but stable
and unstable separatrix incidence is the level-to-level holonomy on that
hyperelliptic family.  See ``docs/hyperelliptic_smale.md``.

THEORY.  For psi-nice models the sublevel set {L <= l} fibers in segments
over {S_l >= 0}, S_l(b) = B^2 + (l - C)A, so pi0(sublevel) = maximal
intervals of {S_l >= 0} and the 2D merge tree is the merge structure of a
monotone one-variable pencil.  Events are the critical values u(b*) plus
the rational degree-drop level u_inf = C - beta^2/alpha where unbounded
rays switch on.

ALGORITHM (v3 -- no discriminant).  v2 formed D(l) = disc_b S_l by exact
Sylvester determinants; its coefficients are 15-fold products of the
moment-scale rationals and everything downstream drowned.  But the tree
never needs D:

  * the event LOCATIONS are the real roots of B*N -- native-scale
    polynomials, isolated by the same primitive-PRS Sturm technology the
    engine already uses;
  * every LEVEL COMPARISON the tree needs is one sign:
        u(b*) > l  <=>  S_l(b*) < 0,
    decided by refining b*'s isolating interval until the fixed integer
    polynomial S_l holds one sign on it;
  * event-vs-event order is bisection over rational l with that test;
  * the against-u_inf verdict -- the far-saddle 37-ulp question -- is
    sign(S_{u_inf}(b*)), and S_{u_inf} = B^2 - (beta^2/alpha) A = -R/alpha:
    the codebase's own R polynomial (u - u_inf = R/(alpha A)).  "An R-root
    shadows the far N-root" was the difficulty; it is now the decision
    procedure.

The genealogy is the classic watershed on the b-ordered critical sequence:
minima are born, each saddle (a local max of u, hence above both flanking
minima) fuses the live components on its two sides, and the two rays are
born at l = u_inf.  Exact order corrections relative to float order are
reported -- that is the payoff column.

    python scripts/pencil_tree.py 1785201004 1143710268        # controls
    python scripts/pencil_tree.py 555999196 634753038          # far saddles
    python scripts/pencil_tree.py 953953598 --pow2             # invariance
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import sturm                                   # noqa: E402
from qualify import directed_model, random_model          # noqa: E402
from pole_portrait import exact_ABCN, pmul, pderiv        # noqa: E402
from pole_portrait import peval as fpeval                 # noqa: E402
from box_experiment_arms import normalised                # noqa: E402

ZERO = Fraction(0)


# ------------------------------------------------------------------ #
# primitive integer polynomials (ascending int lists)
# ------------------------------------------------------------------ #

def itrim(p):
    while p and p[-1] == 0:
        p.pop()
    return p


def primitive(p):
    p = itrim(list(p))
    if not p:
        return p
    g = 0
    for v in p:
        g = math.gcd(g, v)
    return [v // g for v in p] if g > 1 else p


def integerize(fracs):
    fracs = [Fraction(c) for c in fracs]
    den = 1
    for c in fracs:
        den = den * c.denominator // math.gcd(den, c.denominator)
    return primitive([int(c * den) for c in fracs])


def ideriv(p):
    return itrim([k * c for k, c in enumerate(p)][1:])


def sign_at(p, x):
    if not p:
        return 0
    if x == "+inf":
        return (p[-1] > 0) - (p[-1] < 0)
    if x == "-inf":
        s = (p[-1] > 0) - (p[-1] < 0)
        return s * (-1 if (len(p) - 1) % 2 else 1)
    num, den = x.numerator, x.denominator
    acc, q = 0, 1
    for c in reversed(p):
        acc = acc * num + c * q
        q *= den
    return (acc > 0) - (acc < 0)


def _prs_step(f, g):
    """Sign-safe pseudo-remainder ~ rem(f, g), primitive, positive scale."""
    f = list(f)
    lc = g[-1]
    t = 0
    while len(f) >= len(g) and f:
        c = f[-1]
        k = len(f) - len(g)
        f = [lc * v for v in f]
        for i, gc in enumerate(g):
            f[i + k] -= c * gc
        assert f[-1] == 0
        f.pop()
        itrim(f)
        t += 1
    if lc < 0 and t % 2:
        f = [-v for v in f]
    return primitive(f)


def igcd_poly(f, g):
    f, g = primitive(f), primitive(g)
    while g:
        f, g = g, _prs_step(f, g)
    return f


def idivide(f, g):
    fq = [Fraction(c) for c in f]
    q = [ZERO] * max(len(f) - len(g) + 1, 1)
    while len(fq) >= len(g):
        c = fq[-1] / g[-1]
        k = len(fq) - len(g)
        q[k] = c
        for i, gc in enumerate(g):
            fq[i + k] -= c * gc
        while fq and fq[-1] == 0:
            fq.pop()
        if not fq:
            break
    return integerize(q)


class RP:
    """Primitive integer polynomial with cached Sturm machinery."""

    def __init__(self, coeffs, from_fracs=False):
        self.c = integerize(coeffs) if from_fracs else primitive(list(coeffs))
        self._sqf = None
        self._chain = None

    @property
    def deg(self):
        return len(self.c) - 1

    def sqfree(self):
        if self._sqf is None:
            p = self.c
            if len(p) <= 2:
                self._sqf = p
            else:
                g = igcd_poly(p, ideriv(p))
                self._sqf = p if len(g) <= 1 else idivide(p, g)
        return self._sqf

    def chain(self):
        if self._chain is None:
            p = self.sqfree()
            chain = [p, ideriv(p)]
            while len(chain[-1]) > 1:
                r = _prs_step(chain[-2], chain[-1])
                if not r:
                    break
                chain.append([-v for v in r])
            self._chain = [q for q in chain if q]
        return self._chain

    def variations(self, x):
        signs = [s for s in (sign_at(q, x) for q in self.chain()) if s]
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    def count(self, lo, hi):
        return self.variations(lo) - self.variations(hi)

    def bound(self):
        p = self.sqfree()
        if len(p) <= 1:
            return Fraction(1)
        return Fraction(1 + max(abs(v) for v in p[:-1]) // abs(p[-1]) + 1)

    def _nudge(self, lo, mid, hi):
        while sign_at(self.sqfree(), mid) == 0:
            mid = (lo + 2 * mid + hi) / 4
        return mid

    def isolate(self):
        if self.deg < 1:
            return []
        M = self.bound()
        total = self.count("-inf", "+inf")
        assert self.count(-M, M) == total, "root bound failed"
        work = [(-M, M, total)]
        done = []
        while work:
            lo, hi, k = work.pop()
            if k == 0:
                continue
            if k == 1:
                done.append((lo, hi))
                continue
            mid = self._nudge(lo, (lo + hi) / 2, hi)
            left = self.count(lo, mid)
            work.append((lo, mid, left))
            work.append((mid, hi, k - left))
        return sorted(done)

    def refine(self, lo, hi, width):
        while hi - lo > width:
            mid = self._nudge(lo, (lo + hi) / 2, hi)
            if self.count(lo, mid):
                hi = mid
            else:
                lo = mid
        return lo, hi


# ------------------------------------------------------------------ #
# events and exact level comparisons
# ------------------------------------------------------------------ #

class Event:
    """One critical point: an isolating b-interval on BN plus metadata."""

    def __init__(self, name, kind, bn: RP, lo, hi, u_float, b_float):
        self.name = name
        self.kind = kind                     # 'min' or 'saddle'
        self.bn = bn
        self.lo = lo
        self.hi = hi
        self.u_float = u_float
        self.b_float = b_float

    def side_of(self, S_ell: "RP", depth=300):
        """sign(u(b*) - ell) for rational ell with S_ell precomputed.

        Same endpoint signs do NOT prove constant sign on the interval
        (S_ell dips near a tie).  The sound criterion: refine b*'s
        interval until S_ell has NO root in it (exact Sturm count), then
        the sign at the midpoint -- provably not a root -- is the sign at
        b*.  Exact ties (S_ell(b*) = 0, e.g. every a*=0 saddle at
        ell = C) fall through to the gcd certificate.
        """
        lo, hi = self.lo, self.hi
        for _ in range(depth):
            if S_ell.count(lo, hi) == 0:
                s = sign_at(S_ell.c, (lo + hi) / 2)
                self.lo, self.hi = lo, hi
                return -s                    # u > ell  <=>  S_ell(b*) < 0
            lo, hi = self.bn.refine(lo, hi, (hi - lo) / 4)
        g = igcd_poly(S_ell.c, self.bn.c)
        if len(g) > 1 and RP(g).count(lo, hi):
            self.lo, self.hi = lo, hi
            return 0                         # exact: u(b*) == ell
        raise RuntimeError("level side undecided at depth cap")


def s_level(P, Q, ell: Fraction) -> RP:
    return RP([(P[i] if i < len(P) else ZERO)
               + ell * (Q[i] if i < len(Q) else ZERO)
               for i in range(max(len(P), len(Q)))], from_fracs=True)


class LevelOracle:
    def __init__(self, P, Q, u_inf: Fraction):
        self.P, self.Q = P, Q
        self.u_inf = u_inf
        self._cache: dict[Fraction, RP] = {}

    def S(self, ell: Fraction) -> RP:
        if ell not in self._cache:
            self._cache[ell] = s_level(self.P, self.Q, ell)
        return self._cache[ell]

    def vs_uinf(self, ev: Event) -> int:
        """sign(u(b*) - u_inf); S_{u_inf} = -R/alpha, the codebase's R."""
        return ev.side_of(self.S(self.u_inf))

    def compare(self, a: Event, b: Event, depth=200) -> int:
        """sign(u_a - u_b), exactly; 0 means exactly equal levels."""
        lo = Fraction(min(a.u_float, b.u_float))
        hi = Fraction(max(a.u_float, b.u_float))
        span = max(hi - lo, Fraction(1, 1 << 20) * max(1, abs(hi)))
        lo -= span
        hi += span
        while a.side_of(self.S(lo)) < 0 or b.side_of(self.S(lo)) < 0:
            lo -= span
            span *= 2
        while a.side_of(self.S(hi)) > 0 or b.side_of(self.S(hi)) > 0:
            hi += span
            span *= 2
        # invariant: both events lie in (lo, hi)
        for _ in range(depth):
            mid = (lo + hi) / 2
            sa, sb = a.side_of(self.S(mid)), b.side_of(self.S(mid))
            if sa == 0 and sb == 0:
                return 0
            if sa != sb:
                return 1 if sa > 0 else -1   # a above mid, b below => a > b
            if sa > 0:
                lo = mid
            else:
                hi = mid
        # Near-equality after a finite bisection budget is not an equality
        # certificate.  Only the simultaneous exact side_of()==0 branch
        # above may return a tie.
        raise RuntimeError("critical-level comparison undecided at depth cap")


# ------------------------------------------------------------------ #

def build_tree(P, Q, C, alpha, beta, events, oracle: LevelOracle):
    # exact order of all events (insertion sort with the exact comparator;
    # ties become simultaneous events)
    order = []
    comparisons = 0
    for ev in events:
        placed = False
        for i, group in enumerate(order):
            comparisons += 1
            c = oracle.compare(ev, group[0])
            if c == 0:
                group.append(ev)
                placed = True
                break
            if c < 0:
                order.insert(i, [ev])
                placed = True
                break
        if not placed:
            order.append([ev])

    uinf_sides = {ev.name: oracle.vs_uinf(ev) for ev in events}

    # float-order audit: where does the exact order disagree with floats?
    exact_names = [ev.name for group in order for ev in group]
    float_names = [ev.name for ev in sorted(events, key=lambda e: e.u_float)]
    corrections = exact_names != float_names

    # watershed on the b-ordered sequence, events processed in exact order,
    # rays inserted at u_inf
    by_b = sorted(events, key=lambda e: e.b_float)
    position = {ev.name: i for i, ev in enumerate(by_b)}
    live: dict[str, str] = {}                # b-slot name -> component label
    genealogy = []
    uinf_done = False

    def open_rays():
        # A ray is born already CONNECTED through a boundary minimum: u runs
        # monotonically from u_inf into it, so the whole stretch is below
        # level the moment l > u_inf.  Only a boundary saddle (always above
        # u_inf) separates its ray.
        attach = []
        for ray, edge in (("ray-", by_b[0]), ("ray+", by_b[-1])):
            tag = "escape-" if ray == "ray-" else "escape+"
            if edge.kind != "saddle" and edge.name in live:
                live[ray] = live[edge.name]
                attach.append(f"{tag}->{live[edge.name]}")
            else:
                live[ray] = tag
                attach.append(tag)
        genealogy.append({"kind": "escape_open", "level": float(oracle.u_inf),
                          "detail": "  ".join(attach), "vs_uinf": "at"})

    def component_left_of(idx):
        for i in range(idx - 1, -1, -1):
            name = by_b[i].name
            if name in live:
                return name
        return "ray-" if "ray-" in live else None

    def component_right_of(idx):
        for i in range(idx + 1, len(by_b)):
            name = by_b[i].name
            if name in live:
                return name
        return "ray+" if "ray+" in live else None

    for group in order:
        # the exact order is trusted: rays open immediately before the
        # first event group at or above u_inf
        if not uinf_done and all(uinf_sides[ev.name] >= 0 for ev in group):
            open_rays()
            uinf_done = True
        level = float(sum(ev.u_float for ev in group) / len(group))
        tie = len(group) > 1
        for ev in group:
            side = uinf_sides[ev.name]
            vs = "below" if side < 0 else "above" if side > 0 else "AT u_inf"
            if tie:
                vs += " =TIE="
            if ev.kind != "saddle":
                live[ev.name] = ev.name
                genealogy.append({"kind": "birth", "level": level,
                                  "detail": ev.name, "b": ev.b_float,
                                  "vs_uinf": vs})
            else:
                idx = position[ev.name]
                left = component_left_of(idx)
                right = component_right_of(idx)
                merged = sorted({live.get(left, left), live.get(right, right)}
                                - {None})
                genealogy.append({"kind": "fusion", "level": level,
                                  "detail": "+".join(str(x) for x in merged),
                                  "b": ev.b_float, "vs_uinf": vs})
                keep = merged[0]
                for name in list(live):
                    if live[name] in merged:
                        live[name] = keep
                live[ev.name] = keep
    if not uinf_done:
        open_rays()
    final = sorted(set(live.values()))
    assert len(final) == 1, (
        "tree must end in one component (S_l > 0 on all of R for large l); "
        f"got {final}")
    return genealogy, final, corrections, comparisons


def run_seed(seed, mode, degree, pow2):
    generate = directed_model if mode == "directed" else random_model
    built = generate(random.Random(seed), degree)
    if built is None or built[0] is None:
        return {"seed": seed, "error": "generator declined"}
    m, spec = built
    if pow2:
        m = normalised(m, pow2=True)
    A, B, C, N = exact_ABCN(m)
    alpha = A[-1]
    beta = B[-1] if len(B) - 1 == (len(A) - 1) // 2 else ZERO
    B2 = list(pmul(B, B))
    size = max(len(A), len(B2))
    P = [(B2[i] if i < len(B2) else ZERO)
         - C * (A[i] if i < len(A) else ZERO) for i in range(size)]
    Q = list(A)
    u_inf = C - beta * beta / alpha

    def u_of(b):
        return float(C) - fpeval(pmul(B, B), b) / fpeval(A, b)

    # events: real roots of B*N matched to the enumeration's kinds
    e = sturm.enumerate_critical_points(m)
    skel = sorted(((float(q.b), q.kind) for q in e.points))
    bn = RP(pmul(B, N), from_fracs=True)
    brackets = bn.isolate()
    events = []
    used = set()
    for lo, hi in brackets:
        matches = [i for i, (b, _) in enumerate(skel)
                   if lo < Fraction(b) <= hi or
                   abs(b - float((lo + hi) / 2)) <=
                   1e-9 * (1 + abs(b))]
        matches = [i for i in matches if i not in used]
        if not matches:
            continue                          # spurious BN root (B*N overlap)
        i = matches[0]
        used.add(i)
        b_f, kind = skel[i]
        kind = "saddle" if kind == "saddle" else "min"
        events.append(Event(f"{kind[0]}{i}@{b_f:.6g}", kind, bn, lo, hi,
                            u_of(b_f), b_f))
    if len(events) != len(skel):
        return {"seed": seed, "spec": str(spec),
                "error": f"event matching failed: {len(events)} events "
                         f"for {len(skel)} skeleton points"}

    oracle = LevelOracle(P, Q, u_inf)
    genealogy, final, corrections, comparisons = build_tree(
        P, Q, C, alpha, beta, events, oracle)
    return {"seed": seed, "spec": str(spec), "pow2": bool(pow2),
            "u_inf": float(u_inf), "events": genealogy,
            "final_components": final,
            "exact_vs_float_order_differs": bool(corrections),
            "exact_comparisons": comparisons,
            "births": sum(1 for g in genealogy if g["kind"] == "birth"),
            "fusions": sum(1 for g in genealogy if g["kind"] == "fusion"),
            "skeleton_minima": sum(1 for _, k in skel if k != "saddle"),
            "skeleton_saddles": sum(1 for _, k in skel if k == "saddle")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--mode", choices=("directed", "random"),
                    default="directed")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--pow2", action="store_true",
                    help="dyadic-normalise first; the tree must be invariant")
    ap.add_argument("--out", default=str(REPO / "out" / "pencil_tree.json"))
    args = ap.parse_args(argv)

    results = []
    for seed in args.seeds:
        r = run_seed(seed, args.mode, args.degree, args.pow2)
        results.append(r)
        if "error" in r:
            print(f"\nseed {seed}: {r['error']}")
            continue
        ok = (r["births"] == r["skeleton_minima"]
              and r["fusions"] == r["skeleton_saddles"])
        print(f"\nseed {seed}   {r['spec']}   u_inf {r['u_inf']:.9g}   "
              f"counts {'MATCH' if ok else 'MISMATCH'} "
              f"(births {r['births']}/{r['skeleton_minima']}, "
              f"fusions {r['fusions']}/{r['skeleton_saddles']})   "
              f"exact comparisons {r['exact_comparisons']}"
              + ("   FLOAT ORDER CORRECTED"
                 if r["exact_vs_float_order_differs"] else ""))
        for ev in r["events"]:
            loc = "" if ev.get("b") is None else f"   b ~ {ev['b']:.6g}"
            print(f"  {ev['level']:>18.9g}  {ev['kind']:<12}"
                  f"{(ev['detail'] or ''):<24} {ev['vs_uinf']:>9}{loc}")
        print(f"  final components: {r['final_components']}")
    Path(args.out).write_text(
        json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
