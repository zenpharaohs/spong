"""Inverse construction: prescribe critical points, solve for the target f.

Regression-suite generator.  The point is CONTROL: sampling random f, g lands
where it lands, while prescription puts a critical point at a chosen b and
therefore at a chosen stiffness (see docs/inverse_construction.md).

Mechanism.  From the model construction (model.py `__post_init__`):

    A(b) = sum_ij g_i g_j mu_{i+j} b^{i+j}          depends only on (g, mu)
    B(b) = sum_j g_j (sum_i f_i mu_{i+j}) b^j       LINEAR in f for fixed g
    N    = A'B - 2B'A                               LINEAR in B

and the critical b-values are the real roots of N union the real roots of B.
So with g and mu FIXED, both root families are linear conditions on the
coefficients of f, and a prescription is a single homogeneous linear system

    M f = 0,        f in Q^{deg_f + 1}

solved exactly over the rationals.  Any nonzero null vector is a target
polynomial whose model has the prescribed points among its critical set.

Exactness.  Everything here is Fraction arithmetic, and `model.Model` accepts
Fraction coefficients, so a prescribed b IS an exact root -- the Sturm
enumeration finds it exactly rather than nearby.  Prescribe dyadic or rational
b-values; irrational targets are not representable and are not supported.

Extras.  deg N = 3*deg g - 2 and deg B = deg g, so the construction pins only
a few of the available roots; the rest are determined but not controlled.  The
result CONTAINS the prescription.  `report()` returns the full critical set so
the extras are visible rather than implicit.

Stiffness gate.  `depth_gauge_floor` (= `2|a*'| / |w1'|`, `charts.kappa_saddle`)
is the dispatcher's gauge -- the quantity `trace_unstable` branches on at
KAPPA_HI.  It is NOT `sounding` (`kappa_spectral_saddle`), which is diagnostic
only, lies at u-inflections, and was measured to miss shallow-water cases
one-sidedly.  Gate designs on the depth gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import _poly as P
from . import charts, model, sturm
from ._poly import Poly


# --------------------------------------------------------------------- #
# exact linear algebra over Q                                           #
# --------------------------------------------------------------------- #

def null_space(rows: list[list[Fraction]], ncols: int) -> list[list[Fraction]]:
    """Exact basis for the null space of the matrix with the given rows."""
    mat = [list(r) for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        piv = next((k for k in range(r, len(mat)) if mat[k][c] != 0), None)
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        inv = Fraction(1) / mat[r][c]
        mat[r] = [v * inv for v in mat[r]]
        for k in range(len(mat)):
            if k != r and mat[k][c] != 0:
                fac = mat[k][c]
                mat[k] = [a - fac * b for a, b in zip(mat[k], mat[r])]
        pivots.append(c)
        r += 1
        if r == len(mat):
            break

    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -mat[i][fc]
        basis.append(v)
    return basis


# --------------------------------------------------------------------- #
# the construction                                                      #
# --------------------------------------------------------------------- #

def _condition_row(kind: str, b: Fraction, gpoly: Poly, mu, deg_f: int,
                   A: Poly, Ap: Poly) -> list[Fraction]:
    """One linear condition on f, as a row of coefficients.

    B_j = g_j * sum_i f_i mu_{i+j}, so any linear functional of B is a linear
    functional of f.  For the N family use N = sum_j B_j * Pj with

        Pj(b) = A'(b) b^j - 2 j A(b) b^{j-1}

    which is just A'B - 2B'A collected by B's coefficients.
    """
    dg = P.degree(gpoly)
    weight: list[Fraction] = []
    for j in range(dg + 1):
        if kind == "B":
            wj = b ** j
        else:
            term = P.eval_at(Ap, b) * (b ** j)
            if j > 0:
                term -= Fraction(2 * j) * P.eval_at(A, b) * (b ** (j - 1))
            wj = term
        weight.append(gpoly[j] * wj)
    return [sum((weight[j] * mu[i + j] for j in range(dg + 1)), Fraction(0))
            for i in range(deg_f + 1)]


@dataclass(frozen=True)
class Design:
    model: model.Model
    f: Poly
    prescribed: tuple[Fraction, ...]
    families: tuple[str, ...]
    freedom: int          # dimension of the solution space


def design(prescribed, gpoly, mu, deg_f: int | None = None,
           families=None, combo=None) -> Design:
    """Build a model whose critical set contains `prescribed`.

    `gpoly` and `mu` are fixed by the caller (they determine A); `f` is solved
    for.  `families[k]` selects whether prescribed[k] is imposed as a root of
    "N" (default, the larger family) or of "B".

    `deg_f` defaults to one more than the number of conditions, the smallest
    degree guaranteeing a nonzero solution.  `combo` selects which null vector
    to return when the solution space has dimension > 1 (default: the sum of
    the basis, which avoids the sparse axis-aligned degenerate cases).
    """
    pres = tuple(P.as_fraction(b) for b in prescribed)
    fams = tuple(families) if families is not None else ("N",) * len(pres)
    if len(fams) != len(pres):
        raise ValueError("families must match prescribed in length")
    if any(k not in ("N", "B") for k in fams):
        raise ValueError("family must be 'N' or 'B'")

    g = P.poly(gpoly)
    if not g:
        raise ValueError("g must not be identically zero")
    if g[0] == 0:
        # A(b) = E[g(bX)^2] is > 0 for every b != 0 whenever g is nonzero, but
        # A(0) = g_0^2 mu_0.  A vanishing constant term puts a pole of
        # a* = B/A at the origin and the model is not well posed there.
        raise ValueError("g(0) must be nonzero: A(0) = g_0^2 mu_0 would vanish")
    dg_check = P.degree(g)
    if len(pres) > dg_check:
        # Every condition is a linear functional on B, and B has only
        # deg(g)+1 coefficients.  deg(g)+1 independent conditions force
        # B == 0 identically -- a model with no critical points at all, in
        # which the prescription is satisfied vacuously.  Raising deg_f does
        # NOT help; the bottleneck is deg(g).
        raise ValueError(
            f"at most deg(g)={dg_check} points can be prescribed "
            f"(got {len(pres)}); raise the degree of g")
    if deg_f is None:
        deg_f = max(len(pres), P.degree(g)) + 1
    mu = tuple(P.as_fraction(m) for m in mu)
    need = 2 * max(deg_f, P.degree(g)) + 1
    if len(mu) < need:
        raise ValueError(f"need moments mu_0..mu_{need - 1}, got {len(mu)}")

    # A depends only on (g, mu) -- exactly as model.__post_init__ builds it.
    dg = P.degree(g)
    alpha = [Fraction(0)] * (2 * dg + 1)
    for i in range(dg + 1):
        for j in range(dg + 1):
            alpha[i + j] += g[i] * g[j] * mu[i + j]
    A = P.trim(tuple(alpha))
    Ap = P.deriv(A)

    rows = [_condition_row(k, b, g, mu, deg_f, A, Ap)
            for b, k in zip(pres, fams)]
    basis = null_space(rows, deg_f + 1)
    if not basis:
        raise ValueError(
            f"no nonzero f satisfies {len(pres)} conditions at deg_f={deg_f}; "
            "raise deg_f")

    def b_image(fv):
        """The B coefficients a candidate f induces (B_j = g_j sum_i f_i mu_ij)."""
        return tuple(g[j] * sum((fv[i] * mu[i + j] for i in range(deg_f + 1)),
                                Fraction(0)) for j in range(dg + 1))

    if combo is not None:
        cands = [[sum((Fraction(w) * v[i] for w, v in zip(combo, basis)),
                      Fraction(0)) for i in range(deg_f + 1)]]
    else:
        # Prefer the generic combination; fall back to individual basis
        # vectors.  Reject any candidate whose B vanishes identically: it
        # satisfies every condition vacuously and has no critical points.
        cands = [[sum(col, Fraction(0)) for col in zip(*basis)]] + basis

    for vec in cands:
        f = P.trim(tuple(vec))
        if f and any(x != 0 for x in b_image(vec)):
            break
    else:
        raise ValueError(
            "every solution has B identically zero (no critical points); "
            "the prescription is degenerate for this g and mu")

    return Design(model.build(f, g, mu), f, pres, fams, len(basis))


# --------------------------------------------------------------------- #
# depth-gauge gate                                                      #
# --------------------------------------------------------------------- #

def depth_gauge_at(m: model.Model, b, rel_offset: float = 1e-6) -> float:
    """The dispatcher's gauge near b.

    `depth_gauge_floor` is 0/0 AT a critical point, so read just off it, which
    is what `trace_unstable` does (`kap[0] <- kap[1]` on its grid).  Both sides
    are read and the smaller taken, matching the floor's conservative role.
    """
    b = float(b)
    off = max(rel_offset, rel_offset * abs(b))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vals = [float(charts.depth_gauge_floor(m, b + off)),
                float(charts.depth_gauge_floor(m, b - off))]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return min(vals) if vals else float("inf")


def is_shallow(m: model.Model, b) -> bool:
    """Would the engine hand this point to the Hadamard fixed point?"""
    return depth_gauge_at(m, b) >= charts.KAPPA_HI


@dataclass(frozen=True)
class Report:
    realised: tuple[Fraction, ...]      # prescribed points that are exact roots
    missing: tuple[Fraction, ...]       # prescribed but NOT found (a bug)
    critical: tuple[float, ...]         # full enumerated critical set
    extras: tuple[float, ...]           # critical points not prescribed
    gauges: tuple[float, ...]           # depth gauge at each prescribed point
    shallow: tuple[bool, ...]           # gauge >= KAPPA_HI
    morse: bool
    alternates: bool


def report(d: Design, tol: float = 1e-9) -> Report:
    """Containment check plus the depth-gauge gate for every prescribed point.

    Containment (`prescribed subset of enumerated`) catches MISSED critical
    points; the enumeration's own alternation and index certificates catch
    spurious ones.  Neither is two-sided alone.
    """
    e = sturm.enumerate_critical_points(d.model)
    crit = tuple(float(p.b) for p in e.points)

    realised, missing = [], []
    for b in d.prescribed:
        fb = float(b)
        if any(abs(c - fb) <= tol * max(1.0, abs(fb)) for c in crit):
            realised.append(b)
        else:
            missing.append(b)

    extras = tuple(c for c in crit
                   if not any(abs(c - float(b)) <= tol * max(1.0, abs(float(b)))
                              for b in d.prescribed))
    gauges = tuple(depth_gauge_at(d.model, b) for b in d.prescribed)
    return Report(tuple(realised), tuple(missing), crit, extras, gauges,
                  tuple(g >= charts.KAPPA_HI for g in gauges),
                  bool(e.morse), bool(e.alternates))


def stiffness_ladder(radii, gpoly, mu, deg_f: int | None = None, **kw):
    """Design one model per radius; return (radius, Design, Report) triples.

    The intended use: sweep |b| and read the gauge ladder that comes with it.
    Designs that fail to realise their prescription are reported, not hidden.
    """
    out = []
    for r in radii:
        try:
            d = design([r], gpoly, mu, deg_f=deg_f, **kw)
            out.append((r, d, report(d)))
        except Exception as exc:                      # noqa: BLE001
            out.append((r, None, exc))
    return out


# --------------------------------------------------------------------- #
# transition-straddling suite                                           #
# --------------------------------------------------------------------- #
#
# The instrument is stressed by the mild/shallow HANDOFF, not by the magnitude
# of the gauge: past the transition a branch is uniformly shallow, the Hadamard
# fixed point owns all of it, and the seam falls back to roundoff.  So the cases
# worth generating are the ones that cross the boundary.
#
# Screening must be HYSTERETIC.  The engine enters shallow water at KAPPA_HI and
# only leaves below KAPPA_EXIT, so raw crossings of KAPPA_HI do not cause chart
# switches -- counting them predicts nothing (measured: median switches 0 in
# every raw-crossing group).  `hysteretic_zones` reproduces the engine's state
# machine, which is what correlates with the seam.

def branch_span(m: model.Model):
    """(saddle, nearest target minimum) for the first traceable branch, or None."""
    e = sturm.enumerate_critical_points(m)
    if not e.saddles or not e.minima:
        return None
    s = e.saddles[0]
    side = [p for p in e.minima if p.b > s.b] or [p for p in e.minima if p.b < s.b]
    if not side:
        return None
    return s, min(side, key=lambda p: abs(p.b - s.b)), e


def zone_profile(m: model.Model, b0: float, b1: float, n: int = 3001):
    """Depth gauge sampled along the span (endpoints dropped: 0/0 there)."""
    bs = np.linspace(float(b0), float(b1), n)[1:-1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gg = np.asarray(charts.depth_gauge_floor(m, bs), dtype=float)
    ok = np.isfinite(gg)
    return bs[ok], gg[ok]


def hysteretic_zones(gauges, hi: float | None = None,
                     exit_: float | None = None) -> int:
    """Zone transitions the ENGINE would make, hysteresis included."""
    hi = charts.KAPPA_HI if hi is None else hi
    exit_ = charts.KAPPA_EXIT if exit_ is None else exit_
    if len(gauges) == 0:
        return 0
    shallow = gauges[0] >= hi
    n = 0
    for v in gauges[1:]:
        if shallow and v < exit_:
            shallow, n = False, n + 1
        elif not shallow and v >= hi:
            shallow, n = True, n + 1
    return n


@dataclass(frozen=True)
class StraddleCase:
    design: Design
    saddle_b: float
    target_b: float
    predicted_zones: int      # hysteretic transitions along the gauge profile
    band_fraction: float      # fraction of the span inside [KAPPA_EXIT, KAPPA_HI)
    shallow_fraction: float
    actual_zones: int = -1    # from tracing; -1 if not verified
    switches: int = -1
    worst_seam: float = float("nan")
    angle_energy: float = float("nan")
    term: str = ""

    @property
    def mispredicted(self) -> bool:
        """Screening disagreed with the engine.

        The gauge is sampled on a straight b-grid while the engine follows the
        trajectory and may run BACKWARD in b inside a shallow zone, so the
        cheap screen can miss a genuinely straddling case.  The worst case
        found so far (g4, b* = 20480: four zones, seam 3.42, angle energy 32)
        was screened as zero transitions -- so a suite built on prediction
        alone omits exactly the case it most needs.
        """
        return self.actual_zones >= 0 and abs(self.actual_zones - 1
                                              - self.predicted_zones) > 1


def straddle_case(pres, gpoly, mu, **kw) -> StraddleCase | None:
    """Design and screen one candidate; None if it has no traceable branch."""
    d = design(pres, gpoly, mu, **kw)
    sp = branch_span(d.model)
    if sp is None:
        return None
    s, t, _ = sp
    bs, gg = zone_profile(d.model, s.b, t.b)
    if gg.size < 10:
        return None
    band = (gg >= charts.KAPPA_EXIT) & (gg < charts.KAPPA_HI)
    return StraddleCase(d, float(s.b), float(t.b), hysteretic_zones(gg),
                        float(np.mean(band)), float(np.mean(gg >= charts.KAPPA_HI)))


def verify(c: StraddleCase, box=(-1e9, 1e9, -1e9, 1e9)) -> StraddleCase:
    """Trace the case and record what the engine actually did."""
    from dataclasses import replace
    sp = branch_span(c.design.model)
    if sp is None:
        return c
    s, t, _ = sp
    br = charts.trace_unstable(c.design.model, s.b, (t.a, t.b), box=box,
                               ds=abs(t.b - s.b) / 2000.0)
    seams = br.certs.get("seam_residuals", [])
    return replace(
        c,
        actual_zones=len(br.diag.get("zones", [])),
        switches=int(br.diag.get("switches", 0)),
        worst_seam=max((abs(float(x)) for x in seams), default=0.0),
        angle_energy=abs(float(br.certs.get("angle_energy") or 0.0)),
        term=br.term)


def straddling_suite(gpoly, mu, radii=None, min_zones: int = 1,
                     limit: int | None = None,
                     verify_all: bool = False) -> list[StraddleCase]:
    """Cases whose branch crosses the mild/shallow boundary, hardest first.

    With `verify_all`, every candidate is traced and the ranking uses what the
    engine ACTUALLY did (zones, then seam).  That is the honest mode: the cheap
    screen alone omits the worst known case -- see `StraddleCase.mispredicted`.
    Without it, ranking is by predicted transitions, then by how evenly the span
    splits between zones (a 50/50 branch spends the most length near the
    handoff, which is where the seam is paid).
    """
    if radii is None:
        radii = [Fraction(2) ** k for k in range(1, 20)]
        radii += [Fraction(3) * Fraction(2) ** k for k in range(1, 17)]
        radii += [Fraction(5) * Fraction(2) ** k for k in range(1, 15)]
    seen, uniq = set(), []
    for r in radii:                       # dedupe: 20480 = 5*2^12 = 2^12 * 5
        rf = Fraction(r)
        if rf not in seen:
            seen.add(rf)
            uniq.append(rf)
    radii = uniq

    out: list[StraddleCase] = []
    for r in radii:
        try:
            c = straddle_case([r], gpoly, mu)
        except Exception:                                   # noqa: BLE001
            continue
        if c is None:
            continue
        if verify_all:
            try:
                c = verify(c)
            except Exception:                               # noqa: BLE001
                continue
            # keep anything the ENGINE found interesting, whatever the screen said
            if max(c.predicted_zones, c.actual_zones - 1) < min_zones:
                continue
        elif c.predicted_zones < min_zones:
            continue
        out.append(c)

    if verify_all:
        out.sort(key=lambda c: (-(c.actual_zones), -c.worst_seam))
    else:
        out.sort(key=lambda c: (-c.predicted_zones,
                                -min(c.shallow_fraction, 1.0 - c.shallow_fraction)))
    return out[:limit] if limit else out
