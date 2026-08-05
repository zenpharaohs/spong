"""Compare the Hilbertian L2 portrait with the quartic L4 oracle.

The L2 panels are production SPONG portraits with the usual certificate
ledger.  The L4 panels are numerical oracles: polynomial coefficients come
from exact rational moments, hyperbolic critical points are refined by Newton,
and separatrices are traced with high-accuracy stiff implicit integration, but
no claim of complete critical-
point enumeration or topological certification is made.

Usage:
  PYTHONPATH=src:. python3 demos/l2_l4_phase_portraits.py
  open out/l2_l4_phase_portraits/comparison.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from html import escape
import json
from pathlib import Path

import numpy as np

from spong import model, portrait


F = (Fraction(1), Fraction(1), Fraction(1))
G = F


def _mul(*polynomials):
    out = np.array([1.0])
    for polynomial in polynomials:
        out = np.convolve(out, np.asarray(polynomial, dtype=float))
    return out


def _sym2_eigenpair(H, smaller=True):
    """Closed-form eigenpair of a real symmetric 2x2 matrix."""
    aa, cc, dd = float(H[0, 0]), float(H[0, 1]), float(H[1, 1])
    radius = float(np.hypot(aa-dd, 2.0*cc))
    eigenvalue = 0.5*(aa+dd-radius if smaller else aa+dd+radius)
    first = np.array([cc, eigenvalue-aa])
    second = np.array([eigenvalue-dd, cc])
    if np.hypot(first[0], first[1]) >= np.hypot(second[0], second[1]):
        vector = first
    elif np.hypot(second[0], second[1]) > 0.0:
        vector = second
    else:
        vector = np.array([1.0, 0.0]) if (
            (aa <= dd) == smaller) else np.array([0.0, 1.0])
    vector /= float(np.hypot(vector[0], vector[1]))
    return eigenvalue, vector


class QuarticModel:
    """Binary64 evaluator of an exactly specified polynomial L4 loss."""

    def __init__(self, f, g, moments):
        self.f_exact = tuple(Fraction(x) for x in f)
        self.g_exact = tuple(Fraction(x) for x in g)
        self.mu_exact = tuple(Fraction(x) for x in moments)
        self.degree = max(len(self.f_exact), len(self.g_exact))-1
        need = 4*self.degree+1
        if len(self.mu_exact) < need:
            raise ValueError(f"quartic loss needs moments mu_0..mu_{need-1}")
        self.f = np.asarray([float(x) for x in self.f_exact], dtype=float)
        self.g = np.asarray([float(x) for x in self.g_exact], dtype=float)
        self.mu = np.asarray([float(x) for x in self.mu_exact], dtype=float)

    def _expect(self, polynomial):
        p = np.asarray(polynomial, dtype=float)
        return float(p @ self.mu[:len(p)])

    def _pieces(self, a, b):
        activation = np.array([
            self.g[j]*b**j for j in range(len(self.g))])
        activation_p = np.array([
            0.0 if j == 0 else j*self.g[j]*b**(j-1)
            for j in range(len(self.g))])
        activation_pp = np.array([
            0.0 if j < 2 else j*(j-1)*self.g[j]*b**(j-2)
            for j in range(len(self.g))])
        n = max(len(self.f), len(activation))
        target = np.pad(self.f, (0, n-len(self.f)))
        activation = np.pad(activation, (0, n-len(activation)))
        return target-a*activation, activation, activation_p, activation_pp

    def values(self, a, b):
        a, b = float(a), float(b)
        residual, activation, activation_p, activation_pp = \
            self._pieces(a, b)
        r2 = _mul(residual, residual)
        r3 = _mul(r2, residual)
        loss = self._expect(_mul(r2, r2))
        grad_a = -4.0*self._expect(_mul(r3, activation))
        grad_b = -4.0*a*self._expect(_mul(r3, activation_p))
        h_aa = 12.0*self._expect(_mul(r2, activation, activation))
        h_ab = (12.0*a*self._expect(
            _mul(r2, activation, activation_p))
            - 4.0*self._expect(_mul(r3, activation_p)))
        h_bb = (12.0*a*a*self._expect(
            _mul(r2, activation_p, activation_p))
            - 4.0*a*self._expect(_mul(r3, activation_pp)))
        return (loss, np.array([grad_a, grad_b]),
                np.array([[h_aa, h_ab], [h_ab, h_bb]]))

    def L(self, a, b):
        return self.values(a, b)[0]

    def gradL(self, a, b):
        return self.values(a, b)[1]

    def hessL(self, a, b):
        return self.values(a, b)[2]

    def a_star(self, b):
        """Unique conditional L4 minimizer in a, by safeguarded Newton."""
        b = float(b)
        a = 1.0
        for _ in range(30):
            _loss, gradient, hessian = self.values(a, b)
            if abs(gradient[0]) <= 2e-14*(1.0+abs(a)):
                return a
            if hessian[0, 0] <= 0.0 or not np.isfinite(hessian[0, 0]):
                break
            candidate = a-gradient[0]/hessian[0, 0]
            if not np.isfinite(candidate):
                break
            a = candidate
        lo, hi = -1.0, 1.0
        while self.gradL(lo, b)[0] > 0.0:
            lo *= 2.0
        while self.gradL(hi, b)[0] < 0.0:
            hi = 2.0*hi+1.0
        for _ in range(100):
            mid = 0.5*(lo+hi)
            if self.gradL(mid, b)[0] < 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5*(lo+hi)

    def normalized_field(self, _time, state, ascent=False):
        gradient = self.gradL(*state)
        scale = float(np.sqrt(1.0+gradient @ gradient))
        field = -gradient/scale
        return -field if ascent else field

    def normalized_jacobian(self, _time, state, ascent=False):
        gradient = self.gradL(*state)
        hessian = self.hessL(*state)
        scale = float(np.sqrt(1.0+gradient @ gradient))
        hg = hessian @ gradient
        jacobian = -hessian/scale + np.outer(gradient, hg)/(scale**3)
        return -jacobian if ascent else jacobian

    def geometric_field(self, _time, state, ascent=False):
        """Unit-speed reparametrization of the gradient curve."""
        gradient = self.gradL(*state)
        scale = float(np.hypot(gradient[0], gradient[1]))
        if scale == 0.0:
            raise FloatingPointError("geometric field reached a critical point")
        field = -gradient/scale
        return -field if ascent else field

    def geometric_jacobian(self, _time, state, ascent=False):
        gradient = self.gradL(*state)
        hessian = self.hessL(*state)
        scale = float(np.hypot(gradient[0], gradient[1]))
        if scale == 0.0:
            raise FloatingPointError("geometric field reached a critical point")
        hg = hessian @ gradient
        jacobian = -hessian/scale + np.outer(gradient, hg)/(scale**3)
        return -jacobian if ascent else jacobian


@dataclass(frozen=True)
class OracleCriticalPoint:
    a: float
    b: float
    kind: str
    loss: float
    hessian: np.ndarray


@dataclass
class OracleBranch:
    kind: str
    Y: np.ndarray
    term: str


def _newton_critical(q, seed, box):
    z = np.asarray(seed, dtype=float).copy()
    for _ in range(60):
        loss, gradient, hessian = q.values(*z)
        scale_g = max(1.0, abs(loss), float(np.max(np.abs(hessian))))
        if np.max(np.abs(gradient)) <= 2e-12*scale_g:
            return z
        aa, cc, dd = hessian[0, 0], hessian[0, 1], hessian[1, 1]
        det = aa*dd-cc*cc
        row_product = max(abs(aa), abs(cc))*max(abs(cc), abs(dd))
        if row_product == 0.0 or abs(det) < 1e-12*row_product:
            return None
        correction = np.array([
            (dd*gradient[0]-cc*gradient[1])/det,
            (aa*gradient[1]-cc*gradient[0])/det])
        phi = float(gradient @ gradient)
        alpha = 1.0
        while alpha >= 2.0**-14:
            candidate = z-alpha*correction
            candidate_gradient = q.gradL(*candidate)
            if (np.all(np.isfinite(candidate))
                    and candidate_gradient @ candidate_gradient
                    <= phi*(1.0-1e-4*alpha)):
                break
            alpha *= 0.5
        if alpha < 2.0**-14:
            return None
        z = candidate
        if not (box[0]-20 <= z[0] <= box[1]+20
                and box[2]-50 <= z[1] <= box[3]+50):
            return None
    return None


def quartic_critical_points(q, p2_points, view):
    """Numerical-oracle critical set; no completeness claim is attached."""
    seeds = [(point.a, point.b) for point in p2_points]
    seeds.extend((a, b)
                 for a in np.linspace(view[0], view[1], 13)
                 for b in np.linspace(view[2], view[3], 17))
    roots = []
    for seed in seeds:
        root = _newton_critical(q, seed, view)
        if root is None:
            continue
        if np.hypot(root[0]-1.0, root[1]-1.0) < 2e-2:
            continue
        if not (view[0] <= root[0] <= view[1]
                and view[2] <= root[1] <= view[3]):
            continue
        loss, gradient, hessian = q.values(*root)
        det = hessian[0, 0]*hessian[1, 1]-hessian[0, 1]**2
        row_product = max(abs(hessian[0, 0]), abs(hessian[0, 1]))*max(
            abs(hessian[1, 0]), abs(hessian[1, 1]))
        if (np.max(np.abs(gradient)) > 2e-9*max(1.0, abs(loss))
                or row_product == 0.0 or abs(det) < 1e-8*row_product):
            continue
        if any(np.hypot(*(root-other)) < 2e-5 for other in roots):
            continue
        roots.append(root)
    roots.sort(key=lambda z: z[1])
    points = []
    for root in roots:
        loss, _gradient, hessian = q.values(*root)
        det = hessian[0, 0]*hessian[1, 1]-hessian[0, 1]**2
        trace = hessian[0, 0]+hessian[1, 1]
        kind = "saddle" if det < 0.0 else "min" if trace > 0.0 else "degenerate"
        points.append(OracleCriticalPoint(
            float(root[0]), float(root[1]), kind, loss, hessian))
    # f=g gives this exact zero-residual point.  Its Hessian vanishes for L4,
    # so Newton cannot isolate it and must not be asked to pretend otherwise.
    points.append(OracleCriticalPoint(
        1.0, 1.0, "degenerate-min", 0.0, np.zeros((2, 2))))
    return tuple(sorted(points, key=lambda point: point.b))


def _inside(point, view):
    return (view[0] <= point[0] <= view[1]
            and view[2] <= point[1] <= view[3])


def _trace_oracle_branch(q, start, view, critical_points, ascent,
                         max_steps=2400):
    from scipy.integrate import solve_ivp

    span = max(view[1]-view[0], view[3]-view[2])
    step_size = min(0.022, span/700.0)
    capture = 2.5*step_size
    state = np.asarray(start, dtype=float)
    destinations = ([] if ascent else [p for p in critical_points
                    if p.kind in ("min", "degenerate-min")])

    def leave_view(_time, z):
        return min(z[0]-view[0], view[1]-z[0],
                   z[1]-view[2], view[3]-z[1])

    leave_view.terminal = True
    leave_view.direction = -1
    events = [leave_view]
    for destination in destinations:
        def capture_destination(_time, z, p=destination):
            return np.hypot(z[0]-p.a, z[1]-p.b)-capture
        capture_destination.terminal = True
        capture_destination.direction = -1
        events.append(capture_destination)
    solution = solve_ivp(
        lambda t, y: q.geometric_field(t, y, ascent=ascent),
        (0.0, min(max_steps*step_size, 4.0*span+4.0)), state,
        method="Radau", jac=lambda t, y: q.geometric_jacobian(
            t, y, ascent=ascent), rtol=2e-10, atol=1e-12,
        max_step=step_size, events=events)
    points = solution.y.T
    term = "budget" if solution.success else "solver-refusal"
    if solution.t_events[0].size:
        term = "view-exit"
    else:
        for destination, hits in zip(destinations, solution.t_events[1:]):
            if hits.size:
                points = np.vstack((points,
                                    np.array([destination.a, destination.b])))
                term = "capture"
                break
    return points, term


def quartic_branches(q, critical_points, view):
    branches = []
    span = max(view[1]-view[0], view[3]-view[2])
    # This is a visualization oracle rather than a certified local chart.
    # Start far enough along the exact eigendirection that the unit-gradient
    # reparametrization is numerically regular; the omitted segment is below a
    # display pixel in the chosen views and is restored explicitly below.
    offset = 2e-3*span
    for point in critical_points:
        if point.kind != "saddle":
            continue
        _negative, unstable = _sym2_eigenpair(point.hessian, smaller=True)
        _positive, stable = _sym2_eigenpair(point.hessian, smaller=False)
        center = np.array([point.a, point.b])
        for sign in (-1.0, 1.0):
            Y, term = _trace_oracle_branch(
                q, center+sign*offset*unstable, view, critical_points,
                ascent=False)
            Y = np.vstack((center, Y))
            branches.append(OracleBranch("unstable", Y, term))
            Y, term = _trace_oracle_branch(
                q, center+sign*offset*stable, view, critical_points,
                ascent=True)
            Y = np.vstack((center, Y))
            branches.append(OracleBranch("stable", Y, term))
    return branches


def _root_on_interval(loss, level, b, lo, hi):
    flo, fhi = loss(lo, b)-level, loss(hi, b)-level
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo*fhi > 0.0:
        return np.nan
    for _ in range(58):
        mid = 0.5*(lo+hi)
        fm = loss(mid, b)-level
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5*(lo+hi)


def _contours(loss, backbone, view, critical_losses):
    a_grid = np.linspace(view[0], view[1], 41)
    b_grid = np.linspace(view[2], view[3], 51)
    sampled = np.array([[loss(a, b) for a in a_grid] for b in b_grid])
    finite = sampled[np.isfinite(sampled)]
    ceiling = float(np.quantile(finite, 0.82))
    floor = max(ceiling*2e-5, 1e-10)
    levels = list(np.geomspace(floor, ceiling, 13))
    levels.extend(x for x in critical_losses if floor < x < ceiling)
    levels = sorted(set(round(float(x), 13) for x in levels))
    bs = np.linspace(view[2], view[3], 301)
    stars = np.array([backbone(b) for b in bs])
    curves = []
    for level in levels:
        low = np.full(len(bs), np.nan)
        high = np.full(len(bs), np.nan)
        for i, (b, star) in enumerate(zip(bs, stars)):
            split = float(np.clip(star, view[0], view[1]))
            intervals = [(view[0], split), (split, view[1])]
            values = [_root_on_interval(loss, level, b, *interval)
                      if interval[1] > interval[0] else np.nan
                      for interval in intervals]
            low[i], high[i] = values
        curves.extend(((low, bs), (high, bs)))
    return curves, (stars, bs)


def _runs(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    start = None
    for i, flag in enumerate(good):
        if flag and start is None:
            start = i
        if (not flag or i == len(good)-1) and start is not None:
            stop = i if not flag else i+1
            if stop-start >= 2:
                yield np.column_stack((a[start:stop], b[start:stop]))
            start = None


def _decimate(points, limit):
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points)-1, limit).round().astype(int)
    return points[np.unique(indices)]


def _path(points, mapper, color, width, opacity=1.0, limit=260):
    points = _decimate(np.asarray(points), limit)
    pixels = [mapper(*point) for point in points if np.all(np.isfinite(point))]
    if len(pixels) < 2:
        return ""
    data = "M"+" L".join(f"{x:.5g},{y:.5g}" for x, y in pixels)
    return (f'<path d="{data}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            'stroke-linejoin="round" stroke-linecap="round" '
            'vector-effect="non-scaling-stroke"/>')


def _panel(title, subtitle, loss, backbone, critical_points, branches, view,
           certified, width=560, height=430):
    pad = 44
    sx = (width-2*pad)/(view[1]-view[0])
    sy = (height-2*pad)/(view[3]-view[2])
    mapper = lambda a, b: (pad+(a-view[0])*sx,
                           height-pad-(b-view[2])*sy)
    clip_id = "plot-"+str(sum((i+1)*ord(char)
                              for i, char in enumerate(title)))
    critical_losses = [point.loss for point in critical_points]
    contours, backbone_curve = _contours(loss, backbone, view, critical_losses)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}: {escape(subtitle)}">',
        f'<title>{escape(title)}</title>',
        f'<desc>{escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="var(--card)"/>',
        f'<defs><clipPath id="{clip_id}"><rect x="{pad}" y="{pad}" '
        f'width="{width-2*pad}" height="{height-2*pad}"/>'
        '</clipPath></defs>',
        f'<g clip-path="url(#{clip_id})">']
    for aa, bb in contours:
        for run in _runs(aa, bb):
            parts.append(_path(run, mapper, "var(--border)", 0.7, .72))
    for run in _runs(*backbone_curve):
        parts.append(_path(run, mapper, "var(--viz-series-3)", 1.4, .9))
    for branch in branches:
        color = ("var(--viz-series-2)" if branch.kind == "unstable"
                 else "var(--viz-series-1)")
        parts.append(_path(branch.Y, mapper, color, 1.65, .95))
    for point in critical_points:
        x, y = mapper(point.a, point.b)
        if point.kind == "saddle":
            parts.append(
                f'<path d="M{x:.5g},{y-5:.5g} L{x-4.5:.5g},{y+3.5:.5g} '
                f'L{x+4.5:.5g},{y+3.5:.5g} Z" fill="var(--foreground)"/>')
        elif point.kind == "degenerate-min":
            parts.append(
                f'<rect x="{x-4:.5g}" y="{y-4:.5g}" width="8" height="8" '
                'fill="var(--card)" stroke="var(--viz-series-4)" '
                'stroke-width="2"/>')
        else:
            parts.append(
                f'<circle cx="{x:.5g}" cy="{y:.5g}" r="4" '
                'fill="var(--foreground)"/>')
    parts.append('</g>')
    parts.append(
        f'<rect x="{pad}" y="{pad}" width="{width-2*pad}" '
        f'height="{height-2*pad}" fill="none" stroke="var(--border)"/>')
    for value, x in ((view[0], pad), (view[1], width-pad)):
        parts.append(
            f'<text x="{x}" y="{height-18}" text-anchor="middle" '
            f'fill="var(--muted-foreground)" font-size="11">{value:g}</text>')
    for value, y in ((view[2], height-pad), (view[3], pad)):
        parts.append(
            f'<text x="{pad-8}" y="{y+4}" text-anchor="end" '
            f'fill="var(--muted-foreground)" font-size="11">{value:g}</text>')
    parts.extend([
        f'<text x="{width/2}" y="{height-3}" text-anchor="middle" '
        'fill="var(--foreground)" font-size="12">a</text>',
        f'<text x="12" y="{height/2}" text-anchor="middle" '
        'fill="var(--foreground)" font-size="12" '
        f'transform="rotate(-90 12 {height/2})">b</text>',
        f'<text x="{pad+4}" y="{pad+16}" fill="var(--foreground)" '
        f'font-size="13" font-weight="500">{escape(title)}</text>',
        f'<text x="{pad+4}" y="{pad+32}" fill="var(--muted-foreground)" '
        f'font-size="11">{escape(subtitle)}</text>',
        f'<text x="{width-pad-4}" y="{pad+16}" text-anchor="end" '
        f'fill="var(--muted-foreground)" font-size="11">'
        f'{"CERTIFIED" if certified else "NUMERICAL ORACLE"}</text>',
        '</svg>'])
    return "\n".join(parts)


def _p2_branches(p):
    return [OracleBranch(branch.kind, np.asarray(branch.Y), branch.term)
            for branch in p.branches]


def _p2_points(p):
    return tuple(OracleCriticalPoint(
        point.a, point.b, point.kind, float(p.model.L(point.a, point.b)),
        p.model.hessL(point.a, point.b))
        for point in p.enumeration.points)


def _report_point(point):
    return {"a": point.a, "b": point.b, "kind": point.kind,
            "loss": point.loss}


def build_comparison():
    cases = (
        ("U(0,1)", model.moments_uniform01(17),
         (-0.5, 2.7, -12.0, 3.2)),
        ("N(0,1)", model.moments_normal01(17),
         (0.4, 3.25, -1.25, 1.75)),
    )
    panels = []
    report_cases = []
    for name, moments, view in cases:
        p2_model = model.build(F, G, moments[:5])
        p2 = portrait.certified_compute(p2_model, view=view)
        p2_points = _p2_points(p2)
        p4_model = QuarticModel(F, G, moments[:9])
        p4_points = quartic_critical_points(
            p4_model, p2.enumeration.points, view)
        p4_branches = quartic_branches(p4_model, p4_points, view)
        p2_status = p2.ledger["topology"]["status"]
        panels.extend((
            _panel(
                f"{name} · p=2", "quadratic fibers · hyperbolic exact fit",
                p2_model.L, p2_model.a_star, p2_points,
                _p2_branches(p2), view, p2_status == "certified"),
            _panel(
                f"{name} · p=4",
                "quartic fibers · cubic backbone equation",
                p4_model.L, p4_model.a_star, p4_points,
                p4_branches, view, False),
        ))
        report_cases.append({
            "distribution": name,
            "view": view,
            "p2": {"topology_status": p2_status,
                   "critical_points": [_report_point(x) for x in p2_points]},
            "p4": {"status": "numerical-oracle-not-certified",
                   "critical_points": [_report_point(x) for x in p4_points],
                   "branch_terms": [branch.term for branch in p4_branches]},
        })
    return panels, {"format": "spong-l2-l4-comparison-v1",
                    "cases": report_cases}


def _fragment(panels):
    return f"""<div id="spong-l2-l4-comparison">
<style>
#spong-l2-l4-comparison .l2l4-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
#spong-l2-l4-comparison .l2l4-panel {{ min-width:0; }}
#spong-l2-l4-comparison .l2l4-panel svg {{ display:block; width:100%; height:auto; }}
#spong-l2-l4-comparison .l2l4-legend {{ justify-content:center; margin-top:10px; color:var(--muted-foreground); }}
#spong-l2-l4-comparison .l2l4-legend span {{ display:inline-flex; align-items:center; gap:5px; }}
#spong-l2-l4-comparison .l2l4-legend i {{ width:18px; height:2px; display:inline-block; }}
#spong-l2-l4-comparison .unstable-swatch {{ background:var(--viz-series-2); }}
#spong-l2-l4-comparison .stable-swatch {{ background:var(--viz-series-1); }}
#spong-l2-l4-comparison .backbone-swatch {{ background:var(--viz-series-3); }}
@media (max-width:620px) {{ #spong-l2-l4-comparison .l2l4-grid {{ grid-template-columns:1fr; }} }}
</style>
<div class="l2l4-grid">
{''.join(f'<div class="l2l4-panel">{panel}</div>' for panel in panels)}
</div>
<div class="viz-row l2l4-legend text-small" aria-label="Plot legend">
  <span><i class="unstable-swatch"></i>unstable manifold</span>
  <span><i class="stable-swatch"></i>stable manifold</span>
  <span><i class="backbone-swatch"></i>conditional minimizer</span>
  <span>▲ saddle</span><span>● Morse minimum</span><span>□ quartic minimum</span>
</div>
</div>
"""


def _standalone(fragment):
    return f"""<!doctype html>
<meta charset="utf-8">
<title>SPONG L2 versus L4 phase portraits</title>
<style>
:root {{ --card:#fff; --foreground:#202124; --muted-foreground:#65676b;
--border:#c8cbd0; --viz-series-1:#d43f3a; --viz-series-2:#159947;
--viz-series-3:#d29a16; --viz-series-4:#7357c7; }}
body {{ margin:24px; font-family:system-ui,sans-serif; background:#f7f7f8; color:var(--foreground); }}
main {{ max-width:1240px; margin:auto; }}
.viz-row {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; }}
.text-small {{ font-size:12px; }}
</style>
<main>
<h1>L2 versus L4 phase portraits</h1>
<p>The p=2 panels are certified SPONG portraits. The p=4 panels are a high-accuracy stiff implicit numerical oracle; the square marks the zero-loss quartic minimum with zero Hessian.</p>
{fragment}
</main>
"""


def _comparison_svg(panels):
    colors = {
        "var(--card)": "#ffffff",
        "var(--foreground)": "#202124",
        "var(--muted-foreground)": "#65676b",
        "var(--border)": "#c8cbd0",
        "var(--viz-series-1)": "#d43f3a",
        "var(--viz-series-2)": "#159947",
        "var(--viz-series-3)": "#d29a16",
        "var(--viz-series-4)": "#7357c7",
    }
    positioned = []
    for index, panel in enumerate(panels):
        x, y = 560*(index % 2), 430*(index // 2)
        panel = panel.replace(
            '<svg viewBox="0 0 560 430"',
            f'<svg x="{x}" y="{y}" width="560" height="430" '
            'viewBox="0 0 560 430"', 1)
        for token, color in colors.items():
            panel = panel.replace(token, color)
        positioned.append(panel)
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 1120 860">\n'
            + "\n".join(positioned)+"\n</svg>\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/l2_l4_phase_portraits"))
    parser.add_argument("--fragment-output", type=Path)
    args = parser.parse_args(argv)
    panels, report = build_comparison()
    fragment = _fragment(panels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"comparison.html").write_text(_standalone(fragment))
    (args.output_dir/"comparison.svg").write_text(_comparison_svg(panels))
    (args.output_dir/"comparison.json").write_text(
        json.dumps(report, indent=2)+"\n")
    if args.fragment_output is not None:
        args.fragment_output.parent.mkdir(parents=True, exist_ok=True)
        args.fragment_output.write_text(fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
