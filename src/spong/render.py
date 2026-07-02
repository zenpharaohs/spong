"""Rendering: pure-stdlib SVG.  No plotting-suite black boxes.

The contour layer is spong's own (the user's design decision: "we will
end up rolling our own contour plot"): level curves of L are CLOSED FORM,
a_±(b; c) = a*(b) ± √((c − u(b))/A(b)), so contours are exact polylines on
a dense b-grid — no marching squares, no grid artifacts, no fcontour.

House palette (mse-bundle heritage): light-gray contours, gold backbone,
green unstable branches, red separatrices; minima as filled dots, N-root
saddles as triangles, B-root saddles as open diamonds (they are saddles
by Theorem 2's B-root clause and deserve their own glyph).

Output is vector SVG: crisp at any zoom (the polylines beneath are the
certified, chord-uniform ones from spong.charts), viewable in any
browser, and — per the corpus's merch clause — silkscreen-ready.
"""

from __future__ import annotations

import numpy as np

from .model import Model
from .portrait import Portrait

PALETTE = {
    "contour": "#c8c8c8",
    "backbone": "#e6a817",
    "unstable": "#00b400",
    "stable": "#d40000",
    "min_fill": "#111111",
    "saddle_fill": "#111111",
    "bsaddle_stroke": "#111111",
    "text": "#333333",
    "rim": "#888888",
}


def _fmt(x: float) -> str:
    return f"{x:.6g}"


class _SVG:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>']

    def polyline(self, pts, color, width=1.0, opacity=1.0):
        if len(pts) < 2:
            return
        d = "M" + " L".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')

    def circle(self, x, y, r, fill, stroke="white", sw=1.2):
        self.parts.append(
            f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')

    def triangle(self, x, y, r, fill, stroke="white", sw=1.2):
        p = [(x, y - r), (x - 0.87 * r, y + 0.5 * r),
             (x + 0.87 * r, y + 0.5 * r)]
        d = "M" + " L".join(f"{_fmt(px)},{_fmt(py)}" for px, py in p) + " Z"
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{sw}"/>')

    def diamond(self, x, y, r, stroke, sw=1.6):
        p = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        d = "M" + " L".join(f"{_fmt(px)},{_fmt(py)}" for px, py in p) + " Z"
        self.parts.append(f'<path d="{d}" fill="white" stroke="{stroke}" '
                          f'stroke-width="{sw}"/>')

    def text(self, x, y, s, size=11, color=PALETTE["text"], anchor="start"):
        self.parts.append(
            f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-family="Helvetica" '
            f'font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}">{s}</text>')

    def to_string(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def _mapper(view, width, height, pad):
    a_lo, a_hi, b_lo, b_hi = view
    sx = (width - 2 * pad) / (a_hi - a_lo)
    sy = (height - 2 * pad) / (b_hi - b_lo)

    def to_px(a, b):
        return (pad + (a - a_lo) * sx,
                height - pad - (b - b_lo) * sy)   # b increases upward
    return to_px


def _clip_runs(X, Y, view):
    """Split (X, Y) into runs of points inside the view; NaN-safe."""
    a_lo, a_hi, b_lo, b_hi = view
    ok = (np.isfinite(X) & np.isfinite(Y)
          & (X >= a_lo) & (X <= a_hi) & (Y >= b_lo) & (Y <= b_hi))
    runs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    if start is not None and len(ok) - start >= 2:
        runs.append((start, len(ok)))
    return runs


def contour_levels(m: Model, view, n_levels: int = 48):
    """Quantile level ladder: levels at quantiles of L sampled over the
    view, so contour density follows where the landscape actually lives
    (a geometric ladder starves the mid-range when corner values are
    astronomically larger than the valley floor)."""
    a = np.linspace(view[0], view[1], 160)
    b = np.linspace(view[2], view[3], 160)
    A, B = np.meshgrid(a, b)
    Lv = m.L(A, B).ravel()
    Lv = Lv[np.isfinite(Lv)]
    q = np.linspace(0.02, 0.995, n_levels)
    return np.unique(np.quantile(Lv, q))


def plane_view(p: Portrait, view=None, width=1200, height=900,
               n_levels: int = 48, n_grid: int = 1501, title=None) -> str:
    """The portrait in the (a, b) plane.  Returns SVG text.

    Curves are computed on the compute box and CLIPPED to the view
    (§8b: clipped, never truncated mid-chart).
    """
    m = p.model
    view = tuple(view if view is not None else (p.view or p.box))
    to_px = _mapper(view, width, height, pad=40)
    svg = _SVG(width, height)

    # ---- contour layer: exact closed-form level curves ----------------- #
    b = np.linspace(view[2], view[3], n_grid)
    for c in contour_levels(m, view, n_levels):
        lo_arm, hi_arm = m.level_curve(c, b)
        for arm in (lo_arm, hi_arm):
            for i0, i1 in _clip_runs(arm, b, view):
                pts = [to_px(arm[k], b[k]) for k in range(i0, i1)]
                svg.polyline(pts, PALETTE["contour"], 0.7)

    # ---- backbone ------------------------------------------------------ #
    bb = m.a_star(b)
    for i0, i1 in _clip_runs(bb, b, view):
        pts = [to_px(bb[k], b[k]) for k in range(i0, i1)]
        svg.polyline(pts, PALETTE["backbone"], 1.6)

    # ---- branches ------------------------------------------------------ #
    for br in p.branches:
        color = PALETTE["unstable"] if br.kind == "unstable" \
            else PALETTE["stable"]
        X, Y = br.Y[:, 0], br.Y[:, 1]
        for i0, i1 in _clip_runs(X, Y, view):
            pts = [to_px(X[k], Y[k]) for k in range(i0, i1)]
            svg.polyline(pts, color, 2.2)

    # ---- critical points ------------------------------------------------ #
    for q in p.enumeration.points:
        if not (view[0] <= q.a <= view[1] and view[2] <= q.b <= view[3]):
            continue
        x, y = to_px(q.a, q.b)
        if q.kind == "min":
            svg.circle(x, y, 5, PALETTE["min_fill"])
        elif q.source == "B":
            svg.diamond(x, y, 6, PALETTE["bsaddle_stroke"])
        else:
            svg.triangle(x, y, 6, PALETTE["saddle_fill"])

    if title:
        svg.text(width / 2, 24, title, size=15, anchor="middle")
    led = p.ledger.get("summary", {})
    svg.text(42, height - 14,
             f"certificates: E_max = {led.get('worst_angle_energy', 0):.2e}"
             f" | index balanced: {led.get('balanced')}"
             f" | max turn = {led.get('worst_max_turn_deg', 0):.3f}°",
             size=10)
    return svg.to_string()


def disk_view(p: Portrait, width=900, height=900, n_levels=24,
              title=None) -> str:
    """The global portrait on the Poincaré disk: z → z/√(1 + |z|²)
    (the canonical Poincaré-compactification projection: gnomonic to the
    hemisphere, viewed from above).

    Nothing is off-canvas, by theorem: separatrices run to the rim
    diagonals b = ±√d_eff·a; unbounded unstable branches to the b-poles.

    GEOMETRY CAVEAT (not a defect — a theorem): no bounded single-chart
    picture of the whole plane can be conformal (Liouville / Riemann
    mapping: the plane is not conformally a disk), so angles — in
    particular the orthogonality of invariant manifolds to level curves —
    are NOT preserved here.  They are preserved exactly in the plane
    view, and would be preserved in a conformal two-chart atlas (finite
    chart = the plane view; rim chart = the inversion ζ = 1/z, conformal
    onto a disk with the equatorial equilibria finite).  The ζ-chart view
    is on the render backlog.
    """
    from .atlas import effective_degree
    m = p.model
    pad = 30
    R = min(width, height) / 2 - pad
    cx, cy = width / 2, height / 2

    def to_px(a, b):
        f = 1.0 / np.sqrt(1.0 + a * a + b * b)
        return (cx + a * f * R, cy - b * f * R)

    svg = _SVG(width, height)
    svg.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
        f'stroke="{PALETTE["rim"]}" stroke-width="1.5"/>')

    box = p.box
    b = np.linspace(box[2], box[3], 2001)

    for c in contour_levels(m, box, n_levels):
        lo_arm, hi_arm = m.level_curve(c, b)
        for arm in (lo_arm, hi_arm):
            ok = np.isfinite(arm)
            if ok.sum() < 2:
                continue
            idx = np.where(ok)[0]
            pts = [to_px(arm[k], b[k]) for k in idx]
            svg.polyline(pts, PALETTE["contour"], 0.6)

    bb = m.a_star(b)
    svg.polyline([to_px(bb[k], b[k]) for k in range(len(b))],
                 PALETTE["backbone"], 1.4)

    for br in p.branches:
        color = PALETTE["unstable"] if br.kind == "unstable" \
            else PALETTE["stable"]
        pts = [to_px(x, y) for x, y in br.Y if np.isfinite(x)]
        svg.polyline(pts, color, 2.0)

    # rim equilibria: diagonals at slope ±√d_eff and the b-poles
    d_eff = effective_degree(m)
    s = np.sqrt(d_eff)
    for (da, db) in [(1, s), (1, -s), (-1, s), (-1, -s)]:
        n = np.hypot(da, db)
        svg.circle(cx + da / n * R, cy - db / n * R, 4.5, "white",
                   stroke=PALETTE["stable"], sw=2.0)
    for db in (1, -1):
        svg.circle(cx, cy - db * R, 4.5, "white",
                   stroke=PALETTE["backbone"], sw=2.0)

    for q in p.enumeration.points:
        x, y = to_px(q.a, q.b)
        if q.kind == "min":
            svg.circle(x, y, 4.5, PALETTE["min_fill"])
        elif q.source == "B":
            svg.diamond(x, y, 5, PALETTE["bsaddle_stroke"])
        else:
            svg.triangle(x, y, 5, PALETTE["saddle_fill"])

    if title:
        svg.text(width / 2, 22, title, size=14, anchor="middle")
    return svg.to_string()


def save(svg_text: str, path: str) -> str:
    with open(path, "w") as f:
        f.write(svg_text)
    return path
