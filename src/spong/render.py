"""Rendering: pure-stdlib SVG.  No plotting-suite black boxes.

The contour layer is spong's own (the user's design decision: "we will
end up rolling our own contour plot"): level curves of L are CLOSED FORM,
a_±(b; c) = a*(b) ± √((c − u(b))/A(b)), so contours are exact polylines on
a dense b-grid — no marching squares, no grid artifacts, no fcontour.

House palette (mse-bundle heritage): light-gray contours, gold backbone,
green unstable branches, red separatrices; local nonglobal minima as filled
dots, global minima as open circles, N-root saddles as triangles, B-root
saddles as open diamonds (they are saddles by Theorem 2's B-root clause and
deserve their own glyph).

Output is vector SVG: crisp at any zoom (the polylines beneath are the
certified, chord-uniform ones from spong.charts), viewable in any
browser, and — per the corpus's merch clause — silkscreen-ready.
"""

from __future__ import annotations

from html import escape

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
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'vector-effect="non-scaling-stroke"/>')

    def circle(self, x, y, r, fill, stroke="white", sw=1.2):
        self.parts.append(
            f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" '
            f'vector-effect="non-scaling-stroke"/>')

    def triangle(self, x, y, r, fill, stroke="white", sw=1.2):
        p = [(x, y - r), (x - 0.87 * r, y + 0.5 * r),
             (x + 0.87 * r, y + 0.5 * r)]
        d = "M" + " L".join(f"{_fmt(px)},{_fmt(py)}" for px, py in p) + " Z"
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{sw}" '
                          f'vector-effect="non-scaling-stroke"/>')

    def diamond(self, x, y, r, stroke, sw=1.6):
        p = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        d = "M" + " L".join(f"{_fmt(px)},{_fmt(py)}" for px, py in p) + " Z"
        self.parts.append(f'<path d="{d}" fill="white" stroke="{stroke}" '
                          f'stroke-width="{sw}" '
                          f'vector-effect="non-scaling-stroke"/>')

    def text(self, x, y, s, size=11, color=PALETTE["text"], anchor="start"):
        self.parts.append(
            f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-family="Helvetica" '
            f'font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}">{escape(str(s))}</text>')

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


def _global_minima(p: Portrait) -> set[int]:
    minima = [(i, q) for i, q in enumerate(p.enumeration.points)
              if q.kind == "min"]
    if not minima:
        return set()
    vals = np.array([p.model.L(q.a, q.b) for _, q in minima], dtype=float)
    best = float(np.min(vals))
    tol = 1e-9 * max(1.0, abs(best), float(np.max(np.abs(vals))))
    return {i for (i, _q), val in zip(minima, vals) if val <= best + tol}


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


def _clip_segment_to_view(x0, y0, x1, y1, view):
    """Liang-Barsky clip of one segment against the data-coordinate view."""
    a_lo, a_hi, b_lo, b_hi = view
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - a_lo),
        (dx, a_hi - x0),
        (-dy, y0 - b_lo),
        (dy, b_hi - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    return ((x0 + t0 * dx, y0 + t0 * dy),
            (x0 + t1 * dx, y0 + t1 * dy))


def _clip_polylines(X, Y, view):
    """Return clipped data-coordinate polylines, including boundary crossings.

    The older point-in-view splitter is sufficient for densely sampled
    curves, but a far-field slaved branch can cross a tight inspection view
    with only one sampled vertex inside.  Segment clipping preserves that
    visible chord instead of silently dropping it.
    """
    runs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    last_end = None
    for k in range(len(X) - 1):
        x0, y0 = float(X[k]), float(Y[k])
        x1, y1 = float(X[k + 1]), float(Y[k + 1])
        if not (np.isfinite(x0) and np.isfinite(y0)
                and np.isfinite(x1) and np.isfinite(y1)):
            if len(cur) >= 2:
                runs.append(cur)
            cur, last_end = [], None
            continue
        clipped = _clip_segment_to_view(x0, y0, x1, y1, view)
        if clipped is None:
            if len(cur) >= 2:
                runs.append(cur)
            cur, last_end = [], None
            continue
        p0, p1 = clipped
        if last_end is None or np.hypot(p0[0] - last_end[0],
                                        p0[1] - last_end[1]) > 1e-10:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [p0]
        cur.append(p1)
        last_end = p1
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _adaptive_backbone(m: Model, view, to_px, n_seed: int = 96,
                       max_depth: int = 14, max_chord_px: float = 8.0,
                       max_err_px: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Sample a*(b) adaptively in screen space.

    A uniform b-grid can make the rational backbone look broken in tight
    views near a sharp bend.  Contours can stay on their grid; the backbone is
    a single curve, so a cheap midpoint subdivision gives a smoother
    metrological guide.
    """
    b0, b1 = float(view[2]), float(view[3])
    out: list[tuple[float, float]] = []

    def aval(b):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float(m.a_star(b))

    def flat_enough(a0, b0, a1, b1):
        if not (np.isfinite(a0) and np.isfinite(a1)):
            return True, None
        bm = 0.5 * (b0 + b1)
        am = aval(bm)
        if not np.isfinite(am):
            return True, (am, bm)
        x0, y0 = to_px(a0, b0)
        x1, y1 = to_px(a1, b1)
        xm, ym = to_px(am, bm)
        chord = float(np.hypot(x1 - x0, y1 - y0))
        lx, ly = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        err = float(np.hypot(xm - lx, ym - ly))
        return chord <= max_chord_px and err <= max_err_px, (am, bm)

    def rec(a0, b0, a1, b1, depth):
        ok, mid = flat_enough(a0, b0, a1, b1)
        if ok or depth >= max_depth:
            out.append((a0, b0))
            return
        am, bm = mid
        rec(a0, b0, am, bm, depth + 1)
        rec(am, bm, a1, b1, depth + 1)

    b_seed = np.linspace(b0, b1, max(2, n_seed))
    a_seed = [aval(float(bi)) for bi in b_seed]
    for k in range(len(b_seed) - 1):
        rec(a_seed[k], float(b_seed[k]), a_seed[k + 1], float(b_seed[k + 1]), 0)
    out.append((a_seed[-1], float(b_seed[-1])))
    Y = np.array(out, dtype=float)
    return Y[:, 0], Y[:, 1]


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


def _sample_polyline(Y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(Y) <= max_points:
        idx = np.arange(len(Y))
        return Y, idx
    idx = np.unique(np.linspace(0, len(Y) - 1, max_points).astype(int))
    return Y[idx], idx


def close_unstable_zooms(p: Portrait, n: int = 2, samples: int = 1800,
                         trim_capture_radius: float = 1e-2,
                         margin: float = 0.12) -> list[dict]:
    """Find tight views around close approaches of distinct unstable branches.

    Only branch pairs with the same captured target are considered, and a
    small neighborhood of that target is excluded.  The returned views are
    ordinary plane-view boxes; no branch geometry or stroke width changes.
    """
    candidates = []
    unstable = [(i, br) for i, br in enumerate(p.branches)
                if br.kind == "unstable" and br.term == "capture"
                and br.diag.get("target") is not None and len(br.Y) >= 2]
    for ka, (ia, ba) in enumerate(unstable):
        ta = ba.diag.get("target")
        for ib, bb in unstable[ka + 1:]:
            tb = bb.diag.get("target")
            if abs(ta[0] - tb[0]) > 1e-8 or abs(ta[1] - tb[1]) > 1e-8:
                continue
            Ya = ba.Y[np.hypot(ba.Y[:, 0] - ta[0],
                               ba.Y[:, 1] - ta[1]) > trim_capture_radius]
            Yb = bb.Y[np.hypot(bb.Y[:, 0] - tb[0],
                               bb.Y[:, 1] - tb[1]) > trim_capture_radius]
            if len(Ya) < 2 or len(Yb) < 2:
                continue
            Sa, ia_s = _sample_polyline(Ya, samples)
            Sb, ib_s = _sample_polyline(Yb, samples)
            d = Sa[:, None, :] - Sb[None, :, :]
            d2 = np.sum(d * d, axis=2)
            k = np.unravel_index(int(np.argmin(d2)), d2.shape)
            pa, pb = Sa[k[0]], Sb[k[1]]
            sep = float(np.sqrt(d2[k]))
            center = 0.5 * (pa + pb)
            scale = max(sep * 80.0, 1e-6)
            a_half = max(scale, abs(pa[0] - pb[0]) * 20.0)
            b_half = max(scale, abs(pa[1] - pb[1]) * 20.0)
            a_half *= 1.0 + margin
            b_half *= 1.0 + margin
            view = (float(center[0] - a_half), float(center[0] + a_half),
                    float(center[1] - b_half), float(center[1] + b_half))
            candidates.append({
                "view": view,
                "separation": sep,
                "center": (float(center[0]), float(center[1])),
                "branches": (ia, ib),
                "target": (float(ta[0]), float(ta[1])),
                "sample_indices": (int(ia_s[k[0]]), int(ib_s[k[1]])),
            })
    candidates.sort(key=lambda z: z["separation"])
    return candidates[:n]


def plane_view(p: Portrait, view=None, width=1200, height=900,
               n_levels: int = 48, n_grid: int = 1501, title=None,
               overlays=None) -> str:
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
    bb_a, bb_b = _adaptive_backbone(m, view, to_px)
    for run in _clip_polylines(bb_a, bb_b, view):
        pts = [to_px(x, y) for x, y in run]
        svg.polyline(pts, PALETTE["backbone"], 1.6)

    # ---- branches ------------------------------------------------------ #
    for br in p.branches:
        color = PALETTE["unstable"] if br.kind == "unstable" \
            else PALETTE["stable"]
        X, Y = br.Y[:, 0], br.Y[:, 1]
        for run in _clip_polylines(X, Y, view):
            pts = [to_px(x, y) for x, y in run]
            svg.polyline(pts, color, 2.2)

    # ---- critical points ------------------------------------------------ #
    global_minima = _global_minima(p)
    for qi, q in enumerate(p.enumeration.points):
        if not (view[0] <= q.a <= view[1] and view[2] <= q.b <= view[3]):
            continue
        x, y = to_px(q.a, q.b)
        if qi in global_minima:
            svg.circle(x, y, 6, "white", stroke=PALETTE["min_fill"],
                       sw=1.8)
        elif q.kind == "min":
            svg.circle(x, y, 5, PALETTE["min_fill"])
        elif q.source == "B":
            svg.diamond(x, y, 6, PALETTE["bsaddle_stroke"])
        else:
            svg.triangle(x, y, 6, PALETTE["saddle_fill"])

    # ---- overlays (demo consumers: optimizer trajectories etc.) -------- #
    if overlays:
        ly = 44
        for ov in overlays:
            Yov = np.asarray(ov["Y"], dtype=float)
            X, Yb = Yov[:, 0], Yov[:, 1]
            for i0, i1 in _clip_runs(X, Yb, view):
                pts = [to_px(X[k], Yb[k]) for k in range(i0, i1)]
                svg.polyline(pts, ov.get("color", "#3060ff"),
                             ov.get("width", 1.6),
                             opacity=ov.get("opacity", 0.95))
            if len(Yov) and np.all(np.isfinite(Yov[0])):
                x0, y0 = to_px(Yov[0, 0], Yov[0, 1])
                svg.circle(x0, y0, 3.2, ov.get("color", "#3060ff"))
            if ov.get("label"):
                svg.text(width - 250, ly, ov["label"], size=11,
                         color=ov.get("color", "#3060ff"))
                ly += 15

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
        f'stroke="{PALETTE["rim"]}" stroke-width="1.5" '
        f'vector-effect="non-scaling-stroke"/>')

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

    global_minima = _global_minima(p)
    for qi, q in enumerate(p.enumeration.points):
        x, y = to_px(q.a, q.b)
        if qi in global_minima:
            svg.circle(x, y, 5.5, "white", stroke=PALETTE["min_fill"],
                       sw=1.7)
        elif q.kind == "min":
            svg.circle(x, y, 4.5, PALETTE["min_fill"])
        elif q.source == "B":
            svg.diamond(x, y, 5, PALETTE["bsaddle_stroke"])
        else:
            svg.triangle(x, y, 5, PALETTE["saddle_fill"])

    if title:
        svg.text(width / 2, 22, title, size=14, anchor="middle")
    return svg.to_string()


def save(svg_text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg_text)
    return path
