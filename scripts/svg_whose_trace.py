#!/usr/bin/env python3
"""Whose trace is that?  Read an explorer SVG export and, for every path,
report its identity (the data- attributes) and how close it comes to the
critical point nearest the view centre.

    python scripts/svg_whose_trace.py out/spong-....svg [--near-px 60]

Paths that never enter the view are listed as such.  Distances are in the
SVG's pixel coordinates (the view at export time).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET

NS = {"svg": "http://www.w3.org/2000/svg"}


def parse_path(d: str):
    nums = re.findall(r"[ML]?(-?[\d.]+(?:e-?\d+)?) (-?[\d.]+(?:e-?\d+)?)", d)
    return [(float(x), float(y)) for x, y in nums]


def _model_from_comments(comments):
    """Rebuild L(a, b) and the view from the export's comment lines, or
    (None, None) if they cannot be parsed or spong is unavailable."""
    text = "\n".join(comments)
    fm = re.search(r"f = \[([^\]]*)\]", text)
    gm = re.search(r"g = \[([^\]]*)\]", text)
    mm = re.search(r"moments = (\w+)", text)
    vm = re.search(r"view a \[([^,]+), ([^\]]+)\] b \[([^,]+), ([^\]]+)\]", text)
    view = tuple(float(x) for x in vm.groups()) if vm else None
    if not (fm and gm and mm):
        return None, view
    try:
        from spong import model
    except ImportError:
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        from spong import model
    f = [float(x) for x in fm.group(1).split(",")]
    g = [float(x) for x in gm.group(1).split(",")]
    n = 2 * max(len(f), len(g)) - 1
    kind = mm.group(1)
    if kind == "uniform01":
        mu = model.moments_uniform01(n)
    elif kind == "normal01":
        mu = model.moments_normal01(n)
    else:
        return None, view
    m = model.build(f, g, mu)
    return (lambda a, b: float(m.L(a, b))), view


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("svg")
    ap.add_argument("--near-px", type=float, default=60.0,
                    help="report paths passing within this many pixels of "
                         "the central critical point")
    ap.add_argument("--detail", action="append", default=[],
                    help="path id whose in-view vertices are printed in "
                         "model coordinates with the loss (repeatable)")
    ap.add_argument("--first", type=int, default=12,
                    help="also print this many vertices from the START of "
                         "each detailed path (the launch end)")
    args = ap.parse_args(argv)

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(args.svg, parser=parser).getroot()
    W = float(root.get("width")); H = float(root.get("height"))
    comments = []
    for child in root:
        if child.tag == ET.Comment:
            comments.append(child.text.strip())
            print(child.text.strip())
    model_L, view = _model_from_comments(comments)
    # everything is in the default namespace or none; accept both
    paths = [e for e in root.iter()
             if isinstance(e.tag, str) and e.tag.endswith("path")]
    circles = [e for e in root.iter()
               if isinstance(e.tag, str) and e.tag.endswith("circle")]
    cx, cy = W / 2, H / 2
    if not circles:
        print("no critical points in the SVG"); return 1
    centre = min(circles, key=lambda c: math.hypot(
        float(c.get("cx")) - cx, float(c.get("cy")) - cy))
    px, py = float(centre.get("cx")), float(centre.get("cy"))
    print(f"\ncentral critical point: {centre.get('data-kind')} "
          f"(source {centre.get('data-source')}) at a={centre.get('data-a')} "
          f"b={centre.get('data-b')}  -> pixel ({px:.1f}, {py:.1f})\n")
    print(f"{'id':28s} {'kind':9s} {'saddle_b':>10s} {'dir':>4s} {'sgn':>4s} "
          f"{'term':18s} {'n':>7s} {'in view':>8s} {'closest px':>11s} "
          f"{'at index':>9s}")
    for p in paths:
        pts = parse_path(p.get("d", ""))
        inside = [(i, q) for i, q in enumerate(pts)
                  if -2 <= q[0] <= W + 2 and -2 <= q[1] <= H + 2]
        if pts:
            dist, idx = min((math.hypot(q[0] - px, q[1] - py), i)
                            for i, q in enumerate(pts))
        else:
            dist, idx = float("nan"), -1
        flag = "  <-- near" if inside and dist <= args.near_px else ""
        print(f"{p.get('id',''):28s} {p.get('data-kind',''):9s} "
              f"{p.get('data-saddle-b', p.get('data-source-b','')):>10.10s} "
              f"{p.get('data-direction',''):>4s} {p.get('data-stable-sign',''):>4s} "
              f"{p.get('data-term',''):18s} {len(pts):>7d} "
              f"{len(inside):>8d} {dist:>11.2f} {idx:>9d}{flag}")
        if flag and len(inside) >= 2:
            # Where it enters the view, passes closest, and leaves, as pixel
            # offsets from the central critical point (SVG y points DOWN,
            # so negative dy is UP, i.e. larger b).  Path order is trace
            # order: unstable branches run away from their saddle in
            # descent, stable branches away from theirs in ascent.
            rel = lambda q: f"({q[0]-px:+8.1f}, {q[1]-py:+8.1f})"
            (i0, q0), (i1, q1) = inside[0], inside[-1]
            print(f"{'':28s} enters {rel(q0)} @{i0}  closest {rel(pts[idx])} "
                  f"@{idx}  leaves {rel(q1)} @{i1}")
        if p.get("id") in args.detail and view is not None:
            a0, a1, b0, b1 = view
            to_a = lambda x: a0 + x / W * (a1 - a0)
            to_b = lambda y: b1 - y / H * (b1 - b0)
            sa, sb = float(centre.get("data-a")), float(centre.get("data-b"))
            print(f"{'':4s}vertices of {p.get('id')} in view, model coordinates; "
                  f"saddle at a={sa:.9g} b={sb:.9g}"
                  + (f" L={model_L(sa, sb):.12g}" if model_L else ""))
            print(f"{'':4s}{'index':>7s} {'a':>16s} {'b':>16s} {'a-a_s':>12s} "
                  f"{'b-b_s':>12s} {'L':>18s}")
            for i, q in inside:
                a, b = to_a(q[0]), to_b(q[1])
                Lv = f"{model_L(a, b):.12g}" if model_L else ""
                print(f"{'':4s}{i:>7d} {a:>16.9g} {b:>16.9g} {a-sa:>12.3e} "
                      f"{b-sb:>12.3e} {Lv:>18s}")
            if args.first > 0:
                print(f"{'':4s}first {args.first} vertices of {p.get('id')} "
                      f"(the launch end; offsets from THIS path's own first vertex)")
                a_start, b_start = to_a(pts[0][0]), to_b(pts[0][1])
                for i, q in enumerate(pts[:args.first]):
                    a, b = to_a(q[0]), to_b(q[1])
                    Lv = f"{model_L(a, b):.12g}" if model_L else ""
                    print(f"{'':4s}{i:>7d} {a:>16.9g} {b:>16.9g} "
                          f"{a-a_start:>12.3e} {b-b_start:>12.3e} {Lv:>18s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
