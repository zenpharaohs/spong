"""Merge tree: cost at degree, and agreement with the fates inventory.

    python scripts/merge_tree_probe.py                       # cheap cases
    python scripts/merge_tree_probe.py tricky-d11
    python scripts/merge_tree_probe.py linear-target-d17-thrash

Two questions, both measured rather than argued.

COST.  The claim is that skeleton connectivity needs one root isolation per
gap between distinct critical values, at degree ~2 deg g, shared by every
branch -- against the audit's per-branch search at 4 deg A (136 at d17).
This prints the tree's build time and the degree actually used.

AGREEMENT.  merge_tree reaches a component through rational levels and exact
root counts; topology's `_sublevel_component_inventory` reaches it through a
slack-derived level and its own isolation.  The routes share no code, so
agreement is evidence and disagreement indicts one of them.  For every
unstable branch this compares the terminal-sample verdicts: the minimum each
route forces, or the fact that neither does.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

from spong import (fates, merge_tree, model, portrait,     # noqa: E402
                   sturm, zoo)
from spong import _poly as P                                # noqa: E402

CHEAP = ["near-slide-d2", "dead-neuron-far-saddle-d3", "quadratic-stiff",
         "minimal-quartet", "nonnearest-attachment"]


def build_model(name):
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu)


def run(name):
    m = build_model(name)
    t0 = time.perf_counter()
    e = sturm.enumerate_critical_points(m)
    t_enum = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = merge_tree.build(m, e)
    t_tree = time.perf_counter() - t0

    level_degree = P.degree(merge_tree.level_polynomial(m, tree.levels[0]))
    print(f"\n=== {name}")
    print(f"  critical points {len(e.points)}  "
          f"({len(e.minima)}m/{len(e.saddles)}s)   "
          f"levels {len(tree.levels)}   "
          f"level-poly degree {level_degree}   "
          f"(funnel would be 4*deg A = {4 * P.degree(m.alpha)})")
    print(f"  enumerate {t_enum:.2f}s    merge tree {t_tree:.2f}s")
    if tree.sequence.unseparated:
        print(f"  unseparated value classes: {tree.sequence.unseparated}")
    for k, (c, comps) in enumerate(zip(tree.levels, tree.components)):
        shape = "  ".join(
            f"[{'B' if x.bounded else 'U'} {len(x.minima)}m/{len(x.saddles)}s]"
            for x in comps)
        print(f"    level {k} c={float(c):.9g}   {shape}")
    u_inf = merge_tree.backbone_level_at_infinity(m)
    print(f"  u(infinity) = "
          f"{'unbounded below' if u_inf is None else float(u_inf)}")
    print(f"  escape-eligible saddles: "
          f"{[float(e.points[i].b) for i in merge_tree.escape_eligible(m, e, tree)]}")

    t0 = time.perf_counter()
    try:
        p = portrait.compute(m, _skip_audit=True)
    except TypeError:
        p = portrait.compute(m)
    print(f"  geometry {time.perf_counter() - t0:.2f}s")

    report = {r["branch"]: r for r in
              fates.fate_report(m, p.enumeration, p.branches)}
    agree = disagree = 0
    for i, branch in enumerate(p.branches):
        if branch.kind != "unstable" or len(branch.Y) < 2:
            continue
        a, b = (float(branch.Y[len(branch.Y) - 2, 0]),
                float(branch.Y[len(branch.Y) - 2, 1]))
        tree_fate = merge_tree.fate_from_tree(m, e, tree, a, b)
        entry = report.get(i, {})
        inv = entry.get("terminal_fates") or entry
        # Compare LIKE WITH LIKE.  fates.forced means capture-forced only,
        # so an escape-forced component is not a disagreement.  And the two
        # routes report minima DIFFERENTLY -- merge_tree gives indices into
        # the enumeration, fates gives (a, b) coordinate pairs -- so the
        # comparison goes through b values, not raw containers.
        tree_capture = bool(tree_fate and tree_fate["fate"] == "capture")
        inv_capture = bool(inv.get("forced"))
        ok = tree_capture == inv_capture
        if ok and tree_capture:
            tree_bs = sorted(float(e.points[j].b)
                             for j in tree_fate["minima"])
            inv_bs = sorted(float(y) for _, y in inv.get("minima", ()))
            ok = (len(tree_bs) == len(inv_bs)
                  and all(abs(x - y) <= 1e-6 * (1 + abs(y))
                          for x, y in zip(tree_bs, inv_bs)))
            if not ok:
                print(f"        tree minima b={tree_bs}  "
                      f"inventory minima b={inv_bs}")
        agree += ok
        disagree += not ok
        note = "" if ok else "   <-- DISAGREE"
        where = (tree_fate["fate"] if tree_fate else "unlocated")
        print(f"    br{i:<3d} {branch.term:<10s} tree={where:<10s} "
              f"capture tree/inv={tree_capture}/{inv_capture} "
              f"splits={tree_fate['splits_remaining'] if tree_fate else '-'}"
              f"{note}")
    print(f"  capture verdicts: {agree} agree, {disagree} disagree")


def main() -> int:
    names = sys.argv[1:] or CHEAP
    for name in names:
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
