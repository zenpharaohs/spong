"""Why are contact events not being discharged?

    python scripts/contact_probe.py quadratic-stiff
    python scripts/contact_probe.py near-slide-d2

Runs the audit at geometry_level 0 ONLY -- the census reports the escalated
level 2, which is a consequence of the level-0 refusal, not its cause.

For every branch it prints length, terminal-suffix kind and start, and the
endpoint certificate's method.  Then, for each forbidden intersection, it
prints the exact reason the discharge test failed:

    both branches' suffix kinds and starts,
    whether each segment index is at or past its own suffix start,
    and, for two captures, the distance between their terminal minima.

`same_sublevel_end` requires ALL of: both suffix kinds `minimum_sublevel`,
both starts present, si >= start_i, sj >= start_j, and the two terminal
minima within allowed_radius.  One column will be False; that column is
the defect.  No inference -- just the failing conjunct.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("SPONG_ENGINE", "native")
os.environ.setdefault("SPONG_WORKERS", str(os.cpu_count() or 1))

import numpy as np                                          # noqa: E402
from spong import merge_tree, model, portrait, topology, zoo   # noqa: E402
from qualify import directed_model, random_model             # noqa: E402


def build_case(argv):
    """A zoo case by name, or an ensemble case by its seed.

        python scripts/contact_probe.py quadratic-stiff
        python scripts/contact_probe.py 499170635 directed 5

    The seed form reproduces exactly the model an ensemble record names,
    so a refusal found at scale can be dissected at geometry level 0
    without first minting a zoo entry for it.
    """
    if argv and argv[0].isdigit():
        seed = int(argv[0])
        mode = argv[1] if len(argv) > 1 else "directed"
        degree = int(argv[2]) if len(argv) > 2 else 5
        generate = directed_model if mode == "directed" else random_model
        m, spec = generate(random.Random(seed), degree)
        return m, None, f"{mode} seed {seed} ({spec})"
    name = argv[0] if argv else "quadratic-stiff"
    z = zoo.get(name)
    n = 2 * max(len(z.f), len(z.g)) - 1
    mu = (model.moments_normal01(n) if z.moment_dist == "normal01"
          else model.moments_uniform01(n))
    return model.build(list(z.f), list(z.g), mu), z.default_view, name


def main() -> int:
    m, view, label = build_case(sys.argv[1:])
    p = portrait.certified_compute(m, view=view, max_geometry_level=0)
    top = p.ledger["topology"]
    print(f"{label}  status={top.get('status')}  "
          f"reason={top.get('resolution_reason')}  "
          f"forbidden={top.get('forbidden_count')}  "
          f"ambiguous={top.get('ambiguous_count')}")

    suffixes = top.get("terminal_suffixes", [])
    ends = {x["branch"]: x for x in top.get("unstable_ends", ())}
    print("\nbranches:")
    for i, br in enumerate(p.branches):
        s = suffixes[i] if i < len(suffixes) else {}
        e = ends.get(i, {})
        print(f"  br{i:<3d} {br.kind:<9s} {br.term:<10s} n={len(br.Y):<7d} "
              f"suffix={str(s.get('kind')):<18s} start={s.get('start')} "
              f"method={e.get('method')} "
              f"certified={e.get('certified')}")

    forbidden = top.get("forbidden_intersections", [])

    # Any UNCERTIFIED capture is the thing to explain first: it gets no
    # terminal suffix, so every contact involving it stays undischarged and
    # one branch can exhaust the whole event budget.  Print, per level, what
    # the tree actually saw at that branch's terminal sample -- that
    # distinguishes "above every level" from "component holds 2+ minima"
    # from "forced on the wrong minimum", which want different fixes.
    declined = [x for x in top.get("unstable_ends", ())
                if x["kind"] == "finite_capture" and not x["certified"]]
    if declined:
        tree = merge_tree.build(m, p.enumeration)
        print(f"\nuncertified captures ({len(declined)}):")
        for entry in declined:
            br = p.branches[entry["branch"]]
            last = len(br.Y) - 2
            a, b = float(br.Y[last, 0]), float(br.Y[last, 1])
            loss = merge_tree.exact_loss(m, a, b)
            print(f"  br{entry['branch']} reason={entry.get('reason')} "
                  f"target={br.diag.get('target')} n={len(br.Y)}")
            print(f"    terminal a={a:.6g} b={b:.6g} L={float(loss):.12g}")
            for k, c in enumerate(tree.levels):
                comp = merge_tree.locate(m, p.enumeration, c, a, b,
                                         components=tree.components[k])
                if comp is None:
                    print(f"    level {k} c={float(c):.12g}  "
                          f"NOT IN ANY COMPONENT (L >= c)"
                          if loss >= c else
                          f"    level {k} c={float(c):.12g}  unlocated")
                    continue
                print(f"    level {k} c={float(c):.12g}  "
                      f"{'bounded' if comp.bounded else 'UNBOUNDED'} "
                      f"{len(comp.minima)}m/{len(comp.saddles)}s "
                      f"minima_b="
                      f"{[float(p.enumeration.points[i].b) for i in comp.minima]}")

    print(f"\nforbidden intersections (showing up to 12 of "
          f"{top.get('forbidden_count')}):")
    for item in forbidden[:12]:
        i, j = item["branches"]
        si, sj = item["segments"]
        ti = suffixes[i] if i < len(suffixes) else {}
        tj = suffixes[j] if j < len(suffixes) else {}
        gi, gj = ti.get("start"), tj.get("start")
        past_i = gi is not None and si >= gi
        past_j = gj is not None and sj >= gj
        same_kind = ti.get("kind") == tj.get("kind")
        distance = None
        if ti.get("kind") == tj.get("kind") == "minimum_sublevel":
            distance = float(np.hypot(
                ti["terminal"][0] - tj["terminal"][0],
                ti["terminal"][1] - tj["terminal"][1]))
        print(f"  br{i}xbr{j} seg=({si},{sj}) "
              f"kinds=({ti.get('kind')},{tj.get('kind')}) same={same_kind} "
              f"starts=({gi},{gj}) past=({past_i},{past_j}) "
              f"minima_distance={distance}")

    print("\nlegend: discharge needs same kind, both past, and for two "
          "captures a minima distance within allowed_radius "
          f"(~{max(1024*np.finfo(float).eps*max(1.0, 1.0), 1e-11):.2e} "
          "scaled by the box).")

    # The surviving AMBIGUOUS contacts are the interesting residue now that
    # removable ones are filtered: the curves are separated by MORE than
    # their combined sagitta, yet the orientation predicate still cannot
    # order them.  Print the margin -- depth against the sagitta sum -- so
    # a genuine FP64 resolution failure is distinguishable from a contact
    # sitting just over the threshold.
    ambiguous = top.get("ambiguous_contacts", [])
    sagittae = {}
    print(f"\nambiguous contacts (showing up to 12 of "
          f"{top.get('ambiguous_count')}):")
    for item in ambiguous[:12]:
        i, j = item["branches"]
        si, sj = item["segments"]
        for index in (i, j):
            if index not in sagittae:
                sagittae[index] = topology._sagitta_bounds(
                    p.branches[index].Y)
        Yi = np.asarray(p.branches[i].Y)
        Yj = np.asarray(p.branches[j].Y)
        depth = topology._crossing_depth(Yi, si, Yj, sj)
        allowance = sagittae[i][si] + sagittae[j][sj]
        ti = suffixes[i] if i < len(suffixes) else {}
        tj = suffixes[j] if j < len(suffixes) else {}
        print(f"  br{i}xbr{j} seg=({si},{sj}) "
              f"kinds=({ti.get('kind')},{tj.get('kind')}) "
              f"starts=({ti.get('start')},{tj.get('start')}) "
              f"depth={depth:.3e} sagitta_sum={allowance:.3e} "
              f"ratio={depth/allowance if allowance else float('inf'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
