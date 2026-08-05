#!/usr/bin/env python3
"""Measure branch-trace parallelism, without modifying the library.

    python demos/explorer/bench_parallel.py [zoo-name] [workers]

Traces the STABLE branches only.  They are the simple case -- one call each,
no discovery/refine two-stage logic to duplicate -- and they are enough to
answer the two questions that decide the parallelisation strategy:

  1. Do THREADS help?  They should not while charts.py and gauss.py are pure
     Python/NumPy: the trace loop holds the GIL, and the 2x2 array work is far
     too small for NumPy to release it.  If threads DO help, that assumption
     is wrong and the C migration is less urgent than it looks.

  2. How bad is the load imbalance?  Speedup is bounded by
     total_work / longest_branch, so the per-branch spread is the ceiling on
     any scheme.  A single stiff branch dominating means neither threads nor
     processes will save the hard cases.

Nothing here is certified: this measures wall clock and discards the curves.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

try:
    from spong import atlas, charts, model, portrait, sturm, zoo
except ImportError:
    sys.path.insert(0, str(REPO / "src"))
    from spong import atlas, charts, model, portrait, sturm, zoo


QUAD = ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def setup(name: str | None):
    """Reproduce exactly the pre-trace state of portrait.compute."""
    if name:
        z = zoo.get(name)
        f, g = list(z.f), list(z.g)
        mu = (model.moments_normal01 if z.moment_dist == "normal01"
              else model.moments_uniform01)(2 * max(len(f), len(g)) - 1)
        view = z.default_view
    else:
        f, g = QUAD
        mu = model.moments_uniform01(2 * max(len(f), len(g)) - 1)
        view = None

    m = model.build(f, g, mu)
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    display_view = atlas.compute_box(m, e, view=view)
    box = portrait._trace_box(m, display_view, scale=1.35)
    span = max(display_view[3] - display_view[2],
               display_view[1] - display_view[0])
    return m, e, box, span / 30000.0


def stable_task(args):
    """One stable branch.  Module level so ProcessPoolExecutor can pickle it."""
    m, b, sign, box, ds, local, stub = args
    t0 = time.perf_counter()
    br = charts.trace_stable(m, b, sign, box=box, ds=ds,
                             critical_local=local, critical_stub=stub)
    return (float(b), int(sign), br.term, int(len(br.Y)),
            time.perf_counter() - t0)


def build_tasks(m, e, box, ds):
    tasks = []
    for s in e.saddles:
        for sign in (+1, -1):
            tasks.append((m, s.b, sign, box, ds, s.local,
                          portrait._stable_stub(s, sign, m)))
    return tasks


def run(label, tasks, executor=None, workers=None):
    t0 = time.perf_counter()
    if executor is None:
        results = [stable_task(t) for t in tasks]
    else:
        with executor(max_workers=workers) as ex:
            results = list(ex.map(stable_task, tasks))
    wall = time.perf_counter() - t0
    return label, wall, results


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    m, e, box, ds = setup(name)
    tasks = build_tasks(m, e, box, ds)
    print(f"case: {name or 'f = g = 1 + x + x^2'}   "
          f"{len(e.saddles)} saddles -> {len(tasks)} stable branches   "
          f"ds = {ds:.3e}")

    label, serial, results = run("serial", tasks)
    per = sorted((r[4] for r in results), reverse=True)
    total = sum(per)
    print(f"\n{'branch (b, sign)':>22s} {'term':>10s} {'pts':>8s} {'sec':>8s}")
    for b, sign, term, n, dt in sorted(results, key=lambda r: -r[4]):
        print(f"{b:>15.6f} {sign:>+3d}   {term:>10s} {n:>8d} {dt:>8.3f}")

    print(f"\nserial wall            {serial:7.3f}s")
    print(f"sum of branch times    {total:7.3f}s")
    print(f"longest branch         {per[0]:7.3f}s")
    print(f"imbalance ceiling      {total / per[0]:7.2f}x "
          f"(best possible speedup, any worker count)")

    # Longest-first dispatch: with a dynamic queue this is what keeps the tail
    # from stranding a worker.
    ordered = [t for _, t in sorted(
        zip((r[4] for r in results), tasks),
        key=lambda p: -p[0])]

    for lbl, ex in (("threads", ThreadPoolExecutor),
                    ("processes", ProcessPoolExecutor)):
        try:
            _, wall, _ = run(lbl, ordered, ex, workers)
            print(f"{lbl:<22s} {wall:7.3f}s   speedup {serial / wall:5.2f}x "
                  f"({workers} workers)")
        except Exception as exc:
            print(f"{lbl:<22s} FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
