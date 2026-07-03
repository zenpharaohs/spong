"""The founding demo: SGD, Adam, and L-BFGS on the graph paper where
descent methods go to die.

Run:  PYTHONPATH=src:. python demos/showcase_tricky.py

The portrait is the mean-field object (Ljung): handing the reader the
noiseless flow forecloses "it's just SGD noise" as an explanation.  The
overlays are the optimizers' actual trajectories; the certificates in
the footer are why the graph paper itself cannot be blamed.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from spong import model, portrait, render                     # noqa: E402
from demos import optimizers as opt                           # noqa: E402
from tests.test_enumeration import TRICKY_F                   # noqa: E402


def main():
    t0 = time.time()
    m = model.build(TRICKY_F, TRICKY_F, model.moments_uniform01(23))
    p = portrait.compute(m, view=(-1.5, 2.5, -4.0, 3.0))
    print(f"portrait: {time.time()-t0:.0f}s  "
          f"E_max={p.ledger['summary']['worst_angle_energy']:.1e}  "
          f"balanced={p.ledger['summary']['balanced']}")

    box = p.box
    rng = np.random.default_rng(7)
    sgrad = opt.BatchGradient(TRICKY_F, TRICKY_F, batch_size=32, rng=rng)
    fgrad = lambda a, b: m.gradL(a, b)

    starts = [(0.55, -3.30),      # in the fan below the tricky saddle
              (-0.90, 2.20),      # left plateau, above the B-saddle
              (2.20, -1.20)]      # right slope near the bend

    overlays = []
    report = []
    palette = {"sgd": "#2060e0", "adam": "#9020c0", "lbfgs": "#00a0a8",
               "sgd-stable": "#70b0ff"}
    for z0 in starts:
        tr_sgd = opt.run_sgd(sgrad, z0, lr=2e-3, n_steps=4000, box=box)
        tr_adam = opt.run_adam(sgrad, z0, lr=2e-2, n_steps=20000, box=box)
        tr_lb = opt.run_lbfgs(m, z0, n_steps=200, box=box)
        runs = [("sgd", tr_sgd), ("adam", tr_adam), ("lbfgs", tr_lb)]
        if z0 == starts[0]:
            # the exhaustion exhibit: in the fan the STOCHASTIC gradient
            # is single-sample dominated (g(bx)^2 ~ b^22 amplifies batch
            # noise by ~1e11), so SGD's largest stable lr is ~1e-11 — at
            # which 20k steps buy ~12% of the journey to the minimum
            tr_stable = opt.run_sgd(sgrad, z0, lr=1e-11, n_steps=20000,
                                    box=box)
            runs.append(("sgd-stable", tr_stable))
        for name, tr in runs:
            overlays.append({"Y": tr, "color": palette[name], "width": 1.4})
            z_end = tr[-1]
            cp, dist = opt.nearest_critical(p.enumeration, z_end)
            moved = float(np.hypot(tr[-1][0] - z0[0], tr[-1][1] - z0[1]))
            fate = (f"{cp.kind}@b={cp.b:.3f}" if dist < 0.5 else
                    ("exploded" if not np.all(np.isfinite(tr[-1])) or
                     m.L(*z_end) > 1e3 else
                     (f"stuck (moved {moved:.1e})" if moved < 1e-2
                      else "wandering")))
            report.append((name, z0, len(tr), m.L(*z_end), fate))
    labels = {"sgd": "SGD lr=2e-3 (explodes in the fan)",
              "adam": "Adam: 20k steps, parked at a saddle",
              "lbfgs": "L-BFGS (full batch)",
              "sgd-stable": "SGD, largest stable lr (1e-11): 20k steps = 12% of the journey"}
    seen = set()
    for ov, (name, _z, _n, _L, _f) in zip(overlays, report):
        if name not in seen:
            ov["label"] = labels[name]
            seen.add(name)

    svg = render.plane_view(
        p, overlays=overlays,
        title="descent methods on the graph paper where they go to die "
              "(tricky d=11, kappa = 8.5e8)")
    render.save(svg, "docs/gallery/tricky_vs_optimizers.svg")

    print(f"{'method':8s} {'start':>16s} {'steps':>6s} {'final L':>12s}  fate")
    for name, z0, n, L, fate in report:
        print(f"{name:8s} ({z0[0]:5.2f},{z0[1]:5.2f}) {n:6d} {L:12.4e}  {fate}")


if __name__ == "__main__":
    main()
