# demos — consumers, not library code

Descent methods (SGD, Adam, L-BFGS, ...) live here and only here, overlaid on
certified portraits: the portraits judge the optimizers, never the reverse.
Here now:
- `optimizers.py` — demo-grade SGD (honest batches: the stochastic gradient
  is computed from raw U(0,1) samples, so batch noise IS moment-space
  jitter), Adam, and full-batch L-BFGS (two-loop, Armijo).
- `showcase_tricky.py` — the founding demo: all three on the tricky d=11
  portrait.  Findings on first run: naive SGD explodes in 2 steps from any
  stiff start; in the fan the stochastic gradient is single-sample
  dominated (g(bx)² ~ b²² amplifies batch noise ~1e11) so SGD's largest
  stable lr is ~1e-11, at which 20k steps buy ~12% of the journey; Adam
  survives by normalizing and then parks AT a saddle for 20k steps (both
  the tricky saddle and the B-saddle); L-BFGS reaches good minima here — but
  by LUCK: its early line-search steps are large enough to jump basins
  entirely, and basin-jumping cuts both ways (the same leaps that escape a
  bad basin will, on average, carry it out of a good basin into a bad
  one).  The skeleton is what makes this diagnosis possible: overlay the
  trajectory on the certified separatrices and every basin crossing is
  visible and attributable.
  Output: docs/gallery/tricky_vs_optimizers.svg.

Planned next: batch-morphing across moment space with certified
discriminant walls (the walls are algebraic: disc(B·N) and psi-positivity).
