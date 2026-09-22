# The global normal form L = u(b) + v^2 — measured, not adopted

*Status: experimental evidence, 2026-09-22.  Nothing in `src/` uses this.
The probe is `scripts/normal_form_probe.py`.*

## The transformation

With `a* = B/A` and `u = C - B^2/A`, set

    v = sqrt(A(b)) * (a - a*(b)).

Under hypothesis (A) this is a **global diffeomorphism** of the plane — for
each `b`, `a -> v` is affine with slope `sqrt(A) > 0` — and it puts the loss in
exact normal form

    L = u(b) + v^2.

`v^2 = L - u(b)` is the excess loss above the backbone: a scale-free quantity.
Going further, on any interval where `u` is monotone `s = u(b)` is itself a
coordinate and `L = s + v^2`; near a critical point the Morse lemma gives
`L = u* ± t^2 + v^2`.  The whole landscape reduces to three universal local
forms, and every model-specific fact moves into the pulled-back metric.

That metric is explicit.  With `p = a*' - v A'/(2 A^{3/2})`, the Euclidean
gradient flow in `(b, v)` is

    b' = -u' + 2 p sqrt(A) v,      v' = -2 A (1 + p^2) v + p sqrt(A) u'.

This is a **change of coordinates with the metric carried along**, so the
orbits are exactly those of the original flow.  It is not the rheostat, which
changes the metric and therefore the orbits.

The motivation was the "|b| is the villain" intuition, made precise: the
villain is the scale of `A(b)` in either direction.  Large `A` at large `|b|`
makes the fast direction stiff (`L_aa = 2A`); small `A` makes `a* = B/A` huge
and stretches the geometry across `a`.  `sqrt(A)` normalises both, and no
constant rescaling can, since the right factor varies by tens of orders along
`b` — and a linear change of coordinates leaves the flow's Jacobian spectrum
where it was anyway.

## The round trip

Repatriation is `a = a*(b) + v/sqrt(A(b))`, with

    J = d(a,b)/d(b,v) = [[p, 1/sqrt(A)], [1, 0]],   det J = -1/sqrt(A).

In operator norm `cond(J)` grows like `(1+p^2) sqrt(A)` — as bad as the
original problem, as the general principle predicts.  But a traced branch is a
**set**, blind to error along its own tangent, so what matters is the normal
component.  For a unit error normal to a curve with unit tangent `t`:

    normal amplification = |det J| / |J t| = 1 / (sqrt(A) |J t|).

## Measurements

**Exactness.**  `|L - (u + v^2)|` relative, at 2000 random points:
`2.7e-16` (case 165), `8.4e-16` (near-slide-d2).  Holds.

**The conditioning formula.**  Measured by perturbing actual trace points and
repatriating, it matches `|det J|/|J t|` to three or four digits on every
sample (e.g. `2.683e6` against `2.688e6`; `19.77` against `19.77`).  The
formula is right.

**The regime prediction failed.**  It was predicted that branches mostly sit
along the slow manifold, where `t ~ d/db` and the amplification is
`1/(sqrt(A) sqrt(1+p^2)) < 1`, and that the bad direction — a branch running
parallel to the `b`-axis across a steep backbone — would be rare.  Measured
amplification was **1.8 to 20 on near-slide-d2** and **2.7e6 to 2.0e7 on case
165**.  Not damped anywhere measured.

**Case 165** (directed seed 953953598; `A(0) = 9.1e-4`, `B(0) = -4.05e4`,
so `a*` dives from 0 at the B-saddles `b = ±0.21` to `-4.45e7` at `b = 0`):

- *The earlier account of this case was wrong.*  The inner unstable branch
  does **not** ride the backbone.  Traced in `(b,v)` from its stub, after
  20,000 steps it was at `b = -0.109`, **`a = -0.115`**, with `v = 3.15e5`,
  while the backbone beneath it was already at `a* ~ -3.3e6`.  The branch
  traverses along `b` at nearly constant `a` while the backbone dives; to
  express that, `v` swings through hundreds of thousands, and the traverse —
  simple in `(a,b)` — is violent in `(b,v)`.  That is exactly the worst
  direction above, sustained over the whole traverse.
- *The plunge* at `b ~ 0`, where production had crawled to `a = -1.3e6` in
  304,706 steps (~3% of the excursion): from there, `(b,v)` reached
  `a = -4.416e7`, **99.2% of the way** to the minimum at `-4.4497e7`, in 20,000
  steps.  Better than `(a,b)` by a wide margin on that segment — but not the
  short exponential decay predicted, since `|p| = 3.2e5` at the start: `a*` is
  steep right up to its extremum, and the stiffness `2A(1+p^2) ~ 1.8e8`
  persists.  The prototype integrator's 38% rejection rate is part of the cost.

**Agreement on near-slide-d2.**  Captured branches repatriated from `(b,v)`
sit on the production traces with medians `1e-7` to `1e-10` and maxima up to
`8e-6`.  Plausibly the prototype's global error rather than a defect in the
map; inconclusive at production accuracy.

## What this says

1. **No global chart is uniformly good.**  The normal form is exact and
   elegant, and it moves the conditioning — as the general principle says it
   must — onto precisely the branches that don't follow the backbone.  Case
   165's single branch has two phases wanting *opposite* charts: the
   traverse wants `(a,b)`, the plunge wants `(b,v)`.
2. **The indicator is validated and cheap.**  `|det J|/|J t|` is computable
   along any trace and tells you which chart is locally well-conditioned.  A
   candidate chart-switching criterion.
3. **The direction this points: an atlas of series charts.**  The invariant
   manifold jet (`local._manifold_series`) is already one working instance —
   exact coefficients from the polynomial field, a recursion never singular at
   a saddle, convergence visible order by order, an error estimate from the
   tail.  Generalised: a chart is a centre plus a choice of independent
   variable in which the branch's local graph has O(1) slope; within a chart
   the branch's Taylor series comes by recursion (Cauchy products — triangular
   Toeplitz); between charts, analytic continuation (re-centring is a Taylor
   shift, a Pascal-matrix operation; re-parametrising is composition and
   reversion).  All structured, all driven by exact coefficients.
4. **Two constraints to design around.**  *Series do not beat stiffness if the
   parameter is time*: a Taylor step on a mode of rate `lambda` has truncation
   `~ (lambda h)^N / N!`, so the step stays `~ N/lambda`.  The jet escaped this
   because its parameter was the manifold's own `t` with `t' = rho t`.  Charts
   should be graphs or loss, never time.  And *accurate* acceleration of the
   Taylor shift at large order likely means structure-exploiting O(N^2) with
   exact or high-precision coefficients; the FFT-based O(N log N) shift loses
   accuracy badly as N grows.

Not pursued further for now.  To be designed, not grown one probe at a time.
