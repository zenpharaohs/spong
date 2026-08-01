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
- `optimizer_moustaches.py` — reproducible 100-start galleries using Halton
  low-discrepancy or best-candidate blue-noise initialization.  Separate
  panels overlay minibatch SGD, Nesterov momentum SGD, and Adam trajectories
  on one certified portrait; schedules, batch sizes, and common random-number
  streams are recorded in JSON.  Muon is not a default panel because `(a,b)`
  is a vector rather than a matrix-valued hidden weight: practical Muon sends
  this parameter group to auxiliary AdamW.  The duplicate AdamW fallback and
  a 2-by-1 normalized-momentum surrogate remain explicit optional diagnostics.

  ```sh
  PYTHONPATH=src:. python demos/optimizer_moustaches.py \
      --zoo quadratic-stiff --starts 100
  open out/optimizer_moustaches/quadratic-stiff_optimizer_moustaches.html
  ```
- `thompson_moustaches.py` — equal-budget versus adaptive continuation.
  Every initialization first receives one short optimizer continuation.
  Thereafter the arm with the smallest exact continuous-Bernoulli posterior
  draw receives the next continuation and its posterior is immediately
  updated with `L/(1+L)`.  Capture balls and portrait topology never enter
  allocation; they are interpretation equipment after the run.  The exact
  posterior implementation remains in the separate `continuous-bernoulli`
  package:

  ```sh
  clang -O3 -std=c99 -dynamiclib \
    ../continuous-bernoulli/src/c/cb_core.c -o /tmp/libcb_core.dylib -lm
  PYTHONPATH=src:. python demos/thompson_moustaches.py \
    --cb-library /tmp/libcb_core.dylib --zoo quadratic-stiff \
    --starts 100 --rounds 20000 --chunk-steps 10
  open out/thompson_moustaches/quadratic-stiff_low-discrepancy_adam.html
  ```
- `adam_phase_portrait.py` — first comparison between the certified
  steepest-descent portrait and the autonomous Adam vector field of
  Dereich--Jentzen--Kassing.  Both panels use exactly the same finite
  empirical distribution.  The Adam panel is explicitly a numerical oracle:
  two independent stationary-history chains provide a field-disagreement
  diagnostic, and its grid zero cells are nominations rather than
  certifications.

  ```sh
  PYTHONPATH=src:. python demos/adam_phase_portrait.py \
    --zoo quadratic-stiff
  open out/adam_phase_portrait/quadratic-stiff_empirical_n256.html
  ```
- `search_saddle_connection.py` — targeted search for the codimension-one
  failure of Morse--Smale structural stability.  High-degree random
  polynomial arms are screened by the separation of a stable and an
  energy-feasible unstable branch belonging to distinct saddles on one
  common regular loss level.  Unconstrained closest-vertex distance is not
  used: spatially close vertices at different loss values cannot converge to
  the same noncritical orbit point and previously produced false leads.
  Exact continuous-Bernoulli posteriors concentrate SPSA coefficient updates
  on promising arms.  Coefficients are normalized and quantized to bounded
  dyadics before every exact Morse analysis.  The coarse distance is a
  discovery objective; a promising pair must subsequently be bracketed with
  a signed level-section shooting residual.

  ```sh
  clang -O3 -std=c99 -dynamiclib \
    ../continuous-bernoulli/src/c/cb_core.c -o /tmp/libcb_core.dylib -lm
  PYTHONPATH=src:. python demos/search_saddle_connection.py \
    --cb-library /tmp/libcb_core.dylib --degree 11 --arms 12 --rounds 36
  ```
- `continue_saddle_connection.py` — local signed continuation after the
  search has identified the same stable/unstable pair in two candidates.
  It extrapolates the regular-level shooting residual to opposite signs,
  then bisects an exact rational affine coefficient segment.  No
  requantization is permitted during the final bracket because that would
  destroy continuity of the parameter family.

  ```sh
  PYTHONPATH=src:. python demos/continue_saddle_connection.py \
    out/search_phase2.json out/search_phase3.json \
    --output out/saddle_connection_bracket.json
  ```
- `hybrid_saddle_connection.py` — removes an unwanted remote
  saddle--minimum pair before the shooting continuation.  It solves the
  inverse saddle-node equations `N(r)=N'(r)=0` in the full `(f,g,r)` space,
  crosses to dyadic Morse models, and exact-Sturm screens the resulting
  candidates.  A candidate shooting residual is accepted only when both
  section crossings lie in the same local level-set chart; a small tangent
  projection between distant level-set components is not a bracket.

- `saddle_connection_triptych.py` — renders the registered
  `nonnearest-saddle-connection` zoo wall family as three panels: the
  Λ=2 chamber, the geometric B→N wall limit at Λ*≈2.177709563954844, and the
  Λ=4 chamber.  The main triptych keeps all three portraits unobstructed; a
  separate three-panel strip enlarges the connection/turning region.  The
  side panels have clean individual branches, certified global contact
  audits, and robustly different landing fates.  Exact terminal product
  completions prevent coincident samples inside a named minimum tube or
  stable superlevel end from being mistaken for finite-skeleton crossings.
  The center is explicitly not passed off as an ordinary certified portrait:
  the two off-wall continuations are removed and their common
  saddle-to-saddle limit is drawn once.

  ```sh
  PYTHONPATH=src python3 demos/saddle_connection_triptych.py
  open out/saddle_connection_triptych/nonnearest-saddle-connection.html
  ```
- `saddle_connection_comparison.py` — holds the exact wall model and exact
  Sturm critical-point inventory fixed, then replaces only the invariant-
  manifold geometry with Forward/Backward Euler, explicit/implicit midpoint,
  RKF45, or the `ode23s`-like ROS2 method.  For unstable traces, casual
  finite-radius saddle capture is disabled: otherwise entering a ball around
  the N saddle would make an approximate curve look like the exact B→N
  connection.  Each panel contains only the green/red phase portrait produced
  by that discretization—no optimizer or highlighted trajectory overlay.
  Continuing the discretized B-unstable branch reveals which outgoing N
  branch, and hence which minimum, is selected by accumulated launch and
  integration error.  The fixed-step default is deliberately moderate
  (`h=0.05`), rather than the visually flattering `h=0.01`.  A separate
  centerline-subtracted transverse-section figure removes the common motion
  along the connection and displays the otherwise subpixel disagreement of
  the independently traced `W^u(B)` and `W^s(N)` curves.

  ```sh
  PYTHONPATH=src:. python3 demos/saddle_connection_comparison.py
  open out/saddle_connection_comparison/nonnearest-saddle-connection-casual-comparison.html
  ```

- `batch_moment_portraits.py` — fixes `f` and `g` at the registered
  population handle-slide member, draws independent raw samples from U(0,1),
  and builds each batch loss from the exact empirical moments of those
  binary64 samples.  Up to six independently certified skeletons are
  superposed for each selected batch size; any unresolved portrait is withheld
  and reported.  The sample streams are nested across batch sizes, so
  increasing `N` adds observations instead of silently replacing the
  experiment.

  The demo intentionally does **not** give a source-labelled branch an
  identity across batches.  Even when the endpoint Morse inventories agree,
  the affine span between population and empirical moments may contain a
  critical-point bifurcation.  `--span-probes` performs exact Morse analyses
  at rational points on that segment, but this finite probe is a diagnostic,
  not an exclusion proof for an algebraic wall between probes.  Every drawn
  batch skeleton is independently certified by the geometry engine.

  ```sh
  PYTHONPATH=src:. python3 demos/batch_moment_portraits.py \
      --batch-sizes 32,128,512 --batches 6 --jobs 6
  open out/batch_moment_portraits/nonnearest-saddle-connection-batches.html
  ```
