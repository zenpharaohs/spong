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
  Λ=4 chamber.  Each panel includes an enlarged connection-region inset.  The
  side panels have clean individual branches and robustly different landing
  fates; their current full-plane contact audits remain FP64-unresolved on
  asymptotically adjacent stable tails and are reported as such.  The center
  is explicitly not passed off as an ordinary certified portrait: the two
  off-wall continuations are removed and their common saddle-to-saddle limit
  is drawn once.

  ```sh
  PYTHONPATH=src python3 demos/saddle_connection_triptych.py
  open out/saddle_connection_triptych/nonnearest-saddle-connection.html
  ```

Planned next: batch-morphing across moment space.  Loss of the Morse
critical-point inventory is controlled by exact algebraic walls
(`disc(B·N)`/the reduced numerator and ψ-positivity).  These are not all the
walls of the phase portrait: global saddle--saddle handle slides occur where
a separatrix shooting map vanishes while the algebraic Morse data can remain
fixed.  Such walls require geometric bracketing and a-posteriori topology
certification.  The present Λ construction varies `f` and `g`; demonstrating
a handle slide on a fixed-`(f,g)`, moment-only batch path remains a separate
experiment.
