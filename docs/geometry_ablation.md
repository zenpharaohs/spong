# Geometry-engine ablation program

SPONG's production portraitist is a composition of numerical choices, not
"GL8 instead of Forward Euler."  Qualification must therefore ablate those
choices sequentially.  Exact Sturm critical points and exact local Hessian
spectra remain fixed throughout the geometry experiment; replacing them with
grid Newton is a separate critical-point experiment.

## Ladder

| rung | local invariant-manifold construction | continuation | audit |
|---|---|---|---|
| full production | selective Poincare conditioning + Hadamard graph fixed point | chart-dispatched IRK-GL8 with full/two-half checks and dense collocation | global topology certificate |
| no audit | same | same | none; measures what certification adds rather than changing the curve |
| continuation ablation | same certified materialized stubs | named textbook IVP method | diagnostic only |
| Poincare ablation | centered-jet Hadamard graph fixed point | production GL8 | diagnostic only |
| graph-transform ablation | exact centered local flow/eigenvector launch | production GL8 | diagnostic only |
| launch ablation | fixed offset in the exact saddle eigenvector | production GL8 | diagnostic only |
| vanilla portraitist | fixed exact-eigenvector offset | Forward/Backward Euler, explicit/implicit midpoint, RKF45, or ROS2 | none |

The current `comparison.casual_portrait` is already the bottom rung.  It does
not call the Poincare transform, Hadamard graph transform, scalar graph-chart
dispatcher, Gauss collocation, Richardson checks, or topology audit.  It holds
the exact critical inventory fixed so failures can be attributed to geometry.

The intermediate rungs still need explicit production switches.  They should
not be simulated by deleting data after a full run: every rung must construct
its own launch and continuation from the information it is allowed to use.

## Measurements

Every rung reports more than whether a low-resolution picture looks plausible:

1. branch incidence and terminal critical point;
2. signed separation of independently traced branches on common regular loss
   sections;
3. forward/backward anadromy defect when the same initial state and interval
   are used;
4. local invariant-graph residual and certified physical reach;
5. handoff seam residuals and full/two-half disagreement;
6. global branch intersections and topology-audit result; and
7. work: field evaluations, nonlinear iterations, rejected steps, and wall
   time.

The common-section mismatch is distinct from a pure anadromy test.  At the
handle-slide wall it compares independently launched `W^u(B)` and `W^s(N)`, so
it includes both launch and continuation error.  Pure anadromy starts once and
composes the numerical map with its negative-step adjoint.

## Witness cases

### Saddle-connection wall: two-sided consistency and topology

At the midpoint regular level of the registered wall example, the current
measurements at the old, unusually fine fixed step `h=0.01` are:

| geometry | signed section mismatch |
|---|---:|
| full conditioned-stub/GL8 computation | +9.19e-10 |
| Forward Euler | -5.91e-4 |
| Backward Euler | +5.99e-4 |
| explicit midpoint | -3.49e-6 |
| implicit midpoint | +1.33e-6 |
| RKF45 | -3.16e-4 |
| ROS2 | -2.84e-4 |

All casual separations are below a pixel in the ordinary portrait, which is why
several pictures look deceptively good.  The sign and magnitude become
macroscopic when the branch passes the target saddle and selects an outgoing
basin.  This case tests the combined launch/continuation consistency.  The
intermediate continuation-ablation rung will isolate how much GL8 anadromy adds
after the same certified stubs are supplied to every method.

The fixed-step convergence sequence shows why `h=0.01` should not be the
display default:

| method | `h=0.20` | `h=0.10` | `h=0.05` | `h=0.02` | `h=0.01` |
|---|---:|---:|---:|---:|---:|
| Forward Euler | -1.04e-2 | -5.56e-3 | -2.87e-3 | -1.17e-3 | -5.91e-4 |
| Backward Euler | +1.42e-2 | +6.43e-3 | +3.09e-3 | +1.21e-3 | +5.99e-4 |
| explicit midpoint | -1.12e-3 | -2.97e-4 | -7.80e-5 | -1.29e-5 | -3.49e-6 |
| implicit midpoint | +6.70e-4 | +1.58e-4 | +3.99e-5 | +6.25e-6 | +1.33e-6 |

The expected first- and second-order trends are visible.  At `h=0.01`, Euler
uses thousands of accepted steps across the two traces, so its ambient-plane
portrait is a poor illustration of a casual Euler portraitist.  The gallery
now defaults to `h=0.05` and reports the exact setting.

Adaptive runs need a different warning.  RKF45 gave signed mismatches
`+1.75e-3`, `-1.27e-3`, `-2.46e-4`, `+4.24e-4`, and `+4.48e-5` at representative
tolerances `1e-1`, `1e-2`, `1e-3`, `1e-4`, and `1e-5`: refinement was not
monotone and even the selected chamber changed.  ROS2 failed to produce both
common-section crossings at the loosest tolerances, then approached the wall
from one side.  That behavior is exactly why a local adaptive error estimate
cannot replace the two-sided geometric and topology certificates.

### `tricky-d11`: Hadamard graph transform

This is the canonical graph-transform witness.  A valid ablation must show that
the no-Hadamard rung fails, refuses, or follows the nearby backbone, while the
conditioned invariant-graph stub hands a certified branch to deep water.  A
smaller runtime or smoother polyline is irrelevant if branch incidence is
wrong.

### Poincare-conditioning witness: to be selected metrologically

Do not choose this example because its final portrait looks dramatic.  Search
the qualification corpus for a reproducible saddle where, at the same grid and
certificate tolerances, Poincare conditioning materially increases one of:

- maximum certified invariant-graph reach;
- global-field resolution margin at handoff;
- injectivity margin;
- contraction rate / fixed-point iteration count; or
- the centered graph's ability to reach the continuation-ready region at all.

Register the strongest ordinary-scale witness in the zoo, then retain more
extreme cases as stress tests.

## What can be proved

Universal completeness is not a reasonable goal: arbitrary polynomial data
can place a portrait beyond a fixed arithmetic or geometric resolution.  The
appropriate theorem is conditional soundness, not universal success:

> If the exact Morse analysis succeeds, every local and continuation
> certificate is satisfied, and the global topology audit certifies, then the
> returned embedded Morse skeleton has the reported incidence within the
> stated arithmetic and geometric bounds.

Thus the engine can return `certified`, a mathematically classified non-Morse
case, or an explicit unresolved result.  Ablation validates why each component
enlarges the set of instances reaching `certified`; it does not pretend that
the set is all SPONG instances.
