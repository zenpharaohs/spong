# Continuing rheostat crossings on a fixed loss section

`spong.wall_continue` is an experimental numerical oracle. It keeps the
original rational model fixed and puts the rheostat entirely in the metric
`D = diag(Lambda**2, 1)`. It does not change the inspector's production
shooting path or replace its stored wall parameters.

For the `nonnearest-saddle-connection` family, the experiment resolves the
connection near

```
Lambda = 2.177709563954816
```

This is numerical convergence evidence, not an interval certificate. A tiny
root residual alone does not establish this accuracy.

## Representation and algorithm

Both coordinates are functions of loss. Away from critical points the branch
equation is

```
dZ/dell = D grad L / (grad L . D grad L).
```

No spatial monotonicity is required. Each local launch is a degree-10
parameterization germ `K(t)` satisfying `F(K(t)) = rho*t*K'(t)`, with its
coefficients obtained recursively. The default launch amplitude is 0.001.
The saddle coordinates are refined from the exact enumerator's isolating
intervals. Original rational coefficients are converted once to 50-digit
Decimal arithmetic, rather than rebuilding floating f/sqrt(Lambda) and
sqrt(Lambda)*g for every evaluation.

Each branch is propagated to a fixed regular loss section using GL4, GL6,
or GL8. Logarithmic loss distance from its saddle regularizes the departure
parameter. Forward-mode Decimal derivatives carry Lambda sensitivity through
the eigendirection, local germ, parameter-dependent launch loss, and Gauss
stage equations. This differentiates the discrete transport map; refinement
is still needed to recover the continuous-map derivative.

At the section, a fixed tangent projection supplies a local scalar coordinate.
Endpoints are corrected to the level in the fixed normal direction, and that
correction is differentiated too. An exact rational interval bound verifies
strict normal monotonicity on the crossing rectangle, excluding opposite-sheet
aliasing inside that chart. This is a local coordinate with explicit refusal
when the chart cannot be verified, not a global arclength parameter.

The outer continuation uses cubic Hermite approximations of the two crossing
coordinates versus Lambda. Their difference predicts a collision. Each
prediction is corrected by two fresh shots before the actual sign bracket is
updated. Sensitivity-Newton prediction is also available for comparison.

Thus Gauss collocation is used for branch/sensitivity transport in loss;
Hermite prediction and shooting correction are used in Lambda. This is not
an implementation of an outer Gauss method with nested shooting at every
implicit parameter stage.

## Measured convergence

For the midpoint loss section, 50 Decimal digits, degree-10 germ and amplitude
0.001, GL8 gave:

| Steps per branch | Root estimate |
| ---: | --- |
| 64 | 2.177709563971202928 |
| 128 | 2.177709563954888416 |
| 256 | 2.177709563954816286 |
| 512 | 2.177709563954815994 |
| 1024 | 2.177709563954815993 |

The 512-to-1024 change is approximately 1.15e-18. GL6 with 2048 steps gave
2.17770956395481599886. A section 35% down the saddle loss gap, rather than
50%, gave 2.17770956395481599334 at GL8/1024. Changing to 60 digits, a degree-12
germ, and amplitude 0.002 gave 2.17770956395481599332 at GL8/1024.

These runs agree beyond binary64 precision. Only the shorter value above is
recommended for ordinary reporting; the long strings are retained for
convergence comparisons, not presented as certified digits.

The local gap derivative is about -1.15733. Over Lambda offsets +/-0.0001,
it varies smoothly from about -1.15743 to -1.15724. With the initial bracket
[2.1776, 2.1778], Hermite prediction needed four shooting evaluations total
(two endpoints and two corrections), versus five for sensitivity-Newton.
GL8/1024 with Hermite took approximately 20 seconds on the test machine.

## Existing launch bias

The original native shooter's root stays near 2.17770956136537 as its spatial
step shrinks from 0.008 to 0.0005. This is approximately 2.59e-9 below the
converged result. Replacing its launch points with the new Decimal germ while
retaining native GL8 transport and fractional-step event location gives:

| Native step | New-launch/native-transport root |
| ---: | --- |
| 0.004 | 2.1777095639548136 |
| 0.002 | 2.1777095639548167 |
| 0.001 | 2.1777095639548140 |

This isolates the dominant discrepancy to the existing launch path, rather
than the final root solver or native transport step size. It does not locate
the specific defect inside the production launch construction. The stored
inspector wall parameter 2.17770956082838 also differs from the new result;
neither that value nor the production launch code was changed by this experiment.

## Reproduce

```
PYTHONPATH=src python scripts/wall_continue_probe.py \
  --steps 64 128 256 512 1024 --output out/wall_continue.json

PYTHONPATH=src python scripts/wall_continue_probe.py \
  --stages 3 --steps 2048 --output out/wall_continue_gl6.json

PYTHONPATH=src python scripts/wall_continue_probe.py \
  --steps 1024 --level-fraction 0.35 --output out/wall_continue_section.json

PYTHONPATH=src python scripts/wall_launch_probe.py \
  --output out/wall_launch.json

PYTHONPATH=src python -m pytest tests/test_wall_continue.py tests/test_wall_shoot.py
```

## Warm starts from old-method brackets

`find_root_from_brackets` accepts one or more legacy intervals. It takes their
envelope rather than their intersection, pads by half the envelope width on
each side, and checks both endpoints with the new section-gap calculation.
The checked evaluations are reused by the Hermite corrector.

An ulp-wide legacy interval may miss the new root because of launch bias.
If the new signs agree, twice the best sensitivity-Newton displacement
proposes a point beyond the estimated new root. Only a sign change in fresh
new-method evaluations establishes the numerical bracket. Repairs obey an
explicit evaluation budget and optional parameter bounds; failure refuses.
Agreement of two legacy methods is not treated as an error certificate.

Generate one coarse bracket automatically, then reuse it across refinement
runs:

```
PYTHONPATH=src python scripts/wall_continue_probe.py \
  --legacy-brackets 1 --steps 64 1024 --output out/wall_warm_start.json
```

Use `--legacy-brackets 2` to compare two native step sizes, or repeat
`--seed-bracket LO HI` to supply brackets already computed by an inspector or
another run. The old method defaults to a coarse parameter tolerance of 1e-6;
it need not chase its launch-biased root to machine precision.

On the same wall, one coarse legacy bracket reduced the GL8/1024 work from
four high-precision evaluations to three: two endpoint checks and one Hermite
correction. Refinement took approximately 15.4 seconds instead of 20.1;
generating the bracket cost approximately 2.1 seconds. The result was again
2.17770956395481599332. Two legacy brackets gave the same three-evaluation
refinement but cost approximately 4.1 seconds to generate, with no further
speedup in this example. Prefer reusing one available bracket.

The focused suite passed 18 tests. It checks high-precision Gauss quadrature,
sensitivities against independent parameter perturbations, tangency and loss
constraints, GL8 refinement, a section spanning a b-fold, refusal of an
unsigned bracket, and agreement between Hermite and Newton prediction, plus
the existing shooting regressions. Warm-start tests check one/two bracket
reuse without repeated endpoints, repair of a biased ulp-wide interval,
refusal when bounds prevent repair, and rejection of invalid seed brackets.

The remaining proof obligation is to replace observed transport/launch error
convergence with validated bounds. In particular, final section projection
removes normal level drift; it does not remove accumulated tangential orbit
error. An interval containing the connection cannot be inferred from the
discrete root solver's narrow bracket alone.

## Native successor

`wall_native.py` now adapts the C99/GMP implementation. The Decimal code in
this module remains the independent oracle. See `wall_native.md` for the
native safeguards, parity checks, and finer convergence study. Neither
implementation certifies connections or a distance to the nearest wall.
