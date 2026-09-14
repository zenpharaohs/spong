# Two-ended saddle-connection shooting

Status: development diagnostic.  This method does not yet alter a zoo wall
parameter or the production portrait certificate.

## Two layers

The intended use is deliberately split:

1. **Exploration/proposal of non-Smale parameters.**  Floating local stubs,
   pure-order two-ended shooting, several meeting levels, and the split clock
   produce a candidate wall parameter together with numerical convergence
   evidence.  This layer should be cheap enough for rheostat searches and
   zoo qualification.
2. **Certification.**  Validated source and recovery rectangles are propagated
   to a regular meeting level.  Disjoint signed discrepancy intervals certify
   opposite sides of a wall; an interval Newton or equivalent transverse
   argument encloses its zero.  This layer is more expensive and may be
   optional when a portrait is being explored rather than cited as a
   certified Morse--Smale result.

The proposal layer must never acquire a certification label merely because
its floating convergence is unusually clean.

## Witnesses

Fix an unstable half-branch of a higher saddle, a stable half-branch of a
lower saddle, and a regular loss level `c` between their critical values.
Shoot the first branch downward and the second upward (backward in descent
time).  On an oriented component of `{L = c}`, define

```
D_c(Lambda) = b_down(Lambda; c) - b_up(Lambda; c).
```

In a chart where `b` is not a local coordinate, it can be replaced by signed
level-curve arclength.  A zero of `D_c` glues the two shots into one gradient
trajectory by ODE uniqueness.  Repeating the calculation on several meeting
levels is an overdetermined witness: the inferred parameter must not depend
on where the connection was split.

Start the two proper-time clocks on fixed regular loss sections
`c_high - epsilon_high` and `c_low + epsilon_low`.  If

```
S_c = T_down(c) + T_up(c),
```

then `S_c` is independent of `c` on a connected orbit.  Away from the
connection it combines pieces of two different orbits and generally varies
with the split.  This split invariance, rather than an assumed maximum of
`S_c` at one level, is the clock witness.

## Numerical realization

[`scripts/two_ended_connection.py`](../scripts/two_ended_connection.py) uses
the materialized local invariant-manifold stubs at both ends.  Outside those
stubs, both shots solve the regular constant-loss ODE

```
dz/dL = grad(L) / |grad(L)|^2
```

with one pure GL order and no fallback.  Full versus two-half steps control
position and the collocation-stage proper-time quadrature independently; the
accepted endpoint and time are both taken from the two-half composition.
Steps are shortened to land on every clock or meeting section.  The loss-step
schedule depends only on the saddle loss gap, not on a guessed terminal
minimum.

The default nonnearest-connection experiment is run with

```
python scripts/two_ended_connection.py
```

Order, local-stub refinement, and loss-step refinement are separate axes:

```
python scripts/two_ended_connection.py --orders=-4,-6,-8 \
    --launch-levels 4,5 --step-levels 0,1
```

## First measurement

For the `nonnearest-saddle-connection` family, portrait branch 10 and the
`stable_sign=-1` half-branch of the saddle near `b=0.640274` give the following
raw GL6 roots at 4,000 loss steps:

| local-stub level | source grid error | target grid error | root Lambda |
|---:|---:|---:|---:|
| 2 | 3.878e-8 | 8.306e-8 | 2.1777095568825904 |
| 3 | 9.697e-9 | 2.077e-8 | 2.1777095621867000 |
| 4 | 2.424e-9 | 5.192e-9 | 2.1777095635127890 |
| 5 | 6.061e-10 | 1.298e-9 | 2.1777095638443131 |
| 6 | 1.515e-10 | 3.245e-10 | 2.1777095639271846 |
| 7 | 3.788e-11 | 8.113e-11 | 2.1777095639479089 |

The increments shrink by four per rung, in agreement with the recorded
second-order stub grid errors.  Second-order Richardson extrapolation gives

```
levels 5 -> 6:  2.1777095639548083
levels 6 -> 7:  2.1777095639548172
```

Their difference is 8.9e-15.  At stub level 6 the pure GL4, GL6, and GL8 raw
roots span 7.1e-15.  At stub level 4, changing the loss schedule from 2,000 to
8,000 steps moves the root by 2.4e-14.  Thus the endpoint charts, not the
regular middle integration, are the visible leading error.

At every tested raw root, meeting fractions 0.35, 0.50, and 0.65 agree to a
few binary64 ulps.  Their proper-time sums agree at the same scale.  The clock
test is non-vacuous: with five meeting levels its split-time spread is about
`5e-9` at parameter offsets of `1e-7`, and `8e-2` in the `Lambda=2` control
chamber, while falling to about `5e-15` at the two-ended root.

The extrapolated value is about `3.12e-9` above the near-target landing/sector
flip measured by the long one-ended trace.  The two calculations therefore
must not be combined into a narrow wall bracket.  The present evidence says
that the long near-saddle passage loses the incidence before the regular
two-ended BVP does; certification still requires an endpoint-error enclosure
or a validated graph-stub refinement argument.

## Independent near-slide result

The stored `near-slide-d2` case is an ordinary Morse--Smale member at
`Lambda=1`, about eight percent from a previously reported rheostat wall.  It
therefore tests the method on a different polynomial and geometry rather than
on another tuning of the first family.  The candidate pair is portrait branch
6 (the increasing-b unstable branch of the saddle near `b=-0.651233`) and the
`stable_sign=+1` half-branch of the saddle near `b=0.984296`:

```
python scripts/two_ended_connection.py --case near-slide-d2 \
    --branch 6 --target-b 0.984296 --stable-sign 1 \
    --lo 1.0798783 --hi 1.07987865 --launch-levels 2,3,4,5,6,7
```

At 4,000 loss steps, the raw GL6 roots are:

| local-stub level | source grid error | target grid error | root Lambda |
|---:|---:|---:|---:|
| 2 | 2.188e-8 | 3.375e-8 | 1.0798784747861894 |
| 3 | 5.471e-9 | 8.439e-9 | 1.0798784755911739 |
| 4 | 1.368e-9 | 2.110e-9 | 1.0798784757924196 |
| 5 | 3.420e-10 | 5.275e-10 | 1.0798784758427340 |
| 6 | 8.549e-11 | 1.319e-10 | 1.0798784758553137 |
| 7 | 2.137e-11 | 3.297e-11 | 1.0798784758584579 |

Again, both grid errors and root increments fall by four.  The last five
successive Richardson values lie between `1.0798784758595015` and
`1.0798784758595070`, giving the floating proposal

```
Lambda ~= 1.079878475859506.
```

This lies inside the historical independently reported Radau/DOP853 bracket
`[1.0798784758546, 1.0798784758605]`.  Coarse local stubs put that narrow
bracket entirely on one side, but endpoint refinement converges into it and
explains the apparent disagreement.  Those SciPy methods are deprecated in
SPONG because they bypass the backend integrators; the agreement is retained
only as an independent consistency observation, not as the current protocol.

At local-stub level 6, pure GL4/GL6/GL8 raw roots span `4.7e-14`; 2,000,
4,000, and 8,000 loss steps span `2.7e-15`.  Three meeting levels agree to a
few ulps and their proper-time sums agree to about `2e-14`.  At the stored
`Lambda=1` control, the correct pair has `D_b` between `0.135` and `0.189`
and a split-time spread of `0.164`; the other target half-branch is farther
away still.  The two witnesses therefore distinguish the ordinary stored
case from the proposed wall rather than reporting a structural identity.

## Ordinary random-ensemble screen

[`scripts/explore_non_smale.py`](../scripts/explore_non_smale.py) connects the
ordinary portrait explorer to the two-ended proposal.  For each unstable
half-branch and every lower `N`-saddle it:

1. discards the local launch prefix;
2. locates the first chord crossing of the target saddle's loss;
3. measures its distance to the saddle, with the chord's sagitta allowance;
4. normalizes by the target's nearest-critical-point spacing; and
5. sends only the ranked tail to the regular two-ended shooter.

The `N` filter is algebraic, not heuristic.  A descending connection cannot
terminate at a `B`-saddle: all `B`-saddles have the common loss `C`, so a
`B`--`B` connection cannot strictly descend.  Measuring on the target level
also matters.  In the first attempted screen, the smallest unrestricted
Euclidean approaches occurred several saddle-level gaps away and were not
wall proximity at all.

The stored `near-slide-d2` case calibrates the normalized screen at `0.1120`.
On 2,000 fresh ordinary generator cases, numbered `[200, 2200)` under master
seed `20260806` and maximum degree 5, 850 cases supplied at least one
admissible pair (1,537 pairs total).  Six cases screened below the near-slide
value.  Every one of those six had a common three-level shooting bracket in
`|log Lambda| <= 0.32`; across the top fifteen candidates, twelve bracketed.
Thus the screen is selective without being a surrogate wall verdict.

Two of the closest proposed walls were refined further:

| generator case | seed | source branch | target N-saddle | section miss / target spacing | GL6 launch-5 root |
|---:|---:|---|---:|---:|---:|
| 1587 | 1035355622 | `B:+0.855787`, decreasing b | -1.403999 | 0.00864 | 0.9960397507738134 |
| 1030 | 695079654 | `B:-0.419486`, increasing b | +0.838728 | 0.02085 | 1.0031241986028361 |

Their launch-level sequences at 2,000 loss steps were

```
case 1587:  L2 0.9960397504788539  L3 0.9960397507035836
            L4 0.9960397507597687  L5 0.9960397507738134
case 1030:  L2 1.0031242031642531  L3 1.0031241996888944
            L4 1.0031241988200452  L5 1.0031241986028361
```

The increments fall by four.  Successive second-order launch extrapolants
agree to a few `1e-15` and give the floating proposals

```
case 1587: Lambda ~= 0.996039750778495
case 1030: Lambda ~= 1.003124198530433
```

At launch level 5, pure GL4 converges to pure GL6 as the loss-step count runs
2,000, 4,000, 8,000; the finest-order differences are `1.6e-14` and `8.5e-15`
for cases 1587 and 1030.  GL6 is already at its binary64 floor.  At the root,
three meeting-level coordinates agree to a few ulps and the split proper-time
sums agree to `5e-15` or better.

The ordinary portrait also changes fate across both proposals at
`Delta log Lambda = +/-1e-3`.  Case 1587 switches between the minima near
`b=-2.40525` and `b=-0.05873`, on opposite sides of its target saddle.  Case
1030 switches between the minima near `b=1.05997` and `b=0.14666`, likewise
on opposite sides of its target.  These are reproducible floating
handle-slide proposals, not validated wall certificates.

Reproduce the screen and the refinement grid with

```
python scripts/explore_non_smale.py --generated-cases 2000 --start-case 200 \
    --jobs 8 --screen-only --out out/random-screen.json
python scripts/explore_non_smale.py --reuse-screen out/random-screen.json \
    --shoot-cases 1587,1030 --orders 4,6 --launch-levels 5 \
    --step-levels 0,1,2 --out out/random-shoot.json
```

## Directed random-ensemble screen

The same screen was applied to the existing 200-case directed degree-5
ledger.  Of its 174 multi-saddle cases, 162 completed the geometric screen,
eight returned an explicit 20-second timeout, and four remained inside long
native calls after the Python alarm.  The checkpoint is therefore marked as
an incomplete screen; those twelve cases are missing data, not negative
observations.  The directed constructor is reconstructed from each recorded
seed, rather than passing its seed through the ordinary random constructor.

The directed sample was not built to seek near-connections.  Its closest
completed target-level miss was `0.1516` of the target critical spacing, and
none beat the `near-slide-d2` calibration of `0.1120`.  Nevertheless the
closest case produced a common two-ended bracket:

```
case 139, seed 628285345, directed d4 |b|=256 uniform01
branch 11: B saddle b=-0.2236478786, decreasing b
target:    N saddle b=-1.2415454702
screen:    distance/target spacing = 0.1515536
```

At pure GL6 with 2,000 loss steps, launch refinement gives

```
L2  0.7457034522825702
L3  0.7457034531386385
L4  0.7457034533526519
L5  0.7457034534061574
```

The increments divide by four.  The three successive second-order
extrapolants agree to `5e-15`, giving the floating proposal

```
Lambda ~= 0.745703453423992
```

At launch level 4, pure GL4 moves from `0.7457034533519933` to
`0.7457034533526067` as the loss-step count doubles from 1,000 to 2,000.
Pure GL6 gives `0.7457034533526507` and `0.7457034533526519`.  Thus the
GL4/GL6 difference at 2,000 steps is `4.5e-14`, while GL6 step refinement is
already at `1.2e-15`.  At every root, the three meeting-level roots and
split proper-time sums agree to a few `1e-15` or better.

Only one of the top twenty directed candidates bracketed in
`|log Lambda| <= 0.32`; the others were clean non-brackets or reported that
the target stable half-branches could not furnish the requested regular
shooting interval.  This supports using the target-loss distance as a
screen, not as a wall score.

## Multiple connections in one portrait

The screen retains every `(source half-branch, target N-saddle)` pair and
assigns it a stable candidate id.  It must not collapse a portrait to its
closest pair.  In the completed directed sample, 29 portraits have at least
two candidates below 1.5 target spacings, but only case 18 has two below one
spacing (`0.829` and `0.989`).  They use distinct source half-branches and
distinct N targets.  The first did not bracket in the tested rheostat range;
the second could not yet shoot either target stable half-branch.  Case 139's
two additional remote-target candidates were likewise not shootable.  Thus
this sample contains no proposed simultaneous multiple connection.

For a one-parameter rheostat, each pair should first be rooted independently.
Distinct root intervals are separate codimension-one wall crossings by the
same family.  A multiple connection is proposed only when two or more root
intervals overlap after launch, step, and pure-order refinement.  Generically
two independent connections are codimension two, so coincidence in one
parameter calls for a symmetry or algebraic dependence, or for evidence that
the two witnesses are not independent.  Certification must propagate every
source and target rectangle at the same parameter and retain itinerary and
component labels; spatial proximity of several manifolds is not enough.

Reproduce the directed screen and case-139 refinement with

```
python scripts/explore_non_smale.py --mode directed --screen-only --jobs 8 \
    --case-timeout 20 --per-case 8 --out out/directed-screen.json
python scripts/explore_non_smale.py --reuse-screen out/directed-screen.json \
    --shoot-cases 139 --order 6 --launch-levels 2,3,4,5 \
    --out out/directed-launch.json
python scripts/explore_non_smale.py --reuse-screen out/directed-screen.json \
    --shoot-cases 139 --orders 4,6 --launch-level 4 --n-steps 1000 \
    --step-levels 0,1 --out out/directed-order-step.json
```

## Remaining qualification

Before promotion to a wall certificate, the method still needs:

1. completion or explicit external termination of the twelve stiff directed
   screen cases, while retaining the present close nonconnections;
2. a coordinate-independent signed level-curve discrepancy when `b` ceases
   to be a valid local coordinate;
3. an interval or validated bound propagating both stub rectangles to the
   meeting level and converting their discrepancy width through
   `dD/dLambda`; and
4. explicit itinerary/component checks when several invariant manifolds pass
   through the same small spatial neighborhood.
