# Effective Bounds, Near Poles, and Compute Boxes

This note records a lesson from random portrait inspection: theorem-level
boundedness is not the same thing as an effective plotting bound.

## Backbone Poles

The backbone is

```text
a*(b) = B(b) / A(b)
```

where `A(b) = E[g(bX)^2]`.  A genuinely unbounded finite-`b` backbone event is
a pole: `A(b0) = 0` while `B(b0) != 0`.

For the standard positive training measures used by spong, the Morse/generic
contract includes positivity of `A` on the real line.  Therefore a real pole
of the backbone is not a difficult Morse case; it is a failure of the
denominator/positivity condition and should be caught by the exact positivity
test on `alpha = A`.

However, this does not give an effective bound.  A nearby training
distribution can make `A` strictly positive but very small near a would-be
pole.  Then the model is technically inside the Morse/generic regime, but

```text
a*(b) = B(b) / A(b)
```

can make a huge finite excursion.  Such cases may produce bounded orbits whose
practical enclosing box is enormous.

This is a likely source of a future zoo family:

```text
near-pole Morse portraits
```

These would test the difference between qualitative boundedness and
metrological boundedness.

## Bounded Orbits Are Not Effectively Bounded

Lojasiewicz-type convergence results say that bounded gradient trajectories
terminate in critical points.  They do not supply a useful plotting box.

In spong, a bounded invariant-manifold branch can leave the naive padded hull
of the critical points and later return to a finite target.  The branch is
bounded and topologically ordinary, but the enclosing box needed to draw it
faithfully may be much larger than the box suggested by the critical points
alone.

The random quadratic case discovered at seed `2195314464` is the basic
example.  It is a degree-2 `f=g` portrait with two minima and two saddles.  The
lower unstable branch from the bottom saddle targets the lower finite minimum,
but the initial compute box was too narrow in `a`; the branch exited the box
before returning to its target.  Widening the box made the same trace capture
cleanly.

This exposed a contract distinction:

```text
critical-point containment  !=  branch containment
```

Critical-point containment is an enumeration/ledger contract.  Branch
containment is a plotting/instrument contract.

## Implementation Rule

For bounded unstable branches, premature compute-box exit is not acceptable
when the target is a finite minimum.  It means the compute box was too small.

Current policy:

1. Trace finite unstable branches first.
2. If a branch exits the current compute box before capturing its finite
   target, expand the relevant `a` side and retry.
3. Once finite connections are settled, trace unbounded branches and stable
   separatrices against the final enlarged box.

This makes the expensive behavior demand-driven: spong only pays for a larger
box when a finite branch proves the existing box inadequate.

## Display Views and Trace Boxes

The readable picture and the integration domain are not the same object.

The default display view is chosen to show the finite critical skeleton with
moderate padding.  That is the right first picture for inspection: minima,
saddles, local separatrix geometry, and branch adjacency should not be pushed
into a tiny corner by a far excursion.

The trace box must be larger.  A stable separatrix can leave the display view
and later re-enter it; if tracing stops at the display boundary, the renderer
has no data for the later visible chord.  Segment clipping can only preserve
computed curve pieces.  It cannot display manifold history that was never
traced.

Current policy:

1. Build a display view from the critical-point hull, unless the user supplies
   `--view`.
2. Trace against a modestly larger box around that view.
3. Render by clipping the computed branches back to the display view.
4. When the user asks for a new interactive view that exceeds the current
   compute box, recompute the portrait for a new box before rendering.

Thus:

```text
display view  ⊂  trace box  ⊂  legal maximum box
```

The distinction matters most for correctness-by-looking.  A small view should
not silently erase separatrix portions that pass through that view merely
because those portions lie beyond a too-small integration boundary.

## Open Problem

Find useful a priori bounds for bounded invariant manifolds in terms of the
polynomial data and the moment sequence.

Even if the backbone itself has an effective bound under the positivity
contract, bounded invariant manifolds are harder.  They can make excursions
away from the backbone before returning to finite critical points.  A good
bound would immediately improve performance and make the box contract more
predictive; until then, adaptive branch-driven enlargement is part of the
instrument.

## Near Saddle-Nodes and Batch-Moment Resonance

The random quadratic case discovered at seed `323153948` is a useful
stochastic-training thought experiment.  It is a degree-2 `f=g` portrait with
three minima and three saddles, including:

```text
local min:  b = -0.8484098186,  a = 1.0320435751,  L ~= 8.67e-05
saddle:     b =  0.0674421703,  a = 0.8329925420,  L ~= 1.50e-02
global min: b =  1,             a = 1,             L ~= 0
```

The local minimum is close to the global minimum in both parameter space and
loss value, and it has a nearby saddle.  This suggests a possible
saddle-node mechanism under perturbations of the training distribution.

The moment sequence is part of the loss geometry.  Changing the training
distribution changes `A`, `B`, `N`, and the reduced potential `u(b)`.  A
small perturbation of the moments can therefore move a nearby min/saddle pair,
or even annihilate/create it when `N` develops a double root and `u'' -> 0`.

For minibatch methods, the effective moment sequence fluctuates from batch to
batch.  Near such a saddle-node threshold, stochastic training is not merely
"gradient descent plus noise."  The sampled batch can temporarily change the
local Morse structure:

```text
batch moments fluctuate
=> A, B, N fluctuate
=> near-saddle-node geometry fluctuates
=> basin boundary / shallow well shifts, appears, or disappears
=> stochastic descent may cross a bottleneck that full-batch descent respects
```

This is a candidate mechanism for stochastic resonance in batch-trained
methods.  The full-distribution portrait gives the baseline Morse skeleton;
batch-induced portraits explain when the optimizer is being pushed across a
geometric bottleneck rather than merely escaping a static well by noise.

Future harness idea: fix `f=g` from seed `323153948`, perturb the moments
along a one-parameter family, and track the relevant roots of `N` near
`b=-0.8484` and `b=0.06744`.  The expected event is a collision of that
local min/saddle pair, visible as loss of squarefreeness and `u'' = 0`.

## Overparameterized Fits and Local-Minimum Access

The random case discovered at seed `1158725111` is a useful
overparameterization stress test:

```text
deg f = 1,  deg g = 17
```

This is the single-polynomial-neuron analogue of fitting a high-degree
polynomial model to essentially linear data.  The resulting portrait has
fourteen critical points and a rich Morse skeleton.  In the observed portrait,
the extra degrees of freedom make it dynamically easy for descent to terminate
at a finite minimum, but that minimum need not be global.

This is a small exact-dynamics version of a familiar overparameterization
claim:

```text
many parameter choices fit the data well
=> many descent routes can reach a good-enough finite attractor
=> the attractor reached by a given descent route may be only local
```

The same experiment run in the opposite direction,

```text
deg f high,  deg g low
```

is underparameterized in the approximation sense.  Empirically it produces a
simpler phase portrait: fewer representational degrees of freedom give the
flow fewer ways to create competing saddle/minimum arrangements.

This pair is worth keeping because it phrases a neural-net training slogan in
something measurable.  Overparameterization is not just "more parameters make
optimization easier"; in this model it can mean "more nearby Morse basins make
descent to some local finite minimum easier."

The seed `1158725111` also exposed a numerical instrumentation issue.  One
finite unstable branch from a `B`-source saddle was nearly horizontal in the
`(a,b)` plane: small `|Delta b|`, large `|Delta a|`.  A branch step budget
based only on `|Delta b|` over-resolved that branch until the continuation
engine hit `max_steps` before reaching the target.  Finite-branch tracing now
uses a conservative Euclidean chord budget when that is more informative.

## The compute box has an upper bound too, and it is metrological

The rule above ("if a branch exits before capturing, expand and retry") and the
blanket `_trace_box` inflation both push the box OUTWARD.  Measurement says the
box also has a hard *upper* limit, for a reason unrelated to topology: **past a
certain radius the certificates cannot measure anything, and the flow does not
need to be integrated.**

Found by bisecting a regression in the founding MATLAB-parity gate
(`test_tricky_branch_parity`).  It passes at `1ca22ed` and fails at `3938060`
(`angle_energy` 3.70e-14 -> 2.01e-08 against a `< 1e-12` bound).  The cause is
NOT the closed-form `_sym2_eigh` (verified bit-identical to `np.linalg.eigh`
here) but `_trace_box`, which enlarged the integration box 1.35x:

| | b-range | arc | vertices | E | E/vertex |
|---|---|--:|--:|--:|--:|
| `1ca22ed` | [-7.069, -2.738] | 4.330 | 4001 | 3.70e-14 | 9.26e-18 |
| `3938060` | [-11.049, -2.738] | 8.310 | 4001 | 2.01e-08 | 5.02e-12 |

The branch is *escaping*: it runs to the b-boundary in BOTH commits, so no
finite box is ever the right answer for it.  It exits along the **degenerate
b-pole**, not a diagonal — `asymptote_certificate` extrapolates slope 16116
against target sqrt(d_eff) = 3.3166 (residual 1.06e+03), which is the
certificate correctly declining to apply.  `rim_directions` already documents
this direction: `bdot ~ -C_inf/b^2`, integrable in closed form as
`b^3/3 = -C_inf t`, with `C_inf` an EXACT rational.

**Where the energy actually comes from.**  Per-chord decomposition of the
2.01e-08:

| b window | E | % of total | med \|grad L\| | med floor | ratio |
|---|--:|--:|--:|--:|--:|
| [-4.0, -2.7) | 7.73e-20 | 0.0% | 8.66e-03 | 5.84e-10 | 1.5e+07 |
| [-7.0, -5.0) | 3.41e-14 | 0.0% | 2.96e-03 | 4.06e-07 | 7.3e+03 |
| [-9.0, -7.0) | 5.65e-11 | 0.3% | 1.41e-03 | 1.01e-05 | 1.4e+02 |
| **[-11.1, -9.0)** | **2.00e-08** | **99.7%** | 7.86e-04 | 1.24e-04 | **6.3** |

99.7% of it is in the arc that exists only because of the inflation.  `|grad L|`
DECAYS (as `C_inf/b^2`) while its evaluation floor RISES (the cancellation scale
grows with `|b|`), so the two converge: the direction of `grad L` carries ~7
digits at b=-4 and **less than one** at b=-11.  `angle_energy` already documents
this exact regime — "measures its own evaluation noise, not the curve (seen on
far valley stretches where |grad L| ~ C_inf/b^2)" — but its guard fires only on
`ng < floor`, a cliff that never triggers, so a continuous loss of meaning is
reported as curve error.  **0 of 3999 vertices were skipped.**

Note the branch is a single `shallow` zone with `switches=0`: it is traced
entirely by the Hadamard fixed point, not the engine.  This is not an integrator
accuracy problem, and cannot be fixed by more steps.

**Rule.**  The outward box limit is the radius at which `|grad L|` falls to
within a chosen digit budget of `g_floor` — a *metrological* bound, computable a
priori from `C_inf` and the coefficient scales, not a heuristic multiple of the
view.  Beyond it: attach the closed-form b-pole tail, certify with `C_inf`, and
integrate nothing.  Two consequences for the code:

1. `_trace_box`'s 1.35x inflation should be replaced by (or intersected with)
   this bound.  Inflating costs resolution exactly where resolution is already
   meaningless, since the chord budget stays fixed while the arc grows.
2. `angle_energy`'s noise guard should be graded (skip when `ng < K*g_floor`
   for a digit budget K, K >> 1) rather than a cliff at `K = 1`.  Otherwise the
   certificate silently reports evaluation noise as geometry.

### Replace the flow, don't just stop — but replace it with the EXACT backbone

The rule above says where to stop.  The better statement is that past that radius
the flow should be *replaced* by an analytically known one.  Two candidates, and
measurement picks the second decisively.

**Candidate 1, the principal-terms (leading form) flow.**  With `A ~ a_L b^2d`
and `B ~ b_L b^d`, slaving gives `a -> (b_L/a_L) b^-d`, and substituting into
`bdot = 2aB' - a^2 A'` the leading term cancels identically (the Wronskian
cancellation recorded in `C_inf`'s docstring), leaving `bdot = -C_inf/b^2`,
solvable as `b^3/3 = -C_inf t`.  **Measured, it converges far too slowly to
serve.**  On the tricky escaping branch `P b^2 / C_inf` reads

| b | -3 | -4 | -5 | -7 | -9 | -11 |
|---|--:|--:|--:|--:|--:|--:|
| `P b^2/C_inf` | 2.12 | 3.76 | 3.58 | 2.99 | 2.58 | 2.30 |

still 2.3x off at b=-11, with the residual decaying only as `O(1/b)` (fitted
exponent 0.93, 0.98 on the last two windows; roughly `1 + 14/b`).  Three digits
would need `|b| ~ 1.4e4`.  Asymptotically correct, metrologically useless at the
radii that actually occur.

**Candidate 2, the exact backbone.**  Far out the separatrix simply *is* the
backbone, and `a*(b) = B(b)/A(b)` is an exact rational function — no asymptotics
required, correct immediately rather than in the limit:

| b | chord digits `log10(\|grad L\|/g_floor)` | fp residual | fp iters | `w_s` |
|---|--:|--:|--:|--:|
| -4 | 6.27 | 0 | 2 | 1.79e-19 |
| -11 | 0.25 | 0 | 1 | 9.29e-36 |
| -20 | **-3.27** | 0 | 1 | 2.81e-45 |
| -200 | **-16.49** | 0 | 1 | 1.60e-81 |

`w_s` collapses like `~b^-36` here while the fixed point returns residual 0 in
one iteration — there is nothing left to solve.  Chord digits go **negative**
past `|b| ~ 12`: `grad L` is entirely beneath its own evaluation floor, so
`angle_energy` is measuring noise, which is exactly how the box inflation broke
the founding gate.

**So the far-field chart is: `a = a*(b)`, emitted analytically at whatever
resolution rendering wants, certified ALGEBRAICALLY (`|w_s|` below tolerance
against an exact rational) rather than geometrically (chord-vs-gradient angles
that have run out of digits).**  The handoff radius is where `w_s` becomes
numerically indistinguishable from 0 relative to `a*` — which on this branch has
already happened by `b = -4`, far inside either compute box.

This is the same architectural pattern as the shallow/deep handoff, with the
certificate swapped rather than the integrator: a region where a closed form is
exact, entered on a computable criterion.  Note what it does NOT require —
`C_inf`, the leading form, or any asymptotic expansion.  Those describe the
*time parameterization* `bdot = -C_inf/b^2`, which the portrait never needs,
since the drawn object is the curve, not its schedule.
