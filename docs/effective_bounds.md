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
