# Ordinary portrait certification

The ordinary certificate assesses whether a proposed phase portrait is
credible. Its four obligations are:

1. The complete inventory of invariant half-branches has connections consistent
   with the exact critical-point inventory and merge tree.
2. Branches attach to their critical points with the correct local geometry.
3. Branch trajectories are credible pseudo-orbits of the given flow, assessed
   independently of construction primarily through orthogonality to loss levels.
4. Unbounded branches are acceptably oriented as they leave the compute region.

This is the routine health check for the ordinary Smale case. Validated
trajectory enclosures, trapping tubes, or comparable exceptional work are
reserved for claims that require them, such as resolving a suspected saddle
connection. They are not a missing prerequisite for every ordinary portrait.
The ledger must distinguish the exact structural facts from geometric residual
evidence. A branch near a suspected connection can require escalation without
imposing that cost on every other branch.

## Technology and remaining integration

| Obligation | Existing machinery | What needs scrutiny or completion |
|---|---|---|
| Complete, consistent connections | Exact Sturm inventory; exact merge tree and terminal component tests; oriented inventory with opposing launch sections | Keep identity and endpoint evidence explicit. Merge-tree compatibility alone need not determine the attaching map. |
| Attachment geometry | Centered local jets and invariant graph residuals; local level-section witnesses; arrival charts and preconnector capture tests | Independently assess arrival geometry and distinguish drawing connectors from measured trajectories. A capture basin certifies fate, not the fidelity of a connector drawn to the minimum. |
| Pseudo-orbits | Native gradient/chord diagnostics; backbone measurements; contact audit | Replace sampling-dependent energy as the primary scrutiny quantity; justify arithmetic resolution, sampling error, coverage, and oriented flow tests. Calibrate an explicit acceptance policy. |
| Unbounded ends | Exact superlevel/escape tests; shrinking backbone funnels; asymptote residuals | Separate fate from boundary orientation, and apply the appropriate diagonal or backbone end model at the actual boundary or declared analytic extension. A fitted far asymptote need not be accurate at an arbitrary finite box. |

Contact checks support the branch geometry obligations. Step-size and turn
controls in the trajectory constructor support producing a good proposal;
they are not substitutes for an independent check of the proposal. The final
audit also uses its own turn check for representation/contact reliability.

## Independent level-normal measurements

`spong.credibility.level_normal_reference` is a native GMP reference evaluator.
For each chord `d`, it evaluates the exact model gradient `g` at the exact
midpoint of the supplied binary64 endpoints and reports

\[
  r^2 = \frac{\det(d,g)^2}{(d\cdot d)(g\cdot g)}.
\]

This is the squared sine of the misalignment with the level normal. It is
dimensionless. Unlike the old sum of squared transverse chord lengths, it
does not shrink merely by subdividing a straight misaligned trajectory.
Each chord also gets the exact signs of the oriented dot product and endpoint
loss change. Orthogonality alone does not distinguish ascent from descent.
Stable branches stored away from their saddle are checked in ascent;
unstable branches are checked in descent.

The coefficient evaluation, midpoint construction, ratio, and signs use exact
rational arithmetic in C/GMP. Only the displayed ratio is rounded to binary64;
an exact nonzero flag survives ratio underflow. Repeated points, stationary
midpoints, and nonfinite input are explicit unmeasured entries. The additive
loss constant cancels. No numerical ODE solver is used.

This supplies reference evidence, **not a new production acceptance gate**.
There is deliberately no tolerance chosen to make the current zoo pass.

Its limits matter:

- A midpoint chord test measures the proposed sampled geometry. Curvature and
  step spacing affect its approximation to a smooth trajectory's tangent.
  It is not a bound on the unsampled curve.
- Exact arithmetic does not recover a transverse offset lost by rounding the
  trajectory to global coordinates. Very stiff branches may require retained
  local/chart coordinates or an explicitly justified representation allowance.
- Backbone proximity by itself is not invariance. The backbone is generally
  not an integral curve: on `a=a*(b)`, the gradient's a-component vanishes while
  the backbone tangent can have nonzero a-component. A complementary check
  must use the applicable invariant-graph or asymptotic relation.
- The existing raw angle energy and its gradient-resolution guard remain
  historical diagnostics. Their sampling dependence and heuristic arithmetic
  guard must not silently become a universal acceptance threshold.

`scripts/credibility_probe.py` measures every chord and reports local launches,
continuation, and terminal representations separately, including all omitted-
measurement reasons and evaluation time. It is the calibration tool for deciding
where an inexpensive floating check suffices and where conditioned or exact
evaluation is needed. Exceptional validated transport remains a separate tool.

### Initial measurement: chord approximation matters

On `dead-neuron-far-saddle-d3`, the first nonzero chord of the far saddle's
positive unstable launch gives a nearly 90-degree midpoint misalignment.
The local series predicts a transverse chord-midpoint sag at fixed b of
`-4.543450056191112e-16`. Independently, one exact rational Newton correction
of the midpoint's a-coordinate to align its normal with that chord is
`+4.543447289127111e-16`. The opposing values agree to about six relative
decimal places. This identifies a chord-approximation effect amplified by
stiffness; it is not explained by a sub-ulp a-coordinate rounding allowance.
It also does not establish that every large angle in other regions has the
same explanation. Drawing connectors and near-backbone tails need their own
scrutiny. An acceptance policy must use the available smooth chart/tangent
or account for its geometric representation error, rather than judging each
straight chord as though it were the exact trajectory.
