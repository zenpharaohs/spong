# Validated Frobenius/Poincare graph launch

## The local statement

The production Hadamard/Poincare graph transform is an accurate floating
proposal, but a small fixed-point residual is not an enclosure of the exact
algebraic saddle.  `spong.local_certificate` supplies the missing local
statement independently.

Let the Sturm isolating interval contain the critical coordinate `b_c`, put
`a_c=B(b_c)/A(b_c)`, and use the binary64 Hessian eigenframe as an **exact
dyadic rational** matrix.  Substitution in the exact polynomial gradient
produces interval coefficients for

    du/dt = F_u(u,s),        ds/dt = F_s(u,s).

The uncertainty in `b_c`, including the fact that the dyadic frame is only an
approximate eigenframe, remains in those coefficients.  The constant gradient
is zero by the exact critical-point identity; it is imposed structurally
rather than recovered by subtracting overlapping intervals.

For each manifold and each orientation the certifier searches a bounded
dyadic family of cones

    u = orientation*t,      |s| <= K*t,      0 <= t <= R.

It selects the time orientation `sigma` in which the requested eigendirection
departs from the saddle, removes the common factor `t`, and proves the exact
rational inequalities

    orientation*sigma*F_u/t > 0,
    sigma*F_s(t,-K t)/t + K*orientation*sigma*F_u(t,-K t)/t > 0,
    K*orientation*sigma*F_u(t,+K t)/t - sigma*F_s(t,+K t)/t > 0.

These are the transverse and two Nagumo face conditions.  Morse
hyperbolicity and the invariant-cone theorem then put the selected one-sided
local invariant manifold inside the cone.  This is the enclosure form of the
graph transform: it certifies a graph in `|s/u| <= K`, without claiming that
the current linear cone resolves all of its higher Frobenius coefficients.

## Exact Poincare section and holonomy handoff

The centered potential is integrated from the centered gradient with constant
term zero.  This preserves the quadratic saddle cancellation exactly.  Two
cone faces are found whose loss ranges are disjoint, and an exact rational
level between them is selected.  Rational bisection localizes the unique
crossing slab.  Mapping that slab to

    b,
    y = A(b)a-B(b) = L_a/2

gives a rational `(b,y)` rectangle over the exact loss section.  The crossing
is locally regular because the certified transverse field is nonzero there;
global regularity of the whole fibre remains the job of `certify_fibre`.  No
midpoint is asserted to be an exact point of the fibre; the theorem is that
the desired separatrix crosses the rectangle.

`spong.hyperelliptic.certify_flow_tube_from_launch` uses this rectangle as the
first trapping-tube box.  Both directions are supported: stable launches move
toward increasing loss and unstable launches toward decreasing loss.  All
later centres are proposals.  Exact interval face inequalities, not those
centres, prove the global lifted-flow enclosure.

## Deterministic refusal and work bounds

The Python oracle returns a fixed integer status:

| code | meaning |
|---:|---|
| 0 | validated |
| 1 | invalid input |
| 2 | singular rational frame |
| 3 | invariant cone unresolved within budget |
| 4 | regular loss section unresolved within budget |
| 5 | rational endpoint-bit budget reached |

Its work record contains coefficient count, cone tests, reach halvings, slope
doublings, section bisections, and peak rational endpoint size.  A refusal is
therefore a bounded algorithmic result, not a tolerance-dependent false
certificate.

The ordinary degree-2 qualification model closes all eight stable/unstable
oriented launches at `R=0.1`; the test suite replays every positive exact face
margin and the handoff of a decreasing-loss launch.

The handoff itself is tested on the same model
(`tests/test_local_certificate.py`).  On 2026-08-29 five of the eight
launches closed into rational trapping tubes seeded from lifted stub or
traced-branch vertices; both unstable section rectangles at the far saddle
`b=-9.445` contained `y=0` (`y` in `[-0.042, 0.018]` with cone slope
`K=2.4e-4`).  The cause is the stretching of the cone by `A`: the cone is
thin in `(a,b)`, but `y=A(a-a*)` and `A` is in the thousands there, so the
transverse slack `K t` becomes a `y`-width of about `2A|v01|Kt`.  At a
backbone-tangent saddle the branch's own departure from the backbone is
second order, `y ~ a*'u''(b_c)(b-b_c)/2`, linear in the reach `R`; the cone
must hold the branch's curvature `s ~ c u^2`, so `K ~ cR` and the slack is
`~2A|v01|cR^2`, quadratic.  Slack over signal is proportional to `R`, so
the remedy is a *smaller* reach.  The reach loop now owns the section stage:
a rectangle that fails either contract halves the reach and re-closes the
cone.  The two contracts are (i) flow box: `|grad L|^2` excludes zero on the
rectangle, exactly what the tube demands of its first knot; (ii) one sheet:
`y` is one-signed on it, which is what a same-sheet comparison downstream
means.  The second is not implied by the first: `L_b` can exclude zero on a
rectangle that straddles `y=0`, and such a launch hands off but then
inflates, because the `2Ay/|grad L|^2` term of `dy/dlevel` straddles with a
spread of order `A*width(y)`.  On 2026-08-30 with both contracts all eight
launches hand off and their tubes follow the separatrices through the long
tails of the continuation test, the backbone-hugging escape included.

What this does not yet cover is the genuine far field: along an
increasing-loss direction neighbouring orbits diverge from a stable
separatrix and tube radii only grow, so a very long run fattens until it
meets something.  A validated tube's terminal box cannot be tightened
without new information; a comparison should be made at a fibre near the
saddles, and following a branch to the box wall will want the slaved
asymptotics of the stable-escape work rather than level slabs.

## C backend shape and timing boundary

The oracle is deliberately arranged like a future GMP C kernel:

- two rectangular interval-coefficient arrays for `(F_u,F_s)` plus arrays for
  centered loss and `y`;
- exact dyadic frame entries and rational critical-centre endpoints;
- fixed integer budgets for reach, slope, section bisection, and endpoint
  bits;
- an integer status, exact `R`, `K`, three face margins, exact section level,
  `(b,y)` endpoints, and fixed work counters as output;
- after a reach halving, the slope search resumes from the previous closing
  slope over four rather than from the initial slope (`K` scales like the
  reach, so the new slope is expected near `K/2`; the extra octave costs
  one test).  This is a cost policy, not a soundness one: any slope whose
  face inequalities close is a valid cone.

The C port should preserve these statuses and counters and be differentially
tested against the `Fraction` oracle before it replaces any production path.
Timing should report three quantities separately:

1. the existing floating Poincare graph proposal;
2. exact interval launch certification, excluding already-shared exact Sturm
   enumeration;
3. proposal plus certification end to end.

For now `sturm.materialize_validated_launches` is opt-in.  This keeps current
portrait timings comparable while the Python exact oracle is intentionally
slow, and establishes a stable contract for the later C benchmark.
