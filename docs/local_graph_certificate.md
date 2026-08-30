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

## C backend shape and timing boundary

The oracle is deliberately arranged like a future GMP C kernel:

- two rectangular interval-coefficient arrays for `(F_u,F_s)` plus arrays for
  centered loss and `y`;
- exact dyadic frame entries and rational critical-centre endpoints;
- fixed integer budgets for reach, slope, section bisection, and endpoint
  bits;
- an integer status, exact `R`, `K`, three face margins, exact section level,
  `(b,y)` endpoints, and fixed work counters as output.

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
