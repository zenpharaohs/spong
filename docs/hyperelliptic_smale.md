# Hyperelliptic holonomy for the Smale attaching map

## What the pencil already decides

For a regular loss level `ell`, define

    y = A(b)a-B(b),
    S_ell(b) = B(b)^2+(ell-C)A(b).

Then

    L(a,b)=ell  <=>  y^2=S_ell(b),
    a=(B+y)/A.

Thus every real level component is a real component of the hyperelliptic
curve `X_ell: y^2=S_ell(b)`.  Its real branch points decide the sublevel
components and hence the merge/Reeb tree.  That is the Morse part currently
computed by `spong.merge_tree` and `scripts/pencil_tree.py`.

The merge tree does **not** decide which unstable saddle branch attaches to
which minimum, nor whether a stable and unstable separatrix form a
saddle-saddle connection.  Those are the Smale data.  They depend on the
transport of points between distinct fibres `X_ell`, not merely on the
topology of any one fibre.

## The exact lifted gradient flow

On `X_ell`, with `N=A'B-2B'A`, the physical gradient is

    L_a = 2y,
    L_b = (B+y)(N+A'y)/A^2.

Along any noncritical gradient trajectory, use `ell=L` as the independent
variable.  Since `dL/dt=-|grad L|^2` and `db/dt=-L_b`,

    db/dell = L_b/(L_a^2+L_b^2).

Differentiating `y=Aa-B` along the normalized gradient gives the equivalent,
branch-point-regular formula

    dy/dell = [2Ay + (A'a-B')L_b] / (L_a^2+L_b^2).

These are algebraic functions on the total hyperelliptic surface.  They are
implemented exactly in `spong.hyperelliptic`.  Unlike the divided formula,
the second expression remains regular at an ordinary branch point `y=0`;
only a critical point makes the normalized field singular.  A
Frobenius/Poincare germ at a saddle supplies the initial branch; the equations
above carry it from one regular fibre to the next.

## The genus-zero residue--logarithm stratum

The algebraic-plus-logarithmic formula applies exactly after a meromorphic
differential has been pulled back to a rational parametrization of a conic.
If the reduced pullback is `N(x) dx/D(x)`, exact polynomial division and a
squarefree pole divisor give

    integral N/D dx = Q(x)
        + sum_{D(alpha)=0} R(alpha)/D'(alpha) log(x-alpha).

Thus a definite conic integral is an algebraic term plus algebraic
coefficients times logarithms of algebraic quantities.  This is the
far-field Melnikov fast path described in `wall_theory.md`; it is not a claim
that generic positive-genus Abelian integrals are elementary.

`certify_genus_zero_integral` records that exact root-sum identity, certifies
every complex pole with Lehmer--Schur disks, excludes real poles by Sturm
counting, and encloses the real definite integral by an independent
exact-rational midpoint remainder.  The value certificate therefore does not
rely on unvalidated complex floating logarithms.

## Abel coordinates and the connection gap

Choose an oriented real component of `X_ell` and a continuously transported
base point.  An Abel coordinate begins with

    xi_ell(p) = integral^p db/y.

For genus `g`, the holomorphic differentials are

    db/y, b db/y, ..., b^(g-1) db/y.

Their level derivatives are the second-kind differentials

    partial_ell (b^k db/y) = -A(b)b^k db/(2y^3).

Hermite reduction modulo exact differentials gives the Gauss-Manin system for
the period/Abel vector.  The inverse Abel map then supplies a globally
unwrapped coordinate on each real oval, including passage through a branch
point where raw `b` ceases to be a good coordinate.

Let `u_i(ell)` be an unstable branch crossing and `s_j(ell)` a stable branch
crossing on the same oriented component.  The Smale decision function is

    Delta_ij(ell) = xi_ell(u_i(ell))-xi_ell(s_j(ell))  modulo the period.

The branches connect exactly when `Delta_ij=0`.  Inside a critical-value-free
slab their order cannot change without such a zero.  The existing
`spong.order_sweep` computes a sampled tangent-coordinate shadow of this
statement.  The validated transport uses several complementary scalar charts.
On a strict sheet, `certify_sheet_flow_tube` eliminates `y` and proves a
trapping tube for `b(ell)`.  Near `y=0`, straight `(ell,b)` chords can leave
the real fibre even when both endpoints lie on it.  There the local Poincare
graph is cut at a fixed rational `b`, loss is eliminated by

    ell = C + (y^2-B(b)^2)/A(b),

and `certify_b_parameter_flow_tube` proves a scalar tube for `y(b)`.  Its
regularity obligation is `L_b != 0` throughout every slab.  Once this tube
strictly brackets the requested loss, exact subbox pruning projects it back
onto that common fibre.  The older two-coordinate `(b,y)` tube remains a
conservative fallback.  In all charts the vector field is proved regular on
the entire trapping slab as well as inward-pointing on its lateral faces.

`certify_fibre_separation` first excludes equality whenever two terminal boxes
are disjoint in either `b` or `y`.  This direct test also handles opposite
sheets and needs no period coordinate: a noncritical gradient trajectory
crosses a regular loss fibre only once.  `certify_abel_gap` then
records the oriented gap whenever two terminal boxes lie on one sheet and the
joining chart contains no branch point.
On one sheet `db/y` has constant sign, so this is exactly `b`-order
disjointness of the two crossing boxes; the Abel integral is enclosed for
continuity with the positive-genus version, not because it decides
anything the `b` order does not.  The load-bearing fact is simpler: a
saddle connection is a single trajectory, so it crosses a regular fibre
once, and two validated tubes whose terminal boxes on one exact fibre are
disjoint cannot enclose the same trajectory.  A comparison spanning a branch
point needs the unwrapped period coordinate only when a global order coordinate
is requested.

## Certificate contract

A hyperelliptic Smale certificate should contain:

1. Exact rational regular levels separating every distinct critical-value
   class.
2. Complete certified root disks for each `S_ell`, paired into its real
   components, with a fixed homology/sheet labelling across the slab.
3. Validated Frobenius/Poincare launch boxes for every stable and unstable
   local germ (implemented by `spong_local_launch_decimal`, with
   `spong.local_certificate` retained as the Python exact oracle).  The hard
   optional path replays the selected quadratic Poincare coordinate map
   exactly, traps the
   invariant graph between rational piecewise-linear faces, checks the chart
   Jacobian throughout every slab, and may cut either a fixed-loss or fixed-b
   departure section without moving rectangles artificially closer to the
   saddle.
4. Rational trapping-tube enclosures of the lifted holonomy (implemented).
   Direct rectangle separation on a common fibre is implemented and is enough
   for pairwise connection exclusion.  Ball/interval enclosures of periods
   remain useful for a globally unwrapped order coordinate, but are not a
   prerequisite for this yes/no decision.
5. For every candidate stable/unstable pair, disjoint validated crossing
   rectangles on one exact regular fibre.  An interval enclosure of
   `Delta_ij` additionally records order where an Abel chart is available.
   An exact wall requires interval Newton in the model parameter together
   with `Delta_ij=0`; a small floating gap is not an equality certificate.
6. Once an unstable branch enters a one-minimum or one-ended component, the
   existing exact merge-tree terminal certificate supplies its final fate.

Consequently, static complex roots and static periods are infrastructure, not
the verdict.  The verdict is validated hyperelliptic **holonomy** plus the
exact terminal component.  The local launch, chart-switching holonomy,
common-fibre gap, and portrait-wide finite-plane orchestration are implemented.
Unwrapped positive-genus comparison across sheet transitions remains separate
global-order machinery.
See `local_graph_certificate.md` for the exact cone theorem and the C-backend
contract.

## Pointwise orchestration and the margin boundary

`spong.smale.certify_finite_plane` is the fail-closed **development**
orchestrator.  It first uses exact loss order and the fact that only an
N-saddle can be the lower endpoint.  A validated unstable cone also certifies
which of the two sublevel components adjacent to its source saddle it enters;
if the target saddle is absent from that component, the pair is excluded
without global continuation.  Only the remaining half-branch pairs request
validated tubes on a common fibre.  The cheap path uses a fixed-sheet scalar
tube.  A branch whose launch is too close to the backbone requests the exact
Poincare graph tube and the fixed-b `y(b)` handoff.  The latter trapping tube
and its target-fibre projection now execute as one GMP C operation; exact
real-case parity with the Fraction oracle is pinned.  Floating Poincare and
portrait branches supply tube centres but carry no part of the verdict:
changing or refining those centres can turn a refusal into a proof, never
manufacture a positive certificate.  Local launch certificates are also
materialized incidence-by-incidence: stable germs of a source and unstable
germs of a target that cannot occur in any admissible saddle connection are
not paid for.

The returned status is deliberately
`certified_finite_plane_morse_smale`, not the less specific
`certified_portrait`: identifying equilibria of a chosen compactification is a
separate certificate.  Every missing launch or branch, unresolved exact loss
order, tube refusal, or overlapping terminal rectangle produces `unresolved`.
The ordinary fixed-sheet tube and its initial fibre projection now execute in
the GMP backend.  The native Frobenius cone also supplies the fixed-`b`
section used by the complementary `y(b)` handoff.  The old piecewise numerical
graph tube and two-coordinate Python/Fraction fallback are development-only
and disabled by default; they can be replayed explicitly with
`allow_development_fallbacks=True`.  The serialized result still says
`production_ready: false` while portrait-wide orchestration and incidence
reduction remain Python; no frontend may promote that status.

This pointwise result is not yet a backward-error radius.  Given a declared
weighted norm on the finite coefficient data `(f,g,mu[0:M])`, a certified
lower radius additionally requires the same Morse and pair-exclusion
inequalities to hold uniformly on the admissible input box.  The efficient
far-from-wall query is therefore `margin >= R?`: validate one requested box
and stop.  Adaptive doubling and bisection can estimate the largest certified
lower radius.  A separately validated saddle-connection wall supplies an
upper radius; a floating two-ended shot remains a proposal until interval
Newton encloses its zero.
