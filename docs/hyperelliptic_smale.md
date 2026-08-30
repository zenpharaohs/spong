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
statement.  `certify_flow_tube` now proves level holonomy by checking exact
inward inequalities on every lateral face of a piecewise-linear tube in
`(ell,b,y)`.  `certify_abel_gap` then excludes zero whenever two terminal
boxes lie on one sheet and the joining chart contains no branch point.
On one sheet `db/y` has constant sign, so this is exactly `b`-order
disjointness of the two crossing boxes; the Abel integral is enclosed for
continuity with the positive-genus version, not because it decides
anything the `b` order does not.  The load-bearing fact is simpler: a
saddle connection is a single trajectory, so it crosses a regular fibre
once, and two validated tubes whose terminal boxes on one exact fibre are
disjoint cannot enclose the same trajectory.  A comparison spanning a
branch point still needs the unwrapped period coordinate described above.

## Certificate contract

A hyperelliptic Smale certificate should contain:

1. Exact rational regular levels separating every distinct critical-value
   class.
2. Complete certified root disks for each `S_ell`, paired into its real
   components, with a fixed homology/sheet labelling across the slab.
3. Validated Frobenius/Poincare launch boxes for every stable and unstable
   local germ (implemented by `spong.local_certificate`; opt-in
   materialization while it remains the Python exact oracle).
4. Rational trapping-tube enclosures of the lifted holonomy (implemented),
   plus ball/interval enclosures of periods for comparisons which cannot stay
   in one sheet chart (not yet implemented).
5. For every candidate stable/unstable pair, an interval enclosure of
   `Delta_ij` (on one sheet, presently just the exact `b` gap).  Exclusion
   of zero certifies preserved order and hence absence of a connection.
   An exact wall requires interval Newton in the model parameter together
   with `Delta_ij=0`; a small floating gap is not an equality certificate.
6. Once an unstable branch enters a one-minimum or one-ended component, the
   existing exact merge-tree terminal certificate supplies its final fate.

Consequently, static complex roots and static periods are infrastructure, not
the verdict.  The verdict is validated hyperelliptic **holonomy** plus the
exact terminal component.  The holonomy and same-chart gap engines now exist;
the local launch and same-sheet holonomy composition now exist.  The remaining
global promotion step is unwrapped positive-genus comparison across sheet
transitions, plus portrait-wide orchestration of the implemented certificates.
See `local_graph_certificate.md` for the exact cone theorem and the C-backend
contract.
