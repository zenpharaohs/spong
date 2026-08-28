# Stable branches: escape, labels, and tails

Stable manifolds are the easy half of the portrait, and SPONG has been
treating them as though they were the hard half.  This note records why they
are easy, what replaces tracing them, and what each claim actually rests on.

Throughout, `L = u(b) + A(b)(a - a*(b))^2` with `A > 0` (psi-positivity),
`a* = B/A` the backbone, `u = C - B^2/A`, and `d = d_eff`.  Stable manifolds
are traced by **ascent**, `x' = +grad L`.

## 1. Escape is decided by a level, not by a box

A bounded forward ascent orbit converges to a critical point: the flow is
gradient, critical points are isolated, there are no cycles, and there are no
maxima because `L_aa = 2A > 0`.  Every critical point lies on the backbone
with `L = u(b) <= U*`, where

    U* := max_b u(b)

and `U*` is attained at a **saddle** -- `u' = -B*N/A^2`, so the critical points
of `u` are the roots of `B` (level `C`) and of `N`, and the maxima among them
are the `u'' < 0` ones, which are exactly the saddles.  Minima are irrelevant
here since a minimum is a repeller for ascent.

Therefore:

> If a traced sample has `L(p) > U*`, the orbit can never reach a critical
> point, so the branch is unbounded.

One exact rational comparison.  No Sturm chain, no box, no geometry.

**The bar is sharp.**  The fiber of `{L >= c}` over `b` is all of `R` when
`u(b) >= c` and two disjoint rays when `u(b) < c`.  For `c > U*` no fiber is
full, so the superlevel set is exactly two components, neither containing a
critical point.  For `c <= U*` some `b0` has `u(b0) >= c`, its whole vertical
line lies in the set and joins the two sides, so the set is connected and
contains the saddle attaining `U*`.  Below the bar, level information alone
can never decide.  That is where the merge tree runs out on the stable side.

**It is a one-sided witness.**  Clearing the bar proves escape.  *Not*
clearing it proves nothing: the branch may be bounded (a saddle connection) or
merely under-traced.  Those two are separated only by tracing further, and the
required length diverges as a connection is approached -- the trace length is
the Morse-Smale margin.  No positive certificate for boundedness exists; a
saddle connection is codimension one, so binary64 can bracket it and never
confirm standing on it.

Meaning check: a connecting orbit from `s'` down to `s` lies entirely in
`{L <= u(s')}`, so clearing the top saddle level *is* "above every possible
connection partner".

**Complement.**  The half-plane `H = {a <= a0}` with `a0 < min a*` is
forward-invariant under ascent (`a' = 2A(a - a*) < 0` throughout) and contains
no critical point (they all sit on the backbone).  It certifies escape
*regardless of level*, so it can fire on a branch whose loss has not cleared
the bar.  The test is one positivity question, `B - a0*A > 0` on `R`, at
degree `2d`.  Mirror with `a1*A - B > 0`.

## 2. The exact invariant

Away from the backbone the `B` terms are negligible -- precisely, dropping them
needs

    |a| >> |a*(b)|      and      |a| >> 2|B'(b)/A'(b)|

which is the half-plane condition, *not* a far-field condition.  Under that
truncation the orbit equation becomes separable:

    db/da = (a/2) * A'(b)/A(b)

so

    J(a, b) = a^2/2 - integral_b 2A/A' ds

is conserved **exactly** -- no series, no truncation order.  And `A/A'` is
rational, so the integral is closed form by partial fractions:

    J = a^2/2 - b^2/(2d) - 2*c1*b - 2*sum_j [A(rho_j)/A''(rho_j)] * ln(b - rho_j)

where `rho_j` are the roots of `A'` (degree `2d-1`, Sturm-isolable) and `c1` is
the constant term of the polynomial part of `A/A'`.

The logarithms are residues at the critical points of `A`.  Those same
`rho_j` are the regime-2 traps of section 3, so one formula covers both
far-field behaviours: near a `rho_j` the log dominates and the tail is
vertical; away from all of them the `b^2/(2d)` term dominates and the tail is
the offset diagonal

    b = sqrt(d)*a - A_{2d-1}/(2*d*A_{2d}) + O(1/a)

`J` locks onto its final value within tens of steps of leaving the backbone,
not at the box edge.  On tricky-d11, four of seven stable branches have `J`
correct to 1e-6 at the *stub endpoint*, and the rest within 120 steps of a
104000-step trace.  The lock point tracks `|a| >> max(|a*|, 2|B'/A'|)` case by
case.

## 3. Two tails, and the number that chooses between them

Escaping stable branches come in two families.

**Diagonal.**  `|b|` past `A`'s outermost root modulus; the tail is the offset
diagonal above, labelled by `J` (equivalently by `K0`, with the caveat in
section 4).

**Vertical.**  `b` pinned at a root `b*` of `A'` with `A''(b*) < 0` -- a local
maximum of `A`, attracting for the ascent `b`-dynamics.  With `w = b - b*`,

    dw/da = kappa * a * w,   kappa = A''(b*)/(2A(b*)) < 0
    =>  w = w0 * exp(kappa (a^2 - a0^2)/2)

Gaussian pinning, not slow drift.  Restoring `B` (`A'(b*) = 0` makes `-2B'`
the leading numerator) gives the actual asymptote

    b_inf(a) = b* + 2B'(b*)/(A''(b*) * a) + O(a^-2)

a vertical line with a `1/a` approach -- the branch riding a slaved
equilibrium, structurally the same object as the backbone slaved continuation
used for unstable branches.

**The dispatcher is one number:** compare `|a|` against `2|B'/A'|`.  Finite
means the `B`-truncation holds and `J` applies; blowing up means the branch is
sitting on a trap and wants the vertical form.  The traps are enumerable
before any tracing -- roots of `A'` with `A'' < 0`.

Higher degree does not widen the strip in `b` (the wall converges to `|b| = 1`)
but it does put **more** roots of `A'` inside it -- measured 0.17 -> 2.00 real
roots within `|b| < 1` from degree 1 to 11 -- so more traps, and more branches
that crawl.

## 4. Labels, separation, and what is actually certified

Branches escaping along the same ray differ only in their label, and their
separation in `b` is a power law, `dK0 / (2 sqrt(d) a)`.  So they stay apart
and the separation is exactly what a user zooming in would inspect.

Since distinct orbits never meet, non-crossing is a **theorem**, not something
to check.  Where the tail is supplied analytically and the family is ordered by
a conserved label, disjoint label intervals discharge the contact test rather
than performing it.  That is stronger than scanning polylines in three ways:

* **Reach.**  Non-crossing holds on `[handoff, inf)`; the scan only covers the
  traced arc.
* **Conditioning.**  Near-collinear tails are exactly where an orientation
  determinant cancels; a label gap is a difference of scalars.
* **Regime 2.**  Two branches converging on the same `b*` sit at separation
  `~exp(-|kappa| a^2 / 2)`, thousands of digits below representable.  Any
  polyline scan sees identical curves.  But both satisfy the same linear
  equation, so `w/w'` is constant: ordering preserved, crossing impossible.

**Caveat on labels obtained by tracing.**  An error committed at position `b`
produces a permanent label shift `dK0 = 2b*db`, and coordinate roundoff is
`~eps*b` per step over `~b/ds` steps, so

    dK0 ~ 2*eps*b^3/ds

The label degrades like `b^3` along the path, and *more* steps make it worse.
Measured on tricky-d11: halving `ds` at fixed endpoint changed `K0` by up to
0.14, all one sign.  This is why the handoff belongs early and why a wide
compute box actively damages the label it would extend.  `J` largely sidesteps
this by locking before roundoff has any path to accumulate over.

## 5. What this does not do

* It says nothing about **bounded** stable branches.  Those are exactly the
  saddle connections, and they still need whatever the near-field machinery
  can prove.
* It does not transfer to unstable branches.  The same algebra gives a linear
  equation for `w = a - a*` and a conserved constant `C = mu (w - w_part)`
  with `mu = exp(integral 2A/u')` -- but `mu` is the contraction factor, so `C`
  is the amplitude of a decaying mode and is exponentially ill-conditioned.
  Measured drift of `J` along unstable branches is 80% to 1200%.
  The compensation is that unstable orbits collapse onto a *single* slaved
  curve `w1 = -a*' u' / (2A)` with no free constant, which is why the existing
  slaved continuation needs no label.
* Unstable **captures** still need the merge tree.  That is where the
  remaining work is.

## 6. Launch directions at a saddle

At a critical point, with `M := B A' - A B'` and `a*' = -M/A^2`,

    L_ab = 2(a* A' - B') = 2M/A = -2A a*'

so the Hessian is `[[2A, -2A a*'], [-2A a*', L_bb]]`, and applying it to the
backbone tangent `(a*', 1)` gives

    H (a*', 1)^T = (0, L_bb - 2A a*'^2)^T = (0, u'')^T

The image is purely vertical, so **the backbone tangent is an eigenvector iff
`a*' = 0`.**  It is *not* generally true that unstable branches leave along the
backbone; that happens only in the dead-neuron regime, where `a* ~ C b^-d`
makes `a*'` vanish to machine precision.  Measured angles between the unstable
eigendirection and the backbone tangent: 0.000 deg at `a*' = 7e-4`, 0.96 deg at
6e-2, 28.2 deg at -2.38, 43.9 deg at 41.0.  Note the B-saddle at `b = 1.35756`
has `a* = 0` but `a*' = -2.38`, so sitting on the axis does not imply
alignment.

What `L_aa = 2A > 0` does say is that the `a`-direction is strongly stable, so
the unstable direction stays away from it -- measured 45 to 90 degrees off the
`a`-axis in every case.  That constrains where it cannot point, not where it
does.

Consequence for basin combinatorics: the sector an unstable branch launches
into is a local `2x2` eigenvector computation at the saddle, well defined
whenever the eigendirections are distinct -- which Morse-ness already
certifies.  No claim about backbone alignment is needed.
