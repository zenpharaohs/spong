# The wall theory: separatrix handoff via the complex domain

*2026-08-29.  Companion to the level-pencil machinery in
`scripts/pencil_tree.py`.  Status: working derivation; phase-1 sweep in
`scripts/wall_sweep.py`; the far-field (small-eps) experiment is phase 2.*

## Setting and the exact normal form

Loss `L(a,b) = C - 2aB(b) + a^2 A(b)`, backbone `a*(b) = B/A`, valley
profile `u = C - B^2/A`.  With the deviation coordinate `w = a - a*(b)`,

    L(a,b) = u(b) + A(b) w^2        -- exactly, no remainder.

The model is its own algebraic normal form.  All critical points lie on
`w = 0`.  In the complex domain four divisors must be kept distinct:

    A                           transverse zero divisor,
    denominator(B/A)            poles of the valley chart a*,
    D = denominator(B^2/A)      poles of the reduced backbone u,
    H = numerator(u')           critical divisor.

Common factors make these genuinely different.  Reporting raw zeros of `A`
as poles of `u` creates removable ``poles`` and false proximity alarms.
`spong.complex_structure` reduces first and then uses exact Lehmer-Schur
(Schur-Cohn) disk counts or a cheaper exact linear Rouche witness; floating
roots are proposals only.

## The orbit equation and its singular points

Orbits of the gradient flow, graphed over `b`:

    dw/db = 2Aw/L_b - a*'(b),
    L_b   = u' + A' w^2 - 2A a*' w,     u' = B N / A^2 .

Quasi-static balance recovers the engine's slaved graph
`w = a*' u' / (2A)` exactly.  At each critical point `b_c` the graph
equation is singular.  The branch slope is fixed by an eigenvector of the
full Hessian, and the invariant exponent is the ratio of the two
eigenvalues of the linearized flow (the same spectral ratio used by the
Poincare chart).  The tempting scalar

    2 A(b_c) / u''(b_c)

equals that ratio only when the valley shear `a*'(b_c)` vanishes.  In
general `(w,b)` is a non-orthogonal shear of `(a,b)`, so the scalar is not a
spectral invariant.  Frobenius and Poincare describe the same local
invariant germ only after this metric/shear bookkeeping.  Consequences:

* at its own nonresonant departure saddle the selected separatrix has an
  analytic Taylor germ whose first slope is the unstable Hessian
  eigenvector and whose later coefficients obey an algebraic recursion;
* the fractional powers `(b-b_c)^nu` describe the transverse foliation
  (peel-off of neighbouring orbits) and the branch's ARRIVAL at
  downstream critical points (at a minimum it carries
  `(b-b_m)^{nu_m}`, `nu_m > 0`: a funnel);
* resonance tests must use the actual spectral ratio, not `2A/u''`;
* exact complex root disks certify which fixed divisors are absent from a
  launch disk.  They do **not** by themselves certify convergence of the
  nonlinear invariant graph: movable complex singularities may occur
  first.  A coefficient majorant or validated analytic continuation is
  still required;
* a downstream funnel can support a capture theorem only after its local
  invariant sector and the branch's entry into it are both validated.

The linear transport along the valley has integrating factor

    mu(b) = exp( - int 2A^3/(BN) db )
          = e^{poly(b)} * prod_r (b - r)^{-rho_r},

where the rational integrand must be reduced before its denominator roots
`r` are enumerated.  Those real and complex roots, with the
partial-fraction residues, determine the characteristic exponents and
phases of the linear valley transport.

## Wall-blindness of the pointwise algebra (a theorem)

Crossing a wall (saddle-connection locus) changes Smale data only; the
merge tree, and every pointwise algebraic invariant of the level pencil,
is constant along the wall family.  This is forced by the separation of
distance-to-non-Morse (algebraic) from distance-to-non-Smale
(transcendental): a pointwise algebraic wall criterion would contradict
it.  The wall must be an INTEGRAL of the algebraic data, not a value.

## The wall functional

The unstable branch of `s1` passes the col `s2` iff the loss it still
carries on arrival exceeds the col height:

    Sigma = [ L|_{W^u(s1)} - u ](b_2),     wall  <=>  Sigma = 0.

Dissipation-along-the-path versus col height: the same currency as the
level bar and the order sweep.

## The far-field integrable limit and the splitting integral

In the monomial far field `L_0 = alpha a^2 b^{2m}` (`m = deg g`) the
gradient flow conserves

    K = b^2 - m a^2,        grad K . grad L_0 == 0,

and orbits are hyperbola branches.  Writing `L = L_0 + L_1`, the
connection condition for `s1 -> s2` is

    K(s1) - K(s2) + M = O(eps^2),
    M = int_{gamma_0} grad K . grad L_1 dt
      = - oint_{K = k*} (grad K . grad L_1) / L_{0,b} db .

Two structural facts make this unusually clean:

1. the measured drift is of the first integral itself, so NO
   Abel/divergence weight appears (the usual dissipative-Melnikov
   complication cancels by construction);
2. `gamma_0` is a CONIC -- genus zero -- so at leading order `M`
   evaluates by residues into algebraic terms plus logarithms of
   algebraic quantities, with poles at the complex roots pulled back to
   the hyperbola.

Hence, at leading order, the wall location is EXPLICIT in the roots:
transcendental only through logs of algebraic numbers, computable to
certified precision by interval arithmetic.  Genuine hyperelliptic
periods of the pencil curves `y^2 = S_l(b)` enter only at the next
order (the infinitesimal-Hilbert-16 structure).  The smallness
parameter `eps` shrinks in the far field, so the formula is most
accurate exactly where tracing dies: the horizon saddles are its home
turf.

The reusable exact layer is now
`spong.hyperelliptic.certify_genus_zero_integral`: after the conic
parametrization supplies the reduced rational differential, it certifies the
complete complex pole
divisor, records the algebraic residue--log root sum, excludes real path
poles by Sturm counting, and returns an independent exact-rational enclosure
of the definite integral.  Constructing and bounding the model-specific
`L_0+L_1+O(eps^2)` decomposition remains part of the far-field wall proof,
not something this generic rational-integral certificate assumes.

## Open care points

(a) the correct `L_0 + L_1` decomposition and smallness parameter for a
GIVEN wall family (the central-path families of `connect_saddles.py`
live at O(1) `b`, where `eps` is not small: they test wall-blindness
and calibrate `Delta K`, not the asymptotics; a far-field wall family,
certified by tracing in normalised units, is the designed experiment);
(b) the organizing-centre bookkeeping (`k*`, and the honest `O(eps^2)`
statement) for heteroclinics between saddles at different loss levels.

## The two-chart conjecture

The valley graph chart fails where `L_b = 0` (swing-past segments); the
loss-level chart (the order sweep's parameterisation) is valid exactly
there.  Conjecture: a two-chart atlas -- Frobenius series near critical
points, monotone-level transport between them -- covers every branch
with no arclength continuation, leaving the wall integral `M` as the
only transcendental object in certification.

## The hyperelliptic Smale target

The level pencil is not merely a conditioning diagnostic.  With
`y=Aa-B`, each regular level is the hyperelliptic curve
`y^2=B^2+(ell-C)A`, and the gradient defines algebraic level-to-level
holonomy on the total family.  Stable/unstable incidence should be measured
as an Abel-coordinate gap on a common fibre.  Static periods alone do not
decide that gap; their Gauss-Manin transport, the inverse Abel map, and the
Frobenius endpoint germs do.  The exact equations and certificate contract
are in `docs/hyperelliptic_smale.md`.

## Validation plan

Phase 1 (`scripts/wall_sweep.py`, existing certified family): sweep the
central path `theta(t)`, `t: 1 -> 0`, recording the traced separation
`delta(t)` (ground truth), the validated reduced complex clearances, the
actual per-critical-point spectral ratio, and
`Delta K(t)` for the target pair.  Predictions: `delta -> 0` at `t = 0`
by construction; every pointwise algebraic column varies smoothly and
without signature through the wall (wall-blindness, confirmed
negatively); `Delta K` trends but does not vanish (out of regime --
its residual calibrates `eps` at O(1) `b`).

Phase 2 (far-field family): construct the connection at large `|b|`,
certify `t*` by tracing in normalised units, and test the ordering
`|t_{K+M} - t*| << |t_K - t*|`, both shrinking under far-field scaling.
If that ordering appears, the geometric coordinate exists and its
formula is the residue-log expression above.

## First results (2026-08-29)

The original phase-1 output in
`out/wall_sweep-nonnearest-attachment.json` predates the corrections above:
its `minImA`, `minEpsN`, and `nu` columns came from floating roots of
unreduced polynomials and the non-invariant `2A/u''` scalar.  It is useful as
a historical probe, not a complex certificate, and must be regenerated by
the revised script before quantitative use.  Qualitatively, delta reaches
-1.1e-8 at t = 0; the pointwise algebraic columns vary smoothly through the
wall with no signature, consistent with wall-blindness.  Delta K trends
37 -> 14.28
without vanishing: the measured eps at O(1) b, and the reason phase 2
needs a far-field family.  The old `nu_src` column cannot support its
reported tracing-conditioning interpretation; that question must be
retested using the true spectral ratio.

Frobenius launch race (`scripts/frobenius_launch.py`): on tractable
saddles the jet agrees with the stub germ to ~1e-6..1e-8 and with the
flow (fine RK referee) to ~1e-9 at ten times the stub radius
(555999196, b = -1.284: w ~ 1.0 at the test point).  That is evidence of
reach on those examples, not a convergence certificate.  Prototype build
cost (~100-230 ms/saddle,
FD-Jacobian Newton) exceeds stub materialisation; the exact recursion
is the obvious optimisation.  Open items: (1) the jet Newton fails at
the B-root saddles of 953953598 (--pow2) -- near-transverse eigenvector
vs ill-conditioned FD Jacobian, not yet isolated; (2) at the horizon
saddle b = 196608 the STUB itself is degenerate (zero-length curve --
the fp64 launch failure that defines the class), so stub-relative radii
test nothing there.  The series-seeded comparison is now labelled
self-consistency because its RK initial value comes from the same series.
The old gap `R_empirical ~ 5.5e3 << R_fixed ~ 2e5` is exactly why the
nearest fixed complex root cannot be called a convergence radius: it may
reflect double-precision conditioning, a movable nonlinear singularity, or
both.  A majorant calculation is the next certification step.
