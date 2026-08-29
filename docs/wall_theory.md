# The wall theory: separatrix handoff via the complex domain

*2026-08-29.  Companion to the level-pencil machinery in
`scripts/pencil_tree.py`.  Status: derivation complete; phase-1 sweep in
`scripts/wall_sweep.py`; the far-field (small-eps) experiment is phase 2.*

## Setting and the exact normal form

Loss `L(a,b) = C - 2aB(b) + a^2 A(b)`, backbone `a*(b) = B/A`, valley
profile `u = C - B^2/A`.  With the deviation coordinate `w = a - a*(b)`,

    L(a,b) = u(b) + A(b) w^2        -- exactly, no remainder.

The model is its own normal form.  All critical points lie on `w = 0`.
`A` is simultaneously the transverse stiffness, the denominator of `a*`,
and the polynomial whose complex zeros are the poles of `u` (the psi
strip is where the valley walls collapse).

## The orbit equation and its singular points

Orbits of the gradient flow, graphed over `b`:

    dw/db = 2Aw/L_b - a*'(b),
    L_b   = u' + A' w^2 - 2A a*' w,     u' = B N / A^2 .

Quasi-static balance recovers the engine's slaved graph
`w = a*' u' / (2A)` exactly.  At each critical point `b_c` the equation
has a regular singular point of Briot-Bouquet type with indicial
exponents `{analytic, nu_c}`,

    nu_c = 2 A(b_c) / u''(b_c) = (transverse eigenvalue)/(valley
    eigenvalue) of -Hess L,

so the Frobenius indicial equation IS the Poincare resonance condition,
and the analytic Frobenius solution IS the stub germ in graph
coordinates.  Consequences:

* at its own departure saddle (`nu < 0`) the separatrix is PURELY
  ANALYTIC: a Taylor series whose coefficients obey a rational
  recursion -- algebraic jets to all orders;
* the fractional powers `(b-b_c)^nu` describe the transverse foliation
  (peel-off of neighbouring orbits) and the branch's ARRIVAL at
  downstream critical points (at a minimum it carries
  `(b-b_m)^{nu_m}`, `nu_m > 0`: a funnel);
* resonance (`nu` a non-negative integer, log terms) is decidable in
  advance by exact arithmetic;
* the radius of convergence of the launch series is the distance from
  `b_c` to the nearest COMPLEX root of `B N` (or zero of `A`): the
  complex portrait is the certified launch radius;
* `nu_m > 0` funnels convert capture from a radius heuristic into a
  theorem: enter the certified disk inside the funnel and termination
  at the minimum is guaranteed.

The linear transport along the valley has integrating factor

    mu(b) = exp( - int 2A^3/(BN) db )
          = e^{poly(b)} * prod_r (b - r)^{-rho_r},

the product over ALL roots `r` of `B N` -- real and complex -- with
exponents the partial-fraction residues.  The complex roots are the
characteristic exponents and phases of the valley transport.

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

## Validation plan

Phase 1 (`scripts/wall_sweep.py`, existing certified family): sweep the
central path `theta(t)`, `t: 1 -> 0`, recording the traced separation
`delta(t)` (ground truth), the pointwise complex portrait (pole `Im`s,
N-pair `eps`, the new per-critical-point invariant `nu`), and
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
