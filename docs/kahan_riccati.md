# Design note: Kahan–Li anadromic Riccati steps and spong's folds

Reference: W. Kahan and R.-C. Li, *A family of anadromic numerical methods
for matrix Riccati differential equations*, Math. Comp. **81** (2012),
233–265.  Kahan & Li define the solution of a matrix Riccati equation PAST
a pole analytically (the Generalized Inverse Property) and construct
anadromic schemes that compute it — continuing gracefully *through* poles
that lie directly on the solution path.  Mechanism: the Riccati flow is
the projection W = YX⁻¹ of a LINEAR flow; a pole of W is X dropping rank —
a regular point of the lifted flow on the Grassmannian.  Kahan's linearly
implicit scheme is equivalent to Euler on the lift, hence pole-transparent;
its step map is a Möbius transformation, regular at infinity.

## The mapping to spong

- The direction of a traced curve lives on RP¹; the slope m = dw/db
  transported along the flow obeys a Riccati equation (projectivized
  linear dynamics).  **A fold is a pole of m on the solution path** —
  verbatim Kahan–Li.
- The Phase-3 continuation engine's slow/fast charts are the two affine
  charts of RP¹ = Gr(1,2); chart-switching with R_SWITCH + hysteresis is
  the MANUAL Grassmannian navigation.  A Kahan/GIP step for the direction
  variable is the AUTOMATIC version: linearly implicit (one linear solve,
  no Newton), self-adjoint (anadromic — doctrine-compliant), and the fold
  is not an event.
- Same circle of facts as the Phase-2 "order-6 fluke" (GL4 on y' = y²):
  Gauss stability functions are Padé rationals, Riccati flows are Möbius.

## Refinement candidates (not scheduled; record only)

1. charts: position + projective direction with Kahan-stepped direction —
   folds become non-events; keep chart-switching as the independent
   cross-check (two mechanisms agreeing at former folds = a certificate).
2. render: contour tracer — closed level curves cross vertical tangency
   twice per circuit; projective direction removes those events.
3. atlas: the degenerate backbone poles (our weakest certificates) are
   candidates for a GIP-style lift: define past the singularity
   analytically, compute with a scheme preserving the defining property.
4. cross-repo: the matrix Riccati equation is load-bearing in the
   polynomial-root untrainable-net repo; a `kahan` addition to the
   integrator module would serve both instruments from one trusted core.

## Addendum: scope corrections and the gauge slate (2026-07-02)

**Anadromicity does NOT confer pole passage.**  Kahan's pole transparency
comes from LINEAR implicitness (polarization makes the step map Möbius —
one branch, regular at infinity, Euler-on-the-lift).  Gauss methods on a
quadratic RHS are QUADRATICALLY implicit: two algebraic branches — the
observed d=17 Newton ping-pong was branch alternation of a non-Möbius
implicit map.  Symmetric ≠ sphere-regular.

**Exact scope of the Riccati statement.**  The state ODEs are not Riccati
and their fold "poles" are chart artifacts (the state stays finite) —
cured by coordinates, not schemes.  The trajectory DIRECTION m = v_w/v_b
obeys exactly  dm/dt = J21 + (J22 − J11)m − J12 m²  (Jacobian = 2-jet of
L): a genuine scalar nonautonomous Riccati, quadratic in m, hence
Kahan-polarizable.  That is the precise entry point.

**No explicit anadromic one-step method exists** (adjoint of explicit is
implicit); the alternatives are parasitic (two-step midpoint) or
separable-only (Verlet).  IMM/IRK4-GL "good on all counts" is a theorem.

**Gauge slate for the dispatcher trigger (Phase-4 comparison):**
1. spectral ratio κ = 2A/|u''| — cheapest; PROVEN to lie at u-inflections
   (survives only via fixed-point self-certification);
2. implicit–explicit differentiator (mse-bundle heritage):
   IMM − EMM = (h³/4)·J²f + O(h⁴) — isolates the J²f elementary
   differential; prices the implicit advantage directly; jet-computable;
3. Hessian commutator rate (MTC ω_s) — frame rotation / nonnormality axis;
4. RHS cancellation ratio  (|2Aw/P| + |a*'|)/|2Aw/P − a*'| — measures the
   ACTUAL failure mode of the slow chart (evaluation cancellation on the
   slaved floor, which is the slaving balance itself); cannot spike at
   inflections; zero extra cost.
Criterion: with always-implicit Gauss, the shallow-water handoff is a
chart-CONDITIONING decision, not a stability decision; the trigger should
measure the cancellation, not a spectral proxy.  (4) is the candidate
simplest cat; (2) is the right tool wherever implicit-vs-explicit is a
live question (demos; other repos).
