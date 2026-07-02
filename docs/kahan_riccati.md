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
