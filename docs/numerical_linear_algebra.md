# Numerical linear-algebra audit

`np.linalg` is not a neutral implementation detail in SPONG.  Model
coefficients, centered critical jets, and Morse signatures may carry exact
information across dynamic ranges for which a matrix rounded entrywise to
binary64 no longer carries the same rank or inertia.

The production audit has the following disposition.

| Former operation | Data provenance | Disposition |
|---|---|---|
| Hessian `eigvalsh` during critical-point elaboration | Exact rational centered jet | Replaced by the exact trace and determinant, a high-precision algebraic square root, and `det/lambda_large` for the smaller eigenvalue. |
| Hessian `eigvalsh` in the minimum-basin certificate | Same critical point, but formerly reevaluated globally in FP64 | Replaced by the already-conditioned critical-point spectral data. |
| Rotation of a centered jet into an eigenframe | Exact rational polynomial followed by a severely anisotropic frame | Performed coefficientwise in high precision; only the transformed jet is rounded for the C graph kernel. |
| Poincare polynomial evaluation | Rotation and normal-form pullback produce all powers of both normal coordinates, although the physical gradient is only quadratic in `a` | The complete polynomial `adj(DT) F(T(z))` is composed coefficientwise in high precision and rounded once into the native C kernel.  The omitted positive determinant changes only parametrization.  Runtime subtraction of theoretically removable terms is forbidden. |
| Generic Newton `solve` in the Python Gauss oracle | Arbitrary user vector field in FP64 | Replaced by row equilibration, pivoted elimination, a pivot-resolution guard, and an a posteriori backward-error check. |
| Native GL4/GL6/GL8 stage Newton solve | Structured 4x4, 6x6, or 8x8 block system assembled from analytic 2D Jacobians | Uses the same row equilibration and partial pivoting, rejects pivots below \(64\epsilon_{64}\), and certifies the correction against the original unequilibrated system by a normwise backward-error bound. |
| Poincare map `solve` | A certified near-identity 2x2 Jacobian | Replaced by a row-scaled closed form with a determinant guard. |
| RRE normal-equation `solve` | Empirical differences of FP64 graph iterates | Removed.  RRE coefficients are computed directly from the SVD, avoiding the squared condition number. |
| Euclidean `norm` calls | Planar geometry or diagnostic residuals | Replaced in production by `hypot` or a scaled sum of squares, avoiding overflow and underflow. |

One production `np.linalg` operation remains intentionally:

- `np.linalg.svd` in the optional RRE comparison.  Its input is empirical
  FP64 iterate data rather than exact model data; rank revelation is the
  required operation; singular values are explicitly guarded; and a proposed
  extrapolate is accepted only when an independently evaluated graph-transform
  residual improves.  RRE is not needed to construct the production stubs.

Occurrences retained under `tests/` are independent or adversarial oracles:

- determinant, condition-number, solve, and inverse comparisons for small
  Gauss stage matrices whose conditioning is explicitly bounded by the test;
- Euclidean norms on moderate synthetic test data;
- one `eigvalsh` call that deliberately demonstrates the loss of the small
  negative eigenvalue after entrywise FP64 rounding.

The audit rule is therefore: exact or conditioned model data must never be
reclassified by a generic FP64 factorization.  Floating linear algebra is
permitted only when its input is intrinsically floating, its conditioning is
measured, and its output is independently certified or safely rejected.

The critical-chart handoff follows the same rule.  At the stub endpoint it
compares the global gradient direction with the independently evaluated
pulled-back polynomial, checks both the Poincare map's Jacobian Hadamard ratio
and the graph's invariance angle, and requires the global signal to exceed its
componentwise polynomial-evaluation rounding floor by at least 1024.  The
allowed angular disagreement is the corresponding forward-error budget.  A
certified graph is doubled while these tests remain valid.  A branch that
cannot establish overlap before graph contraction or invariance is lost
terminates explicitly as `abort_conditioning_handoff`; it is never passed to
the global continuation engine on a cosine heuristic alone.

That refusal also records the binary64 spectral-resolution margin

\[
\rho_{\rm spec} =
\frac{\min_i |\lambda_i|}
     {\epsilon_{64}\max_i |\lambda_i|}.
\]

The diagnostic classifies both eigendirections as robustly resolved at the
scale of binary64 coefficient perturbations only when
\(\rho_{\rm spec}\geq64\).  In the first 660-case out-of-sample corpus, all 181 local
handoff refusals had \(\rho_{\rm spec}<64\), and 170 had
\(\rho_{\rm spec}<1\).  Thus those refusals identify an arithmetic boundary,
not an arbitrary launch-radius rule.  A higher-degree normal form may still
help the small residual set with \(1\leq\rho_{\rm spec}<64\), but it must not
be presented as a binary64 certificate without a separate backward-error
argument.
