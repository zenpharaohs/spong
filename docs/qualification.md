# Qualification boundary

SPONG distinguishes an **advertised input sandbox** from the larger adversarial
stress zoo.  A case outside the sandbox may compute correctly and is useful
evidence, but it does not determine the calibrated reliability claim.  A case
inside the sandbox can still be reported as `fp64_unresolved`; refusal to issue
a certificate is an intended outcome.

## Total public outcome

`spong.resolve` gives every valid model exactly one terminal response:

1. `certified_non_morse`: exact analysis proves that the critical set is not
   Morse; no geometry is attempted.
2. `morse_numerically_unresolved`: the model is Morse exactly, but either a
   prospective arithmetic policy refuses it or the a-posteriori geometry
   certificate does not close.
3. `certified_portrait`: exact Morse analysis and the complete geometry and
   topology ledger certify the returned portrait.

The unresolved result is structured, not an unanswered call.  It includes the
first refusal reason, all applicable margin diagnostics, the exact enumeration,
and any partial portrait that was safely computed.  Expected arithmetic
failures are translated into this result; programming errors are deliberately
not swallowed.

`ResolutionPolicy` contains prospective root-collision, Hessian-condition, and
scaled-local-gamma thresholds plus the geometry escalation budget.  Thresholds
remain explicit policy data until calibration is frozen.  The default policy
does not mistake exploratory pilot bands for proved constants: it observes
the margins and permits the a-posteriori certificate to decide.

Qualification has three successive gates.

## 1. Finite exact input and construction

Only the moment prefix actually consumed by the model is relevant:

\[
  \mu_0,\ldots,\mu_{2d},\qquad
  d=\max(\deg f,\deg g).
\]

An MGF hypothesis can imply useful moment bounds for a population, but it is
neither necessary nor sufficient for the realized computation.  In particular,
Gaussian and finite empirical distributions have entire MGFs at every scale,
while their finite moment prefixes can still create poorly scaled monomial
arithmetic.

The input profile therefore records:

- degree and required moment order;
- maximum exact numerator/denominator bit length;
- raw moment exponent range;
- variance and scale-free standardized central-moment growth; and
- exact coefficient profiles for \(f,g,A,B,N\).

Raw moment size is primarily a bound on exact preprocessing resources.  The
derived \(A,B,N\) profiles are the more direct binary64 admission measurements.
This distinction permits an exactly compensated scaled distribution: large raw
moments can cancel against correspondingly scaled \(f,g\) coefficients and
leave the actual model well conditioned.

Every nonzero derived coefficient must survive conversion to a finite, nonzero
binary64 value.  The conversion error, exponent span, dynamic range, and exact
bit complexity are recorded rather than hidden.

Before root isolation, the Morse preflight records the primitive integer
degree, coefficient height, sparsity, Cauchy root bound, and a conservative
Mignotte separation-depth bound for the \(A\)-positivity, \(B\), \(N\), and
reduced-\(u'\) polynomials.  An optional second preflight constructs the
primitive Sturm chains and records their coefficient swell.  Those chains are
cached, so this sharper estimate is reused by the subsequent enumeration.
Worst-case separation bounds are safety envelopes rather than useful timing
predictions; held-out calibration maps the structural and chain measurements
to actual work budgets.

The native GMP preflight additionally performs the exact primitive-PRS
analysis and records total PRS steps, all chain coefficients, and peak
intermediate rational coefficient bits for each required polynomial.  Its
caller-supplied ceilings produce an explicit exact-work refusal rather than a
truncated root count.  Unbounded real-root counting is now a production C
operation; bounded isolation remains under C/Python parity qualification.

The degrees of \(f\) and \(g\) are recorded as input and preprocessing resource
descriptors, but are not by themselves admission criteria.  The directly
relevant Morse object is the reduced rational function

\[
  u(b)=C-\frac{B(b)^2}{A(b)}.
\]

After exact cancellation, qualification records the variable numerator \(B^2\),
the denominator, the absolute-loss numerator \(CA-B^2\), the derivative
numerator, and its squarefree part.  The derivative data determine the
root-enumeration workload much more directly than the source degrees.
Calibration is stratified by this backbone-rational complexity.  Degree-six
source polynomials are merely the width of the current random generator, not a
proposed advertised boundary.

The additive constant \(C\) is tracked separately.  It is relevant when
reporting absolute loss values, but disappears from the vector field, Morse
skeleton, and every potential difference.  It therefore must not by itself
exclude an otherwise representable phase portrait.

## 2. Exact Morse skeleton and binary64 admission

The exact layer must certify:

- \(A>0\);
- no common real root of the reduced \(u'\) numerator and its derivative
  (the exact Morse condition);
- the exact critical-point classification and alternation invariant; and
- distinct isolating intervals.

The handoff to binary64 additionally measures:

- whether distinct exact critical coordinates remain distinct as binary64;
- the minimum adjacent-root separation in local binary64 ulps;
- a normwise perturbation-to-root-collision margin for the squarefree
  numerator of \(u'\);
- the range of \(|u''|\) at the critical points;
- the range of transverse curvature \(2A\) and the binary64 evaluation
  margin for \(A(b_0)\);
- the maximum backbone shear \(|a_*'|\) and minimum relative Hessian
  nonsingularity;
- an exact-jet, Hessian-preconditioned coefficient-\(\ell_1\) upper bound
  analogous to Smale's \(\gamma\), both unscaled and multiplied by half the
  nearest-critical-point distance;
- maximum critical-coordinate magnitude;
- the minimum local spectral resolution margin; and
- the minimum global field/handoff resolution margin.

The existing local continuation rules use margins of 64 for resolved spectral
data and 1024 for a global handoff.  Calibration may impose larger admission
margins, but it must not weaken these soundness floors.

## 3. A-posteriori geometry certificate

Admission does not promise that every portrait is numerically easy.  The
geometry engine must still certify its own result:

- all stage solves pass their backward-error tests;
- all local charts and dense-output handoffs certify;
- all invariant branches terminate in a certified endpoint or end regime;
- the intersection scan finds no forbidden crossing or unresolved contact;
- the index balance and topology audit are complete; and
- all work and event budgets remain within the declared qualification policy.

Failure of any test yields `fp64_unresolved`, not a guessed portrait.

## Calibration experiment

Thresholds are fixed only after an exploratory pilot, then frozen before the
held-out run.  The held-out corpus is stratified by:

- source degrees and reduced backbone-rational complexity;
- moment family, raw scale, and standardized moment-growth bands;
- input and derived coefficient bit complexity/dynamic range;
- number and separation of critical points;
- local spectral and global handoff margin bands; and
- critical-coordinate/trace-box scale.

The report separates:

1. **admission rate** into the advertised sandbox;
2. **certification rate conditional on admission**;
3. **false-certificate count** under independent exact/topological checks; and
4. runtime and memory distributions.

Untargeted calibration samples, targeted boundary samples, and the deterministic
stress zoo are reported separately.  Only the first estimates ordinary-case
rates.  Boundary cases locate the advertised edge; the zoo guards against
regressions beyond it.

For zero observed failures among \(n\) independent held-out admitted cases, the
one-sided 95% binomial upper bound is approximately \(3/n\).  Thus about 3,000
zero-failure admitted cases support a 0.1% upper bound, while about 30,000
support a 0.01% upper bound.  Exact Clopper--Pearson bounds should be printed in
the final calibration report, including nonzero-failure strata.

`scripts/qualify_parallel.py` writes the observation-only `arithmetic` and
`skeleton_arithmetic` records alongside the existing topology ledger.  No
policy threshold is presently embedded in those measurements.

## Near-Morse geometry pilot

The first calibration indicates that no single scalar describes all portrait
difficulty.  Two independent gates are needed:

1. The perturbation-to-root-collision margin measures whether the exact
   one-dimensional Morse skeleton has enough room for a binary64 realization.
2. Relative Hessian nonsingularity and the preconditioned nonlinear-jet bound
   measure whether a local critical-point chart can hand off reliably.

The local jet measurement is the conservative coefficient-norm analogue

\[
 \widehat\gamma_1
 =\max_{k\geq2}
   \left\|H^{-1}\frac{D^k\nabla L(q)}{k!}\right\|_1^{1/(k-1)} .
\]

It is computed from exact centered coefficients and the exact inverse of the
two-by-two Hessian.  It is therefore invariant under multiplying the entire
loss by a nonzero scalar.  Multiplication by a physical target radius tests
whether the certified near-linear region has enough reach.

An exact inverse-designed pilot prescribed pairs of backbone critical points
at separation \(2^{-k}\), with six independently generated cases at each
\(k=4,8,\ldots,32\).  The median root-collision margin fell from 31 binary64
epsilon-bits at \(k=4\), to 8.9 at \(k=16\), to -22.9 at \(k=32\).
All twelve \(k=4,8\) traces were clean; all 24 cases with \(k\geq20\) refused
certification.  Across the 48 cases, Spearman correlation of collision margin
with clean outcome was 0.81 in magnitude and with maximum graph-transform
iterations was 0.71 in magnitude.  The neighbor-scaled gamma bound was nearly
uncorrelated in this controlled family: shrinking separation and increasing
local nonlinearity compensate under that scaling.  One \(k=16\) failure had a
well-conditioned Hessian and small scaled gamma, demonstrating why the root
margin cannot be replaced by the local-jet gate.

A separate held-out set of 128 ordinary random models (source degrees two
through six) produced no exceptions or forbidden-crossing certificates.
After the normal geometry escalation, 115 certified, seven stopped with a
branch conditioning refusal, and six retained ambiguous topology contacts.
Every branch refusal had scaled-gamma log2 above 51 and Hessian condition loss
above 55 bits.  Conversely, all 79 ordinary cases losing fewer than 35 Hessian
bits certified, as did all 50 with scaled-gamma log2 below 20.  These are
exploratory separation bands, not advertised thresholds; substantially larger
held-out strata are required before freezing policy.

Timing of the 22 escalated ordinary cases separated the phases.  Median exact
enumeration was 0.44 seconds and median stub materialization 0.31 seconds,
versus 28.7 seconds for geometry; maximum geometry time was 157 seconds.
Thus, in this degree range, reduced-polynomial complexity predicts exact
preprocessing work, while the near-Morse and local-conditioning measurements
principally predict geometry robustness and long-tail cost.

## Morse pathway pilot

`scripts/compare_morse_paths.py` compares the generic factorized \(B,N\)
enumeration with direct isolation of the reduced numerator of \(u'\).  It
requires identical critical roots and interval-certified \(u''\) signs, and
records portable exact-work counters in addition to elapsed time.

The initial 24-case mixed-degree pilot had 24/24 route agreements.  Factorized
enumeration was faster in all 24 cases: the median time ratio was 4.7, while
the combined Sturm chains contained about 3.8 times as many coefficient bits.
The three deterministic zoo cases also agreed; factorized speed ratios were
2.8, 5.2, and 2.1 with interval-certified signs (the last is the degree-17
case).  Eagerly constructing both candidate chains is therefore itself too
expensive to be the generic selector.

The targeted cancellation model

\[
 X\sim N(0,1),\qquad f=1,\qquad g=1+2x+2x^2
\]

has \(A=(1+2b^2)(1+6b^2)\), \(B=1+2b^2\), and reduced
\(B^2/A=(1+2b^2)/(1+6b^2)\).  Here the reduced route wins and exposes the
correct real-Morse condition: the shared factor has only complex roots, so it
does not make the real loss non-Morse.

The resulting production policy is:

1. Use factorized \(B,N\) enumeration when both are algebraically squarefree
   and coprime.
2. Otherwise use the reduced numerator of \(u'\), testing for multiple
   **real** roots.
3. Never build the combined chain speculatively in the generic case.
