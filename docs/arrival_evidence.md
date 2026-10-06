# Minimum arrival representations

The tested constructor now has a native GMP implementation and an explicit
checked portrait mode. The complete 1,400-entry replay matches the Python
reference after correction of high-degree coefficient transport. See
`native_analytic_arrival.md` for the implemented API, timings and the 223
remaining refusals/failures. The prototype and calibration history below
describe the mathematics retained by that port.

## Diagnosis

The original constructor can end a potential prefix or chart continuation
with a straight capture connector. Its centered raw-time arrival integrator
is used only on other exits from that dispatcher. Neither mechanism supplies
the minimum entry jet called for by SPONG_FOUNDING §7. The existing
PoincareData proposals and Frobenius launch certificates are saddle-specific.

On minimal-quartet and dead-neuron-far-saddle-d3 the six terminal connectors
are approximately 10^12–10^14 scaled binary64 rounding units long. Their
angular failures are not attributable to rounding of the target. Extending
the centered Gauss solve improves some arrivals, but shrinking its stopping
radius can make stiff dense curves worse: small endpoint/Richardson error
does not establish a small directional residual between collocation nodes.
The original integrator and its parity corpus remain unchanged.

## Experimental entry charts

`arrival_series.refine_capture` replaces only an explicit terminal connector
with a proposed analytic arrival. It preserves the preceding branch bit for
bit, matches its incoming point, preserves its declared target, and clears
inherited branch certificates. The original branch is not mutated. This is
an opt-in high-precision Python proposal, not a new native production policy.

Let the minimum's positive Hessian eigenvalues be lambda_s <= lambda_f.
For moderate spectral ratios, construct a truncated Poincare parameterization
K satisfying

    grad L(c + K(u,v)) = DK(u,v) diag(lambda_s,lambda_f) (u,v).

The coefficient at u^i v^j solves

    [H - (i lambda_s + j lambda_f) I] K_ij = -R_ij.

The same local exact polynomial used elsewhere supplies the proposal, with
80-digit arithmetic by default. Every required divisor through the requested
order is screened. A minimum can be resonant; the saddle's absence-of-resonance
argument cannot be reused. Resonant or small-divisor cases refuse explicitly.
No logarithmic resonant terms or Hadamard fallback are implemented here.

For spectral ratio >= 10^4, full normalization of mixed fast/slow terms can
produce a badly scaled coordinate map. Use the slow entry jet K_s(u) plus
the incoming homogeneous fast Frobenius term instead. The threshold follows
the existing stiff-chart split; it is not an acceptance threshold. Omitted
mixed couplings must pass independent residual scrutiny. This is a finite
proposal, not a claim that the omitted terms vanish.

In both cases solve for BOTH incoming amplitudes u0,v0, then retain

    c + K(u0 exp(-s), v0 exp(-rho s)),  rho = lambda_f/lambda_s,

or its stated stiff truncation. The fast amplitude is not set to zero to
project the incoming orbit onto the slow graph. Store the resulting finite
sum of vector coefficients times exp(-rate*s), along with its analytic
limit c at s=infinity. No final finite straight chord is used as flow evidence.
The displayed polyline still joins its last sample to the target, but ownership
of that interval is the analytic arrival representation.

The parameterization approach and the need for nonresonance conditions are
consistent with [Cabre, Fontich and de la Llave, *The Parameterization Method
for Invariant Manifolds I*](https://web.mat.upc.edu/xavier.cabre/docs/indi1jour.pdf).
The finite proposals here do not implement that paper's existence proof.

## Independent checks

`arrival_check` imports no constructor or integrator. It reconstructs the
curve and its derivative from the retained exponential sum and compares them
against the original rational model, evaluated at 90 digits. It checks all
retained parameters and midparameters, adds its own samples on both modal
scales (so an omitted fast boundary layer cannot disappear from the check),
and checks the limiting eigendirection and inward orientation.

The claimed center is bound to a separately refined exact isolating interval
at 200 bits and to a=B(b)/A(b). Point binding, zero tangents, stationary samples,
wrong flow, malformed exponents and budget exhaustion remain explicit.
Decimal arithmetic is used because a general Frobenius exponent cannot be
evaluated over Q alone. These are sampled measurements plus an asymptotic
check, not a uniform bound on the unsampled tail or a convergence proof.

`portrait_check` verifies exact endpoint fate at the entry to the analytic
piece, separately from local attachment and trajectory geometry. It allows
exactly one terminal analytic arrival OR one terminal drawing connector.
All other obligations, including unsupported upstream representations,
remain in force. No production verdict is promoted.

## Reproduction and measured results

```sh
SPONG_ENGINE=native python scripts/arrival_probe.py --orders 6 8
SPONG_ENGINE=native python scripts/credibility_probe.py \
  --refine-arrivals --max-sin-squared 1e-6 \
  minimal-quartet dead-neuron-far-saddle-d3
```

At order 8, the independent sampled squared-sine residuals are:

| Case / branch (zero-based) | Original connector | Analytic arrival |
|---|---:|---:|
| minimal-quartet / 4 | 8.01e-5 | 1.11e-20 |
| minimal-quartet / 6 | 3.68e-5 | 4.51e-21 |
| minimal-quartet / 7 | 8.47e-6 | 1.11e-20 |
| dead-neuron-far-saddle-d3 / 4 | 2.26e-6 | 2.40e-38 |
| dead-neuron-far-saddle-d3 / 5 | 0.996 | 6.80e-17 |
| dead-neuron-far-saddle-d3 / 7 | 0.0133 | 4.74e-9 |

All six measured arrivals have positive inward orientation, no opposed flow,
and point binding below one scaled machine epsilon. Order 6 versus 8 reduces
the largest analytic truncation residual from 1.13e-7 to 4.74e-9. The already
small stiff-mode residuals are limited by other retained/omitted effects;
they do not exhibit monotone improvement merely by adding terms. Tests also
compare 60/80-digit construction and 70/90-digit checking, test a known
mixed-mode linear solution, hide a fast transient from the producer samples,
and deliberately corrupt the attachment and representation.

Qualification across other minima, robust handling of resonances and entry
charts outside their useful radius, uniform sampling policy, native migration,
and any eventual production gate are separate remaining work. Hadamard or
resonant Frobenius representations are needed where this proposal refuses;
no such case is accepted by silently reverting to a drawing connector.

With these proposed replacements, minimal-quartet meets the experimental
four-obligation policy. Dead-neuron-far-saddle-d3 has no remaining arrival
failures but remains unresolved on upstream launch/handoff/trajectory evidence.
These are candidate-policy results, not production certificates.

## Fixed-cohort qualification

`scripts/qualify_arrivals.py` compares the actual terminal connector, an
additional existing native centered raw solve, and the analytic replacement
from the same incoming point and target. It covers 200 random models, 200
directed distant-critical-point models, all seven zoo cases and all 22
recorded raw-arrival calls. The recorded calls retain their original caps
and work budgets. Every traced capture is included, even if other branches
in that portrait are unresolved. This tests the terminal arrival only;
whole-portrait topology audit/escalation is not run.

The frozen policy is order 8, 80-digit construction, independent 90-digit
analytic checking, sin² <= 1e-6, binding/join <= 64 scaled eps, positive
orientation, and independently checked minimum attachment. Exact endpoint
fate is recorded separately. Native dense representations and their final
connector are checked independently too. Construction failures, checker
failures and timeouts remain in the results. The per-model process ceiling
and work budgets are experimental censoring rules, not production changes.

```sh
SPONG_ENGINE=native python scripts/qualify_arrivals.py \
  --output /tmp/arrival-run --random 200 --directed 200 --workers 3
python scripts/summarize_arrivals.py /tmp/arrival-run \
  --output /tmp/arrival-summary
```

The output directory must be new. A manifest freezes seeds, policy and source
hashes; streaming logs preserve baselines if later proposals fail. The
summary retains unprocessed/refused entries and reports paired wins/losses,
cohort/stiffness strata, exact fate, and construction/checking time separately.
These are sampled local geometry measurements, not a uniform tail bound or a
qualified four-obligation acceptance gate. Branches from one model are
correlated; the corpus reuses zoo models.

### 2026-09-25, seed 20260925

All 429 jobs constructed and completed, yielding 1,400 arrival entries;
30 constructed models had no captures. No model/worker timeouts or entry
refusals occurred. Four raw comparison stages timed out at 30 seconds and
are counted as non-passes (the stage includes independent checking).

| Cohort | Entries | Connector passes | Extra raw passes | Analytic passes |
|---|---:|---:|---:|---:|
| Random | 598 | 187 | 468 | 577 |
| Directed | 737 | 181 | 371 | 554 |
| Zoo | 43 | 9 | 30 | 40 |
| Original raw calls | 22 | n/a | 2 | 6 |

The analytic proposal adds 794 passes over the connector on 1,378 traced
tails, and 306 over the raw arm on all 1,400 entries. Neither baseline has a
pass where analytic fails. Analytic nevertheless has 223 non-passes: 183
small-divisor refusals, 27 amplitude-matching failures, and 13 constructed
curves rejected for angular error (11 also have opposed flow). The global
relative-divisor floor already refuses spectral ratios around 1e12 at the
degree-two slow term; these refusals are not evidence of actual resonance.
Exact endpoint fate is forced on 1,396 entries, including every method pass.

Directed seed 427028989/branch 6 and 387639321/branch 4 illustrate why correct
attachment is insufficient: both order-8 curves contain opposed-flow
samples. Increasing to order 12 reduces their sin² residuals to 0.0360 and
0.00603, still above policy; 60- versus 80-digit construction reproduces the
order-8 failures. The current prototype needs controlled entry radius/order
and treatment of omitted couplings, not just more precision.

Median measured construction time is 0.545 s for the Python Decimal proposal
versus 89 microseconds for native raw integration. Independent checking is
separate (medians 1.63 s and 0.098 s). The prototype eagerly renders typically
514 vertices and does not cache charts; these timings do not predict a native
analytic implementation. The original raw-call cohort's 6/22 analytic passes
also shows that the nearer terminal-tail results cannot justify wholesale
replacement at arbitrary existing raw-entry points.

Recommendation: implement a native analytic representation with checked entry
selection and explicit unresolved outcomes; replay this frozen cohort before
changing defaults. A qualified portrait gate still requires all four
obligations and stronger control of unsampled geometry. No production policy
or native solver changed during this qualification. Six harness/aggregation
tests and 59 focused arrival/trajectory/checker tests passed.


Subsequent implementation: see [native analytic arrivals](native_analytic_arrival.md)
and [condition-aware qualification](conditioned_analytic_arrival.md). The
prototype qualification above is retained as the historical baseline.

The [checked handoff implementation](checked_arrival_handoff.md) now retains
and independently checks a native prefix when moving the analytic entry.
