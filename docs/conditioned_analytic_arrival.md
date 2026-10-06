# Condition-aware native analytic arrivals

Completed 2026-10-02. The subsequent
[condition-aware matching update](conditioned_arrival_matching.md) supersedes
the matching policy and remaining-case counts below. This document retains
the divisor milestone's results.

Divisor milestone: The native constructor now accepts 1,237 of the 1,400
frozen arrival inputs: 60 additional passes and zero regressions from the
initial native/Python baseline. Whole-portrait default remains `legacy`;
select `arrival_policy='checked_native'` to use the independently checked
native proposal. This is local sampled evidence, not a certification gate.

## Numerical policy

The old native guard divided every homological divisor by a scale containing
the fast eigenvalue. It therefore rejected nonresonant slow equations when
the eigenvalue ratio exceeded roughly 1e12. The new `componentwise-v1` policy
separates cancellation within a modal subtraction from conditioning of the
physical-coordinate linear system.

For degree `(i,j)`, let `rho = i*lambda_s + j*lambda_f`. Each modal gap is
measured as `abs(lambda_k-rho)/(lambda_k+rho)`. It must exceed `10**(20-p)`,
where `p` is the declared input precision (50–100 decimal digits). Exact zero
gaps still refuse. The threshold reserves 20 digits against cancellation;
adding working bits cannot recover spectral information absent from inputs.
The request must supply a consistent symmetric Hessian and its spectral data.

The precision planner estimates the largest ratio of absolute modal gaps
across the retained homological systems. Its binary exponent is added to
`ceil(p*log2(10))+64`, with two extra bits for the symmetric 2x2 norm
conversion. A 4096-bit budget cap produces an explicit precision refusal.
Reported `working_bits` is this budget; GMP can round allocated precision up.
The policy is fixed and reproducible, rather than an unbounded retry loop.

Each homological solve now equilibrates rows and uses partial pivoting.
The residual is measured against the original, unscaled matrix. For symmetric
`A = [[a,b],[b,d]]`, its infinity-norm condition is
`max(abs(a)+abs(b), abs(d)+abs(b))**2 / abs(a*d-b*b)`.
The componentwise backward error has an arithmetic floor of `2**(-bits+8)`.
With `beta = condition * backward_error`, the forward-error indicator is
`2*beta/(1-beta)`; `beta >= 1` or an indicator exceeding `1e-20` refuses.
These finite-arithmetic diagnostics are not directed-rounding error bounds
and do not control propagation through the nonlinear coefficient recurrence.

`spong_arrival_chart_inspect` and `spong_arrival_conditioning_destroy` expose
owned diagnostic strings through an additive C API without changing existing
request/result layouts. Python evidence includes policy, budgeted bits,
minimum relative divisor, maximum matrix condition, backward error and
forward-error indicator. The existing `divisor_margin` now means the minimum
componentwise relative divisor; it is not comparable to old global margins.

The Python reference retains its historical global guard. The stiffness
switch at 1e4, order-8 production proposal, 24-iteration matching, retained
analytic representation and independent acceptance thresholds are unchanged.
No resonant normal form or Hadamard graph-transform fallback was added.

## Frozen replay

All 399 model jobs with arrival entries completed successfully. All 1,400
original start/target inputs were covered once, without retracing prefixes.
The replay required identical cold/cached native outputs and independently
checked each constructed proposal. Numerical source and binary hashes were
unchanged during the replay.

| Cohort | Entries | Baseline passes | Condition-aware passes |
|---|---:|---:|---:|
| Random | 598 | 577 | 593 |
| Directed | 737 | 554 | 597 |
| Zoo | 43 | 40 | 41 |
| Original raw calls | 22 | 6 | 6 |
| Total | 1,400 | 1,177 | 1,237 |

All 183 old divisor refusals passed the new arithmetic screen: 60 passed
independent admission, 117 failed geometry, and 6 failed amplitude matching.
There were no new divisor, working-precision or homological-solve refusals in
this cohort. This does not imply that the guards cannot fire: focused tests
exercise genuine resonance, insufficient declared precision and the bit cap.
The largest recovered passing spectral ratio was 5.10e29.

For constructed proposals, working budgets ranged from 333 to 580 bits,
maximum reported matrix condition was 4.47e74, and the largest solve
forward-error indicator was 8.46e-98. These small solve indicators do not
imply accurate truncated arrival geometry.

The primary independent checker used 90 digits. Every one of the original
183 divisor cases was independently revisited: 177 constructed curves were
checked at both 120 and 180 digits (354 measurements), while 6 reproduced
matching refusals. There were no worker failures, measurement errors or
pass/fail disagreements with the primary check. All 60 recovered admissions
therefore agree at all three checking precisions. Geometry retains the
sin-squared threshold 1e-6, binding threshold 64 scaled machine epsilons,
orientation tests and attachment eigen-direction threshold 1e-12.

Median native construction/display time was 18.3 ms cold and 16.8 ms cached
under this replay's concurrent workload. These are not controlled comparisons
with the earlier port timings; no speedup or slowdown claim follows from them.

## Remaining cases

| Failure | Random | Directed | Zoo | Raw calls | Total |
|---|---:|---:|---:|---:|---:|
| Amplitude matching | 2 | 24 | 0 | 7 | 33 |
| Geometry | 3 | 116 | 2 | 9 | 130 |

All 130 geometry rejections use the slow-Frobenius proposal and exceed the
fixed angular residual threshold; 49 also contain opposed-flow samples.
None exceeds the binding threshold. Increasing checker precision leaves
these decisions unchanged in the previously divisor-blocked group.

The next investigation is a condition-aware entry-region and chart-inversion
policy for matching, followed by truncation control and omitted fast
couplings for geometry. The evidence does not justify accepting these
remaining curves by adding precision or relaxing the checker. In particular,
raw-call performance remains 6/22; this is not yet a replacement at arbitrary
legacy entry points. Convergence bounds and the other portrait obligations
remain necessary before promoting an ordinary certification gate.

## Validation and artifacts

- Full default Python suite: 993 passed; 10 slow/heavy tests deselected.
- Native/Python arrival, portrait checker and topology integration: 85 passed.
- Focused native-arrival suite: 24 passed, including extreme diagonal and
  rotated stiffness, near/exact resonance and precision-budget exhaustion.
- All 12 standalone C tests passed via direct clang/GMP compilation; CMake
  was unavailable. The analytic-arrival C test also passed ASan/UBSan.
- `git diff --check` passed.

The task's `outputs/conditioned-arrival/` contains the report, summary,
per-entry JSON/CSV, all 60 recovered entries, all 163 remaining entries,
higher-precision checks, source manifests, validation logs and a source/log
archive. Historical baseline artifacts were preserved separately.

Reproduce the primary replay with:

```sh
python setup.py build_ext --inplace
python scripts/replay_native_arrivals.py /path/to/arrival-qualification/entries.json \
  --output /tmp/conditioned-arrival-replay --workers 3
```

The archived higher-precision and aggregation scripts record their exact
input paths and policies. Use the matching baseline inputs from the earlier
qualification bundle when relocating those scripts.
