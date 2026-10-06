# Condition-aware native arrival matching

The subsequent [checked handoff](checked_arrival_handoff.md) adds a retained,
independently checked prefix when a later analytic entry is needed. The
counts below document the preceding inverse-solver milestone.

The 1,400-entry replay accepts **1,249 arrivals**, up from 1,237 after the
divisor update: **12 recoveries and zero regressions**. All 399 model jobs
completed. Of the former 33 matching failures, 12 now pass, 6 match but fail
geometry, and 15 still refuse. Overall unresolved: **136 geometry failures
and 15 matching failures**. Default portrait policy remains `legacy`.

| Cohort | Entries | Previous passes | Current passes |
|---|---:|---:|---:|
| Random | 598 | 593 | 595 |
| Directed | 737 | 597 | 607 |
| Zoo | 43 | 41 | 41 |
| Original raw calls | 22 | 6 | 6 |
| Total | 1,400 | 1,237 | 1,249 |

The native inverse now checks Jacobian conditioning and has a bounded damped
continuation fallback. It matches the original incoming point and retains the
same independent geometry and attachment admission policy. A successful
inverse does not establish convergence of the truncated chart or certify the
resulting curve.

## Inversion policy

The fast path retains the original linear-frame initialization and 24 Newton
updates, with an additional convergence evaluation after the last update.
Every general 2x2 Jacobian system now uses row equilibration and partial
pivoting. The condition number is computed in the infinity norm from the
original matrix and its inverse. The original-system componentwise residual,
with the same working-arithmetic floor used for homological solves, gives a
forward-error indicator `2*beta/(1-beta)`, where `beta` is condition times
backward error. The solve refuses if `beta >= 1` or the indicator exceeds
`1e-20`. The chart's existing condition-based GMP precision budget is reused;
matching does not silently request extra precision.

Convergence requires both the original residual tolerance
`10**(-(p-20))*(1+norm(offset,inf))` and a resolved Newton correction no larger
than `1e-20*(1+norm(amplitudes,inf))`. Thus a small residual alone cannot
justify convergence when the resolved amplitude correction is still large.
These are numerical indicators, not interval error bounds.

If the fast path fails, the solver restarts at amplitude zero and solves
`K(u,v)=t*offset`, advancing `t` from zero to one. Each stage uses damped
Newton with an Armijo decrease in the infinity-norm residual. The initial
increment is 1/8; successful stages double it, failures halve it. A stage
allows 48 updates and at most 32 trial step lengths per update. Overall limits
are 128 attempted stages, 2,048 Newton updates and a minimum increment of
2^-20. Singular or numerically unresolved Jacobians and exhausted budgets
remain explicit matching refusals.

Intermediate targets are internal inverse calculations. They are never
returned as a shortened geometric arrival or used to replace an unchecked
connecting segment. The solver must reach the original target offset before
exporting the analytic representation. This continuation is numerical; it
does not prove branch uniqueness or that the path avoided every chart fold.
The independent curve checker remains responsible for admission.

## API and diagnostics

The additive `spong_arrival_chart_match_inspect` API returns the existing
arrival result plus a separate `spong_arrival_matching` record. The original
`spong_arrival_chart_match` entry point remains a wrapper with unchanged
signature and result layout. Per-call scratch and diagnostics leave cached
charts immutable and safe for concurrent matches.

Destroy diagnostic strings with `spong_arrival_matching_destroy` after
success or refusal. Records include Newton updates, backtracking count,
accepted continuation stages, whether fallback was used, maximum Jacobian
condition, maximum solve forward-error indicator, and the last residual.
On refusal the last residual may belong to an intermediate stage, not the
full original offset. Maxima include unsuccessful attempts; they are not
properties only of the final root.

Python exports this record under `evidence['matching']`, policy
`conditioned-continuation-v1`. Refusal exceptions retain it as `.matching`,
and the replay harness records it. Neither homological nor inverse-solve
conditioning diagnostics replace the independent geometric checks.

## Investigation

A 100-digit Python diagnostic of the 33 prior matching failures found that
19 could find a root within 192 undamped Newton updates. This is diagnostic
only: finding a root is not sufficient for arrival admission.

The native pilot recovered 12 independently admitted arrivals. Six other
cases now match but fail the unchanged geometry policy. Fifteen still refuse
matching. Root-image probes for the remaining slow-Frobenius cases eliminate
the linear fast amplitude and use exact Sturm counts on the resulting
rounded prototype polynomial. The 80- and 100-digit probes distinguish lack
of a real polynomial inverse at the chosen entry from mere Newton failure.
These counts concern rounded finite polynomials, not validated enclosures of
the true analytic chart.

A closer handoff must be reached by a retained and independently checked
continuation segment. Shrinking an offset in an inverse diagnostic is not
such a segment and is not a production fallback.

All 14 remaining slow-Frobenius cases have zero real inverses at the original
offset in both rounded-precision probes. Ten have real inverses after an
offset reduction somewhere between one half and one sixteenth; four still
have none at one sixteenth. The remaining full-Poincare case (job 239,
entry 7) is not covered by the scalar elimination. Its bounded inverse
continuation still refuses; no claim of nonexistence is made for that case.

The independent higher-precision replay covered all 33 former matching
failures: 18 constructed curves received both 120- and 180-digit checks,
and 15 reproduced construction refusal. All 36 measurements agreed with
the 90-digit primary decisions, including all 12 recovered admissions.
There were no worker or measurement errors.

## Validation and reproducibility

- Default suite: 996 passed, 10 slow/heavy tests deselected.
- Focused native-arrival suite: 27 passed, including a recovered frozen case,
  a difficult but invertible polynomial chart, and a point with no real
  polynomial inverse that must remain refused.
- All 12 standalone C tests passed via direct clang/GMP compilation.
- Standalone arrival tests passed ASan/UBSan. An allocator-injection harness
  exercises each of 480 allocation positions in a successful match, including
  diagnostic ownership and cleanup on refusal.
- All 1,400 replay decisions were compared with the previous native baseline;
  cold/cached native results and diagnostics were required to match exactly.
- All 33 former matching cases were separately revisited at higher checker
  precision, with complete coverage and no decision disagreements.

Numerical source and extension hashes were unchanged throughout the full
replay. Subsequent failure-path inspection added checks for allocation
failure while parsing internal tolerance constants: such a failure now
refuses that solve instead of using a stale scratch value. This does not
change successful arithmetic. The final build reran the 27 focused tests,
standalone sanitizer test and allocation-failure probe; the full suite and
cohort replay preceded this error-path-only correction. Both the replayed C
source and final source hashes are retained for audit.

Task artifacts are under `outputs/conditioned-matching/`: per-entry results,
12 recovered cases, 151 unresolved cases, summary, higher-precision checks,
prototype Newton and polynomial-image probes, replay manifests, test logs,
and a source/evidence archive. Existing divisor and initial-port artifacts
remain separate.

The next implementation step is a checked continuation segment to a later
analytic handoff for entries outside the polynomial image. That segment
must preserve the original incoming point and retain independently checked
geometry; offset shrinking alone is insufficient. Slow-Frobenius truncation
and missing fast couplings remain separate causes of geometric refusal.
