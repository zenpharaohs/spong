# Native analytic arrivals: implementation and qualification

Initial port completed 2026-10-02 after recovery from the interrupted session.

**Current update:** the native solver now uses condition-aware divisor
screening and adaptive working precision. See
[conditioned analytic arrivals](conditioned_analytic_arrival.md) for the
subsequent divisor replay, and [condition-aware matching](conditioned_arrival_matching.md)
for the inverse policy, followed by [checked handoffs](checked_arrival_handoff.md)
for the latest complete-path admission and remaining cases. The cohort counts, timing
measurements and remaining-work table below document the initial port with
the historical divisor guard.

The tested Poincare / slow-Frobenius arrival constructor now runs in the
frontend-independent C core with GMP. Chart construction, two-mode amplitude
matching and display evaluation are native. Exact coefficients cross the
adapter without binary64 rounding. The exported analytic representation is
checked independently against the original model, including its endpoint.

## Using the implemented path

```python
from spong import portrait
p = portrait.compute(m, arrival_policy="checked_native")
# Or include the existing topology audit/escalation workflow:
p = portrait.certified_compute(m, arrival_policy="checked_native")
```

`arrival_series.refine_capture` also uses checked native construction by
default; `backend="python"` retains the experimental reference constructor.
Whole-portrait defaults remain `legacy`. The new mode is explicitly selected
while the remaining arrival regions are developed; no general certification
policy has been promoted.

An accepted arrival replaces only the terminal connector, preserves its
incoming point and target, and clears old branch certificates. Its evidence
retains both modes, coefficients, exponents and analytic limit. Failed
proposals remove the unaccepted connector and return an explicit
`abort_analytic_arrival` prefix, with the reason and proposed target in
diagnostics. Endpoint-fate certification cannot override that representation
failure. Other existing continuation-abort policies are unchanged.

Charts are immutable, reusable and bounded by a 128-entry adapter cache.
Matching uses per-call scratch storage; GMP precision is per object. The C
ABI exposes separate create/match/destroy calls for other frontends. See
`docs/c_api.md` and `include/spong/spong_analytic_arrival.h` for ownership and
precision limits. The independent sampled checker still runs in Python.

## Frozen-cohort replay

All 1,400 original arrival inputs were replayed without retracing prefixes.
Model enumeration/local data were reconstructed, native curves were checked
independently, and native cold/cached construction was measured separately.
Every native admission decision matches the Python reference.

| Cohort | Entries | Python passes | Native passes |
|---|---:|---:|---:|
| Random | 598 | 577 | 577 |
| Directed | 737 | 554 | 554 |
| Zoo | 43 | 40 | 40 |
| Original raw calls | 22 | 6 | 6 |
| Total | 1,400 | 1,177 | 1,177 |

The first replay exposed Python's 4,300-digit integer-string conversion
limit on 38 high-degree entries, causing 21 admission regressions. Exact
Decimal integer formatting now transports these coefficients without
changing the process-wide digit limit. All 38 affected entries were rerun;
there are zero remaining admission disagreements. The original logs are
retained alongside the corrective replay, not overwritten.

The native C core, extension and independent numerical checker hashes were
unchanged throughout the replay. Two adapter/policy files changed after its
initial manifest: `native_arrival.py` received the serialization fix, and
`topology.py` prevents exact forced completion from overriding an explicitly
failed analytic representation. The latter path is not called by the local
replay and was separately integration-tested.

Median native construction over 1,190 constructed proposals is **12.4 ms
cold** and **11.5 ms cached**, including display vertices. The median ratio
against archived Python timings is about 39x, but those runs had different
concurrent workloads. A fresh alternating-order paired sample of six full
Poincare and six slow-Frobenius cases measured median speedups of **39.7x**
and **14.9x**, respectively. Display points differed by less than one scaled
machine epsilon in that sample. These measurements include display
evaluation and exclude independent checking; they do not imply a comparable
whole-portrait speedup. Independent checking remains the dominant cost.

## What remains

| Failure class | Entries | Observed location | Next investigation |
|---|---:|---|---|
| Small-divisor screen | 183 | Eigenvalue ratios 1.09e12–4.47e74 | Condition-aware divisor/precision policy; retain genuine resonance refusal |
| Amplitude matching | 27 | 2 random, 20 directed, 5 original raw calls | Valid entry regions and safeguarded chart inversion |
| Rejected constructed geometry | 13 | 7 directed, 6 original raw calls; 11 have opposed flow | Truncation/entry-radius control and omitted fast couplings |

Every small-divisor refusal in this cohort lies beyond the current globally
scaled cutoff. That does not prove these cases are resonant, nor that lowering
the cutoff alone is safe. Eigenvalue ratios span many more digits than the
current precision budget can uniformly resolve.

Two reproducible geometry failures are directed seed 427028989/branch 6 and
387639321/branch 4. The earlier order/precision probes already showed that
raising order to 12 improves their residuals but does not meet the fixed
threshold. Merely adding arithmetic precision does not fix these entries.
The native port intentionally preserves these refusals and the original
order-8 policy instead of changing the mathematics during migration.

The 1,177 passes are sampled local arrival evidence, not uniform enclosures,
convergence proofs or complete four-obligation portrait certificates. The
original raw-entry subset remains only 6/22, so choosing an appropriate entry
region is essential before replacing the entire legacy raw-arrival routine.

## Validation and reproduction

- Default suite: 985 passed, 10 slow/heavy tests deselected.
- Native-arrival/topology integration: 51 passed after the refusal-policy fix.
- Final native-arrival focused tests: 19 passed, including large exact
  coefficient transport and a real rejected geometric counterexample.
- All 12 standalone C tests passed by direct compilation/linking; CMake was
  unavailable on this machine.
- The new standalone C test also passed AddressSanitizer and UBSan.
- An audited minimal-quartet portrait uses all three native analytic tails.

The final serialization correction was covered by the 19 focused tests and
the 38-entry corrective replay; the 985-test run preceded that correction.

```sh
python setup.py build_ext --inplace
python scripts/replay_native_arrivals.py /path/to/original/entries.json \
  --output /tmp/native-arrival-run --workers 3
python scripts/summarize_native_arrivals.py /path/to/original/entries.json \
  /tmp/native-arrival-run --output /tmp/native-arrival-summary
```

For the historical run, pass `--correction` with the separate 38-entry replay
to the summary script. It requires exactly the entries affected by the
serialization error and rejects incomplete primary or corrective coverage.
The saved report bundle includes original/corrective logs and manifests,
per-entry JSON/CSV, the 223-entry remaining-work list and fresh paired timings.
