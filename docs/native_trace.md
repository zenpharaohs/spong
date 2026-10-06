# Native trajectory evidence, schema 1

The optional `*_recorded` C entry points for continuation, potential segments,
and centered arrival retain representations used by the accepted solve.
Legacy entry points call these with a null trace, preserving their signatures
and results. Python production wrappers request evidence; direct extension
calls retain their legacy return values unless the trailing `recording` flag
is set. This is observational transport, not a solver acceptance criterion.

Initialize `spong_trace` to zero, pass its address, inspect `failed`, and free
`records` after use. Allocation failure is sticky; Python raises MemoryError.
Capacity retries reset the event count and replay the complete segment.
Storage is 192 bytes per event, plus allocator capacity; large portraits can
retain substantial evidence. The Python byte stream uses native-endian
binary64 and is an in-process representation, not a portable archival format.

Each record has 24 doubles. Record numbers are zero-based, including vertex
and trim events. Unused fields are zero.

| Kind (`v[0]`) | Payload |
|---|---|
| 1: collocation | `v[1]` chart; `v[2]` order 4/6/8; `v[3]` signed step; `v[4:6]` start; `v[6:14]` stage derivative pairs; `v[14:18]` nodes; `v[18:20]` physical endpoint; `v[21]` method |
| 2: Hermite | Physical chart 0; `v[3]` signed step; `v[4:6]` start; `v[6:8]` end; `v[8:10]`, `v[10:12]` endpoint derivatives |
| 3: emitted vertex | `v[4:6]` physical point; `v[20]` output vertex index; `v[22]` parameter in [0,1]; `v[23]` source record number, or -1 for an anchor/connector |
| 4: terminal trim | `v[20]` surviving vertex count; remove all higher vertex indices |

Charts are 0: physical `(a,b)`, 1: slow `(b,w)`, 2: fast `(w,b)`,
3: centered physical coordinates with center in `v[22:24]`. Charts 1 and 2
map by `a=B(b)/A(b)+w`; the checker differentiates this map over Q.
Continuation's method codes are 0: primary, 1: alternate chart/order ladder,
2: normalized-plane rescue, 3: retry-exhausted acceptance inherited from the
reference. Code 3 is explicitly ineligible for candidate acceptance.
Potential/arrival modes retain actual accepted half-step stages, including
arclength fallback and the selected proper-time event trial. Potential ascent
retains the actual cubic Hermite interpolant used to emit samples.

A repeated vertex index replaces its previous mapping. The checker decodes
these replacements and terminal trims before selecting the surviving dense
prefix. It ignores a discarded final half-step and an erased hover tail.
The stream may contain such discarded records: their presence never counts
as evidence for the surviving path. Native continuation's centered-chart
rescue still delegates an entire segment to Python; that returned segment
has no native stream and remains unresolved, rather than inheriting evidence
from an abandoned native attempt.

`native_trace.measure_native` reconstructs the collocation polynomial by
integrating the Lagrange interpolant of the retained stages, or reconstructs
the Hermite polynomial. It evaluates the original exact model and the
proposed tangent with Fraction arithmetic. Each surviving step is checked
at endpoints, quartiles and every emitted parameter. Final partial steps are
cut at their surviving endpoint. A sample budget refuses the measurement
explicitly; it never silently subsamples.

Reports separate field-angle residual, oriented flow, stationary points,
zero tangents, and binding/join gaps in `eps*(1+abs(coordinate))` units.
This checks a proposed smooth curve; it does not prove the nonlinear stage
solve or bound the error between sample parameters. Unknown schemas, methods,
charts, missing vertex sources, malformed maps and nonfinite records refuse.

`portrait_check.assess` additionally recomputes the exact critical inventory,
reruns the contact/branch auditor, checks launch eigenspaces and signed sides,
checks exact capture/escape fate, checks capture and handoff geometry against
the original field, and checks actual dense tangents at declared boundaries.
Its policy requires explicit angular, eigenvector and binding thresholds.
`meets_candidate_policy` is experimental evidence, never a production
certificate; `certified` remains false. Grids, Hadamard graphs and delegated
Python continuations currently prevent candidate acceptance where encountered.

## Calibration experiment, 2026-09-25

Run:

```sh
SPONG_ENGINE=native python scripts/credibility_probe.py \
  --max-sin-squared 1e-6 --max-eigen-sin-squared 1e-12 \
  --max-binding-eps 64 minimal-quartet dead-neuron-far-saddle-d3
```

These are candidate accuracy targets, not a calibrated default. The angular
threshold is about 0.057 degrees. Both portraits remain unresolved even
though the existing topology audit certifies them. Minimal-quartet's three
capture connectors have squared-sine residuals about `8.01e-5`, `3.68e-5`
and `8.47e-6`; its retained smooth paths and other obligations pass this
candidate. Dead-neuron-far-saddle-d3 additionally contains unsupported local
and continuation representations, and one capture connector measures about
`0.996`. Exact arithmetic makes small or rounded drawing connectors visible;
correct endpoint fate does not establish their geometric accuracy.

Analytic straight/curved controls and adversarial stage, vertex, direction,
launch, endpoint, boundary and budget tests validate measurement/refusal
behavior. These two zoo cases are a diagnostic experiment, not qualification
on a held-out calibration suite. No production gate is promoted. Completing
that qualification requires smooth evidence for remaining construction types,
a justified arrival representation, and acceptance/rejection calibration
across degree, stiffness and precision. Probe timings now include independent
trajectory and attachment work as well as the chord reference.
