# Checked native prefix to an analytic arrival

This page records the initial handoff milestone. The subsequent
[resolution-aware bridge study](native_arrival_bridge.md) preserves anchors,
resolves ambiguous rounded-jet loss changes, and investigates all 144
remaining cases. Its checked handoff policy is `checked-native-handoff-v2`;
the v1 results below remain the comparison baseline.

Full frozen replay: **1,256/1,400 arrivals admitted**, up from 1,249:
**7 recoveries, zero regressions**. All 399 model jobs completed with unchanged
numerical source and binary hashes. The seven new paths preserve their
original entry and target and include independently checked prefix, tail
and join evidence. **144 cases remain explicitly unresolved.**

| Cohort | Entries | Previous passes | Current passes |
|---|---:|---:|---:|
| Random | 598 | 595 | 595 |
| Directed | 737 | 607 | 614 |
| Zoo | 43 | 41 | 41 |
| Original raw calls | 22 | 6 | 6 |
| Total | 1,400 | 1,249 | 1,256 |

The checked-native portrait path can now retain a native continuation segment
from the original incoming point and hand off to an analytic tail at a later
point. Both smooth representations and their join must pass independent
checks. Moving the handoff never silently discards the original entry or
replaces the intervening path by a chord.

## Construction and admission

`native_arrival.checked_path` first tries the existing condition-aware
analytic proposal at the original point. A successful direct arrival is
returned unchanged as one `minimum_arrival` piece. Only matching or
independent-geometry refusal enables the handoff search; resonance,
precision-budget, invalid-request and other errors are not bypassed.

The fallback calls the existing native target-centered raw-flow integrator
with dense-step recording, a 512-step budget, the established turn/order
policy, and zero requested capture radius. The tracer still applies its
roundoff-based finish radius. No global engine setting is changed. The raw
tracer's capture connector is never accepted as part of this path.

Later inverse attempts are scheduled at the first emitted vertex inside
each radius shell from one half to one 4,096th of the original distance,
with duplicate candidates removed. The target itself is excluded. These
shells are only a bounded search schedule: closeness does not imply chart
validity. Every candidate uses the condition-checked native inverse and the
unchanged independent analytic geometry/attachment checker.

For a candidate with an admitted analytic tail:

1. Reconstruct every retained native dense step from its exported evidence
   and independently measure its tangent against the original rational model.
   Check endpoints, quartiles and emitted parameters, along with display
   binding and step joins. Missing evidence, unresolved measurement budgets,
   unvalidated steps, stationary/zero tangents or opposed flow refuse.
2. Require prefix sin-squared error at most 1e-6 and binding/step joins within
   64 scaled binary64 epsilons, matching the existing geometry targets.
3. Reconstruct the native endpoint and analytic start over exact rationals.
   Require their positional join within 64 scaled epsilons, tangent
   sin-squared mismatch at most 1e-6, and the same tangent orientation.
4. Return a `centered_arrival` piece followed by `minimum_arrival`, sharing a
   single vertex. Their chord ranges partition the entire replacement.

If the prefix fails, the search stops: later candidates would retain that
same rejected prefix. If a join fails, the search may try a later candidate.
The original input vertex must remain bit-identical; if the raw tracer
replaces it during point coalescing, this fallback refuses rather than
inventing a missing segment. Every failure leaves the original branch prefix
intact and the existing explicit `abort_analytic_arrival` policy in force.

## Evidence and API

`checked_proposal` remains the direct analytic-only operation.
`checked_path` returns `(points, pieces, checks)`, where pieces own all chords
and checks explicitly state `certified=False`. Both accept independent
checker precision and sample-budget options; the defaults remain 90 digits
and 10,000 samples per piece. The qualification replay uses 50,000 samples
per piece. Each candidate is subject to its own bounded checker budget.

`arrival_series.refine_capture(..., backend='native')` now uses this complete
path. It preserves all existing points before and including the old entry,
clears stale certificates, and shifts the new piece ranges and native trace
indices consistently when inserting them into the branch. Trace evidence
retains its original source vertex count even when its tail is discarded.

Successful handoffs retain prefix, tail, attachment and join measurements,
along with the candidate history. Refusal exceptions carry `.handoff`, which
is preserved in branch refusal diagnostics. The policy identifier is
`checked-native-handoff-v1`.

The independent join measurement is `arrival_check.measure_handoff`; it
imports no constructor or integrator. Existing trajectory inspection already
understands both piece representations and can remeasure their geometry.
The join check supplements, rather than substitutes for, those measurements.

## Remaining-case investigation

The replay of the previous 151 unresolved arrivals recovered seven complete
paths: six with full-Poincare tails and one with a slow-Frobenius tail. This
includes the last previously unresolved full-Poincare matching case. The
other 144 cases remain explicitly unresolved:

| Refusal category | Entries | Evidence |
|---|---:|---|
| No candidate in the scheduled contraction shells | 121 | Native tracing stopped before reaching a usable later entry |
| Incoming anchor replaced by raw-trace point coalescing | 18 | Refused rather than discarding the original point |
| Independently rejected native prefix | 5 | Angular error exceeded 1e-6 despite an acceptable later tail |

All 121 no-candidate outcomes have an empty candidate history. These are
limitations of this bounded native bridge search, not proofs that a checked
handoff cannot exist. The five rejected prefixes have worst sin-squared
errors ranging from 1.19e-6 to 0.997; no threshold was relaxed to admit them.

All seven saved complete paths were decoded and rechecked independently,
without rerunning the constructor. At both 120 and 180 digits, their analytic
tails and endpoint attachments passed; rational prefix and join checks also
passed. Across these paths, the maximum prefix sin-squared error was
1.97e-10, tail error 5.61e-7, join angular error 4.92e-7, and positional join
error 0.271 scaled epsilons. All original incoming points and targets remained
bit-identical, and every chord had exactly one owner.

The next numerical work is the stiff native bridge: resolved progress from
the original entry and preservation of its anchor. Slow-Frobenius truncation
and omitted fast couplings still require attention. A bridge that fails its
independent test cannot be rescued merely by finding a good tail farther on.

## Scope

This is sampled geometric admission. It is not an interval enclosure, a
series-convergence proof, a guarantee that the inverse remains on a unique
branch, or a complete portrait certificate. Whole-portrait default remains
`legacy`; `arrival_policy='checked_native'` selects this path explicitly.
Neither the analytic nor prefix geometry threshold has been loosened.

Replay with:

```sh
python scripts/replay_arrival_handoffs.py /path/to/frozen/entries.json \
  --output /tmp/handoff-replay --workers 3
```

The harness freezes numerical source and binary hashes, checks exact
endpoints and chord coverage, records explicit failures, and saves both
representations of every hybrid path. Saved trace bytes are base64-encoded
for lossless JSON transport. Higher-precision artifact rechecks use these
saved representations, rather than reconstructing a possibly different path.

## Final validation and artifacts

- Default suite: **1,002 passed**, 10 slow/heavy tests deselected.
- Targeted arrival/trace/portrait/topology integration: 106 passed before the
  last two adversarial tests were added; both are covered by the final suite.
- Regression coverage includes a real recovered handoff, preservation of the
  prior branch prefix, shifted trace indices, complete chord ownership,
  displaced or reversed joins, a bad prefix with a good tail, and loss of
  the original incoming anchor.
- All 1,400 outcomes were compared with the preceding native baseline.
- All seven saved hybrid paths passed 120- and 180-digit independent
  rechecks. Their evidence hashes exactly match the full replay's artifacts.
- `git diff --check` passed. No native C or ABI change was made in this step.

The task's `outputs/checked-handoff/` contains this report, summaries,
per-entry JSON/CSV, seven recovered entries, 144 unresolved entries, the seven
complete saved paths, independent rechecks, replay manifests, test logs and
an evidence/source archive. Older arrival qualification artifacts were kept.
No commit or push was made.
