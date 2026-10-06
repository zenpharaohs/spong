# Resolution-aware native arrival bridge

The checked handoff uses an opt-in native bridge that preserves its incoming
anchor and separates potential subtraction roundoff from an actual loss
increase. This is a construction improvement, not a new certification gate.
The default portrait policy remains `legacy`.

## Native policy

`spong_centered_arrival_bridge_recorded` is additive to the existing C API.
It requires a trace, diagnostics, and `clock == 0`. Existing request/result
layouts and legacy arrival entry points retain their behavior.

The bridge makes three changes:

1. Point coalescing cannot replace the original incoming vertex. Subsequent
   small steps can still replace later display vertices, retaining their
   native dense representation in the trace.
2. When two finite potential values differ by no more than
   `256 * DBL_EPSILON * max(abs(before), abs(after))`, GMP rational arithmetic
   compares the path-integrated potential at both binary64 coordinates.
   Both rounded increases and rounded decreases in this band are checked.
   A zero or positive exact change refuses the trial. The trigger is a
   numerical heuristic, not an error enclosure. The comparison is exact for
   the **rounded jet coefficients**, not the original rational model; rounded
   cross-partials need not agree. No claim of an original-model Lyapunov
   certificate follows from it.
3. When both order trials return exactly the same centered coordinates,
   probe a fourfold larger timestep, up to the existing `4 / slow` cap.
   If other rejection criteria require returning to a known stalled
   timestep, refuse explicitly. The retry limit is 64 for this bridge;
   legacy tracing retains 16. Growth never bypasses the existing Richardson
   or turn criteria. Nonfinite and nonpositive initial floating potential
   still refuse.

No angular, binding, attachment, join, or Richardson threshold is weakened.
The full retained prefix must pass the independent original-model check,
then its analytic tail, target attachment, and representation join must pass.
A bridge may take accepted solver steps yet fail any of those obligations.

Diagnostics count solver failures, loss rejections, Richardson rejections,
zero-movement trials, exact potential comparisons and strict decreases,
and timestep growth retries. `anchor_retained` counts the first append after
the incoming point. Counters include rejected order trials and do not
identify a unique root cause by themselves.

The CPython `centered_arrival` adapter accepts a final optional `bridge`
flag following `recording`. Bridge calls require recording and a disabled
clock; they append a diagnostics dictionary after the usual trace bytes.
Old call shapes and return shapes remain unchanged. `checked_path` uses
this mode and identifies its policy as `checked-native-handoff-v2`.

## Frozen replay result

Full replay remains **1,256/1,400 admitted**, with **zero recoveries and zero
regressions**. All 399 jobs completed with unchanged numerical source and
binary hashes. The seven retained hybrid paths pass all 14 saved-evidence
rechecks at 120/180 digits. Python validation: 1,002 passed, 10 slow/heavy
cases deselected; nine focused tests also passed. All 13 standalone C tests
passed, and the new bridge regressions passed AddressSanitizer and
UndefinedBehaviorSanitizer. CMake was unavailable; the C suite was built
directly with clang and GMP. Task artifacts are in `outputs/native-bridge/`.

## Validation scope

Standalone C regressions use a diagonal linear field with a closed-form
exponential solution. They demonstrate loss decreases hidden by binary64
subtraction, zero-movement timestep growth, and exact preservation of the
incoming point. A second small-finish-radius test verifies that growth
still refuses when the unchanged Richardson criterion cannot resolve the
step. Unsupported clock mode is rejected.

The study preserves complete native traces for the 144 previously unresolved
entries and independently measures every available prefix against the
original rational model. These prefix measurements are diagnostic: a passing
short prefix does not admit a complete arrival. A separate experiment tries
an analytic tail at the last resolved non-target display vertex, even if the
usual contraction shell was not reached. It also retains prefix and join
checks; no endpoint-only shortcut is introduced into production.

## The 144-case frontier

All 144 cases completed both the bridge measurement and endpoint experiment,
with no worker errors. All original incoming vertices were preserved.

| Available native prefix | Cases |
|---|---:|
| Independently admitted short prefix | 77 |
| Rejected prefix | 57 |
| No retained movement | 10 |

Among the 57 rejected prefixes, 33 exceed the angular threshold and 37
contain opposed-flow samples (13 have both defects). No binding or join
position threshold caused those prefix rejections. The 77 passing prefixes
are not complete arrivals and do not establish an admissible analytic tail.

The raw bridge terminates on step failure in 139 cases, invalid floating
potential in three, and the 512-step budget in two. These are termination
labels, not proofs of a unique numerical cause. Exact potential comparison
finds strict rounded-jet decreases in 112 cases; 135 probe timestep growth.
Compared with anchor-preservation-only tracing, 103 cases take more accepted
solver steps, totaling 2,749 versus 2,115. None of the 144 reaches complete
independent arrival admission.

The endpoint experiment skips 71 entries whose last retained display point
is still the original input. It tests 73 changed endpoints: 57 tails fail
independent checks, nine fail inverse matching, and seven tails pass. All
seven passing tails have rejected prefixes and joins, so there are zero
complete successes. This experiment does not change production candidate
scheduling.

## Next numerical work

The next candidate is a native bridge that carries sufficient precision
through field evaluation, state updates, and retained evidence. Increasing
precision only in a scalar loss comparison is demonstrably insufficient.
The rational translation center and original-model coefficients need to
remain authoritative; recomputing a rounded binary64 jet with more precision
does not restore information already lost. The current local-jet builder
also imposes a fixed root-refinement scale, which should be examined against
conditioning before making stronger claims.

The representation handed to the analytic inverse needs the same scrutiny:
71 endpoint probes had no new binary64 display point despite some internal
stepping. A future high-precision bridge must export its actual smooth
endpoint and stages, not round them away before matching or checking. The
independent prefix, analytic tail, attachment, and join obligations remain
separate. Slow-Frobenius truncation and fast/slow coupling can still reject a
tail after the bridge itself is valid.
