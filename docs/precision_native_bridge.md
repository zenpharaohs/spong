# Experimental precision-aware native stiff bridge

`spong.precision_bridge.checked_precision_path` first preserves every path
admitted by the existing checked-native policy. Only a refused handoff can
activate the new experimental bridge. The ordinary portrait default and
`checked_native` policy are unchanged. This is sampled geometric admission,
not a convergence proof or an ordinary portrait certification gate.

## Original-model construction

The adapter translates the **original rational model gradient** around the
exact binary64 incoming point, using rational arithmetic. It does not remove
the constant gradient, round the coefficients, or substitute a rounded
critical-point center. The complete translated polynomial, positive spectral
rates, precision request and step budget are passed to the C core. All
integration and stage solves execute natively with GMP.

The method is three-stage Radau IIA, an L-stable implicit method of order
five. Its cubic collocation curve is constructed by integrating the stage
Lagrange polynomials; nodes and coefficients are computed at working
precision. The retained coordinates are offsets from the original rational
input. Each accepted step exports both cubic components as decimal strings
at the planned working precision. Display vertices may coincide even when
the smooth representation has resolved nonzero movement.

Working bits are planned as requested decimal digits converted to bits,
plus 128 guard bits and the binary exponent of the fast/slow spectral ratio.
Requests exceeding 4,096 bits refuse. The six-dimensional Newton stage solve
uses pivoting, an explicit numerical infinity-norm condition estimate and a
recomputed backward-error check. It refuses when
`condition * 2**(-working_bits + 32)` consumes the requested numerical reserve.
Newton convergence is componentwise, so a large coordinate cannot mask an
unresolved small-coordinate correction. These are numerical indicators, not
interval bounds. The solver does not silently exceed its precision budget.

The default construction budget is 512 steps, with at most 32 timestep trials
per step and 24 Newton corrections per trial. The C API permits up to 2,048
steps. Initial timestep is `0.25 / fast`, capped at `4 / slow`; failures halve
it, and admitted steps grow it conservatively. The native constructor checks
field alignment and orientation at endpoints and quartiles, requiring
sin-squared angular error at most 1e-8. This construction screen does not
replace independent admission. No proper-time/clock certificate is emitted.

## Exact entry and independent checks

The analytic matcher has an additive exact-rational entry API. The existing
binary64 start is retained for display only when exact coordinates are
supplied. The inverse receives the actual endpoint reconstructed from the
exported cubic, including movements smaller than a display ulp. Existing
match entry points and default behavior remain unchanged.

Candidate scheduling combines radius shells with powers-of-two step indices
and the last retained endpoint. It is bounded and can test points whose
display coordinates have not changed. No prefix is discarded to manufacture
a new incoming point.

`precision_bridge_check` imports neither the constructor nor the integrator.
It derives the gradient independently from the original model and measures
all retained steps at nine points (endpoints and eighths), including
points between the constructor's samples. Checker precision has a floor
above the bridge working precision; a conservative cancellation indicator
refuses unresolved evaluations. The existing admission thresholds remain:

- Prefix and analytic-tail angular error at most 1e-6, correct flow orientation,
  no stationary or zero-tangent samples, and resolved measurement budgets.
- Display binding and step joins within 64 scaled binary64 epsilons.
- Target binding, positive Hessian and inward attachment, with attachment
  angular error at most 1e-12.
- Exact-rational reconstruction of the native/analytic representation join,
  checking position, angle and orientation separately.

The old checked paths are returned before this experiment is attempted.
New successes have a `precision_bridge` piece followed by `minimum_arrival`;
all chords are owned, original display endpoints are preserved, and checks
state `certified=False`. Failure carries candidate history and native
precision/solver diagnostics and returns no partial accepted path.

## Usage

```python
from spong.precision_bridge import checked_precision_path
points, pieces, checks = checked_precision_path(
    model, minimum, original_entry, target,
    sample_budget=50000,
)
```

This explicit experimental function is not automatically selected by ordinary
portrait construction. `scripts/replay_precision_bridges.py` qualifies it
against frozen entry metadata and saves the complete evidence for every
hybrid path. `trajectory_check.inspect_trajectory` understands the new piece
when native measurement is requested.

The C interface is `spong_precision_bridge`, declared in
`spong_precision_bridge.h`. Input coefficient strings are ascending
`[component][a power][b power]` with explicit rectangular dimensions.
`spong_precision_bridge_destroy` frees all result strings and can be called
on cleared results. The additive analytic matcher entry is
`spong_arrival_chart_match_exact_inspect`; independent display binding and
representation join checks remain required.

## Qualification result

Full frozen replay: **1,289/1,400 admitted**, up from 1,256, with **33
recoveries and zero regressions**. All 399 jobs completed. Numerical source,
headers, adapter and binary hashes stayed unchanged during replay. All 598
random entries now pass; directed entries are 637/737, zoo entries 42/43,
and original raw calls 12/22. Recovered paths use 412–502 working bits.

All 33 recovered prefixes also pass exact-rational checks against the
original global model. All 40 saved hybrid paths pass rechecks at both 240
and 320 digits: 80/80. Default Python tests: 1,007 passed, 10 slow/heavy
cases deselected. Focused tests: 41 passed. All 14 standalone C tests passed,
and new solver tests passed ASan/UBSan. CMake was unavailable, so C tests
were built directly with clang and GMP.

**111 remained unresolved at this milestone.** All reached the default
512-step construction budget. Their last candidates failed analytic
independent checks in 101 cases and matching in ten. This result alone did
not qualify the optional larger budgets. Task evidence is in
`outputs/precision-bridge/`, including saved complete paths and recheck hashes.

## Controlled budget and chart-order continuation — 2026-10-09

The replay driver now accepts explicit `--max-steps`, `--order`, and
`--retry-order` options, plus `--save-all` when complete representations of
direct analytic arrivals are needed. The backend's opt-in `retry_order`
returns an existing admitted path first and tries the second chart order
only after an explicit `precision_bridge_no_checked_entry` refusal. Its
default is `None`, so previous behavior and ordinary portrait policy are
unchanged.

```python
from spong.precision_bridge import checked_precision_path
points, pieces, checks = checked_precision_path(
    model, minimum, original_entry, target,
    max_steps=2048, retry_order=12, sample_budget=50000,
)
```

On the frozen 111 refusals, order 8 recovers six at 1,024 steps and twelve
at 2,048 steps. The order-8-then-12 policy at 2,048 steps recovers **73**:
twelve at the first order and 61 at the second. It leaves **38**. A complete
1,400-entry replay confirms **1,362/1,400** admissions against the original
1,289 baseline, with zero regressions and all 399 jobs complete. Each frontier
sweep completed all 109 jobs; the second-order policy lost no first-order
admissions. Frontier and full replay numerical source/binary hashes match.
Forty new paths are direct analytic arrivals and 33
retain a precision-bridge prefix. All 73 saved representations pass fresh
240- and 320-digit checks (**146/146**). All 33 retained native prefixes
also pass exact-rational checks against the original global model at nine
points per step. These are sampled admissions, never certificates.
The current default Python suite passes 1,008 tests.

At order 8 and 2,048 steps, 95 of 99 last candidates failed the independent
tail-angle limit, including 71 with reversed flow; four inverse matches did
not converge. Order 12 recovered some of the reversed-flow cases, so the
order-8 orientation alone does not justify skipping the second order. Of the
38 final refusals, last-candidate reasons are 29 analytic checks, five
native prefix checks and four inverse matches. They need individual
obligation diagnostics and further chart/fast-coupling work. The 2,048-step
limit and existing admission thresholds were not widened.

See task `outputs/precision-bridge-budget-sweep/report.md` for complete
frontier and full replay inputs, manifests, saved paths, independent rechecks
and the entry-by-entry comparison. The compact repo checkpoints are
`docs/evidence/precision_bridge_order_ladder_2026-10-09.json` and
`docs/evidence/precision_bridge_remaining_2026-10-09.json`; the latter is a
replay-compatible list of all 38 still-refused original entries.
