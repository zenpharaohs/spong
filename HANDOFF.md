# Handoff — spong

Current checkpoint: `trajectory-checker`, 2026-10-09. Milestone sections
below retain their original dates and status.

## Opt-in order-aware precision bridge — 2026-10-09

The experimental `checked_precision_path` accepts `max_steps=2048` and
`retry_order=12`. It returns any admitted order-8 path first, then retries a
refusal at order 12. The ordinary portrait and checked-native defaults remain
unchanged; all results still say `certified=False`.

On the frozen 111 refusals from the 512-step replay, a 1,024-step order-8
budget recovers six, a 2,048-step order-8 budget recovers twelve, and the
2,048-step order-8/order-12 ladder recovers **73**, leaving **38**. Combined
with the preserved 1,289 admissions, this gives **1,362/1,400**. The final
full-corpus replay confirmed this directly: all 399 jobs complete, 73
recoveries, zero regressions. All 109 frontier jobs completed at each
budget, with no lost order-8 admissions.
Forty of the 73 new paths are direct analytic arrivals; 33 retain native
precision prefixes. All 73 saved paths pass 240- and 320-digit independent
rechecks (**146/146**), and all 33 native prefixes pass exact-rational checks
against the original model. Frontier and full replay source/binary hashes
match. The current default Python suite passes **1,008 tests**.

Among the 38 frontier refusals, the last candidate has 29 analytic-check
failures, five prefix refusals, and four inverse matching failures. These
are unresolved, not evidence of impossible connections. Next investigate
the separate failed obligations on those fixed cases; do not widen the
acceptance limits. See `docs/precision_native_bridge.md` and task
`outputs/precision-bridge-budget-sweep/report.md` for replay inputs,
manifests, saved paths, and rechecks. The repo also retains the compact
machine-readable summary and all 38 replayable remaining inputs in
`docs/evidence/precision_bridge_order_ladder_2026-10-09.json` and
`docs/evidence/precision_bridge_remaining_2026-10-09.json`.

## Precision-aware native bridge — 2026-10-04

Implemented an explicit experimental GMP Radau-IIA fallback in
`precision_bridge.checked_precision_path`. It preserves old accepted paths
first, then integrates the original rational model translated around the
incoming point. Condition-planned precision and componentwise stage solves
retain small coordinates. Exported decimal cubics feed an additive exact
entry API in the analytic matcher. The independent checker uses additional
sample locations and a precision floor; prefix, tail, attachment and join
requirements remain separate. Ordinary portrait/checked_native policies are
unchanged and all successful checks still say certified=False.

Final frozen replay: **1,289/1,400 admitted, 33 recoveries, zero regressions**;
all 399 jobs complete. Source/header/adapter/binary hashes stayed unchanged.
All 33 recovered prefixes pass exact-rational checks against the original
model; all 40 saved hybrid paths pass 80/80 rechecks at 240/320 digits.
Validation: 1,007 default Python tests passed, 10 slow/heavy deselected;
41 focused tests and 14 standalone C tests passed; new solver tests passed
ASan/UBSan.

**111 remain**: every bridge reaches the 512-step budget; the last candidate
fails analytic independent checks in 101 cases and matching in ten. Next:
a controlled 1,024/2,048-step budget sweep, preserving the same independent
thresholds, to separate continuation limits from series/fast-coupling errors.
Larger budgets and certification promotion have not been qualified.
See `docs/precision_native_bridge.md`, task `outputs/precision-bridge/report.md`,
and its saved paths, source hashes, recovered/remaining tables and rechecks.


## Resolution-aware native bridge — 2026-10-03

Completed the 144-case investigation. Added an opt-in recorded C bridge
that preserves the incoming anchor, compares ambiguous rounded-jet loss
changes over exact rationals, and probes bounded larger timesteps for zero
movement. Legacy entry points/default portrait policy and all geometric
admission thresholds remain unchanged. Checked handoff policy is now v2.

Full frozen replay: **1,256/1,400 admitted, zero recoveries, zero regressions**;
all 399 jobs completed. The 144 remain unresolved: 77 independently valid
short prefixes, 57 rejected prefixes (33 angular, 37 reversed-flow, 13 both),
and 10 with no retained movement. An additional final-endpoint experiment
finds seven acceptable tails but zero acceptable complete paths. There are
71 cases without a new final binary64 display point. Increasing solver
steps or broadening endpoint search alone does not resolve this frontier.

Validation: 1,002 default Python tests passed, 10 slow/heavy deselected;
13 standalone C tests passed; new bridge tests passed ASan/UBSan. All seven
saved hybrid paths pass 14/14 independent 120/180-digit rechecks. Source and
binary hashes stayed fixed during replay. No commit or push.

Next candidate: a native precision-aware stiff bridge carrying the rational
translation center, original-model field information, state, and smooth
endpoint through export and inverse matching. Assess root refinement against
conditioning; do not just increase precision on already rounded coefficients.
Fast/slow coupling and Frobenius truncation still need scrutiny. Keep all
four independent acceptance obligations. See `docs/native_arrival_bridge.md`
and task `outputs/native-bridge/report.md` for results and saved traces.

The sections below record prior milestones.

## Checked analytic handoff — completed

`native_arrival.checked_path` preserves the original entry and can retain a
native dense prefix before an analytic tail. Every prefix, tail, attachment
and representation join must pass independently. `refine_capture` inserts
both pieces with shifted trace indices; default portrait policy stays legacy.
No C ABI or native solver change was required.

Full frozen replay: 1,256/1,400 admitted, 7 recoveries, zero regressions.
All 399 jobs completed with unchanged numerical source/binary hashes.
Remaining: 144 explicit refusals (121 without a candidate contraction,
18 with an incoming-anchor overwrite, 5 with independently rejected prefix
geometry). All seven saved hybrid paths pass fresh 120/180-digit tail and
attachment checks plus rational prefix/join checks; saved-path hashes match
the full replay artifacts. All original inputs and chord coverage are retained.

Validation: 1,002 default tests passed, 10 slow/heavy tests deselected;
106 targeted integration tests passed before the final two adversarial tests
were added. Both are covered by the final default suite. No commit or push.
See `docs/checked_arrival_handoff.md` and task `outputs/checked-handoff/`.

Next: condition-aware stiff native bridge progress and original-anchor
preservation. A useful analytic tail does not excuse an unchecked or missing
prefix. Slow-Frobenius truncation/fast-coupling control and all four portrait
obligations remain; this is sampled evidence, not a certification gate.

Earlier milestone sections retain their historical counts.

## Condition-aware arrival matching — completed

Native matching now uses condition-checked pivoted Jacobian solves and a
bounded damped inverse-continuation fallback. It still matches the original
entry; intermediate inverse targets never become returned curve segments.
The additive match-inspect C API and Python evidence expose per-call
iterations, backtracking, stages, Jacobian conditioning and residuals.

Frozen replay: 1,249/1,400 admitted, 12 recoveries, zero regressions versus
the preceding divisor milestone. Prior 33 matching failures: 12 passes,
6 geometry refusals, 15 matching refusals. Overall remaining: 136 geometry
and 15 matching failures. Every recovered case agrees at 90/120/180 checker
digits. All 14 remaining slow-Frobenius entries lack a real inverse in both
80- and 100-digit rounded prototype polynomials at the original offset;
1 remaining full-Poincare entry is not classified by scalar elimination.

Validation: 996 default tests, 27 focused tests, all 12 C tests and sanitizer
checks passed. Final allocation-error guard was covered by the focused and
sanitizer suites plus a 480-position allocator-injection probe; full replay
and default suite preceded that error-path-only correction. Details and
provenance: `docs/conditioned_arrival_matching.md`; task artifacts:
`outputs/conditioned-matching/`. No commit or push made.

Next: a retained, independently checked continuation segment to a later
analytic handoff, followed by slow-Frobenius truncation/fast-coupling control.
No default-policy change or certification-gate promotion.

Earlier milestone sections below retain their historical counts.

## Condition-aware arrival divisors — completed 2026-10-02

Native homological construction now uses componentwise modal gaps, a
condition-based GMP bit budget, row equilibration and pivoting, and explicit
backward/forward-error indicators. Exact resonances and unresolved precision
or solves still refuse. The additive diagnostics API preserves existing C
request/result layouts; `divisor_margin` now has componentwise semantics.

Final frozen replay: 1,237/1,400 passes, up from 1,177, with 60 recoveries and
zero regressions. All 183 former divisor refusals progressed: 60 passes,
117 geometry refusals and 6 matching refusals. Remaining overall: 130
geometry and 33 matching failures. All 177 constructed former-divisor cases
agree at 90/120/180 checker digits; the other 6 reproduce matching refusal.
All 399 replay jobs succeeded with unchanged numerical source/binary hashes.

Validation: 993 default Python tests, 85 integration tests, 24 focused tests,
all 12 standalone C tests and ASan/UBSan passed. Default portrait policy is
still legacy; this is sampled evidence, not a uniform certification gate.
Next: condition-aware entry-region/chart inversion and slow-Frobenius
truncation/fast-coupling control. See `docs/conditioned_analytic_arrival.md`
and task artifacts `outputs/conditioned-arrival/`. No commit or push made.

The initial-port section below is historical and its counts are superseded.

## Native analytic arrival backend — completed 2026-10-02

The tested order-8 Poincare / slow-Frobenius proposal now lives in
`src/c/spong_analytic_arrival.c`, with a public create/match/destroy API in
`include/spong/spong_analytic_arrival.h`. GMP construction, two-mode matching
and display evaluation are native; immutable charts are reusable. The Python
adapter retains an independent sampled geometry/attachment admission check.

Use `portrait.compute(..., arrival_policy='checked_native')` or the same
option on `certified_compute`. `arrival_series.refine_capture` defaults to
checked native construction; its `backend='python'` remains the oracle.
Whole-portrait default is still legacy. Refusal removes the terminal chord
and produces `abort_analytic_arrival`. The topology audit explicitly prevents
forced endpoint completion from overriding this new geometric refusal.

All 1,400 frozen inputs match Python admission decisions: 1,177 passes and
223 unresolved entries (183 divisor-screen refusals, 27 matching failures,
13 geometry failures). A 38-entry corrective replay fixed exact-coefficient
transport beyond Python's 4,300-digit string limit without a global override.
Native construction medians: 12.4 ms cold / 11.5 ms cached. Fresh paired
timings: about 40x faster for full Poincare and 15x for slow Frobenius.
Independent checking remains Python and dominates checked-path runtime.

Validation: 985 default tests passed before the serialization correction;
51 integration tests cover the explicit-refusal policy; 19 final focused
tests and the corrective replay cover the final adapter. All 12 standalone
C tests pass; the new C test also passes ASan/UBSan. CMake is unavailable, so
the standalone suite was compiled/linked directly. No commit or push made.

See `docs/native_analytic_arrival.md` for implementation, remaining work,
reproduction and qualification limits. Task outputs are saved under
`outputs/native-arrival/`, including all 223 residual cases and both replay
logs/manifests. Next work: condition-aware divisor screening, validated entry
regions/chart inversion, and truncation/fast-coupling control. No sampled
arrival result has been promoted to a general portrait acceptance gate.

## Large arrival qualification — 2026-09-25

`scripts/qualify_arrivals.py` freezes a paired cohort of 200 random models,
200 directed models, seven zoo cases and 22 original raw-arrival calls.
`scripts/summarize_arrivals.py` retains refusals/censoring and reports paired
outcomes, exact fate, stiffness strata and separate build/checking runtime.
Seed 20260925: all 429 jobs completed and produced 1,400 entries. Analytic
passes 1,177 versus raw 871; on 1,378 terminal tails, analytic passes 1,171
versus connector 377 and extra raw 869. No baseline-only passes occurred.
The original raw-call subset is only analytic 6/22 versus raw 2/22: entry
selection is essential, and the prototype is not a drop-in full replacement.

Analytic non-passes: 183 small-divisor refusals, 27 matching failures and 13
bad constructed curves (11 opposed-flow). The divisor guard also refuses
extreme stiffness without proving resonance. Four raw checking/comparison
stages hit the fixed 30-second deadline. Exact fate is forced on every pass.
The Python proposal's median construction is 0.545 s versus native raw's
89 microseconds, with independent checking timed separately. See
`docs/arrival_evidence.md` for strata, counterexamples and interpretation.

Native analytic work is justified, with validated entry radius/order,
conditioned/resonant handling, retained two-mode evidence and explicit
refusal. No production solver/default/gate was changed. Six new harness and
aggregation tests plus 59 focused arrival/trajectory/checker tests pass.
The complete logs, manifest, CSV, code snapshots and sensitivity probes are
saved in the task's `outputs/arrival-qualification/` directory.

## Minimum arrival entry charts — 2026-09-25

The six failed connectors in the two calibration cases are geometrically
large (10^12–10^14 scaled eps), not target-rounding artifacts. Extending the
raw-time Gauss arrival is insufficient on the stiff minimum. The founding
minimum-entry-chart design was not wired into these terminal captures.

`arrival_series.refine_capture` now proposes matched two-mode analytic
arrivals: full nonresonant Poincare parameterization for moderate ratios,
slow entry jet plus the incoming fast Frobenius term in the stiff regime.
It preserves the preceding branch, retains both incoming amplitudes and an
analytic endpoint, and clears inherited branch certificates. It is opt-in;
original native integrators and production portraits remain unchanged.
`arrival_check` independently checks original-field residuals, point binding,
fast and slow parameter scales, exact minimum identity and limiting geometry.
The experimental four-obligation assessor recognizes these pieces.

`scripts/arrival_probe.py` compares orders/precisions in seconds;
`credibility_probe.py --refine-arrivals --max-sin-squared 1e-6` performs the
full assessment. All six arrival residuals are below 1e-6 (worst 4.74e-9);
the very stiff case improves from 0.996 to 6.80e-17. No production gate is
promoted. Minimal-quartet meets the candidate four-obligation policy; the
far-saddle case remains unresolved only on upstream evidence.
Resonance/near-resonance refuses; no logarithmic or Hadamard arrival
fallback has been implemented. Validation: 963 default tests passed; all seven
zoo goldens hold. The 28 focused arrival/attachment tests also passed. No C
solver was changed in this arrival milestone. See `docs/arrival_evidence.md`
for details.

## Trajectory checker milestone — 2026-09-25

The `trajectory-checker` branch now records every production chord's
construction phase, with explicit handoffs and capture connectors. Jets retain
their selected solve's coefficients, parameters, centered samples and tangents;
grid launches and Hadamard pieces retain their centered/deviation samples.
`trajectory_check` independently checks coverage and can re-evaluate the smooth
jet against the exact model over Q. The portrait ledger reports this evidence
separately; production certification remains unchanged.

The native kernels now export optional schema-1 accepted-stage/Hermite streams,
including normalized and alternate-chart rescues, centered arrival, selected
clock-event trials, replacement vertices and terminal trim events. The old C
entry points and extension return shapes remain available unchanged.
`native_trace` independently reconstructs the smooth curves over Q;
`portrait_check` recomputes inventory/contact checks, launch geometry, exact
endpoint fate, connector geometry and finite-boundary orientation. Scheduling
helpers moved to `_parallel` so the auditor no longer imports the tracer via
`engine`; public resolution exports are lazy so importing the checker alone
does not load `charts`, `engine` or `local`. See `docs/native_trace.md` for
the schema and limitations.

An explicit candidate policy (sin² <= 1e-6, eigen sin² <= 1e-12, binding/join
<= 64 scaled eps) was measured on minimal-quartet and
dead-neuron-far-saddle-d3. Both remain unresolved. Minimal-quartet's native
smooth paths pass but all three capture connectors fail the angular target;
the second case also has unsupported grid/Hadamard evidence. This is a
calibration experiment, not a qualified production policy. No gate or
production certificate has been promoted. Python-delegated centered-chart
rescue remains without native dense evidence and explicitly unresolved.

Validation of the final tree: 946 default tests passed, including the three
recording-versus-legacy export tests. All seven zoo goldens hold. All 11
standalone C tests pass after compiling every C source with
`-Wall -Wextra -Werror`. Analytic and adversarial tests exercise dense
reconstruction, coordinate maps, trim/replacement semantics, malformed
sources, wrong flow, attachments, boundaries and measurement-budget refusal.
Focused parity/topology tests passed 374 tests before the new checker tests.
After the import separation, 18 checker/resolution/export tests passed; the
final 14 analytic native-trace tests also passed. Both heavy convergence
tests passed (146 s). Actual normalized-rescue and
hover-trim export checks passed. Probe reference timing now includes the
independent assessments.

The full standalone continuation corpus replay completed: 22 identical,
3 expected centered-chart delegations, 1 historical mismatch at
`directed:1414065525`, segment 8 (10,355 recorded vs 20,927 current points).
Rebuilding the unmodified HEAD C sources reproduces the same mismatch.
Direct old/new C comparison on all eight segments of that case is bit-identical;
the accepted-step instrumentation did not introduce this corpus drift. The
corpus was not regenerated to hide the discrepancy.

The older State and Inventory sections below describe the branch's starting
point; the `trajectory-checker` branch has since been created. Changes remain
uncommitted.

## State

The cleanup on `level-bar` is recorded as three separate commits on top of
`0c52214` (the previously pushed piece-census/handoff tip):

1. `39f64df` — native pivoted 2x2 arithmetic, corrected numerical claims, and
   the three refreshed goldens.
2. `b0493cd` — distinct oriented launch inventory and ABI 11.
3. This commit, `credibility: specify portrait checks and add exact normal
   measurements` — the four-part contract, native GMP reference evaluator,
   probe, tests, and this refreshed handoff.

The default stub construction remains **manifold jet**;
`SPONG_STUB_MODE=grid` is the comparison arm. The planned next branch is
`trajectory-checker`, forked from the resulting `level-bar` tip after these
three commits are pushed. It has not been created as part of this cleanup.

Pre-commit validation of the combined tree:

- `python -m pytest -q -m heavy`: **2 passed** (146 s). The only warning was
  inability to write pytest's cache because the run began before filesystem
  write access was granted; no test failed.
- Default suite: **901 passed**.
- `python scripts/golden_zoo.py check`: **all assertions hold**.
- The credibility staged snapshot builds independently and passes its 13
  reference tests and all 11 standalone C test executables.
- The arithmetic-only staged snapshot builds independently and passes 27
  local/solver tests plus its standalone C local test.
- The inventory staged snapshot builds independently and passes 57
  topology/portrait/resolution tests plus C topology/resolution tests.

The `_solve2` native adapter and all conditioning/truncation docstring and
comment corrections in `src/spong/local.py` were verified intact before
staging. Shared `_native.c` and C API documentation edits were split in the
index, without reverting or rewriting any working source files. The separate
`slow` test selection was not run.

Historical qualification behind the jet switch: directed N=200, grid 174
certified vs jet 180; random N=200 verdict-identical; zoo claims unchanged.

## Arithmetic cleanup — 2026-09-22, subsequent investigation

The earlier diagnosis of divergence at reach 3.2 was **incorrect**. Decaying
coefficients with successive ratios near 0.1 suggest a radius near 10, not
0.1. High-precision coefficients at orders 20/60/90 have ratios
0.1194/0.1244/0.1252, suggesting a radius near 8 (an estimate, not a proof).
At reach 3.2 the centered displacement is (7.35837e-7, 2.70199), not enormous.
A 90-digit recurrence gives an invariance angle about 3.99e-26; repeating
at 65 digits changes its order-90 endpoint by only 1.22e-58.

The problem was arithmetic in the local 2x2 solve. `_solve2` used row-scaled
Cramer quotients, independently rounding the two components and losing the
stiff equation. The order-two system's spectral condition number is about
8e16. Higher precision for the RHS alone did not repair the angle error;
higher precision for the solve alone reduced it by four orders. Ordinary
back-substitution also repairs the component relation. Exact nonsingularity
of the recurrence is not a numerical conditioning guarantee.

The arithmetic commit introduces `spong_local_solve2` in the native C core:
row equilibration, partial pivoting, FMA elimination and back-substitution,
and an independent componentwise backward-error check. The scaled pivot
product retains the old 1e-12 admission threshold. The Python function is
only an adapter. No radius cap and no angle/transverse acceptance relaxation.
The rest of the Python series construction has not been migrated in this fix.

Measured on the far saddle, unstable + branch, order 60 at reach 3.2:
- original float curve evaluated against the exact field: angle 4.88e-3;
- native replacement: angle 1.55e-6 (native field evaluation: 1.45e-6);
- endpoint normal projection error against the high-precision reference:
  1.02e-19 before, 3.24e-23 after.

Tests cover the actual stiff matrix under row/column permutations and extreme
row scales, refused systems, and both unstable orientations across reaches
0.025, 0.2, 1.6 and 3.2 against the exact field. The far-saddle golden's
sample-dependent counters change as geometry sampling changes.

Truncation agreement still cannot bound errors shared by low-order
coefficients. But this example does not establish the claimed divergent-tail
failure, and must not be used to justify a radius cap. The previous assertion
that the far saddle must refuse reach 0.2 is withdrawn.

The earlier transverse-criterion experiment was reverted. Its measurements
at seed 3389434578 may still motivate a separate domain/error analysis, but
no change to acceptance follows from the present arithmetic repair. Likewise,
finite coefficient ratios are not a convergence certificate.

## Rest of the open list

### Portrait inventory fix (2026-09-22)

The topology audit formerly counted stable/unstable branches globally. On
`minimal-quartet`, replacing either half-branch by a duplicate of its opposite
still certified. This is now rejected, including when the duplicate is
relabelled as the missing branch.

The auditor binds each curve prefix and `critical_steps` to a unique
materialized saddle stub. Each pair must cross a common exact loss level:
below the algebraic saddle value for unstable launches, above it for stable
launches. Exact rational crossing brackets must occupy opposing sides of
`b=b_s` (unstable) or `y=Aa-B=0` (stable), and agree with the declared direction.
The ledger records these local section witnesses. They certify separation of
the supplied launch polylines; they do not supply validated global transport.

The C verdict now requires all four distinct `(saddle index, manifold,
orientation)` slots per saddle, not just counts. ABI 11 adds the required
identity array to `spong_topology_analysis`; topology clients must rebuild.
Missing/duplicate/invalid identities use `branch_inventory_incomplete`.

Validation: the 886-test default suite and all 11 standalone C tests passed.
After adding the prefix-length check and two additional tests, all 55 focused
topology/portrait/resolution tests passed. The heavy tests have since passed;
see State above. The separate slow selection was not run.
The geometry-residual gating and global proof-contract issues identified in
the certification audit remain separate work; this fix addresses launch
identity and inventory. This fix is now in the cleanup commit sequence above.

### Ordinary credibility contract and reference measurements (2026-09-22)

The user's four obligations are now recorded in `docs/portrait_credibility.md`:
complete merge-tree-consistent connections, correct local attachments, credible
pseudo-orbits independently scrutinized against level normals, and acceptable
unbounded-end orientation. Ordinary certification does not require validated
trajectory tubes; those are reserved for exceptional claims.

Added `spong_level_normal_reference` (native GMP) and the thin
`spong.credibility.level_normal_reference` adapter. It measures normalized
midpoint/chord misalignment and exact flow/loss directions with explicit
coverage statuses. `scripts/credibility_probe.py` measures every chord and
reports construction/reference cost and local/continuation/terminal regions.
Production verdicts and acceptance tolerances are unchanged.

The reference uncovered why a raw midpoint-angle gate is unsuitable: on the
far saddle, a nearly 90-degree result is explained by transverse chord sag
`-4.543450056191112e-16`, matching the independent normal-alignment correction
`+4.543447289127111e-16`. Stiffness amplifies the difference between the
straight chord midpoint and the smooth trajectory. A sub-ulp a-coordinate
rounding explanation does not account for this example. Other large-angle
regions remain to be classified; do not infer that all are explained.

Next implementation work is to use retained smooth chart/tangent information
or a justified geometric representation allowance in the ordinary check, then
calibrate the inexpensive arithmetic path against the reference. Do not turn
either historical angle energy or raw exact midpoint angles into a universal
gate. Backbone proximity claims were corrected in comments/documentation.

Validation: 901 default tests passed; all 11 standalone C tests passed. Native
reference tests cover independent rational-oracle agreement, resampling,
reversed flow, coefficient cancellation, extreme scaling, nonfinite data,
stationary midpoints, repeated points, and ratio underflow. The heavy tests
have since passed; see State above. This work completes the cleanup commit
sequence; the separate slow selection was not run.

### Previously open work

1. **directed seed 1495454581** — refuses under the jet with same-destination
   crossings that PRECEDE certification.  A correct refusal: two branches
   pressed together by the lambda-lemma, a non-adjacent attachment passing
   near saddle b = -1.0247 on its way to the minimum at b = -1536.  The fix is
   the SEPARATION CERTIFICATE (`scripts/separation_certificate.py`) deciding
   which side of that saddle's stable manifold branch 13 passes — not a
   contact exemption.  This is the substantial item.
2. **seeds 953953598 and 1886674721** — segment budget under the jet, fates
   identical to the grid's.  Cost, not correctness.  `docs/normal_form.md`
   explains why no current chart handles 953953598: its backbone dives to
   a* = -4.45e7 at b = 0, and its inner branch wants (a,b) for the traverse
   and (b,v) for the plunge.
3. **The series atlas** (analytic continuation, multiple charts) — to be
   DESIGNED, not grown one probe at a time.  Evidence and constraints are in
   `docs/normal_form.md`; note especially that charts must be graphs or loss,
   never time, and that accurate Taylor shifts at large order mean structured
   O(N^2), not FFT.
4. **The note** (`note/spong-note.tex`) — sections 1-6 and Appendix A written,
   7-10 stubbed.  Section 7 ("absence": the separation certificate as
   mathematics) is next, and the cross-reference in section 5's remark points
   at `sec:certify` awaiting section 7's label.  Sections 8-9 have their
   citations settled (Dereich-Jentzen-Kassing arXiv:2511.04622 for Adam;
   Asmussen & Glynn ch. V for regression-adjusted control variates).

## Inventory for the checker branch (measured 2026-09-22)

The new branch builds a trajectory OBJECT plus an independent CHECKER.  Its
vocabulary must cover every construction the tracer can emit; it is open by
design (there is always a worse example), so an unrecognised piece is
REFUSED, never passed -- incompleteness costs certifications, never
correctness.  Each new construction ships with its checker step and the seed
that motivated it.  The checker must not import tracer code: it re-derives
its evidence from the model (exact polynomials, enumeration, merge tree), and
should evaluate the flow through a different arithmetic path from the
tracer's (exact rationals, or the C reference evaluator) so a shared bug
cannot hide.

`scripts/piece_census.py` is the instrument.  Default mode certifies every
case; `--trace-only` traces without certifying, only models screened as
difficult from exact data (saddle eigenvalue ratio, A-range along the
backbone), so thousands can be surveyed.  Two runs so far: 207 certified
cases (zoo + 100 random + 100 directed), and 626 difficult cases traced from
1,000 screened.

**Production vocabulary** (what actually fires):

- launch: the manifold jet (~95% of stubs); the Poincare grid only where
  the jet DECLINES (every such stub unready); centred grid with extension in
  three cases; outright refusals.
- continue: zones `shallow`, `engine`, `shallow_rejected`, plus the
  **unzoned** default path -- the LARGEST bucket (1,760 unstable branches and
  every stable branch in the survey) and currently OPAQUE, since only the
  stiff dispatcher records zones.  Making the tracer label every piece is the
  first concrete task: the biggest checker step cannot be specified until
  its insides are visible.
- terminate: `capture`, `box_exit`, `level_bar` (nearly all stable
  branches).
- endpoint: overwhelmingly `exact_merge_tree`, `exact_merge_tree_escape`,
  `exact_superlevel_enclosure` -- already exact and model-level, so the
  topology certifiers are most of a checker nucleus as they stand.
  `forced_completion` is rare.

**Rare kinds and their seeds** (regression cases for the checker):

    directed 206935347   jet DECLINED a stable manifold -> unready Poincare
                         grid stub -> abort_conditioning_handoff (stiff 1e39)
    directed 639780423   abort_nonfinite on an unstable branch
    directed 483412341, 835893398, 61899905
                         centred grid with centred extension, ready
    directed 1283395251, 202251424   forced_completion
    directed 230086462   stable branch abort_step_failure
    random 1216867987, directed 908892439
                         stable tail refused, no_box_boundary_crossing
    directed 908892439   no_bounded_capture_component

**Rare fallback certifiers** -- `exact_sublevel_tube`, `exact_backbone_funnel`,
`exact_superlevel_product` -- never closed a branch in either run.  None was
ever removed; all predate 877a774 (2026-08-10, "merge tree decides
capture"), after which the merge tree wins first.  Live, rare, and needing
coverage: a census mode that DISABLES the winning certificate would force
them to run on ordinary cases, and disagreement with the winner would find a
bug in one of them.

**Left behind: `charts.trace_valley_exit`** (the slaved backbone stretch).
Never fired in 833 cases.  Its production call was removed in caf3448
(2026-07-29, "Add native certified portrait core and qualification"); it
survives only as a comparison arm in `scripts/box_experiment_arms.py`.  The
tracer cannot emit it, so the checker needs no step, and the new branch should
not carry it forward.

**The zone vocabulary looks closed at this level** -- the same four zones
across 833 cases.  New behaviour appeared at LAUNCH and TERMINATION, and only
above stiffness ~1e30.  For larger surveys, raise `--min-arange` to about 12
(1e6 lets ordinary models through: 626 of 1,000 passed) or screen on
stiffness alone.

## Working notes

- The machine was on BATTERY at the end of the session: wall times from then
  are not comparable with earlier ones.
- Ensemble A/B invocation, both arms in one pool, longest-first:

      python scripts/ensemble.py --cases 200 --mode directed --jobs 16 \
          --stub-modes grid,jet --out out/dir.jsonl \
          --order-from out/ens_directed_grid.jsonl \
          --order-from out/ens_directed_jet.jsonl

  Wall time is floored by the single slowest case (1024 s last run, which IS
  case 137's time).  Only making that case faster helps.
- `python -m pytest -q -m heavy` must be run deliberately before a merge; the
  two heaviest parameters are held out of the default loop.
- Probes written this session: `scripts/stub_bias_probe.py`,
  `scripts/manifold_jet_probe.py`, `scripts/normal_form_probe.py`.
