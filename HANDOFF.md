# Handoff — spong, level-bar branch, 2026-09-22 (afternoon)

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
