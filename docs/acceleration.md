# Acceleration: the native engine, parallelism, and what the audit costs

This records the work on the `native-dispatcher` branch: what changed, how it
is verified, what it bought, and — at least as usefully — what was tried and
does not work.  Everything here is measured; the failed attempts are included
because each of them looked obviously right beforehand.

## What changed

**The continuation dispatcher runs in C.**  `charts._continue_curve` now
dispatches to `spong_continue_curve` (`src/c/spong_continue.c`), which marches
one engine segment end to end — chart switching, the shallow handoff, step
halving, the descent-realization test, the turn budget, capture, box exit —
and releases the GIL for the whole call.  The Gauss stages it takes are the
same `Kernel.slow_step` / `fast_step` the Python engine already used; what
moved is the loop around them.

`charts._continue_curve_python` is retained as the reference implementation.
It is not scaffolding: it is what a skeptic runs to check that the fast path
did not change a portrait, and it is selected with `SPONG_ENGINE=python`.

**Paths the segment corpus cannot judge are delegated, not ported.**  The
floor-fallback ladder, the normalized-arclength rescue, the centered rescue
and the stall trim are never reached by any segment the zoo produces, so the C
returns `SPONG_CONT_DELEGATE` and the caller re-runs the *whole* segment in
the reference implementation.  Resuming mid-segment would have to carry the
active chart, the ramped chord, the floor fixed at launch and the stall window
across the boundary; re-running is identical by construction and costs nothing
in a case that has never yet occurred.  Across a full `tricky-d11` portrait —
1.4 million native steps — there were zero delegations.

**Branch tracing and the topology audit run concurrently.**
`engine.map_ordered` assembles results in *submission* order, never a
completion order or a sort key, so raising the worker count cannot move a
ledger entry or change a golden.  `portrait.compute` maps both branch loops
through it; `topology.audit` maps its per-branch endpoint-certificate loop.
The GIL is released around `SturmPlan.refine`, `isolate` and
`sign_polynomial_at_root`, which is what makes the audit's concurrency real —
the plan is `const` for a query, results are allocated per call, and
`spong_exact_gmp.c` has no mutable file statics.

**Exact root isolation and refinement are memoized.**  `isolate_roots` and
`refine` are pure in their arguments but were recomputed from scratch on every
call.  Cached on the primitive integer polynomial and the isolating interval;
calls that request `stats` bypass the cache, since work counters belong to the
call that did the work.

**The step budget is derived from the geometry.**  `max_steps` was a constant
200000 while `ds` shrinks and the box grows with the escalation level, so
headroom fell from ~7x at level 0 to 0.70x at level 2 and seven stable
branches aborted there in every escalating case; `topology.audit` then refused
on `branch_set_incomplete`.  It is now `max(200000, 8*diagonal/ds)`, floored so
that no trace which completes today can newly fail.

## How it is verified

Four tiers, each finer than the one below it:

| tier | what it pins | where |
|---|---|---|
| step | one Gauss stage solve, bit-comparable | `tests/test_native_parity.py` |
| segment | one `_continue_curve` call, bit-identical | `tests/corpus/continue_curve.json` |
| portrait | skeleton, certificates, topology verdict | `tests/golden/` |
| sampling | random and directed models | `scripts/qualify.py` |

The segment corpus records what each call was *asked* and what it *answered*
— term, switch count, final state, vertex count, and a SHA-256 of the packed
polyline.  The native port reproduces all 45 segments bit-identically.

`scripts/qualify.py` has three independent legs: invariants that need no
oracle at all, determinism across worker counts, and reference agreement
sampled at **100% of refusals** plus a configurable fraction of certifying
cases.  Refusals are sampled exhaustively on purpose: an engine difference
flips a verdict far more readily than it alters a certified skeleton.

Directed cases matter more than random ones.  `inverse.straddle_case`
prescribes a critical point at a chosen large `|b|`, which is where the
stiffness is; 18 of 20 directed models refuse against 11 of 60 undirected.
Cumulative to date: 80 models, 28 reference checks, 11 determinism checks,
zero failures.

## Configuration

    SPONG_ENGINE=python|native     default python; the explorer defaults native
    SPONG_WORKERS=N                default 1

Threads help only under the native engine.  Under the Python engine the trace
loop holds the GIL and a pool costs contention for no speed — measured at
1.00x on eight threads.

## What it bought, and where the time now goes

On `tricky-d11`: the engine took a portrait from 188s to 80s (2.35x wall,
2.51x geometry).  Its level-0 audit went 24.0s serial, 7.7s at eight workers,
1.66s with memoization as well — about 14x on what was 80% of the runtime.

The distribution is worth knowing, because it is not where anyone guesses.
On `tricky-d11` under the native engine: branch tracing 8.2s over 16 branches
and 1.87M vertices, `build_ledger` 0.03s, and `topology.audit` the rest.
**Branch tracing is about 6% of a portrait.**

Audit cost is not proportional to vertex count.  It is the exact
contact-resolution and certificate work: sign of a polynomial at a real
algebraic number, decided by GMP Sturm machinery.  On
`linear-target-d17-thrash` at level 0 that is roughly 151 distinct
degree-98/136 test polynomials, each needing a Sturm chain at seconds apiece,
generated by `_unstable_far_field_funnel` and `_sublevel_component_inventory`.
That case takes 808s for a level-0 portrait of which 714s is the audit, and it
is inherent: see the negative results below.

## Negative results

Five things that looked right and are not.  Each cost a measured experiment.

**Integer convolution in `_poly.mul` is much slower.**  Clearing to the LCM of
all denominators inflates every coefficient to the size of the worst one, and
denominators compound through repeated products (`A`, `A^2`, `A^4` reaches
degree 136 in the far-field funnel), so the big-int operands outgrow what
`Fraction`'s per-coefficient reduction keeps them at.  Many small gcds beat a
few enormous multiplies.

**Subresultant PRS would be worse, not better.**  Measured peak chain
coefficient bits, primitive versus subresultant: A 8100 vs 10381, N 24385 vs
27873, A^2 8257 vs 20116, A^4 8573 vs 39554 — ratios 0.78, 0.87, 0.41, 0.22.
Primitive PRS is the coefficient-*optimal* variant; subresultant trades larger
coefficients for avoiding content computation.  `sturm_chain_build` already
uses the right one.

**`peak_coefficient_bits` is not the chain's coefficient size.**  It is the
conservative *intermediate* PRS peak, as `sturm.py`'s docstring says.  Reading
it as chain size suggests a 150-250x growth factor that is not there.

**The `_native_sturm_plan` cache was not thrashing.**  6823 hits against 151
misses with a working set of 151; raising `maxsize` from 512 to 4096 changed a
718s audit by 4s.  The cost is the 151 misses themselves.

**Hoisting loop-invariants out of `scaled_radial` bought 0.1s in 807s.**  The
hoists are kept — they are provably value-preserving and harmless — but the
width ladder exits early enough that the products were not being rebuilt
anything like as often as the code shape suggests.

## A methodological note

`cProfile` instruments the main thread only.  Once the audit runs in a worker
pool, every profile of it is blind to the actual work and shows
`threading.join` at the top.  Use an explicit timing wrapper with a lock.

Profile the **cold** path.  A warm audit on `linear-target-d17-thrash` takes
26s and appears to be dominated by `_poly.mul`; the cold one takes 713s and is
dominated by Sturm chain construction, because the warm run had the chains
cached.  Two of the five negative results above came from profiling warm and
inferring cold.

Timings on a laptop swing by 2-5x with thermal and power state.  Only
within-run comparisons are trustworthy; `caffeinate -disu` helps and does not
fully prevent it.

## Not done

`topology.audit` carries a fixed `segment_budget = 1000000` which does not
scale with the escalation ladder, so level 2 now traces to completion and then
refuses with `certification_segment_budget`.  Unlike `max_steps`, lifting it
makes the contact scan do work that grows with the material, and
`linear-target-d17-thrash` already spends minutes there at level 0 — measure
before changing.

The self-contact scan is still serial.  It carries `nonlocal` counters and an
order-dependent early break, so parallelising it needs per-pair event lists
merged in submission order with the budget applied during the merge.  It is
cheap on every case measured so far.

For interactive use the remaining win is architectural, not numerical: the
level-0 portrait exists after ~95s of the 808s on the worst case, and the
escalation ladder refines the *verdict*, not the picture.  A viewer should
show the geometry as soon as it exists.
