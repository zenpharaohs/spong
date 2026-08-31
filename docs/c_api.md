# Native core and frontend contract

SPONG has one computational implementation and multiple adapters.  Python,
MATLAB, command-line tools, and mobile applications must not reproduce
qualification thresholds or terminal-state logic.

The public C99 ABI begins in `include/spong/spong_resolution.h`.  The
resolution-policy portion is:

- allocation-free and exception-free;
- expressed with fixed-width fields;
- versioned by `SPONG_ABI_VERSION` and `spong_abi_version()`;
- usable as a static or shared library;
- independent of Python and NumPy; and
- covered by a standalone C executable in addition to Python parity tests.

The terminal values are stable:

```c
SPONG_CERTIFIED_NON_MORSE
SPONG_MORSE_NUMERICALLY_UNRESOLVED
SPONG_CERTIFIED_PORTRAIT
```

`spong_resolution_preflight` applies the common numerical admission policy.
`spong_resolution_finalize` maps the a-posteriori geometry certificate to the
same terminal result.  The reason mask preserves every simultaneous refusal
while `primary_reason` gives frontends a deterministic headline.

Exact real-root analysis is exposed by `include/spong/spong_exact.h`.
Multiprecision is presently supplied by GMP behind that SPONG-owned ABI; GMP
objects and limb layouts never cross the boundary.  Polynomial coefficients
cross as canonical decimal integers, so MATLAB, Python, Swift, and a future
alternative multiprecision backend see the same representation.

`spong_sturm_analyze_decimal` returns:

- the distinct real-root count;
- the number of real roots of \(\gcd(p,p')\), which is the exact real
  non-Morse test;
- input and squarefree degrees; and
- PRS steps, chain size, and peak coefficient bits.

Caller-supplied bit, chain-size, and PRS-step ceilings turn excessive exact
work into `SPONG_EXACT_WORK_LIMIT`; no result is silently truncated.
The decimal input is screened against the bit ceiling before GMP parsing, and
every rational polynomial produced during division is observed—not only the
final primitive chain—so temporary coefficient swell is part of the recorded
peak.

For repeated bounded queries, `spong_sturm_plan_create_decimal` constructs one
opaque, persistent squarefree Sturm chain.  `spong_sturm_plan_count` then
counts roots on exact rational intervals with the convention `(lower, upper]`;
a null lower or upper numerator denotes the corresponding infinity.  Plan
ownership never crosses into GMP: callers release it with
`spong_sturm_plan_destroy`.

`spong_sturm_plan_isolate` uses the same persistent chain to isolate every
distinct real root.  Its explicit work policy independently bounds:

- subdivision nodes;
- puncture halvings around an exact rational root;
- rational endpoint bit length; and
- output interval count.

It returns exact decimal numerator/denominator strings and never returns a
partial list on refusal.  The library owns this memory, so every frontend must
release it with `spong_root_intervals_destroy`; this is required in particular
when the core is a Windows DLL.  The isolator uses an iterative stack rather
than C recursion, and certifies that a puncture contains exactly the rational
root it removes before continuing on either side.

`spong_sturm_plan_refine` validates a one-root interval and bisects it to an
exact rational relative-width target.  Bisection count and endpoint bit length
have independent limits.  `spong_sturm_plan_sign_at` evaluates the sign of the
original polynomial retained by the plan—not its squarefree reduction—so
multiplicity and global-sign information are preserved for interval
certificates.

`spong_sturm_plan_sign_polynomial_at_root` certifies the sign of a second
integer polynomial at the unique algebraic root in an isolating interval.
Exact Horner interval arithmetic and direct root bisection are fused in GMP;
the policy bounds both bisections and endpoint growth.  If the query may share
the algebraic root, the result is explicitly unresolved instead of assigning a
sign.  This fused path also avoids a historical frontend refinement stall in
which a relative-width request could already exceed a symmetric interval.

The production Python path delegates unbounded counts, bounded rational
counts, root isolation, refinement, and interval-sign evaluation to this C
implementation.  It caches the persistent plan by primitive integer
polynomial.  The original Fraction/bigint implementations remain independent
differential oracles.

An installed package exposes `spong-validate-native`.  The command exercises
this differential boundary on seeded mundane random inputs, targeted
close/far/repeated-root families, and the complete named zoo.  It emits the
versioned `spong-native-validation-v1` JSON format when requested and exits
nonzero if any case disagrees.  Parallel execution is used when available and
falls back explicitly to serial execution on restricted hosts.  For a quick
loader/ABI smoke test, use:

```sh
spong-validate-native --no-zoo --mundane 16 \
  --targeted-per-exponent 4 --output validation.json
```

GMP's limb allocator is process-global and does not provide a recoverable
per-operation out-of-memory return.  SPONG therefore admits exact work using
conservative bit/chain/step ceilings before relying on GMP allocation.
Array-allocation failures in SPONG itself are returned explicitly.  Replacing
the multiprecision backend later does not alter the public ABI.

Build and test the frontend-independent library with:

```sh
cmake -S . -B build-c -DSPONG_BUILD_C_TESTS=ON
cmake --build build-c
ctest --test-dir build-c --output-on-failure
```

Set `SPONG_BUILD_SHARED=ON` for a dynamic library.  MATLAB MEX may link either
form; Apple and Android application builds normally consume the static target.
Cross builds must supply GMP through `GMP_INCLUDE_DIR` and `GMP_LIBRARY`;
Python builds may set `SPONG_GMP_PREFIX`.

Global contact certification begins in
`include/spong/spong_topology.h`.  `spong_contact_scan_create` builds the same
balanced bounding-volume hierarchy used by the Python oracle, while
`spong_contact_scan_next` streams transverse and FP64-ambiguous segment events
without allocating a potentially enormous result list.  Pair scans compare
two invariant manifolds; self scans visit every nonadjacent segment pair once.
The caller retains the packed `(a,b)` arrays until
`spong_contact_scan_destroy` and can stop immediately when its event budget is
exhausted.  The standalone C test and randomized Python differential test
exercise the same orientation, proximity, and intersection-point formulas.
`spong_topology_decide` then reduces branch counts, work budgets, contacts,
and endpoint-certificate counts to the common status and deterministic refusal
reason.  This state machine is separately compared against the Python oracle
over randomized evidence combinations.  These topology entry points comprise
ABI version 2; version 1 resolution callers remain source- and binary-compatible.

Batched geometric measurements are exposed by
`include/spong/spong_geometry.h`.  `spong_curve_diagnostics` evaluates the
angle-energy certificate and its complementary unresolved-tail backbone
residual in one pass over a packed polyline.  The C implementation explicitly
uses the same separate binary64 multiply/add rounding as the NumPy oracle;
this matters on the high-cancellation tails for which the diagnostic measures
its own evaluation floor.  The scalar Python implementations remain
differential oracles rather than production loops.

High-precision near-critical chart composition is exposed by
`include/spong/spong_local.h`.  `spong_poincare_pullback_decimal` forms
`adj(DT) F(T(z))` for the selected quadratic Poincaré map in GMP floating
arithmetic, restores the structural critical point and diagonal linear part,
and rounds the completed coefficient array to binary64 once.  Decimal spectral
data cross as strings, while the map coefficients cross as the exact doubles
that will map the resulting curve.  The production path uses at least 192 bits
and is differentially checked against the retained Decimal implementation.

The next local entry point is specified by the independent
`spong.local_certificate` oracle.  It will consume rectangular GMP rational
interval-coefficient arrays for the centered two-component field, potential,
and lifted `y`, together with exact dyadic frame/centre data and fixed work
budgets.  It will return the stable integer status contract, exact cone and
face margins, rational section rectangle, and work counters documented in
`local_graph_certificate.md`.  That entry point is intentionally not declared
in the public header until differential parity and refusal behavior are fixed;
the Python oracle is the executable ABI specification.

Two-dimensional Gauss--Legendre collocation on the loss field is exposed by
`include/spong/spong_gauss2.h`.  `spong_field` is the plain-array view of
the eight ascending coefficient arrays and the loss constant (layout
identical to `spong_continue_field`); `spong_irk2_step` takes one implicit
Gauss--Legendre step of order 4, 6 or 8 on any `spong_vec_fj` field with the
damped Newton stage solve and the equilibrated, backward-error-certified
small dense solve, and `spong_normalized_step` / `spong_potential_step` are
the unit-speed and constant-potential-rate fields on the loss.  Loss,
gradient and Hessian evaluation are provided alongside.  These bodies were
relocated verbatim from the CPython extension (the goldens are bit-identical
across the move); the extension's `Kernel.normalized_step`,
`Kernel.potential_step` and the `LocalKernel` steps are now thin adapters.
A standalone C test exercises the steps on analytic fields.  This is the
stepper the potential-rate segment entry point (below) is built on.

The three constant-potential-rate phases of chart continuation are exposed
as one entry point by `include/spong/spong_potential.h`.
`spong_potential_rate_segment` runs a `spong_potential_request` in one of
three modes -- `SPONG_POT_PREFIX` (descent toward one target minimum),
`SPONG_POT_LEVEL_EVENT` (descent to the next candidate level, with capture
against a target list) or `SPONG_POT_ASCENT` (ascent to the box boundary) --
on a `spong_field`, with the target and critical-point lists as packed
`(a,b)` arrays, the box, the retry budgets and the step-fraction and
primary-order policy constants passed in explicitly, so the library holds no
tunables of its own.  Vertices are written to a caller-supplied packed
buffer; when it overflows the call returns `SPONG_POT_NEED_CAPACITY` with the
exact count required in `n_points`, and the caller regrows and retries.  The
`spong_potential_result` carries the terminal condition (the same eight
terms the Python loops report, `SPONG_POT_NEAR_TARGET` through
`SPONG_POT_UNAVAILABLE`), the endpoint, the captured target and event level,
and every counter of the phase's diagnostics -- accepted, rejected,
critical-capped and arclength steps, GL8 attempts and acceptances, and the
maximal Richardson and interpolation errors -- so the extension's engine
diagnostics are assembled without recomputation.  The GIL is released for
the whole segment.

Parity is defined by `tests/corpus/potential_rate.json`, recorded by
`scripts/potential_corpus.py` from the requests the phases receive during
ordinary zoo portraits and the answers the Python loops give; the check and
`tests/test_potential_parity.py` demand that both the Python oracle and the
C entry point reproduce every vertex, endpoint, term and counter to the
last bit.  This is meaningful only under the shared-arithmetic doctrine now
in force: the oracle loops (`charts._potential_rate_*_python`, the
executable specification) evaluate the loss, gradient and Hessian through
`Kernel.loss`, `Kernel.gradient` and `Kernel.hessian` -- the library's
Horner kernels, the arithmetic the segment uses -- rather than the model's
range-guarded evaluators, and the curvature cap is written out scalar by
scalar rather than through a small NumPy product.  The comparison therefore
judges loop logic, not evaluator ulps.  Both doctrine changes alter the
specification itself, not just the port, and may move goldens; such drift
is engine-agnostic by construction and is re-recorded, not chased.

## Migration boundary

The present production geometry kernels are already written in C, but some are
still hosted inside the CPython extension type in `src/spong/_native.c`.
Migration proceeds by moving computation—not translating it—into `src/c`:

1. resolution policy and terminal states;
2. binary64 qualification and small exact matrix calculus;
3. GL4/GL6/GL8 and dense collocation;
4. local Poincaré/graph-transform charts;
5. global continuation and topology ledger (the streaming BVH contact kernel,
   batched curve diagnostics, and certificate state machine are native;
   endpoint-proof orchestration and ledger assembly remain); and
6. exact rational polynomial/Sturm analysis through a portable multiprecision
   backend.

The Python implementations remain parity oracles during migration.  A migrated
operation is complete only when the Python, standalone C, and extension-binding
tests agree.  Frontend presentation and serialization may remain in their host
languages; mathematical decisions and numerical work may not.
