# Native core and frontend contract

SPONG has one computational implementation and multiple adapters.  Python,
MATLAB, command-line tools, and mobile applications must not reproduce
qualification thresholds or terminal-state logic.

The public C99 ABI begins in `include/spong/spong_resolution.h`.  It is:

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

## Migration boundary

The present production geometry kernels are already written in C, but some are
still hosted inside the CPython extension type in `src/spong/_native.c`.
Migration proceeds by moving computation—not translating it—into `src/c`:

1. resolution policy and terminal states;
2. binary64 qualification and small exact matrix calculus;
3. GL4/GL6/GL8 and dense collocation;
4. local Poincaré/graph-transform charts;
5. global continuation and topology ledger; and
6. exact rational polynomial/Sturm analysis through a portable multiprecision
   backend.

The Python implementations remain parity oracles during migration.  A migrated
operation is complete only when the Python, standalone C, and extension-binding
tests agree.  Frontend presentation and serialization may remain in their host
languages; mathematical decisions and numerical work may not.
