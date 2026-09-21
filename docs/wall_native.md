# Native numerical rheostat

The C99/GMP implementation in `src/c/spong_rheostat_gmp.c` replaces the
numerical work of the Python rheostat prototype. Its public interface is
`include/spong/spong_rheostat.h` (ABI 10). The Python adapter `wall_native.py`
marshals inputs and outputs; the independent Decimal implementation remains
in `wall_continue.py` for differential testing.

This is a **numerical proposal engine**, not an interval certificate. It can
propose a saddle connection parameter or an estimated distance along the
rheostat. It cannot certify a connection, uniqueness, Smale, or distance to
the nearest wall. The experimental connectivity orchestrator uses this
backend for its optional diagnostics; those diagnostics cannot promote a
certification verdict. The connectivity orchestrator itself remains Python
and has not been promoted to the production resolution core.

## Arithmetic and numerical decisions

- Model A, B, C and isolating intervals enter as rational strings. Native
  code reconstructs B or N=2AB'-A'B, verifies the selected root interval
  with native Sturm machinery, and refines it by exact rational bisection.
  It never substitutes an unchecked Newton root. The Python adapter checks
  the existing exact A>0/Morse hypotheses and selects saddles from that
  inventory. The C API is a numerical engine, not a replacement Morse test.
- GMP arithmetic uses 192 requested bits by default and 64 extra guard bits.
  All precision is local to a context; no global GMP precision is changed.
  Supported requested precision is 128–512 bits. Polynomial degree is capped
  at 64, germ order at 20, nominal step count at 65,536, and temporary scalar
  storage at 262,144 slots. Exact Sturm work also has explicit budgets.
- The large eigenvalue uses the non-cancelling trace/discriminant expression;
  the small one uses determinant/large. Determinant cancellation and small
  spectral resolution have explicit refusal guards. Launch recurrence
  systems use the same guarded solver as collocation.
- All small solves use row equilibration and partial pivoting, reject pivots
  below 64 times the requested-precision epsilon, and independently check
  both value and sensitivity corrections against the original system.
  Reported backward errors are numerical metrology, not interval proofs.
- The degree-p germ and its Lambda derivative are recomputed in C.
  GL4/6/8 integrate in logarithmic loss distance; GMP logarithm/exponential
  use bounded range reduction and series. Primal and sensitivity Newton
  residuals have separate scales. Failed stage solves trigger bounded step
  halving; exhausted budgets produce explicit refusals.
- A coefficient-magnitude gradient floor rejects unresolved global field
  evaluation. Projection denominators have cancellation guards. Root-finder
  tolerances below the requested working-precision floor are rejected.
- Projection to the fixed loss section and its sensitivity are native.
  An exact rational interval normal-derivative check prevents aliasing
  between sheets on the computed crossing rectangle, including b-turns.
  This check does not enclose the true trajectories.
- Safeguarded sensitivity-Newton locates the numerical gap zero. Legacy
  brackets can be padded and repaired within explicit search bounds, but
  fresh native gap signs must establish the bracket. Numerical stopping
  conditions are kept in C, not duplicated in adapters.

The fixed step count is a discretization choice, not an accuracy guarantee.
The Newton correction tolerance applies to the discrete shooting map.
`numerical_bracket_width` can remain broad when Newton's correction converges;
it must not be described as a tight interval for the true connection.
Launch, step, section, and precision studies remain necessary before quoting
extra digits. Native contexts fix these settings at construction; create a
new context to change them.

## Use

```python
from spong.wall_native import NativeSectionContinuation

c = NativeSectionContinuation(model, source_index=i, target_index=j,
                              steps=1024, precision_bits=192)
result = c.find_root('2.1776', '2.1778', parameter_tol='1e-25')
print(result.evaluation.lam, result.max_backward_error)

# An old numerical bracket is a seed, not a certificate.
result = c.find_root_from_brackets(
    [('2.1777095613653', '2.1777095613654')], bounds=('2.17', '2.19'))
```

`source_b`/`target_b` remain convenience selectors for comparison with the
old studies. Inventory indices are preferred; tied approximate selectors
are refused. Returned crossings include the pre-projection loss residual,
launch invariance residual, sensitivity, and projection distance. Step counts
are nominal; extra subdivisions are reported separately.

## Qualification checks and reference study

The focused Python suite passed 92 tests, covering new native/oracle parity,
independent finite differences, convergence, failure paths, warm starts,
connectivity proof boundaries, and existing Gauss/resolution regressions.
Standalone C ABI and private arithmetic tests passed with Clang address and
undefined-behavior sanitizers. The arithmetic test covers row scales from
1e-100 to 1e100, near-singular rejection, pivoting, elementary-function
round trips, cancellation-resistant eigenvalues, and projection across a
b-turn. CMake targets are provided; CMake was unavailable on this host, so
the standalone tests were compiled directly with Clang.

On `nonnearest-saddle-connection`, GL8, order-10 germ, radius .001,
midpoint section, and root correction tolerance 1e-40:

| Nominal steps | Numerical Lambda | Elapsed seconds |
|---:|---|---:|
| 512 | 2.1777095639548159944696686155459093 | 3.17 |
| 1024 | 2.1777095639548159933205552120783913 | 5.25 |
| 2048 | 2.1777095639548159933160564860326184 | 10.45 |
| 4096 | 2.1777095639548159933160389030833429 | 21.08 |

At 4096 steps, changing requested precision to 256 bits agrees through about
57 decimal places. Order 12/radius .002 gives
2.1777095639548159933160388648278968; section fraction .35 gives
2.1777095639548159933160392255082116. These differences indicate remaining
trajectory-discretization effects, not arithmetic precision exhaustion.
A conservative summary of this study is Lambda approximately
**2.177709563954815993316**. The rounded binary64 answer remains
**2.177709563954816**. The port therefore confirms the previous binary64
estimate and makes finer convergence studies substantially faster; it does
not reveal a correction at binary64 precision.

Reproduce with:

```sh
PYTHONPATH=src .venv/bin/python scripts/wall_native_probe.py \
    --steps 512 1024 2048 4096 --cross-checks --output /tmp/wall-native.json
```


`NativeSectionContinuation.launch(Lambda)` exposes the same native GMP
invariant-manifold germs used by loss continuation. Its points are in base
(alpha,b) coordinates; convert alpha to physical a=alpha/Lambda and divide
its base-loss section by Lambda for shots of the scaled model. The returned
sensitivities differentiate launch points, not section crossings. The C API
is `spong_rheostat_launch`, using the existing result point/sensitivity slots;
gap and projection fields are zero placeholders, not connection evidence.


## Native paired connections and display sections (ABI 10)

`spong_rheostat_pair.h` implements the entire bounded two-parameter solve in
C99/GMP. `locate_connection_pair` assembles the exact rational family
A=A0+eta*A1+eta²*A2, B=B0+eta*B1, C fixed, from g=g_base+eta*g_direction.
Python selects the initial four saddles and supplies fixed rational boxes;
every native evaluation revalidates those same boxes. A lost or ambiguous
root is refused, never replaced by whichever saddle is now nearest.

The two numerical section charts stay fixed during the solve. Lambda
sensitivities come from the native differentiated trajectory. Eta derivatives
use centered differences at h and h/2 with Richardson extrapolation and bounded
refinement. Their difference is an error *estimate*, not a rigorous enclosure.
The Jacobian is scaled by the explicit parameter spans and row magnitudes;
its determinant must resolve against derivative error and arithmetic epsilon.
The existing equilibrated pivoted solver checks backward error. Bounded
backtracking must reduce the scaled residual, and success requires both the
scaled residual and scaled correction to meet tolerance. Duplicate constraints
are refused as rank deficient, including near a numerical root.

The exact PRS coefficient budget for saddle validation is 512 times the internal
precision in bits (requested precision plus 64), with the existing operation
and subdivision budgets unchanged. This allows the larger rational coefficients
produced by native eta iterates without substituting a floating root finder.
Budget exhaustion remains a refusal.

`spong_section.h` owns the inspector's numerical section tracing. GL8 full/two
half-step comparison controls local integration error; a safeguarded scalar
bracket refines the event using fractional GL8 integrations. There is no chord
fallback. Unresolved gradients, loss-event conditioning, failed solves, escape,
and exhausted budgets have distinct refusal statuses. The roundoff scale and
step-difference estimate are numerical diagnostics, not validated bounds.
Large loss offsets can cause refusal when the requested coordinate tolerance
cannot be resolved in binary64. Python draws the returned points and raises an
explicit error for refused crossings.

The paired solver's small residual certifies neither a true connection nor
codimension two. Step, launch, section and precision studies are still required.
The inspector's stored rounded eta fixture and a newly refined eta proposal
are slightly different landscapes; do not interpret their Lambda difference
as solely numerical error. Zoo values remain unchanged by these probes.

Reproduce both studies (JSON path is optional):

```sh
.venv/bin/python scripts/asymmetric_connection_probe.py /tmp/weak.json --steps 64 128 256 512
.venv/bin/python scripts/strong_asymmetric_connection_probe.py /tmp/strong.json --steps 64 128 256 512
```

The former Python section-event routine is retained only in
`tests/section_oracle.py` for differential testing. The Decimal scalar shooting
prototype remains an independent oracle in `wall_continue.py`; production
paired Newton and section-event decisions do not call it.


ABI-10 qualification: 77 focused Python tests passed (8 paired-solver,
22 inspector/section, 47 existing rheostat/connectivity/resolution checks).
The paired suite checks step refinement for weak and strong asymmetry,
independent section charts, 192/256-bit arithmetic and eta-step agreement,
duplicate-constraint rank refusal, invalid saddle boxes, precision floors,
iteration exhaustion and recovery. Four standalone C programs (section,
pair, scalar rheostat and private arithmetic) passed address/undefined-behavior
sanitizers. Inspector tests verify the blue candidate curves at the stored
parameter and separate invariant manifolds above and below it.
