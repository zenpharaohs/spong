# Trajectory objects and independent scrutiny

Each production `Branch` carries a tuple of `TrajectoryPiece` objects.
`start` and `stop` are half-open **chord** indices: a piece owns
`Y[start:stop+1]`. Pieces partition the branch's chords, including local
launches, representation handoffs and drawing connectors. A refusal can
have a recorded prefix or a single vertex and no chords. A constructor
without explicit instrumentation produces an `unknown` piece, which the
checker refuses. No checker imports the tracer.

The initial inventory is at the **construction-phase** level:

| Construction | Retained representation | Independent scrutiny available |
|---|---|---|
| Manifold jet | Center, polynomial coefficients, parameters, centered samples, analytic tangents | Exact rational polynomial/derivative reconstruction, binding to displayed samples, sampled level-normal and oriented-flow measurements |
| Poincaré or centered grid, including extension | Center and centered samples | Ownership; smooth geometry check pending |
| Hadamard graph | Solved `(b,w)` samples and sample offset | Ownership; invariance/representation check pending |
| Potential prefix, candidate-level event, ascent | Accepted half-step collocation / actual ascent Hermite | Exact rational sampled field/tangent, binding and join checks |
| Centered arrival | Centered accepted half-step collocation | Exact rational sampled geometry |
| Native chart continuation, including rejected-shallow fallback | Accepted chart/normalized-rescue stages and vertex/trim events | Exact rational sampled geometry; retry-exhausted steps refused |
| Compatibility centered/linear/backbone launch | Returned physical polyline | Ownership; launch evidence check pending |
| Experimental minimum arrival | Matched two-mode exponential/Frobenius sum and analytic endpoint | Independent high-precision trajectory, modal-scale, exact center and limiting-direction checks |
| Handoff, capture connector | Explicit chord range | Independent chord geometry, launch and fate checks in `portrait_check` |

Native phases now also carry schema-1 accepted-step streams. See
[native_trace.md](native_trace.md) for the representation, rescue and trim
semantics, independent four-obligation assessment and calibration results.
The vocabulary remains open: unknown or unsupported construction is unresolved.

`trajectory_check.inventory(Y, pieces)` checks exact coverage, ordered ranges,
known kinds, finite coordinates and reports repeated chords. `complete`
means only that every chord has an owner; it is not a geometry verdict.

`trajectory_check.inspect_trajectory(model, branch, measure_jets=True)` adds
independent jet measurements. It uses only the exact model coefficients and
the proposed representation. It recomputes `K(t)` and `K'(t)` with Python
`Fraction` arithmetic at every retained parameter and each interval midpoint;
it does not trust the stored tangent or the constructor's residuals. Centered
offsets are added to the center in rational arithmetic, before any conversion
to global coordinates. A Horner rounding bound binds the proposed polynomial
to the displayed samples. That bound does not certify the coefficient
recurrence, truncation error, or convergence of the series.

The anchor at `t=0` is reported separately: its floating center approximates
an algebraic critical point, and its tiny exact gradient at that proxy has
no meaningful launch direction. This does not discharge the attachment
obligation. Stationary samples, zero tangents, opposed flow, and orthogonal
flow are explicit counts; a nonzero residual flag survives ratio underflow.
Measurements concern sampled smooth geometry, not the unsampled intervals.

The checker currently returns `invalid` for malformed coverage/evidence or
geometry mismatch, and otherwise `unresolved`. Every report has
`certified: false`. It deliberately chooses no universal angle tolerance.
The existing topology verdict is unchanged. `portrait.ledger` has a separate
`trajectory_evidence` entry with cheap coverage reports; exact jet measurements
are opt-in because their cost grows with degree and order.

Inspect actual emitted phases with `scripts/piece_census.py`; its `pieces`
and `piece_coverage` tallies supplement the old dispatcher `zones` tallies.
Run exact smooth-jet measurements alongside the chord reference with:

```sh
SPONG_ENGINE=native python scripts/credibility_probe.py --measure-jets minimal-quartet
```

Use `--measure-native` for native geometry alone, or `--max-sin-squared 1e-6`
for the experimental four-obligation assessment. The candidate policy is
explicit and is **not calibrated for production promotion**. The first two
zoo cases expose arrival-connector failures and unsupported representations;
the existing production verdict remains unchanged.

Minimum arrival work is available through the opt-in `--refine-arrivals`
probe. See [arrival_evidence.md](arrival_evidence.md) for the constructor,
independent scrutiny, measured improvements and explicit limitations.
