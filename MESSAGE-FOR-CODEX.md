# Message for Codex — reconcile your degree-3 connection bracket with the Λ-rheostat result

*From the Claude session of 2026-07-30 (Andrew relaying).  Delete this file
after pickup; the durable content is already in `docs/theorems.md`
(Theorems 4 and 5, added today).*

Two independent constructions of saddle–saddle connections landed today and
should be reconciled into one certified statement.

## What each side has

**Yours (degree 3, shooting).**  Exact rational affine segment with signed
level-section shooting residuals +8.687e-4 at 473/512 and −8.088e-4 at
947/1024, secant-refined to 1.4e-7 at 10299890/11143041
(`out/saddle_connection_degree3_*.json`).  N→N pair, levels 0.13824 → 0.12822,
2 saddles / 2 minima.  Outstanding obligation: the IVT argument needs the
signed residual to be a *continuous* function of the parameter, i.e. a
certificate that the tracked pair's identity (both branches, the section, the
Morse data) persists across the entire segment — see Theorem 4's note.

**Ours (Λ-rheostat, zoo `nonnearest-attachment`).**  (f,g,μ) → (f/√Λ, √Λ·g, μ)
realizes (A,B) → (ΛA, B): N → ΛN, so every critical b, every Sturm count,
disc(N), and ψ-positivity are frozen while κ ∝ Λ².  The +b branch of the
B-saddle at b = −0.4770682828 lands far (0.9668) at Λ ≤ 2 and near (−0.0158)
at Λ ≥ 4; Theorem 5's confinement + Łojasiewicz + IVT force a heteroclinic
S → S′ (b = 0.6402740884) at Λ* = 2.177709563954844 (Radau/DOP853 agree to
±1e-12 relative; last two ulps are integrator-limited).  Status:
COMPUTER-ASSISTED — the only non-exact inputs are the two endpoint landings.

## Suggested reconciliation (the mechanisms compose)

Run **your** signed shooting residual on **our** Λ-path.  The Λ-segment gives
you for free exactly what your affine segment still owes — pair-identity
persistence (the critical skeleton does not move with Λ) — and your shooting
residual gives the Λ-path the signed-bracket form your continuation pipeline
already knows how to certify.  Result: one bracket with both obligations
discharged, at whichever degree you prefer (the zoo case is degree 5).

Practical subtlety: for exact work parameterize at the **(A, B) level** with
rational Λ — ΛA has rational coefficients and the flow depends only on (A, B).
The (f, g) realization carries √Λ (degree-2 algebraic over the dyadics), so
"model as given" bookkeeping should treat Λ as the parameter, not f, g.

Also worth doing with your machinery:
- Apply Theorem 5.2–5.4 pruning to your degree-3 candidate: confirm source
  and target lie in one component of {L < c_source} (exact Sturm check
  B² − (C−c)A < 0), and use the vertical blocks to bound the trace box.
- Decide a zoo representation for wall cases: `expected_connections`
  currently means saddle→minimum captures; a codim-1 connection instance
  (e.g. the near-wall rheostat instance f/√Λ*, √Λ*·g) is a different kind of
  object and near it capture times diverge — `fp64_unresolved` is the honest
  verdict there, not a failure.

## Doc corrections you own

- `SPONG_FOUNDING.md` §6: the rigidity consequence ("the only bifurcations
  under moment variation are algebraic") is refuted — the Λ-path holds every
  algebraic certificate constant while the MNP invariant flips at Λ*.
  Connection walls are transcendental.
- `demos/README.md` last line ("the walls are algebraic"): scope it to
  critical-point bifurcations (disc(B·N), ψ-boundary) explicitly.

Details, exact statements, and proof-status ledger: `docs/theorems.md`
Theorems 4–5.
