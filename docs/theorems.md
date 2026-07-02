# spong theorems — statements and proof obligations

Named theorems the instrument relies on.  Each entry records: statement,
proof status, what the code assumes, and how the claim is enforced at
runtime regardless of status.  A theorem may be cited as EXACT in the
certificate ledger only when its status here is PROVED.

---

## Theorem 1 (Adjacency; no saddle-saddle connections in the finite plane)

**Statement.**  Let L(a,b) = C − 2aB(b) + a²A(b) be ψ-nice (A > 0 on ℝ) and
Morse.  Then, for the descent flow ż = −∇L: every unstable branch of every
saddle converges to the nearest MINIMUM on its side of the backbone (B-root
saddles between it and the target are not attractors and do not capture),
on the side selected by the b-component of the unstable eigenvector; every
stable branch of every saddle escapes to infinity.  NOTE (2026-07-02): the
B-root correction to Theorem 2 means "adjacent critical point" and
"adjacent minimum" differ — branch targeting must skip B-saddles.  In particular there are
no saddle-saddle connections in the finite plane, and the separatrix skeleton
is combinatorially rigid across the ψ-nice Morse family.

**Status: PROOF OBLIGATION** (asserted with sketch; consistent with all
computed evidence to date).

**Proof sketch.**
(i) A > 0 ⇒ L strictly convex on vertical slices ⇒ no local maxima ⇒ a
backward-ascending stable branch has no finite critical point available as a
backward limit other than a saddle — which is exactly what must be excluded.
(ii) Each stable manifold (both branches plus the saddle) is a properly
embedded line from infinity to infinity and separates the plane; unstable
branches of other saddles cannot cross it, by uniqueness of solutions.
(iii) Transverse contraction ẇw ≈ −2Aw² slaves each unstable branch to
w* = a*′P/(2A), while ḃ = −P moves b monotonically across the interval
between the saddle and the adjacent minimum, where u′ is single-signed.

**Gaps a written proof must close.**
1. Stable branches never re-cross the open backbone segment between two
   consecutive saddles (needed for (ii) to confine as claimed; ascent
   crossings of the backbone away from critical points are not excluded
   pointwise, since ẇ = −a*′u′ ≠ 0 there).
2. The slaving estimate in (iii) at moderate κ: bound the correction terms in
   P so that P retains the sign of u′ along the slaved tube for every ψ-nice
   Morse instance, not just asymptotically in κ.

**Runtime enforcement (independent of status).**  Every computed unstable
branch is checked against adjacency; every stable branch against box/rim
exit.  If the theorem holds, a violation is a numerical defect and is
reported as such.  If a counterexample exists, this check is precisely the
instrument that will find and report it.  Sound either way.

---

## Theorem 2 (Backbone reduction — proved)

**Statement (corrected 2026-07-02, simplified same day).**  All critical
points of L lie on the backbone a = B/A, at roots of B·N (u′ = B·N/A²;
N = α′β − 2β′α).  The identities

    H₁₂ = −2A·a*′,   H₂₂ = 2A·a*′² + u″,   det H = 2A·u″

hold at EVERY backbone point — no critical-point substitution required
(verified symbolically and numerically at arbitrary non-critical b).
Hence classification is uniformly one-dimensional: u″ > 0 ⇔ minimum,
u″ < 0 ⇔ saddle.  At a simple B-root, N(b₀) = −2B′A, so
u″ = B′N/A² = −2B′²/A < 0 automatically: **every simple B-root is a
saddle**, with det H = 2A·u″ = −4B′² — the SAME identity, not an
exception to it.  The SIGNS of u″ alternate along b (1D Morse on u);
consecutive L-saddles occur wherever a B-root falls.  L is Morse iff B·N
is squarefree with gcd(B, N) constant.

**Status: PROVED.**  History: the original min/saddle-alternation
phrasing was FALSE (caught by the Poincaré–Hopf index certificate); the
first correction wrongly claimed det H = 2A·u″ fails at B-roots (caught
in external review — Codex — by direct algebra: the identity is
universal).  Strip-localized exact winding confirms index −1 at every
B-root.

---

## Theorem 3 (Equatorial structure — proved under genericity)

**Statement.**  Under the §8 genericity conditions (effective degree d_eff,
μ_{2·d_eff} > 0), the Poincaré compactification of −∇L has equatorial
equilibria exactly at the axes directions and the four diagonals
b = ±√d_eff·a; the equatorial radial component is ≤ 0; separatrices approach
the diagonal equilibria asymptotic to those lines.

**Status: PROVED at leading order** (direct computation of the leading form;
the degenerate backbone poles' local model ḃ ~ −C_inf/b² is established in
mse_morse_info's compactification analysis).  The quasi-homogeneous blow-up
at the degenerate poles carries asymptotic-series-grade certificates only —
recorded as the atlas's weakest link in SPONG_FOUNDING §8.
