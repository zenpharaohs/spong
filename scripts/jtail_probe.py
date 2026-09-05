#!/usr/bin/env python3
"""Measure the J-tail handoff: how the drawn level set J = J0 deviates from
the true (continued) ascent flow as a function of the half-plane margin
|a|/max(|a*|, 2|B'/A'|) at the switchover.

For each stable branch: continue the ascent far beyond the certified
terminal (reference), then for each candidate switchover point along
that reference (sampled by margin) root-find the level set J = J0 at the
reference's later a-values and record the b mismatch relative to r.
The output is the curve err(margin).  RESULT (2026-09-05): err ~
c*margin^-p with c and p per case (c 1e-2 ... 1e-14, p 1 ... 2), so no
fixed margin threshold serves the family; and on pinned branches the
margin is identically 1 (the branch rides the b-nullcline a A' = 2 B',
on which every critical point with a != 0 lies).  That is what replaced
`need` by the measured, octave-attested licensing in serve._j_tails.
Also exports build()/j_invariant()/j_value() for bar_probe.py.

    python scripts/jtail_probe.py [case ...]      (custom:1+x+x^2 allowed)
"""
import math, os, sys, time
os.environ.setdefault("SPONG_ENGINE", "native")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from spong import atlas, model, portrait, sturm, zoo, charts

CASES = ["dead-neuron-far-saddle-d3", "minimal-quartet", "custom:1+x+x^2",
         "linear-target-d17-thrash"]
if len(sys.argv) > 1:
    CASES = sys.argv[1:]


def build(name):
    if name.startswith("custom:"):
        f = g = [1.0, 1.0, 1.0]
        spec = "uniform01"
    else:
        z = zoo.get(name)
        f = [float(x) for x in z.f]; g = [float(x) for x in z.g]
        spec = z.moment_dist
    n = 2*max(len(f), len(g)) - 1
    mu = (model.moments_uniform01(n) if spec == "uniform01"
          else model.moments_normal01(n))
    return model.build(f, g, mu)


def j_invariant(m, d):
    A = np.asarray(m._fa, dtype=float)
    Ap = np.polyder(A[::-1]); App = np.polyder(Ap)
    q, _ = np.polydiv(A[::-1], Ap)
    c1 = float(q[-1]) if len(q) else 0.0
    rhos = np.roots(Ap)
    res = np.array([np.polyval(A[::-1], z)/np.polyval(App, z) for z in rhos],
                   dtype=complex)
    real = sorted(float(z.real) for z in rhos if abs(z.imag) < 1e-9)
    return c1, rhos, res, real


def j_value(a, b, d, c1, rhos, res):
    v = 0.5*a*a - b*b/(2.0*d) - 2.0*c1*b
    v -= 2.0*float(np.real(np.sum(res*np.log(complex(b) - rhos))))
    return v


def newton_b(aa, b, J0, d, c1, rhos, res, A, Ap, lo, hi):
    for _ in range(60):
        f = j_value(aa, b, d, c1, rhos, res) - J0
        Av = float(np.polyval(A[::-1], b)); Apv = float(np.polyval(Ap, b))
        if Apv == 0.0 or not math.isfinite(f):
            return None
        bn = b - f/(-2.0*Av/Apv)
        if not (lo < bn < hi):
            bn = b - 0.5*(b - (lo if bn <= lo else hi))
        if abs(bn - b) <= 1e-13*max(1.0, abs(bn)):
            return bn
        b = bn
    return b


def run(name):
    m = build(name)
    d = float(atlas.effective_degree(m))
    bmax = atlas.legal_max_b(m)
    amax = bmax/max(1.0, math.sqrt(max(1, d)))
    tv = (-amax, amax, -bmax, bmax)
    t0 = time.perf_counter()
    e = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    dv = atlas.compute_box(m, e, view=tv)
    p = portrait.compute(m, view=tv, geometry_level=0, _enumeration=e,
                         _display_view=dv, _genericity=atlas.genericity(m),
                         _skip_audit=True)
    print(f"\n### {name}  d={d:g}  legal box b±{bmax:.3g}  "
          f"portrait {time.perf_counter()-t0:.1f}s  box={tuple(round(x,3) for x in p.box)}")
    c1, rhos, res, real = j_invariant(m, d)
    print("   A' real roots:", [round(r, 6) for r in real])
    A = np.asarray(m._fa, dtype=float); Ap = np.polyder(A[::-1])
    B = np.asarray(m._fb, dtype=float); Bp = np.polyder(B[::-1])
    critical = np.array([[float(q.a), float(q.b)] for q in e.points])

    def margin_at(a, b):
        Av = float(np.polyval(A[::-1], b)); Apv = float(np.polyval(Ap, b))
        Bv = float(np.polyval(B[::-1], b)); Bpv = float(np.polyval(Bp, b))
        if not Av > 0: return 0.0
        sc = max(abs(Bv/Av), 2*abs(Bpv/Apv) if Apv != 0 else math.inf)
        return abs(a)/sc if sc > 0 else math.inf

    for i, br in enumerate(p.branches):
        if br.kind != "stable" or len(br.Y) < 2:
            continue
        cur = [(float(q[0]), float(q[1])) for q in br.Y]
        # reference: continue far
        ascent = br.diag.get("potential_rate_ascent") or {}
        ds = (float(ascent["geometric_ds"])/4.0 if "geometric_ds" in ascent
              else math.hypot(p.box[1]-p.box[0], p.box[3]-p.box[2])/30000.0)
        r_stop = 200.0*max(abs(x) for x in p.box)
        t0 = time.perf_counter()
        try:
            grown, term = charts._potential_rate_box_exit(
                m, cur[-1], (-r_stop, r_stop, -r_stop, r_stop), ds, {},
                max_steps=400000, critical=critical)
            ref = cur + [(float(q[0]), float(q[1])) for q in grown[1:]]
        except Exception as ex:
            print(f" br{i} stable b*={br.diag.get('saddle_b')}: ref failed {ex}")
            continue
        tgrow = time.perf_counter() - t0
        marg = np.array([margin_at(a, b) for a, b in ref])
        a_ref = np.array([q[0] for q in ref]); b_ref = np.array([q[1] for q in ref])
        r_ref = np.hypot(a_ref, b_ref)
        a_end, b_end = ref[-1]
        print(f" br{i} b*={float(br.diag.get('saddle_b', float('nan'))):.5f} "
              f"term={br.term} n={len(cur)} ref +{len(ref)-len(cur)} ({tgrow:.1f}s, {term}) "
              f"end a={a_end:.4g} b={b_end:.6g} margin_end={marg[-1]:.3g} "
              f"margin_term={marg[len(cur)-1]:.3g}")
        # candidate switchovers: first index where margin >= need, for a ladder
        seen = set()
        rows = []
        for need in (1, 2, 5, 10, 20, 50, 100, 300, 1000):
            idx = next((k for k in range(len(ref)) if marg[k] >= need), None)
            if idx is None or idx in seen or idx >= len(ref) - 5:
                continue
            seen.add(idx)
            a0, b0 = ref[idx]
            J0 = j_value(a0, b0, d, c1, rhos, res)
            lo = max([z for z in real if z < b0], default=-math.inf)
            hi = min([z for z in real if z > b0], default=math.inf)
            # follow the level set at the reference's later a-values
            worst = 0.0; worst_at = None; b = b0; nbad = 0
            samp = np.unique(np.linspace(idx+1, len(ref)-1, 400).astype(int))
            for k in samp:
                aa = a_ref[k]
                bn = newton_b(aa, b, J0, d, c1, rhos, res, A, Ap, lo, hi)
                if bn is None or not math.isfinite(bn):
                    nbad += 1; continue
                b = bn
                err = abs(b - b_ref[k])/r_ref[k]
                if err > worst:
                    worst, worst_at = err, (aa, b_ref[k], b)
            rows.append((need, idx, marg[idx], a0, b0, J0, worst, worst_at, nbad, lo, hi))
        for need, idx, mg, a0, b0, J0, worst, wat, nbad, lo, hi in rows:
            wa = "" if wat is None else f" at a={wat[0]:.4g} b_ref={wat[1]:.6g} b_J={wat[2]:.6g}"
            print(f"    need>={need:>5}: idx {idx:>6} margin {mg:9.3g} a0={a0:9.4g} b0={b0:10.6g} "
                  f"J0={J0:11.5g} trap({lo:.4g},{hi:.4g})  max|db|/r = {worst:.3e}{wa}"
                  f"{'  bad='+str(nbad) if nbad else ''}")


if __name__ == "__main__":
    for nm in CASES:
        run(nm)
