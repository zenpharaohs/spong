#!/usr/bin/env python3
"""Stable-tail licensing at the LEVEL-BAR terminal, measured.

For each stable branch: emulate the bar (truncate the trace at the first
vertex with L > U*(1+1e-9)+1e-12, U* the highest saddle loss -- the bar
index is a property of the model, so this reproduces the level-bar
tree's terminals on a main-tree trace), then report

  * reference refinement: the ascent continued from the bar terminal at
    ds and ds/4, compared at matched |a| (the tail references are only
    as good as this number);
  * whether either sector licenses AT the bar terminal (it never does on
    the four cases measured -- the bar fires at launch);
  * whether it licenses on the full trace, and how much growth from the
    bar terminal (radius ratio) it takes.

Licensing here mirrors serve._j_tails: pinned = nullcline lag <= tol over
the last radius octave and falling; diagonal = remaining J drift D_k <= tol
with r doubling after k.  This is the probe that fixed the octave rule.

    python scripts/bar_probe.py <case> [tol]
"""
import os, sys, math
os.environ.setdefault("SPONG_ENGINE", "native")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from fractions import Fraction
from jtail_probe import build, j_invariant, j_value
from spong import atlas, portrait, sturm, charts, _poly as P
name=sys.argv[1]; tol=float(sys.argv[2]) if len(sys.argv)>2 else 1e-4
m=build(name); d=float(atlas.effective_degree(m))
bmax=atlas.legal_max_b(m); amax=bmax/math.sqrt(d); tv=(-amax,amax,-bmax,bmax)
e=sturm.materialize_stubs(m, sturm.enumerate_critical_points(m)); dv=atlas.compute_box(m,e,view=tv)
p=portrait.compute(m,view=tv,geometry_level=0,_enumeration=e,_display_view=dv,_genericity=atlas.genericity(m),_skip_audit=True)
c1,rhos,res,real=j_invariant(m,d); A=np.asarray(m._fa,float); Ap=np.polyder(A[::-1]); App=np.polyder(Ap)
B=np.asarray(m._fb,float); Bp=np.polyder(B[::-1]); Bpp=np.polyder(Bp)
critical=np.array([[float(q.a),float(q.b)] for q in e.points])
Ustar=max(float(m.L(q.a,q.b)) for q in e.points if q.kind=="saddle")
stop=Ustar*(1+1e-9)+1e-12
traps=[(float(iv.mid), sturm.interval_sign(P.deriv(P.deriv(m.alpha)),iv)) for iv in (sturm.refine(P.deriv(m.alpha),iv,Fraction(1,2**40)) for iv in sturm.isolate_roots(P.deriv(m.alpha)))]
print(f"### {name} U*={Ustar:.6g} traps={[(round(t,5),'attr' if s<0 else 'rep') for t,s in traps]}")
def nullcline_b(a,b):
    for _ in range(50):
        f=a*np.polyval(Ap,b)-2*np.polyval(Bp,b); fp=a*np.polyval(App,b)-2*np.polyval(Bpp,b)
        if fp==0: return b
        bn=b-f/fp
        if abs(bn-b)<1e-14*max(1,abs(bn)): return bn
        b=bn
    return b
def grow(start, ds, steps):
    r_stop=200*max(abs(x) for x in p.box)
    g,t=charts._potential_rate_box_exit(m,start,(-r_stop,r_stop,-r_stop,r_stop),ds,{},max_steps=steps,critical=critical)
    return [(float(q[0]),float(q[1])) for q in g], t
def license(pts):
    """returns (kind, index, measure) or (None, n, nan)"""
    a=np.array([q[0] for q in pts]); b=np.array([q[1] for q in pts]); r=np.hypot(a,b); n=len(pts)
    # pinned: nullcline lag over the last octave in r, all <= tol and falling
    k0=next((k for k in range(n) if r[k]>=r[-1]/2),n-1)
    if k0<n-1 and abs(a[-1])>abs(a[k0]):
        lag=np.array([abs(b[k]-nullcline_b(a[k],b[k]))/r[k] for k in range(k0,n)])
        if np.all(np.isfinite(lag)) and lag.max()<=tol and lag[-1]<=lag[0]:
            return ("nullcline", k0, lag.max())
    # diagonal: J drift after k in relative-b units, with r doubling after k
    J=np.array([j_value(x,y,d,c1,rhos,res) for x,y in pts])
    if np.all(np.isfinite(J)):
        w=np.abs(np.polyval(Ap,b)/(2*np.polyval(A[::-1],b)))/r
        for k in range(0,n-1,max(1,n//3000)):
            if r[-1]<2*r[k]: break
            D=float(np.max(np.abs(J[k+1:]-J[k])*w[k+1:]))
            if D<=tol: return ("J",k,D)
    return (None, n, float("nan"))
for i,br in enumerate(p.branches):
    if br.kind!="stable": continue
    cur=[(float(q[0]),float(q[1])) for q in br.Y]
    L=[float(m.L(x,y)) for x,y in cur]
    kb=next((k for k in range(len(cur)) if L[k]>stop), len(cur)-1)
    bar=cur[:kb+1]
    asc=br.diag.get("potential_rate_ascent") or {}; ds=float(asc["geometric_ds"])/4
    # reference refinement: ds vs ds/4 from the bar terminal
    g1,t1=grow(bar[-1],ds,200000); g2,t2=grow(bar[-1],ds/4,800000)
    a1=np.array([q[0] for q in g1]); b1=np.array([q[1] for q in g1]); a2=np.array([q[0] for q in g2]); b2=np.array([q[1] for q in g2])
    # compare at matched |a| where both monotone in a (use interpolation on the shorter reach)
    amax_c=min(abs(a1[-1]),abs(a2[-1])); sel=(np.abs(a1)<=amax_c)&(np.abs(a1)>abs(a1[0]))
    if np.all(np.diff(np.abs(a2))>=0) and sel.sum()>10:
        bi=np.interp(np.abs(a1[sel]),np.abs(a2),b2); ref_err=float(np.max(np.abs(bi-b1[sel])/np.hypot(a1[sel],b1[sel])))
    else: ref_err=float("nan")
    lic_bar=license(bar); lic_full=license(cur)
    # growth needed from the bar terminal
    need_grow=None
    pts=list(bar)
    for chunk in range(12):
        lic=license(pts)
        if lic[0]: need_grow=(chunk, lic, math.hypot(*pts[-1])/math.hypot(*bar[-1])); break
        g,t=grow(pts[-1],ds,2000); pts+=g[1:]
        if len(g)<3: break
    print(f"br{i} b*={float(br.diag['saddle_b']):.4f} n={len(cur)} bar@{kb} term={br.term} | ref ds vs ds/4: {ref_err:.2e} ({t1}/{t2}) | at bar: {lic_bar[0]} | full trace: {lic_full[0]} k={lic_full[1]} meas={lic_full[2]:.2e} | grow: {need_grow}")
