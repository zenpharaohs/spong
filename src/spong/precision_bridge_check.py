"""Independent original-model checks of exported decimal bridge cubics.

No constructor/integrator imports. Arithmetic has a conservative cancellation
screen; measurements remain sampled numerical evidence, not interval proofs.
"""
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q
from math import comb, ceil
import numpy as np


def _number(q):
    q=Q(q);return D(q.numerator)/D(q.denominator)


def _shift(c,b):
    return tuple(sum((Q(c[j])*comb(j,i)*b**(j-i) for j in range(i,len(c))),Q(0)) for i in range(len(c)))


def _gradient(m,origin):
    a,b=map(Q,origin)
    A,B=_shift(m.alpha,b),_shift(m.beta,b)
    n=max(len(A),len(B));A+= (Q(0),)*(n-len(A));B+=(Q(0),)*(n-len(B))
    Ap=tuple(i*A[i] for i in range(1,n));Bp=tuple(i*B[i] for i in range(1,n))
    return ((tuple(2*(a*x-y) for x,y in zip(A,B)),tuple(2*x for x in A)),
            (tuple(a*a*x-2*a*y for x,y in zip(Ap,Bp)),tuple(2*a*x-2*y for x,y in zip(Ap,Bp)),Ap))


def _horner(c,t):
    v=D(0)
    for x in reversed(c):v=v*t+x
    return v


def _value(p,t):
    return tuple(_horner(c,t) for c in p),tuple(_horner(tuple(i*c[i] for i in range(1,4)),t) for c in p)


def _decode(e):
    if e.get('representation')!='precision_bridge_cubics' or e.get('schema')!=1:raise ValueError('bridge_representation')
    origin=tuple(Q(x) for x in e['origin']);polys=e['polynomials'];bits=e['working_bits']
    if len(origin)!=2 or type(bits) is not int or not 128<=bits<=4096 or not 1<=len(polys)<=2048:raise ValueError('bridge_shape')
    for p in polys:
        if len(p)!=2 or any(len(c)!=4 for c in p) or any(not D(x).is_finite() for c in p for x in c):raise ValueError('bridge_polynomial')
    return origin,polys,bits


def endpoint(e,index):
    """Exact endpoint and tangent of the retained decimal polynomial."""
    origin,polys,_=_decode(e)
    if not 1<=index<=len(polys):raise ValueError('bridge_index')
    p=tuple(tuple(Q(x) for x in c) for c in polys[index-1])
    return tuple(origin[d]+sum(p[d]) for d in range(2)),tuple(sum(i*p[d][i] for i in range(1,4)) for d in range(2))


def measure_precision_bridge(m,points,piece,*,precision=90,sample_budget=10000):
    try:
        e=piece.evidence;origin,polys,bits=_decode(e)
        offset=e.get('branch_offset',0);start,stop=piece.start-offset,piece.stop-offset
        if start!=0 or not 0<stop<=len(polys):raise ValueError('bridge_range')
        Y=np.asarray(points,dtype=float)
        if Y.ndim!=2 or Y.shape[1]!=2 or not np.all(np.isfinite(Y)) or piece.stop>=len(Y):raise ValueError('bridge_points')
        if stop*9>sample_budget:return {'status':'unresolved','reason':'measurement_budget','required_samples':stop*9}
        if type(precision) is not int or not 50<=precision<=2000:raise ValueError('bridge_checker_precision')
        with localcontext() as ctx:
            ctx.prec=max(precision,ceil(bits*0.3010299956639812)+20)
            grad=tuple(tuple(tuple(_number(x) for x in c) for c in g) for g in _gradient(m,origin))
            center=tuple(_number(x) for x in origin)
            eps=D(2)**-52;roundoff=D(10)**(-ctx.prec+8)
            binding=joins=worst=error=D(0);opposed=zero=stationary=0
            previous=(D(0),D(0))
            for j,p0 in enumerate(polys[:stop]):
                p=tuple(tuple(D(x) for x in c) for c in p0)
                beginning,_=_value(p,D(0))
                joins=max(joins,*(abs(a-b)/(eps*(1+abs(center[d]+a))) for d,(a,b) in enumerate(zip(beginning,previous))))
                for t in (D(k)/8 for k in range(9)):
                    x,v=_value(p,t)
                    g=tuple(_horner(tuple(_horner(row,x[1]) for row in component),x[0]) for component in grad)
                    magnitude=max(_horner(tuple(_horner(tuple(abs(c) for c in row),abs(x[1])) for row in component),abs(x[0])) for component in grad)
                    ng=sum(a*a for a in g);nv=sum(a*a for a in v)
                    if not ng:stationary+=1;continue
                    if not nv:zero+=1;continue
                    # Input rounding plus accumulated Horner operations. Refuse
                    # cancellation that consumes the large guard reserve.
                    indicator=roundoff*magnitude/max(map(abs,g))
                    tangent_mag=max(sum(abs(i*c[i])*(abs(t)**(i-1) if i>1 else D(1)) for i in range(1,4)) for c in p)
                    indicator=max(indicator,roundoff*tangent_mag/max(map(abs,v)))
                    error=max(error,indicator)
                    if indicator>D('1e-20'):return {'status':'unresolved','reason':'bridge_checker_cancellation','precision':ctx.prec}
                    opposed+=sum(a*b for a,b in zip(g,v))>=0
                    worst=max(worst,(g[0]*v[1]-g[1]*v[0])**2/(ng*nv))
                    if t==0 and j==0 or t==1:
                        vertex=Y[offset+j+(t==1)]
                        binding=max(binding,*(abs(center[d]+x[d]-D.from_float(float(vertex[d])))/(eps*(1+abs(center[d]+x[d]))) for d in range(2)))
                previous,_=_value(p,D(1))
            return {'status':'measured','steps':stop,'samples':stop*9,'precision':ctx.prec,'max_sin_squared':float(worst),
                    'binding_eps':float(binding),'join_eps':float(joins),'opposed_flow':opposed,'zero_tangents':zero,
                    'stationary_samples':stationary,'unvalidated_steps':0,'max_roundoff_indicator':float(error),'scope':'sampled_precision_bridge'}
    except (ValueError,KeyError,TypeError,IndexError,ArithmeticError) as exc:return {'status':'invalid','reason':str(exc)}


def measure_precision_join(points,prefix,arrival):
    try:
        if prefix.stop!=arrival.start:raise ValueError('bridge_partition')
        e=prefix.evidence;x,dx=endpoint(e,prefix.stop-e.get('branch_offset',0))
        ae=arrival.evidence
        if ae.get('representation')!='poincare_frobenius_arrival':raise ValueError('bridge_tail_representation')
        c=tuple(Q(v) for v in ae['center']);terms=[tuple(Q(v) for v in row) for row in ae['terms']]
        y=tuple(c[d]+sum(t[d+1] for t in terms) for d in range(2));dy=tuple(-sum(t[0]*t[d+1] for t in terms) for d in range(2))
        nx,ny=sum(v*v for v in dx),sum(v*v for v in dy)
        if not nx or not ny:raise ValueError('bridge_zero_join_tangent')
        return {'status':'measured','join_eps':float(max(abs(a-b)/(Q(1,2**52)*(1+max(abs(a),abs(b)))) for a,b in zip(x,y))),
                'sin_squared':float((dx[0]*dy[1]-dx[1]*dy[0])**2/(nx*ny)),'same_orientation':sum(a*b for a,b in zip(dx,dy))>0}
    except (ValueError,KeyError,TypeError,IndexError,ArithmeticError) as exc:return {'status':'invalid','reason':str(exc)}
