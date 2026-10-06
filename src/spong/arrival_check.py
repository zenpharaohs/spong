"""Independent high-precision scrutiny of proposed Frobenius arrivals.

No local chart, series constructor, or integrator is imported. The retained
exponential polynomial defines the tangent; the original rational model
supplies the field. These are sampled residual checks, not interval proofs.
"""
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q

import numpy as np


def _number(x):
    q = Q(x)
    return D(q.numerator)/D(q.denominator)


def _horner(c,x):
    y = D(0)
    for a in reversed(c):
        y = y*x+a
    return y


def _derivative(c):
    return tuple(i*x for i,x in enumerate(c) if i)


def _decode(e):
    if e.get('representation')!='poincare_frobenius_arrival' or e.get('schema')!=1:
        raise ValueError('unknown_arrival_representation')
    if e.get('terminal')!='analytic_limit':
        raise ValueError('missing_analytic_limit')
    center = tuple(_number(x) for x in e['center'])
    if len(center)!=2:
        raise ValueError('invalid_arrival_center')
    terms = [tuple(D(x) for x in row) for row in e['terms']]
    if (not terms or len(terms)>1000 or any(len(t)!=3 for t in terms)
            or any(not x.is_finite() for t in terms for x in t)
            or any(t[0]<=0 for t in terms)):
        raise ValueError('invalid_arrival_terms')
    return center,terms


def _curve(center,terms,s):
    z = [D(0),D(0)]; tangent = [D(0),D(0)]
    for rate,a,b in terms:
        exponential = (-rate*s).exp()
        for i,c in enumerate((a,b)):
            z[i] += c*exponential
            tangent[i] -= rate*c*exponential
    return tuple(c+x for c,x in zip(center,z)),tuple(tangent)


def _angle(v,w):
    nv = sum(x*x for x in v); nw = sum(x*x for x in w)
    return None if not nv or not nw else (v[0]*w[1]-v[1]*w[0])**2/(nv*nw)


def _limit(m,center,terms):
    # Aggregate coincident exponents before finding the leading nonzero mode.
    grouped = {}
    for rate,a,b in terms:
        v = grouped.setdefault(rate,[D(0),D(0)])
        v[0] += a; v[1] += b
    modes = [(r,v) for r,v in sorted(grouped.items()) if any(v)]
    if not modes:
        raise ValueError('constant_arrival')
    _,v = modes[0]
    a,b = center
    A,B = tuple(map(_number,m.alpha)),tuple(map(_number,m.beta))
    Ap,Bp = _derivative(A),_derivative(B)
    h00 = 2*_horner(A,b)
    h01 = 2*a*_horner(Ap,b)-2*_horner(Bp,b)
    h11 = a*a*_horner(_derivative(Ap),b)-2*a*_horner(_derivative(Bp),b)
    Hv = (h00*v[0]+h01*v[1],h01*v[0]+h11*v[1])
    angle = _angle(v,Hv)
    return {'status':'measured','sin_squared':None if angle is None else float(angle),
            'inward':sum(x*y for x,y in zip(v,Hv))>0,
            'positive_hessian':h00>0 and h00*h11-h01*h01>0}


def check_target(m,minimum,evidence,*,precision=90):
    """Independently bind the claimed center to its exact isolated minimum."""
    from . import sturm, merge_tree, _poly as P
    try:
        with localcontext() as ctx:
            ctx.prec = precision
            center,terms = _decode(evidence)
            a,b = map(Q,evidence['center'])
            root = merge_tree._root_poly(m,minimum)
            interval = sturm.refine(root,minimum.interval,Q(1,2**200))
            if (abs(b-interval.mid)>Q(1,2**150)*(1+abs(interval.mid))
                    or a != P.eval_at(m.beta,b)/P.eval_at(m.alpha,b)):
                raise ValueError('arrival_center_not_bound_to_minimum')
            result = _limit(m,center,terms)
            return {**result,'center_bound':True}
    except (ValueError,KeyError,TypeError,ArithmeticError):
        return {'status':'invalid','reason':'invalid_arrival_attachment'}


def measure_arrival(m,points,evidence,direction=-1,*,precision=90,sample_budget=10000):
    """Check all retained parameters, midparameters, both modal scales and limit.

    Original exact coefficients are evaluated at high precision; Decimal exp
    is necessary because the Frobenius exponents need not be rational powers
    with rational values. Coefficients/tangents supplied by the constructor
    are never interpreted as a precomputed field residual.
    """
    try:
        if direction!=-1:
            raise ValueError('arrival_requires_descent')
        with localcontext() as ctx:
            ctx.prec = precision
            center,terms = _decode(evidence)
            parameters = evidence['parameters']
            if len(parameters)!=len(points) or parameters[-1] is not None:
                raise ValueError('arrival_parameter_count')
            ts = [D(x) for x in parameters[:-1]]
            if (not ts or ts[0]!=0 or any(not t.is_finite() for t in ts)
                    or any(a>=b for a,b in zip(ts,ts[1:]))):
                raise ValueError('invalid_arrival_parameters')
            Y = np.asarray(points,dtype=float)
            if Y.shape!=(len(parameters),2) or not np.all(np.isfinite(Y)):
                raise ValueError('invalid_arrival_points')
            samples = set(ts)|{(a+b)/2 for a,b in zip(ts,ts[1:])}
            # Independent scales: do not trust the producer to sample a fast
            # transient, or to approach far enough toward the analytic limit.
            rates = {rate for rate,_,_ in terms}
            for rate in (min(rates),max(rates)):
                samples.update(D(k)/(4*rate) for k in range(129))
            if len(samples)>sample_budget:
                return {'status':'unresolved','reason':'measurement_budget',
                        'required_samples':len(samples),'sample_budget':sample_budget}
            A,B = tuple(map(_number,m.alpha)),tuple(map(_number,m.beta))
            Ap,Bp = _derivative(A),_derivative(B)
            binding = D(0); worst = D(0); opposed=zero=stationary=0
            values = {}
            for s in sorted(samples):
                point,tangent = _curve(center,terms,s)
                values[s] = point
                a,b = point
                g = (2*(a*_horner(A,b)-_horner(B,b)),
                     a*a*_horner(Ap,b)-2*a*_horner(Bp,b))
                if not any(tangent): zero+=1; continue
                if not any(g): stationary+=1; continue
                opposed += sum(x*y for x,y in zip(tangent,g))>=0
                worst = max(worst,_angle(tangent,g))
            for point,t in zip(Y,parameters):
                exact = center if t is None else values[D(t)]
                binding = max(binding,*(abs(x-D.from_float(float(y)))/
                                        (D(2)**-52*(1+abs(x))) for x,y in zip(exact,point)))
            limit = _limit(m,center,terms)
            if limit['sin_squared'] is not None:
                worst = max(worst,D.from_float(limit['sin_squared']))
            return {'status':'measured','samples':len(samples),'max_sin_squared':float(worst),
                    'cross_nonzero':bool(worst),'opposed_flow':opposed,
                    'zero_tangents':zero,'stationary_samples':stationary,
                    'binding_eps':float(binding),'limit':limit,'precision':precision,
                    'scope':'sampled_frobenius_arrival_and_asymptotic_direction'}
    except (ValueError,KeyError,TypeError,ArithmeticError,IndexError) as exc:
        return {'status':'invalid','reason':str(exc)}


def measure_handoff(m, points, prefix, arrival):
    """Bind the retained native endpoint to the analytic start, over Q.

    This checks the two actual representations, not just their shared display
    vertex. Each piece's field alignment must be checked separately.
    """
    from .native_trace import decode, polynomial, physical
    try:
        if (prefix.stop != arrival.start or prefix.stop <= prefix.start
                or arrival.evidence.get('representation') != 'poincare_frobenius_arrival'):
            raise ValueError('handoff_partition')
        e = prefix.evidence
        dense, vertices, _ = decode(e['trace'])
        source, t, vertex = vertices[prefix.stop-e['branch_offset']]
        if source < 0 or not np.array_equal(vertex, points[prefix.stop]):
            raise ValueError('handoff_source')
        x, dx = physical(m, dense[source], polynomial(dense[source]), t)
        ae = arrival.evidence
        center = tuple(Q(c) for c in ae['center'])
        terms = [tuple(Q(c) for c in row) for row in ae['terms']]
        y = tuple(center[d]+sum(row[d+1] for row in terms) for d in range(2))
        dy = tuple(-sum(row[0]*row[d+1] for row in terms) for d in range(2))
        nx, ny = sum(v*v for v in dx), sum(v*v for v in dy)
        if not nx or not ny:
            raise ValueError('handoff_zero_tangent')
        gap = max(abs(a-b)/(Q(1,2**52)*(1+max(abs(a),abs(b)))) for a,b in zip(x,y))
        angle = (dx[0]*dy[1]-dx[1]*dy[0])**2/(nx*ny)
        return {'status': 'measured', 'join_eps': float(gap),
                'sin_squared': float(angle),
                'same_orientation': sum(a*b for a,b in zip(dx,dy)) > 0,
                'scope': 'native_analytic_join'}
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
        return {'status': 'invalid', 'reason': str(exc)}
