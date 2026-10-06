"""Experimental minimum entry charts, separate from production certification.

For moderate spectral ratios a truncated Poincare parameterization K
conjugates the centered gradient field to diag(lambda_s, lambda_f). For
stiff nodes it retains the slow entry jet and the leading homogeneous fast
Frobenius term, leaving omitted couplings to independent residual scrutiny.
The incoming point determines BOTH modal amplitudes. Along descent,
K(u0 exp(-s), v0 exp(-rho s)) supplies a proposed analytic arrival.
No integration to a capture radius or straight terminal connector is used.
Resonance is a refusal, not a pole regularization. This high-precision
proposal is not a validated enclosure or an acceptance policy.
"""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q

import numpy as np

from .trajectory import TrajectoryPiece


def _decimal(x):
    if isinstance(x, Q):
        return D(x.numerator)/D(x.denominator)
    return D(x)


def _mul(a, b, order):
    out = {}
    for (i,j), x in a.items():
        for (k,l), y in b.items():
            key = i+k,j+l
            if sum(key) <= order:
                out[key] = out.get(key,D(0))+x*y
    return out


def _compose(field, K, order):
    pa = [{(0,0):D(1)}]
    pb = [{(0,0):D(1)}]
    for _ in range(max(len(c) for c in field)-1):
        pa.append(_mul(pa[-1],K[0],order))
    for _ in range(max(len(r) for c in field for r in c)-1):
        pb.append(_mul(pb[-1],K[1],order))
    out = [{},{}]
    for d,c in enumerate(field):
        for i,row in enumerate(c):
            for j,value in enumerate(row):
                if not value or i+j < 2:
                    continue
                for key,x in _mul(pa[i],pb[j],order).items():
                    out[d][key] = out[d].get(key,D(0))+value*x
    return out


def parameterization(local, *, order=8, precision=80, divisor_floor=D('1e-12')):
    """Polynomial K and rates; guarded homological equations over Decimal.

    Unlike a saddle's one-dimensional launch, a minimum CAN be resonant.
    Every homological divisor through the requested order is screened.
    Higher-order resonances and convergence remain outside this proposal.
    """
    if type(order) is not int or not 2 <= order <= 12:
        raise ValueError('arrival_order')
    if type(precision) is not int or not 50 <= precision <= local.spectral.decimal_precision:
        raise ValueError('arrival_precision')
    with localcontext() as ctx:
        ctx.prec = precision
        ls,lf = local.spectral.decimal_eigenvalues
        if not 0 < ls <= lf:
            raise ValueError('arrival_requires_minimum')
        R = local.spectral.decimal_frame
        H = tuple(tuple(_decimal(x) for x in row) for row in local.exact_hessian)
        field = tuple(tuple(tuple(_decimal(x) for x in row) for row in c)
                      for c in local.exact_grad)
        # Full normal-form mixed coefficients divide by multiples of the
        # slow rate even for terms involving the fast mode. In the stiff
        # regime this coordinate change can be uselessly large. Retain the
        # slow entry jet and the incoming homogeneous Frobenius mode; leave
        # its omitted nonlinear couplings to independent residual scrutiny.
        mode = 'slow_frobenius' if lf/ls >= D(10000) else 'full_poincare'
        K = [{(1,0):+R[d][0],(0,1):+R[d][1]} for d in range(2)]
        margin = D(1)
        for degree in range(2,order+1):
            nonlinear = _compose(field,K,degree)
            for i in range(degree+1):
                j = degree-i
                if mode == 'slow_frobenius' and j:
                    continue
                rate = i*ls+j*lf
                gap = min(abs(rate-ls),abs(rate-lf))/max(rate,lf)
                margin = min(margin,gap)
                if gap <= divisor_floor:
                    raise ValueError('arrival_resonance_or_small_divisor')
                a,b,c = H[0][0]-rate,H[0][1],H[1][1]-rate
                det = a*c-b*b
                f,g = (-nonlinear[d].get((i,j),D(0)) for d in range(2))
                if not det:
                    raise ValueError('arrival_singular_homological_equation')
                K[0][i,j],K[1][i,j] = (c*f-b*g)/det,(a*g-b*f)/det
        return {'K':K,'rates':(+ls,+lf),'order':order,'precision':precision,
                'divisor_margin':str(margin),'mode':mode}


def _evaluate(K,u,v,derivative=False):
    if not derivative:
        return tuple(sum((c*(u**i if i else 1)*(v**j if j else 1)
                          for (i,j),c in p.items()),D(0)) for p in K)
    return tuple(tuple(sum((c*(i if d==0 else j)
                           *(u**(i-(d==0)) if i-(d==0) else 1)
                           *(v**(j-(d==1)) if j-(d==1) else 1)
                           for (i,j),c in p.items() if (i if d==0 else j)),D(0))
                       for d in range(2)) for p in K)


def _propose(local, start, target, *, chart):
    """Match both amplitudes and retain an exponential-polynomial arrival."""
    with localcontext() as ctx:
        ctx.prec = chart['precision']
        # The exact translation point used by build_local_jet, not its rounded
        # display coordinates. a is recovered independently by the caller's
        # retained exact translation identity supplied below.
        center = chart['center']
        c = tuple(_decimal(x) for x in center)
        z = tuple(D.from_float(float(x))-y for x,y in zip(start,c))
        R = local.spectral.decimal_frame
        u,v = (sum(R[d][k]*z[d] for d in range(2)) for k in range(2))
        K = chart['K']
        for _ in range(24):
            x,y = _evaluate(K,u,v)
            f,g = x-z[0],y-z[1]
            if max(abs(f),abs(g)) < D(10)**(-(ctx.prec-20))*(1+max(map(abs,z))):
                break
            J = _evaluate(K,u,v,True)
            a,b = J[0]; d,e = J[1]; det = a*e-b*d
            if not det:
                raise ValueError('arrival_matching_singular')
            du,dv = (e*f-b*g)/det,(a*g-d*f)/det
            u -= du; v -= dv
        else:
            raise ValueError('arrival_matching_not_converged')
        rho = chart['rates'][1]/chart['rates'][0]
        terms = []
        for i,j in sorted(K[0]):
            scale = (u**i if i else 1)*(v**j if j else 1)
            coeff = tuple(K[d][i,j]*scale for d in range(2))
            if any(coeff):
                terms.append((str(i+j*rho),*(str(x) for x in coeff)))
        # Resolve the fast boundary layer as well as the slow approach.
        # Infinity is the analytic endpoint, represented by None, never by
        # a finite chord whose direction is guessed from the target.
        parameters = sorted({D(k)/16 for k in range(257)} |
                            {D(k)/(16*rho) for k in range(257)})
        points = []
        for s in parameters:
            q = list(c)
            for rate,a,b in terms:
                exp = (-D(rate)*s).exp()
                q[0] += D(a)*exp; q[1] += D(b)*exp
            points.append(tuple(float(x) for x in q))
        points[0] = tuple(map(float,start))
        points.append(tuple(map(float,target)))
        evidence = {'representation':'poincare_frobenius_arrival','schema':1,
                    'center':tuple(str(x) for x in center),'terms':tuple(terms),
                    'parameters':tuple(str(x) for x in parameters)+(None,),
                    'order':chart['order'],'precision':chart['precision'],'mode':chart['mode'],
                    'divisor_margin':chart['divisor_margin'],
                    'amplitudes':(str(u),str(v)), 'spectral_ratio':str(rho),
                    'terminal':'analytic_limit'}
        return np.asarray(points),evidence


def refine_capture(m, enumeration, branch, *, order=8, precision=80,
                   backend='native'):
    """Replace only an explicit terminal connector; keep the original branch.

    The returned proposal has no inherited certificate or cached verdict.
    Failure leaves the caller's original branch untouched and raises ValueError.
    """
    from . import _poly as P
    if (branch.kind!='unstable' or branch.term!='capture' or not branch.pieces
            or branch.pieces[-1].kind!='capture_connector'
            or branch.pieces[-1].start!=len(branch.Y)-2
            or branch.pieces[-1].stop!=len(branch.Y)-1):
        raise ValueError('arrival_requires_terminal_connector')
    targets = [q for q in enumeration.minima
               if np.array_equal(branch.Y[-1],(q.a,q.b))]
    if len(targets)!=1 or targets[0].local is None:
        raise ValueError('arrival_minimum_chart_unavailable')
    q = targets[0]; local = q.local
    checks = None
    if backend == 'native':
        from .native_arrival import checked_path
        points,arrival_pieces,checks = checked_path(
            m,q,branch.Y[-2],branch.Y[-1],order=order,precision=precision)
        evidence = arrival_pieces[-1].evidence
    elif backend == 'python':
        chart = parameterization(local,order=order,precision=precision)
        b = local.center_interval.mid
        chart['center'] = (P.eval_at(m.beta,b)/P.eval_at(m.alpha,b),b)
        points,evidence = _propose(local,branch.Y[-2],branch.Y[-1],chart=chart)
        arrival_pieces = (TrajectoryPiece('minimum_arrival', 0, len(points)-1, evidence),)
    else:
        raise ValueError('arrival_backend')
    start = len(branch.Y)-2
    Y = np.vstack((branch.Y[:start],points))
    shifted = []
    for piece in arrival_pieces:
        e = piece.evidence
        if e.get('representation') == 'native_dense_trace':
            e = {**e, 'branch_offset': e['branch_offset']+start}
        shifted.append(replace(piece, start=piece.start+start, stop=piece.stop+start, evidence=e))
    pieces = (*branch.pieces[:-1], *shifted)
    diag = {**branch.diag,'arrival_refinement':{
        'representation':evidence['representation'],'order':order,
        'backend':backend,'checks':checks,
        'replaced_connector':branch.Y[-2:].tolist()}}
    return replace(branch,Y=Y,pieces=pieces,diag=diag,certs={})
