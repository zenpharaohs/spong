"""Experimental opt-in native stiff bridge; all admissions are independent."""
from fractions import Fraction as Q
from math import ceil
import numpy as np
from .native_arrival import _rational_text,checked_path,checked_proposal,_prefix_admitted
from .trajectory import TrajectoryPiece


def propose_bridge(m,minimum,start,*,digits=80,max_steps=512):
    from . import _native,_poly as P
    from .local import _shift,_lincomb
    a,b=map(lambda x:Q(float(x)),start)
    A,B=_shift(m.alpha,b),_shift(m.beta,b)
    Ap,Bp=_shift(P.deriv(m.alpha),b),_shift(P.deriv(m.beta),b)
    gradient=((_lincomb((2*a,A),(Q(-2),B)),_lincomb((Q(2),A))),
              (_lincomb((a*a,Ap),(-2*a,Bp)),_lincomb((2*a,Ap),(Q(-2),Bp)),Ap))
    rows=max(map(len,gradient));cols=max(len(row) for g in gradient for row in g)
    strings=tuple(_rational_text(gradient[d][i][j]) if i<len(gradient[d]) and j<len(gradient[d][i]) else '0'
                  for d in range(2) for i in range(rows) for j in range(cols))
    raw=_native.precision_bridge(strings,rows,cols,*map(str,minimum.local.spectral.decimal_eigenvalues),digits,max_steps)
    polynomials=raw.pop('polynomials')
    if not polynomials:
        error=ValueError('precision_bridge_no_steps')
        error.handoff={'policy':'precision-native-bridge-experimental-v1','native':raw}
        raise error
    evidence={'representation':'precision_bridge_cubics','schema':1,'origin':tuple(map(str,(a,b))),
              'polynomials':polynomials,'working_bits':raw['working_bits'],'branch_offset':0,'diagnostics':raw}
    # Display only; inverse entry below uses the retained exact polynomial.
    points=[tuple(map(float,start))]
    for p in polynomials:points.append(tuple(float(origin+sum(map(Q,row))) for origin,row in zip((a,b),p)))
    return np.array(points),evidence


def checked_precision_path(m,minimum,start,target,*,order=8,precision=80,check_precision=90,
                           sample_budget=10000,max_steps=512,retry_order=None):
    """Preserve existing admissions; optionally retry a refused path at another chart order."""
    if retry_order is not None:
        if type(retry_order) is not int or not 2<=retry_order<=12 or retry_order==order:
            raise ValueError('precision_bridge_retry_order')
        options=dict(precision=precision,check_precision=check_precision,
                     sample_budget=sample_budget,max_steps=max_steps)
        try:
            return checked_precision_path(m,minimum,start,target,order=order,**options)
        except ValueError as first:
            if str(first)!='precision_bridge_no_checked_entry':raise
            prior=first
        try:
            points,pieces,checks=checked_precision_path(m,minimum,start,target,
                                                        order=retry_order,**options)
        except ValueError as second:
            if str(second)!='precision_bridge_no_checked_entry':raise
            error=ValueError('precision_bridge_no_checked_entry')
            error.handoff={'policy':'precision-bridge-order-retry-v1','orders':(order,retry_order),
                           'first':getattr(prior,'handoff',None),
                           'second':getattr(second,'handoff',None)}
            raise error from second
        return points,pieces,{**checks,'order_retry':{'first_order':order,
                                                     'second_order':retry_order,
                                                     'first_reason':str(prior)}}
    from .precision_bridge_check import endpoint,measure_precision_bridge,measure_precision_join
    kwargs=dict(order=order,precision=precision,check_precision=check_precision,sample_budget=sample_budget)
    try:return checked_path(m,minimum,start,target,**kwargs)
    except ValueError as exc:
        if str(exc) not in ('arrival_handoff_no_checked_entry','arrival_handoff_prefix_rejected','arrival_handoff_trace_unavailable'):raise
        prior={'reason':str(exc),'handoff':getattr(exc,'handoff',None)}
    points,evidence=propose_bridge(m,minimum,start,digits=precision,max_steps=max_steps)
    check_precision=max(check_precision,ceil(evidence['working_bits']*0.3010299956639812)+20)
    kwargs['check_precision']=check_precision
    diagnostics={'check_precision':check_precision,'policy':'precision-native-bridge-experimental-v1','previous':prior,'native':evidence['diagnostics'],'attempts':[]}
    # Fast transients need not halve Euclidean radius. Both bounded schedules
    # are used, including exact endpoints invisible in the display polyline.
    distance=np.hypot(*(points-np.asarray(target)).T)
    candidates={len(points)-1}
    candidates.update(2**k for k in range(3,12) if 2**k<len(points))
    for k in range(1,13):
        indices=np.flatnonzero((distance<=distance[0]/2**k)&(np.arange(len(points))>0))
        if len(indices):candidates.add(int(indices[0]))
    for index in sorted(candidates):
        trial={'index':index};diagnostics['attempts'].append(trial)
        exact,_=endpoint(evidence,index)
        try:tail,te,tc=checked_proposal(m,minimum,points[index],target,exact_start=exact,**kwargs)
        except ValueError as exc:trial['reason']=str(exc);continue
        prefix=TrajectoryPiece('precision_bridge',0,index,evidence)
        g=measure_precision_bridge(m,points,prefix,precision=check_precision,sample_budget=sample_budget);trial['prefix']=g
        if not _prefix_admitted(g):trial['reason']='precision_bridge_prefix_rejected';break
        joined=np.vstack((points[:index+1],tail[1:]));tp=TrajectoryPiece('minimum_arrival',index,len(joined)-1,te)
        join=measure_precision_join(joined,prefix,tp);trial['join']=join
        if join.get('status')!='measured' or join['join_eps']>64 or join['sin_squared']>1e-6 or not join['same_orientation']:
            trial['reason']='precision_bridge_join_rejected';continue
        trial['accepted']=True;diagnostics['selected_index']=index
        return joined,(prefix,tp),{**tc,'prefix':g,'join':join,'handoff':diagnostics,'certified':False,'scope':'sampled_precision_bridge_and_analytic_arrival'}
    error=ValueError('precision_bridge_no_checked_entry');error.handoff=diagnostics;raise error
