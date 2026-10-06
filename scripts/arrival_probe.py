#!/usr/bin/env python3
"""Compare original capture connectors with independently checked entry charts.

SPONG_ENGINE=native python scripts/arrival_probe.py --orders 6 8
This is an arrival-only experiment; no portrait verdict is promoted.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from spong import model,portrait,zoo
from spong.arrival_series import refine_capture
from spong.arrival_check import measure_arrival,check_target
from spong.credibility import level_normal_reference


def probe(name,orders=(6,8),precision=80):
    z=zoo.get(name)
    moments=model.moments_normal01 if z.moment_dist=='normal01' else model.moments_uniform01
    m=model.build(list(z.f),list(z.g),moments(2*max(len(z.f),len(z.g))-1))
    p=portrait.certified_compute(m,view=z.default_view)
    branches=[]
    for i,b in enumerate(p.branches):
        if b.term!='capture': continue
        target=next(q for q in p.enumeration.minima if np.array_equal(b.Y[-1],(q.a,q.b)))
        length=float(np.hypot(*(b.Y[-2]-b.Y[-1])))
        scale=np.finfo(float).eps*(1+float(np.hypot(*b.Y[-1])))
        trials=[]
        for order in orders:
            start=time.perf_counter()
            try:
                candidate=refine_capture(m,p.enumeration,b,order=order,precision=precision)
                piece=candidate.pieces[-1]
                measured=measure_arrival(m,candidate.Y[piece.start:],piece.evidence)
                trials.append({'order':order,'precision':precision,
                    'mode':piece.evidence['mode'],'geometry':measured,
                    'attachment':check_target(m,target,piece.evidence),
                    'seconds':time.perf_counter()-start})
            except ValueError as exc:
                trials.append({'order':order,'status':'unresolved','reason':str(exc),
                               'seconds':time.perf_counter()-start})
        branches.append({'branch':i,'original_owner':b.pieces[-2].kind,
                         'connector_length':length,'connector_scaled_eps':length/scale,
                         'original_connector':level_normal_reference(m,b.Y[-2:],-1)[0],
                         'trials':trials})
    return {'case':name,'branches':branches,'certified':False,
            'scope':'arrival_representation_experiment'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',nargs='*',default=['minimal-quartet','dead-neuron-far-saddle-d3'])
    parser.add_argument('--orders',type=int,nargs='+',default=[6,8])
    parser.add_argument('--precision',type=int,default=80)
    args=parser.parse_args()
    for name in args.names:
        print(json.dumps(probe(name,args.orders,args.precision)),flush=True)
