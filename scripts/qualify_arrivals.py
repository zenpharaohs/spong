#!/usr/bin/env python3
"""Fixed-cohort paired arrival qualification; never promotes production policy.

Each model runs in an isolated process. A fixed tracing work ceiling and
per-stage deadlines make censoring explicit. Results stream to JSONL, so a
failed/slow proposal cannot erase already measured baselines or branches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))

POLICY={'max_sin_squared':1e-6,'max_binding_eps':64.,'max_eigen_sin_squared':1e-12,
        'order':8,'precision':80,'sample_budget':50000,'trace_step_ceiling':50000,
        'raw_step_budget':4096,'stage_seconds':30,'model_seconds':90}


@contextmanager
def deadline(seconds):
    def expired(_signum,_frame):
        raise TimeoutError('stage_deadline')
    old=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)


def attempt(fn):
    start=time.perf_counter()
    try:
        with deadline(POLICY['stage_seconds']):
            result=fn()
        return {**result,'seconds':time.perf_counter()-start}
    except Exception as exc:
        return {'status':'timeout' if isinstance(exc,TimeoutError) else 'error',
                'reason':f'{type(exc).__name__}: {exc}','seconds':time.perf_counter()-start}


def geometry_ok(r):
    return (r.get('status')=='measured' and r.get('max_sin_squared',1)<=POLICY['max_sin_squared']
            and not any(r.get(k,0) for k in ('opposed_flow','orthogonal_flow','zero_tangents',
                                           'stationary_samples','unvalidated_steps'))
            and r.get('binding_eps',0)<=POLICY['max_binding_eps']
            and r.get('join_eps',0)<=POLICY['max_binding_eps'])


def connector_ok(r):
    return (r.get('status')=='measured' and r.get('sin_squared',1)<=POLICY['max_sin_squared']
            and r.get('flow_alignment')==1 and r.get('loss_direction')==1)


def emit(row):
    print(json.dumps(row,allow_nan=False),flush=True)


def compare_entry(m,e,q,start,target,*,entry,original=None,raw_cap=None,raw_budget=None):
    import numpy as np
    from spong import charts,merge_tree
    from spong.arrival_series import parameterization,_propose
    from spong.arrival_check import measure_arrival,check_target
    from spong import _poly as P
    from spong.credibility import level_normal_reference
    from spong.native_trace import measure_native
    from spong.trajectory import PieceRecorder,trace_evidence

    distance=float(np.hypot(*(np.asarray(start)-target)))
    scale=np.finfo(float).eps*(1+float(np.hypot(*target)))
    common={'entry':entry,'start':list(map(float,start)),'target':list(map(float,target)),
            'distance':distance,'distance_scaled_eps':distance/scale,
            'spectral_ratio':float(q.local.spectral.eigenvalues[1]/q.local.spectral.eigenvalues[0])}
    emit({'event':'entry',**common})
    if original is not None:
        r=attempt(lambda:{'status':'measured','connector':level_normal_reference(m,original,-1)[0]})
        r['pass']=connector_ok(r.get('connector',{}));emit({'event':'method','entry':entry,'method':'current_connector',**r})
    def fate():
        tree=merge_tree.build(m,e)
        found=merge_tree.widest_forcing_component(m,e,tree,*map(float,start))
        index=next(i for i,p in enumerate(e.points) if p is q)
        return {'status':'measured','forced':found is not None and found[1].minima==(index,)}
    emit({'event':'fate','entry':entry,**attempt(fate)})

    def raw():
        diag={};t=time.perf_counter()
        pts,term=charts._centered_raw_arrival(start,target,q.local,
                    distance if raw_cap is None else raw_cap,diag,
                    max_steps=POLICY['raw_step_budget'] if raw_budget is None else raw_budget)
        build=time.perf_counter()-t
        rec=PieceRecorder();xy=[tuple(start)]
        rec.append(xy,pts,'centered_arrival',term=term,evidence=trace_evidence(diag))
        measures=[measure_native(m,np.asarray(xy),p,-1,sample_budget=POLICY['sample_budget'])
                  for p in rec.pieces if p.kind=='centered_arrival']
        connector=(level_normal_reference(m,np.asarray(pts[-2:]),-1)[0]
                   if term=='capture' and len(pts)>1 else None)
        passed=term=='capture' and bool(measures) and all(map(geometry_ok,measures)) and connector_ok(connector or {})
        return {'status':'measured','term':term,'points':len(pts),'build_seconds':build,
                'geometry':measures,'connector':connector,'pass':passed,
                'accepted_steps':diag.get('centered_arrival',{}).get('accepted_steps')}
    r=attempt(raw);r.setdefault('pass',False)
    emit({'event':'method','entry':entry,'method':'raw_arrival',**r})

    def analytic():
        t=time.perf_counter();chart=parameterization(q.local,order=POLICY['order'],precision=POLICY['precision'])
        b=q.local.center_interval.mid;chart['center']=(P.eval_at(m.beta,b)/P.eval_at(m.alpha,b),b)
        points,evidence=_propose(q.local,start,target,chart=chart)
        build=time.perf_counter()-t
        r=measure_arrival(m,points,evidence,sample_budget=POLICY['sample_budget'])
        a=check_target(m,q,evidence)
        passed=(geometry_ok(r) and a.get('status')=='measured' and a.get('center_bound')
                and a.get('positive_hessian') and a.get('inward')
                and a.get('sin_squared',1)<=POLICY['max_eigen_sin_squared'])
        return {'status':'measured','build_seconds':build,'geometry':r,'attachment':a,
                'mode':evidence['mode'],'pass':bool(passed),'points':len(points)}
    r=attempt(analytic);r.setdefault('pass',False)
    emit({'event':'method','entry':entry,'method':'analytic',**r})


def worker(spec):
    os.environ['SPONG_ENGINE']='native';os.environ['SPONG_WORKERS']='1'
    os.environ['SPONG_STUB_MODE']='jet'
    import numpy as np
    from spong import charts,engine,model,portrait,sturm,zoo
    import qualify
    engine.use('native');charts.STEP_CEILING=POLICY['trace_step_ceiling']
    emit({'event':'begin','spec':spec,'policy':POLICY})
    started=time.perf_counter()
    try:
        with deadline(POLICY['model_seconds']):
            if spec['cohort'] in ('zoo','corpus'):
                z=zoo.get(spec['name'])
                moments=model.moments_normal01 if z.moment_dist=='normal01' else model.moments_uniform01
                m=model.build(list(z.f),list(z.g),moments(2*max(len(z.f),len(z.g))-1))
            else:
                generate=qualify.random_model if spec['cohort']=='random' else qualify.directed_model
                m,_=generate(random.Random(spec['seed']),spec['max_degree'])
            if m is None:
                emit({'event':'model','status':'generator_declined'});return
            exact=sturm.enumerate_critical_points(m)
            if not exact.morse:
                emit({'event':'model','status':'non_morse'});return
            e=sturm.materialize_stubs(m,exact)
            if spec['cohort']=='corpus':
                item=json.loads((REPO/'tests/corpus/centered_arrival.json').read_text())[spec['index']]
                entries=[(spec['index'],item['input'])]
                branches=None
            else:
                p=portrait.compute(m,_enumeration=e,_skip_audit=True,
                                   view=z.default_view if spec['cohort']=='zoo' else None)
                branches=p.branches
                entries=[(i,b) for i,b in enumerate(branches) if b.term=='capture']
            emit({'event':'model','status':'constructed','seconds':time.perf_counter()-started,
                  'degrees':[len(m.f)-1,len(m.g)-1],'minima':len(e.minima),
                  'entries':len(entries),'branch_terms':None if branches is None else [b.term for b in branches],
                  'topology':'not_audited'})
    except Exception as exc:
        emit({'event':'model','status':'timeout' if isinstance(exc,TimeoutError) else 'error',
              'reason':f'{type(exc).__name__}: {exc}','seconds':time.perf_counter()-started});return
    for index,obj in entries:
        try:
            if spec['cohort']=='corpus':
                start=np.array(obj['start']);target=np.array(obj['target'])
                raw_cap=obj['cap_r'];raw_budget=min(obj['max_steps'],POLICY['raw_step_budget']);original=None
            else:
                if not obj.pieces or obj.pieces[-1].kind!='capture_connector':
                    emit({'event':'entry_refused','entry':index,'reason':'no_terminal_connector'});continue
                start,target=obj.Y[-2:];raw_cap=raw_budget=None;original=obj.Y[-2:]
            q=next(q for q in e.minima if np.array_equal(target,(q.a,q.b)))
            compare_entry(m,e,q,start,target,entry=index,original=original,raw_cap=raw_cap,raw_budget=raw_budget)
        except Exception as exc:
            emit({'event':'entry_refused','entry':index,'reason':f'{type(exc).__name__}: {exc}'})
    emit({'event':'end','seconds':time.perf_counter()-started})


def cohort(seed,random_cases,directed_cases):
    from spong import zoo
    rng=random.Random(seed);specs=[]
    for mode,count in [('random',random_cases),('directed',directed_cases)]:
        for i in range(count):
            specs.append({'cohort':mode,'index':i,'seed':rng.randrange(2**31),'max_degree':5})
    specs += [{'cohort':'zoo','name':name} for name in zoo.names()]
    specs += [{'cohort':'corpus','name':r['case'],'index':i}
              for i,r in enumerate(json.loads((REPO/'tests/corpus/centered_arrival.json').read_text()))]
    return specs


def run_one(index,spec,out):
    stem=f'{index:04d}'
    start=time.perf_counter()
    with (out/(stem+'.jsonl')).open('w') as log,(out/(stem+'.stderr')).open('w') as err:
        try:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(spec)],
                             cwd=REPO,stdout=log,stderr=err,timeout=600,
                             env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            return {'index':index,'spec':spec,'returncode':p.returncode,'seconds':time.perf_counter()-start}
        except subprocess.TimeoutExpired:
            return {'index':index,'spec':spec,'status':'worker_timeout','seconds':time.perf_counter()-start}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker');ap.add_argument('--output',type=Path)
    ap.add_argument('--seed',type=int,default=20260925)
    ap.add_argument('--random',type=int,default=200);ap.add_argument('--directed',type=int,default=200)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    if args.worker:worker(json.loads(args.worker));return
    if args.output is None:ap.error('--output is required')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    specs=cohort(args.seed,args.random,args.directed)
    paths=[*sorted((REPO/'src/spong').glob('*.py')),Path(__file__),REPO/'scripts/qualify.py',
           REPO/'tests/corpus/centered_arrival.json',*sorted((REPO/'src/spong').glob('_native*.so'))]
    manifest={'seed':args.seed,'policy':POLICY,'workers':args.workers,'specs':specs,
              'sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'completion.jsonl').open('w') as completed,ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,i,spec,out) for i,spec in enumerate(specs)]
        for n,f in enumerate(as_completed(futures),1):
            row=f.result();completed.write(json.dumps(row)+'\n');completed.flush()
            print(f'{n}/{len(specs)} completed: {row["index"]} {row["spec"]["cohort"]} ({row["seconds"]:.1f}s)',flush=True)


if __name__=='__main__':main()
