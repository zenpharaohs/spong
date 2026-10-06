#!/usr/bin/env python3
"""Replay frozen arrival entries against native GMP, without retracing paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def worker(source, job):
    os.environ['SPONG_ENGINE'] = 'native'
    os.environ['SPONG_WORKERS'] = '1'
    os.environ['SPONG_STUB_MODE'] = 'jet'
    import numpy as np
    from spong import model, sturm, zoo, _poly as P
    from spong.native_arrival import propose, clear_cache
    from spong.arrival_check import measure_arrival, check_target
    from qualify_arrivals import deadline, attempt, geometry_ok
    import qualify
    entries = [e for e in json.loads(source.read_text()) if e['job'] == job]
    spec = entries[0]
    emit({'event': 'begin', 'job': job, 'entries': len(entries)})
    try:
        with deadline(90):
            if spec['cohort'] in ('random', 'directed'):
                generate = qualify.random_model if spec['cohort'] == 'random' else qualify.directed_model
                m, _ = generate(random.Random(spec['seed']), spec['max_degree'])
            else:
                z = zoo.get(spec['name'])
                moments = model.moments_normal01 if z.moment_dist == 'normal01' else model.moments_uniform01
                m = model.build(list(z.f), list(z.g), moments(2*max(len(z.f), len(z.g))-1))
            enumeration = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    except Exception as exc:
        emit({'event': 'model_error', 'job': job, 'reason': str(exc)})
        return
    for entry in entries:
        def run():
            q = next(q for q in enumeration.minima if [q.a, q.b] == entry['target'])
            b = q.local.center_interval.mid
            center = (P.eval_at(m.beta, b)/P.eval_at(m.alpha, b), b)
            clear_cache()
            t = time.perf_counter()
            try:
                points, evidence = propose(q.local, entry['start'], entry['target'], center)
            except ValueError as exc:
                return {'status': 'error', 'pass': False,
                        'reason': f'{type(exc).__name__}: {exc}',
                        'matching': getattr(exc, 'matching', None)}
            cold = time.perf_counter()-t
            t = time.perf_counter()
            warm_points, warm_evidence = propose(q.local, entry['start'], entry['target'], center)
            warm = time.perf_counter()-t
            if not np.array_equal(points, warm_points) or evidence != warm_evidence:
                raise AssertionError('native_cached_chart_disagreement')
            g = measure_arrival(m, points, evidence, sample_budget=50000)
            a = check_target(m, q, evidence)
            passed = (geometry_ok(g) and a.get('status') == 'measured'
                      and all(a.get(k) for k in ('center_bound', 'positive_hessian', 'inward'))
                      and a.get('sin_squared', 1) <= 1e-12)
            return {'status': 'measured', 'pass': bool(passed), 'cold_seconds': cold,
                    'warm_seconds': warm, 'geometry': g, 'attachment': a, 'mode': evidence['mode'],
                    'conditioning': evidence.get('conditioning'), 'matching': evidence.get('matching')}
        r = attempt(run)
        r.setdefault('pass', False)
        old = entry['methods']['analytic']
        emit({'event': 'arrival', 'job': job, 'entry': entry['entry'],
              'cohort': entry['cohort'], 'python_pass': old['pass'],
              'python_build_seconds': old.get('build_seconds'),
              'python_status': old['status'], 'python_reason': old.get('reason'), **r})
    emit({'event': 'end', 'job': job})


def run_one(source, out, job):
    with (out/f'{job:04d}.jsonl').open('w') as log, (out/f'{job:04d}.stderr').open('w') as err:
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source),
                                '--worker', str(job)], stdout=log, stderr=err, cwd=REPO,
                               timeout=600, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            return {'job': job, 'returncode': p.returncode}
        except subprocess.TimeoutExpired:
            return {'job': job, 'status': 'worker_timeout'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--worker', type=int)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.source, args.worker)
        return
    if args.output is None:
        ap.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.source.read_text())
    jobs = sorted({e['job'] for e in entries})
    paths = [*sorted((REPO/'src/spong').glob('*.py')), *sorted((REPO/'src/spong').glob('_native*.so')),
             REPO/'src/c/spong_analytic_arrival.c', REPO/'include/spong/spong_analytic_arrival.h',
             REPO/'src/spong/_native_analytic_arrival.inc', Path(__file__).resolve()]
    manifest = {'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'entries': len(entries), 'jobs': jobs, 'workers': args.workers,
                'sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (args.output/'completion.jsonl').open('w') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, args.source, args.output, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            log.write(json.dumps(result)+'\n');log.flush()
            print(f'{n}/{len(jobs)} completed: {result}', flush=True)


if __name__ == '__main__':
    main()
