"""Reproducible Decimal Gauss/sensitivity study of a rheostat wall.

PYTHONPATH=src python scripts/wall_continue_probe.py --steps 64 128 256 512 \
    --output out/wall_continue.json

Strings retain all computed digits.  Results are numerical-oracle grade;
the last root-iteration residual does NOT bound manifold discretization error.
"""

import argparse
from dataclasses import asdict
from decimal import Decimal, localcontext
import json
from pathlib import Path
import time

from spong import model, zoo
from spong.wall_continue import SectionContinuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',nargs='+',type=int,default=[64,128,256,512])
    parser.add_argument('--stages',type=int,choices=[2,3,4],default=4)
    parser.add_argument('--precision',type=int,default=50)
    parser.add_argument('--launch-order',type=int,default=10)
    parser.add_argument('--launch-radius',default='0.001')
    parser.add_argument('--level-fraction',default='0.5')
    parser.add_argument('--predictor',choices=['hermite','newton'],default='hermite')
    parser.add_argument('--lo',default='2.1776')
    parser.add_argument('--hi',default='2.1778')
    parser.add_argument('--seed-bracket',nargs=2,action='append',default=[],
                        metavar=('LO','HI'),help='reuse an existing old-method bracket')
    parser.add_argument('--legacy-brackets',type=int,choices=[0,1,2],default=0,
                        help='generate coarse legacy brackets at one or two step sizes')
    parser.add_argument('--legacy-width',type=float,default=1e-6)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    family = zoo.get_wall_family('nonnearest-saddle-connection')
    case = zoo.get(family.base_case)
    m = model.build(case.f,case.g,model.moments_uniform01(2*max(len(case.f),len(case.g))-1))
    report = {'family':family.name,'grade':'NUMERICAL_ORACLE',
              'coordinates':'fixed original loss; metric diag(Lambda^2,1)',
              'outer_method':f'sensitivity-{args.predictor} predictor with two-shot correction',
              'inner_method':f'GL{2*args.stages} in logarithmic loss distance',
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'exact_input':{'f':list(map(str,m.f)),'g':list(map(str,m.g)),
                             'mu':list(map(str,m.mu))},'runs':[]}
    seed_brackets = list(args.seed_bracket)
    if args.legacy_brackets:
        from spong import wall_shoot
        started = time.perf_counter()
        legacy = []
        for ds in (.004,.002)[:args.legacy_brackets]:
            seed = wall_shoot.find_wall(family,ds=ds,xtol=args.legacy_width,
                                       ftol=0,target_direction=-1)
            seed_brackets.append(seed.bracket)
            legacy.append({'ds':ds,'bracket':seed.bracket,'evaluations':seed.evaluations,
                           'history':seed.history})
        report['legacy_generation'] = {'seconds':time.perf_counter()-started,'runs':legacy}
    report['seed_brackets'] = seed_brackets
    previous = None
    for steps in args.steps:
        c = SectionContinuation(m,family.source_b,family.target_b,steps=steps,
             stages=args.stages,precision=args.precision,launch_order=args.launch_order,
             launch_radius=args.launch_radius,level_fraction=args.level_fraction)
        start = time.perf_counter()
        if seed_brackets:
            root,history = c.find_root_from_brackets(seed_brackets,
                bounds=(family.below_parameter,family.above_parameter),predictor=args.predictor)
        else:
            root,history = c.find_root(args.lo,args.hi,predictor=args.predictor)
        with localcontext() as ctx:
            ctx.prec = args.precision
            difference = None if previous is None else root.lam-previous
            euclidean_gap = sum((x-y)**2 for x,y in zip(root.unstable.point,root.stable.point)).sqrt()
            correction = abs(root.gap/root.slope)
        run = {'steps':steps,'seconds':time.perf_counter()-start,
               'root':asdict(root),'evaluations':len(history),
               'difference_from_previous':difference,
               'root_iteration_correction':correction,'euclidean_gap':euclidean_gap,
               'level':c.level,'history':[{'Lambda':e.lam,'gap':e.gap,'slope':e.slope}
                                        for e in history]}
        report['runs'].append(run)
        previous = root.lam
        print(f'GL{2*args.stages} n={steps}: Lambda={root.lam}, change={difference}, '
              f'evaluations={len(history)}, seconds={run["seconds"]:.2f}',flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__ == '__main__':
    main()
