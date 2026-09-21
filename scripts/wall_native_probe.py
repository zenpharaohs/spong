"""Native rheostat convergence study; results are numerical proposals only."""
import argparse
from dataclasses import asdict
from decimal import Decimal,localcontext
import json
from pathlib import Path
import time
from spong import model,zoo
from spong.wall_native import NativeSectionContinuation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,nargs='+',default=[256,512,1024,2048])
    p.add_argument('--cross-checks',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    family=zoo.get_wall_family('nonnearest-saddle-connection')
    case=zoo.get(family.base_case)
    m=model.build(case.f,case.g,model.moments_uniform01(11))
    configs=[{'steps':n} for n in args.steps]
    if args.cross_checks:
        n=max(args.steps)
        configs.extend([{'steps':n,'precision_bits':256},
                        {'steps':n,'launch_order':12,'launch_radius':'0.002'},
                        {'steps':n,'level_fraction':'0.35'}])
    report={'grade':'NUMERICAL_ORACLE','family':family.name,
            'method':'native C99/GMP, GL8, sensitivity-Newton',
            'parameter_tolerance':'1e-40','runs':[]}
    previous=None
    for config in configs:
        t=time.perf_counter()
        c=NativeSectionContinuation(m,family.source_b,family.target_b,**config)
        result=c.find_root('2.1776','2.1778',parameter_tol='1e-40')
        with localcontext() as ctx:
            ctx.prec=80
            change=None if previous is None else result.evaluation.lam-previous
        entry={'settings':config,'seconds':time.perf_counter()-t,
               'difference_from_previous':change,'result':asdict(result)}
        report['runs'].append(entry)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,default=str)+'\n')
        print(config,result.evaluation.lam,entry['seconds'],flush=True)
        previous=result.evaluation.lam

if __name__=='__main__':
    main()
