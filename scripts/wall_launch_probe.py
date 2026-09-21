"""Isolate local-launch bias while retaining the existing native shooter.

PYTHONPATH=src python scripts/wall_launch_probe.py --output out/wall_launch.json

Both protocols are numerical oracles.  Only the initial points change;
the native GL8 transport and fractional-step event locator are shared.
"""

import argparse
from decimal import Decimal as D, localcontext
import json
from pathlib import Path

from spong import model, wall_shoot, zoo
from spong.wall_continue import Dual, SectionContinuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ds',nargs='+',type=float,default=[.004,.002,.001])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    family = zoo.get_wall_family('nonnearest-saddle-connection')
    case = zoo.get(family.base_case)
    base = model.build(case.f,case.g,model.moments_uniform01(11))
    continuation = SectionContinuation(base,family.source_b,family.target_b)
    if getattr(base,'_native_kernel',None) is None:
        raise RuntimeError('this comparison requires the native GL kernel')
    records = []
    for ds in args.ds:
        old = wall_shoot.find_wall(family,ds=ds,ftol=1e-15,target_direction=-1)

        def gap(lam):
            member = wall_shoot.rheostat_model(family,lam)
            with localcontext() as ctx:
                ctx.prec = continuation.precision
                parameter = D.from_float(lam)
                launches = [continuation._launch(Dual(parameter,D(1)),i)[0]
                            for i in (0,1)]
                level = float(continuation.level/parameter)
                starts = [(float(p[0].v/parameter),float(p[1].v)) for p in launches]
            curves = [wall_shoot._level_crossing(
                member,member._native_kernel,*starts[i],bool(i),level,ds,400000)
                for i in (0,1)]
            if any(curve is None for curve in curves):
                raise RuntimeError('native crossing failed')
            return float(curves[0][-1,1]-curves[1][-1,1])

        lo,hi = 2.17770955,2.17770958
        root,value,bracket,evaluations = wall_shoot._brent(
            gap,lo,hi,gap(lo),gap(hi),xtol=0,ftol=1e-16,max_iter=80)
        record = {'ds':ds,'existing_launch_root':old.lam,
                  'new_launch_native_transport_root':root,'new_launch_b_gap':value,
                  'new_launch_evaluations':evaluations+2,
                  'note':'numerical oracle; bracket width is not an error bound'}
        records.append(record)
        print(record,flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
