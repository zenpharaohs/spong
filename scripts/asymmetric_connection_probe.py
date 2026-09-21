"""Reproduce/refine a double-connection proposal with the native GMP driver.

Python assembles exact family coefficients and serializes diagnostics only.
Convergence of the discrete shooting map is not a connection certificate.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spong import zoo
from spong.wall_native import locate_connection_pair


def run(name, steps, precision=192):
    case=zoo.get_numerical_connection(name)
    g=list(case.base_g);g[1]='0'
    strong='strongly' in name
    return locate_connection_pair(case.base_f,g,['0','1'],case.moments(21),
        case.connection_pairs,initial=(case.parameter,case.base_g[1]),
        lower=('1.30','-.30') if strong else ('1.29','-.021'),
        upper=('1.33','-.27') if strong else ('1.33','-.018'),
        steps=steps,precision_bits=precision)


def main(default_case='two-asymmetric-connections'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path)
    parser.add_argument('--case',default=default_case,choices=[
        'two-asymmetric-connections','two-strongly-asymmetric-connections'])
    parser.add_argument('--steps',nargs='+',type=int,default=[64,128,256,512])
    parser.add_argument('--precision',type=int,default=192)
    args=parser.parse_args();results=[]
    for steps in args.steps:
        result=run(args.case,steps,args.precision)
        record={'case':args.case,'steps':steps,'precision_bits':args.precision,
                **asdict(result)}
        results.append(record)
        print(json.dumps(record,default=str),flush=True)
        if args.output:args.output.write_text(json.dumps(results,indent=2,default=str)+'\n')

if __name__=='__main__':main()
