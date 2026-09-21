"""Thin adapter for the C99/GMP numerical rheostat engine.

These results propose connections; they do not certify connections, Smale,
wall uniqueness, or distance to the nearest wall. All numerical decisions
and arithmetic reside in the shared native core. The Decimal implementation
in wall_continue remains an independent convergence-study oracle.
"""
from dataclasses import dataclass
from decimal import Decimal

from . import _native, sturm
from .wall_continue import Crossing, Evaluation


@dataclass(frozen=True)
class NativeWallResult:
    evaluation: Evaluation
    numerical_bracket_width: Decimal
    evaluations: int
    stage_solves: int
    step_halvings: int
    max_backward_error: float
    grade: str = 'NUMERICAL_ORACLE'


class NativeSectionContinuation:
    def __init__(self, model, source_b=None, target_b=None, *,
                 source_index=None, target_index=None,
                 source_direction=1, target_direction=-1,
                 precision_bits=192, stages=4, steps=64, launch_order=10,
                 launch_radius='0.001', level_fraction='0.5',
                 max_newton=24, max_halvings=4):
        e=sturm.enumerate_critical_points(model)
        if not e.psi_positive or not e.morse:
            raise ValueError('requires exactly certified A>0 and Morse loss')
        points=[]
        for index,approximate in ((source_index,source_b),(target_index,target_b)):
            if index is None:
                if approximate is None:
                    raise ValueError('provide saddle indices or approximate b coordinates')
                distances=sorted((abs(p.b-approximate),i) for i,p in enumerate(e.points)
                                 if p.kind=='saddle')
                if not distances or (len(distances)>1 and distances[0][0]==distances[1][0]):
                    raise ValueError('ambiguous saddle selection; use exact inventory indices')
                index=distances[0][1]
            if index < 0 or index >= len(e.points) or e.points[index].kind!='saddle':
                raise ValueError('selected inventory entry is not a saddle')
            points.append(e.points[index])
        self._steps=steps
        self._precision_bits=precision_bits
        self.last_diagnostics=None
        self._context=_native.rheostat_create(
            tuple(map(str,model.alpha)),tuple(map(str,model.beta)),str(model.C),
            tuple(str(endpoint) for p in points for endpoint in (p.interval.lo,p.interval.hi)),
            *(0 if p.source=='B' else 1 for p in points),
            source_direction,target_direction,precision_bits,stages,steps,launch_order,
            max_newton,max_halvings,str(launch_radius),str(level_fraction))

    @property
    def steps(self):
        return self._steps

    @property
    def precision_bits(self):
        return self._precision_bits

    def _decode(self, result, *, steps=None):
        self.last_diagnostics={k:v for k,v in result.items() if k!='values'}
        if result['status']:
            raise ArithmeticError(f"native rheostat refusal ({result['status']}): {result['reason']}")
        v=tuple(map(Decimal,result['values']))
        crossings=tuple(Crossing((v[4+4*k],v[5+4*k]),(v[6+4*k],v[7+4*k]),
                                v[12+k],v[14+k],self.steps if steps is None else steps,v[17+k]) for k in range(2))
        self.level=v[3]
        return NativeWallResult(Evaluation(v[0],v[1],v[2],*crossings),v[16],
                               result['evaluations'],result['stage_solves'],
                               result['step_halvings'],result['max_backward_error'])

    def launch(self, lam):
        """Native germs in base (alpha,b), plus their base-loss section.

        This is numerical display data, never a connection certificate.
        Sensitivities differentiate the germs, not section crossings.
        """
        result = self._decode(_native.rheostat_launch(self._context, str(lam)), steps=0)
        return result.evaluation

    def evaluate(self, lam):
        return self._decode(_native.rheostat_evaluate(self._context,str(lam))).evaluation

    def find_root(self, lo, hi, *, parameter_tol='1e-25', max_iterations=32):
        """Safeguarded native sensitivity-Newton with fresh two-sided shots.

        The tolerance applies to the numerical shooting map, not the true
        connection parameter. Require independent launch/step/precision studies.
        """
        return self._decode(_native.rheostat_locate(self._context,str(lo),str(hi),
                            str(parameter_tol),max_iterations))

    def find_root_from_brackets(self, brackets, *, bounds,
                               parameter_tol='1e-25', max_iterations=32):
        """Aggregate old brackets; native code must recover its own signs."""
        pairs=[(Decimal(str(a)),Decimal(str(b))) for a,b in brackets]
        if not pairs or any(not a.is_finite() or not b.is_finite() or not 0<a<b for a,b in pairs):
            raise ValueError('need positive ordered finite seed brackets')
        lo=min(a for a,b in pairs)
        hi=max(b for a,b in pairs)
        return self._decode(_native.rheostat_locate_seeded(self._context,str(lo),str(hi),
                            str(bounds[0]),str(bounds[1]),str(parameter_tol),max_iterations))


@dataclass(frozen=True)
class NativePairResult:
    lam: Decimal
    eta: Decimal
    gaps: tuple
    scaled_corrections: tuple
    scaled_determinant: Decimal
    eta_derivative_errors: tuple
    scaled_correction_norm: Decimal
    scaled_residual_norm: Decimal
    diagnostics: dict
    grade: str = 'NUMERICAL_ORACLE'


def locate_connection_pair(f, g_base, g_direction, moments, pairs, *,
                           initial, lower, upper, steps=128, precision_bits=192,
                           stages=4, launch_order=10, launch_radius='0.001',
                           level_fraction='0.5', eta_difference_step='1e-5',
                           scaled_tolerance='1e-22', max_iterations=16,
                           saddle_bounds=None):
    """Marshal an exact affine-G family to the native two-parameter solver.

    g(eta)=g_base+eta*g_direction. pairs gives two (source_b,target_b,
    source_b_direction) selections at the initial eta. Exact initial isolating
    boxes are widened only to neighboring critical midpoints, then stay FIXED
    throughout the native solve. Optional saddle_bounds supplies four explicit
    rational (lo,hi) boxes instead. Python performs exact coefficient algebra
    and initial saddle selection only, never Newton/FD/stopping decisions.
    """
    from fractions import Fraction as Q
    from . import model
    f=tuple(map(Q,f));g=tuple(map(Q,g_base));q=tuple(map(Q,g_direction));mu=tuple(map(Q,moments))
    n=max(len(g),len(q));g=g+(Q(0),)*(n-len(g));q=q+(Q(0),)*(n-len(q))
    if len(pairs)!=2:
        raise ValueError('need exactly two connection pairs')
    # These are rational input coefficients, not a floating numerical solver.
    A=[[Q(0) for _ in range(2*n-1)] for _ in range(3)]
    B=[[Q(0) for _ in range(n)] for _ in range(2)]
    for i in range(n):
        for j in range(n):
            A[0][i+j]+=g[i]*g[j]*mu[i+j]
            A[1][i+j]+=(g[i]*q[j]+q[i]*g[j])*mu[i+j]
            A[2][i+j]+=q[i]*q[j]*mu[i+j]
        for j in range(len(f)):
            B[0][i]+=g[i]*f[j]*mu[i+j]
            B[1][i]+=q[i]*f[j]*mu[i+j]
    eta=Q(str(initial[1]))
    seed=model.build(f,[a+eta*b for a,b in zip(g,q)],mu)
    inventory=sturm.enumerate_critical_points(seed)
    if not inventory.morse or not inventory.psi_positive:
        raise ValueError('initial family member must have exact A>0 and Morse skeleton')
    bounds=[];sources=[];directions=[]
    for source_b,target_b,sign in pairs:
        for approximate,direction in ((source_b,sign),(target_b,-sign)):
            candidates=sorted((abs(p.b-approximate),i) for i,p in enumerate(inventory.points) if p.kind=='saddle')
            if not candidates or (len(candidates)>1 and candidates[0][0]==candidates[1][0]):
                raise ValueError('ambiguous initial saddle selection')
            i=candidates[0][1];p=inventory.points[i]
            lo=(inventory.points[i-1].interval.hi+p.interval.lo)/2 if i else p.interval.lo-1
            hi=(p.interval.hi+inventory.points[i+1].interval.lo)/2 if i+1<len(inventory.points) else p.interval.hi+1
            bounds.extend((lo,hi));sources.append(0 if p.source=='B' else 1);directions.append(direction)
    if saddle_bounds is not None:
        if len(saddle_bounds)!=4 or any(len(b)!=2 for b in saddle_bounds):
            raise ValueError('need four saddle interval pairs')
        bounds=[Q(x) for interval in saddle_bounds for x in interval]
    result=_native.rheostat_pair_locate(tuple(tuple(map(str,a)) for a in A+B),str(seed.C),
        tuple(map(str,bounds)),tuple(sources),tuple(directions),tuple(map(str,initial)),
        tuple(map(str,lower)),tuple(map(str,upper)),str(launch_radius),str(level_fraction),
        str(eta_difference_step),str(scaled_tolerance),
        (precision_bits,stages,steps,launch_order,24,4),max_iterations)
    if result['status']:
        raise ArithmeticError(f"native pair refusal ({result['status']}): {result['reason']}")
    v=tuple(map(Decimal,result.pop('values')))
    return NativePairResult(v[0],v[1],v[2:4],v[4:6],v[6],v[7:9],v[9],v[10],result)
