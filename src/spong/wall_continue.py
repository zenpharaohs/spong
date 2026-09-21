"""Experimental high-precision continuation of rheostat section crossings.

The model is kept EXACTLY fixed; Lambda enters only D=diag(Lambda**2,1).
Decimal forward derivatives propagate the Lambda sensitivity through a local
parameterization germ and Gauss collocation.  Loss, not a spatial coordinate,
parameterizes each shot, so folded branches are supported.

This is a convergence-study oracle, NOT an interval certificate.  No scipy,
mpmath, native extension, or coefficient re-rounding is involved in transport.
"""

from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from functools import lru_cache
from fractions import Fraction

from . import sturm


@dataclass(frozen=True)
class Dual:
    """A Decimal value and its first Lambda derivative."""
    v: D
    d: D = D(0)

    def __add__(self, other):
        other = dual(other)
        return Dual(self.v + other.v, self.d + other.d)
    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __sub__(self, other):
        return self + -dual(other)

    def __rsub__(self, other):
        return dual(other) + -self

    def __mul__(self, other):
        other = dual(other)
        return Dual(self.v * other.v, self.d * other.v + self.v * other.d)
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = dual(other)
        value = self.v / other.v
        return Dual(value, (self.d - value * other.d) / other.v)

    def __rtruediv__(self, other):
        return dual(other) / self

    def sqrt(self):
        value = self.v.sqrt()
        return Dual(value, self.d / (2 * value))

    def ln(self):
        return Dual(self.v.ln(), self.d / self.v)

    def exp(self):
        value = self.v.exp()
        return Dual(value, value * self.d)


def decimal(value):
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return D(value.numerator) / D(value.denominator)
    return D(str(value))


def dual(value):
    return value if isinstance(value, Dual) else Dual(decimal(value))


def polyval(coefficients, x):
    value = x * 0
    for c in reversed(coefficients):
        value = value * x + c
    return value


def deriv(p):
    return tuple(i * p[i] for i in range(1, len(p)))


def _mul(p, q, n):
    return [sum((p[j] * q[k-j] for j in range(max(0, k-len(q)+1),
                min(k+1, len(p)))), dual(0)) for k in range(n+1)]


def _compose(p, q, n):
    result = [dual(0)] * (n+1)
    for c in reversed(p):
        result = _mul(result, q, n)
        result[0] += c
    return result


def _solve(matrix, right):
    """Pivoted Decimal elimination, with Decimal or dual right-hand sides."""
    a = [list(row) + [rhs] for row, rhs in zip(matrix, right)]
    n = len(a)
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(a[i][k]))
        a[k], a[pivot] = a[pivot], a[k]
        if not a[k][k]:
            raise ArithmeticError("singular collocation matrix")
        for i in range(k+1, n):
            factor = a[i][k] / a[k][k]
            for j in range(k+1, n+1):
                a[i][j] -= factor * a[k][j]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][j]*x[j] for j in range(i+1, n))) / a[i][i]
    return x


@lru_cache(maxsize=16)
def _tableau(stages, precision):
    """Gauss nodes and integrated Lagrange basis, generated at working precision."""
    with localcontext() as ctx:
        ctx.prec = precision
        if stages == 2:
            r = D(3).sqrt() / 6
            nodes = [D('.5')-r, D('.5')+r]
        elif stages == 3:
            r = D(15).sqrt() / 10
            nodes = [D('.5')-r, D('.5'), D('.5')+r]
        elif stages == 4:
            r = (D(6)/5).sqrt()
            outer = ((D(3)+2*r)/7).sqrt()/2
            inner = ((D(3)-2*r)/7).sqrt()/2
            nodes = [D('.5')-outer, D('.5')-inner,
                     D('.5')+inner, D('.5')+outer]
        else:
            raise ValueError("stages must be 2, 3, or 4")
        columns, weights = [], []
        for j, node in enumerate(nodes):
            p, denominator = [D(1)], D(1)
            for k, other in enumerate(nodes):
                if k == j:
                    continue
                q = [D(0)] * (len(p)+1)
                for i, coefficient in enumerate(p):
                    q[i] -= other*coefficient
                    q[i+1] += coefficient
                p = q
                denominator *= node-other
            primitive = [D(0)] + [v / ((i+1)*denominator)
                                       for i, v in enumerate(p)]
            columns.append([polyval(primitive, c) for c in nodes])
            weights.append(polyval(primitive, D(1)))
        return (tuple(nodes), tuple(tuple(columns[j][i] for j in range(stages))
                                  for i in range(stages)), tuple(weights))


@dataclass(frozen=True)
class Crossing:
    point: tuple[D, D]
    sensitivity: tuple[D, D]
    loss_residual: D
    launch_residual: D
    steps: int
    projection_distance: D = D(0)


@dataclass(frozen=True)
class Evaluation:
    lam: D
    gap: D
    slope: D
    unstable: Crossing
    stable: Crossing


def _hermite_parameter(left, right):
    """Zero of the cubic difference of the two crossing trajectories.

    Values and sensitivities at both ends define a Hermite polynomial in
    the fixed tangent section coordinate.  This is only a predictor: a
    fresh two-sided shot must correct it before updating the real bracket.
    """
    width = right.lam-left.lam
    f0,f1 = left.gap,right.gap
    d0,d1 = width*left.slope,width*right.slope
    coefficients = (f0,d0,3*(f1-f0)-2*d0-d1,2*(f0-f1)+d0+d1)
    t = -f0/(f1-f0)
    lo,hi = D(0),D(1)
    for _ in range(100):
        value = polyval(coefficients,t)
        if not value:
            break
        if f0*value > 0:
            lo = t
        else:
            hi = t
        slope = polyval(deriv(coefficients),t)
        candidate = t-value/slope if slope else (lo+hi)/2
        if candidate == t:
            break
        t = candidate if lo < candidate < hi else (lo+hi)/2
    return left.lam+width*t


class SectionContinuation:
    """A fixed saddle pair and a regular, fixed-loss local section chart.

    The section coordinate is a fixed tangent projection.  Unlike a bare b
    gap it detects opposite-sheet points, including at a b-turn.  This is a
    LOCAL coordinate: every evaluation verifies uniqueness of a level point
    at each coordinate in the crossing rectangle, using an exact rational
    interval bound on the normal derivative of the original polynomial.
    """

    def __init__(self, model, source_b, target_b, *, source_direction=1,
                 target_direction=-1, precision=50, stages=4, steps=64,
                 launch_order=10, launch_radius="0.001", level_fraction="0.5"):
        if precision < 30 or steps < 1 or launch_order < 2:
            raise ValueError("need precision>=30, steps>=1, launch_order>=2")
        if source_direction not in (-1, 1) or target_direction not in (-1, 1):
            raise ValueError("branch directions must be +/-1")
        self.precision, self.stages, self.steps = precision, stages, steps
        self.order, self.radius = launch_order, decimal(launch_radius)
        if self.radius <= 0:
            raise ValueError("launch_radius must be positive")
        self.directions = (source_direction, target_direction)
        self._exact_A, self._exact_B = model.alpha, model.beta
        enumeration = sturm.enumerate_critical_points(model)
        if not enumeration.psi_positive or not enumeration.morse:
            raise ValueError("requires exactly certified A>0 and Morse loss")
        with localcontext() as ctx:
            ctx.prec = precision
            self.A, self.B = [tuple(decimal(c) for c in p)
                             for p in (model.alpha, model.beta)]
            self.C = decimal(model.C)
            self.Ap, self.Bp = deriv(self.A), deriv(self.B)
            self.App, self.Bpp = deriv(self.Ap), deriv(self.Bp)
            saddles = []
            for approximate in (source_b, target_b):
                point = min(enumeration.saddles, key=lambda p: abs(p.b-approximate))
                p = tuple(decimal(c) for c in
                          (model.beta if point.source == "B" else model.N))
                b = decimal(point.interval.mid)
                for _ in range(20):
                    correction = polyval(p, b)/polyval(deriv(p), b)
                    b -= correction
                    if abs(correction) < D(10)**(-precision+8):
                        break
                saddles.append((polyval(self.B,b)/polyval(self.A,b), b))
            self.saddles = tuple(saddles)
            self.losses = tuple(self.loss(p) for p in saddles)
            fraction = decimal(level_fraction)
            if not 0 < fraction < 1 or not self.losses[0] > self.losses[1]:
                raise ValueError("need ordered saddles and 0<level_fraction<1")
            self.level = self.losses[0] + fraction*(self.losses[1]-self.losses[0])
        self.anchor = self.tangent = None

    def _check_chart(self, points, normal):
        """Strict normal monotonicity on a convex box excludes sheet aliasing."""
        def add(x,y):
            return x[0]+y[0],x[1]+y[1]
        def mul(x,y):
            values = [a*b for a in x for b in y]
            return min(values),max(values)
        def scale(x,c):
            return mul(x,(c,c))
        def ev(coefficients,x):
            out = (Fraction(0),Fraction(0))
            for c in reversed(coefficients):
                out = add(mul(out,x),(c,c))
            return out
        a,b = [(min(Fraction(p[i]) for p in points),
                max(Fraction(p[i]) for p in points)) for i in range(2)]
        A,B = self._exact_A,self._exact_B
        ga = scale(add(mul(a,ev(A,b)),scale(ev(B,b),-1)),2)
        gb = add(mul(mul(a,a),ev(deriv(A),b)),scale(mul(a,ev(deriv(B),b)),-2))
        bound = add(scale(ga,Fraction(normal[0])),scale(gb,Fraction(normal[1])))
        if bound[0] <= 0:
            raise ValueError("normal monotonicity failed: reduce parameter window or change section chart")

    def _project(self, crossing, normal):
        z = [Dual(v,d) for v,d in zip(crossing.point,crossing.sensitivity)]
        offset = dual(0)
        tolerance = D(10)**(-self.precision+8)
        for _ in range(12):
            p = [z[i]+offset*normal[i] for i in range(2)]
            residual = self.loss(p)-self.level
            g = self.gradient(p)
            correction = residual/sum(g[i]*normal[i] for i in range(2))
            offset -= correction
            if max(abs(correction.v),abs(correction.d)) < tolerance:
                break
        else:
            raise ArithmeticError("section projection failed")
        p = [z[i]+offset*normal[i] for i in range(2)]
        return Crossing(tuple(v.v for v in p),tuple(v.d for v in p),
                        crossing.loss_residual,crossing.launch_residual,
                        crossing.steps,abs(offset.v))

    def loss(self, z):
        a,b = z
        return self.C - 2*a*polyval(self.B,b) + a*a*polyval(self.A,b)

    def gradient(self, z):
        a,b = z
        return (2*(a*polyval(self.A,b)-polyval(self.B,b)),
                a*a*polyval(self.Ap,b)-2*a*polyval(self.Bp,b))

    def hessian(self, z):
        a,b = z
        ab = 2*(a*polyval(self.Ap,b)-polyval(self.Bp,b))
        return ((2*polyval(self.A,b),ab),
                (ab,a*a*polyval(self.App,b)-2*a*polyval(self.Bpp,b)))

    def _launch(self, lam, index):
        saddle = self.saddles[index]
        h = self.hessian(saddle)
        j = ((-lam*lam*h[0][0], -lam*lam*h[0][1]),
             (dual(-h[1][0]), dual(-h[1][1])))
        trace = j[0][0]+j[1][1]
        root = ((j[0][0]-j[1][1])*(j[0][0]-j[1][1])+4*j[0][1]*j[1][0]).sqrt()
        rho = (trace+root)/2 if index == 0 else (trace-root)/2
        candidates = [(j[0][1],rho-j[0][0]), (rho-j[1][1],j[1][0])]
        v = max(candidates, key=lambda v: v[0].v*v[0].v+v[1].v*v[1].v)
        norm = (v[0]*v[0]+v[1]*v[1]).sqrt()
        v = [x/norm for x in v]
        if not v[1].v:
            raise ValueError("b-direction cannot label a horizontal eigendirection")
        if v[1].v < 0:
            v = [-x for x in v]
        a,b = [dual(saddle[0]),v[0]], [dual(saddle[1]),v[1]]
        for n in range(2,self.order+1):
            aa, bb = a+[dual(0)], b+[dual(0)]
            A, B = _compose(self.A,bb,n), _compose(self.B,bb,n)
            Ap, Bp = _compose(self.Ap,bb,n), _compose(self.Bp,bb,n)
            f0 = -2*lam*lam*(_mul(aa,A,n)[n]-B[n])
            f1 = -_mul(_mul(aa,aa,n),Ap,n)[n]+2*_mul(aa,Bp,n)[n]
            m00,m01 = n*rho-j[0][0],-j[0][1]
            m10,m11 = -j[1][0],n*rho-j[1][1]
            determinant = m00*m11-m01*m10
            a.append((m11*f0-m01*f1)/determinant)
            b.append((m00*f1-m10*f0)/determinant)
        t = self.radius*self.directions[index]
        z = [polyval(a,t),polyval(b,t)]
        g = self.gradient(z)
        residual = max(abs((-lam*lam*g[0]-rho*t*polyval(deriv(a),t)).v),
                       abs((-g[1]-rho*t*polyval(deriv(b),t)).v))
        return z,residual

    def _field(self, z, lam, factor):
        p,r = self.gradient(z)
        weighted = (lam*lam*p,r)
        q = p*weighted[0]+r*r
        return [factor*v/q for v in weighted]

    def _jacobian(self, z, lam, factor):
        p,r = self.gradient(z)
        h = self.hessian(z)
        w = (lam*lam*p,r)
        q = p*w[0]+r*r
        h_w = [sum(h[i][j]*w[j] for j in range(2)) for i in range(2)]
        return [[factor*((lam*lam if i==0 else 1)*h[i][j]/q
                         -2*w[i]*h_w[j]/(q*q)) for j in range(2)] for i in range(2)]

    def _shot(self, lam, index):
        z, launch_residual = self._launch(lam,index)
        initial = self.loss(z)-self.losses[index]
        target = self.level-self.losses[index]
        if initial.v*target <= 0 or abs(initial.v) >= abs(target):
            raise ValueError("launch does not lie between saddle and section")
        # Log loss-distance spreads the singular departure over a finite,
        # smooth interval x in [0,1].  Initial is dual: its Lambda dependence
        # must be included to obtain fixed-final-loss sensitivities.
        rate = (target/initial).ln()
        nodes, a, weights = _tableau(self.stages,self.precision)
        step = D(1)/self.steps
        tolerance = D(10)**(-self.precision+10)
        for k in range(self.steps):
            factors = [initial*(rate*((D(k)+node)*step)).exp()*rate for node in nodes]
            slopes = [self._field(z,lam,factor) for factor in factors]
            for iteration in range(20):
                states = [[z[d]+step*sum(a[i][j]*slopes[j][d]
                          for j in range(self.stages)) for d in range(2)]
                          for i in range(self.stages)]
                fields = [self._field(states[i],lam,factors[i]) for i in range(self.stages)]
                residual = [slopes[i][d]-fields[i][d] for i in range(self.stages) for d in range(2)]
                scale = 1+max(max(abs(s.v),abs(s.d)) for row in slopes for s in row)
                if max(max(abs(r.v),abs(r.d)) for r in residual) < tolerance*scale:
                    break
                jac = [self._jacobian([v.v for v in states[i]],lam.v,factors[i].v)
                       for i in range(self.stages)]
                matrix = [[D(i==j and d==e)-step*a[i][j]*jac[i][d][e]
                           for j in range(self.stages) for e in range(2)]
                          for i in range(self.stages) for d in range(2)]
                correction = _solve(matrix,[-r for r in residual])
                slopes = [[slopes[i][d]+correction[2*i+d] for d in range(2)]
                          for i in range(self.stages)]
            else:
                raise ArithmeticError("Gauss stages failed; increase steps")
            z = [z[d]+step*sum(weights[i]*slopes[i][d] for i in range(self.stages))
                 for d in range(2)]
        return Crossing(tuple(v.v for v in z),tuple(v.d for v in z),
                        self.loss(z).v-self.level,launch_residual,self.steps)

    def evaluate(self, lam):
        """Crossings and analytic derivative of the DISCRETE GL shooting map.

        The tangent chart is fixed by the first evaluation and then reused.
        Its gap is a local tangent coordinate, not global arclength.
        Loss residuals are reported separately; convergence in steps and
        launches is required before interpreting a small gap as a connection.
        """
        with localcontext() as ctx:
            ctx.prec = self.precision
            lam = dual(lam)
            if lam.v <= 0:
                raise ValueError("Lambda must be positive")
            lam = Dual(lam.v,D(1))
            u,s = self._shot(lam,0),self._shot(lam,1)
            if self.tangent is None:
                self.anchor = tuple((x+y)/2 for x,y in zip(u.point,s.point))
                p,r = self.gradient(self.anchor)
                norm = (p*p+r*r).sqrt()
                self.tangent = (r/norm,-p/norm)
            normal = (-self.tangent[1],self.tangent[0])
            raw = (u.point,s.point)
            u,s = self._project(u,normal),self._project(s,normal)
            self._check_chart([self.anchor,*raw,u.point,s.point],normal)
            gap = sum(self.tangent[i]*(u.point[i]-s.point[i]) for i in range(2))
            slope = sum(self.tangent[i]*(u.sensitivity[i]-s.sensitivity[i]) for i in range(2))
            return Evaluation(lam.v,gap,slope,u,s)

    def find_root(self, lo, hi, *, parameter_tol="1e-25", max_iterations=16,
                  predictor="hermite"):
        """Sensitivity-Hermite (or Newton) continuation; returns full history.

        Its bracket is for the DISCRETIZED shooting map, not a proof bracket.
        Every proposed parameter is corrected by two fresh manifold shots.
        """
        with localcontext() as ctx:
            ctx.prec = self.precision
            if predictor not in ("hermite","newton"):
                raise ValueError("predictor must be hermite or newton")
            lo,hi = decimal(lo),decimal(hi)
            if not lo < hi:
                raise ValueError("require lo < hi")
            left,right = self.evaluate(lo),self.evaluate(hi)
            if left.gap*right.gap > 0:
                raise ValueError("section gap does not change sign")
            history = [left,right]
            return self._refine_bracket(left,right,history,parameter_tol,
                                        max_iterations,predictor)

    def find_root_from_brackets(self, brackets, *, bounds=None,
                                parameter_tol="1e-25", max_iterations=16,
                                predictor="hermite", max_repairs=4):
        """Warm-start from one or more OLD-method brackets, checked afresh.

        Combine their envelope (not their intersection) and pad by half its
        width on each side. Evaluate the new shooting map at both ends. If
        launch bias moved the root outside, use twice the best sensitivity-
        Newton displacement to propose the opposite side. Only observed signs
        of NEW evaluations establish the numerical bracket. Optional bounds
        limit repairs to a caller-supplied parameter domain.

        All evaluations, including initialization/repairs, appear in history;
        checked endpoints are reused by the corrector. Failure to recover a
        sign change is an explicit refusal, never acceptance of legacy signs.
        """
        with localcontext() as ctx:
            ctx.prec = self.precision
            if predictor not in ("hermite","newton"):
                raise ValueError("predictor must be hermite or newton")
            intervals = [(decimal(lo),decimal(hi)) for lo,hi in brackets]
            if not intervals or any(not 0 < lo < hi for lo,hi in intervals):
                raise ValueError("need positive, ordered seed brackets")
            lo,hi = min(p[0] for p in intervals),max(p[1] for p in intervals)
            width = hi-lo
            lo,hi = max(lo/2,lo-width/2),hi+width/2
            lower,upper = (D(0),None) if bounds is None else tuple(map(decimal,bounds))
            if lower < 0 or (upper is not None and not lower < upper):
                raise ValueError("invalid warm-start bounds")
            lo,hi = max(lo,lower),hi if upper is None else min(hi,upper)
            if not 0 < lo < hi:
                raise ValueError("seed brackets do not overlap positive bounds")
            history = [self.evaluate(lo),self.evaluate(hi)]
            for repair in range(max_repairs+1):
                ordered = sorted(history,key=lambda e:e.lam)
                for left,right in zip(ordered,ordered[1:]):
                    if left.gap*right.gap <= 0:
                        return self._refine_bracket(left,right,history,
                            parameter_tol,max_iterations,predictor)
                if repair == max_repairs:
                    break
                best = min(history,key=lambda e:abs(e.gap))
                if best.slope:
                    proposed = best.lam-2*best.gap/best.slope
                else:
                    span = ordered[-1].lam-ordered[0].lam
                    proposed = (ordered[-1].lam+2*span if repair % 2 == 0
                                else ordered[0].lam-2*span)
                if proposed <= lower:
                    proposed = lower if lower > 0 else ordered[0].lam/2
                if upper is not None:
                    proposed = min(proposed,upper)
                if any(proposed == e.lam for e in history):
                    break
                history.append(self.evaluate(proposed))
            raise ValueError("new section gap did not rebracket within warm-start repair budget/bounds")

    def _refine_bracket(self,left,right,history,parameter_tol,max_iterations,predictor):
        current = min((left,right),key=lambda e:abs(e.gap))
        for _ in range(max_iterations):
            lo,hi = left.lam,right.lam
            if not current.gap:
                return current,tuple(history)
            correction = current.gap/current.slope if current.slope else None
            if correction is not None and abs(correction) <= decimal(parameter_tol):
                return current,tuple(history)
            if hi-lo <= decimal(parameter_tol):
                return current,tuple(history)
            if predictor == "hermite":
                proposed = _hermite_parameter(left,right)
            else:
                proposed = current.lam-correction if correction is not None else (lo+hi)/2
            if not lo < proposed < hi:
                proposed = (lo+hi)/2
            current = self.evaluate(proposed)
            history.append(current)
            if left.gap*current.gap <= 0:
                right = current
            else:
                left = current
        raise ArithmeticError("continuation root iteration budget exhausted")
