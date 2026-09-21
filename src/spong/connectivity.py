"""Determine unstable-branch destinations before drawing a portrait.

Exact merge-tree inventories and validated local launches resolve the easy
branches. Ambiguous branches are transported only to decision sections; a
trapping tube, never a plotted point, is the witness for a refined inventory.
Rheostat diagnostics are NUMERICAL proposals and cannot promote a verdict.

This development API separates connectivity, finite-plane Smale exclusion,
and distance-to-wall evidence. An unresolved tube is not a proved nearby wall.
"""

from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace
import math
import time

import numpy as np

from . import _poly as P
from . import local_certificate, merge_tree, smale, sturm
from .charts import Branch


@dataclass(frozen=True)
class ConnectivityPolicy:
    transport: bool = True
    proposal_step: float = .01
    max_proposal_steps: int = 10000
    max_sections_per_branch: int = 8
    centre_limit: int = 192
    tube_options: dict = field(default_factory=lambda: {
        "max_inflations": 32, "max_slab_bisections": 8,
        "max_endpoint_bits": 16384})
    rheostat_diagnostics: bool = False
    max_rheostat_pairs: int = 4
    rheostat_steps: int = 64
    rheostat_radius: float = .05


@dataclass(frozen=True)
class BranchConnectivity:
    source: int
    direction: int
    orientation: int
    status: str
    minimum: int | None = None
    end: str | None = None
    candidate_minima: tuple[int, ...] = ()
    candidate_saddles: tuple[int, ...] = ()
    candidate_ends: tuple[str, ...] = ()
    witness: str | None = None
    level: Fraction | None = None
    sections: int = 0
    proposal_steps: int = 0
    reason: str | None = None
    proof: object | None = field(default=None,repr=False,compare=False)

    @property
    def certified(self):
        return self.status in ("capture", "escape")

    def as_dict(self):
        return {"source":self.source,"direction":self.direction,
                "orientation":self.orientation,"status":self.status,
                "certified":self.certified,"minimum":self.minimum,"end":self.end,
                "candidate_minima":self.candidate_minima,
                "candidate_saddles":self.candidate_saddles,
                "candidate_ends":self.candidate_ends,"witness":self.witness,
                "level":None if self.level is None else str(self.level),
                "sections":self.sections,"proposal_steps":self.proposal_steps,
                "reason":self.reason}


@dataclass(frozen=True)
class Connectivity:
    status: str
    enumeration: object
    branches: tuple[BranchConnectivity, ...]
    candidate_connections: tuple[tuple[int,int,int,int], ...]
    rheostat: tuple[dict, ...]
    finite_plane_smale: bool
    diagnostics: tuple[str, ...]
    elapsed_seconds: float

    @property
    def complete(self):
        return self.status == "certified_connectivity"

    def as_dict(self):
        return {"status":self.status,"engine_maturity":"development_oracle",
                "connectivity_certified":self.complete,
                "finite_plane_smale_certified":self.finite_plane_smale,
                "compactified_smale_certified":False,
                "rheostat_distance_lower_bound":None,
                "distance_scope":"Lambda direction only; numerical estimates are not bounds",
                "branches":[branch.as_dict() for branch in self.branches],
                "candidate_connections":self.candidate_connections,
                "rheostat":self.rheostat,"diagnostics":self.diagnostics,
                "elapsed_seconds":self.elapsed_seconds}


def _ends(component):
    return tuple(name for exists,name in (
        (component.lo is None,"b_minus_infinity"),
        (component.hi is None,"b_plus_infinity")) if exists)


def _decision(source,stub,component,witness,proof,**metadata):
    ends = _ends(component)
    capture = component.bounded and not component.saddles and len(component.minima)==1
    escape = not component.minima and not component.saddles and len(ends)==1
    return BranchConnectivity(source,stub.b_direction,stub.orientation,
        "capture" if capture else "escape" if escape else "unresolved",
        minimum=component.minima[0] if capture else None,
        end=ends[0] if escape else None,candidate_minima=component.minima,
        candidate_saddles=component.saddles,candidate_ends=ends,
        witness=witness,level=component.level,proof=proof,**metadata)


def _checked_components(m,e,level):
    if any(merge_tree.value_sign(m,p,level) not in (-1,1) for p in e.points):
        raise ValueError("component inventory has unresolved critical-value signs")
    return merge_tree.components_at(m,e,level)


def _launch_component(m,e,point,stub):
    """Use a short dyadic level between the exact launch and saddle losses.

    The native launch section can have thousands of coefficient bits. Raising
    its loss slightly preserves the local-component argument while avoiding
    that unnecessary height in a new level polynomial's Sturm chain.
    """
    launch = stub.validated_launch
    if launch is None or not launch.validated or launch.time_direction != -1:
        raise ValueError("local decreasing-loss launch is not validated")
    direction = smale._certified_b_direction(point,stub,launch)
    if not direction or direction != stub.b_direction:
        raise ValueError("local cone does not certify the branch direction")
    for bits in (32,48,64,96,128,192,256):
        scaled = launch.section_level*(1<<bits)
        level = Fraction(scaled.numerator//scaled.denominator+1,1<<bits)
        if merge_tree.value_sign(m,point,level) == 1:
            gap = merge_tree.critical_gap_index(m,level,point)+direction
            return next(c for c in _checked_components(m,e,level) if c.gap==gap)
    raise ValueError("no short dyadic launch upper level was separated")


def _component_from_tube(m,e,upper,tube):
    if tube is None or tube.status != "validated" or not tube.knots:
        raise ValueError("transport has no validated terminal interval")
    terminal = tube.knots[-1]
    if not terminal.level < upper:
        raise ValueError("terminal loss must be strictly below inventory level")
    interval = terminal.b_interval
    polynomial = merge_tree.level_polynomial(m,upper)
    lo = sturm.count_roots(polynomial,None,interval.lo)
    hi = sturm.count_roots(polynomial,None,interval.hi)
    if lo != hi or P.eval_at(polynomial,interval.lo)==0:
        raise ValueError("terminal enclosure overlaps a component boundary")
    return next(c for c in _checked_components(m,e,upper) if c.gap==lo)


def _lower_same_slab(m,e,upper):
    """A lower regular transport section with the same exact critical signs."""
    signs = tuple(merge_tree.value_sign(m,p,upper) for p in e.points)
    if any(s not in (-1,1) for s in signs):
        raise ValueError("inventory level has unresolved critical signs")
    step = Fraction(max(1.,abs(float(upper))))/16
    for _ in range(160):
        lower = upper-step
        if tuple(merge_tree.value_sign(m,p,lower) for p in e.points)==signs:
            return lower
        step /= 2
    raise ValueError("no lower rational section in the critical-value slab")


def _extend_proposal(m,point,stub,branch,target,policy):
    kernel = getattr(m,"_native_kernel",None)
    if kernel is None:
        raise ValueError("native section proposal integrator unavailable")
    points = list(branch.Y)
    a,b = points[-1]
    previous = float(m.L(a,b))
    stagnant = 0
    used = 0
    while previous >= float(target):
        if used >= policy.max_proposal_steps:
            raise ValueError("section proposal step budget exhausted")
        a,b = kernel.normalized_step(float(a),float(b),-policy.proposal_step,8)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ArithmeticError("nonfinite section proposal")
        current = float(m.L(a,b))
        stagnant = stagnant+1 if current >= previous else 0
        points.append((a,b))
        used += 1
        if stagnant >= 4:
            raise ValueError("section proposal stopped decreasing before requested level")
        previous = current
    # Fractional event rounding may put a nominal crossing on the wrong side;
    # the exact proposal-to-level routine later checks whether it really crossed.
    branch.Y = np.asarray(points,dtype=float)
    return used


def _rheostat_diagnostic(m,e,pair,policy):
    from .wall_native import NativeSectionContinuation
    source,direction,target,stable_direction = pair
    record = {"pair":pair,"grade":"NUMERICAL_ORACLE",
              "engine":"native_gmp","distance_lower_bound":None}
    try:
        c = NativeSectionContinuation(m,source_index=source,target_index=target,
            source_direction=direction,target_direction=stable_direction,
            steps=policy.rheostat_steps)
        evaluation = c.evaluate(1)
        record.update(gap=str(evaluation.gap),slope=str(evaluation.slope),
                      status="numerical_separation")
        radius = Fraction(str(policy.rheostat_radius))
        result = c.find_root(1-radius,1+radius)
        root = result.evaluation
        # Reporting only; all wall-location decisions were made in C.
        with localcontext() as ctx:
            ctx.prec = 80
            distance = abs(root.lam-Decimal(1))
        record.update(status="numerical_wall_candidate",wall_parameter=str(root.lam),
                      wall_distance_estimate=str(distance),evaluations=result.evaluations,
                      stage_solves=result.stage_solves,step_halvings=result.step_halvings,
                      max_backward_error=result.max_backward_error)
        return record
    except (ArithmeticError,ValueError,RuntimeError) as exc:
        return dict(record,status="unresolved",reason=str(exc))


def determine_connectivity(m, *, enumeration=None, policy=None):
    """Return a validated attaching map, or explicit remaining obligations.

    No portrait or full stable manifold is drawn. Local stubs and short
    section proposals are proof inputs; only exact/validated decisions can
    mark a destination. Rheostat diagnostics are optional and never proofs.
    """
    start = time.perf_counter()
    policy = policy or ConnectivityPolicy()
    if (not math.isfinite(policy.proposal_step) or policy.proposal_step <= 0
            or policy.max_sections_per_branch < 0 or policy.max_proposal_steps < 0
            or policy.centre_limit < 2 or policy.max_rheostat_pairs < 0
            or not 0 < policy.rheostat_radius < 1 or policy.rheostat_steps < 1):
        raise ValueError("invalid connectivity policy")
    e = enumeration
    if not sturm.is_positive(m.alpha):
        return Connectivity("model_hypothesis_failed",e,(),(),(),False,
                            ("requires A(b)>0 on the real line",),time.perf_counter()-start)
    e = e if e is not None else sturm.enumerate_critical_points(m)
    if not e.morse:
        return Connectivity("certified_non_morse",e,(),(),(),False,(),time.perf_counter()-start)
    if not e.saddles:
        return Connectivity("certified_connectivity",e,(),(),(),True,(),time.perf_counter()-start)
    tree = merge_tree.build(m,e)
    if any(not p.stubs for p in e.saddles):
        e = sturm.materialize_stubs(m,e)
    points = list(e.points)
    outcomes,diagnostics = [],[]
    for index,point in enumerate(points):
        if point.kind != "saddle":
            continue
        charts = {chart.manifold:chart for chart in point.local.poincare}
        stubs = []
        for stub in point.stubs:
            if stub.manifold != "unstable":
                stubs.append(stub)
                continue
            sections = used = 0
            outcome = BranchConnectivity(index,stub.b_direction,stub.orientation,"unresolved",
                candidate_minima=tuple(i for i,p in enumerate(e.points) if p.kind=="min"),
                candidate_saddles=tuple(i for i,p in enumerate(e.points) if p.kind=="saddle" and i!=index),
                candidate_ends=("b_minus_infinity","b_plus_infinity"))
            try:
                if stub.validated_launch is None:
                    launch = local_certificate.certify_poincare_launch_native(
                        m,point,charts["unstable"],stub.orientation,require_frobenius=False)
                    stub = replace(stub,validated_launch=launch)
                component = _launch_component(m,e,point,stub)
                outcome = _decision(index,stub,component,"validated_launch_component",
                                    stub.validated_launch)
                if not outcome.certified and policy.transport:
                    # A farther local graph section makes the near-saddle
                    # handoff tractable; retain the linear launch if it fails.
                    refined = local_certificate.certify_poincare_launch_native(
                        m,point,charts["unstable"],stub.orientation,
                        require_frobenius=True,max_reach_halvings=8,
                        max_slope_doublings=24)
                    if refined.validated:
                        stub = replace(stub,validated_launch=refined)
                    branch = Branch("unstable",np.array(stub.curve),"section_proposal",
                                    diag={"saddle_b":point.b,"unstable_direction":stub.b_direction})
                    proposal = SimpleNamespace(branches=[branch])
                    for upper in reversed(tree.levels):
                        if upper >= stub.validated_launch.section_level:
                            continue
                        if sections >= policy.max_sections_per_branch:
                            break
                        lower = _lower_same_slab(m,e,upper)
                        used += _extend_proposal(m,point,stub,branch,lower,policy)
                        sections += 1
                        tube,reason = smale._certify_half_sheet_tube(
                            m,proposal,point,stub,lower,policy.centre_limit,
                            policy.tube_options,False)
                        if tube is None or tube.status != "validated":
                            outcome = replace(outcome,reason=reason)
                            diagnostics.append(f"source {index}, branch {stub.orientation}, "
                                               f"section {sections}: {reason}")
                            continue
                        component = _component_from_tube(m,e,upper,tube)
                        outcome = _decision(index,stub,component,"validated_section_component",tube)
                        if outcome.certified:
                            break
            except (ArithmeticError,OverflowError,ValueError,RuntimeError,ImportError,StopIteration) as exc:
                outcome = replace(outcome,reason=str(exc) or type(exc).__name__)
            if not outcome.certified and outcome.reason is None:
                outcome = replace(outcome,reason="merge-tree inventory still has multiple possible destinations")
            outcomes.append(replace(outcome,sections=sections,proposal_steps=used))
            stubs.append(stub)
        points[index] = replace(point,stubs=tuple(stubs))
    e = replace(e,points=tuple(points))
    complete_inventory = all(
        sorted(b.orientation for b in outcomes if b.source==i)==[-1,1]
        for i,p in enumerate(e.points) if p.kind=="saddle")
    if not complete_inventory:
        diagnostics.append("complete two-branch unstable inventory unavailable")
    candidates = []
    for branch in outcomes:
        if branch.certified:
            continue
        for target in branch.candidate_saddles:
            if e.points[target].source == "B":
                continue  # level C is the roof; no descending saddle can reach it.
            _level,order = smale._common_level(tree.sequence,m,e.points[branch.source],e.points[target])
            if order < 0:
                continue
            candidates.extend((branch.source,branch.direction,target,d) for d in (-1,1))
    complete = complete_inventory and all(b.certified for b in outcomes)
    # With a complete launch-component inventory, all possible saddle targets
    # excluded is enough for finite-plane Smale, even if escape vs capture is open.
    valid_inventories = all(b.witness is not None for b in outcomes)
    finite_smale = complete or (complete_inventory and valid_inventories and not candidates)
    rheostat = tuple(_rheostat_diagnostic(m,e,pair,policy)
                    for pair in candidates[:policy.max_rheostat_pairs]) if policy.rheostat_diagnostics else ()
    return Connectivity("certified_connectivity" if complete else "unresolved",e,
                        tuple(outcomes),tuple(candidates),rheostat,finite_smale,
                        tuple(diagnostics),time.perf_counter()-start)
