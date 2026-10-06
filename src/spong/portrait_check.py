"""Four-obligation ordinary portrait scrutiny, independent of construction.

The checker reuses the exact structural/contact auditor, not its cached ledger.
An explicit policy is experimental until separately qualified; the production
resolution verdict is never promoted by this module.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as Q
from math import isfinite

import numpy as np

from . import merge_tree, sturm, topology
from .credibility import level_normal_reference
from .trajectory_check import _eval, _derivative, inspect_trajectory


@dataclass(frozen=True)
class CredibilityPolicy:
    """Explicit accuracy targets; no thresholds inferred from a passing zoo."""
    max_sin_squared: float
    max_eigen_sin_squared: float
    max_binding_eps: float
    sample_budget: int = 250000

    def __post_init__(self):
        if (not 0 < self.max_sin_squared < 1
                or not 0 < self.max_eigen_sin_squared < 1
                or not isfinite(self.max_binding_eps) or self.max_binding_eps < 0
                or type(self.sample_budget) is not int or self.sample_budget < 1):
            raise ValueError("invalid credibility policy")


def _gradient(m, point):
    a,b = map(lambda x: Q(float(x)), point)
    A,B = _eval(m.alpha,b),_eval(m.beta,b)
    Ap,Bp = _eval(_derivative(m.alpha),b),_eval(_derivative(m.beta),b)
    return (2*(a*A-B),a*a*Ap-2*a*Bp)


def _ratio(v,w):
    nv=sum(x*x for x in v); nw=sum(x*x for x in w)
    return None if not nv or not nw else float((v[0]*w[1]-v[1]*w[0])**2/(nv*nw))


def check_launch(m, enumeration, branch):
    if not branch.pieces or branch.pieces[0].kind != "manifold_jet":
        return {"status":"unresolved","reason":"launch_representation"}
    matches=[(i,p) for i,p in enumerate(enumeration.points)
             if p.kind == "saddle" and np.array_equal(branch.Y[0],(p.a,p.b))]
    if len(matches)!=1:
        return {"status":"invalid","reason":"launch_center"}
    i,p=matches[0]; e=branch.pieces[0].evidence
    try:
        ca,cb=map(lambda x: Q(float(x)),e["center"])
        if (float(ca),float(cb)) != (float(p.a),float(p.b)):
            raise ValueError("center_mismatch")
        v=tuple(Q(float(c[1])) for c in e["coefficients"])
        if any(c[0] != 0 for c in e["coefficients"]):
            raise ValueError("nonzero_launch_offset")
        A,B=_eval(m.alpha,cb),_eval(m.beta,cb)
        Ap,Bp=_eval(_derivative(m.alpha),cb),_eval(_derivative(m.beta),cb)
        App,Bpp=_eval(_derivative(_derivative(m.alpha)),cb),_eval(_derivative(_derivative(m.beta)),cb)
        h00,h01,h11=2*A,2*ca*Ap-2*Bp,ca*ca*App-2*ca*Bpp
        Hv=(h00*v[0]+h01*v[1],h01*v[0]+h11*v[1])
        eigen=sum(x*y for x,y in zip(v,Hv))
        stable=branch.kind=="stable"
        orientation=(A*v[0]+(ca*Ap-Bp)*v[1]) if stable else v[1]
        sign=(orientation>0)-(orientation<0)
        declared=branch.diag.get("stable_sign" if stable else "unstable_direction")
        if not sign or sign!=declared or (eigen<=0 if stable else eigen>=0):
            return {"status":"invalid","reason":"launch_orientation_or_eigenspace"}
        return {"status":"measured","saddle":i,"orientation":sign,
                "eigen_sin_squared":_ratio(v,Hv),"critical_proxy":True}
    except (KeyError,IndexError,TypeError,ValueError,ZeroDivisionError):
        return {"status":"invalid","reason":"invalid_launch_evidence"}


def endpoint_tangent(m, branch):
    """Use the actual proposed smooth tangent, never set it equal to the field."""
    from .native_trace import decode, polynomial, physical
    last=len(branch.Y)-1
    for p in reversed(branch.pieces):
        if p.kind=="capture_connector":
            continue
        if p.stop!=last:
            return None
        if p.kind=="manifold_jet":
            e=p.evidence; t=Q(float(e["parameters"][p.stop-p.start]))
            return tuple(_eval(_derivative(tuple(Q(float(x)) for x in c)),t)
                         for c in e["coefficients"])
        if p.evidence.get("representation")=="native_dense_trace":
            e=p.evidence; dense,vertices,_=decode(e["trace"])
            source,t,_=vertices[last-e["branch_offset"]]
            if source<0: return None
            return physical(m,dense[source],polynomial(dense[source]),t)[1]
        return None
    return None


def check_end(m, enumeration, tree, branch, box):
    """Exact fate is separate from finite-boundary and connector geometry."""
    try:
        if branch.term=="capture" and branch.kind=="unstable":
            if len(branch.Y)<2:
                raise ValueError("missing_preconnector")
            matches=[i for i,p in enumerate(enumeration.points)
                     if p.kind=="min" and np.array_equal(branch.Y[-1],(p.a,p.b))]
            arrival = (branch.pieces[-1] if branch.pieces
                       and branch.pieces[-1].kind == "minimum_arrival" else None)
            entry = branch.Y[arrival.start] if arrival is not None else branch.Y[-2]
            forcing=merge_tree.widest_forcing_component(m,enumeration,tree,*map(float,entry))
            if len(matches)!=1 or forcing is None or forcing[1].minima != (matches[0],):
                return {"status":"unresolved","reason":"capture_not_forced"}
            if arrival is not None:
                from .arrival_check import check_target
                attachment = check_target(m,enumeration.points[matches[0]],arrival.evidence)
                return {"status":"measured","kind":"analytic_arrival","fate_exact":True,
                        "minimum":matches[0],"attachment":attachment}
            row,=level_normal_reference(m,branch.Y[-2:],-1)
            return {"status":"measured","kind":"capture","fate_exact":True,
                    "minimum":matches[0],"connector":row}
        if branch.term not in ("box_exit","level_bar"):
            return {"status":"unresolved","reason":"unfinished_branch"}
        tangent=endpoint_tangent(m,branch)
        if tangent is None:
            return {"status":"unresolved","reason":"boundary_tangent_unavailable"}
        g=_gradient(m,branch.Y[-1]); direction=1 if branch.kind=="stable" else -1
        alignment=direction*sum(x*y for x,y in zip(g,tangent))
        fate=False
        if branch.kind=="stable":
            level=merge_tree.exact_loss(m,*map(float,branch.Y[-1]))
            fate=all(merge_tree.value_sign(m,p,level)<0 for p in enumeration.points)
        elif branch.term=="box_exit":
            fate=merge_tree.widest_escape_component(m,enumeration,tree,*map(float,branch.Y[-1])) is not None
        if branch.term=="level_bar":
            outward=branch.kind=="stable" and alignment>0
            kind="loss_level_boundary"
        else:
            a,b=branch.Y[-1]; outward=False
            for d,low,high in ((0,box[0],box[1]),(1,box[2],box[3])):
                x=(a,b)[d]
                for sign,crossed in ((-1,x<low),(1,x>high)):
                    if crossed and sign*tangent[d]>0 and sign*direction*g[d]>0:
                        outward=True
            kind="finite_box_boundary"
        return {"status":"measured","kind":kind,"fate_exact":bool(fate),
                "outward":bool(outward),"flow_aligned":alignment>0,
                "sin_squared":_ratio(tangent,g)}
    except (KeyError,ValueError,TypeError,ArithmeticError,IndexError):
        return {"status":"unresolved","reason":"endpoint_evidence_failure"}


def assess(portrait, policy, *, measure=True):
    """Recompute all obligations; candidate acceptance is NOT production promotion."""
    m=portrait.model
    exact=sturm.enumerate_critical_points(m)
    supplied=portrait.enumeration
    signatures=lambda e: [(p.kind,p.source,float(p.a),float(p.b)) for p in e.points]
    if not exact.morse or signatures(exact)!=signatures(supplied):
        return {"status":"unresolved","reason":"critical_inventory","certified":False}
    # The existing exact/contact auditor is shared checker machinery. It is
    # rerun on the proposal; its old cached ledger is not accepted as evidence.
    top=topology.audit(m,supplied,portrait.branches,portrait.box)
    tree=merge_tree.build(m,exact)
    branches=[]; all_ok=top["status"]=="certified"
    for b in portrait.branches:
        launch=check_launch(m,exact,b)
        end=check_end(m,exact,tree,b,portrait.box)
        geometry=inspect_trajectory(m,b,measure_jets=measure,measure_native=measure,
                                    sample_budget=policy.sample_budget)
        issues=[]
        connectors = [p for p in b.pieces if p.kind == "capture_connector"]
        arrivals = [p for p in b.pieces if p.kind == "minimum_arrival"]
        if b.term == "capture" and arrivals:
            if (len(arrivals) != 1 or arrivals[0] is not b.pieces[-1]
                    or arrivals[0].stop != len(b.Y)-1 or connectors):
                issues.append("arrival_geometry")
        elif b.term == "capture":
            if (len(connectors) != 1 or connectors[0].stop != len(b.Y)-1
                    or connectors[0].start != len(b.Y)-2):
                issues.append("arrival_geometry")
        elif connectors or arrivals:
            issues.append("arrival_geometry")
        if launch["status"]!="measured" or launch.get("eigen_sin_squared") is None or launch["eigen_sin_squared"]>policy.max_eigen_sin_squared:
            issues.append("local_attachment")
        if end["status"]!="measured" or not end.get("fate_exact"):
            issues.append("endpoint_fate")
        elif end["kind"]=="analytic_arrival":
            attachment = end["attachment"]
            if (attachment["status"] != "measured" or not attachment.get("center_bound")
                    or not attachment.get("inward") or not attachment.get("positive_hessian")
                    or attachment.get("sin_squared") is None
                    or attachment["sin_squared"] > policy.max_eigen_sin_squared):
                issues.append("arrival_geometry")
        elif end["kind"]=="capture":
            c=end["connector"]
            if c["status"]!="measured" or c["flow_alignment"]!=1 or c["loss_direction"]!=1 or c["sin_squared"]>policy.max_sin_squared:
                issues.append("arrival_geometry")
        elif not end["outward"] or not end["flow_aligned"] or end["sin_squared"] is None or end["sin_squared"]>policy.max_sin_squared:
            issues.append("boundary_orientation")
        if geometry["coverage"]["status"]!="complete": issues.append("piece_coverage")
        for piece,r in zip(b.pieces,geometry["pieces"]):
            if piece.kind=="capture_connector": continue # checked above
            if piece.kind=="handoff":
                source=piece.evidence.get("from")
                if source is None:
                    issues.append("handoff_evidence")
                else:
                    gap=max(abs(float(x)-float(y))/(np.finfo(float).eps*(1+max(abs(float(x)),abs(float(y)))))
                            for x,y in zip(b.Y[piece.start],source))
                    if gap>policy.max_binding_eps: issues.append("handoff_gap")
                rows = level_normal_reference(m, b.Y[piece.start:piece.stop+1],
                                              1 if b.kind == "stable" else -1)
                if any(r["status"] != "measured" or r["flow_alignment"] != 1
                       or r["loss_direction"] != 1
                       or r["sin_squared"] > policy.max_sin_squared for r in rows):
                    issues.append("handoff_geometry")
                continue
            if (r["status"]!="measured" or r.get("max_sin_squared",1)>policy.max_sin_squared
                    or r.get("opposed_flow",0) or r.get("orthogonal_flow",0)
                    or r.get("zero_tangents",0) or r.get("stationary_samples",0)
                    or r.get("unvalidated_steps",0)
                    or r.get("binding_eps",0)>policy.max_binding_eps
                    or r.get("join_eps",0)>policy.max_binding_eps):
                issues.append("trajectory_geometry")
        all_ok &= not issues
        branches.append({"launch":launch,"end":end,"trajectory":geometry,
                         "issues":sorted(set(issues))})
    return {"status":"meets_candidate_policy" if all_ok else "unresolved",
            "certified":False,"policy":asdict(policy),"topology":{
                "status":top["status"],"reason":top["resolution_reason"]},
            "branches":branches,"scope":"ordinary_credibility_calibration"}
