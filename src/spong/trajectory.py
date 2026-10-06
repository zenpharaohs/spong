"""Trajectory evidence transport; no model evaluation or acceptance policy.

Piece ranges are half-open CHORD ranges: [start, stop) owns vertices
Y[start:stop+1]. A complete inventory partitions every chord exactly once.
Evidence describes the construction, never a certificate of its correctness.
"""
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class TrajectoryPiece:
    kind: str
    start: int
    stop: int
    evidence: dict = field(default_factory=dict)

    def summary(self):
        return {"kind": self.kind, "start": self.start, "stop": self.stop,
                "representation": self.evidence.get("representation", "polyline")}


class PieceRecorder:
    """Record assembly without changing any emitted coordinate."""

    def __init__(self):
        self.pieces = []

    def add(self, kind, start, stop, evidence=None):
        if stop > start:
            self.pieces.append(TrajectoryPiece(kind, start, stop, evidence or {}))

    def truncate(self, stop):
        self.pieces = [replace(p, stop=min(p.stop, stop))
                       for p in self.pieces if p.start < stop]

    def append(self, points, segment, kind, *, term=None, evidence=None):
        """Append with the dispatcher's existing shared-vertex convention.

        If the two representations disagree at the shared vertex, its first
        outgoing chord is a handoff, not silently part of the smooth chart.
        Capture's last chord is conservatively a drawing connector.
        """
        start = max(0, len(points)-1)
        mismatch = bool(points and len(segment) > 1
                        and not np.array_equal(points[-1], segment[0]))
        points.extend(segment[1:] if points else segment)
        stop = len(points)-1
        if evidence and evidence.get("representation") == "native_dense_trace":
            evidence = {**evidence, "branch_offset": start,
                        "source_points": len(segment)}
        connector = term == "capture" and stop > start
        body_stop = stop-int(connector)
        if mismatch and body_stop > start:
            self.add("handoff", start, start+1,
                     {"from": tuple(segment[0]), "to_kind": kind})
            start += 1
        self.add(kind, start, body_stop, evidence)
        if connector:
            self.add("capture_connector", stop-1, stop)


def launch_evidence(stub):
    """Use the representation retained by the actual selected stub solve."""
    if stub is None:
        return "linear_launch", {}
    evidence = getattr(stub, "trajectory_evidence", None) or {}
    certs = dict(stub.certificates)
    if certs.get("stub_jet"):
        kind = "manifold_jet"
    elif certs.get("centered_extension"):
        kind = "centered_grid_extension"
    elif certs.get("poincare_conditioned"):
        kind = "poincare_grid"
    else:
        kind = "centered_grid"
    return kind, evidence


def trace_evidence(diag):
    blob = diag.pop("_last_trace", None)
    if blob is None:
        return {}
    return {"representation": "native_dense_trace", "schema": 1, "trace": blob}
