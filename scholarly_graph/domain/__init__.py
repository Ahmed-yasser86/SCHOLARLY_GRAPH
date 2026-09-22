"""Domain services: query decomposition, evidence fusion, contradiction detection."""
from scholarly_graph.domain import ports
from scholarly_graph.domain.services import (
    ContradictionPair,
    FusedEvidence,
    StructuredQuery,
    detect_contradictions,
    decompose_query,
    fuse_evidence,
)

__all__ = [
    "ports",
    "ContradictionPair",
    "FusedEvidence",
    "StructuredQuery",
    "detect_contradictions",
    "decompose_query",
    "fuse_evidence",
]
