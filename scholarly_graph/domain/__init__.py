"""Domain services: query decomposition, evidence fusion, contradiction detection."""
from scholarly_graph.domain import ports, query
from scholarly_graph.domain.evidence import (
    ContradictionPair,
    FusedEvidence,
    claim_rank_score,
    detect_contradictions,
    fuse_evidence,
)
from scholarly_graph.domain.ports import (
    ClaimExtractor,
    GraphStore,
    PaperRepository,
    TextExtractor,
    VectorStore,
)
from scholarly_graph.domain.query import StructuredQuery, decompose_query

__all__ = [
    "ClaimExtractor",
    "ContradictionPair",
    "FusedEvidence",
    "GraphStore",
    "PaperRepository",
    "StructuredQuery",
    "TextExtractor",
    "VectorStore",
    "claim_rank_score",
    "decompose_query",
    "detect_contradictions",
    "fuse_evidence",
    "ports",
    "query",
]
