"""Domain entities for the scholarly graph.

Standard library only. No third-party imports are permitted in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

VALID_SECTIONS = frozenset(
    {
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "unknown",
    }
)

VALID_RELATIONSHIP_TYPES = frozenset(
    {
        "increases",
        "decreases",
        "associated_with",
        "mediates",
        "moderates",
        "causes",
        "no_significant_effect",
        "contradicts",
    }
)

VALID_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})


@dataclass(frozen=True)
class ClaimConfidence:
    level: str = "medium"

    def __post_init__(self) -> None:
        if self.level not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"Unknown confidence {self.level!r}; "
                f"expected one of {sorted(VALID_CONFIDENCE_LEVELS)}"
            )

    def __str__(self) -> str:
        return self.level

VALID_STUDY_DESIGNS = frozenset(
    {
        "rct",
        "quasi_experimental",
        "observational",
        "meta_analysis",
        "simulation",
        "theoretical",
    }
)

MAX_EVIDENCE_WORDS = 500


@dataclass(frozen=True)
class DocumentId:
    """Identifier in the format ``source:id`` (e.g. ``openalex:W2741809807``)."""

    value: str

    def __post_init__(self) -> None:
        parts = self.value.split(":", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                f"DocumentId must look like 'source:id', got {self.value!r}"
            )

    @property
    def source(self) -> str:
        return self.value.split(":", 1)[0]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EvidenceSpan:
    """Verbatim excerpt grounding a claim. Immutable."""

    text: str
    section: str = "unknown"

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("EvidenceSpan text must be non-empty")
        if len(self.text.split()) > MAX_EVIDENCE_WORDS:
            raise ValueError(
                "EvidenceSpan exceeds 500 words; it must be a verbatim excerpt, "
                "not a paraphrase of the whole paper"
            )
        if self.section not in VALID_SECTIONS:
            raise ValueError(
                f"Unknown section {self.section!r}; "
                f"expected one of {sorted(VALID_SECTIONS)}"
            )


@dataclass(frozen=True)
class Paper:
    """A scholarly document in the corpus."""

    document_id: DocumentId
    title: str
    authors: tuple = ()
    year: int = 2000
    journal: str = ""
    doi: str = ""
    sources: tuple = ()
    open_access: bool = False
    pdf_url: str = ""
    data_countries: tuple = ()
    claim_scope: tuple = ()
    study_design: str = "observational"
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("Paper title must be non-empty")
        current_year = datetime.now().year + 1
        if self.year < 1900 or self.year > current_year:
            raise ValueError(f"Paper year {self.year} is implausible")
        if self.study_design not in VALID_STUDY_DESIGNS:
            raise ValueError(
                f"Unknown study design {self.study_design!r}; "
                f"expected one of {sorted(VALID_STUDY_DESIGNS)}"
            )
        object.__setattr__(self, "authors", tuple(self.authors))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "data_countries", tuple(self.data_countries))
        object.__setattr__(self, "claim_scope", tuple(self.claim_scope))


@dataclass(frozen=True)
class Claim:
    """A structured assertion about a relationship between two concepts."""

    subject: str
    relationship: str
    obj: str
    evidence: EvidenceSpan
    paper_id: DocumentId | None = None
    conditions: str | None = None
    country_scope: tuple = ()
    time_period: str = ""
    confidence: str = "medium"

    def __post_init__(self) -> None:
        if not self.subject or not self.subject.strip():
            raise ValueError("Claim subject must be non-empty")
        if not self.obj or not self.obj.strip():
            raise ValueError("Claim object must be non-empty")
        if self.relationship not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(
                f"Unknown relationship {self.relationship!r}; "
                f"expected one of {sorted(VALID_RELATIONSHIP_TYPES)}"
            )
        if self.confidence not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"Unknown confidence {self.confidence!r}; "
                f"expected one of {sorted(VALID_CONFIDENCE_LEVELS)}"
            )
        object.__setattr__(self, "country_scope", tuple(self.country_scope))

    def plain_language(self) -> str:
        condition = f", {self.conditions}" if self.conditions else ""
        return (
            f"{self.subject} {self.relationship.replace('_', ' ')} "
            f"{self.obj}{condition}".strip()
        )


@dataclass(frozen=True)
class CorpusSnapshot:
    """Manifest record for one ingestion run."""

    snapshot_id: str
    query: str = ""
    created_utc: str = ""
    paper_count: int = 0
    open_access_count: int = 0
    papers: tuple = field(default_factory=tuple)
