"""Abstract repository and service ports."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PaperRepository(ABC):
    @abstractmethod
    def save(self, paper: Any) -> None: ...

    @abstractmethod
    def get(self, document_id: Any) -> Any | None: ...

    @abstractmethod
    def with_full_text(self) -> list: ...

    @abstractmethod
    def count(self) -> int: ...


class VectorStore(ABC):
    @abstractmethod
    def collection_exists(self, name: str) -> bool: ...

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, chunk_id: str, vector: list, payload: dict) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: list, limit: int = 25, filters: dict | None = None
    ) -> list: ...


class GraphStore(ABC):
    @abstractmethod
    def save_paper(self, paper: Any) -> None: ...

    @abstractmethod
    def save_claim(self, paper_id: Any, claim: Any) -> None: ...

    @abstractmethod
    def save_citation(
        self, citing_id: Any, cited_id: Any, role: str = "neutral"
    ) -> None: ...

    @abstractmethod
    def mechanisms_between(self, subject: str, obj: str) -> list: ...

    @abstractmethod
    def country_subgraph(self, country: str) -> list: ...


class TextExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: str) -> dict: ...

    @abstractmethod
    def quality_score(self, text: str) -> float: ...


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, paper: Any, chunk_text: str, section: str) -> list: ...
