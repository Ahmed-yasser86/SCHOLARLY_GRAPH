"""Application use cases orchestrating ingestion and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_QUERY = (
    "economic inequality intergenerational mobility social mobility income inequality"
)


@dataclass
class IngestResult:
    snapshot_id: str
    discovered: int = 0
    downloaded: int = 0
    processed: int = 0
    claims: int = 0
    log: list = field(default_factory=list)


def _log(entries: list, kind: str, message: str) -> None:
    entries.append({"type": kind, "message": message})
    logger.info("%s: %s", kind, message)


@dataclass
class IngestPapersUseCase:
    discovery: object
    downloader: object
    extractor: object
    claim_extractor_client: object | None
    vector_store: object
    graph_store: object
    snapshots: object
    pdf_dir: str = "data/pdfs"

    def run(
        self,
        query: str = DEFAULT_DOMAIN_QUERY,
        target_count: int = 100,
        year_from: int = 2000,
        year_to: int = 2025,
        open_access_only: bool = True,
    ) -> IngestResult:
        import pathlib

        from scholarly_graph.domain.entities import DocumentId, Paper
        from scholarly_graph.extraction.claims import extract_claims_from_chunk
        from scholarly_graph.extraction.normalize import ConceptNormalizer
        from scholarly_graph.ingestion.chunking import chunk_paper
        from scholarly_graph.storage.vectors import embed_text

        entries: list = []
        snapshot_id = self.snapshots.new_snapshot_id()
        _log(entries, "info", f"Starting ingestion snapshot {snapshot_id}")
        try:
            papers = self.discovery.collect(
                query,
                target_count=target_count,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            )
        except Exception as exc:
            _log(entries, "error", f"Discovery failed: {exc}")
            raise
        _log(entries, "success", f"Discovered {len(papers)} papers")
        normalizer = ConceptNormalizer()
        processed_records: list = []
        claim_total = 0
        downloaded = 0
        for paper_dict in papers:
            title = paper_dict.get("title") or "Untitled"
            try:
                paper = Paper(
                    document_id=DocumentId(paper_dict["document_id"]),
                    title=title,
                    authors=tuple(paper_dict.get("authors", [])),
                    year=int(paper_dict.get("year") or 2000),
                    journal=paper_dict.get("journal", ""),
                    doi=paper_dict.get("doi", ""),
                    open_access=bool(paper_dict.get("open_access")),
                    pdf_url=paper_dict.get("pdf_url", ""),
                    data_countries=tuple(paper_dict.get("data_countries", [])),
                    snapshot_id=snapshot_id,
                )
            except Exception as exc:
                _log(entries, "warning", f"Skipping paper with bad metadata: {exc}")
                continue
            _log(entries, "info", f"Retrieved paper: {paper.title} ({paper.year})")
            self.graph_store.save_paper(paper)
            processed_records.append(paper_dict)
            if not paper.pdf_url:
                _log(entries, "skip", f"No open-access PDF for {paper.title}")
                continue
            safe_name = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_"
                for ch in str(paper.document_id)
            )
            destination = str(pathlib.Path(self.pdf_dir) / f"{safe_name}.pdf")
            outcome = self.downloader.download(paper.pdf_url, destination)
            if outcome.get("status") not in {"success", "cached"}:
                _log(entries, "skip", f"PDF unavailable for {paper.title}")
                continue
            downloaded += 1
            try:
                extraction = self.extractor.extract(destination)
            except Exception as exc:
                _log(entries, "warning", f"Text extraction failed: {exc}")
                continue
            _log(entries, "info", f"Quality score: {extraction['quality']:.2f}")
            if not extraction.get("processable"):
                _log(entries, "skip", "Excluded for low PDF quality")
                continue
            chunks = chunk_paper(
                str(paper.document_id),
                extraction["text"],
                metadata={"paper": paper.title, "year": paper.year},
            )
            for chunk in chunks:
                self.vector_store.upsert(
                    chunk.chunk_id,
                    embed_text(chunk.text),
                    {
                        "text": chunk.text,
                        "section": chunk.section,
                        "paper": paper.title,
                        "year": paper.year,
                        "collection": "scholarly_graph",
                    },
                )
                if self.claim_extractor_client is None:
                    continue
                for claim in extract_claims_from_chunk(
                    chunk.text,
                    chunk.section,
                    paper_id=str(paper.document_id),
                    client=self.claim_extractor_client,
                    normalizer=normalizer,
                ):
                    self.graph_store.save_claim(paper.document_id, claim)
                    claim_total += 1
            _log(entries, "success", f"Processed {paper.title}: {len(chunks)} chunks")
        manifest_path = self.snapshots.save_manifest(
            snapshot_id,
            {"query": query, "papers": processed_records, "claim_count": claim_total},
        )
        _log(entries, "success", f"Snapshot manifest saved to {manifest_path}")
        return IngestResult(
            snapshot_id=snapshot_id,
            discovered=len(papers),
            downloaded=downloaded,
            processed=len(processed_records),
            claims=claim_total,
            log=entries,
        )


@dataclass
class StandardRagUseCase:
    vector_store: object
    synthesis_client: object | None = None

    def run(self, question: str, limit: int = 25, filters: dict | None = None) -> dict:
        from scholarly_graph.storage.vectors import embed_text

        query_vector = embed_text(question)
        hits = self.vector_store.search(query_vector, limit=limit, filters=filters)
        papers = sorted({hit["payload"].get("paper", "") for hit in hits} - {""})
        answer = ""
        if self.synthesis_client is not None:
            from scholarly_graph.retrieval.synthesis import synthesize_standard

            answer = synthesize_standard(
                question, hits, client=self.synthesis_client
            )
        return {
            "mode": "standard",
            "question": question,
            "answer": answer,
            "chunks_retrieved": len(hits),
            "papers": papers,
            "sources": hits,
            "trace": [
                "Embedded query with BAAI/bge-small-en-v1.5",
                f"Retrieved {len(hits)} chunks from {len(papers)} papers",
                "Synthesis grounded only in retrieved passages",
            ],
        }


@dataclass
class NetworkAwareRagUseCase:
    vector_store: object
    graph_store: object
    synthesis_client: object | None = None

    def run(self, question: str, limit: int = 25, filters: dict | None = None) -> dict:
        from scholarly_graph.domain.evidence import (
            detect_contradictions,
            fuse_evidence,
        )
        from scholarly_graph.domain.query import decompose_query
        from scholarly_graph.storage.vectors import embed_text

        structured = decompose_query(question)
        hits = self.vector_store.search(
            embed_text(question), limit=limit, filters=filters
        )
        graph_claims: list = []
        if structured.subject_concept and structured.object_concept:
            graph_claims = self.graph_store.mechanisms_between(
                structured.subject_concept, structured.object_concept
            )
        graph_claims = [c for c in graph_claims if hasattr(c, "subject")]
        fused = fuse_evidence(graph_claims)
        contradictions = detect_contradictions(fused.claims)
        countries: dict = {}
        for claim in fused.claims:
            for country in getattr(claim, "country_scope", ()):
                countries[country] = countries.get(country, 0) + 1
        answer = ""
        if self.synthesis_client is not None:
            from scholarly_graph.retrieval.synthesis import synthesize_network_aware

            answer = synthesize_network_aware(
                question, fused.claims, contradictions, client=self.synthesis_client
            )
        return {
            "mode": "network_aware",
            "question": question,
            "answer": answer,
            "structured_query": {
                "subject": structured.subject_concept,
                "object": structured.object_concept,
                "type": structured.question_type,
                "countries": list(structured.countries),
            },
            "vector_hits": hits,
            "mechanisms": [
                {
                    "subject": c.subject,
                    "relationship": c.relationship,
                    "object": c.obj,
                    "confidence": c.confidence,
                    "countries": list(c.country_scope),
                    "evidence": c.evidence.text,
                }
                for c in fused.claims
            ],
            "contradictions": [
                {
                    "subject": pair.subject,
                    "object": pair.obj,
                    "first": pair.first.plain_language(),
                    "second": pair.second.plain_language(),
                }
                for pair in contradictions
            ],
            "geography": countries,
            "trace": [
                f"Decomposed query as {structured.question_type}",
                f"Vector retrieval returned {len(hits)} chunks",
                f"Graph traversal returned {len(graph_claims)} claims",
                f"Fused to {len(fused.claims)} ranked claims",
                f"Detected {len(contradictions)} contradictions",
            ],
        }


@dataclass
class CompareRagUseCase:
    standard: StandardRagUseCase
    network: NetworkAwareRagUseCase

    def run(self, question: str) -> dict:
        standard = self.standard.run(question)
        network = self.network.run(question)
        return {
            "question": question,
            "standard": standard,
            "network_aware": network,
            "metrics": {
                "chunks_retrieved": standard["chunks_retrieved"],
                "vector_hits_network": len(network["vector_hits"]),
                "mechanisms_identified": len(network["mechanisms"]),
                "contradictions_detected": len(network["contradictions"]),
                "distinct_papers": len(standard["papers"]),
            },
        }
