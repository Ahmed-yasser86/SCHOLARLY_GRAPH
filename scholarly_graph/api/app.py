"""FastAPI application exposing ingestion, search, graph, and evaluation."""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from scholarly_graph.application.use_cases import (
    CompareRagUseCase,
    IngestPapersUseCase,
    NetworkAwareRagUseCase,
    StandardRagUseCase,
)
from scholarly_graph.evaluation.questions import EVALUATION_QUESTIONS
from scholarly_graph.ingestion.chunking import PdfTextExtractor
from scholarly_graph.ingestion.discovery import (
    CoreClient,
    OpenAlexClient,
    PdfDownloader,
    SemanticScholarClient,
    SnapshotStore,
)
from scholarly_graph.storage.graph_store import FileGraphStore
from scholarly_graph.storage.vectors import LocalVectorStore

_JOBS: dict = {}


class TransparencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.monotonic()
        print(f"--> {request.method} {request.url.path}")
        response = await call_next(request)
        elapsed = time.monotonic() - started
        print(f"<-- {request.method} {request.url.path} {response.status_code} {elapsed:.3f}s")
        return response


class IngestRequest(BaseModel):
    query: str = Field(default="economic inequality intergenerational mobility")
    target_count: int = Field(default=10, ge=1, le=500)
    year_from: int = Field(default=2000)
    year_to: int = Field(default=2025)
    open_access_only: bool = Field(default=True)


class SearchRequest(BaseModel):
    question: str
    limit: int = Field(default=25, ge=1, le=100)


class MechanismRequest(BaseModel):
    subject: str
    obj: str = Field(alias="object")


def _services() -> dict:
    vector_store = LocalVectorStore(
        os.environ.get("VECTOR_PATH", "data/vector_chunks.json")
    )
    graph_store = FileGraphStore(os.environ.get("GRAPH_PATH", "data/graph.json"))
    return {"vector_store": vector_store, "graph_store": graph_store}


def _ingest_use_case() -> IngestPapersUseCase:
    services = _services()
    return IngestPapersUseCase(
        discovery=OpenAlexClient(),
        downloader=PdfDownloader(),
        extractor=PdfTextExtractor(),
        claim_extractor_client=None,
        vector_store=services["vector_store"],
        graph_store=services["graph_store"],
        snapshots=SnapshotStore(os.environ.get("SNAPSHOT_DIR", "data/snapshots")),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Scholarly Graph", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TransparencyMiddleware)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ingest")
    def start_ingest(payload: IngestRequest, background: BackgroundTasks) -> dict:
        use_case = _ingest_use_case()
        snapshot_id = SnapshotStore(
            os.environ.get("SNAPSHOT_DIR", "data/snapshots")
        ).new_snapshot_id()
        _JOBS[snapshot_id] = {"status": "running", "result": None}

        def _run() -> None:
            try:
                result = use_case.run(
                    query=payload.query,
                    target_count=payload.target_count,
                    year_from=payload.year_from,
                    year_to=payload.year_to,
                    open_access_only=payload.open_access_only,
                )
                _JOBS[result.snapshot_id] = {"status": "complete", "result": result.__dict__}
            except Exception as exc:  # noqa: BLE001
                _JOBS[snapshot_id] = {"status": "failed", "error": str(exc)}

        background.add_task(_run)
        return {"snapshot_id": snapshot_id, "status": "running"}

    @app.get("/ingest/{snapshot_id}")
    def ingest_status(snapshot_id: str) -> dict:
        return _JOBS.get(snapshot_id, {"status": "unknown"})

    @app.post("/search/standard")
    def search_standard(payload: SearchRequest) -> dict:
        services = _services()
        return StandardRagUseCase(vector_store=services["vector_store"]).run(
            payload.question, limit=payload.limit
        )

    @app.post("/search/network")
    def search_network(payload: SearchRequest) -> dict:
        services = _services()
        return NetworkAwareRagUseCase(
            vector_store=services["vector_store"], graph_store=services["graph_store"]
        ).run(payload.question, limit=payload.limit)

    @app.post("/search/compare")
    def search_compare(payload: SearchRequest) -> dict:
        services = _services()
        return CompareRagUseCase(
            standard=StandardRagUseCase(vector_store=services["vector_store"]),
            network=NetworkAwareRagUseCase(
                vector_store=services["vector_store"],
                graph_store=services["graph_store"],
            ),
        ).run(payload.question)

    @app.post("/graph/mechanisms")
    def graph_mechanisms(payload: MechanismRequest) -> dict:
        services = _services()
        claims = services["graph_store"].mechanisms_between(
            payload.subject, payload.obj
        )
        return {"subject": payload.subject, "object": payload.obj, "claims": claims}

    @app.get("/graph/country/{country}")
    def graph_country(country: str) -> dict:
        services = _services()
        return {"country": country, "claims": services["graph_store"].country_subgraph(country)}

    @app.get("/graph/concepts")
    def graph_concepts() -> dict:
        from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS

        return {"concepts": list(CANONICAL_CONCEPTS)}

    @app.get("/evaluation/questions")
    def evaluation_questions() -> dict:
        return {"count": len(EVALUATION_QUESTIONS), "questions": EVALUATION_QUESTIONS}

    @app.get("/corpus/status")
    def corpus_status() -> dict:
        import glob
        import json

        manifests = sorted(glob.glob("data/snapshots/*.json"))
        latest = None
        if manifests:
            latest = json.loads(open(manifests[-1], encoding="utf-8").read())
        return {"snapshots": len(manifests), "latest": latest}

    return app


app = create_app()
