"""FastAPI application exposing ingestion, search, graph, and evaluation."""

from __future__ import annotations

import json
import os
import pathlib
import time

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

import structlog

from scholarly_graph.application.use_cases import (
    CompareRagUseCase,
    IngestPapersUseCase,
    NetworkAwareRagUseCase,
    StandardRagUseCase,
)
from scholarly_graph.evaluation.questions import EVALUATION_QUESTIONS, run_evaluation
from scholarly_graph.ingestion.discovery import SnapshotStore
from scholarly_graph.ingestion.extractors import PdfTextExtractor
from scholarly_graph.storage.graph_store import FileGraphStore
from scholarly_graph.storage.vectors import LocalVectorStore, QdrantVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
app_log = structlog.get_logger("scholarly_graph.api")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_tokens: int = 1500
    anthropic_api_key: str = ""
    core_api_key: str = ""
    semantic_scholar_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "scholarly_graph"
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    vector_path: str = "data/vector_chunks.json"
    graph_path: str = "data/graph.json"
    snapshot_dir: str = "data/snapshots"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"


def settings() -> Settings:
    return Settings()


_JOBS: dict = {}


class TransparencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.monotonic()
        app_log.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=dict(request.query_params),
        )
        response = await call_next(request)
        app_log.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_s=round(time.monotonic() - started, 3),
        )
        return response


class IngestRequest(BaseModel):
    query: str = Field(default="economic inequality intergenerational mobility")
    target_count: int = Field(default=100, ge=1, le=500)
    year_from: int = Field(default=2000)
    year_to: int = Field(default=2025)
    open_access_only: bool = Field(default=True)


class SearchRequest(BaseModel):
    question: str
    limit: int = Field(default=25, ge=1, le=100)
    country: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    study_design: str | None = None
    section: str | None = None

    def filters(self) -> dict:
        year: tuple | None = None
        if self.year_from is not None or self.year_to is not None:
            year = (self.year_from or 0, self.year_to or 9999)
        return {
            "country": self.country,
            "year": year,
            "study_design": self.study_design,
            "section": self.section,
        }


class MechanismRequest(BaseModel):
    subject: str
    obj: str = Field(alias="object")


def _vector_store(cfg: Settings):
    if cfg.qdrant_url:
        return QdrantVectorStore(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            collection=cfg.qdrant_collection,
        )
    return LocalVectorStore(cfg.vector_path)


def _graph_store(cfg: Settings):
    if cfg.neo4j_uri:
        from scholarly_graph.storage.graph_store import Neo4jGraphStore

        return Neo4jGraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    return FileGraphStore(cfg.graph_path)


def _services(cfg: Settings | None = None) -> dict:
    cfg = cfg or settings()
    return {"vector_store": _vector_store(cfg), "graph_store": _graph_store(cfg)}


def _llm_client(cfg: Settings):
    from scholarly_graph.llm.client import LlmClient, LlmConfig

    api_key = cfg.llm_api_key or cfg.anthropic_api_key
    if not api_key and not cfg.llm_base_url:
        return None
    return LlmClient(
        LlmConfig(
            model=cfg.llm_model,
            api_key=api_key,
            base_url=cfg.llm_base_url,
            max_tokens=cfg.llm_max_tokens,
        )
    )


def _claim_client(cfg: Settings):
    return _llm_client(cfg)


def _ingest_use_case(cfg: Settings | None = None) -> IngestPapersUseCase:
    from scholarly_graph.ingestion.discovery import OpenAlexClient, PdfDownloader

    cfg = cfg or settings()
    services = _services(cfg)
    return IngestPapersUseCase(
        discovery=OpenAlexClient(),
        downloader=PdfDownloader(),
        extractor=PdfTextExtractor(),
        claim_extractor_client=_claim_client(cfg),
        vector_store=services["vector_store"],
        graph_store=services["graph_store"],
        snapshots=SnapshotStore(cfg.snapshot_dir),
    )


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings()
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
        use_case = _ingest_use_case(cfg)
        store = SnapshotStore(cfg.snapshot_dir)
        snapshot_id = store.new_snapshot_id()
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
                _JOBS[result.snapshot_id] = {
                    "status": "complete",
                    "result": result.__dict__,
                }
            except Exception as exc:
                _JOBS[snapshot_id] = {"status": "failed", "error": str(exc)}

        background.add_task(_run)
        return {"snapshot_id": snapshot_id, "status": "running"}

    @app.get("/ingest/{snapshot_id}")
    def ingest_status(snapshot_id: str) -> dict:
        return _JOBS.get(snapshot_id, {"status": "unknown"})

    @app.post("/search/standard")
    def search_standard(payload: SearchRequest) -> dict:
        services = _services(cfg)
        return StandardRagUseCase(
            vector_store=services["vector_store"],
            synthesis_client=_claim_client(cfg),
        ).run(payload.question, limit=payload.limit, filters=payload.filters())

    @app.post("/search/network")
    def search_network(payload: SearchRequest) -> dict:
        services = _services(cfg)
        return NetworkAwareRagUseCase(
            vector_store=services["vector_store"],
            graph_store=services["graph_store"],
            synthesis_client=_claim_client(cfg),
        ).run(payload.question, limit=payload.limit, filters=payload.filters())

    @app.post("/search/compare")
    def search_compare(payload: SearchRequest) -> dict:
        services = _services(cfg)
        return CompareRagUseCase(
            standard=StandardRagUseCase(
                vector_store=services["vector_store"],
                synthesis_client=_claim_client(cfg),
            ),
            network=NetworkAwareRagUseCase(
                vector_store=services["vector_store"],
                graph_store=services["graph_store"],
                synthesis_client=_claim_client(cfg),
            ),
        ).run(payload.question)

    @app.post("/graph/mechanisms")
    def graph_mechanisms(payload: MechanismRequest) -> dict:
        services = _services(cfg)
        claims = services["graph_store"].mechanisms_between(
            payload.subject, payload.obj
        )
        return {"subject": payload.subject, "object": payload.obj, "claims": claims}

    @app.get("/graph/country/{country}")
    def graph_country(country: str) -> dict:
        services = _services(cfg)
        return {
            "country": country,
            "claims": services["graph_store"].country_subgraph(country),
        }

    @app.get("/graph/concepts")
    def graph_concepts() -> dict:
        from scholarly_graph.domain.concepts import CANONICAL_CONCEPTS

        return {"concepts": list(CANONICAL_CONCEPTS)}

    @app.get("/evaluation/questions")
    def evaluation_questions() -> dict:
        return {"count": len(EVALUATION_QUESTIONS), "questions": EVALUATION_QUESTIONS}

    @app.post("/evaluation/run")
    def evaluation_run(payload: dict) -> dict:
        return run_evaluation(
            payload.get("standard", []), payload.get("network_aware", [])
        )

    @app.get("/evaluation/export")
    def evaluation_export() -> JSONResponse:
        return JSONResponse(
            {"count": len(EVALUATION_QUESTIONS), "questions": EVALUATION_QUESTIONS}
        )

    @app.get("/corpus/status")
    def corpus_status() -> dict:
        manifests = sorted(pathlib.Path(cfg.snapshot_dir).glob("*.json"))
        latest = None
        papers = 0
        if manifests:
            latest = json.loads(manifests[-1].read_text(encoding="utf-8"))
            papers = latest.get("paper_count", 0)
        return {
            "snapshots": len(manifests),
            "latest": latest,
            "papers": papers,
            "validation": {"status": "pending", "accuracy": None},
        }

    return app


app = create_app()
