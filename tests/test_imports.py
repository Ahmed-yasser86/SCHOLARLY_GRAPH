import importlib


def test_core_imports_succeed():
    modules = [
        "fastapi",
        "uvicorn",
        "httpx",
        "litellm",
        "sentence_transformers",
        "qdrant_client",
        "neo4j",
        "fitz",
        "pdfplumber",
        "pydantic",
        "pydantic_settings",
        "dotenv",
        "tenacity",
        "structlog",
        "nltk",
        "tiktoken",
        "rapidfuzz",
        "pycountry",
        "scholarly_graph.api.app",
        "scholarly_graph.application.use_cases",
        "scholarly_graph.domain.entities",
        "scholarly_graph.ingestion.discovery",
        "scholarly_graph.ingestion.chunking",
        "scholarly_graph.ingestion.sections",
        "scholarly_graph.ingestion.extractors",
        "scholarly_graph.extraction.claims",
        "scholarly_graph.extraction.normalize",
        "scholarly_graph.extraction.embeddings",
        "scholarly_graph.storage.vectors",
        "scholarly_graph.storage.graph_store",
        "scholarly_graph.retrieval.synthesis",
        "scholarly_graph.evaluation.questions",
        "scholarly_graph.llm.client",
    ]
    for name in modules:
        assert importlib.import_module(name) is not None
    print("all-core-imports-ok")
