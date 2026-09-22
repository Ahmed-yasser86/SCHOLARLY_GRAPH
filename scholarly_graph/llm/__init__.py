"""LLM provider package: generic interface over any configured model."""
from scholarly_graph.llm.client import DEFAULT_MODEL, LlmClient, LlmConfig, LlmResponse

__all__ = ["DEFAULT_MODEL", "LlmClient", "LlmConfig", "LlmResponse"]
