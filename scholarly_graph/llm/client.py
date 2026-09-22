"""Generic chat-completion interface: any LiteLLM model string via configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1500


@dataclass
class LlmConfig:
    model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    extra: dict = field(default_factory=dict)


@dataclass
class LlmResponse:
    text: str
    model: str
    input_chars: int = 0


def _litellm_kwargs(config: LlmConfig) -> dict:
    kwargs: dict = {"model": config.model, "max_tokens": config.max_tokens}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["api_base"] = config.base_url
    kwargs.update(config.extra)
    return kwargs


class LlmClient:
    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig()

    def complete(self, system: str, user: str) -> LlmResponse:
        import litellm

        logger.info(
            "llm request model=%s system=%d chars user=%d chars",
            self.config.model,
            len(system),
            len(user),
        )
        raw = litellm.completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **_litellm_kwargs(self.config),
        )
        text = raw.choices[0].message.content or ""
        logger.info("llm response model=%s %d chars", self.config.model, len(text))
        return LlmResponse(text=text, model=self.config.model, input_chars=len(user))

    def extract_json_array(self, system: str, user: str) -> list:
        import json

        from scholarly_graph.extraction.json_util import parse_json_array

        response = self.complete(system, user)
        parsed = parse_json_array(response.text)
        logger.info("llm json parsed %d items model=%s", len(parsed), response.model)
        return parsed

    def close(self) -> None:
        return None
