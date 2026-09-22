"""Robust JSON-array parsing for LLM output: fences, repair, salvage."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _salvage_objects(text: str) -> list:
    objects: list = []
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def parse_json_array(text: str) -> list:
    stripped = _strip_fences(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        logger.warning("llm output parsed to non-list; returning []")
        return []
    except Exception:
        logger.warning("json parse failed, trying json_repair")
    try:
        import json_repair

        parsed = json_repair.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        logger.warning("json_repair parsed to non-list; returning []")
        return []
    except Exception as exc:
        logger.warning("json repair failed: %s", exc)
    salvaged = _salvage_objects(stripped)
    if salvaged:
        logger.warning("salvaged %d objects from malformed llm output", len(salvaged))
    return salvaged
