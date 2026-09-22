"""Section detection for academic papers.

Recognises numbered headings, case variations, and the multi-word method /
results variants from the plan, plus literature-review, data, empirical,
findings, and concluding sections. Headings may carry numeric prefixes
(``2. Methods``), trailing punctuation, or parenthetical notes, and must be a
single line below a length cap so body text is never misclassified.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MAX_HEADING_CHARS = 80

_SECTION_ALIASES: dict = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "background", "literature review"),
    "methods": (
        "method",
        "methods",
        "methodology",
        "data and methods",
        "empirical strategy",
        "research design",
        "data",
        "study design",
        "materials and methods",
    ),
    "results": (
        "result",
        "results",
        "finding",
        "findings",
        "empirical results",
        "analysis",
    ),
    "discussion": ("discussion",),
    "conclusion": ("conclusion", "conclusions", "concluding remarks"),
}

_PATTERNS: dict = {}
for _section, _aliases in _SECTION_ALIASES.items():
    _PATTERNS[_section] = [
        re.compile(
            r"^\s*(?:\d+(?:\.\d+)*[\.\)\s:|-]+)?"
            + re.escape(alias)
            + r"(?:\s*[:\-–—]\s*.*|\s*\(\d+\)|\s*\d+)?\s*$",
            re.IGNORECASE,
        )
        for alias in _aliases
    ]


def _is_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None
    if stripped.endswith((".", "?", "!")) and len(stripped) > 30:
        return None
    for section, patterns in _PATTERNS.items():
        if any(pattern.match(stripped) for pattern in patterns):
            return section
    return None


def detect_sections(text: str) -> list:
    sections: list = []
    current = "unknown"
    buffer: list = []
    for line in text.splitlines():
        matched = _is_heading(line)
        if matched is not None:
            if buffer:
                sections.append((current, "\n".join(buffer)))
            logger.debug("detected section heading: %r -> %s", line.strip(), matched)
            current = matched
            buffer = []
            continue
        buffer.append(line)
    if buffer or not sections:
        sections.append((current, "\n".join(buffer)))
    return [(name, body) for name, body in sections if body.strip()]
