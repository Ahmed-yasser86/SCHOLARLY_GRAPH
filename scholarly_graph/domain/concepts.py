"""Canonical concept vocabulary for the inequality-mobility domain.

Loaded from ``vocabulary.yaml`` so domain knowledge stays out of application
logic. Falls back to the bundled defaults when the YAML file is absent.
"""

from __future__ import annotations

import pathlib

_DEFAULT_CONCEPTS: tuple = (
    "economic inequality",
    "income inequality",
    "wealth inequality",
    "intergenerational mobility",
    "social mobility",
    "educational inequality",
    "educational attainment",
    "residential segregation",
    "neighborhood effects",
    "social capital",
    "wealth concentration",
    "capital accumulation",
    "labor market segmentation",
    "political institutions",
    "redistribution",
    "public education",
    "school quality",
    "parental income",
    "child earnings",
    "earnings elasticity",
    "gini coefficient",
    "great gatsby curve",
    "opportunity",
    "upward mobility",
    "downward mobility",
    "poverty",
    "affluence",
    "elite universities",
    "private schooling",
    "tutoring",
    "school funding",
    "property taxes",
    "racial segregation",
    "professional networks",
    "recommendation letters",
    "hiring",
    "credential barriers",
    "vocational training",
    "informal labor markets",
    "minimum wage",
    "labor protections",
    "taxation of capital",
    "public investment",
    "healthcare access",
    "early childhood",
    "college completion",
    "earnings",
    "wealth transfers",
    "inheritance",
    "other",
)

_DEFAULT_CATEGORIES: dict = {
    "inequality": (
        "economic inequality",
        "income inequality",
        "wealth inequality",
        "gini coefficient",
        "poverty",
        "affluence",
    ),
    "mobility": (
        "intergenerational mobility",
        "social mobility",
        "upward mobility",
        "downward mobility",
        "opportunity",
        "great gatsby curve",
        "earnings elasticity",
        "child earnings",
        "parental income",
    ),
    "mechanisms": (
        "educational inequality",
        "educational attainment",
        "residential segregation",
        "neighborhood effects",
        "social capital",
        "wealth concentration",
        "capital accumulation",
        "labor market segmentation",
        "political institutions",
    ),
    "education": (
        "public education",
        "school quality",
        "elite universities",
        "private schooling",
        "tutoring",
        "school funding",
        "property taxes",
        "early childhood",
        "college completion",
    ),
    "labor": (
        "professional networks",
        "recommendation letters",
        "hiring",
        "credential barriers",
        "vocational training",
        "informal labor markets",
        "minimum wage",
        "labor protections",
        "earnings",
    ),
    "policy": (
        "redistribution",
        "taxation of capital",
        "public investment",
        "healthcare access",
        "racial segregation",
        "wealth transfers",
        "inheritance",
    ),
}


def _load_from_yaml(path: pathlib.Path) -> tuple | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    concepts = data.get("concepts")
    if not concepts:
        return None
    categories = data.get("categories", _DEFAULT_CATEGORIES)
    return tuple(concepts), {k: tuple(v) for k, v in categories.items()}


_loaded = _load_from_yaml(pathlib.Path(__file__).with_name("vocabulary.yaml"))
if _loaded is not None:
    CANONICAL_CONCEPTS, CONCEPT_CATEGORIES = _loaded
else:
    CANONICAL_CONCEPTS = _DEFAULT_CONCEPTS
    CONCEPT_CATEGORIES = _DEFAULT_CATEGORIES
