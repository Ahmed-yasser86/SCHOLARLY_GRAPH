"""CORE client tests: unauthenticated guard plus live search when keyed."""

from __future__ import annotations

import os

import pytest

from scholarly_graph.ingestion.discovery import CoreClient

pytestmark = pytest.mark.live


def test_search_without_key_raises():
    with pytest.raises(RuntimeError):
        CoreClient(api_key="").search("inequality mobility")


def test_live_search_when_key_present():
    api_key = os.environ.get("CORE_API_KEY", "")
    if not api_key:
        pytest.skip("CORE_API_KEY not set")
    results = CoreClient(api_key=api_key).search("inequality mobility")
    assert results
    assert results[0].get("title")
