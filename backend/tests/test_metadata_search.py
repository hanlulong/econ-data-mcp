from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.services.metadata_search import MetadataSearchService


class _DummyLLM:
    pass


@pytest.mark.asyncio
async def test_search_sdmx_handles_none_description_entries():
    service = MetadataSearchService(llm_provider=_DummyLLM())
    service._sdmx_catalogs = {
        "IMF": {
            "FLOW_A": {
                "name": "Exports of goods and services",
                "description": None,
            }
        }
    }

    with patch("backend.services.metadata_search.cache_service.get", return_value=None), \
         patch("backend.services.metadata_search.cache_service.set"):
        results = await service.search_sdmx("exports", provider_filter="IMF")

    assert len(results) == 1
    assert results[0]["code"] == "FLOW_A"


@pytest.mark.asyncio
async def test_search_sdmx_treats_empty_cached_results_as_cache_hit():
    service = MetadataSearchService(llm_provider=_DummyLLM())
    service._sdmx_catalogs = {
        "IMF": {
            "FLOW_A": {
                "name": "Exports of goods and services",
                "description": "Trade flow",
            }
        }
    }

    with patch("backend.services.metadata_search.cache_service.get", return_value=[]), \
         patch.object(service, "_load_sdmx_catalogs", side_effect=AssertionError("catalog load should not run on cache hit")):
        results = await service.search_sdmx("exports", provider_filter="IMF")

    assert results == []
