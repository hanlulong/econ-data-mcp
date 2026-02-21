from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend.models import NormalizedData
from backend.services.redis_cache import RedisCacheService


def _sample_series(country: str = "US") -> NormalizedData:
    return NormalizedData.model_validate(
        {
            "metadata": {
                "source": "FRED",
                "indicator": "Real GDP",
                "country": country,
                "frequency": "quarterly",
                "unit": "Billions",
                "lastUpdated": "2026-01-01",
                "seriesId": "GDP",
                "apiUrl": "https://example.com",
            },
            "data": [{"date": "2025-01-01", "value": 100.0}],
        }
    )


def _build_service() -> RedisCacheService:
    settings = SimpleNamespace(redis_url=None)
    with patch("backend.services.redis_cache.get_settings", return_value=settings):
        return RedisCacheService()


def test_to_jsonable_converts_pydantic_models():
    service = _build_service()

    payload = service._to_jsonable(_sample_series())  # pylint: disable=protected-access

    assert isinstance(payload, dict)
    assert payload["metadata"]["seriesId"] == "GDP"


def test_restore_cached_payload_reconstructs_normalized_data_list():
    service = _build_service()
    serialized = [
        service._to_jsonable(_sample_series("US")),  # pylint: disable=protected-access
        service._to_jsonable(_sample_series("CN")),  # pylint: disable=protected-access
    ]

    restored = service._restore_cached_payload(serialized)  # pylint: disable=protected-access

    assert isinstance(restored, list)
    assert all(isinstance(item, NormalizedData) for item in restored)
    assert restored[0].metadata.country == "US"
    assert restored[1].metadata.country == "CN"

