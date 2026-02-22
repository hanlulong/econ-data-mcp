from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

from backend.memory.conversation_state import DataReference
from backend.models import NormalizedData

_AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
_PKG_NAME = "backend.agents_testshim"
_MOD_NAME = f"{_PKG_NAME}.data_agent"

if _PKG_NAME not in sys.modules:
    shim_pkg = types.ModuleType(_PKG_NAME)
    shim_pkg.__path__ = [str(_AGENTS_DIR)]
    sys.modules[_PKG_NAME] = shim_pkg

if _MOD_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _AGENTS_DIR / "data_agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load data_agent.py for tests")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG_NAME
    sys.modules[_MOD_NAME] = module
    spec.loader.exec_module(module)

DataAgent = sys.modules[_MOD_NAME].DataAgent


def _series(country: str, indicator: str = "Imports of goods and services (% of GDP)") -> NormalizedData:
    return NormalizedData.model_validate(
        {
            "metadata": {
                "source": "World Bank",
                "indicator": indicator,
                "country": country,
                "frequency": "annual",
                "unit": "%",
                "lastUpdated": "2026-01-01",
                "seriesId": "NE.IMP.GNFS.ZS",
                "apiUrl": "https://example.com",
                "startDate": "2020-01-01",
                "endDate": "2025-01-01",
            },
            "data": [{"date": "2024-01-01", "value": 20.0}],
        }
    )


class DataAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DataAgent()

    def test_follow_up_modifications_preserve_multi_country_scope(self) -> None:
        ref = DataReference(
            provider="WORLDBANK",
            indicator="NE.IMP.GNFS.ZS",
            country="China",
            countries=["China", "United States"],
            time_range=("2020-01-01", "2025-01-01"),
        )

        params = self.agent._apply_follow_up_modifications(ref, {})  # pylint: disable=protected-access

        self.assertEqual(params.get("countries"), ["China", "United States"])
        self.assertNotIn("country", params)

    def test_create_data_reference_collects_all_series_countries(self) -> None:
        data = [_series("China"), _series("United States")]

        ref = self.agent._create_data_reference("compare import share", data, None)  # pylint: disable=protected-access

        self.assertEqual(ref.country, "China")
        self.assertEqual(ref.countries, ["China", "United States"])


if __name__ == "__main__":
    unittest.main()
