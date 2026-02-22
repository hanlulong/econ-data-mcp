from __future__ import annotations

from backend.services.catalog_service import find_concept_by_term, reload_catalog


def test_debt_service_ratio_maps_to_debt_service_concept():
    reload_catalog()
    concept = find_concept_by_term("debt service ratio in china")
    assert concept == "debt_service_ratio"


def test_debt_to_gdp_maps_to_government_debt():
    reload_catalog()
    concept = find_concept_by_term("government debt to gdp ratio")
    assert concept == "government_debt"
