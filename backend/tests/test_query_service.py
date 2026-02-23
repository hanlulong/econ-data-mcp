from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from backend.models import GeneratedFile, NormalizedData, ParsedIntent, QueryResponse
from backend.routing.unified_router import RoutingDecision
from backend.services.cache import cache_service
from backend.services.query import QueryService
from backend.tests.utils import run
from backend.utils.retry import DataNotAvailableError


def sample_series() -> NormalizedData:
    return NormalizedData.model_validate(
        {
            "metadata": {
                "source": "FRED",
                "indicator": "Real GDP",
                "country": "US",
                "frequency": "quarterly",
                "unit": "Billions",
                "lastUpdated": "2024-01-01",
                "seriesId": "GDP",
                "apiUrl": "https://example.com",
            },
            "data": [
                {"date": "2020-01-01", "value": 100.0},
                {"date": "2020-04-01", "value": 90.0},
            ],
        }
    )


class QueryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        cache_service.clear()
        self.service = QueryService(openrouter_key="test", fred_key="fred", comtrade_key="demo")

    def test_process_query_returns_data(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["GDP"],
            parameters={"seriesId": "GDP"},
            clarificationNeeded=False,
        )

        with patch.object(self.service.openrouter, "parse_query", return_value=intent):
            with patch.object(self.service.fred_provider, "fetch_series", return_value=sample_series()):
                response = run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(len(response), 1)
        self.assertEqual(response[0].metadata.indicator, "Real GDP")

    def test_cache_hit_skips_fetch(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["GDP"],
            parameters={"seriesId": "GDP"},
            clarificationNeeded=False,
        )

        first_series = sample_series()
        with patch.object(self.service.fred_provider, "fetch_series", return_value=first_series):
            data = run(self.service._fetch_data(intent))  # pylint: disable=protected-access
            self.assertEqual(data[0].metadata.seriesId, "GDP")

        with patch.object(self.service.fred_provider, "fetch_series", side_effect=AssertionError("Should not refetch")):
            cached = run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(cached[0].metadata.indicator, "Real GDP")

    def test_build_cache_params_adds_version_without_mutating_input(self) -> None:
        raw_params = {"indicator": "NE.IMP.GNFS.ZS", "countries": ["China", "US"]}

        cache_params = self.service._build_cache_params("World Bank", raw_params)  # pylint: disable=protected-access

        self.assertNotIn("_cache_version", raw_params)
        self.assertNotIn("_provider", raw_params)
        self.assertEqual(cache_params["_cache_version"], self.service.CACHE_KEY_VERSION)
        self.assertEqual(cache_params["_provider"], "WORLDBANK")
        self.assertEqual(cache_params["indicator"], "NE.IMP.GNFS.ZS")

    def test_coerce_parsed_intent_sets_original_query_when_missing(self) -> None:
        raw_intent = {
            "apiProvider": "WORLDBANK",
            "indicators": ["NE.IMP.GNFS.ZS"],
            "parameters": {"countries": ["China", "US"]},
            "clarificationNeeded": False,
        }

        intent = self.service._coerce_parsed_intent(raw_intent, "import share of gdp China and US")  # pylint: disable=protected-access

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.originalQuery, "import share of gdp China and US")

    def test_serialize_cache_query_is_deterministic(self) -> None:
        first = {"b": 1, "a": 2}
        second = {"a": 2, "b": 1}

        first_serialized = self.service._serialize_cache_query(first)  # pylint: disable=protected-access
        second_serialized = self.service._serialize_cache_query(second)  # pylint: disable=protected-access

        self.assertEqual(first_serialized, second_serialized)

    def test_cache_version_change_invalidates_prior_entries(self) -> None:
        raw_params = {"seriesId": "GDP"}
        version_1_params = self.service._build_cache_params("FRED", raw_params)  # pylint: disable=protected-access
        cache_service.cache_data("FRED", version_1_params, sample_series())
        self.assertIsNotNone(cache_service.get_data("FRED", version_1_params))

        self.service.CACHE_KEY_VERSION = "test-next-version"
        version_2_params = self.service._build_cache_params("FRED", raw_params)  # pylint: disable=protected-access

        self.assertIsNone(cache_service.get_data("FRED", version_2_params))

    def test_process_query_records_processing_steps(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["GDP"],
            parameters={"seriesId": "GDP"},
            clarificationNeeded=False,
        )

        with patch.object(self.service.openrouter, "parse_query", return_value=intent), \
             patch("backend.services.query.ParameterValidator.validate_intent", return_value=(True, None, None)), \
             patch("backend.services.query.ParameterValidator.check_confidence", return_value=(True, None)), \
             patch("backend.services.query.QueryComplexityAnalyzer.detect_complexity", return_value={"pro_mode_required": False, "complexity_factors": []}), \
             patch.object(self.service.fred_provider, "fetch_series", return_value=sample_series()):

            response = run(self.service.process_query("Show me US GDP"))

        self.assertTrue(response.processingSteps)
        step_names = {step.step for step in response.processingSteps or []}
        # Service now uses LangGraph, so step names have changed
        # Check for either old-style or new-style step names
        has_parsing = "parsing_query" in step_names or "langgraph_execution" in step_names
        has_fetching = "fetching_data" in step_names or "cache_hit" in step_names
        self.assertTrue(has_parsing, f"Expected parsing step, got: {step_names}")
        self.assertTrue(has_fetching, f"Expected fetching step, got: {step_names}")

    def test_select_indicator_query_uses_original_when_cues_mismatch(self) -> None:
        intent = ParsedIntent(
            apiProvider="World Bank",
            indicators=["Gross PSD, Central Gov., All maturities, % of GDP"],
            parameters={"countries": ["China", "US"]},
            clarificationNeeded=False,
            originalQuery="import share of gdp China and US",
        )

        selected = self.service._select_indicator_query_for_resolution(intent)  # pylint: disable=protected-access
        self.assertEqual(selected, "import share of gdp China and US")

    def test_select_indicator_query_uses_original_when_only_generic_gdp_overlap_exists(self) -> None:
        intent = ParsedIntent(
            apiProvider="World Bank",
            indicators=["Gross domestic savings (% of GDP)"],
            parameters={"countries": ["China", "US"]},
            clarificationNeeded=False,
            originalQuery="import share of gdp China and US",
        )

        selected = self.service._select_indicator_query_for_resolution(intent)  # pylint: disable=protected-access
        self.assertEqual(selected, "import share of gdp China and US")

    def test_worldbank_multi_indicator_collapses_to_resolved_code_after_override(self) -> None:
        intent = ParsedIntent(
            apiProvider="World Bank",
            indicators=[
                "Gross PSD, Central Gov., All maturities, % of GDP",
                "Gross PSD, Central Gov., All maturities, % of GDP",
            ],
            parameters={"countries": ["China", "US"]},
            clarificationNeeded=False,
            originalQuery="import share of gdp China and US",
        )

        class _Resolved:
            code = "NE.IMP.GNFS.ZS"
            confidence = 0.9
            source = "database"

        class _Resolver:
            def resolve(self, query, provider=None, **kwargs):
                return _Resolved()

        with patch("backend.services.query.get_indicator_resolver", return_value=_Resolver()), \
             patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.world_bank_provider, "fetch_indicator", return_value=[sample_series()]) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(intent.indicators, ["NE.IMP.GNFS.ZS"])
        self.assertTrue(fetch_mock.called)

    def test_worldbank_fetch_prefers_resolved_indicator_param(self) -> None:
        intent = ParsedIntent(
            apiProvider="World Bank",
            indicators=["Import Share of GDP"],
            parameters={"countries": ["China", "US"], "indicator": "NE.IMP.GNFS.ZS"},
            clarificationNeeded=False,
            originalQuery="import share of gdp China and US",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.world_bank_provider, "fetch_indicator", return_value=[sample_series()]) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(fetch_mock.call_args.kwargs.get("indicator"), "NE.IMP.GNFS.ZS")

    def test_bis_fetch_prefers_resolved_indicator_param(self) -> None:
        intent = ParsedIntent(
            apiProvider="BIS",
            indicators=["policy rate"],
            parameters={"country": "US", "indicator": "BIS.CBPOL"},
            clarificationNeeded=False,
            originalQuery="policy rate in us",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.bis_provider, "fetch_indicator", return_value=[sample_series()]) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(fetch_mock.call_args.kwargs.get("indicator"), "BIS.CBPOL")

    def test_fetch_data_replaces_explicit_fred_code_when_query_conflicts(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["interest rate"],
            parameters={"country": "US", "indicator": "FEDFUNDS"},
            clarificationNeeded=False,
            originalQuery="US 10-year government bond yield from 2000 to 2024",
        )

        class _Resolved:
            code = "DGS10"
            confidence = 0.95
            source = "catalog"

        class _Resolver:
            def resolve(self, query, provider=None, **kwargs):
                return _Resolved()

        with patch("backend.services.query.get_indicator_resolver", return_value=_Resolver()), \
             patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.fred_provider, "fetch_series", return_value=sample_series()) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        params = fetch_mock.call_args.args[0]
        self.assertEqual(params.get("indicator"), "DGS10")

    def test_is_resolved_indicator_plausible_rejects_bis_debt_service_for_debt_gdp_query(self) -> None:
        self.assertFalse(
            self.service._is_resolved_indicator_plausible(  # pylint: disable=protected-access
                provider="BIS",
                indicator_query="gdp to debt ratio in china",
                resolved_code="WS_DSR",
            )
        )

    def test_is_resolved_indicator_plausible_rejects_bis_debt_securities_for_debt_gdp_query(self) -> None:
        self.assertFalse(
            self.service._is_resolved_indicator_plausible(  # pylint: disable=protected-access
                provider="BIS",
                indicator_query="gdp to debt ratio in china",
                resolved_code="WS_DEBT_SEC2_PUB",
            )
        )

    def test_is_resolved_indicator_plausible_rejects_fred_policy_rate_for_bond_yield_query(self) -> None:
        self.assertFalse(
            self.service._is_resolved_indicator_plausible(  # pylint: disable=protected-access
                provider="FRED",
                indicator_query="US 10-year government bond yield from 2000 to 2024",
                resolved_code="FEDFUNDS",
            )
        )

        self.assertTrue(
            self.service._is_resolved_indicator_plausible(  # pylint: disable=protected-access
                provider="FRED",
                indicator_query="US 10-year government bond yield from 2000 to 2024",
                resolved_code="DGS10",
            )
        )

        self.assertTrue(
            self.service._is_resolved_indicator_plausible(  # pylint: disable=protected-access
                provider="BIS",
                indicator_query="debt service ratio in china",
                resolved_code="WS_DSR",
            )
        )

    def test_select_indicator_query_uses_original_for_discontinued_indicator(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["Composite Yield on U.S. Treasury Bonds with Maturity over 10 Years (DISCONTINUED)"],
            parameters={"country": "US"},
            clarificationNeeded=False,
            originalQuery="US 10-year government bond yield from 2000 to 2024",
        )

        selected = self.service._select_indicator_query_for_resolution(intent)  # pylint: disable=protected-access
        self.assertEqual(selected, "US 10-year government bond yield from 2000 to 2024")

    def test_apply_concept_provider_override_reroutes_unavailable_provider(self) -> None:
        intent = ParsedIntent(
            apiProvider="BIS",
            indicators=["WS_DSR"],
            parameters={"country": "CN"},
            clarificationNeeded=False,
            originalQuery="gdp to debt ratio in china",
        )

        with patch("backend.services.catalog_service.find_concept_by_term", return_value="government_debt"), \
             patch("backend.services.catalog_service.is_provider_available", return_value=False), \
             patch("backend.services.catalog_service.get_best_provider", return_value=("IMF", "GGXWDG_NGDP", 0.95)):
            provider, params = self.service._apply_concept_provider_override(  # pylint: disable=protected-access
                "BIS",
                intent,
                dict(intent.parameters),
            )

        self.assertEqual(provider, "IMF")
        self.assertEqual(intent.apiProvider, "IMF")
        self.assertEqual(params.get("indicator"), "GGXWDG_NGDP")

    def test_normalize_bis_metadata_labels_replaces_opaque_code(self) -> None:
        bis_series = NormalizedData.model_validate(
            {
                "metadata": {
                    "source": "BIS",
                    "indicator": "WS_DSR",
                    "country": "CN",
                    "frequency": "quarterly",
                    "unit": "percent",
                    "lastUpdated": "",
                    "seriesId": "WS_DSR",
                },
                "data": [{"date": "2020-01-01", "value": 1.0}],
            }
        )

        self.service._normalize_bis_metadata_labels([bis_series])  # pylint: disable=protected-access

        self.assertNotEqual(bis_series.metadata.indicator, "WS_DSR")
        self.assertIn("debt service", bis_series.metadata.indicator.lower())

    def test_normalize_metadata_labels_promotes_description_for_code_like_indicator(self) -> None:
        imf_series = NormalizedData.model_validate(
            {
                "metadata": {
                    "source": "IMF",
                    "indicator": "GGXCNL_NGDP",
                    "country": "CN",
                    "frequency": "annual",
                    "unit": "percent",
                    "lastUpdated": "",
                    "seriesId": "GGXCNL_NGDP",
                    "description": "General government net lending/borrowing (% of GDP)",
                },
                "data": [{"date": "2020-01-01", "value": -2.1}],
            }
        )

        self.service._normalize_bis_metadata_labels([imf_series])  # pylint: disable=protected-access

        self.assertEqual(
            imf_series.metadata.indicator,
            "General government net lending/borrowing (% of GDP)",
        )

    def test_has_implausible_top_series_detects_wrong_fred_tenor(self) -> None:
        wrong_series = NormalizedData.model_validate(
            {
                "metadata": {
                    "source": "FRED",
                    "indicator": "Composite Yield on U.S. Treasury Bonds with Maturity over 10 Years (DISCONTINUED)",
                    "country": "US",
                    "frequency": "daily",
                    "unit": "Percent",
                    "lastUpdated": "",
                    "seriesId": "DLTBOARD",
                },
                "data": [{"date": "2020-01-01", "value": 1.0}],
            }
        )

        self.assertTrue(
            self.service._has_implausible_top_series(  # pylint: disable=protected-access
                "US 10-year government bond yield from 2000 to 2024",
                [wrong_series],
            )
        )

    def test_build_uncertain_result_clarification_returns_ranked_options(self) -> None:
        intent = ParsedIntent(
            apiProvider="BIS",
            indicators=["GDP to Debt Ratio"],
            parameters={"country": "CN"},
            clarificationNeeded=False,
            originalQuery="gdp to debt ratio in china",
        )
        uncertain_data = [
            NormalizedData.model_validate(
                {
                    "metadata": {
                        "source": "BIS",
                        "indicator": "Debt service ratios",
                        "country": "CN",
                        "frequency": "quarterly",
                        "unit": "percent",
                        "lastUpdated": "",
                        "seriesId": "WS_DSR",
                    },
                    "data": [
                        {"date": "2020-01-01", "value": 1.0},
                    ],
                }
            )
        ]

        class _Resolved:
            def __init__(self, provider: str, code: str, name: str, confidence: float):
                self.provider = provider
                self.code = code
                self.name = name
                self.confidence = confidence
                self.source = "catalog"

        class _Resolver:
            def resolve(self, query, provider=None, **kwargs):
                mapping = {
                    "IMF": _Resolved("IMF", "GGXWDG_NGDP", "General government gross debt (% of GDP)", 0.95),
                    "WORLDBANK": _Resolved("WORLDBANK", "GC.DOD.TOTL.GD.ZS", "Central government debt, total (% of GDP)", 0.92),
                    "BIS": _Resolved("BIS", "WS_DSR", "Debt service ratios", 0.90),
                }
                key = (provider or "").upper()
                return mapping.get(key)

        with patch("backend.services.query.get_indicator_resolver", return_value=_Resolver()), \
             patch.object(self.service, "_get_fallback_providers", return_value=["IMF", "WORLDBANK"]):
            clarification = self.service._build_uncertain_result_clarification(  # pylint: disable=protected-access
                conversation_id="conv-1",
                query="gdp to debt ratio in china",
                intent=intent,
                data=uncertain_data,
                processing_steps=None,
            )

        self.assertIsNotNone(clarification)
        assert clarification is not None
        self.assertTrue(clarification.clarificationNeeded)
        joined = "\n".join(clarification.clarificationQuestions or [])
        self.assertIn("IMF", joined)
        self.assertIn("WorldBank", joined)

    def test_needs_indicator_clarification_detects_catalog_concept_mismatch(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["imports as % of GDP"],
            parameters={"countries": ["CN", "US"]},
            clarificationNeeded=False,
            originalQuery="import share of gdp in china and us",
        )
        mismatched_series = [
            NormalizedData.model_validate(
                {
                    "metadata": {
                        "source": "WorldBank",
                        "indicator": "Gross savings (% of GDP)",
                        "country": "CN",
                        "frequency": "annual",
                        "unit": "%",
                        "lastUpdated": "",
                        "seriesId": "NY.GNS.ICTR.ZS",
                    },
                    "data": [{"date": "2020-01-01", "value": 44.0}],
                }
            )
        ]

        needs = self.service._needs_indicator_clarification(  # pylint: disable=protected-access
            query="import share of gdp in china and us",
            data=mismatched_series,
            intent=intent,
        )
        self.assertTrue(needs)

    def test_build_multi_concept_query_clarification_for_single_indicator_parse(self) -> None:
        intent = ParsedIntent(
            apiProvider="IMF",
            indicators=["inflation"],
            parameters={"countries": ["US", "GB"]},
            clarificationNeeded=False,
            originalQuery="compare unemployment and inflation for g7 countries from 2010 to 2024",
        )

        clarification = self.service._build_multi_concept_query_clarification(  # pylint: disable=protected-access
            conversation_id="conv-multi",
            query=intent.originalQuery,
            intent=intent,
            is_multi_indicator=False,
            processing_steps=None,
        )

        self.assertIsNotNone(clarification)
        assert clarification is not None
        self.assertTrue(clarification.clarificationNeeded)
        joined = "\n".join(clarification.clarificationQuestions or []).lower()
        self.assertIn("multiple indicator families", joined)
        self.assertIn("labor market", joined)
        self.assertIn("prices/inflation", joined)

    def test_build_multi_concept_query_clarification_skips_trade_openness_phrasing(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["trade openness"],
            parameters={"country": "SG"},
            clarificationNeeded=False,
            originalQuery="trade openness ratio (exports plus imports to gdp) in singapore",
        )

        clarification = self.service._build_multi_concept_query_clarification(  # pylint: disable=protected-access
            conversation_id="conv-openness",
            query=intent.originalQuery,
            intent=intent,
            is_multi_indicator=False,
            processing_steps=None,
        )

        self.assertIsNone(clarification)

    def test_collect_indicator_choice_options_prefers_raw_query_on_cue_mismatch(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["GDP (current US$)"],
            parameters={"countries": ["China", "UK"]},
            clarificationNeeded=False,
            originalQuery="gdp to debt ratio in china and uk",
        )

        class _Resolved:
            def __init__(self, provider: str, code: str, name: str, confidence: float):
                self.provider = provider
                self.code = code
                self.name = name
                self.confidence = confidence
                self.source = "catalog"

        captured_queries = []

        class _Resolver:
            def resolve(self, query, provider=None, **kwargs):
                captured_queries.append(str(query))
                mapping = {
                    "IMF": _Resolved("IMF", "GGXWDG_NGDP", "General government gross debt (% of GDP)", 0.95),
                    "WORLDBANK": _Resolved("WORLDBANK", "GC.DOD.TOTL.GD.ZS", "Central government debt, total (% of GDP)", 0.92),
                }
                return mapping.get((provider or "").upper())

        with patch.object(self.service, "_select_indicator_query_for_resolution", return_value="GDP (current US$)"), \
             patch("backend.services.query.get_indicator_resolver", return_value=_Resolver()), \
             patch.object(self.service, "_get_fallback_providers", return_value=["IMF"]):
            options = self.service._collect_indicator_choice_options(  # pylint: disable=protected-access
                query="gdp to debt ratio in china and uk",
                intent=intent,
                max_options=3,
            )

        self.assertTrue(options)
        self.assertTrue(any("debt" in option.lower() for option in options))
        self.assertTrue(captured_queries)
        self.assertEqual(captured_queries[0], "gdp to debt ratio in china and uk")

    def test_eurostat_fetch_prefers_resolved_indicator_param(self) -> None:
        intent = ParsedIntent(
            apiProvider="Eurostat",
            indicators=["harmonized inflation"],
            parameters={"country": "DE", "indicator": "prc_hicp_manr"},
            clarificationNeeded=False,
            originalQuery="hicp inflation germany",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.eurostat_provider, "fetch_indicator", return_value=sample_series()) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(fetch_mock.call_args.kwargs.get("indicator"), "prc_hicp_manr")

    def test_oecd_fetch_prefers_resolved_indicator_param(self) -> None:
        intent = ParsedIntent(
            apiProvider="OECD",
            indicators=["unemployment rate"],
            parameters={"country": "USA", "indicator": "LFS_UNEM_A"},
            clarificationNeeded=False,
            originalQuery="oecd unemployment rate us",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.oecd_provider, "fetch_indicator", return_value=sample_series()) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(fetch_mock.call_args.kwargs.get("indicator"), "LFS_UNEM_A")

    def test_imf_fetch_prefers_resolved_indicator_param_for_multi_country_batch(self) -> None:
        intent = ParsedIntent(
            apiProvider="IMF",
            indicators=["government deficit as % of gdp", "fiscal balance"],
            parameters={
                "countries": ["China", "Brazil"],
                "indicator": "GGXCNL_NGDP",
                "startDate": "2015-01-01",
                "endDate": "2024-01-01",
            },
            clarificationNeeded=False,
            originalQuery="from imf government deficit share of gdp in china and brazil",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.imf_provider, "_resolve_countries", side_effect=lambda x: [str(x).upper()]), \
             patch.object(self.service.imf_provider, "fetch_batch_indicator", new_callable=AsyncMock, return_value=[sample_series()]) as fetch_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(fetch_mock.await_count, 1)
        self.assertEqual(fetch_mock.await_args.kwargs.get("indicator"), "GGXCNL_NGDP")

    def test_fetch_data_concept_override_prefers_stronger_provider_match(self) -> None:
        intent = ParsedIntent(
            apiProvider="IMF",
            indicators=["REAL_EFFECTIVE_EXCHANGE_RATE"],
            parameters={"country": "JP"},
            clarificationNeeded=False,
            originalQuery="real effective exchange rate in japan",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.bis_provider, "fetch_indicator", return_value=[sample_series()]) as bis_fetch, \
             patch.object(self.service.imf_provider, "fetch_indicator", side_effect=AssertionError("should reroute before IMF fetch")):
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertTrue(bis_fetch.called)
        self.assertEqual(bis_fetch.call_args.kwargs.get("indicator"), "WS_XRU")

    def test_process_query_enforces_explicit_provider_request(self) -> None:
        intent = ParsedIntent(
            apiProvider="FRED",
            indicators=["GDP"],
            parameters={"country": "US"},
            clarificationNeeded=False,
        )

        class _Settings:
            use_langchain_orchestrator = False

        with patch("backend.config.get_settings", return_value=_Settings()), \
             patch.object(self.service.openrouter, "parse_query", return_value=intent), \
             patch("backend.services.query.QueryComplexityAnalyzer.detect_complexity", return_value={"pro_mode_required": False, "complexity_factors": []}), \
             patch("backend.services.query.ParameterValidator.validate_intent", return_value=(True, None, None)), \
             patch("backend.services.query.ParameterValidator.check_confidence", return_value=(True, None)), \
             patch.object(self.service, "_fetch_data", return_value=[sample_series()]), \
             patch.object(self.service, "_select_routed_provider", side_effect=AssertionError("router should not override explicit provider")):
            response = run(self.service.process_query("gdp from world bank for us", auto_pro_mode=False))

        self.assertIsNotNone(response.intent)
        assert response.intent is not None
        self.assertEqual(response.intent.apiProvider, "WORLDBANK")

    def test_select_routed_provider_prefers_semantic_router_when_available(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["government debt to gdp"],
            parameters={"country": "CN"},
            clarificationNeeded=False,
        )

        class _SemanticRouter:
            async def route(self, **kwargs):
                return RoutingDecision(
                    provider="IMF",
                    confidence=0.88,
                    fallbacks=["WorldBank"],
                    reasoning="semantic-router match",
                    match_type="semantic",
                )

        class _HybridRouter:
            async def route(self, **kwargs):
                raise AssertionError("Hybrid router should not run when semantic router is enabled")

        self.service.semantic_provider_router = _SemanticRouter()
        self.service.hybrid_router = _HybridRouter()

        provider = run(self.service._select_routed_provider(intent, "government debt in china"))  # pylint: disable=protected-access
        self.assertEqual(provider, "IMF")

    def test_select_routed_provider_keeps_high_confidence_deterministic_match(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["gdp to debt ratio"],
            parameters={"country": "CN"},
            clarificationNeeded=False,
        )

        class _UnifiedRouter:
            def route(self, **kwargs):
                return RoutingDecision(
                    provider="IMF",
                    confidence=0.90,
                    fallbacks=["WorldBank", "BIS"],
                    reasoning="deterministic macro debt rule",
                    match_type="indicator",
                )

        class _SemanticRouter:
            async def route(self, **kwargs):
                return RoutingDecision(
                    provider="BIS",
                    confidence=0.58,
                    fallbacks=["IMF", "WorldBank"],
                    reasoning="semantic-router similarity match (0.58)",
                    match_type="semantic",
                )

        self.service.unified_router = _UnifiedRouter()
        self.service.semantic_provider_router = _SemanticRouter()
        self.service.hybrid_router = None

        provider = run(self.service._select_routed_provider(intent, "gdp to debt ratio in china"))  # pylint: disable=protected-access
        self.assertEqual(provider, "IMF")

    def test_select_routed_provider_uses_unified_router_baseline(self) -> None:
        intent = ParsedIntent(
            apiProvider="WorldBank",
            indicators=["gdp growth"],
            parameters={"country": "DE"},
            clarificationNeeded=False,
        )

        class _UnifiedRouter:
            def route(self, **kwargs):
                return RoutingDecision(
                    provider="IMF",
                    confidence=0.9,
                    fallbacks=["WorldBank"],
                    reasoning="unified baseline",
                    match_type="indicator",
                )

        self.service.unified_router = _UnifiedRouter()
        self.service.semantic_provider_router = None
        self.service.hybrid_router = None

        with patch("backend.services.query.ProviderRouter.route_provider", side_effect=AssertionError("legacy baseline should not run")):
            provider = run(self.service._select_routed_provider(intent, "gdp growth germany"))  # pylint: disable=protected-access
        self.assertEqual(provider, "IMF")

    def test_catalog_provider_reroute_remaps_indicator_code(self) -> None:
        intent = ParsedIntent(
            apiProvider="CoinGecko",
            indicators=["renewable energy share"],
            parameters={},
            clarificationNeeded=False,
            originalQuery="renewable energy share in germany",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.world_bank_provider, "fetch_indicator", return_value=[sample_series()]) as wb_fetch:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertEqual(intent.apiProvider, "WorldBank")
        self.assertEqual(intent.parameters.get("indicator"), "EG.FEC.RNEW.ZS")
        self.assertEqual(wb_fetch.call_args.kwargs.get("indicator"), "EG.FEC.RNEW.ZS")

    def test_is_fallback_relevant_uses_country_resolver_aliases(self) -> None:
        series = sample_series()
        series.metadata.country = "GB"
        series.metadata.indicator = "Imports of goods and services (% of GDP)"

        self.assertTrue(
            self.service._is_fallback_relevant(  # pylint: disable=protected-access
                ["imports of goods and services"],
                [series],
                target_countries=["UK"],
            )
        )

    def test_is_fallback_relevant_rejects_country_mismatch_for_multi_country_queries(self) -> None:
        series = sample_series()
        series.metadata.country = "IN"
        series.metadata.indicator = "Imports of goods and services (% of GDP)"

        self.assertFalse(
            self.service._is_fallback_relevant(  # pylint: disable=protected-access
                ["imports of goods and services"],
                [series],
                target_countries=["China", "US"],
            )
        )

    def test_try_with_fallback_sanitizes_provider_specific_indicator_params(self) -> None:
        intent = ParsedIntent(
            apiProvider="IMF",
            indicators=["imports of goods and services (% of GDP)"],
            parameters={
                "country": "CN",
                "indicator": "BM_GDP",
                "seriesId": "BM_GDP",
                "startDate": "2020-01-01",
                "endDate": "2024-01-01",
            },
            clarificationNeeded=False,
            originalQuery="imports share of gdp in china",
        )
        captured_intent = {}

        async def _fake_fetch_data(fallback_intent):
            captured_intent["intent"] = fallback_intent
            return [sample_series()]

        with patch.object(self.service, "_get_fallback_providers", return_value=["WORLDBANK"]), \
             patch.object(self.service, "_fetch_data", side_effect=_fake_fetch_data), \
             patch.object(self.service, "_is_fallback_relevant", return_value=True):
            result = run(
                self.service._try_with_fallback(  # pylint: disable=protected-access
                    intent,
                    DataNotAvailableError("primary failed"),
                )
            )

        self.assertEqual(len(result), 1)
        fallback_intent = captured_intent["intent"]
        self.assertEqual(fallback_intent.apiProvider, "WORLDBANK")
        self.assertNotIn("indicator", fallback_intent.parameters)
        self.assertNotIn("seriesId", fallback_intent.parameters)
        self.assertEqual(intent.parameters.get("indicator"), "BM_GDP")
        self.assertEqual(intent.parameters.get("seriesId"), "BM_GDP")

    def test_get_fallback_providers_passes_country_context_to_resolver(self) -> None:
        class _Resolved:
            confidence = 0.8

        class _Resolver:
            def __init__(self):
                self.calls = []

            def resolve(self, *args, **kwargs):
                self.calls.append(kwargs)
                return _Resolved()

        resolver = _Resolver()
        with patch("backend.services.indicator_resolver.get_indicator_resolver", return_value=resolver):
            _ = self.service._get_fallback_providers(  # pylint: disable=protected-access
                "IMF",
                indicator="imports",
                country="CN",
                countries=["CN", "GB"],
            )

        self.assertTrue(resolver.calls)
        first_call = resolver.calls[0]
        self.assertEqual(first_call.get("country"), "CN")
        self.assertEqual(first_call.get("countries"), ["CN", "GB"])

    def test_execute_with_langgraph_handles_generated_file_models(self) -> None:
        class _FakeGraph:
            async def ainvoke(self, _initial_state, _config):
                return {
                    "is_pro_mode": True,
                    "code_execution": {
                        "code": "print('ok')",
                        "output": "ok",
                        "error": None,
                        "files": [
                            GeneratedFile(
                                url="/static/promode/report.png",
                                name="report.png",
                                type="image",
                            )
                        ],
                    },
                }

        class _FakeStateManager:
            def get(self, _conversation_id):
                return None

            def update(self, *_args, **_kwargs):
                return None

        import sys
        import types

        fake_agents_module = types.ModuleType("backend.agents")
        fake_agents_module.get_agent_graph = lambda: _FakeGraph()
        fake_agents_module.set_query_service_provider = lambda _provider=None: None

        fake_messages_module = types.ModuleType("langchain_core.messages")

        class _Message:
            def __init__(self, content: str = ""):
                self.content = content

        fake_messages_module.HumanMessage = _Message
        fake_messages_module.AIMessage = _Message
        fake_langchain_core = types.ModuleType("langchain_core")
        fake_langchain_core.messages = fake_messages_module

        with patch.dict(
            sys.modules,
            {
                "backend.agents": fake_agents_module,
                "langchain_core": fake_langchain_core,
                "langchain_core.messages": fake_messages_module,
            },
        ), \
             patch("backend.memory.state_manager.get_state_manager", return_value=_FakeStateManager()):
            response = run(
                self.service._execute_with_langgraph(  # pylint: disable=protected-access
                    query="create inflation forecast chart",
                    conversation_id="conv-lg-pro",
                    conversation_history=[],
                    tracker=None,
                )
            )

        self.assertTrue(response.isProMode)
        self.assertIsNotNone(response.codeExecution)
        assert response.codeExecution is not None
        self.assertEqual(response.codeExecution.files[0].name, "report.png")

    def test_execute_with_langgraph_retries_standard_path_when_empty_result_lacks_intent(self) -> None:
        class _FakeGraph:
            async def ainvoke(self, _initial_state, _config):
                return {
                    "result": {"data": []},
                    "parsed_intent": None,
                    "current_provider": "unknown",
                }

        class _FakeStateManager:
            def get(self, _conversation_id):
                return None

            def update(self, *_args, **_kwargs):
                return None

        standard_response = QueryResponse(
            conversationId="conv-lg-empty",
            clarificationNeeded=False,
            data=[sample_series()],
        )

        import sys
        import types

        fake_agents_module = types.ModuleType("backend.agents")
        fake_agents_module.get_agent_graph = lambda: _FakeGraph()
        fake_agents_module.set_query_service_provider = lambda _provider=None: None

        fake_messages_module = types.ModuleType("langchain_core.messages")

        class _Message:
            def __init__(self, content: str = ""):
                self.content = content

        fake_messages_module.HumanMessage = _Message
        fake_messages_module.AIMessage = _Message
        fake_langchain_core = types.ModuleType("langchain_core")
        fake_langchain_core.messages = fake_messages_module

        with patch.dict(
            sys.modules,
            {
                "backend.agents": fake_agents_module,
                "langchain_core": fake_langchain_core,
                "langchain_core.messages": fake_messages_module,
            },
        ), \
             patch("backend.memory.state_manager.get_state_manager", return_value=_FakeStateManager()), \
             patch.object(self.service, "_standard_query_processing", new_callable=AsyncMock, return_value=standard_response) as standard_mock:
            response = run(
                self.service._execute_with_langgraph(  # pylint: disable=protected-access
                    query="which asean country has highest import share of gdp since 2015",
                    conversation_id="conv-lg-empty",
                    conversation_history=[],
                    tracker=None,
                )
            )

        self.assertEqual(len(response.data or []), 1)
        standard_mock.assert_awaited_once()

    def test_process_query_tries_fallback_when_primary_returns_empty_data(self) -> None:
        intent = ParsedIntent(
            apiProvider="IMF",
            indicators=["imports share of gdp"],
            parameters={"country": "CN"},
            clarificationNeeded=False,
            originalQuery="imports share of gdp in china",
        )

        class _Settings:
            use_langchain_orchestrator = False

        with patch("backend.config.get_settings", return_value=_Settings()), \
             patch.object(self.service.openrouter, "parse_query", return_value=intent), \
             patch("backend.services.query.QueryComplexityAnalyzer.detect_complexity", return_value={"pro_mode_required": False, "complexity_factors": []}), \
             patch("backend.services.query.ParameterValidator.validate_intent", return_value=(True, None, None)), \
             patch("backend.services.query.ParameterValidator.check_confidence", return_value=(True, None)), \
             patch.object(self.service, "_fetch_data", return_value=[]), \
             patch.object(self.service, "_try_with_fallback", return_value=[sample_series()]) as fallback_mock:
            response = run(self.service.process_query("imports share of gdp in china", auto_pro_mode=False))

        self.assertIsNone(response.error)
        self.assertEqual(len(response.data or []), 1)
        self.assertTrue(fallback_mock.called)

    def test_fetch_data_coingecko_normalizes_empty_coin_ids_and_vs_currency(self) -> None:
        intent = ParsedIntent(
            apiProvider="CoinGecko",
            indicators=["bitcoin market cap"],
            parameters={"coinIds": None, "vsCurrency": "right"},
            clarificationNeeded=False,
            originalQuery="bitcoin market cap right now",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.coingecko_provider, "get_historical_data_range", return_value=[sample_series()]) as range_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertTrue(range_mock.called)
        self.assertEqual(range_mock.call_args.kwargs.get("vs_currency"), "usd")

    def test_fetch_data_coingecko_uses_query_text_for_coin_and_metric_when_indicator_is_dynamic(self) -> None:
        intent = ParsedIntent(
            apiProvider="CoinGecko",
            indicators=["dynamic"],
            parameters={},
            clarificationNeeded=False,
            originalQuery="solana trading volume over the last 90 days",
        )

        with patch.object(self.service, "_get_from_cache", return_value=None), \
             patch.object(self.service.coingecko_provider, "get_historical_data", return_value=[sample_series()]) as historical_mock:
            run(self.service._fetch_data(intent))  # pylint: disable=protected-access

        self.assertTrue(historical_mock.called)
        self.assertEqual(historical_mock.call_args.kwargs.get("coin_id"), "solana")
        self.assertEqual(historical_mock.call_args.kwargs.get("metric"), "volume")
        self.assertEqual(historical_mock.call_args.kwargs.get("days"), 90)


if __name__ == "__main__":
    unittest.main()
