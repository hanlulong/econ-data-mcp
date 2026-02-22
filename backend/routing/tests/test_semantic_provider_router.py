from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.routing.semantic_provider_router import SemanticProviderRouter
from backend.routing.unified_router import RoutingDecision


@dataclass
class _FakeSettings:
    use_semantic_provider_router: bool = True
    semantic_router_similarity_threshold: float = 0.58
    semantic_router_top_k: int = 5
    semantic_router_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_litellm_router_fallback: bool = True
    semantic_router_litellm_timeout: int = 20
    llm_provider: str = "vllm"
    llm_model: str = "gpt-oss-120b"
    openrouter_api_key: str | None = None
    llm_base_url: str | None = "http://localhost:8000"
    vllm_api_key: str | None = "EMPTY"


class _FakeDeterministicRouter:
    def __init__(self, provider: str = "WorldBank", fallbacks: list[str] | None = None):
        self.provider = provider
        self.fallbacks = fallbacks or ["IMF", "OECD"]

    def route(self, **kwargs):
        return RoutingDecision(
            provider=self.provider,
            confidence=0.8,
            fallbacks=self.fallbacks,
            reasoning="deterministic baseline",
            match_type="default",
        )


class _FakeChoice:
    def __init__(self, name: str, similarity_score: float):
        self.name = name
        self.similarity_score = similarity_score


class _FakeSemanticEngine:
    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score

    def __call__(self, text: str, route_filter=None, limit=1):
        assert route_filter is not None
        return _FakeChoice(name=self.name, similarity_score=self.score)


@pytest.mark.asyncio
async def test_semantic_router_overrides_baseline_when_confident():
    router = SemanticProviderRouter(
        settings=_FakeSettings(),
        deterministic_router=_FakeDeterministicRouter(provider="WorldBank", fallbacks=["IMF", "OECD"]),
        semantic_engine=_FakeSemanticEngine(name="IMF", score=0.91),
    )

    decision = await router.route("government debt to gdp in china")

    assert decision.provider == "IMF"
    assert decision.match_type == "semantic"
    assert decision.confidence >= 0.9


@pytest.mark.asyncio
async def test_low_semantic_similarity_falls_back_to_baseline_when_litellm_disabled():
    settings = _FakeSettings(use_litellm_router_fallback=False)
    router = SemanticProviderRouter(
        settings=settings,
        deterministic_router=_FakeDeterministicRouter(provider="WorldBank", fallbacks=["IMF", "OECD"]),
        semantic_engine=_FakeSemanticEngine(name="IMF", score=0.15),
    )

    decision = await router.route("cross-country gdp growth")

    assert decision.provider == "WorldBank"
    assert decision.match_type == "default"


@pytest.mark.asyncio
async def test_litellm_fallback_used_when_semantic_is_low():
    class _RouterWithLitellmStub(SemanticProviderRouter):
        async def _litellm_route_choice(self, *args, **kwargs):
            return ("OECD", 0.77, "LiteLLM chose OECD")

    router = _RouterWithLitellmStub(
        settings=_FakeSettings(),
        deterministic_router=_FakeDeterministicRouter(provider="WorldBank", fallbacks=["IMF", "OECD"]),
        semantic_engine=_FakeSemanticEngine(name="IMF", score=0.20),
    )

    decision = await router.route("oecd productivity comparison")

    assert decision.provider == "OECD"
    assert decision.match_type == "litellm"
    assert decision.confidence == pytest.approx(0.77, abs=1e-6)
