from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from common.llm.cache import LLMCache
from common.llm.router import LLMRouter, TaskType
from polymarket.models import Market
from polymarket.search.semantic_filter import ClassificationResult, SemanticFilter


def _mock_router_with_provider(mock_provider):
    router = LLMRouter()
    router._providers[TaskType.CLASSIFICATION] = mock_provider
    return router


@pytest.mark.asyncio
async def test_classify_weather_market():
    mock_provider = AsyncMock()
    mock_provider.complete_json = AsyncMock(
        return_value=ClassificationResult(
            is_weather_related=True,
            confidence=0.95,
            reasoning="Hurricane prediction market",
            keywords=["hurricane"],
        )
    )

    router = _mock_router_with_provider(mock_provider)
    sf = SemanticFilter(router)

    market = Market(
        id="1",
        question="Will a hurricane hit Florida?",
        description="Category 4+ hurricane landfall",
    )
    result = await sf.classify(market)

    assert result is not None
    assert result.relevance_score == 0.95
    assert "hurricane" in result.matched_keywords


@pytest.mark.asyncio
async def test_classify_non_weather_market():
    mock_provider = AsyncMock()
    mock_provider.complete_json = AsyncMock(
        return_value=ClassificationResult(
            is_weather_related=False,
            confidence=0.1,
            reasoning="Cryptocurrency market",
            keywords=[],
        )
    )

    router = _mock_router_with_provider(mock_provider)
    sf = SemanticFilter(router)

    market = Market(id="2", question="Will BTC hit 100k?", description="Bitcoin price")
    result = await sf.classify(market)

    assert result is None


@pytest.mark.asyncio
async def test_classify_uses_cache():
    mock_provider = AsyncMock()
    mock_provider.complete_json = AsyncMock(
        return_value=ClassificationResult(
            is_weather_related=True,
            confidence=0.9,
            reasoning="Snowfall market",
            keywords=["snow"],
        )
    )

    router = _mock_router_with_provider(mock_provider)
    cache = LLMCache()
    sf = SemanticFilter(router, cache=cache)

    market = Market(id="3", question="Snow in NYC?", description="Snowfall prediction")

    result1 = await sf.classify(market)
    result2 = await sf.classify(market)

    assert result1 is not None
    assert result2 is not None
    assert result2.relevance_score == 0.9
    mock_provider.complete_json.assert_called_once()


@pytest.mark.asyncio
async def test_filter_markets():
    responses = [
        ClassificationResult(
            is_weather_related=True, confidence=0.95, keywords=["hurricane"]
        ),
        ClassificationResult(
            is_weather_related=False, confidence=0.1, keywords=[]
        ),
        ClassificationResult(
            is_weather_related=True, confidence=0.7, keywords=["snow"]
        ),
    ]

    mock_provider = AsyncMock()
    mock_provider.complete_json = AsyncMock(side_effect=responses)

    router = _mock_router_with_provider(mock_provider)
    sf = SemanticFilter(router)

    markets = [
        Market(id="1", question="Hurricane in Florida?", description="Hurricane prediction"),
        Market(id="2", question="BTC to 100k?", description="Crypto"),
        Market(id="3", question="Snow in Chicago?", description="Snow prediction"),
    ]

    results = await sf.filter_markets(markets)
    assert len(results) == 2
    assert results[0].relevance_score >= results[1].relevance_score
    assert results[0].matched_keywords == ["hurricane"]
    assert results[1].matched_keywords == ["snow"]
