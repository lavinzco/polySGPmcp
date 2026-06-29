import httpx
import pytest
import respx

from polymarket.client import GammaClient, WEATHER_TAG_ID
from polymarket.markets import filter_weather_markets, score_market
from polymarket.models import Market


@pytest.mark.asyncio
async def test_get_markets(sample_gamma_markets):
    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(200, json=sample_gamma_markets)
        )
        client = GammaClient()
        markets = await client.get_markets()

    assert len(markets) == 4
    assert markets[0].question == "Will a Category 4+ hurricane hit Florida in 2026?"


@pytest.mark.asyncio
async def test_get_all_markets_pagination(sample_gamma_markets):
    with respx.mock:
        route = respx.get("https://gamma-api.polymarket.com/markets")
        route.side_effect = [
            httpx.Response(200, json=sample_gamma_markets),
            httpx.Response(200, json=[]),
        ]
        client = GammaClient()
        markets = await client.get_all_markets(max_pages=3, page_size=100)

    assert len(markets) == 4


def test_score_market_weather():
    m = Market(
        id="1",
        question="Will a hurricane hit Florida?",
        description="Category 4 hurricane landfall prediction",
    )
    result = score_market(m)
    assert result is not None
    assert result.relevance_score == 1.0
    assert "hurricane" in result.matched_keywords


def test_score_market_non_weather():
    m = Market(
        id="2",
        question="Will Bitcoin reach $100k?",
        description="BTC price prediction",
    )
    result = score_market(m)
    assert result is None


def test_score_market_multiple_keywords():
    m = Market(
        id="3",
        question="Will NYC temperature exceed 110°F?",
        description="Heat wave record in fahrenheit",
    )
    result = score_market(m)
    assert result is not None
    assert len(result.matched_keywords) >= 2
    assert result.relevance_score > 0.5


def test_filter_weather_markets(sample_gamma_markets):
    markets = [Market.model_validate(m) for m in sample_gamma_markets]
    results = filter_weather_markets(markets)

    assert len(results) == 3
    assert results[0].relevance_score >= results[1].relevance_score
    questions = [wm.market.question for wm in results]
    assert not any("Bitcoin" in q for q in questions)


def test_filter_with_min_score(sample_gamma_markets):
    markets = [Market.model_validate(m) for m in sample_gamma_markets]
    results = filter_weather_markets(markets, min_score=0.9)
    assert all(wm.relevance_score >= 0.9 for wm in results)


@pytest.mark.asyncio
async def test_get_events_by_tag(sample_temperature_events):
    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/events").mock(
            return_value=httpx.Response(200, json=sample_temperature_events)
        )
        client = GammaClient()
        events = await client.get_events_by_tag(WEATHER_TAG_ID)

    assert len(events) == 3
    assert events[0].title == "Highest temperature in NYC on June 26?"
    assert len(events[0].markets) == 3


@pytest.mark.asyncio
async def test_get_events_by_tag_pagination(sample_temperature_events):
    with respx.mock:
        route = respx.get("https://gamma-api.polymarket.com/events")
        route.side_effect = [
            httpx.Response(200, json=sample_temperature_events),
            httpx.Response(200, json=[]),
        ]
        client = GammaClient()
        events = await client.get_events_by_tag(WEATHER_TAG_ID, max_pages=3)

    assert len(events) == 3
