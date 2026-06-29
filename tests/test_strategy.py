from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agent.models import GammaMarket, WeatherForecast
from agent.strategy import StrategyEngine
from common.llm.router import LLMRouter, TaskType


def _make_weather() -> WeatherForecast:
    return WeatherForecast(
        location="Miami",
        temp_c=33.0,
        temp_f=91.4,
        humidity=80,
        wind_speed_kmph=45,
        wind_dir="ESE",
        weather_desc="Heavy rain",
        feels_like_c=38.0,
        pressure_mb=1005,
        precip_mm=12.5,
        visibility_km=5,
        uv_index=3,
    )


def _make_market(quality: str = "high") -> GammaMarket:
    return GammaMarket(
        id="mkt-001",
        question="Will a hurricane hit Florida on July 4?",
        description="Resolves YES if a named hurricane makes landfall in FL.",
        outcome_yes_price=0.35,
        outcome_no_price=0.65,
        liquidity_usd=50000,
        volume_usd=120000,
        end_date="2026-07-31",
        matched_keywords=["hurricane"],
        quality=quality,
    )


def _mock_router_fixed(raw_output: str, n_repeats: int = 5) -> LLMRouter:
    """All n calls return same output."""
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=raw_output)
    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider
    return router


def _mock_router_sequence(outputs: list[str]) -> LLMRouter:
    """Return different outputs for each call."""
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=outputs)
    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider
    return router


def _buy_yes_json(conf: float = 0.82, size: float = 25.0) -> str:
    return json.dumps({
        "action": "buy_yes",
        "confidence": conf,
        "suggested_size_usd": size,
        "rationale": "Strong weather edge.",
        "weather_factors": ["wind", "pressure"],
    })


def _hold_json(conf: float = 0.0) -> str:
    return json.dumps({
        "action": "hold",
        "confidence": conf,
        "suggested_size_usd": 0,
        "rationale": "Insufficient signal.",
        "weather_factors": [],
    })


def _buy_no_json(conf: float = 0.85, size: float = 50.0) -> str:
    return json.dumps({
        "action": "buy_no",
        "confidence": conf,
        "suggested_size_usd": size,
        "rationale": "Weather contradicts market.",
        "weather_factors": ["temp"],
    })


@pytest.mark.asyncio
async def test_unanimous_buy_yes():
    router = _mock_router_fixed(_buy_yes_json())
    engine = StrategyEngine(router, n_repeats=5)
    signal, raw_outs = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "buy_yes"
    assert signal.confidence == 0.82
    assert signal.agreement_ratio == 1.0
    assert signal.n_samples == 5
    assert len(raw_outs) == 5
    assert len(signal.raw_samples) == 5


@pytest.mark.asyncio
async def test_unanimous_hold():
    router = _mock_router_fixed(_hold_json())
    engine = StrategyEngine(router, n_repeats=5)
    signal, _ = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "hold"
    assert signal.agreement_ratio == 1.0


@pytest.mark.asyncio
async def test_n_repeats_exact_call_count():
    """Verify LLM is called exactly n_repeats times."""
    router = _mock_router_fixed(_buy_yes_json())
    mock_provider = router._providers[TaskType.STRATEGY]
    engine = StrategyEngine(router, n_repeats=5)
    await engine.evaluate(_make_weather(), _make_market())

    assert mock_provider.complete.call_count == 5


@pytest.mark.asyncio
async def test_split_50_50_degrades_to_hold():
    """Scenario 1: 50/50 split should degrade to hold (below 0.7 agreement)."""
    outputs = [
        _hold_json(), _hold_json(),
        _buy_no_json(), _buy_no_json(),
    ]
    router = _mock_router_sequence(outputs)
    engine = StrategyEngine(router, n_repeats=4)
    signal, _ = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "hold"
    assert signal.agreement_ratio == 0.5
    assert signal.confidence == 0.0


@pytest.mark.asyncio
async def test_80_percent_agreement_returns_majority():
    """Scenario 2: 4/5 agree → returns majority action with median stats."""
    outputs = [
        _buy_yes_json(conf=0.70, size=50),
        _buy_yes_json(conf=0.80, size=25),
        _buy_yes_json(conf=0.75, size=50),
        _buy_yes_json(conf=0.85, size=100),
        _hold_json(),
    ]
    router = _mock_router_sequence(outputs)
    engine = StrategyEngine(router, n_repeats=5)
    signal, _ = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "buy_yes"
    assert signal.agreement_ratio == 0.8
    assert signal.confidence == pytest.approx(0.775)  # median of [0.70, 0.75, 0.80, 0.85]
    assert signal.suggested_size_usd == pytest.approx(50.0)  # median of [25, 50, 50, 100]


@pytest.mark.asyncio
async def test_low_quality_short_circuits_no_llm_call():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value="should not be called")
    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    market = _make_market()
    market.quality = "low"

    engine = StrategyEngine(router, n_repeats=5)
    signal, raw_outs = await engine.evaluate(_make_weather(), market)

    assert signal.action == "hold"
    assert signal.quality == "low"
    assert "quality=low" in signal.rationale
    assert raw_outs == []
    mock_provider.complete.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_json_degrades_all_samples_to_hold():
    router = _mock_router_fixed("This is not valid JSON at all")
    engine = StrategyEngine(router, n_repeats=3)
    signal, raw_outs = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "hold"
    assert signal.agreement_ratio == 1.0
    assert len(raw_outs) == 3


@pytest.mark.asyncio
async def test_quality_propagated_to_aggregated_signal():
    router = _mock_router_fixed(_buy_yes_json())
    engine = StrategyEngine(router, n_repeats=3)

    market = _make_market(quality="medium")
    signal, _ = await engine.evaluate(_make_weather(), market)

    assert signal.quality == "medium"


@pytest.mark.asyncio
async def test_llm_exception_degrades_sample_to_hold():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    engine = StrategyEngine(router, n_repeats=3)
    signal, raw_outs = await engine.evaluate(_make_weather(), _make_market())

    assert signal.action == "hold"
    assert signal.n_samples == 3
