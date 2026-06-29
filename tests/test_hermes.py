from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agent.hermes import HermesAgent
from agent.memory import DecisionLog
from agent.models import GammaMarket, PortfolioState, WeatherForecast
from agent.risk import RiskManager
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


def _make_markets() -> list[GammaMarket]:
    return [
        GammaMarket(
            id="mkt-001",
            question="Will a hurricane hit Florida on July 4?",
            description="Named hurricane landfall in FL",
            outcome_yes_price=0.35,
            outcome_no_price=0.65,
            liquidity_usd=50000,
            volume_usd=120000,
            end_date="2026-07-31",
            matched_keywords=["hurricane"],
        ),
        GammaMarket(
            id="mkt-002",
            question="Will Miami see record rainfall on July 4?",
            description="Monthly rainfall exceeds historical record",
            outcome_yes_price=0.20,
            outcome_no_price=0.80,
            liquidity_usd=15000,
            volume_usd=30000,
            end_date="2026-07-31",
            matched_keywords=["rainfall"],
        ),
    ]


_BUY_YES = json.dumps({
    "action": "buy_yes",
    "confidence": 0.82,
    "suggested_size_usd": 25.0,
    "rationale": "Dropping pressure and high winds suggest tropical development.",
    "weather_factors": ["pressure 1005 mb", "wind 45 km/h", "heavy rain"],
})

_HOLD = json.dumps({
    "action": "hold",
    "confidence": 0.4,
    "suggested_size_usd": 0,
    "rationale": "Current rainfall is elevated but not record-breaking.",
    "weather_factors": ["precip 12.5mm"],
})


@pytest.mark.asyncio
async def test_run_once_end_to_end(tmp_path):
    responses = [_BUY_YES] * 5 + [_HOLD] * 5
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=responses)

    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    memory = DecisionLog(tmp_path / "e2e_test.db")
    risk = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=-20)

    agent = HermesAgent(router=router, risk=risk, memory=memory, portfolio=portfolio)
    results, stats = await agent.run_once(_make_weather(), _make_markets())

    assert len(results) == 2
    assert results[0].action == "buy_yes"
    assert results[0].confidence == 0.82
    assert results[0].agreement_ratio == 1.0
    assert results[1].action == "hold"

    assert stats.markets_scanned == 2
    assert stats.markets_evaluated == 2
    assert stats.markets_skipped_dedup == 0

    decisions = memory.get_recent_decisions(10)
    assert len(decisions) == 2
    assert decisions[0]["risk_decision"] in ("approved", "blocked")
    assert decisions[0]["dry_run"] == 1
    assert decisions[0]["market_id"] != ""
    assert decisions[0]["eval_date"] != ""
    memory.close()


@pytest.mark.asyncio
async def test_run_once_risk_blocks_low_confidence(tmp_path):
    raw = json.dumps({
        "action": "buy_yes",
        "confidence": 0.5,
        "suggested_size_usd": 30.0,
        "rationale": "Marginal signal.",
        "weather_factors": ["wind"],
    })
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=raw)

    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    memory = DecisionLog(tmp_path / "risk_test.db")
    risk = RiskManager(min_confidence=0.6)

    agent = HermesAgent(router=router, risk=risk, memory=memory)
    results, stats = await agent.run_once(_make_weather(), _make_markets()[:1])

    assert len(results) == 1
    assert results[0].action == "hold"
    assert "Risk blocked" in results[0].rationale

    decisions = memory.get_recent_decisions(1)
    assert decisions[0]["risk_decision"] == "blocked"
    memory.close()


@pytest.mark.asyncio
async def test_run_once_llm_failure(tmp_path):
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value="NOT JSON")

    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    memory = DecisionLog(tmp_path / "fail_test.db")
    agent = HermesAgent(router=router, memory=memory)
    results, stats = await agent.run_once(_make_weather(), _make_markets()[:1])

    assert len(results) == 1
    assert results[0].action == "hold"
    memory.close()


@pytest.mark.asyncio
async def test_dedup_skips_already_evaluated(tmp_path):
    """Markets evaluated today should be skipped on second run."""
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=_HOLD)

    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    memory = DecisionLog(tmp_path / "dedup_test.db")
    risk = RiskManager()
    agent = HermesAgent(router=router, risk=risk, memory=memory)

    markets = _make_markets()[:1]

    # First run: should evaluate
    _, stats1 = await agent.run_once(
        _make_weather(), markets, skip_if_evaluated_today=True
    )
    assert stats1.markets_evaluated == 1
    assert stats1.markets_skipped_dedup == 0

    # Reset mock call count
    mock_provider.complete.reset_mock()

    # Second run: should skip
    _, stats2 = await agent.run_once(
        _make_weather(), markets, skip_if_evaluated_today=True
    )
    assert stats2.markets_evaluated == 0
    assert stats2.markets_skipped_dedup == 1
    assert stats2.skipped_ids == ["mkt-001"]
    mock_provider.complete.assert_not_called()
    memory.close()


@pytest.mark.asyncio
async def test_dedup_disabled_by_default(tmp_path):
    """Without skip_if_evaluated_today, markets are always re-evaluated."""
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=_HOLD)

    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    memory = DecisionLog(tmp_path / "nodedup_test.db")
    agent = HermesAgent(router=router, memory=memory)

    markets = _make_markets()[:1]

    await agent.run_once(_make_weather(), markets)
    await agent.run_once(_make_weather(), markets)

    decisions = memory.get_recent_decisions(10)
    assert len(decisions) == 2
    memory.close()
