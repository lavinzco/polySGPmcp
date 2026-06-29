from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from agent.aggregation import AggregatedSignal
from agent.hermes import HermesAgent, RunStats
from agent.memory import DecisionLog
from agent.models import GammaMarket, TradeSignal, WeatherForecast
from common.llm.router import LLMRouter, TaskType


def _make_weather() -> WeatherForecast:
    return WeatherForecast(
        location="TestCity",
        temp_c=30.0, temp_f=86.0, humidity=50,
        wind_speed_kmph=10, wind_dir="N", weather_desc="Clear",
        feels_like_c=32.0, pressure_mb=1013, precip_mm=0.0,
        visibility_km=10, uv_index=5,
    )


def _make_market(market_id: str = "mkt-001") -> GammaMarket:
    return GammaMarket(
        id=market_id,
        question="Will temp be 30°C on June 28?",
        description="Test market",
        outcome_yes_price=0.50,
        outcome_no_price=0.50,
    )


_HOLD_JSON = json.dumps({
    "action": "hold",
    "confidence": 0.3,
    "suggested_size_usd": 0,
    "rationale": "Unclear edge.",
    "weather_factors": ["temp"],
})

_BUY_JSON = json.dumps({
    "action": "buy_yes",
    "confidence": 0.85,
    "suggested_size_usd": 40.0,
    "rationale": "Strong forecast alignment.",
    "weather_factors": ["temp"],
})


class TestDedupLogic:
    def test_first_evaluation_passes(self, tmp_path):
        memory = DecisionLog(tmp_path / "dedup.db")
        assert not memory.was_evaluated_today("mkt-001")
        memory.close()

    def test_second_evaluation_blocked(self, tmp_path):
        memory = DecisionLog(tmp_path / "dedup.db")
        signal = AggregatedSignal(
            market_id="mkt-001", action="hold", confidence=0.0,
            suggested_size_usd=0.0, rationale="test",
            agreement_ratio=1.0, n_samples=0,
        )
        memory.log_decision(
            weather_snapshot={}, market_snapshot={},
            llm_raw_outputs=[], final_signal=signal,
            risk_decision="approved",
        )
        assert memory.was_evaluated_today("mkt-001")
        memory.close()

    def test_different_market_id_not_blocked(self, tmp_path):
        memory = DecisionLog(tmp_path / "dedup.db")
        signal = AggregatedSignal(
            market_id="mkt-001", action="hold", confidence=0.0,
            suggested_size_usd=0.0, rationale="test",
            agreement_ratio=1.0, n_samples=0,
        )
        memory.log_decision(
            weather_snapshot={}, market_snapshot={},
            llm_raw_outputs=[], final_signal=signal,
            risk_decision="approved",
        )
        assert not memory.was_evaluated_today("mkt-002")
        memory.close()

    def test_cross_day_resets_dedup(self, tmp_path):
        memory = DecisionLog(tmp_path / "dedup.db")
        signal = AggregatedSignal(
            market_id="mkt-001", action="hold", confidence=0.0,
            suggested_size_usd=0.0, rationale="test",
            agreement_ratio=1.0, n_samples=0,
        )
        memory.log_decision(
            weather_snapshot={}, market_snapshot={},
            llm_raw_outputs=[], final_signal=signal,
            risk_decision="approved",
        )

        assert memory.was_evaluated_today("mkt-001")
        assert not memory.was_evaluated_today("mkt-001", date_str="2025-01-01")
        assert not memory.was_evaluated_today("mkt-001", date_str="2099-12-31")
        memory.close()


class TestRunOnceDedup:
    @pytest.mark.asyncio
    async def test_skip_already_evaluated(self, tmp_path):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_HOLD_JSON)

        router = LLMRouter()
        router._providers[TaskType.STRATEGY] = mock_provider

        memory = DecisionLog(tmp_path / "hermes_dedup.db")
        agent = HermesAgent(router=router, memory=memory)

        markets = [_make_market("mkt-A"), _make_market("mkt-B")]

        # First run evaluates both
        results1, stats1 = await agent.run_once(
            _make_weather(), markets, skip_if_evaluated_today=True,
        )
        assert stats1.markets_evaluated == 2
        assert stats1.markets_skipped_dedup == 0
        assert len(results1) == 2

        # Second run skips both
        mock_provider.complete.reset_mock()
        results2, stats2 = await agent.run_once(
            _make_weather(), markets, skip_if_evaluated_today=True,
        )
        assert stats2.markets_evaluated == 0
        assert stats2.markets_skipped_dedup == 2
        assert len(results2) == 0
        mock_provider.complete.assert_not_called()
        memory.close()

    @pytest.mark.asyncio
    async def test_partial_dedup(self, tmp_path):
        """One market already evaluated, one new."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_HOLD_JSON)

        router = LLMRouter()
        router._providers[TaskType.STRATEGY] = mock_provider

        memory = DecisionLog(tmp_path / "partial_dedup.db")
        agent = HermesAgent(router=router, memory=memory)

        # Evaluate only mkt-A
        _, _ = await agent.run_once(
            _make_weather(), [_make_market("mkt-A")],
            skip_if_evaluated_today=True,
        )

        mock_provider.complete.reset_mock()

        # Now run both: mkt-A should be skipped, mkt-B evaluated
        _, stats = await agent.run_once(
            _make_weather(), [_make_market("mkt-A"), _make_market("mkt-B")],
            skip_if_evaluated_today=True,
        )
        assert stats.markets_skipped_dedup == 1
        assert stats.markets_evaluated == 1
        assert stats.skipped_ids == ["mkt-A"]
        memory.close()

    @pytest.mark.asyncio
    async def test_llm_call_count_in_stats(self, tmp_path):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_BUY_JSON)

        router = LLMRouter()
        router._providers[TaskType.STRATEGY] = mock_provider

        memory = DecisionLog(tmp_path / "llm_count.db")
        agent = HermesAgent(router=router, memory=memory)
        agent.strategy._n_repeats = 3

        _, stats = await agent.run_once(
            _make_weather(), [_make_market()],
            skip_if_evaluated_today=True,
        )
        assert stats.llm_calls == 3
        assert stats.markets_evaluated == 1
        memory.close()

    @pytest.mark.asyncio
    async def test_stats_returned_without_dedup(self, tmp_path):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_HOLD_JSON)

        router = LLMRouter()
        router._providers[TaskType.STRATEGY] = mock_provider

        memory = DecisionLog(tmp_path / "stats.db")
        agent = HermesAgent(router=router, memory=memory)

        _, stats = await agent.run_once(_make_weather(), [_make_market()])
        assert stats.markets_scanned == 1
        assert stats.markets_evaluated == 1
        assert stats.markets_skipped_dedup == 0
        memory.close()
