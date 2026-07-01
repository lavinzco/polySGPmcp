from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from agent.aggregation import AggregatedSignal
from agent.calibration.analyze import analyze_decisions
from agent.calibration.collector import (
    collect_confidence_samples,
    parse_provider_arg,
)
from agent.calibration.daily_report import format_report
from agent.calibration.db import CalibrationDB
from agent.calibration.models import CalibrationSample, ProviderConfig
from agent.memory import DecisionLog
from agent.models import WeatherForecast
from polymarket.models import Market
from polymarket.temperature import TemperatureMarket


# --- Fixtures ---

@pytest.fixture
def cal_db(tmp_path):
    """Legacy CalibrationDB fixture — only for collector tests."""
    db = CalibrationDB(tmp_path / "test_cal.db")
    yield db
    db.close()


@pytest.fixture
def decision_db(tmp_path):
    db = DecisionLog(tmp_path / "test_decisions.db")
    yield db
    db.close()


def _make_weather() -> WeatherForecast:
    return WeatherForecast(
        location="Miami",
        temp_c=33.0, temp_f=91.4, humidity=80,
        wind_speed_kmph=20, wind_dir="SE",
        weather_desc="Sunny", feels_like_c=36.0,
        pressure_mb=1013, precip_mm=0, visibility_km=10, uv_index=8,
    )


def _make_temp_market(market_id: str = "tm-1", city: str = "Miami") -> TemperatureMarket:
    return TemperatureMarket(
        market=Market(
            id=market_id,
            question=f"Will the high temperature in {city} on July 1 exceed 95°F?",
            description="Temperature prediction",
            outcomePrices='["0.45", "0.55"]',
        ),
        city=city,
        date="July 1",
        threshold_temp=95.0,
        threshold_unit="F",
        direction="above",
        outcome_yes_price=0.45,
        outcome_no_price=0.55,
    )


def _make_signal(
    market_id: str = "m1",
    action: str = "buy_yes",
    confidence: float = 0.8,
) -> AggregatedSignal:
    return AggregatedSignal(
        market_id=market_id,
        action=action,
        confidence=confidence,
        suggested_size_usd=25.0,
        rationale="Test signal",
        agreement_ratio=1.0,
        n_samples=1,
    )


def _log_decision(
    db: DecisionLog,
    market_id: str = "m1",
    action: str = "buy_yes",
    confidence: float = 0.8,
    city: str = "Miami",
) -> None:
    db.log_decision(
        weather_snapshot={"location": city, "temp_c": 33.0},
        market_snapshot={"id": market_id, "question": "temp?"},
        llm_raw_outputs=['{"action":"buy_yes"}'],
        final_signal=_make_signal(market_id=market_id, action=action, confidence=confidence),
        risk_decision="approved",
    )


def _make_sample(
    market_id: str = "tm-1",
    provider: str = "deepseek-chat",
    action: str = "buy_yes",
    confidence: float = 0.8,
    settled: bool = False,
    outcome: str | None = None,
) -> CalibrationSample:
    return CalibrationSample(
        market_id=market_id,
        provider_name=provider,
        model_name=provider,
        city="Miami",
        date="July 1",
        threshold_temp=95.0,
        threshold_unit="F",
        direction="above",
        market_yes_price=0.45,
        llm_action=action,
        llm_confidence=confidence,
        llm_rationale="test",
        llm_raw_output='{"action":"buy_yes"}',
        weather_snapshot_json="{}",
        settled=settled,
        actual_outcome=outcome,
    )


# --- CalibrationDB tests (legacy, collector still uses it) ---

class TestCalibrationDB:
    def test_insert_and_retrieve(self, cal_db):
        sample = _make_sample()
        cal_db.insert_sample(sample)
        rows = cal_db.get_all_samples()
        assert len(rows) == 1
        assert rows[0]["provider_name"] == "deepseek-chat"
        assert rows[0]["llm_confidence"] == 0.8

    def test_unsettled_market_ids(self, cal_db):
        cal_db.insert_sample(_make_sample(market_id="a"))
        cal_db.insert_sample(_make_sample(market_id="b"))
        cal_db.insert_sample(_make_sample(market_id="a", settled=True, outcome="YES"))
        ids = cal_db.get_unsettled_market_ids()
        assert "b" in ids

    def test_settle_market(self, cal_db):
        cal_db.insert_sample(_make_sample(market_id="x", provider="p1"))
        cal_db.insert_sample(_make_sample(market_id="x", provider="p2"))
        count = cal_db.settle_market("x", "YES")
        assert count == 2
        settled = cal_db.get_settled_samples()
        assert len(settled) == 2
        assert all(s["actual_outcome"] == "YES" for s in settled)

    def test_get_provider_names(self, cal_db):
        cal_db.insert_sample(_make_sample(provider="alpha"))
        cal_db.insert_sample(_make_sample(provider="beta"))
        names = cal_db.get_provider_names()
        assert names == ["alpha", "beta"]


# --- Collector tests ---

@pytest.mark.asyncio
async def test_collect_dry_run(cal_db):
    providers = [
        ProviderConfig(name="test-model", provider_type="openai_compatible",
                       model="test", base_url="http://fake"),
    ]
    samples = await collect_confidence_samples(
        provider_configs=providers,
        temperature_markets=[_make_temp_market()],
        weather_data={"Miami": _make_weather()},
        db=cal_db,
        dry_run=True,
    )
    assert len(samples) == 1
    assert samples[0].llm_rationale == "[dry-run]"
    assert len(cal_db.get_all_samples()) == 0


@pytest.mark.asyncio
async def test_collect_with_mock_provider(cal_db):
    raw_response = json.dumps({
        "action": "buy_yes",
        "confidence": 0.75,
        "suggested_size_usd": 20,
        "rationale": "Looks hot",
        "weather_factors": ["temp 33C"],
    })

    provider_cfg = ProviderConfig(
        name="mock-model", provider_type="openai_compatible",
        model="mock", base_url="http://fake", api_key="fake",
    )

    from unittest.mock import patch
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=raw_response)

    with patch("agent.calibration.collector._build_provider_from_config", return_value=mock_provider):
        samples = await collect_confidence_samples(
            provider_configs=[provider_cfg],
            temperature_markets=[_make_temp_market()],
            weather_data={"Miami": _make_weather()},
            db=cal_db,
            n_repeats=2,
        )

    assert len(samples) == 2
    assert samples[0].llm_action == "buy_yes"
    assert samples[0].llm_confidence == 0.75
    assert len(cal_db.get_all_samples()) == 2


def test_parse_provider_arg():
    cfg = parse_provider_arg("deepseek-chat")
    assert cfg.name == "deepseek-chat"
    assert cfg.provider_type == "openai_compatible"
    assert cfg.model == "deepseek-chat"

    cfg2 = parse_provider_arg("claude-sonnet-4-6")
    assert cfg2.provider_type == "anthropic"
    assert cfg2.model == "claude-sonnet-4-6"


# --- Analyze tests (using decisions table) ---

def test_analyze_empty(decision_db):
    report = analyze_decisions(decision_db)
    assert report.total_markets == 0
    assert report.total_decisions == 0


def test_analyze_with_settled_data(decision_db):
    _log_decision(decision_db, market_id="m1", action="buy_yes", confidence=0.85)
    _log_decision(decision_db, market_id="m2", action="buy_no", confidence=0.7)
    _log_decision(decision_db, market_id="m3", action="buy_yes", confidence=0.9)

    decision_db.settle_market("m1", "YES")
    decision_db.settle_market("m2", "NO")
    decision_db.settle_market("m3", "NO")

    report = analyze_decisions(decision_db)
    assert report.settled_markets == 3
    assert report.action_stats.overall_accuracy is not None
    assert abs(report.action_stats.overall_accuracy - 2 / 3) < 0.01


# --- Report format tests ---

def test_report_format(decision_db):
    _log_decision(decision_db, market_id="m1", action="buy_yes", confidence=0.8)
    _log_decision(decision_db, market_id="m2", action="buy_no", confidence=0.7)
    _log_decision(decision_db, market_id="m3", action="hold", confidence=0.3)

    decision_db.settle_market("m1", "YES")

    report = analyze_decisions(decision_db)
    text = format_report(report, "2026-06-26")

    assert "2026-06-26" in text
    assert "Decisions: 3" in text
    assert "ACTION DISTRIBUTION" in text
    assert "buy_yes=1" in text
    assert "Accuracy" in text


import sys
