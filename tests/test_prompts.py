from __future__ import annotations

from agent.models import DayForecast, GammaMarket, WeatherForecast
from agent.prompts import (
    _parse_market_date,
    build_strategy_prompt,
    get_system_prompt_with_schema,
)


def _make_weather(forecast_days: list[DayForecast] | None = None) -> WeatherForecast:
    return WeatherForecast(
        location="Beijing",
        temp_c=35.0,
        temp_f=95.0,
        humidity=60,
        wind_speed_kmph=10,
        wind_dir="S",
        weather_desc="Clear",
        feels_like_c=38.0,
        pressure_mb=1010,
        precip_mm=0.0,
        visibility_km=10,
        uv_index=8,
        forecast_3day=forecast_days or [],
    )


def _make_market(quality: str = "high", question: str | None = None) -> GammaMarket:
    return GammaMarket(
        id="mkt-bj",
        question=question or "Will the highest temperature in Beijing be 36°C on June 28?",
        description="Resolves YES if max temp >= 36°C.",
        outcome_yes_price=0.25,
        outcome_no_price=0.75,
        quality=quality,
    )


_SAMPLE_FORECAST = [
    DayForecast(date="2026-06-26", max_temp_c=34.0, min_temp_c=23.0, avg_humidity=50, total_precip_mm=0.0, condition="Sunny"),
    DayForecast(date="2026-06-27", max_temp_c=36.0, min_temp_c=24.0, avg_humidity=45, total_precip_mm=1.2, condition="Partly cloudy"),
    DayForecast(date="2026-06-28", max_temp_c=35.0, min_temp_c=23.0, avg_humidity=55, total_precip_mm=0.5, condition="Cloudy"),
]


class TestParseMarketDate:
    def test_standard_format(self):
        assert _parse_market_date("Will the highest temperature in Beijing be 36°C on June 28?") == "2026-06-28"

    def test_with_year(self):
        assert _parse_market_date("temperature on January 20, 2026") == "2026-01-20"

    def test_short_month(self):
        assert _parse_market_date("forecast for Jan 5") == "2026-01-05"

    def test_no_date(self):
        assert _parse_market_date("Will it rain tomorrow?") is None


class TestBuildStrategyPrompt:
    def test_high_quality_in_prompt(self):
        prompt = build_strategy_prompt(_make_weather(), _make_market("high"))
        assert "Quality: HIGH" in prompt
        assert "reliable market consensus" in prompt

    def test_medium_quality_warning_in_prompt(self):
        prompt = build_strategy_prompt(_make_weather(), _make_market("medium"))
        assert "Quality: MEDIUM" in prompt
        assert "stronger weather evidence" in prompt.lower()

    def test_default_quality_is_high(self):
        prompt = build_strategy_prompt(_make_weather(), _make_market())
        assert "Quality: HIGH" in prompt

    def test_no_forecast_shows_placeholder(self):
        prompt = build_strategy_prompt(_make_weather(), _make_market())
        assert "(no forecast available)" in prompt

    def test_forecast_date_alignment(self):
        weather = _make_weather(_SAMPLE_FORECAST)
        market = _make_market(question="Will X be 36°C on June 27?")
        prompt = build_strategy_prompt(weather, market)
        assert "TARGET DATE 2026-06-27" in prompt
        assert "max 36.0°C" in prompt

    def test_forecast_beyond_range_warning(self):
        weather = _make_weather(_SAMPLE_FORECAST)
        market = _make_market(question="Will X be 36°C on July 5?")
        prompt = build_strategy_prompt(weather, market)
        assert "BEYOND RELIABLE FORECAST RANGE" in prompt
        assert "2026-07-05" in prompt

    def test_forecast_within_range_no_warning(self):
        weather = _make_weather(_SAMPLE_FORECAST)
        market = _make_market(question="Will X be 36°C on June 28?")
        prompt = build_strategy_prompt(weather, market)
        assert "BEYOND RELIABLE FORECAST RANGE" not in prompt
        assert "TARGET DATE 2026-06-28" in prompt

    def test_all_forecast_days_listed(self):
        weather = _make_weather(_SAMPLE_FORECAST)
        market = _make_market(question="Will X be 36°C on June 27?")
        prompt = build_strategy_prompt(weather, market)
        assert "2026-06-26" in prompt
        assert "2026-06-27" in prompt
        assert "2026-06-28" in prompt


class TestSystemPrompt:
    def test_quality_tier_rule_in_system_prompt(self):
        system = get_system_prompt_with_schema()
        assert "quality=" in system
        assert "high" in system.lower()
        assert "medium" in system.lower()
        assert "STRONGER weather evidence" in system

    def test_iron_rule_6_exists(self):
        system = get_system_prompt_with_schema()
        assert "6." in system
        assert "Market quality tiers" in system

    def test_iron_rule_7_forecast_range(self):
        system = get_system_prompt_with_schema()
        assert "7." in system
        assert "BEYOND RELIABLE FORECAST RANGE" in system
