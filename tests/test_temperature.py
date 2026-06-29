from __future__ import annotations

import math

import pytest

from polymarket.models import Event, Market
from polymarket.temperature import (
    TemperatureMarket,
    find_temperature_markets,
    find_temperature_markets_from_events,
    is_temperature_event,
    parse_temperature_market,
)


def _make_market(question: str, desc: str = "", market_id: str = "t-1",
                 prices: str = "[]") -> Market:
    return Market(id=market_id, question=question, description=desc,
                  outcomePrices=prices)


class TestParseTemperatureMarket:
    # --- "or below" / "or above" ---
    def test_or_below_format(self):
        m = _make_market(
            "Will the highest temperature in NYC be 16°F or below on January 22?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "NYC"
        assert tm.date == "January 22"
        assert tm.threshold_temp == 16.0
        assert tm.threshold_unit == "F"
        assert tm.direction == "below"
        assert math.isinf(tm.bucket_width)

    def test_or_above_format(self):
        m = _make_market(
            "Will the highest temperature in NYC be 40°F or above on January 22?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.direction == "above"
        assert math.isinf(tm.bucket_width)

    # --- "or higher" (maps to above) ---
    def test_or_higher_format(self):
        m = _make_market(
            "Will the highest temperature in Austin be 106°F or higher on June 25?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Austin"
        assert tm.threshold_temp == 106.0
        assert tm.direction == "above"
        assert math.isinf(tm.bucket_width)

    def test_or_higher_celsius(self):
        m = _make_market(
            "Will the highest temperature in Seoul be 31°C or higher on June 26?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Seoul"
        assert tm.direction == "above"

    def test_lowest_or_higher(self):
        m = _make_market(
            "Will the lowest temperature in Miami be 88°F or higher on June 25?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Miami"
        assert tm.direction == "above"

    # --- "between X-Y°F" ---
    def test_between_range_format(self):
        m = _make_market(
            "Will the highest temperature in NYC be between 23-24°F on January 22?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "NYC"
        assert tm.threshold_temp == 23.0
        assert tm.threshold_temp_high == 24.0
        assert tm.direction == "between"
        assert tm.bucket_width == 2.0

    def test_between_wider_range(self):
        m = _make_market(
            "Will the highest temperature in NYC be between 72-73°F on June 25?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.bucket_width == 2.0

    # --- exact single-value (°C cities) ---
    def test_exact_celsius(self):
        m = _make_market(
            "Will the highest temperature in London be 33°C on June 26?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "London"
        assert tm.threshold_temp == 33.0
        assert tm.threshold_unit == "C"
        assert tm.direction == "exact"
        assert tm.bucket_width == 1.0

    def test_exact_celsius_lowest(self):
        m = _make_market(
            "Will the lowest temperature in Tokyo be 21°C on June 26?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Tokyo"
        assert tm.threshold_temp == 21.0
        assert tm.direction == "exact"
        assert tm.bucket_width == 1.0

    def test_exact_celsius_various_cities(self):
        for city in ["Seoul", "Hong Kong", "Shanghai", "Jinan"]:
            m = _make_market(
                f"Will the highest temperature in {city} be 25°C on June 26?"
            )
            tm = parse_temperature_market(m)
            assert tm is not None, f"Failed to parse for {city}"
            assert tm.city == city
            assert tm.direction == "exact"

    # --- Legacy "high in" short form ---
    def test_high_short_form(self):
        m = _make_market(
            "Will the high in Washington DC be 19°F or below on January 20?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Washington DC"
        assert tm.threshold_temp == 19.0
        assert tm.direction == "below"

    def test_high_between_format(self):
        m = _make_market(
            "Will the high in Washington DC be between 26-27°F on January 20?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "Washington DC"
        assert tm.direction == "between"

    # --- Legacy "exceed" format ---
    def test_legacy_exceed_format(self):
        m = _make_market(
            "Will the high temperature in New York on July 4, 2026 exceed 95°F?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.city == "New York"
        assert tm.date == "July 4, 2026"
        assert tm.threshold_temp == 95.0
        assert tm.direction == "above"

    # --- Unit conversion ---
    def test_unit_conversion_f_to_c(self):
        m = _make_market(
            "Will the highest temperature in NYC be 100°F or above on July 10?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert abs(tm.threshold_c - 37.78) < 0.1
        assert tm.threshold_f == 100.0

    def test_unit_conversion_c_to_f(self):
        m = _make_market(
            "Will the highest temperature in London be 35°C on June 26?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.threshold_c == 35.0
        assert abs(tm.threshold_f - 95.0) < 0.1

    # --- bucket_width_c property ---
    def test_bucket_width_c_for_celsius(self):
        m = _make_market(
            "Will the highest temperature in Tokyo be 28°C on June 26?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.bucket_width_c == 1.0

    def test_bucket_width_c_for_fahrenheit(self):
        m = _make_market(
            "Will the highest temperature in NYC be between 72-73°F on June 25?"
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert abs(tm.bucket_width_c - 2 * 5 / 9) < 0.01

    # --- Prices ---
    def test_prices_parsed(self):
        m = _make_market(
            "Will the highest temperature in NYC be 75°F or below on June 26?",
            prices='["0.15", "0.85"]',
        )
        tm = parse_temperature_market(m)
        assert tm is not None
        assert tm.outcome_yes_price == 0.15
        assert tm.outcome_no_price == 0.85

    # --- Non-matches ---
    def test_non_temperature_market_returns_none(self):
        m = _make_market("Will Bitcoin reach $100k by 2026?")
        assert parse_temperature_market(m) is None

    def test_hurricane_market_returns_none(self):
        m = _make_market("Will a hurricane hit Florida in 2026?")
        assert parse_temperature_market(m) is None


class TestIsTemperatureEvent:
    def test_temperature_event(self):
        ev = Event(title="Highest temperature in NYC on June 26?")
        assert is_temperature_event(ev)

    def test_lowest_temperature_event(self):
        ev = Event(title="Lowest temperature in London on January 5?")
        assert is_temperature_event(ev)

    def test_non_temperature_event(self):
        ev = Event(title="Where will 2026 rank among the hottest years on record?")
        assert not is_temperature_event(ev)

    def test_temperature_increase_event(self):
        ev = Event(title="April 2024 Temperature Increase (C)")
        assert not is_temperature_event(ev)


class TestFindTemperatureMarkets:
    def test_filters_only_temperature(self):
        markets = [
            _make_market("Will the highest temperature in NYC be 95°F or above on July 4?", market_id="1"),
            _make_market("Will Bitcoin hit 100k?", market_id="2"),
            _make_market("Will the highest temperature in London be 33°C on June 30?", market_id="3"),
            _make_market("Will a hurricane hit FL?", market_id="4"),
            _make_market("Will the highest temperature in Austin be 106°F or higher on June 25?", market_id="5"),
        ]
        results = find_temperature_markets(markets)
        assert len(results) == 3

    def test_empty_input(self):
        assert find_temperature_markets([]) == []


class TestFindTemperatureMarketsFromEvents:
    def test_filters_temperature_events(self, sample_temperature_events):
        events = [Event.model_validate(e) for e in sample_temperature_events]
        results = find_temperature_markets_from_events(events)
        assert len(results) == 5
        cities = {r.city for r in results}
        assert "NYC" in cities
        assert "London" in cities

    def test_skips_non_temperature_events(self, sample_temperature_events):
        events = [Event.model_validate(e) for e in sample_temperature_events]
        results = find_temperature_markets_from_events(events)
        market_ids = {r.market.id for r in results}
        assert "m-6" not in market_ids

    def test_empty_input(self):
        assert find_temperature_markets_from_events([]) == []
