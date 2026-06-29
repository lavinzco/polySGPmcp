from __future__ import annotations

import math

import pytest

from polymarket.models import Market
from polymarket.quality import (
    annotate_quality,
    classify_event_quality,
    compute_yes_sum,
    normalize_market_prices,
)
from polymarket.temperature import TemperatureMarket


def _tm(yes_price: float, no_price: float | None = None, market_id: str = "t") -> TemperatureMarket:
    if no_price is None:
        no_price = 1.0 - yes_price
    return TemperatureMarket(
        market=Market(id=market_id, question="test"),
        city="Test",
        date="June 1",
        threshold_temp=20.0,
        threshold_unit="C",
        direction="exact",
        bucket_width=1.0,
        outcome_yes_price=yes_price,
        outcome_no_price=no_price,
    )


class TestClassifyEventQuality:
    def test_perfect_sum_is_high(self):
        assert classify_event_quality(1.0) == "high"

    def test_within_2pct_is_high(self):
        assert classify_event_quality(1.019) == "high"
        assert classify_event_quality(0.981) == "high"

    def test_just_over_2pct_is_medium(self):
        assert classify_event_quality(1.025) == "medium"
        assert classify_event_quality(0.975) == "medium"

    def test_between_2_and_6pct_is_medium(self):
        assert classify_event_quality(1.04) == "medium"
        assert classify_event_quality(0.955) == "medium"

    def test_just_over_6pct_is_low(self):
        assert classify_event_quality(1.065) == "low"
        assert classify_event_quality(0.935) == "low"

    def test_above_6pct_is_low(self):
        assert classify_event_quality(1.12) == "low"
        assert classify_event_quality(0.85) == "low"

    def test_extreme_deviation_is_low(self):
        assert classify_event_quality(0.5) == "low"
        assert classify_event_quality(2.0) == "low"


class TestComputeYesSum:
    def test_sums_correctly(self):
        markets = [_tm(0.3), _tm(0.5), _tm(0.2)]
        assert abs(compute_yes_sum(markets) - 1.0) < 1e-9

    def test_empty_list(self):
        assert compute_yes_sum([]) == 0.0


class TestNormalizeMarketPrices:
    def test_normalized_sum_equals_one(self):
        markets = [_tm(0.35), _tm(0.50), _tm(0.19)]
        result = normalize_market_prices(markets)
        total = sum(m.outcome_yes_price for m in result)
        assert abs(total - 1.0) < 1e-9

    def test_preserves_relative_proportions(self):
        markets = [_tm(0.2), _tm(0.4)]
        result = normalize_market_prices(markets)
        assert abs(result[0].outcome_yes_price * 2 - result[1].outcome_yes_price) < 1e-9

    def test_no_prices_unchanged(self):
        markets = [_tm(0.0), _tm(0.0)]
        result = normalize_market_prices(markets)
        assert result[0].outcome_yes_price == 0.0

    def test_yes_plus_no_equals_one(self):
        markets = [_tm(0.35), _tm(0.50), _tm(0.19)]
        result = normalize_market_prices(markets)
        for m in result:
            assert abs(m.outcome_yes_price + m.outcome_no_price - 1.0) < 1e-9

    def test_does_not_mutate_input(self):
        markets = [_tm(0.35), _tm(0.50)]
        normalize_market_prices(markets)
        assert markets[0].outcome_yes_price == 0.35


class TestAnnotateQuality:
    def test_high_quality_event(self):
        markets = [_tm(0.3, market_id="1"), _tm(0.5, market_id="2"), _tm(0.2, market_id="3")]
        result = annotate_quality({"ev1": markets})
        assert all(m.quality == "high" for m in result)
        assert all(not m.skip_trading for m in result)
        # Prices unchanged for high
        assert result[0].outcome_yes_price == 0.3

    def test_medium_quality_gets_normalized(self):
        # Sum = 1.04 → medium → normalize
        markets = [_tm(0.35, market_id="1"), _tm(0.50, market_id="2"), _tm(0.19, market_id="3")]
        result = annotate_quality({"ev1": markets})
        assert all(m.quality == "medium" for m in result)
        assert all(not m.skip_trading for m in result)
        total = sum(m.outcome_yes_price for m in result)
        assert abs(total - 1.0) < 1e-9

    def test_low_quality_skip_trading(self):
        # Sum = 1.12 → low
        markets = [_tm(0.50, market_id="1"), _tm(0.40, market_id="2"), _tm(0.22, market_id="3")]
        result = annotate_quality({"ev1": markets})
        assert all(m.quality == "low" for m in result)
        assert all(m.skip_trading for m in result)
        # Prices NOT normalized for low
        assert result[0].outcome_yes_price == 0.50

    def test_mixed_events(self):
        high_markets = [_tm(0.5, market_id="h1"), _tm(0.5, market_id="h2")]
        low_markets = [_tm(0.5, market_id="l1"), _tm(0.7, market_id="l2")]
        result = annotate_quality({"ev_high": high_markets, "ev_low": low_markets})
        high_result = [m for m in result if m.market.id.startswith("h")]
        low_result = [m for m in result if m.market.id.startswith("l")]
        assert all(m.quality == "high" for m in high_result)
        assert all(m.quality == "low" for m in low_result)
