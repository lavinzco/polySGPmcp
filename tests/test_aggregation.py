from __future__ import annotations

import pytest

from agent.aggregation import AggregatedSignal, aggregate_signals
from agent.models import TradeSignal


def _ts(action: str = "buy_yes", confidence: float = 0.8, size: float = 50.0) -> TradeSignal:
    return TradeSignal(
        market_id="mkt-001",
        action=action,
        confidence=confidence,
        suggested_size_usd=size,
        rationale=f"Test {action} @ {confidence}",
        weather_factors=["test"],
        quality="high",
    )


class TestAggregateSignals:
    def test_empty_samples(self):
        result = aggregate_signals([])
        assert result.action == "hold"
        assert result.agreement_ratio == 0.0
        assert result.n_samples == 0

    def test_unanimous_buy_yes(self):
        samples = [_ts("buy_yes", 0.8, 50), _ts("buy_yes", 0.7, 30), _ts("buy_yes", 0.9, 60)]
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "buy_yes"
        assert result.agreement_ratio == 1.0
        assert result.confidence == 0.8  # median of [0.7, 0.8, 0.9]
        assert result.suggested_size_usd == 50.0  # median of [30, 50, 60]
        assert result.n_samples == 3
        assert len(result.raw_samples) == 3

    def test_50_50_split_degrades_to_hold(self):
        samples = [
            _ts("hold", 0.0, 0),
            _ts("hold", 0.0, 0),
            _ts("buy_no", 0.85, 50),
            _ts("buy_no", 0.90, 50),
        ]
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "hold"
        assert result.agreement_ratio == 0.5
        assert result.confidence == 0.0
        assert result.suggested_size_usd == 0.0
        assert "agreement too low" in result.rationale
        assert len(result.raw_samples) == 4

    def test_80_percent_agreement_passes(self):
        samples = [
            _ts("buy_yes", 0.70, 50),
            _ts("buy_yes", 0.80, 25),
            _ts("buy_yes", 0.75, 50),
            _ts("buy_yes", 0.85, 100),
            _ts("hold", 0.0, 0),
        ]
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "buy_yes"
        assert result.agreement_ratio == 0.8
        # Median of majority [0.70, 0.75, 0.80, 0.85]
        assert result.confidence == pytest.approx(0.775)
        # Median of majority [25, 50, 50, 100]
        assert result.suggested_size_usd == pytest.approx(50.0)

    def test_majority_median_excludes_minority(self):
        """Minority extreme values don't affect majority median."""
        samples = [
            _ts("buy_yes", 0.70, 30),
            _ts("buy_yes", 0.75, 40),
            _ts("buy_yes", 0.80, 50),
            _ts("hold", 0.0, 0),  # minority — shouldn't affect median
        ]
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "buy_yes"
        # Median of majority only: [0.70, 0.75, 0.80]
        assert result.confidence == 0.75
        assert result.suggested_size_usd == 40.0

    def test_exactly_at_min_agreement_passes(self):
        # 7/10 = 0.7 exactly
        samples = [_ts("buy_yes", 0.8, 50)] * 7 + [_ts("hold", 0.0, 0)] * 3
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "buy_yes"
        assert result.agreement_ratio == 0.7

    def test_just_below_min_agreement_degrades(self):
        # 6/10 = 0.6 < 0.7
        samples = [_ts("buy_yes", 0.8, 50)] * 6 + [_ts("hold", 0.0, 0)] * 4
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "hold"
        assert result.agreement_ratio == 0.6

    def test_three_way_split_degrades(self):
        samples = [
            _ts("buy_yes", 0.8, 50),
            _ts("buy_yes", 0.7, 40),
            _ts("buy_no", 0.85, 50),
            _ts("buy_no", 0.9, 50),
            _ts("hold", 0.0, 0),
        ]
        result = aggregate_signals(samples, min_agreement=0.7)

        assert result.action == "hold"
        assert result.agreement_ratio == 0.4  # buy_yes and buy_no tied at 2/5

    def test_rationale_from_closest_to_median(self):
        samples = [
            _ts("buy_yes", 0.70, 50),
            _ts("buy_yes", 0.90, 50),
            _ts("buy_yes", 0.80, 50),
        ]
        result = aggregate_signals(samples, min_agreement=0.7)

        # median confidence = 0.80, so rationale from the 0.80 sample
        assert "0.8" in result.rationale

    def test_quality_preserved(self):
        s = _ts("buy_yes", 0.8, 50)
        s = s.model_copy(update={"quality": "medium"})
        result = aggregate_signals([s, s, s], min_agreement=0.7)

        assert result.quality == "medium"
