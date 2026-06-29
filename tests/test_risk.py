from __future__ import annotations

import pytest

from agent.aggregation import AggregatedSignal
from agent.models import PortfolioState
from agent.risk import RiskManager


def _make_signal(
    action: str = "buy_yes",
    confidence: float = 0.8,
    size: float = 25.0,
    quality: str = "high",
    agreement_ratio: float = 1.0,
) -> AggregatedSignal:
    return AggregatedSignal(
        market_id="mkt-001",
        action=action,
        confidence=confidence,
        suggested_size_usd=size,
        rationale="Test signal",
        weather_factors=["test"],
        quality=quality,
        agreement_ratio=agreement_ratio,
        n_samples=5,
    )


def _make_portfolio(daily_pnl: float = 0.0) -> PortfolioState:
    return PortfolioState(total_balance_usd=1000, daily_pnl_usd=daily_pnl)


class TestRiskManager:
    def test_passes_valid_signal(self):
        rm = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(confidence=0.8, size=25)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.action == "buy_yes"
        assert result.suggested_size_usd == 25.0

    def test_blocks_low_confidence(self):
        rm = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(confidence=0.5)
        result = rm.filter(signal, _make_portfolio())

        assert result is None

    def test_confidence_at_threshold_passes(self):
        rm = RiskManager(min_confidence=0.6)
        signal = _make_signal(confidence=0.6)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None

    def test_caps_oversized_position(self):
        rm = RiskManager(max_position_usd=30, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(confidence=0.9, size=100)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 30.0
        assert result.action == "buy_yes"

    def test_blocks_when_daily_loss_exceeded(self):
        rm = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(confidence=0.95, size=10)
        portfolio = _make_portfolio(daily_pnl=-100)
        result = rm.filter(signal, portfolio)

        assert result is None

    def test_allows_when_daily_loss_not_yet_hit(self):
        rm = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(confidence=0.8, size=20)
        portfolio = _make_portfolio(daily_pnl=-99)
        result = rm.filter(signal, portfolio)

        assert result is not None

    def test_hold_passes_through(self):
        rm = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
        signal = _make_signal(action="hold", confidence=0.0, size=0)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.action == "hold"

    def test_medium_tier_size_discount(self):
        rm = RiskManager(
            max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6,
            medium_tier_size_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, quality="medium")
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 20.0

    def test_medium_tier_custom_multiplier(self):
        rm = RiskManager(
            max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6,
            medium_tier_size_multiplier=0.3,
        )
        signal = _make_signal(confidence=0.9, size=30.0, quality="medium")
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert abs(result.suggested_size_usd - 9.0) < 1e-9

    def test_high_tier_no_discount(self):
        rm = RiskManager(
            max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6,
            medium_tier_size_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, quality="high")
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 40.0

    def test_medium_tier_discount_then_cap(self):
        rm = RiskManager(
            max_position_usd=20, max_daily_loss_usd=100, min_confidence=0.6,
            medium_tier_size_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=50.0, quality="medium")
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 20.0

    # --- Agreement ratio tests ---

    def test_weak_agreement_size_discount(self):
        rm = RiskManager(
            max_position_usd=100, min_confidence=0.6,
            weak_agreement_threshold=0.8, weak_agreement_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, agreement_ratio=0.75)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 20.0  # 40 * 0.5

    def test_strong_agreement_no_discount(self):
        rm = RiskManager(
            max_position_usd=100, min_confidence=0.6,
            weak_agreement_threshold=0.8, weak_agreement_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, agreement_ratio=0.9)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 40.0

    def test_agreement_at_threshold_no_discount(self):
        rm = RiskManager(
            max_position_usd=100, min_confidence=0.6,
            weak_agreement_threshold=0.8, weak_agreement_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, agreement_ratio=0.8)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 40.0

    def test_weak_agreement_plus_medium_tier_stacks(self):
        rm = RiskManager(
            max_position_usd=100, min_confidence=0.6,
            medium_tier_size_multiplier=0.5,
            weak_agreement_threshold=0.8, weak_agreement_multiplier=0.5,
        )
        signal = _make_signal(
            confidence=0.8, size=80.0,
            quality="medium", agreement_ratio=0.75,
        )
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 20.0  # 80 * 0.5(medium) * 0.5(weak)

    def test_weak_agreement_discount_then_cap(self):
        rm = RiskManager(
            max_position_usd=15, min_confidence=0.6,
            weak_agreement_threshold=0.8, weak_agreement_multiplier=0.5,
        )
        signal = _make_signal(confidence=0.8, size=40.0, agreement_ratio=0.75)
        result = rm.filter(signal, _make_portfolio())

        assert result is not None
        assert result.suggested_size_usd == 15.0  # 40*0.5=20, capped to 15
