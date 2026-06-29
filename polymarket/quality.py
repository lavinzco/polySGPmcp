from __future__ import annotations

from typing import Literal

from common.config import settings
from polymarket.temperature import TemperatureMarket


QualityTier = Literal["high", "medium", "low"]


def classify_event_quality(yes_sum: float) -> QualityTier:
    deviation = abs(yes_sum - 1.0)
    if deviation <= settings.quality_high_threshold:
        return "high"
    if deviation <= settings.quality_low_threshold:
        return "medium"
    return "low"


def compute_yes_sum(markets: list[TemperatureMarket]) -> float:
    return sum(m.outcome_yes_price for m in markets)


def normalize_market_prices(markets: list[TemperatureMarket]) -> list[TemperatureMarket]:
    yes_sum = compute_yes_sum(markets)
    if yes_sum <= 0:
        return markets
    return [
        m.model_copy(update={
            "outcome_yes_price": m.outcome_yes_price / yes_sum,
            "outcome_no_price": 1.0 - m.outcome_yes_price / yes_sum,
        })
        for m in markets
    ]


def annotate_quality(
    markets_by_event: dict[str, list[TemperatureMarket]],
) -> list[TemperatureMarket]:
    results: list[TemperatureMarket] = []
    for event_key, markets in markets_by_event.items():
        yes_sum = compute_yes_sum(markets)
        tier = classify_event_quality(yes_sum)

        if tier == "medium":
            markets = normalize_market_prices(markets)

        for m in markets:
            updated = m.model_copy(update={
                "quality": tier,
                "skip_trading": tier == "low",
            })
            results.append(updated)

    return results
