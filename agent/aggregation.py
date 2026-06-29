from __future__ import annotations

import os
import statistics

from pydantic import BaseModel, Field

from agent.models import TradeSignal


class AggregatedSignal(BaseModel):
    market_id: str
    action: str = Field(description="buy_yes | buy_no | hold")
    confidence: float = Field(ge=0, le=1)
    suggested_size_usd: float = Field(ge=0)
    rationale: str = ""
    weather_factors: list[str] = []
    quality: str = "high"
    agreement_ratio: float = Field(ge=0, le=1, description="fraction of samples agreeing on action")
    raw_samples: list[TradeSignal] = Field(default_factory=list)
    n_samples: int = 0


def aggregate_signals(
    samples: list[TradeSignal],
    *,
    min_agreement: float | None = None,
) -> AggregatedSignal:
    if min_agreement is None:
        min_agreement = float(os.environ.get("AGENT_MIN_AGREEMENT_RATIO", "0.7"))

    if not samples:
        return AggregatedSignal(
            market_id="",
            action="hold",
            confidence=0.0,
            suggested_size_usd=0.0,
            rationale="no samples to aggregate",
            agreement_ratio=0.0,
            n_samples=0,
        )

    action_counts: dict[str, int] = {}
    for s in samples:
        action_counts[s.action] = action_counts.get(s.action, 0) + 1

    majority_action = max(action_counts, key=action_counts.get)  # type: ignore[arg-type]
    agreement_ratio = action_counts[majority_action] / len(samples)

    majority_samples = [s for s in samples if s.action == majority_action]

    if agreement_ratio < min_agreement:
        rationale_parts = [f"{a}: {c}/{len(samples)}" for a, c in sorted(action_counts.items())]
        return AggregatedSignal(
            market_id=samples[0].market_id,
            action="hold",
            confidence=0.0,
            suggested_size_usd=0.0,
            rationale=f"agreement too low ({agreement_ratio:.0%}): {', '.join(rationale_parts)}",
            quality=samples[0].quality,
            agreement_ratio=agreement_ratio,
            raw_samples=samples,
            n_samples=len(samples),
        )

    conf_median = statistics.median([s.confidence for s in majority_samples])
    size_median = statistics.median([s.suggested_size_usd for s in majority_samples])

    all_factors: list[str] = []
    seen: set[str] = set()
    for s in majority_samples:
        for f in s.weather_factors:
            if f not in seen:
                seen.add(f)
                all_factors.append(f)

    best = min(majority_samples, key=lambda s: abs(s.confidence - conf_median))

    return AggregatedSignal(
        market_id=samples[0].market_id,
        action=majority_action,
        confidence=conf_median,
        suggested_size_usd=size_median,
        rationale=best.rationale,
        weather_factors=all_factors,
        quality=samples[0].quality,
        agreement_ratio=agreement_ratio,
        raw_samples=samples,
        n_samples=len(samples),
    )
