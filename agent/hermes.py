from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent.aggregation import AggregatedSignal
from agent.manual_input import DailySoundingNote
from agent.memory import DecisionLog
from agent.models import GammaMarket, PortfolioState, WeatherForecast
from agent.risk import RiskManager
from agent.strategy import StrategyEngine
from common.llm.router import LLMRouter

logger = logging.getLogger("hermes.agent")


@dataclass
class RunStats:
    markets_scanned: int = 0
    markets_skipped_dedup: int = 0
    markets_evaluated: int = 0
    llm_calls: int = 0
    skipped_ids: list[str] = field(default_factory=list)


class HermesAgent:
    def __init__(
        self,
        *,
        router: LLMRouter,
        risk: RiskManager | None = None,
        memory: DecisionLog | None = None,
        portfolio: PortfolioState | None = None,
    ):
        self.strategy = StrategyEngine(router)
        self.risk = risk or RiskManager()
        self.memory = memory or DecisionLog()
        self.portfolio = portfolio or PortfolioState()

    async def run_once(
        self,
        weather: WeatherForecast,
        markets: list[GammaMarket],
        *,
        skip_if_evaluated_today: bool = False,
        dedup_window_minutes: int | None = None,
        dry_run: bool = True,
        sounding_note: DailySoundingNote | None = None,
    ) -> tuple[list[AggregatedSignal], RunStats]:
        """Evaluate markets with optional dedup.

        Dedup modes (mutually exclusive, dedup_window_minutes takes priority):
        - dedup_window_minutes=N: skip if evaluated within last N minutes
          (for high-frequency polling like Singapore 30-min cycles)
        - skip_if_evaluated_today=True: skip if evaluated any time today
          (for low-frequency polling like all-city 8h cycles)
        """
        results: list[AggregatedSignal] = []
        stats = RunStats(markets_scanned=len(markets))

        for market in markets:
            if dedup_window_minutes is not None:
                if self.memory.was_evaluated_in_window(market.id, dedup_window_minutes):
                    stats.markets_skipped_dedup += 1
                    stats.skipped_ids.append(market.id)
                    logger.info(
                        f"Dedup: skip market {market.id} "
                        f"(evaluated within {dedup_window_minutes}m window)"
                    )
                    continue
            elif skip_if_evaluated_today and self.memory.was_evaluated_today(market.id):
                stats.markets_skipped_dedup += 1
                stats.skipped_ids.append(market.id)
                logger.info(f"Dedup: skip market {market.id} (already evaluated today)")
                continue

            signal, raw_outputs = await self.strategy.evaluate(
                weather, market, sounding_note=sounding_note,
            )
            stats.markets_evaluated += 1
            stats.llm_calls += self.strategy._n_repeats if signal.n_samples > 0 else 0

            filtered = self.risk.filter(signal, self.portfolio)

            if filtered is None:
                risk_decision = "blocked"
                final_signal = AggregatedSignal(
                    market_id=market.id,
                    action="hold",
                    confidence=signal.confidence,
                    suggested_size_usd=0,
                    rationale=f"Risk blocked: {signal.rationale}",
                    weather_factors=signal.weather_factors,
                    quality=signal.quality,
                    agreement_ratio=signal.agreement_ratio,
                    raw_samples=signal.raw_samples,
                    n_samples=signal.n_samples,
                )
            elif filtered.suggested_size_usd < signal.suggested_size_usd:
                risk_decision = f"capped: ${signal.suggested_size_usd:.2f} → ${filtered.suggested_size_usd:.2f}"
                final_signal = filtered
            else:
                risk_decision = "approved"
                final_signal = filtered

            self.memory.log_decision(
                weather_snapshot=weather.model_dump(),
                market_snapshot=market.model_dump(),
                llm_raw_outputs=raw_outputs,
                final_signal=final_signal,
                risk_decision=risk_decision,
                dry_run=dry_run,
            )

            if final_signal.action != "hold":
                logger.info(
                    f"[{'DRY-RUN' if dry_run else 'LIVE'}] Would execute: {final_signal.action} "
                    f"${final_signal.suggested_size_usd:.2f} on market {market.id} "
                    f"(confidence {final_signal.confidence:.2f}, "
                    f"agreement {final_signal.agreement_ratio:.0%})"
                )
            else:
                logger.info(f"Hold on market {market.id}: {final_signal.rationale[:80]}")

            results.append(final_signal)

        logger.info(
            f"Run complete: scanned={stats.markets_scanned} "
            f"skipped={stats.markets_skipped_dedup} "
            f"evaluated={stats.markets_evaluated} "
            f"llm_calls={stats.llm_calls}"
        )

        return results, stats

    async def run_loop(self, interval_seconds: float = 300) -> None:
        """Placeholder for scheduled execution. Not wired to real data fetchers yet."""
        logger.info(f"Starting Hermes loop (interval={interval_seconds}s)")
        while True:
            logger.info("Loop tick — waiting for real data pipeline to be wired")
            await asyncio.sleep(interval_seconds)
