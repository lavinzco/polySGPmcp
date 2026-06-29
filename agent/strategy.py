from __future__ import annotations

import logging
import os

from agent.aggregation import AggregatedSignal, aggregate_signals
from agent.manual_input import DailySoundingNote
from agent.models import GammaMarket, LLMTradeOutput, TradeSignal, WeatherForecast
from agent.prompts import build_strategy_prompt, get_system_prompt_with_schema
from common.llm.router import LLMRouter, TaskType

logger = logging.getLogger("hermes.strategy")


class StrategyEngine:
    def __init__(self, router: LLMRouter, *, n_repeats: int | None = None):
        self._router = router
        self._n_repeats = n_repeats or int(
            os.environ.get("AGENT_STRATEGY_N_REPEATS", "5")
        )

    async def evaluate(
        self,
        weather: WeatherForecast,
        market: GammaMarket,
        *,
        sounding_note: DailySoundingNote | None = None,
    ) -> tuple[AggregatedSignal, list[str]]:
        """Returns (aggregated_signal, list_of_raw_llm_outputs)."""

        if getattr(market, "quality", "high") == "low":
            signal = AggregatedSignal(
                market_id=market.id,
                action="hold",
                confidence=0.0,
                suggested_size_usd=0.0,
                rationale="市场流动性不足 (quality=low)，跳过评估",
                quality="low",
                agreement_ratio=1.0,
                n_samples=0,
            )
            logger.info(f"Strategy: skip low-quality market {market.id}")
            return signal, []

        provider = self._router.get(TaskType.STRATEGY)
        user_prompt = build_strategy_prompt(weather, market, sounding_note=sounding_note)
        system_prompt = get_system_prompt_with_schema()

        samples: list[TradeSignal] = []
        raw_outputs: list[str] = []

        for i in range(self._n_repeats):
            raw_output = ""
            try:
                raw_output = await provider.complete(user_prompt, system=system_prompt)
                llm_result = self._validate_output(raw_output)
                signal = TradeSignal(
                    market_id=market.id,
                    action=llm_result.action,
                    confidence=llm_result.confidence,
                    suggested_size_usd=llm_result.suggested_size_usd,
                    rationale=llm_result.rationale,
                    weather_factors=llm_result.weather_factors,
                    quality=market.quality,
                )
            except Exception as exc:
                reason = f"LLM output validation failed: {exc}"
                logger.warning(f"Strategy sample {i+1}/{self._n_repeats}: {reason}")
                signal = TradeSignal(
                    market_id=market.id,
                    action="hold",
                    confidence=0.0,
                    suggested_size_usd=0.0,
                    rationale=reason,
                    quality=market.quality,
                )

            samples.append(signal)
            raw_outputs.append(raw_output)
            logger.debug(
                f"Strategy sample {i+1}/{self._n_repeats}: "
                f"{signal.action} @ {signal.confidence:.2f}"
            )

        aggregated = aggregate_signals(samples)

        logger.info(
            f"Strategy: {aggregated.action} @ {aggregated.confidence:.2f} "
            f"agreement={aggregated.agreement_ratio:.0%} "
            f"({self._n_repeats} LLM calls) "
            f"for market {market.id} (quality={market.quality})"
        )

        return aggregated, raw_outputs

    def _validate_output(self, raw: str) -> LLMTradeOutput:
        from common.llm.base import LLMProvider

        result = LLMProvider._parse_model(raw, LLMTradeOutput)

        if result.action not in ("buy_yes", "buy_no", "hold"):
            raise ValueError(f"Invalid action '{result.action}'")

        return result
