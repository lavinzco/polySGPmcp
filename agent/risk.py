from __future__ import annotations

import logging
import os

from agent.aggregation import AggregatedSignal
from agent.models import PortfolioState

logger = logging.getLogger("hermes.risk")


class RiskManager:
    def __init__(
        self,
        *,
        max_position_usd: float | None = None,
        max_daily_loss_usd: float | None = None,
        min_confidence: float | None = None,
        medium_tier_size_multiplier: float | None = None,
        weak_agreement_threshold: float | None = None,
        weak_agreement_multiplier: float | None = None,
    ):
        self.max_position_usd = max_position_usd or float(
            os.environ.get("AGENT_MAX_POSITION_USDC", "50")
        )
        self.max_daily_loss_usd = max_daily_loss_usd or float(
            os.environ.get("AGENT_MAX_DAILY_LOSS_USDC", "100")
        )
        self.min_confidence = min_confidence or float(
            os.environ.get("AGENT_MIN_CONFIDENCE", "0.6")
        )
        self.medium_tier_size_multiplier = medium_tier_size_multiplier or float(
            os.environ.get("AGENT_MEDIUM_TIER_SIZE_MULTIPLIER", "0.5")
        )
        self.weak_agreement_threshold = weak_agreement_threshold or float(
            os.environ.get("AGENT_WEAK_AGREEMENT_THRESHOLD", "0.8")
        )
        self.weak_agreement_multiplier = weak_agreement_multiplier or float(
            os.environ.get("AGENT_WEAK_AGREEMENT_MULTIPLIER", "0.5")
        )

    def filter(
        self, signal: AggregatedSignal, portfolio: PortfolioState
    ) -> AggregatedSignal | None:
        if signal.action == "hold":
            logger.info(f"Risk: pass-through hold for market {signal.market_id}")
            return signal

        if signal.confidence < self.min_confidence:
            logger.info(
                f"Risk: blocked market {signal.market_id} — "
                f"confidence {signal.confidence:.2f} < {self.min_confidence}"
            )
            return None

        if portfolio.daily_pnl_usd <= -self.max_daily_loss_usd:
            logger.warning(
                f"Risk: blocked market {signal.market_id} — "
                f"daily loss ${abs(portfolio.daily_pnl_usd):.2f} "
                f">= limit ${self.max_daily_loss_usd:.2f}"
            )
            return None

        if signal.quality == "medium":
            original = signal.suggested_size_usd
            adjusted = original * self.medium_tier_size_multiplier
            signal = signal.model_copy(update={"suggested_size_usd": adjusted})
            logger.info(
                f"Risk: medium-tier discount ${original:.2f} × "
                f"{self.medium_tier_size_multiplier} = ${adjusted:.2f} "
                f"for market {signal.market_id}"
            )

        if signal.agreement_ratio < self.weak_agreement_threshold:
            original = signal.suggested_size_usd
            adjusted = original * self.weak_agreement_multiplier
            signal = signal.model_copy(update={"suggested_size_usd": adjusted})
            logger.info(
                f"Risk: weak-agreement discount ${original:.2f} × "
                f"{self.weak_agreement_multiplier} = ${adjusted:.2f} "
                f"(agreement={signal.agreement_ratio:.0%}) "
                f"for market {signal.market_id}"
            )

        if signal.suggested_size_usd > self.max_position_usd:
            original = signal.suggested_size_usd
            signal = signal.model_copy(
                update={"suggested_size_usd": self.max_position_usd}
            )
            logger.info(
                f"Risk: capped size ${original:.2f} → ${self.max_position_usd:.2f} "
                f"for market {signal.market_id}"
            )

        return signal
