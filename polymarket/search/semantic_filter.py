from __future__ import annotations

from pydantic import BaseModel, Field

from common.llm import LLMRouter, TaskType
from common.llm.cache import LLMCache
from polymarket.models import Market, WeatherMarket


class ClassificationResult(BaseModel):
    is_weather_related: bool = False
    confidence: float = Field(default=0.0, ge=0, le=1)
    reasoning: str = ""
    keywords: list[str] = []


CLASSIFICATION_PROMPT = """Analyze this prediction market and determine if it is related to weather, climate, or natural weather phenomena.

Market question: {question}
Market description: {description}

Consider: hurricanes, tornadoes, temperature records, rainfall, snowfall, droughts, floods, heat waves, cold waves, tropical storms, cyclones, blizzards, wildfires (weather-driven), El Niño/La Niña, and any other meteorological events.

Do NOT classify as weather-related if the market is about politics, economics, sports, crypto, or other non-weather topics, even if it incidentally mentions weather words in a metaphorical sense."""


class SemanticFilter:
    def __init__(self, router: LLMRouter, cache: LLMCache | None = None):
        self._router = router
        self._cache = cache or LLMCache()

    async def classify(self, market: Market) -> WeatherMarket | None:
        content = f"{market.question} | {market.description}"
        cached = self._cache.get(TaskType.CLASSIFICATION, content)
        if cached is not None:
            result = ClassificationResult.model_validate_json(cached)
        else:
            provider = self._router.get(TaskType.CLASSIFICATION)
            result = await provider.complete_json(
                CLASSIFICATION_PROMPT.format(
                    question=market.question,
                    description=market.description,
                ),
                ClassificationResult,
                system="You are a market classification assistant.",
            )
            self._cache.put(TaskType.CLASSIFICATION, content, result.model_dump_json())

        if not result.is_weather_related or result.confidence < 0.5:
            return None

        return WeatherMarket(
            market=market,
            relevance_score=result.confidence,
            matched_keywords=result.keywords,
        )

    async def filter_markets(
        self, markets: list[Market], *, min_confidence: float = 0.5
    ) -> list[WeatherMarket]:
        results: list[WeatherMarket] = []
        for m in markets:
            wm = await self.classify(m)
            if wm and wm.relevance_score >= min_confidence:
                results.append(wm)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
