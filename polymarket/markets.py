from __future__ import annotations

import re

from polymarket.models import Market, WeatherMarket

WEATHER_KEYWORDS: dict[str, float] = {
    "hurricane": 1.0,
    "tornado": 1.0,
    "cyclone": 1.0,
    "typhoon": 1.0,
    "tropical storm": 0.9,
    "blizzard": 0.9,
    "flood": 0.8,
    "flooding": 0.8,
    "drought": 0.8,
    "wildfire": 0.7,
    "heat wave": 0.8,
    "heatwave": 0.8,
    "cold wave": 0.8,
    "temperature": 0.7,
    "rainfall": 0.7,
    "rain": 0.5,
    "snow": 0.6,
    "snowfall": 0.7,
    "precipitation": 0.7,
    "wind speed": 0.6,
    "weather": 0.4,
    "climate": 0.4,
    "storm": 0.6,
    "thunderstorm": 0.7,
    "el nino": 0.7,
    "la nina": 0.7,
    "celsius": 0.5,
    "fahrenheit": 0.5,
}

_KEYWORD_PATTERNS = {
    kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in WEATHER_KEYWORDS
}


def score_market(market: Market) -> WeatherMarket | None:
    text = f"{market.question} {market.description}".lower()
    matched: list[str] = []
    max_score = 0.0

    for kw, weight in WEATHER_KEYWORDS.items():
        if _KEYWORD_PATTERNS[kw].search(text):
            matched.append(kw)
            max_score = max(max_score, weight)

    if not matched:
        return None

    bonus = min(0.1 * (len(matched) - 1), 0.2)
    final_score = min(max_score + bonus, 1.0)

    return WeatherMarket(market=market, relevance_score=final_score, matched_keywords=matched)


def filter_weather_markets(
    markets: list[Market], *, min_score: float = 0.3
) -> list[WeatherMarket]:
    results: list[WeatherMarket] = []
    for m in markets:
        wm = score_market(m)
        if wm and wm.relevance_score >= min_score:
            results.append(wm)
    results.sort(key=lambda x: x.relevance_score, reverse=True)
    return results
