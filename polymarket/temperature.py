from __future__ import annotations

import re

from pydantic import BaseModel, Field

from polymarket.models import Event, Market

# Real Polymarket sub-market question formats (discovered from live API):
#
# °F cities (US) — 2°F-wide "between" buckets:
#   "Will the highest temperature in New York City be 71°F or below on June 25?"
#   "Will the highest temperature in New York City be between 72-73°F on June 25?"
#   "Will the highest temperature in New York City be 90°F or higher on June 25?"
#   "Will the lowest temperature in Miami be 69°F or below on June 25?"
#
# °C cities (non-US) — 1°C-wide "exact" single-value buckets:
#   "Will the highest temperature in London be 32°C or below on June 26?"
#   "Will the highest temperature in London be 33°C on June 26?"
#   "Will the highest temperature in London be 42°C or higher on June 26?"
#   "Will the lowest temperature in Tokyo be 21°C on June 26?"
#
# Legacy (older markets, still in closed data):
#   "Will the high in Washington DC be 19°F or below on January 20?"
#   "Will the high temperature in New York on July 4, 2026 exceed 95°F?"

_TEMP_PATTERNS = [
    # Pattern 1: "or below" / "or above" / "or higher" with single threshold
    re.compile(
        r"(?:will\s+the\s+)?(?:high(?:est)?\s+(?:temperature\s+in\s+)?|low(?:est)?\s+(?:temperature\s+in\s+)?)"
        r"(?:in\s+)?(?P<city>.+?)\s+be\s+"
        r"(?P<threshold>\d+(?:\.\d+)?)\s*°\s*(?P<unit>[FCfc])\s+"
        r"or\s+(?P<dir>below|above|higher)"
        r"\s+(?:on\s+)?(?P<date>[A-Za-z]+\.?\s+\d{1,2}(?:,?\s*\d{4})?)",
        re.IGNORECASE,
    ),
    # Pattern 2: "between X-Y°F" range
    re.compile(
        r"(?:will\s+the\s+)?(?:high(?:est)?\s+(?:temperature\s+in\s+)?|low(?:est)?\s+(?:temperature\s+in\s+)?)"
        r"(?:in\s+)?(?P<city>.+?)\s+be\s+"
        r"between\s+(?P<threshold>\d+(?:\.\d+)?)\s*[-–]\s*(?P<threshold_high>\d+(?:\.\d+)?)\s*°\s*(?P<unit>[FCfc])"
        r"\s+(?:on\s+)?(?P<date>[A-Za-z]+\.?\s+\d{1,2}(?:,?\s*\d{4})?)",
        re.IGNORECASE,
    ),
    # Pattern 3: exact single-value "be X°C on {date}" (no direction word)
    re.compile(
        r"(?:will\s+the\s+)?(?:high(?:est)?\s+(?:temperature\s+in\s+)?|low(?:est)?\s+(?:temperature\s+in\s+)?)"
        r"(?:in\s+)?(?P<city>.+?)\s+be\s+"
        r"(?P<threshold>\d+(?:\.\d+)?)\s*°\s*(?P<unit>[FCfc])\s+"
        r"(?:on\s+)?(?P<date>[A-Za-z]+\.?\s+\d{1,2}(?:,?\s*\d{4})?)",
        re.IGNORECASE,
    ),
    # Pattern 4 (legacy): "Will the high temperature in {city} on {date} exceed {value}°F?"
    re.compile(
        r"(?:will|does)\s+the\s+(?:high|low|max|min|average)?\s*temperature\s+in\s+"
        r"(?P<city>[A-Za-z\s\.]+?)\s+(?:on|for)\s+(?P<date>[A-Za-z]+\s+\d{1,2}(?:,?\s*\d{4})?)"
        r"\s+(?:exceed|be\s+(?:above|over|at\s+least|below|under))\s+"
        r"(?P<threshold>\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FCfc])",
        re.IGNORECASE,
    ),
    # Pattern 5 (legacy): "Will it be above {value}°F in {city} on {date}?"
    re.compile(
        r"(?:will\s+it\s+be|temperature)\s+(?:above|over|below|under|at\s+least)\s+"
        r"(?P<threshold>\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FCfc])\s+in\s+"
        r"(?P<city>[A-Za-z\s\.]+?)\s+(?:on|for)\s+(?P<date>[A-Za-z]+\s+\d{1,2}(?:,?\s*\d{4})?)",
        re.IGNORECASE,
    ),
]

_ABOVE_PATTERN = re.compile(r"exceed|above|over|at\s+least|higher\s+than|more\s+than|higher", re.IGNORECASE)
_BELOW_PATTERN = re.compile(r"below|under|less\s+than|lower\s+than|at\s+most", re.IGNORECASE)

_EVENT_TITLE_PATTERN = re.compile(
    r"(?:highest|lowest)\s+temperature\s+in\s+(?P<city>.+?)\s+on\s+(?P<date>.+?)\??$",
    re.IGNORECASE,
)

# Bucket widths by unit — derived from live API structure:
#   °C: 1-degree exact buckets  (e.g., 33°C, 34°C, 35°C)
#   °F: 2-degree range buckets  (e.g., 72-73°F, 74-75°F)
_DEFAULT_BUCKET_WIDTH = {"C": 1.0, "F": 2.0}


class TemperatureMarket(BaseModel):
    market: Market
    city: str
    date: str
    threshold_temp: float
    threshold_temp_high: float | None = Field(default=None, description="Upper bound for 'between' ranges")
    threshold_unit: str = Field(description="F or C")
    direction: str = Field(default="above", description="above, below, between, or exact")
    bucket_width: float = Field(default=1.0, description="Width of this bucket in native unit degrees")
    outcome_yes_price: float = Field(default=0.0, ge=0, le=1)
    outcome_no_price: float = Field(default=0.0, ge=0, le=1)
    quality: str = Field(default="high", description="high, medium, or low")
    skip_trading: bool = Field(default=False, description="True for low-quality events")

    @property
    def threshold_c(self) -> float:
        if self.threshold_unit.upper() == "C":
            return self.threshold_temp
        return (self.threshold_temp - 32) * 5 / 9

    @property
    def threshold_f(self) -> float:
        if self.threshold_unit.upper() == "F":
            return self.threshold_temp
        return self.threshold_temp * 9 / 5 + 32

    @property
    def bucket_width_c(self) -> float:
        if self.threshold_unit.upper() == "C":
            return self.bucket_width
        return self.bucket_width * 5 / 9


def parse_temperature_market(market: Market) -> TemperatureMarket | None:
    text = market.question
    for pattern in _TEMP_PATTERNS:
        m = pattern.search(text)
        if m:
            city = m.group("city").strip().rstrip(",.")
            date = m.group("date").strip().rstrip(",")
            threshold = float(m.group("threshold"))
            unit = m.group("unit").upper()

            threshold_high: float | None = None
            try:
                threshold_high = float(m.group("threshold_high"))
            except (IndexError, AttributeError):
                pass

            if threshold_high is not None:
                direction = "between"
                bucket_width = threshold_high - threshold + 1
            elif "dir" in m.groupdict() and m.group("dir"):
                raw_dir = m.group("dir").lower()
                direction = "above" if raw_dir in ("above", "higher") else "below"
                bucket_width = float("inf")
            elif _BELOW_PATTERN.search(text):
                direction = "below"
                bucket_width = float("inf")
            elif _ABOVE_PATTERN.search(text):
                direction = "above"
                bucket_width = float("inf")
            else:
                direction = "exact"
                bucket_width = _DEFAULT_BUCKET_WIDTH.get(unit, 1.0)

            prices = _parse_prices(market)

            return TemperatureMarket(
                market=market,
                city=city,
                date=date,
                threshold_temp=threshold,
                threshold_temp_high=threshold_high,
                threshold_unit=unit,
                direction=direction,
                bucket_width=bucket_width,
                outcome_yes_price=prices[0],
                outcome_no_price=prices[1],
            )

    return None


def _parse_prices(market: Market) -> tuple[float, float]:
    import json
    try:
        prices = json.loads(market.outcome_prices)
        if isinstance(prices, list) and len(prices) >= 2:
            return float(prices[0]), float(prices[1])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0.0, 0.0


def is_temperature_event(event: Event) -> bool:
    title = event.title.lower()
    return "temperature" in title and ("highest" in title or "lowest" in title)


def find_temperature_markets(markets: list[Market]) -> list[TemperatureMarket]:
    results: list[TemperatureMarket] = []
    for market in markets:
        tm = parse_temperature_market(market)
        if tm is not None:
            results.append(tm)
    return results


def find_temperature_markets_from_events(
    events: list[Event],
    *,
    apply_quality: bool = True,
) -> list[TemperatureMarket]:
    if not apply_quality:
        results: list[TemperatureMarket] = []
        for event in events:
            if not is_temperature_event(event):
                continue
            for market in event.markets:
                tm = parse_temperature_market(market)
                if tm is not None:
                    results.append(tm)
        return results

    from polymarket.quality import annotate_quality

    markets_by_event: dict[str, list[TemperatureMarket]] = {}
    for event in events:
        if not is_temperature_event(event):
            continue
        event_key = event.id or event.title
        parsed: list[TemperatureMarket] = []
        for market in event.markets:
            tm = parse_temperature_market(market)
            if tm is not None:
                parsed.append(tm)
        if parsed:
            markets_by_event[event_key] = parsed

    return annotate_quality(markets_by_event)
