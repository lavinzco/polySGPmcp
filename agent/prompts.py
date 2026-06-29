from __future__ import annotations

import re
from datetime import datetime, timezone

from agent.manual_input import DailySoundingNote
from agent.models import DayForecast, GammaMarket, LLMTradeOutput, WeatherForecast

STRATEGY_SYSTEM_PROMPT = """\
You are Hermes, a weather-informed trading strategist for prediction markets.

## Iron Rules
1. If you are NOT confident, output action="hold". Do not trade for the sake of trading.
2. You must have a clear, weather-data-driven thesis to recommend buy_yes or buy_no.
3. Never fabricate weather data or market conditions. Only reason from what is provided.
4. When in doubt, HOLD. It is always better to miss a trade than to take a bad one.
5. A confidence below 0.6 means you should hold. Do not force a trade at low confidence.
6. Market quality tiers affect how much you trust market pricing:
   - quality="high": pricing reflects reliable market consensus with deep liquidity. \
You may treat the current YES/NO prices as meaningful signals of crowd belief.
   - quality="medium": prices have been normalized but original liquidity is low. \
The pricing signal is noisier — you need STRONGER weather evidence to justify \
a high-confidence trade. If your weather edge is marginal, hold.
7. If the forecast section says "BEYOND RELIABLE FORECAST RANGE", you must hold. \
Do not speculate on weather beyond the forecast horizon.

## Output Format
Respond with a single JSON object matching this schema:

{schema}

- action: one of "buy_yes", "buy_no", "hold"
- confidence: float 0.0 to 1.0 — your genuine confidence in the recommendation
- suggested_size_usd: dollar amount to risk (0 if hold)
- rationale: 2-3 sentences explaining your reasoning from the weather data
- weather_factors: list of specific weather observations driving your decision

If the weather data does not give you a clear edge on this market, output action="hold" \
with confidence=0.0 and suggested_size_usd=0."""


_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERN = re.compile(
    r"(?:on|for)\s+([A-Za-z]+)\.?\s+(\d{1,2})(?:,?\s*(\d{4}))?",
    re.IGNORECASE,
)


def _parse_market_date(question: str) -> str | None:
    """Extract ISO date (YYYY-MM-DD) from market question text."""
    m = _DATE_PATTERN.search(question)
    if not m:
        return None
    month_str = m.group(1).lower().rstrip(".")
    day = int(m.group(2))
    year = int(m.group(3)) if m.group(3) else datetime.now(timezone.utc).year
    month = _MONTH_MAP.get(month_str)
    if month is None:
        return None
    return f"{year}-{month:02d}-{day:02d}"


def _build_forecast_block(
    weather: WeatherForecast, market_date_iso: str | None
) -> str:
    if not weather.forecast_3day:
        return "  (no forecast available)"

    forecast_dates = {day.date: day for day in weather.forecast_3day}
    lines = []

    matched_day: DayForecast | None = None
    if market_date_iso:
        matched_day = forecast_dates.get(market_date_iso)

    if matched_day:
        lines.append(
            f"  >>> TARGET DATE {matched_day.date}: "
            f"max {matched_day.max_temp_c}°C, min {matched_day.min_temp_c}°C, "
            f"humidity {matched_day.avg_humidity}%, "
            f"precip {matched_day.total_precip_mm:.1f}mm, {matched_day.condition}"
        )
        lines.append("")

    for day in weather.forecast_3day:
        marker = " <<<" if day.date == market_date_iso else ""
        lines.append(
            f"  {day.date}: {day.min_temp_c}–{day.max_temp_c}°C, "
            f"humidity {day.avg_humidity}%, precip {day.total_precip_mm:.1f}mm, "
            f"{day.condition}{marker}"
        )

    if market_date_iso and market_date_iso not in forecast_dates:
        lines.append("")
        lines.append(
            f"  ⚠ BEYOND RELIABLE FORECAST RANGE: market asks about {market_date_iso} "
            f"but forecast only covers {weather.forecast_3day[0].date} to "
            f"{weather.forecast_3day[-1].date}. Do NOT speculate."
        )

    return "\n".join(lines)


def _build_sounding_block(sounding: DailySoundingNote | None) -> str:
    if sounding is None:
        return (
            "\n## Daily Atmospheric Prior\n"
            "No manual sounding note submitted for today. "
            "Base your judgment solely on real-time observations and forecast data.\n"
        )

    inversion_desc = {
        "strong": "STRONG low-level inversion observed — surface heating may be suppressed, "
                  "capping afternoon max temperature below climatological expectation",
        "weak": "Weak low-level inversion — some suppression possible but likely to break "
                "by midday, moderate impact on max temperature",
        "none": "No inversion detected — atmosphere is well-mixed or unstable, "
                "surface heating should translate efficiently to higher max temperatures",
    }[sounding.inversion.value]

    lines = [
        "\n## Daily Atmospheric Prior (human-entered, qualitative)",
        f"- Date: {sounding.target_date}",
        f"- Inversion assessment: {inversion_desc}",
    ]
    if sounding.surface_temp_c is not None:
        lines.append(f"- Early-morning surface temp: {sounding.surface_temp_c}°C")
    if sounding.note:
        lines.append(f"- Observer note: {sounding.note}")
    lines.append(
        "- ⚠ WEIGHT: This is a qualitative human judgment, not a precise measurement. "
        "Treat it as background context — real-time METAR observations carry higher weight."
    )
    return "\n".join(lines) + "\n"


def build_strategy_prompt(
    weather: WeatherForecast,
    market: GammaMarket,
    *,
    sounding_note: DailySoundingNote | None = None,
) -> str:
    market_date_iso = _parse_market_date(market.question)
    forecast_block = _build_forecast_block(weather, market_date_iso)
    sounding_block = _build_sounding_block(sounding_note)

    quality_note = ""
    if market.quality == "medium":
        quality_note = (
            "\n- ⚠ Quality: MEDIUM — prices normalized from low-liquidity source. "
            "Require stronger weather evidence before trading."
        )
    elif market.quality == "high":
        quality_note = "\n- Quality: HIGH — reliable market consensus"

    return f"""\
## Current Weather — {weather.location}
- Temperature: {weather.temp_c}°C / {weather.temp_f}°F (feels like {weather.feels_like_c}°C)
- Humidity: {weather.humidity}%
- Wind: {weather.wind_speed_kmph} km/h {weather.wind_dir}
- Precipitation: {weather.precip_mm} mm
- Pressure: {weather.pressure_mb} mb
- Visibility: {weather.visibility_km} km
- UV Index: {weather.uv_index}
- Conditions: {weather.weather_desc}

## 3-Day Forecast
{forecast_block}
{sounding_block}
## Prediction Market
- Question: {market.question}
- Description: {market.description}
- Current YES price: {market.outcome_yes_price:.2f} (implies {market.outcome_yes_price*100:.0f}% probability)
- Current NO price: {market.outcome_no_price:.2f}
- Liquidity: ${market.liquidity_usd:,.0f}
- Volume: ${market.volume_usd:,.0f}
- Closes: {market.end_date or 'unknown'}
- Weather keywords matched: {', '.join(market.matched_keywords) or 'none'}{quality_note}

Based on the weather data above, should we take a position on this market? \
Analyze whether the current weather conditions and forecast give us an informational edge \
over the current market pricing."""


def get_system_prompt_with_schema() -> str:
    schema = LLMTradeOutput.model_json_schema()
    return STRATEGY_SYSTEM_PROMPT.format(schema=schema)
