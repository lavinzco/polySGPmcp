"""Hermes strategy demo: fake weather + fake market → full decision pipeline (dry-run)."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from agent.hermes import HermesAgent
from agent.memory import DecisionLog
from agent.models import DayForecast, GammaMarket, PortfolioState, WeatherForecast
from agent.risk import RiskManager
from common.llm.router import LLMRouter, TaskType
from common.logging import logger


FAKE_OPUS_RESPONSES = [
    json.dumps({
        "action": "buy_yes",
        "confidence": 0.82,
        "suggested_size_usd": 30.0,
        "rationale": (
            "Current conditions show rapidly dropping pressure (1005 mb) combined with "
            "sustained winds of 45 km/h and heavy rainfall of 12.5mm. The 3-day forecast "
            "shows intensifying conditions with precipitation reaching 45mm. These patterns "
            "are consistent with tropical storm development that could strengthen to "
            "hurricane status before the July 31 market close."
        ),
        "weather_factors": [
            "pressure 1005 mb (dropping)",
            "sustained wind 45 km/h ESE",
            "heavy rain 12.5mm current, 45mm forecast",
            "humidity 80%",
        ],
    }),
    json.dumps({
        "action": "hold",
        "confidence": 0.35,
        "suggested_size_usd": 0,
        "rationale": (
            "While current precipitation of 12.5mm is above average, the 3-day forecast "
            "does not show a clear trajectory toward record-breaking monthly totals. "
            "The market is pricing at 20% which seems roughly fair given current data. "
            "No clear edge to justify a position."
        ),
        "weather_factors": ["precip 12.5mm (above average but not extreme)"],
    }),
    json.dumps({
        "action": "buy_no",
        "confidence": 0.88,
        "suggested_size_usd": 200.0,
        "rationale": (
            "Current temperature of 33°C with a 3-day max forecast of 36°C is well below "
            "the 45°C threshold. Heat index with 80% humidity pushes feels-like to 38°C "
            "but actual temperature records require dry-bulb readings. No realistic path "
            "to 45°C in the forecast window."
        ),
        "weather_factors": [
            "current temp 33°C, max forecast 36°C",
            "45°C threshold unreachable",
            "high humidity suppresses dry-bulb max",
        ],
    }),
]


def build_fake_weather() -> WeatherForecast:
    return WeatherForecast(
        location="Miami, FL",
        temp_c=33.0,
        temp_f=91.4,
        humidity=80,
        wind_speed_kmph=45,
        wind_dir="ESE",
        weather_desc="Heavy rain, thunderstorms",
        feels_like_c=38.0,
        pressure_mb=1005,
        precip_mm=12.5,
        visibility_km=5,
        uv_index=3,
        forecast_3day=[
            DayForecast(
                date="2026-06-27",
                max_temp_c=34.0, min_temp_c=27.0,
                avg_humidity=82, total_precip_mm=18.0,
                condition="Thunderstorms",
            ),
            DayForecast(
                date="2026-06-28",
                max_temp_c=32.0, min_temp_c=26.0,
                avg_humidity=88, total_precip_mm=25.0,
                condition="Heavy rain, strong winds",
            ),
            DayForecast(
                date="2026-06-29",
                max_temp_c=36.0, min_temp_c=28.0,
                avg_humidity=75, total_precip_mm=45.0,
                condition="Tropical storm warning",
            ),
        ],
    )


def build_fake_markets() -> list[GammaMarket]:
    return [
        GammaMarket(
            id="mkt-hurricane-fl",
            question="Will a Category 3+ hurricane hit Florida before August 2026?",
            description="Resolves YES if NHC declares a Cat 3+ hurricane making landfall in FL.",
            outcome_yes_price=0.35,
            outcome_no_price=0.65,
            liquidity_usd=52000,
            volume_usd=128000,
            end_date="2026-08-01",
            matched_keywords=["hurricane"],
        ),
        GammaMarket(
            id="mkt-miami-rain-record",
            question="Will Miami break its July rainfall record in 2026?",
            description="Monthly rainfall must exceed the all-time July record of 330mm.",
            outcome_yes_price=0.20,
            outcome_no_price=0.80,
            liquidity_usd=15000,
            volume_usd=31000,
            end_date="2026-07-31",
            matched_keywords=["rainfall", "rain"],
        ),
        GammaMarket(
            id="mkt-miami-45c",
            question="Will Miami temperature reach 45°C in June-July 2026?",
            description="Resolves YES if any official weather station records ≥45°C.",
            outcome_yes_price=0.05,
            outcome_no_price=0.95,
            liquidity_usd=8000,
            volume_usd=12000,
            end_date="2026-07-31",
            matched_keywords=["temperature"],
        ),
    ]


async def main() -> None:
    logger.info("=" * 70)
    logger.info("  HERMES STRATEGY ENGINE — DRY-RUN DEMO")
    logger.info("=" * 70)

    # Wire up mock LLM
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=FAKE_OPUS_RESPONSES)
    router = LLMRouter()
    router._providers[TaskType.STRATEGY] = mock_provider

    # Set up agent
    db_path = Path(tempfile.mkdtemp()) / "demo_decisions.db"
    memory = DecisionLog(db_path)
    risk = RiskManager(max_position_usd=50, max_daily_loss_usd=100, min_confidence=0.6)
    portfolio = PortfolioState(total_balance_usd=500, daily_pnl_usd=-15)

    agent = HermesAgent(router=router, risk=risk, memory=memory, portfolio=portfolio)

    weather = build_fake_weather()
    markets = build_fake_markets()

    # Print input context
    logger.info("")
    logger.info("--- WEATHER INPUT ---")
    logger.info(f"Location: {weather.location}")
    logger.info(f"Current: {weather.temp_c}°C, {weather.weather_desc}")
    logger.info(f"Wind: {weather.wind_speed_kmph} km/h {weather.wind_dir}")
    logger.info(f"Pressure: {weather.pressure_mb} mb | Humidity: {weather.humidity}%")
    logger.info(f"Precipitation: {weather.precip_mm} mm")
    for day in weather.forecast_3day:
        logger.info(f"  Forecast {day.date}: {day.condition}, {day.total_precip_mm}mm")

    logger.info("")
    logger.info(f"--- MARKETS ({len(markets)}) ---")
    for m in markets:
        logger.info(f"  [{m.id}] {m.question}")
        logger.info(f"    YES={m.outcome_yes_price:.2f} NO={m.outcome_no_price:.2f} "
                     f"liquidity=${m.liquidity_usd:,.0f}")

    # Run the pipeline
    logger.info("")
    logger.info("--- RUNNING STRATEGY PIPELINE ---")
    results = await agent.run_once(weather, markets)

    # Print decisions
    logger.info("")
    logger.info("--- DECISION SUMMARY ---")
    for signal in results:
        logger.info(f"  Market: {signal.market_id}")
        logger.info(f"  Action: {signal.action} | Confidence: {signal.confidence:.2f}")
        logger.info(f"  Size: ${signal.suggested_size_usd:.2f}")
        logger.info(f"  Rationale: {signal.rationale}")
        if signal.weather_factors:
            logger.info(f"  Weather factors: {', '.join(signal.weather_factors)}")
        logger.info("")

    # Print audit log
    logger.info("--- AUDIT LOG (from SQLite) ---")
    decisions = memory.get_recent_decisions(10)
    for i, row in enumerate(reversed(decisions)):
        llm_raw = json.loads(row["llm_raw_output"]) if row["llm_raw_output"] else {}
        logger.info(f"  Decision #{i+1}: risk={row['risk_decision']}")
        logger.info(f"    LLM raw action: {llm_raw.get('action', 'N/A')}")
        logger.info(f"    LLM raw confidence: {llm_raw.get('confidence', 'N/A')}")
        logger.info(f"    LLM raw size: ${llm_raw.get('suggested_size_usd', 0):.2f}")
        logger.info(f"    LLM rationale: {llm_raw.get('rationale', 'N/A')[:100]}")

    memory.close()
    logger.info("")
    logger.info("=" * 70)
    logger.info("  DEMO COMPLETE — all dry-run, no real trades executed")
    logger.info("=" * 70)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
