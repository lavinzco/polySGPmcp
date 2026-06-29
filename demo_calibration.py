"""Demo: full calibration pipeline with synthetic data."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent.calibration.analyze import analyze_calibration
from agent.calibration.collector import collect_confidence_samples
from agent.calibration.daily_report import format_report
from agent.calibration.db import CalibrationDB
from agent.calibration.models import CalibrationSample, ProviderConfig
from agent.calibration.settlement_tracker import check_settled_markets
from agent.models import DayForecast, WeatherForecast
from common.logging import logger
from polymarket.models import Market
from polymarket.temperature import TemperatureMarket, find_temperature_markets


def build_synthetic_markets() -> list[Market]:
    return [
        Market(
            id="temp-nyc-jul4",
            question="Will the high temperature in New York on July 4, 2026 exceed 95°F?",
            description="Resolves YES if the official high exceeds 95°F.",
            outcomePrices='["0.30", "0.70"]',
            active=True,
        ),
        Market(
            id="temp-miami-jul1",
            question="Will the high temperature in Miami on July 1, 2026 exceed 92°F?",
            description="Official high temperature reading.",
            outcomePrices='["0.55", "0.45"]',
            active=True,
        ),
        Market(
            id="temp-chicago-jul2",
            question="Will it be above 88°F in Chicago on July 2, 2026?",
            description="NWS official high.",
            outcomePrices='["0.40", "0.60"]',
            active=True,
        ),
        Market(
            id="btc-100k",
            question="Will Bitcoin reach $100k?",
            description="BTC/USD price.",
            active=True,
        ),
        Market(
            id="temp-dallas-jun30",
            question="Will the high temperature in Dallas on June 30, 2026 exceed 100°F?",
            description="Temperature threshold market.",
            outcomePrices='["0.65", "0.35"]',
            active=True,
        ),
    ]


def build_weather_data() -> dict[str, WeatherForecast]:
    return {
        "New York": WeatherForecast(
            location="New York", temp_c=31, temp_f=88, humidity=65,
            wind_speed_kmph=12, wind_dir="SW", weather_desc="Partly cloudy",
            feels_like_c=34, pressure_mb=1015, precip_mm=0, visibility_km=10, uv_index=9,
            forecast_3day=[
                DayForecast(date="2026-07-03", max_temp_c=33, min_temp_c=24,
                            avg_humidity=60, total_precip_mm=0, condition="Sunny"),
                DayForecast(date="2026-07-04", max_temp_c=36, min_temp_c=26,
                            avg_humidity=55, total_precip_mm=0, condition="Hot, clear"),
                DayForecast(date="2026-07-05", max_temp_c=34, min_temp_c=25,
                            avg_humidity=62, total_precip_mm=2, condition="PM thunderstorms"),
            ],
        ),
        "Miami": WeatherForecast(
            location="Miami", temp_c=33, temp_f=91, humidity=78,
            wind_speed_kmph=15, wind_dir="SE", weather_desc="Hot, humid",
            feels_like_c=38, pressure_mb=1012, precip_mm=3, visibility_km=8, uv_index=11,
            forecast_3day=[
                DayForecast(date="2026-06-30", max_temp_c=34, min_temp_c=27,
                            avg_humidity=80, total_precip_mm=5, condition="Scattered storms"),
                DayForecast(date="2026-07-01", max_temp_c=35, min_temp_c=27,
                            avg_humidity=75, total_precip_mm=2, condition="Hot"),
                DayForecast(date="2026-07-02", max_temp_c=33, min_temp_c=26,
                            avg_humidity=82, total_precip_mm=8, condition="Rain"),
            ],
        ),
        "Chicago": WeatherForecast(
            location="Chicago", temp_c=28, temp_f=82, humidity=55,
            wind_speed_kmph=18, wind_dir="NW", weather_desc="Clear",
            feels_like_c=30, pressure_mb=1018, precip_mm=0, visibility_km=15, uv_index=8,
            forecast_3day=[
                DayForecast(date="2026-07-01", max_temp_c=30, min_temp_c=20,
                            avg_humidity=50, total_precip_mm=0, condition="Sunny"),
                DayForecast(date="2026-07-02", max_temp_c=32, min_temp_c=22,
                            avg_humidity=52, total_precip_mm=0, condition="Hot"),
                DayForecast(date="2026-07-03", max_temp_c=29, min_temp_c=21,
                            avg_humidity=60, total_precip_mm=5, condition="Thunderstorms"),
            ],
        ),
        "Dallas": WeatherForecast(
            location="Dallas", temp_c=38, temp_f=100, humidity=35,
            wind_speed_kmph=10, wind_dir="S", weather_desc="Very hot",
            feels_like_c=40, pressure_mb=1010, precip_mm=0, visibility_km=12, uv_index=11,
            forecast_3day=[
                DayForecast(date="2026-06-29", max_temp_c=39, min_temp_c=28,
                            avg_humidity=32, total_precip_mm=0, condition="Extreme heat"),
                DayForecast(date="2026-06-30", max_temp_c=40, min_temp_c=29,
                            avg_humidity=30, total_precip_mm=0, condition="Extreme heat"),
                DayForecast(date="2026-07-01", max_temp_c=38, min_temp_c=27,
                            avg_humidity=35, total_precip_mm=0, condition="Hot"),
            ],
        ),
    }


MOCK_LLM_RESPONSES = {
    "deepseek-chat": {
        "temp-nyc-jul4": '{"action":"buy_yes","confidence":0.72,"suggested_size_usd":15,"rationale":"Forecast shows 36°C (97°F) high on July 4, exceeding the 95°F threshold. Clear skies and heat dome pattern support this.","weather_factors":["forecast high 36°C = 97°F","clear conditions","heat pattern"]}',
        "temp-miami-jul1": '{"action":"buy_yes","confidence":0.68,"suggested_size_usd":12,"rationale":"Forecast high of 35°C (95°F) for July 1 exceeds the 92°F threshold. Typical summer heat pattern.","weather_factors":["forecast 35°C = 95°F","humidity 75%"]}',
        "temp-chicago-jul2": '{"action":"buy_yes","confidence":0.55,"suggested_size_usd":8,"rationale":"Forecast shows 32°C (90°F) which exceeds 88°F, but Chicago weather is volatile. Moderate confidence.","weather_factors":["forecast 32°C = 90°F","but volatile"]}',
        "temp-dallas-jun30": '{"action":"buy_yes","confidence":0.88,"suggested_size_usd":25,"rationale":"Dallas forecast shows 40°C (104°F) on June 30, well above the 100°F threshold. Extreme heat pattern locked in.","weather_factors":["forecast 40°C = 104°F","extreme heat advisory","no rain"]}',
    },
    "claude-sonnet-4-6": {
        "temp-nyc-jul4": '{"action":"buy_yes","confidence":0.65,"suggested_size_usd":10,"rationale":"The 36°C forecast translates to ~97°F, just above 95°F. But NYC forecasts can shift 2-3°F in 3 days. Moderate edge.","weather_factors":["forecast 97°F vs 95°F threshold","3-day uncertainty window"]}',
        "temp-miami-jul1": '{"action":"hold","confidence":0.45,"suggested_size_usd":0,"rationale":"95°F forecast vs 92°F threshold seems like YES, but market already prices at 55%. No clear edge over market consensus.","weather_factors":["market fairly priced at 55%","no informational edge"]}',
        "temp-chicago-jul2": '{"action":"hold","confidence":0.40,"suggested_size_usd":0,"rationale":"32°C = 90°F barely clears 88°F. Chicago lake effect can easily drop temps 3-4°F. Not enough edge.","weather_factors":["marginal 2°F above threshold","lake effect risk"]}',
        "temp-dallas-jun30": '{"action":"buy_yes","confidence":0.92,"suggested_size_usd":30,"rationale":"104°F forecast is 4°F above threshold with extreme heat advisory. Very high confidence. Market at 65% underprices this.","weather_factors":["forecast 104°F >> 100°F","extreme heat advisory","market underpriced at 65%"]}',
    },
}


async def main() -> None:
    logger.info("=" * 72)
    logger.info("  HERMES CALIBRATION DEMO — Synthetic Data")
    logger.info("=" * 72)

    # Step 1: Parse temperature markets
    raw_markets = build_synthetic_markets()
    temp_markets = find_temperature_markets(raw_markets)
    logger.info(f"\nStep 1: Found {len(temp_markets)} temperature markets out of {len(raw_markets)} total")
    for tm in temp_markets:
        logger.info(f"  [{tm.market.id}] {tm.city} on {tm.date} — "
                     f"{tm.direction} {tm.threshold_temp}°{tm.threshold_unit} "
                     f"(YES={tm.outcome_yes_price:.2f})")

    # Step 2: Collect samples with two mock providers
    weather_data = build_weather_data()
    provider_configs = [
        ProviderConfig(name="deepseek-chat", provider_type="openai_compatible",
                       model="deepseek-chat", base_url="https://api.deepseek.com/v1"),
        ProviderConfig(name="claude-sonnet-4-6", provider_type="anthropic",
                       model="claude-sonnet-4-6"),
    ]

    tmp_dir = Path(tempfile.mkdtemp())
    db = CalibrationDB(tmp_dir / "demo_calibration.db")

    call_counter = {"deepseek-chat": 0, "claude-sonnet-4-6": 0}

    def make_mock_provider(cfg_name: str):
        mock = AsyncMock()
        responses = MOCK_LLM_RESPONSES[cfg_name]
        market_ids = list(responses.keys())

        async def fake_complete(prompt, **kwargs):
            idx = call_counter[cfg_name]
            call_counter[cfg_name] += 1
            mid = market_ids[idx % len(market_ids)]
            return responses[mid]

        mock.complete = AsyncMock(side_effect=fake_complete)
        return mock

    def mock_build(cfg):
        return make_mock_provider(cfg.name)

    logger.info(f"\nStep 2: Collecting calibration samples...")
    with patch("agent.calibration.collector._build_provider_from_config", side_effect=mock_build):
        samples = await collect_confidence_samples(
            provider_configs=provider_configs,
            temperature_markets=temp_markets,
            weather_data=weather_data,
            db=db,
            n_repeats=1,
        )

    logger.info(f"  Collected {len(samples)} samples")

    # Step 3: Simulate settlement (2 of 4 markets resolved)
    logger.info(f"\nStep 3: Simulating market settlement...")
    db.settle_market("temp-nyc-jul4", "YES")      # threshold was 95°F, actual high was 97°F
    db.settle_market("temp-dallas-jun30", "YES")   # threshold was 100°F, actual high was 103°F
    logger.info("  Settled temp-nyc-jul4 → YES (actual 97°F > 95°F)")
    logger.info("  Settled temp-dallas-jun30 → YES (actual 103°F > 100°F)")
    logger.info("  temp-miami-jul1 and temp-chicago-jul2 still pending")

    # Step 4: Generate report
    logger.info(f"\nStep 4: Generating calibration report...")
    report = analyze_calibration(db)
    report_text = format_report(report, "2026-06-26")
    print("\n" + report_text)

    # Save report
    reports_dir = tmp_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "calibration_2026-06-26.txt"
    report_file.write_text(report_text, encoding="utf-8")
    logger.info(f"\nReport saved to: {report_file}")

    # Step 5: Show raw sample data
    logger.info(f"\n--- RAW SAMPLE DATA ---")
    for s in db.get_all_samples():
        status = f"→ {s['actual_outcome']}" if s["settled"] else "(pending)"
        logger.info(
            f"  [{s['provider_name']:>20}] {s['market_id']:>20} | "
            f"{s['llm_action']:>8} conf={s['llm_confidence']:.2f} {status}"
        )

    db.close()
    logger.info("\n  Demo complete.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
