"""Hermes demo: fetch weather data → scan Polymarket for weather markets."""

import asyncio
import sys

from common.config import settings
from common.logging import logger
from polymarket.client import GammaClient
from polymarket.markets import filter_weather_markets
from weather_mcp.tools import WeatherClient


async def main() -> None:
    logger.info("=== Hermes Weather Trading Agent — Demo ===")

    # Step 1: Fetch weather data
    logger.info("Step 1: Fetching weather data...")
    weather = WeatherClient()
    locations = settings.weather_locations
    weather_results = await weather.get_multi(locations)

    for w in weather_results:
        logger.info(
            f"  {w.location}: {w.temp_c}°C ({w.weather_desc}), "
            f"humidity {w.humidity}%, wind {w.wind_speed_kmph} km/h {w.wind_dir}, "
            f"precip {w.precip_mm}mm"
        )

    # Step 2: Scan Polymarket for weather markets
    logger.info("Step 2: Scanning Polymarket for weather-related markets...")
    gamma = GammaClient()
    all_markets = await gamma.get_all_markets(max_pages=5)
    logger.info(f"  Fetched {len(all_markets)} total markets")

    weather_markets = filter_weather_markets(all_markets)
    logger.info(f"  Found {len(weather_markets)} weather-related markets")

    for wm in weather_markets[:10]:
        logger.info(
            f"  [{wm.relevance_score:.2f}] {wm.market.question[:80]}"
            f"  keywords: {', '.join(wm.matched_keywords)}"
        )

    if not weather_markets:
        logger.info("  No weather markets found — this is normal if none are active right now.")

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
