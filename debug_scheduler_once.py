"""Focused scheduler test: 2 cities, verify dedup on second run."""
import asyncio
import sys
import os
import io
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug.scheduler")

from agent.hermes import HermesAgent, RunStats
from agent.memory import DecisionLog
from agent.models import GammaMarket, PortfolioState, WeatherForecast, DayForecast
from agent.risk import RiskManager
from common.llm.router import LLMRouter
from polymarket.client import GammaClient
from polymarket.temperature import find_temperature_markets_from_events
from weather_mcp.tools import WeatherClient
from scheduler.cron import _build_weather_forecast, _temp_market_to_gamma


async def main():
    db_path = "debug_scheduler_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory = DecisionLog(db_path)
    router = LLMRouter()
    risk = RiskManager()
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=0)
    agent = HermesAgent(router=router, risk=risk, memory=memory, portfolio=portfolio)

    gamma = GammaClient()
    weather_client = WeatherClient()

    logger.info("Fetching temperature events...")
    events = await gamma.get_events_by_tag()
    temp_markets = find_temperature_markets_from_events(events)
    logger.info(f"Found {len(temp_markets)} temperature markets total")

    # Pick just 2 cities with smallest market count for speed
    cities_count = {}
    for tm in temp_markets:
        if not tm.skip_trading:
            cities_count[tm.city] = cities_count.get(tm.city, 0) + 1

    sorted_cities = sorted(cities_count.items(), key=lambda x: x[1])
    test_cities = [c for c, _ in sorted_cities[:2]]
    logger.info(f"Testing with cities: {test_cities} (market counts: {[cities_count[c] for c in test_cities]})")

    # Fetch weather for test cities
    weather_by_city = {}
    for city in test_cities:
        raw = await weather_client.get_weather(city)
        weather_by_city[city] = _build_weather_forecast(raw)

    # === RUN 1: First evaluation ===
    logger.info("=" * 60)
    logger.info("RUN 1: First evaluation (nothing deduped yet)")
    logger.info("=" * 60)

    combined1 = RunStats()
    for city in test_cities:
        city_markets = [
            _temp_market_to_gamma(tm) for tm in temp_markets
            if tm.city == city and not tm.skip_trading
        ]
        if not city_markets:
            continue

        logger.info(f"Evaluating {len(city_markets)} markets for {city}")
        _, stats = await agent.run_once(
            weather_by_city[city], city_markets,
            skip_if_evaluated_today=True, dry_run=True,
        )
        combined1.markets_scanned += stats.markets_scanned
        combined1.markets_skipped_dedup += stats.markets_skipped_dedup
        combined1.markets_evaluated += stats.markets_evaluated
        combined1.llm_calls += stats.llm_calls

    print("\n" + "=" * 60)
    print("RUN 1 STATS:")
    print(f"  Markets scanned:  {combined1.markets_scanned}")
    print(f"  Skipped (dedup):  {combined1.markets_skipped_dedup}")
    print(f"  Evaluated:        {combined1.markets_evaluated}")
    print(f"  LLM calls:        {combined1.llm_calls}")
    print("=" * 60)

    # === RUN 2: Dedup should skip everything ===
    logger.info("=" * 60)
    logger.info("RUN 2: Re-run same cities (dedup should skip all)")
    logger.info("=" * 60)

    combined2 = RunStats()
    for city in test_cities:
        city_markets = [
            _temp_market_to_gamma(tm) for tm in temp_markets
            if tm.city == city and not tm.skip_trading
        ]
        if not city_markets:
            continue

        _, stats = await agent.run_once(
            weather_by_city[city], city_markets,
            skip_if_evaluated_today=True, dry_run=True,
        )
        combined2.markets_scanned += stats.markets_scanned
        combined2.markets_skipped_dedup += stats.markets_skipped_dedup
        combined2.markets_evaluated += stats.markets_evaluated
        combined2.llm_calls += stats.llm_calls
        combined2.skipped_ids.extend(stats.skipped_ids)

    print("\n" + "=" * 60)
    print("RUN 2 STATS (should be all skipped):")
    print(f"  Markets scanned:  {combined2.markets_scanned}")
    print(f"  Skipped (dedup):  {combined2.markets_skipped_dedup}")
    print(f"  Evaluated:        {combined2.markets_evaluated}")
    print(f"  LLM calls:        {combined2.llm_calls}")
    print("=" * 60)

    # Verify DB records
    decisions = memory.get_recent_decisions(100)
    print(f"\nDB records: {len(decisions)} total")
    print(f"All dry_run=1: {all(d['dry_run'] == 1 for d in decisions)}")
    print(f"All have market_id: {all(d['market_id'] != '' for d in decisions)}")
    print(f"All have eval_date: {all(d['eval_date'] != '' for d in decisions)}")

    # Verify assertions
    assert combined2.markets_evaluated == 0, f"Expected 0 evaluations in run 2, got {combined2.markets_evaluated}"
    assert combined2.llm_calls == 0, f"Expected 0 LLM calls in run 2, got {combined2.llm_calls}"
    assert combined2.markets_skipped_dedup == combined1.markets_evaluated, \
        f"Expected {combined1.markets_evaluated} skips, got {combined2.markets_skipped_dedup}"

    print("\n✓ ALL ASSERTIONS PASSED")
    memory.close()

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
