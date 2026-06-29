"""Scheduled dry-run data collection for Hermes."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.calibration.db import CalibrationDB
from agent.calibration.settlement_tracker import (
    check_settled_markets_singapore,
    SettlementCheckResult,
)
from agent.hermes import HermesAgent, RunStats
from agent.memory import DecisionLog
from agent.models import GammaMarket, PortfolioState, WeatherForecast, DayForecast
from agent.risk import RiskManager
from common.config import settings
from common.llm.router import LLMRouter
from polymarket.client import GammaClient
from polymarket.temperature import find_temperature_markets_from_events
from weather_mcp.tools import WeatherClient

logger = logging.getLogger("hermes.scheduler")


def _data_path(filename: str) -> str:
    """Resolve a data file path under DATA_DIR."""
    return str(Path(settings.data_dir) / filename)


def _write_healthcheck(success: bool) -> None:
    """Write last-run timestamp to a health check file."""
    path = Path(_data_path("last_run.txt"))
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "ok" if success else "error"
    path.write_text(
        f"{datetime.now(timezone.utc).isoformat()} {status}\n",
        encoding="utf-8",
    )


def _build_weather_forecast(weather_data) -> WeatherForecast:
    forecast_3day = []
    for d in weather_data.forecast:
        forecast_3day.append(DayForecast(
            date=d.date,
            max_temp_c=d.max_temp_c,
            min_temp_c=d.min_temp_c,
            avg_humidity=d.avg_humidity,
            total_precip_mm=d.total_precip_mm,
            condition=d.condition,
        ))
    return WeatherForecast(
        location=weather_data.location,
        temp_c=weather_data.temp_c,
        temp_f=weather_data.temp_f,
        humidity=weather_data.humidity,
        wind_speed_kmph=weather_data.wind_speed_kmph,
        wind_dir=weather_data.wind_dir,
        weather_desc=weather_data.weather_desc,
        feels_like_c=weather_data.feels_like_c,
        pressure_mb=weather_data.pressure_mb,
        precip_mm=weather_data.precip_mm,
        visibility_km=weather_data.visibility_km,
        uv_index=weather_data.uv_index,
        forecast_3day=forecast_3day,
    )


def _temp_market_to_gamma(tm) -> GammaMarket:
    return GammaMarket(
        id=tm.market.id,
        question=tm.market.question,
        description=tm.market.description,
        outcome_yes_price=tm.outcome_yes_price,
        outcome_no_price=tm.outcome_no_price,
        liquidity_usd=float(tm.market.liquidity),
        volume_usd=float(tm.market.volume),
        end_date=tm.market.end_date_iso,
        quality=tm.quality,
    )


async def run_collection_once(
    *,
    memory: DecisionLog | None = None,
    dry_run: bool = True,
    city_filter: set[str] | None = None,
    dedup_window_minutes: int | None = None,
) -> RunStats:
    memory = memory or DecisionLog(db_path=_data_path("hermes_decisions.db"))
    router = LLMRouter()
    risk = RiskManager()
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=0)

    agent = HermesAgent(router=router, risk=risk, memory=memory, portfolio=portfolio)

    gamma = GammaClient()
    weather_client = WeatherClient()

    logger.info("Fetching temperature events from Polymarket...")
    events = await gamma.get_events_by_tag()
    temp_markets = find_temperature_markets_from_events(events)
    logger.info(f"Found {len(temp_markets)} temperature markets")

    if city_filter:
        filter_lower = {c.lower() for c in city_filter}
        temp_markets = [tm for tm in temp_markets if tm.city.lower() in filter_lower]
        logger.info(f"Filtered to {len(temp_markets)} markets for cities: {city_filter}")

    if not temp_markets:
        logger.warning("No temperature markets found, skipping")
        return RunStats()

    cities = {tm.city for tm in temp_markets}
    logger.info(f"Fetching weather for {len(cities)} cities: {', '.join(sorted(cities))}")

    weather_by_city: dict[str, WeatherForecast] = {}
    for city in cities:
        try:
            raw = await weather_client.get_weather(city)
            weather_by_city[city] = _build_weather_forecast(raw)
        except Exception as exc:
            logger.error(f"Failed to fetch weather for {city}: {exc}")

    combined_stats = RunStats()

    for city, weather in weather_by_city.items():
        city_markets = [
            _temp_market_to_gamma(tm)
            for tm in temp_markets
            if tm.city == city and not tm.skip_trading
        ]
        if not city_markets:
            continue

        logger.info(f"Evaluating {len(city_markets)} markets for {city}")
        _, stats = await agent.run_once(
            weather, city_markets,
            dedup_window_minutes=dedup_window_minutes,
            skip_if_evaluated_today=(dedup_window_minutes is None),
            dry_run=dry_run,
        )

        combined_stats.markets_scanned += stats.markets_scanned
        combined_stats.markets_skipped_dedup += stats.markets_skipped_dedup
        combined_stats.markets_evaluated += stats.markets_evaluated
        combined_stats.llm_calls += stats.llm_calls
        combined_stats.skipped_ids.extend(stats.skipped_ids)

    return combined_stats


async def run_settlement_check(
    *,
    cal_db_path: str | None = None,
) -> SettlementCheckResult:
    """Check settled markets and cross-validate outcomes with METAR."""
    path = cal_db_path or _data_path("hermes_calibration.db")
    cal_db = CalibrationDB(path)
    try:
        result = await check_settled_markets_singapore(cal_db)
        logger.info(
            f"Settlement check: {len(result.settled)} newly settled, "
            f"{len(result.metar_checks)} METAR-checked, "
            f"{len(result.discrepancies)} discrepancies"
        )
        return result
    finally:
        cal_db.close()


CITY_MODES: dict[str, set[str]] = {
    "singapore": {"Singapore"},
    "all": set(),
}

DEDUP_WINDOW_BY_MODE: dict[str, int | None] = {
    "singapore": 25,
    "all": None,
}


async def run_daily_collection(
    interval_minutes: float,
    *,
    city_filter: set[str] | None = None,
    dedup_window_minutes: int | None = None,
) -> None:
    interval_seconds = interval_minutes * 60
    shutdown = asyncio.Event()

    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("Received shutdown signal, finishing current cycle...")
        shutdown.set()

    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)
    else:
        signal.signal(signal.SIGINT, lambda *_: _signal_handler())
        signal.signal(signal.SIGTERM, lambda *_: _signal_handler())

    memory = DecisionLog(db_path=_data_path("hermes_decisions.db"))
    cycle = 0

    try:
        while not shutdown.is_set():
            cycle += 1
            logger.info(f"=== Collection cycle {cycle} starting ===")
            success = True

            try:
                stats = await run_collection_once(
                    memory=memory, dry_run=True, city_filter=city_filter,
                    dedup_window_minutes=dedup_window_minutes,
                )
                _print_stats(stats, cycle)
            except Exception:
                logger.exception(f"Cycle {cycle} collection failed")
                success = False

            try:
                await run_settlement_check()
            except Exception:
                logger.exception(f"Cycle {cycle} settlement check failed")

            _write_healthcheck(success)

            if shutdown.is_set():
                break

            logger.info(f"Sleeping {interval_minutes:.0f}m until next cycle...")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        memory.close()
        logger.info("Scheduler shut down cleanly")


def _print_stats(stats: RunStats, cycle: int) -> None:
    logger.info(
        f"=== Cycle {cycle} complete ===\n"
        f"  Markets scanned:  {stats.markets_scanned}\n"
        f"  Skipped (dedup):  {stats.markets_skipped_dedup}\n"
        f"  Evaluated:        {stats.markets_evaluated}\n"
        f"  LLM calls:        {stats.llm_calls}"
    )
    if stats.skipped_ids:
        logger.info(f"  Skipped IDs: {stats.skipped_ids[:10]}{'...' if len(stats.skipped_ids) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Hermes dry-run data collection")
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Hours between cycles (legacy, use --interval-minutes instead)",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=None,
        help="Minutes between cycles (default: 480 = 8h, or from SCHEDULER_INTERVAL_MINUTES env)",
    )
    parser.add_argument(
        "--mode",
        choices=list(CITY_MODES.keys()),
        default=os.environ.get("SCHEDULER_MODE", "all"),
        help="City filter mode: 'singapore' for SG-only, 'all' for everything",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single collection cycle and exit",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="Run only the settlement check (no data collection)",
    )
    args = parser.parse_args()

    if args.interval_minutes is not None:
        interval_minutes = args.interval_minutes
    elif args.interval_hours is not None:
        interval_minutes = args.interval_hours * 60
    elif "SCHEDULER_INTERVAL_MINUTES" in os.environ:
        interval_minutes = float(os.environ["SCHEDULER_INTERVAL_MINUTES"])
    elif "SCHEDULER_INTERVAL_HOURS" in os.environ:
        interval_minutes = float(os.environ["SCHEDULER_INTERVAL_HOURS"]) * 60
    else:
        interval_minutes = 480.0

    city_filter = CITY_MODES.get(args.mode) or None
    dedup_window = DEDUP_WINDOW_BY_MODE.get(args.mode)

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        f"Mode={args.mode}, dedup={'window ' + str(dedup_window) + 'm' if dedup_window else 'daily'}"
    )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if args.settle:
        result = asyncio.run(run_settlement_check())
        logger.info(
            f"Settled: {len(result.settled)}, "
            f"METAR checks: {len(result.metar_checks)}, "
            f"Discrepancies: {len(result.discrepancies)}"
        )
    elif args.once:
        stats = asyncio.run(run_collection_once(
            dry_run=True, city_filter=city_filter,
            dedup_window_minutes=dedup_window,
        ))
        _print_stats(stats, 1)
        _write_healthcheck(True)
    else:
        asyncio.run(run_daily_collection(
            interval_minutes, city_filter=city_filter,
            dedup_window_minutes=dedup_window,
        ))


if __name__ == "__main__":
    main()
