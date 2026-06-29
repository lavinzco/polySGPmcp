"""Live test: strategy pipeline WITH 3-day forecast data + extreme edge case."""
import asyncio
import json
import sys
import io
import os

from dotenv import load_dotenv
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def wd_to_forecast(wd) -> "WeatherForecast":
    from agent.models import DayForecast, WeatherForecast
    days = []
    for fd in wd.forecast:
        days.append(DayForecast(
            date=fd.date,
            max_temp_c=fd.max_temp_c,
            min_temp_c=fd.min_temp_c,
            avg_humidity=fd.avg_humidity,
            total_precip_mm=fd.total_precip_mm,
            condition=fd.condition,
        ))
    return WeatherForecast(
        location=wd.location,
        temp_c=wd.temp_c,
        temp_f=wd.temp_f,
        humidity=wd.humidity,
        wind_speed_kmph=wd.wind_speed_kmph,
        wind_dir=wd.wind_dir,
        weather_desc=wd.weather_desc,
        feels_like_c=wd.feels_like_c,
        pressure_mb=wd.pressure_mb,
        precip_mm=wd.precip_mm,
        visibility_km=wd.visibility_km,
        uv_index=wd.uv_index,
        forecast_3day=days,
    )


def tm_to_gamma(tm, *, event_title: str = "") -> "GammaMarket":
    from agent.models import GammaMarket
    return GammaMarket(
        id=tm.market.id,
        question=tm.market.question,
        description=event_title,
        outcome_yes_price=tm.outcome_yes_price,
        outcome_no_price=tm.outcome_no_price,
        quality=tm.quality,
    )


def print_signal(label, signal, raw_output=""):
    print(f"\n    --- TradeSignal [{label}] ---")
    print(f"    action:     {signal.action}")
    print(f"    confidence: {signal.confidence:.2f}")
    print(f"    size:       ${signal.suggested_size_usd:.2f}")
    print(f"    quality:    {signal.quality}")
    print(f"    rationale:  {signal.rationale}")
    if signal.weather_factors:
        print(f"    factors:    {signal.weather_factors}")
    if raw_output:
        print(f"\n    --- Raw LLM (first 600 chars) ---")
        print(f"    {raw_output[:600]}")


async def run_real_markets():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import is_temperature_event, parse_temperature_market
    from polymarket.quality import compute_yes_sum, classify_event_quality
    from weather_mcp.tools import WeatherClient
    from agent.strategy import StrategyEngine
    from agent.risk import RiskManager
    from agent.models import PortfolioState
    from common.llm.router import LLMRouter

    print("=" * 80)
    print("PART A: Real markets WITH forecast data")
    print("=" * 80)

    gamma = GammaClient()
    events = await gamma.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False)
    temp_events = [e for e in events if is_temperature_event(e)]

    # Find a high-quality Beijing event and a medium-quality event
    high_event = None
    medium_event = None
    for ev in temp_events:
        parsed = [parse_temperature_market(m) for m in ev.markets]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            continue
        yes_sum = compute_yes_sum(parsed)
        tier = classify_event_quality(yes_sum)
        city = parsed[0].city

        if tier == "high" and high_event is None and "Beijing" in city:
            high_event = (ev, parsed, yes_sum, tier)
        elif tier == "medium" and medium_event is None:
            medium_event = (ev, parsed, yes_sum, tier)
        if high_event and medium_event:
            break

    # Fallback if no Beijing high
    if not high_event:
        for ev in temp_events:
            parsed = [parse_temperature_market(m) for m in ev.markets]
            parsed = [p for p in parsed if p is not None]
            if not parsed:
                continue
            yes_sum = compute_yes_sum(parsed)
            if classify_event_quality(yes_sum) == "high":
                high_event = (ev, parsed, yes_sum, "high")
                break

    # Apply quality annotation
    from polymarket.temperature import find_temperature_markets_from_events
    all_evs = [item[0] for item in [high_event, medium_event] if item]
    annotated = find_temperature_markets_from_events(all_evs, apply_quality=True)

    # Group and pick representative bucket per event
    by_event: dict[str, list] = {}
    for tm in annotated:
        key = f"{tm.city}|{tm.date}"
        by_event.setdefault(key, []).append(tm)

    test_cases = {}
    for label, item in [("high", high_event), ("medium", medium_event)]:
        if not item:
            continue
        ev, _, _, _ = item
        city = ev.title.split(" in ")[-1].split(" on ")[0].strip()
        for key, markets in by_event.items():
            if city in key and markets[0].quality == label:
                # Pick a middle "between" or "exact" bucket
                mid = [m for m in markets if m.direction in ("between", "exact")]
                chosen = mid[len(mid) // 2] if mid else markets[len(markets) // 2]
                test_cases[label] = (chosen, ev.title, city)
                break

    # Fetch weather with forecast
    weather_client = WeatherClient()
    weather_cache = {}
    cities = set(tc[2] for tc in test_cases.values())
    for city in cities:
        wd = await weather_client.get_weather(city)
        wf = wd_to_forecast(wd)
        weather_cache[city] = wf
        print(f"\n  Weather: {city} — {wd.temp_c}°C, {wd.weather_desc}")
        for fd in wd.forecast:
            print(f"    Forecast {fd.date}: max={fd.max_temp_c}°C min={fd.min_temp_c}°C "
                  f"precip={fd.total_precip_mm:.1f}mm {fd.condition}")

    # Run strategy
    router = LLMRouter()
    engine = StrategyEngine(router)
    risk = RiskManager()
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=0)

    for label in ["high", "medium"]:
        if label not in test_cases:
            continue
        tm, title, city = test_cases[label]
        weather = weather_cache[city]
        gamma_mkt = tm_to_gamma(tm, event_title=title)

        print(f"\n{'─' * 70}")
        print(f"  [{label.upper()}] {tm.market.question}")
        print(f"  YES={tm.outcome_yes_price:.4f}  quality={tm.quality}  city={city}")

        signal, raw = await engine.evaluate(weather, gamma_mkt)
        print_signal(label, signal, raw)

        filtered = risk.filter(signal, portfolio)
        if filtered and filtered.action != "hold":
            print(f"\n    --- After RiskManager ---")
            print(f"    action: {filtered.action}  size: ${filtered.suggested_size_usd:.2f}")
        elif filtered:
            print(f"\n    --- RiskManager: pass-through hold ---")
        else:
            print(f"\n    --- RiskManager: BLOCKED ---")

    return router, engine, risk, portfolio


async def run_extreme_case(router, engine, risk, portfolio):
    from agent.models import DayForecast, GammaMarket, WeatherForecast

    print(f"\n\n{'=' * 80}")
    print("PART B: Extreme edge case — forecast strongly disagrees with market pricing")
    print("=" * 80)

    # Scenario: Market asks "Will the highest temperature in TestCity be 40°C on June 27?"
    # Market YES price is only 0.08 (8%) — market thinks it's unlikely.
    # But our forecast says max temp on June 27 will be 41°C — strongly suggests YES.
    # This should give the model a clear edge and trigger a buy_yes with high confidence.

    weather = WeatherForecast(
        location="TestCity",
        temp_c=38.0,
        temp_f=100.4,
        humidity=25,
        wind_speed_kmph=5,
        wind_dir="S",
        weather_desc="Clear sky",
        feels_like_c=42.0,
        pressure_mb=1008,
        precip_mm=0.0,
        visibility_km=10,
        uv_index=10,
        forecast_3day=[
            DayForecast(
                date="2026-06-26",
                max_temp_c=39.0,
                min_temp_c=28.0,
                avg_humidity=30,
                total_precip_mm=0.0,
                condition="Sunny",
            ),
            DayForecast(
                date="2026-06-27",
                max_temp_c=41.0,
                min_temp_c=29.0,
                avg_humidity=25,
                total_precip_mm=0.0,
                condition="Sunny",
            ),
            DayForecast(
                date="2026-06-28",
                max_temp_c=40.0,
                min_temp_c=28.0,
                avg_humidity=28,
                total_precip_mm=0.0,
                condition="Sunny",
            ),
        ],
    )

    market = GammaMarket(
        id="extreme-test-001",
        question="Will the highest temperature in TestCity be 40°C on June 27?",
        description="Resolves YES if max temp is exactly 40°C on June 27, 2026.",
        outcome_yes_price=0.08,
        outcome_no_price=0.92,
        quality="high",
    )

    print(f"\n  Scenario: forecast max=41°C on June 27, market asks '40°C' at YES=0.08")
    print(f"  This is an 'exact' bucket — model should reason about whether 41°C")
    print(f"  falls in this bucket or an adjacent one.")
    print(f"  If the market is 'be 40°C' (exact), then 41°C means NO → buy_no.")
    print(f"  Key test: does the model give non-zero confidence with clear data?")

    signal, raw = await engine.evaluate(weather, market)
    print_signal("extreme-exact", signal, raw)

    filtered = risk.filter(signal, portfolio)
    if filtered and filtered.action != "hold":
        print(f"\n    --- After RiskManager ---")
        print(f"    action: {filtered.action}  size: ${filtered.suggested_size_usd:.2f}")

    # Second extreme: market asks about a range that INCLUDES the forecast temp
    print(f"\n{'─' * 70}")
    print(f"  Scenario 2: forecast max=41°C, market asks 'between 40-41°C' at YES=0.12")

    market2 = GammaMarket(
        id="extreme-test-002",
        question="Will the highest temperature in TestCity be between 40-41°C on June 27?",
        description="Resolves YES if max temp is between 40-41°C on June 27, 2026.",
        outcome_yes_price=0.12,
        outcome_no_price=0.88,
        quality="high",
    )

    signal2, raw2 = await engine.evaluate(weather, market2)
    print_signal("extreme-range", signal2, raw2)

    filtered2 = risk.filter(signal2, portfolio)
    if filtered2 and filtered2.action != "hold":
        print(f"\n    --- After RiskManager ---")
        print(f"    action: {filtered2.action}  size: ${filtered2.suggested_size_usd:.2f}")
    elif filtered2:
        print(f"\n    --- RiskManager: pass-through {filtered2.action} ---")
    else:
        print(f"\n    --- RiskManager: BLOCKED ---")


async def main():
    router, engine, risk, portfolio = await run_real_markets()
    await run_extreme_case(router, engine, risk, portfolio)
    print(f"\n{'=' * 80}")
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
