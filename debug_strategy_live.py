"""Live integration test: 3 markets (high/medium/low) through full strategy pipeline."""
import asyncio
import json
import sys
import io
import os

from dotenv import load_dotenv
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


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


def weather_data_to_forecast(wd) -> "WeatherForecast":
    from agent.models import WeatherForecast
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
    )


async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import (
        is_temperature_event,
        find_temperature_markets_from_events,
    )
    from polymarket.quality import compute_yes_sum, classify_event_quality
    from weather_mcp.tools import WeatherClient
    from agent.strategy import StrategyEngine
    from agent.risk import RiskManager
    from agent.models import PortfolioState
    from common.llm.router import LLMRouter

    # --- Step 1: Fetch markets and find one high, one medium, one low ---
    print("=" * 80)
    print("STEP 1: Fetching temperature markets from Gamma API...")
    print("=" * 80)

    gamma = GammaClient()
    events = await gamma.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False)
    temp_events = [e for e in events if is_temperature_event(e)]

    # Classify events by quality without applying quality annotation yet
    from polymarket.temperature import parse_temperature_market

    high_event = None
    medium_event = None
    low_event = None

    for ev in temp_events:
        parsed = [parse_temperature_market(m) for m in ev.markets]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            continue
        yes_sum = compute_yes_sum(parsed)
        tier = classify_event_quality(yes_sum)
        city = parsed[0].city

        if tier == "high" and high_event is None and "Beijing" in city:
            high_event = (ev, parsed, yes_sum)
        elif tier == "medium" and medium_event is None:
            medium_event = (ev, parsed, yes_sum)
        elif tier == "low" and low_event is None and yes_sum > 0:
            low_event = (ev, parsed, yes_sum)

        if high_event and medium_event and low_event:
            break

    if not high_event:
        # Fallback: any high event
        for ev in temp_events:
            parsed = [parse_temperature_market(m) for m in ev.markets]
            parsed = [p for p in parsed if p is not None]
            if not parsed:
                continue
            yes_sum = compute_yes_sum(parsed)
            if classify_event_quality(yes_sum) == "high":
                high_event = (ev, parsed, yes_sum)
                break

    candidates = {"high": high_event, "medium": medium_event, "low": low_event}
    for tier_name, item in candidates.items():
        if item is None:
            print(f"  WARNING: No {tier_name} event found, skipping")
            continue
        ev, parsed, yes_sum = item
        print(f"\n  [{tier_name.upper()}] {ev.title}")
        print(f"    YES sum: {yes_sum:.4f}  deviation: {abs(yes_sum-1.0)*100:.2f}%")
        print(f"    City: {parsed[0].city}  Markets: {len(parsed)}")

    # --- Step 2: Apply quality annotation ---
    print("\n" + "=" * 80)
    print("STEP 2: Apply quality annotation (normalize medium, skip low)...")
    print("=" * 80)

    all_events_to_process = [item[0] for item in candidates.values() if item]
    annotated = find_temperature_markets_from_events(all_events_to_process, apply_quality=True)

    # Group annotated by event title
    by_event: dict[str, list] = {}
    for tm in annotated:
        key = f"{tm.city}|{tm.date}"
        by_event.setdefault(key, []).append(tm)

    # Pick one representative market per tier (the "between" or "exact" bucket closest to current temp)
    test_markets = {}
    for tier_name, item in candidates.items():
        if item is None:
            continue
        ev, _, _ = item
        city = ev.title.split(" in ")[-1].split(" on ")[0].strip() if " in " in ev.title else ""
        for key, markets in by_event.items():
            if city in key and markets[0].quality == tier_name:
                # Pick a middle bucket (not floor/ceil)
                mid = [m for m in markets if m.direction in ("between", "exact")]
                chosen = mid[len(mid) // 2] if mid else markets[len(markets) // 2]
                test_markets[tier_name] = (chosen, ev.title)
                break

    for tier_name, (tm, title) in test_markets.items():
        print(f"\n  [{tier_name.upper()}] {tm.market.question}")
        print(f"    YES={tm.outcome_yes_price:.4f}  NO={tm.outcome_no_price:.4f}  "
              f"quality={tm.quality}  skip={tm.skip_trading}")

    # --- Step 3: Get weather data ---
    print("\n" + "=" * 80)
    print("STEP 3: Fetching live weather data...")
    print("=" * 80)

    weather_client = WeatherClient()
    cities_needed = set()
    for tier_name, (tm, _) in test_markets.items():
        cities_needed.add(tm.city)

    weather_cache = {}
    for city in cities_needed:
        try:
            wd = await weather_client.get_weather(city)
            weather_cache[city] = weather_data_to_forecast(wd)
            print(f"  {city}: {wd.temp_c}°C / {wd.temp_f}°F, {wd.weather_desc}")
        except Exception as exc:
            print(f"  {city}: FAILED — {exc}")

    # --- Step 4: Run strategy engine ---
    print("\n" + "=" * 80)
    print("STEP 4: Running strategy engine (LLM calls for high/medium, skip low)...")
    print("=" * 80)

    router = LLMRouter()
    engine = StrategyEngine(router)
    risk = RiskManager()
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=0)

    for tier_name in ["high", "medium", "low"]:
        if tier_name not in test_markets:
            continue
        tm, title = test_markets[tier_name]
        city = tm.city
        if city not in weather_cache:
            print(f"\n  [{tier_name.upper()}] SKIPPED — no weather data for {city}")
            continue

        weather = weather_cache[city]
        gamma_market = tm_to_gamma(tm, event_title=title)

        print(f"\n  [{tier_name.upper()}] {tm.market.question}")
        print(f"    Market: YES={tm.outcome_yes_price:.4f}  quality={tm.quality}")
        print(f"    Weather: {weather.temp_c}°C, {weather.weather_desc}")

        signal, raw_output = await engine.evaluate(weather, gamma_market)

        print(f"\n    --- TradeSignal ---")
        print(f"    action:     {signal.action}")
        print(f"    confidence: {signal.confidence:.2f}")
        print(f"    size:       ${signal.suggested_size_usd:.2f}")
        print(f"    quality:    {signal.quality}")
        print(f"    rationale:  {signal.rationale}")
        if signal.weather_factors:
            print(f"    factors:    {signal.weather_factors}")

        # Run through risk manager
        filtered = risk.filter(signal, portfolio)
        if filtered:
            print(f"\n    --- After RiskManager ---")
            print(f"    action:     {filtered.action}")
            print(f"    size:       ${filtered.suggested_size_usd:.2f}")
        else:
            print(f"\n    --- RiskManager: BLOCKED ---")

        if raw_output:
            print(f"\n    --- Raw LLM Output ---")
            print(f"    {raw_output[:500]}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
