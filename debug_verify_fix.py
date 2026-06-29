"""Verify the temperature market fix works end-to-end against live API."""
import asyncio, sys, io
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.models import Event
    from polymarket.temperature import (
        find_temperature_markets_from_events,
        is_temperature_event,
    )

    print("=" * 72)
    print("  VERIFY: Temperature market discovery via tag_id=84 pipeline")
    print("=" * 72)

    client = GammaClient()

    # Fetch weather events (both active and closed to get a good sample)
    print("\n  Fetching events with tag_id=84...")
    events = await client.get_events_by_tag(
        WEATHER_TAG_ID, active=True, closed=False, max_pages=5
    )
    print(f"  Active weather events: {len(events)}")

    temp_events = [e for e in events if is_temperature_event(e)]
    print(f"  Temperature events: {len(temp_events)}")

    # Parse temperature markets
    temp_markets = find_temperature_markets_from_events(events)
    print(f"  Temperature markets parsed: {len(temp_markets)}")

    # Show cities and date coverage
    cities = set()
    dates = set()
    for tm in temp_markets:
        cities.add(tm.city)
        dates.add(tm.date)

    print(f"\n  Cities: {sorted(cities)}")
    print(f"  Unique dates: {len(dates)}")
    if dates:
        sample_dates = sorted(dates)[:5]
        print(f"  Sample dates: {sample_dates}")

    # Show a few sample markets
    print(f"\n  --- Sample markets (first 5) ---")
    for tm in temp_markets[:5]:
        dir_str = tm.direction
        if tm.threshold_temp_high:
            thresh_str = f"{tm.threshold_temp}-{tm.threshold_temp_high}"
        else:
            thresh_str = str(tm.threshold_temp)
        print(f"  {tm.city} | {tm.date} | {thresh_str} deg{tm.threshold_unit} {dir_str}")
        print(f"    Q: {tm.market.question}")
        print(f"    Yes={tm.outcome_yes_price:.2f} No={tm.outcome_no_price:.2f}")

    # Summary
    print(f"\n{'=' * 72}")
    success = len(temp_markets) > 0
    print(f"  RESULT: {'PASS' if success else 'FAIL'}")
    print(f"  Found {len(temp_markets)} temperature markets across {len(cities)} cities")
    print(f"{'=' * 72}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
