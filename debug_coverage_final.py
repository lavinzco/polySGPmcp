"""Final coverage check: parse all 1859 markets with updated regex."""
import asyncio, sys, io
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import (
        find_temperature_markets_from_events,
        is_temperature_event,
        parse_temperature_market,
    )
    from polymarket.models import Event
    from collections import Counter

    client = GammaClient()
    events = await client.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False, max_pages=10)
    temp_events = [e for e in events if is_temperature_event(e)]

    all_markets = []
    for ev in temp_events:
        for mk in ev.markets:
            all_markets.append((ev, mk))

    print(f"Total temperature sub-markets: {len(all_markets)}")

    parsed = 0
    failed = []
    direction_counts = Counter()
    bucket_widths = Counter()

    for ev, mk in all_markets:
        tm = parse_temperature_market(mk)
        if tm:
            parsed += 1
            direction_counts[tm.direction] += 1
            if tm.direction == "exact":
                bucket_widths[f"exact {tm.bucket_width}{tm.threshold_unit}"] += 1
            elif tm.direction == "between":
                bucket_widths[f"between {tm.bucket_width}{tm.threshold_unit}"] += 1
            else:
                bucket_widths[f"{tm.direction} (inf)"] += 1
        else:
            failed.append((ev, mk))

    pct = parsed / len(all_markets) * 100 if all_markets else 0
    print(f"\nParsed OK: {parsed}/{len(all_markets)} ({pct:.1f}%)")
    print(f"Failed: {len(failed)}")

    print(f"\nDirection distribution:")
    for d, count in direction_counts.most_common():
        print(f"  {d}: {count}")

    print(f"\nBucket width distribution:")
    for bw, count in bucket_widths.most_common():
        print(f"  {bw}: {count}")

    if failed:
        print(f"\n--- ALL FAILED ({len(failed)}) ---")
        for i, (ev, mk) in enumerate(failed):
            print(f"  [{i+1}] Event: {ev.title}")
            print(f"      Q: {mk.question}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
