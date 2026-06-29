"""Scan all temperature events and report quality tier distribution."""
import asyncio, sys, io
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import (
        is_temperature_event,
        parse_temperature_market,
    )
    from polymarket.quality import classify_event_quality, compute_yes_sum

    client = GammaClient()
    events = await client.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False, max_pages=10)
    temp_events = [e for e in events if is_temperature_event(e)]

    print(f"Temperature events: {len(temp_events)}")
    print(f"{'=' * 80}")

    tier_counts = Counter()
    tier_markets = Counter()
    tier_cities = defaultdict(set)
    tier_events_detail = defaultdict(list)

    for ev in temp_events:
        parsed = []
        for mk in ev.markets:
            tm = parse_temperature_market(mk)
            if tm:
                parsed.append(tm)

        if not parsed:
            continue

        yes_sum = compute_yes_sum(parsed)
        tier = classify_event_quality(yes_sum)
        deviation = abs(yes_sum - 1.0)
        city = parsed[0].city

        tier_counts[tier] += 1
        tier_markets[tier] += len(parsed)
        tier_cities[tier].add(city)
        tier_events_detail[tier].append((ev.title, yes_sum, deviation, city, len(parsed)))

    # Summary table
    print(f"\n{'Tier':<10} {'Events':<10} {'Markets':<10} {'Cities':<10}")
    print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for tier in ["high", "medium", "low"]:
        print(f"{tier:<10} {tier_counts[tier]:<10} {tier_markets[tier]:<10} {len(tier_cities[tier]):<10}")

    total_events = sum(tier_counts.values())
    total_markets = sum(tier_markets.values())
    print(f"{'─'*10} {'─'*10} {'─'*10}")
    print(f"{'TOTAL':<10} {total_events:<10} {total_markets:<10}")

    tradeable = tier_counts["high"] + tier_counts["medium"]
    print(f"\nTradeable events (high + medium): {tradeable}/{total_events} ({tradeable/total_events*100:.0f}%)")
    tradeable_mkts = tier_markets["high"] + tier_markets["medium"]
    print(f"Tradeable markets: {tradeable_mkts}/{total_markets} ({tradeable_mkts/total_markets*100:.0f}%)")

    # Detail per tier
    for tier in ["high", "medium", "low"]:
        items = tier_events_detail[tier]
        if not items:
            continue
        print(f"\n{'=' * 80}")
        print(f"  {tier.upper()} tier — {len(items)} events, cities: {sorted(tier_cities[tier])}")
        print(f"{'=' * 80}")
        items.sort(key=lambda x: x[2])
        for title, yes_sum, deviation, city, n_mkts in items:
            print(f"  {deviation*100:5.2f}%  sum={yes_sum:.4f}  {title} ({n_mkts} markets)")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
