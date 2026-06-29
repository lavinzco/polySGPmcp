"""Diagnose Low Temp variants and unit coverage in live temperature markets."""
import asyncio, sys, io, re, json
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import (
        find_temperature_markets_from_events,
        is_temperature_event,
        parse_temperature_market,
        _TEMP_PATTERNS,
    )
    from polymarket.models import Event, Market

    client = GammaClient()
    events = await client.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False, max_pages=10)
    print(f"Total active weather events fetched: {len(events)}")

    temp_events = [e for e in events if is_temperature_event(e)]
    print(f"Temperature events: {len(temp_events)}")

    all_markets = []
    for ev in temp_events:
        for mk in ev.markets:
            all_markets.append((ev, mk))

    print(f"Total temperature sub-markets: {len(all_markets)}")

    # ================================================================
    # 1. Low/Lowest/Min markets
    # ================================================================
    print(f"\n{'=' * 72}")
    print("  1. LOW TEMP MARKETS (question contains 'low', 'lowest', or 'min')")
    print(f"{'=' * 72}")

    low_pattern = re.compile(r'\b(low|lowest|min)\b', re.IGNORECASE)
    low_markets = [(ev, mk) for ev, mk in all_markets if low_pattern.search(mk.question)]

    print(f"\n  Found {len(low_markets)} low-temp markets")
    cities_seen = set()
    for i, (ev, mk) in enumerate(low_markets):
        if i < 30 or mk.question.split("in ")[-1].split(" be")[0].strip() not in cities_seen:
            print(f"\n  [{i+1}] Event: {ev.title}")
            print(f"      Q: {mk.question}")
            print(f"      Prices: {mk.outcome_prices}")
            city_guess = mk.question.split("in ")[-1].split(" be")[0].strip() if " in " in mk.question else "?"
            cities_seen.add(city_guess)

    # Also check event titles for "lowest"
    print(f"\n  --- Event titles containing 'low' or 'lowest' ---")
    low_events = [e for e in temp_events if re.search(r'\b(low|lowest)\b', e.title, re.IGNORECASE)]
    print(f"  Found {len(low_events)} events with 'low/lowest' in title")
    for ev in low_events[:10]:
        print(f"    Title: {ev.title}")
        print(f"    Markets: {len(ev.markets)}")

    # ================================================================
    # 2. Wording analysis
    # ================================================================
    print(f"\n{'=' * 72}")
    print("  2. WORDING ANALYSIS")
    print(f"{'=' * 72}")

    wording_counts = {}
    for ev, mk in all_markets:
        q = mk.question
        if re.search(r'highest temperature', q, re.IGNORECASE):
            wording_counts.setdefault("highest temperature", 0)
            wording_counts["highest temperature"] += 1
        elif re.search(r'high temperature', q, re.IGNORECASE):
            wording_counts.setdefault("high temperature", 0)
            wording_counts["high temperature"] += 1
        elif re.search(r'\bhigh\b(?!\s*temperature)', q, re.IGNORECASE):
            wording_counts.setdefault("high (no 'temperature')", 0)
            wording_counts["high (no 'temperature')"] += 1
        elif re.search(r'lowest temperature', q, re.IGNORECASE):
            wording_counts.setdefault("lowest temperature", 0)
            wording_counts["lowest temperature"] += 1
        elif re.search(r'low temperature', q, re.IGNORECASE):
            wording_counts.setdefault("low temperature", 0)
            wording_counts["low temperature"] += 1
        elif re.search(r'\blow\b(?!\s*temperature)', q, re.IGNORECASE):
            wording_counts.setdefault("low (no 'temperature')", 0)
            wording_counts["low (no 'temperature')"] += 1
        else:
            wording_counts.setdefault("OTHER", 0)
            wording_counts["OTHER"] += 1

    print("\n  Wording distribution:")
    for wording, count in sorted(wording_counts.items(), key=lambda x: -x[1]):
        print(f"    {wording}: {count}")

    # Direction/range patterns
    dir_counts = {}
    for ev, mk in all_markets:
        q = mk.question
        if "or below" in q.lower():
            dir_counts.setdefault("or below", 0)
            dir_counts["or below"] += 1
        elif "or above" in q.lower():
            dir_counts.setdefault("or above", 0)
            dir_counts["or above"] += 1
        elif "between" in q.lower():
            dir_counts.setdefault("between X-Y", 0)
            dir_counts["between X-Y"] += 1
        else:
            dir_counts.setdefault("OTHER direction", 0)
            dir_counts["OTHER direction"] += 1
            print(f"    OTHER direction: {mk.question}")

    print("\n  Direction/range distribution:")
    for d, count in sorted(dir_counts.items(), key=lambda x: -x[1]):
        print(f"    {d}: {count}")

    # ================================================================
    # 3. Unit analysis (°C vs °F)
    # ================================================================
    print(f"\n{'=' * 72}")
    print("  3. UNIT ANALYSIS (Celsius vs Fahrenheit)")
    print(f"{'=' * 72}")

    celsius_markets = [(ev, mk) for ev, mk in all_markets if '°C' in mk.question]
    fahrenheit_markets = [(ev, mk) for ev, mk in all_markets if '°F' in mk.question]
    neither = [(ev, mk) for ev, mk in all_markets
               if '°C' not in mk.question and '°F' not in mk.question]

    print(f"\n  Celsius markets: {len(celsius_markets)}")
    print(f"  Fahrenheit markets: {len(fahrenheit_markets)}")
    print(f"  Neither: {len(neither)}")

    celsius_cities = set()
    for ev, mk in celsius_markets:
        city_guess = mk.question.split("in ")[-1].split(" be")[0].strip() if " in " in mk.question else "?"
        celsius_cities.add(city_guess)

    fahrenheit_cities = set()
    for ev, mk in fahrenheit_markets:
        city_guess = mk.question.split("in ")[-1].split(" be")[0].strip() if " in " in mk.question else "?"
        fahrenheit_cities.add(city_guess)

    print(f"\n  Celsius cities ({len(celsius_cities)}): {sorted(celsius_cities)}")
    print(f"\n  Fahrenheit cities ({len(fahrenheit_cities)}): {sorted(fahrenheit_cities)}")

    # Show sample celsius questions
    print(f"\n  --- Celsius sample questions ---")
    for i, (ev, mk) in enumerate(celsius_markets[:10]):
        print(f"    [{i+1}] {mk.question}")

    if neither:
        print(f"\n  --- Markets with no degree symbol ---")
        for ev, mk in neither[:5]:
            print(f"    {mk.question}")

    # ================================================================
    # 4. Event/market hierarchy for Low Temp
    # ================================================================
    print(f"\n{'=' * 72}")
    print("  4. EVENT/MARKET HIERARCHY for Low Temp events")
    print(f"{'=' * 72}")

    if low_events:
        for ev in low_events[:3]:
            print(f"\n  Event: {ev.title}")
            print(f"  Event ID: {ev.id}")
            print(f"  Markets: {len(ev.markets)}")
            for mk in ev.markets:
                print(f"    - {mk.question}")
    else:
        print("  No 'lowest' events found in active data. Checking all temp events for 'low' sub-markets...")
        events_with_low = set()
        for ev, mk in low_markets[:5]:
            if ev.id not in events_with_low:
                events_with_low.add(ev.id)
                print(f"\n  Event: {ev.title}")
                print(f"  Event ID: {ev.id}")
                print(f"  Markets ({len(ev.markets)}):")
                for m in ev.markets:
                    marker = " <<<LOW" if low_pattern.search(m.question) else ""
                    print(f"    - {m.question}{marker}")

    # ================================================================
    # 5. Regex match test — how many markets fail to parse?
    # ================================================================
    print(f"\n{'=' * 72}")
    print("  5. REGEX PARSE COVERAGE (current patterns)")
    print(f"{'=' * 72}")

    parsed = 0
    failed = []
    for ev, mk in all_markets:
        tm = parse_temperature_market(mk)
        if tm:
            parsed += 1
        else:
            failed.append((ev, mk))

    print(f"\n  Parsed OK: {parsed}/{len(all_markets)}")
    print(f"  Failed to parse: {len(failed)}")
    if failed:
        print(f"\n  --- Failed samples (up to 20) ---")
        for i, (ev, mk) in enumerate(failed[:20]):
            print(f"    [{i+1}] Event: {ev.title}")
            print(f"        Q: {mk.question}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
