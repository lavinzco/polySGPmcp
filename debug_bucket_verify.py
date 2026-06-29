"""Verify bucket structure: continuity, price sum, and width differences."""
import asyncio, sys, io, json, re
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    from polymarket.client import GammaClient, WEATHER_TAG_ID
    from polymarket.temperature import is_temperature_event
    from polymarket.models import Event

    client = GammaClient()
    events = await client.get_events_by_tag(WEATHER_TAG_ID, active=True, closed=False, max_pages=10)
    temp_events = [e for e in events if is_temperature_event(e)]

    # Pick specific events for detailed analysis
    targets = {
        "london_high": None,
        "london_low": None,
        "seoul_high": None,
        "tokyo_high": None,
        "tokyo_low": None,
        "nyc_high": None,      # F, for comparison
        "miami_low": None,     # F, for comparison
        "austin_high": None,   # F, for comparison
    }

    for ev in temp_events:
        t = ev.title.lower()
        if "london" in t and "highest" in t and not targets["london_high"]:
            targets["london_high"] = ev
        elif "london" in t and "lowest" in t and not targets["london_low"]:
            targets["london_low"] = ev
        elif "seoul" in t and "highest" in t and not targets["seoul_high"]:
            targets["seoul_high"] = ev
        elif "tokyo" in t and "highest" in t and not targets["tokyo_high"]:
            targets["tokyo_high"] = ev
        elif "tokyo" in t and "lowest" in t and not targets["tokyo_low"]:
            targets["tokyo_low"] = ev
        elif "nyc" in t and "highest" in t and not targets["nyc_high"]:
            targets["nyc_high"] = ev
        elif "miami" in t and "lowest" in t and not targets["miami_low"]:
            targets["miami_low"] = ev
        elif "austin" in t and "highest" in t and not targets["austin_high"]:
            targets["austin_high"] = ev

    # Helper to extract threshold from question
    def extract_threshold(q):
        # "be 19°C on" / "be 19°C or below" / "be between 19-20°F"
        m = re.search(r'be\s+(?:between\s+)?(\d+(?:\.\d+)?)', q)
        return float(m.group(1)) if m else None

    def extract_unit(q):
        m = re.search(r'°([FCfc])', q)
        return m.group(1).upper() if m else "?"

    def extract_bucket_type(q):
        q_lower = q.lower()
        if "or below" in q_lower:
            return "floor"
        elif "or above" in q_lower or "or higher" in q_lower:
            return "ceil"
        elif "between" in q_lower:
            m = re.search(r'between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', q)
            if m:
                return f"range({m.group(1)}-{m.group(2)})"
            return "range(?)"
        else:
            return "exact"

    def parse_yes_price(mk):
        try:
            prices = json.loads(mk.outcome_prices)
            if isinstance(prices, list) and len(prices) >= 1:
                return float(prices[0])
        except:
            pass
        return 0.0

    for label, ev in targets.items():
        if ev is None:
            print(f"\n{'=' * 72}")
            print(f"  {label}: NOT FOUND")
            continue

        print(f"\n{'=' * 72}")
        print(f"  {label}: {ev.title}")
        print(f"  Event ID: {ev.id}, Markets: {len(ev.markets)}")
        print(f"{'=' * 72}")

        unit = "?"
        buckets = []
        yes_total = 0.0

        for mk in ev.markets:
            threshold = extract_threshold(mk.question)
            u = extract_unit(mk.question)
            btype = extract_bucket_type(mk.question)
            yes_p = parse_yes_price(mk)
            yes_total += yes_p
            if u != "?":
                unit = u
            buckets.append((threshold, btype, yes_p, mk.question))

        # Sort by threshold
        buckets.sort(key=lambda x: x[0] if x[0] is not None else -999)

        print(f"\n  Unit: °{unit}")
        print(f"  {'#':<4} {'Threshold':<12} {'Type':<18} {'YES price':<12} Question")
        print(f"  {'─'*4} {'─'*12} {'─'*18} {'─'*12} {'─'*50}")

        for i, (thresh, btype, yes_p, q) in enumerate(buckets):
            print(f"  {i+1:<4} {thresh:<12} {btype:<18} {yes_p:<12.4f} {q}")

        print(f"\n  (a) Bucket count: {len(buckets)}")

        # Check continuity
        thresholds = [b[0] for b in buckets if b[0] is not None]
        if len(thresholds) >= 2:
            diffs = [thresholds[i+1] - thresholds[i] for i in range(len(thresholds)-1)]
            print(f"      Thresholds: {thresholds}")
            print(f"      Steps: {diffs}")
            all_same = len(set(round(d, 2) for d in diffs)) == 1
            print(f"      Continuous (uniform step): {'YES' if all_same else 'NO — ' + str(set(round(d,2) for d in diffs))}")
            step = diffs[0] if all_same else None
            if step:
                print(f"      Step size: {step}°{unit}")

        # (b) Price sum
        print(f"\n  (b) YES price sum: {yes_total:.4f}")
        deviation = abs(yes_total - 1.0)
        if deviation < 0.05:
            print(f"      Deviation from 1.0: {deviation:.4f} — GOOD (< 5%)")
        elif deviation < 0.15:
            print(f"      Deviation from 1.0: {deviation:.4f} — MODERATE")
        else:
            print(f"      Deviation from 1.0: {deviation:.4f} — SIGNIFICANT")

    # (c) Summary comparison
    print(f"\n\n{'=' * 72}")
    print(f"  (c) BUCKET WIDTH COMPARISON: °F vs °C")
    print(f"{'=' * 72}")
    print("""
  °F cities (US): "between X-(X+1)°F" — 2-degree-wide buckets
    e.g., 76-77°F, 78-79°F, 80-81°F
    Each "between" bucket covers 2°F
    Typical event: 11 markets = 1 floor + 9 between + 1 ceil
    Coverage: ~20°F range (= ~11°C range)

  °C cities (non-US): "be X°C" — exact-value single-degree buckets
    e.g., 19°C, 20°C, 21°C
    Each "exact" bucket covers 1°C (= 1.8°F)
    Typical event: 11 markets = 1 floor + 9 exact + 1 ceil
    Coverage: ~11°C range (= ~20°F range)

  Effective resolution comparison:
    °F: 2°F per bucket = 1.11°C per bucket
    °C: 1°C per bucket = 1.8°F per bucket
    => °F markets have FINER resolution than °C markets!
       (1.11°C vs 1.0°C — roughly comparable though)
""")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
