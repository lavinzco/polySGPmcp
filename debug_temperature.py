"""Diagnostic: find temperature markets in raw Gamma API data."""

import asyncio
import json
import sys

import httpx

from polymarket.temperature import _TEMP_PATTERNS, parse_temperature_market
from polymarket.models import Market


async def main() -> None:
    base_url = "https://gamma-api.polymarket.com"
    headers = {"Accept-Encoding": "gzip, deflate"}

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:

        # ── Probe 1: keyword search via Gamma API ──
        print("=" * 72)
        print("  PROBE 1: Gamma API /markets?tag_slug= and text_query search")
        print("=" * 72)

        for search_param, search_val in [
            ("tag_slug", "temperature"),
            ("tag_slug", "weather"),
            ("tag_slug", "daily-temperature"),
            ("slug", "temperature"),
        ]:
            params = {"limit": 10, search_param: search_val, "active": True, "closed": False}
            resp = await client.get(f"{base_url}/markets", params=params)
            batch = resp.json() if resp.status_code == 200 else []
            print(f"\n  ?{search_param}={search_val} => {len(batch)} results (HTTP {resp.status_code})")
            for m in batch[:3]:
                print(f"    - {m.get('question', 'N/A')[:90]}")

        # ── Probe 2: brute-force deep pagination ──
        print("\n" + "=" * 72)
        print("  PROBE 2: Deep pagination (up to 3000 markets)")
        print("=" * 72)

        all_markets: list[dict] = []
        for page in range(30):
            offset = page * 100
            params = {"limit": 100, "offset": offset, "active": True, "closed": False}
            try:
                resp = await client.get(f"{base_url}/markets", params=params)
                resp.raise_for_status()
                batch = resp.json()
            except Exception as exc:
                print(f"  Page {page+1} (offset={offset}) failed: {exc}")
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                print(f"  Page {page+1}: {len(batch)} markets (last page)")
                break
            if page % 5 == 4:
                print(f"  ...fetched {len(all_markets)} so far")

        print(f"\n  Total markets fetched: {len(all_markets)}")

        # Scan for temperature-related keywords in questions
        temp_kws = ["temperature", "highest temp", "lowest temp", "high temp",
                     "degrees f", "degrees c"]
        temp_matches = []
        for m in all_markets:
            q = m.get("question", "").lower()
            for kw in temp_kws:
                if kw in q:
                    temp_matches.append(m)
                    break

        print(f"  Markets matching temperature keywords: {len(temp_matches)}")

        if temp_matches:
            print("\n  --- Temperature matches (first 30) ---")
            for i, m in enumerate(temp_matches[:30]):
                print(f"\n  [{i+1}] id: {m.get('id', 'N/A')}")
                print(f"      question: {m.get('question', 'N/A')}")
                print(f"      active: {m.get('active')}, closed: {m.get('closed')}")
                raw_tags = m.get("tags", None)
                print(f"      tags (raw type={type(raw_tags).__name__}): {json.dumps(raw_tags)}")
                print(f"      outcomePrices: {m.get('outcomePrices', 'N/A')}")
                print(f"      endDateIso: {m.get('endDateIso', 'N/A')}")
                # Also print all keys we haven't seen
                extra_keys = set(m.keys()) - {"id", "question", "active", "closed",
                    "tags", "outcomePrices", "endDateIso", "description", "outcomes",
                    "conditionId", "slug", "liquidity", "volume"}
                if extra_keys:
                    print(f"      extra keys: {sorted(extra_keys)}")

        # ── Probe 3: Try /events endpoint (Polymarket groups markets into events) ──
        print("\n" + "=" * 72)
        print("  PROBE 3: /events endpoint (markets may live under events)")
        print("=" * 72)

        for endpoint in ["/events", "/events?tag=weather", "/events?slug=temperature"]:
            full_url = f"{base_url}{endpoint}"
            try:
                resp = await client.get(full_url, params={"limit": 10, "active": True})
                if resp.status_code == 200:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("data", data.get("results", []))
                    print(f"\n  GET {endpoint} => {len(items) if isinstance(items, list) else 'non-list'} results")
                    if isinstance(items, list):
                        for ev in items[:5]:
                            title = ev.get("title", ev.get("question", ev.get("name", "?")))
                            slug = ev.get("slug", "")
                            n_markets = len(ev.get("markets", []))
                            print(f"    - {title[:80]} (slug={slug}, markets={n_markets})")
                else:
                    print(f"\n  GET {endpoint} => HTTP {resp.status_code}")
            except Exception as exc:
                print(f"\n  GET {endpoint} => Error: {exc}")

        # ── Probe 4: If we found temp markets, test regex ──
        if temp_matches:
            print("\n" + "=" * 72)
            print("  PROBE 4: Regex match test on real questions")
            print("=" * 72)

            for q_data in temp_matches[:10]:
                q = q_data.get("question", "")
                print(f"\n  Q: {q!r}")
                matched = False
                for i, pat in enumerate(_TEMP_PATTERNS):
                    match = pat.search(q)
                    if match:
                        print(f"  => Pattern {i+1} matched: {match.groupdict()}")
                        matched = True
                        break
                if not matched:
                    print(f"  => NO MATCH from any pattern")

                market_obj = Market(id="test", question=q)
                result = parse_temperature_market(market_obj)
                print(f"  => parse_temperature_market: {'OK' if result else 'FAILED'}")

        # ── Probe 5: Tag distribution ──
        print("\n" + "=" * 72)
        print("  PROBE 5: Tag distribution (top 30)")
        print("=" * 72)

        all_tags: dict[str, int] = {}
        tag_types_seen = set()
        for m in all_markets[:500]:
            tags = m.get("tags", None)
            tag_types_seen.add(type(tags).__name__)
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, dict):
                        label = t.get("label", t.get("slug", str(t)))
                    else:
                        label = str(t)
                    all_tags[label] = all_tags.get(label, 0) + 1
            elif isinstance(tags, str) and tags:
                all_tags[tags] = all_tags.get(tags, 0) + 1

        print(f"\n  Tag field types seen: {tag_types_seen}")
        sorted_tags = sorted(all_tags.items(), key=lambda x: -x[1])
        for tag, count in sorted_tags[:30]:
            flag = " <<<" if any(w in tag.lower() for w in ["weather", "temp", "climate"]) else ""
            print(f"    {tag}: {count}{flag}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
