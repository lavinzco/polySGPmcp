"""Step 2: Deep tag discovery + event search for temperature markets."""
import asyncio, json, sys
import httpx

async def main():
    base = "https://gamma-api.polymarket.com"
    hdrs = {"Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(timeout=30, headers=hdrs) as c:

        # 1. Paginate /tags to find weather-related ones
        print("=" * 72)
        print("  PHASE 1: Paginate /tags exhaustively")
        print("=" * 72)
        all_tags = []
        for offset in range(0, 5000, 100):
            resp = await c.get(f"{base}/tags", params={"limit": 100, "offset": offset})
            if resp.status_code != 200:
                print(f"  offset={offset}: HTTP {resp.status_code}")
                break
            batch = resp.json()
            if not batch:
                break
            all_tags.extend(batch)
            if len(batch) < 100:
                break

        print(f"  Total tags fetched: {len(all_tags)}")
        weather_hits = []
        for t in all_tags:
            label = str(t.get("label", "")).lower()
            slug = str(t.get("slug", "")).lower()
            if any(kw in label or kw in slug for kw in
                   ["weather", "temperature", "temp ", "climate", "daily temp",
                    "heat", "forecast", "hurricane", "storm"]):
                weather_hits.append(t)
                print(f"  HIT: {json.dumps(t)}")

        if not weather_hits:
            print("  No weather tags found via keyword search")

        # 2. Try text-based search on /events
        print("\n" + "=" * 72)
        print("  PHASE 2: Search /events with text queries")
        print("=" * 72)

        for query_param_name in ["q", "query", "search", "title", "slug"]:
            for query_val in ["temperature", "highest temperature", "weather"]:
                params = {query_param_name: query_val, "limit": 5, "active": True}
                resp = await c.get(f"{base}/events", params=params)
                data = resp.json() if resp.status_code == 200 else []
                items = data if isinstance(data, list) else []
                relevant = [e for e in items if "temp" in str(e.get("title", e.get("question", ""))).lower()]
                marker = f" <<< {len(relevant)} relevant" if relevant else ""
                print(f"  ?{query_param_name}={query_val} => {len(items)} results{marker}")
                for e in relevant[:3]:
                    print(f"    - {e.get('title', e.get('question', '?'))[:90]}")

        # 3. Try /events with different params
        print("\n" + "=" * 72)
        print("  PHASE 3: /events pagination + category exploration")
        print("=" * 72)

        # Check response structure
        resp = await c.get(f"{base}/events", params={"limit": 5, "active": True, "closed": False})
        data = resp.json()
        if isinstance(data, list) and data:
            print(f"  /events returns: list of {len(data)} items")
            print(f"  Event keys: {list(data[0].keys())}")
            print(f"  First event title: {data[0].get('title', 'N/A')}")
            markets_in_event = data[0].get("markets", [])
            print(f"  First event has {len(markets_in_event)} markets")
            if markets_in_event:
                print(f"  Market keys: {list(markets_in_event[0].keys())}")
        elif isinstance(data, dict):
            print(f"  /events returns: dict with keys {list(data.keys())}")
            if "data" in data:
                items = data["data"]
                print(f"  data field has {len(items)} items")

        # Deep paginate events, look for temperature
        print("\n  Deep paginating /events to find temperature markets...")
        events_found = []
        cursor = None
        for page in range(20):
            params = {"limit": 100, "active": True, "closed": False}
            if cursor:
                params["next_cursor"] = cursor
            else:
                params["offset"] = page * 100
            resp = await c.get(f"{base}/events", params=params)
            if resp.status_code != 200:
                print(f"  page {page}: HTTP {resp.status_code}")
                break
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            if not items:
                print(f"  page {page}: empty, stopping")
                break

            for ev in items:
                title = str(ev.get("title", ev.get("question", ""))).lower()
                if any(kw in title for kw in ["temperature", "highest temp", "lowest temp",
                                               "weather", "daily temp"]):
                    events_found.append(ev)

            if isinstance(data, dict) and not data.get("has_more", True):
                print(f"  page {page}: has_more=False")
                break

            if len(items) < 100:
                print(f"  page {page}: {len(items)} items (last page)")
                break

            if page % 5 == 4:
                print(f"  ...scanned {(page+1)*100} events")

        print(f"\n  Temperature events found: {len(events_found)}")
        for ev in events_found[:10]:
            title = ev.get("title", ev.get("question", "?"))
            slug = ev.get("slug", "")
            n_markets = len(ev.get("markets", []))
            tags = ev.get("tags", [])
            print(f"\n  Title: {title}")
            print(f"  Slug: {slug}")
            print(f"  Markets: {n_markets}")
            print(f"  Tags: {json.dumps(tags)}")
            # Print first 3 sub-markets
            for mk in ev.get("markets", [])[:3]:
                print(f"    market: {mk.get('question', '?')[:80]}")
                print(f"      id={mk.get('id', '?')} prices={mk.get('outcomePrices', '?')}")

        # 4. If still nothing, try the Polymarket frontend API
        print("\n" + "=" * 72)
        print("  PHASE 4: Try alternative API paths")
        print("=" * 72)

        alt_urls = [
            f"{base}/markets?_q=temperature&limit=5",
            f"{base}/markets?search=temperature&limit=5",
            f"{base}/events?_q=temperature&limit=5",
            f"{base}/events?tag=Weather&limit=5",
        ]
        for url in alt_urls:
            try:
                resp = await c.get(url)
                data = resp.json() if resp.status_code == 200 else []
                items = data if isinstance(data, list) else data.get("data", [])
                n = len(items) if isinstance(items, list) else "?"
                print(f"  {url.replace(base,'')} => {n} results (HTTP {resp.status_code})")
                if isinstance(items, list):
                    for it in items[:2]:
                        q = it.get("title", it.get("question", "?"))
                        print(f"    - {q[:90]}")
            except Exception as exc:
                print(f"  {url.replace(base,'')} => Error: {exc}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
