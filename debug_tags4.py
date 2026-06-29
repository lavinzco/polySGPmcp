"""Step 4: Find daily temperature markets - try text search and deeper pagination."""
import asyncio, json, sys, io
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    base = "https://gamma-api.polymarket.com"
    hdrs = {"Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(timeout=30, headers=hdrs) as c:

        # 1. Text search on /markets for temperature-related questions
        print("=" * 72)
        print("  TEST 1: /markets text search for 'highest temperature'")
        print("=" * 72)
        for param in ["_q", "q", "query", "search", "text_query"]:
            resp = await c.get(f"{base}/markets",
                              params={param: "highest temperature", "limit": 5, "active": True})
            data = resp.json() if resp.status_code == 200 else []
            items = data if isinstance(data, list) else []
            temp_hits = [m for m in items if "temperature" in m.get("question", "").lower()
                         or "temp" in m.get("question", "").lower()]
            print(f"  ?{param}='highest temperature' => {len(items)} total, {len(temp_hits)} temp-related")
            for m in temp_hits[:3]:
                print(f"    - {m.get('question','')[:90]}")
            if not temp_hits and items:
                print(f"    (first result: {items[0].get('question','')[:70]})")

        # 2. Paginate ALL events under tag_id=84 (Weather)
        print(f"\n{'=' * 72}")
        print(f"  TEST 2: Paginate all events under tag_id=84 (Weather)")
        print(f"{'=' * 72}")
        all_events = []
        for offset in range(0, 500, 100):
            resp = await c.get(f"{base}/events",
                              params={"tag_id": 84, "limit": 100, "offset": offset})
            data = resp.json() if resp.status_code == 200 else []
            items = data if isinstance(data, list) else []
            if not items:
                break
            all_events.extend(items)
            if len(items) < 100:
                break

        print(f"  Total Weather events: {len(all_events)}")
        for ev in all_events:
            title = ev.get("title", "").lower()
            if "temperature" in title or "highest" in title or "temp" in title:
                n_mk = len(ev.get("markets", []))
                print(f"  TEMP HIT: {ev.get('title','')} ({n_mk} markets)")

        # 3. Try searching with CLOB/STRAPI endpoints
        print(f"\n{'=' * 72}")
        print(f"  TEST 3: Try CLOB token search")
        print(f"{'=' * 72}")
        clob_url = "https://clob.polymarket.com"
        try:
            resp = await c.get(f"{clob_url}/markets",
                               params={"next_cursor": "MA=="},
                               timeout=10)
            print(f"  CLOB /markets: HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                    items = data.get("data", data.get("markets", []))
                    if isinstance(items, list) and items:
                        print(f"  {len(items)} markets")
                        print(f"  Market keys: {list(items[0].keys())}")
                        print(f"  First: {items[0].get('question', items[0].get('condition_id',''))[:80]}")
        except Exception as e:
            print(f"  CLOB error: {e}")

        # 4. Try Strapi search
        print(f"\n{'=' * 72}")
        print(f"  TEST 4: Try strapi/search")
        print(f"{'=' * 72}")
        strapi_url = "https://strapi-matic.polymarket.com"
        try:
            resp = await c.get(f"{strapi_url}/markets",
                               params={"_q": "temperature", "_limit": 5},
                               timeout=10)
            print(f"  Strapi /markets?_q=temperature: HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                items = data if isinstance(data, list) else data.get("data", [])
                print(f"  {len(items)} results")
                for m in (items if isinstance(items, list) else [])[:3]:
                    print(f"    - {m.get('question', m.get('title', ''))[:80]}")
        except Exception as e:
            print(f"  Strapi error: {e}")

        # 5. Try gamma with broader tag search
        print(f"\n{'=' * 72}")
        print(f"  TEST 5: Broader tag search (paginate all tags, look for 'daily')")
        print(f"{'=' * 72}")
        all_tags = []
        for offset in range(0, 10000, 500):
            resp = await c.get(f"{base}/tags", params={"limit": 500, "offset": offset})
            batch = resp.json() if resp.status_code == 200 else []
            if not batch:
                break
            all_tags.extend(batch)
            if len(batch) < 500:
                break

        print(f"  Total tags: {len(all_tags)}")
        daily_tags = [t for t in all_tags
                      if any(kw in str(t.get("label","")).lower()
                             for kw in ["daily", "temperature", "highest", "city",
                                        "london", "tokyo", "new york", "paris"])]
        print(f"  Relevant tags: {len(daily_tags)}")
        for t in daily_tags:
            print(f"    id={t.get('id')} label={t.get('label')} slug={t.get('slug')}")
            # Try each tag
            resp = await c.get(f"{base}/events",
                               params={"tag_id": t.get("id"), "active": True, "limit": 3})
            ev_data = resp.json() if resp.status_code == 200 else []
            ev_items = ev_data if isinstance(ev_data, list) else []
            if ev_items:
                print(f"      => {len(ev_items)} events")
                for ev in ev_items[:2]:
                    print(f"         {ev.get('title','')[:70]}")
            else:
                print(f"      => 0 events")

        # 6. Try market slug search for temperature
        print(f"\n{'=' * 72}")
        print(f"  TEST 6: /markets?slug= variants for temperature")
        print(f"{'=' * 72}")
        for slug_val in ["highest-temperature", "temperature-in", "daily-temperature",
                         "london-temperature", "will-the-highest"]:
            resp = await c.get(f"{base}/markets",
                               params={"slug": slug_val, "limit": 5})
            data = resp.json() if resp.status_code == 200 else []
            items = data if isinstance(data, list) else []
            print(f"  ?slug={slug_val} => {len(items)}")
            for m in items[:2]:
                print(f"    - {m.get('question','')[:80]}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
