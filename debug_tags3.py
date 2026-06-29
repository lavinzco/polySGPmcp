"""Step 3: Test tag_id filtering on /events endpoint."""
import asyncio, json, sys
import httpx

async def main():
    base = "https://gamma-api.polymarket.com"
    hdrs = {"Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(timeout=30, headers=hdrs) as c:

        tag_ids = {
            "104615": "temperature",
            "84": "Weather",
            "1474": "climate & weather",
            "104180": "Weather & Science",
            "426": "record temperatures",
            "117": "heat",
        }

        for tag_id, label in tag_ids.items():
            print(f"\n{'=' * 72}")
            print(f"  tag_id={tag_id} ({label})")
            print(f"{'=' * 72}")

            params = {"tag_id": tag_id, "active": True, "closed": False, "limit": 20}
            resp = await c.get(f"{base}/events", params=params)
            data = resp.json() if resp.status_code == 200 else []
            items = data if isinstance(data, list) else data.get("data", [])
            print(f"  HTTP {resp.status_code}, {len(items)} events")

            for ev in items[:5]:
                title = ev.get("title", "?")
                n_mk = len(ev.get("markets", []))
                print(f"  - {title[:90]} ({n_mk} markets)")
                for mk in ev.get("markets", [])[:2]:
                    q = mk.get("question", "?")
                    prices = mk.get("outcomePrices", "?")
                    print(f"      Q: {q[:85]}")
                    print(f"      Prices: {prices}")

            # Check if same results regardless of tag_id (API ignoring param)
            if items:
                first_title = items[0].get("title", "")
                print(f"\n  First result: {first_title[:60]}")

        # Also try /markets with tag_id
        print(f"\n{'=' * 72}")
        print(f"  /markets?tag_id=104615 (temperature)")
        print(f"{'=' * 72}")
        resp = await c.get(f"{base}/markets", params={"tag_id": "104615", "limit": 10})
        data = resp.json()
        items = data if isinstance(data, list) else []
        print(f"  {len(items)} markets")
        for m in items[:5]:
            print(f"  - {m.get('question','?')[:90]}")

        # Try /events with slug filter for temperature
        print(f"\n{'=' * 72}")
        print(f"  /events?slug_contains=temperature (if supported)")
        print(f"{'=' * 72}")
        for param in ["slug", "slug_contains", "title_contains"]:
            resp = await c.get(f"{base}/events",
                              params={param: "temperature", "limit": 5, "active": True})
            data = resp.json() if resp.status_code == 200 else []
            items = data if isinstance(data, list) else []
            titles = [e.get("title","")[:50] for e in items[:3]]
            print(f"  ?{param}=temperature => {len(items)} results: {titles}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
