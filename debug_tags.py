"""Step 1: Discover weather/temperature tag IDs from Gamma API."""
import asyncio, json, sys
import httpx

async def main():
    base = "https://gamma-api.polymarket.com"
    hdrs = {"Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(timeout=30, headers=hdrs) as c:
        # Fetch all tags
        resp = await c.get(f"{base}/tags")
        resp.raise_for_status()
        tags = resp.json()
        print(f"Total tags: {len(tags)}")
        print(f"Tag object keys: {list(tags[0].keys()) if tags else 'empty'}")
        print()

        # Search for weather-related
        weather_kw = ["weather", "temperature", "temp", "climate", "daily"]
        hits = []
        for t in tags:
            name = str(t.get("label", t.get("name", t.get("slug", "")))).lower()
            slug = str(t.get("slug", "")).lower()
            for kw in weather_kw:
                if kw in name or kw in slug:
                    hits.append(t)
                    break

        print(f"Weather-related tags: {len(hits)}")
        for t in hits:
            print(f"  {json.dumps(t)}")

        # If nothing found, dump all tags
        if not hits:
            print("\nAll tags (first 50):")
            for t in tags[:50]:
                print(f"  {json.dumps(t)}")
            if len(tags) > 50:
                print(f"  ... and {len(tags)-50} more")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
