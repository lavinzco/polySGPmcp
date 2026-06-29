"""Check if any Singapore temperature markets have already settled."""
import asyncio
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

import httpx


async def main():
    # Check a few recent Singapore market IDs from our scan
    # (these are June 28 markets — they should settle after June 29 data is published)
    market_ids = ["2688223", "2688224", "2688225", "2688226", "2688227"]

    async with httpx.AsyncClient(timeout=30) as client:
        for mid in market_ids:
            resp = await client.get(f"https://gamma-api.polymarket.com/markets/{mid}")
            data = resp.json()
            print(f"Market {mid}:")
            print(f"  question: {data.get('question', 'N/A')}")
            print(f"  closed:   {data.get('closed', 'N/A')}")
            print(f"  active:   {data.get('active', 'N/A')}")

            prices_str = data.get("outcomePrices", "[]")
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            print(f"  prices:   {prices}")

            resolved = data.get("resolvedTo")
            print(f"  resolvedTo: {resolved}")
            print()

    # Also check an older event that might already be settled
    # Let's look for closed=true markets
    resp2 = await httpx.AsyncClient(timeout=30).get(
        "https://gamma-api.polymarket.com/events",
        params={"tag_id": 84, "limit": 5, "closed": True, "active": False},
    )
    closed_events = resp2.json()
    print(f"\n=== Closed weather events: {len(closed_events)} ===")
    for event in closed_events[:2]:
        print(f"\nEvent: {event.get('title')}")
        for m in event.get("markets", [])[:3]:
            prices = json.loads(m.get("outcomePrices", "[]"))
            print(f"  {m['id']}: {m.get('question', '')[:60]}  closed={m.get('closed')}  prices={prices}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
