"""Step 5: Dump sub-market questions from temperature events for regex design."""
import asyncio, json, sys, io
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def main():
    base = "https://gamma-api.polymarket.com"
    hdrs = {"Accept-Encoding": "gzip, deflate"}
    async with httpx.AsyncClient(timeout=30, headers=hdrs) as c:

        # Get active temperature events from tag_id=84
        resp = await c.get(f"{base}/events", params={
            "tag_id": 84, "active": True, "closed": False, "limit": 20
        })
        events = resp.json() if resp.status_code == 200 else []

        temp_events = [e for e in events
                       if "temperature" in e.get("title","").lower()
                       and "highest" in e.get("title","").lower()]

        print(f"Active temperature events (from first page): {len(temp_events)}")

        for ev in temp_events[:3]:
            print(f"\n{'=' * 72}")
            print(f"  EVENT: {ev.get('title')}")
            print(f"  id: {ev.get('id')}")
            print(f"  slug: {ev.get('slug')}")
            print(f"  active: {ev.get('active')}, closed: {ev.get('closed')}")
            print(f"  tags: {json.dumps(ev.get('tags', []))}")
            print(f"  Markets ({len(ev.get('markets',[]))}):")

            for mk in ev.get("markets", []):
                print(f"\n    question: {mk.get('question')}")
                print(f"    id: {mk.get('id')}")
                print(f"    outcomePrices: {mk.get('outcomePrices')}")
                print(f"    active: {mk.get('active')}, closed: {mk.get('closed')}")
                print(f"    groupItemTitle: {mk.get('groupItemTitle')}")
                # Dump all keys to see what's available
                interesting_keys = {k: v for k, v in mk.items()
                                    if k not in ('description','icon','image','conditionId',
                                                 'slug','outcomes','clobTokenIds')}
                print(f"    other fields: {json.dumps(interesting_keys, default=str)[:200]}")

        # Also dump one closed event to see settlement format
        print(f"\n\n{'=' * 72}")
        print("  CLOSED temperature events (for settlement format)")
        print(f"{'=' * 72}")
        resp = await c.get(f"{base}/events", params={
            "tag_id": 84, "closed": True, "limit": 50
        })
        closed_events = resp.json() if resp.status_code == 200 else []
        closed_temp = [e for e in closed_events
                       if "highest temperature" in e.get("title","").lower()]

        print(f"Closed temperature events: {len(closed_temp)}")
        for ev in closed_temp[:2]:
            print(f"\n  EVENT: {ev.get('title')}")
            for mk in ev.get("markets", [])[:3]:
                print(f"    Q: {mk.get('question')}")
                print(f"    prices: {mk.get('outcomePrices')}")
                print(f"    closed: {mk.get('closed')}, resolved: {mk.get('resolved')}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
