"""Check Singapore temperature market descriptions for settlement source."""
import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from polymarket.client import GammaClient
from polymarket.temperature import find_temperature_markets_from_events, is_temperature_event


async def main():
    gamma = GammaClient()
    events = await gamma.get_events_by_tag()

    sg_events = [e for e in events if is_temperature_event(e) and "singapore" in e.title.lower()]
    print(f"Found {len(sg_events)} Singapore temperature events\n")

    for event in sg_events:
        print("=" * 80)
        print(f"EVENT TITLE: {event.title}")
        print(f"EVENT ID:    {event.id}")
        print(f"MARKETS:     {len(event.markets)}")
        print()

        for i, m in enumerate(event.markets[:3]):
            print(f"  --- Market {i+1} ---")
            print(f"  ID:       {m.id}")
            print(f"  Question: {m.question}")
            print(f"  Description (full):")
            print(f"  {m.description}")
            print()

        if len(event.markets) > 3:
            print(f"  ... and {len(event.markets) - 3} more markets")
        print()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
