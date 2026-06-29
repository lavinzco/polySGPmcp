"""Repeat-sampling stability test for Part B extreme scenarios."""
import asyncio
import sys
import io
import statistics

from dotenv import load_dotenv
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

N_REPEATS = 10


def build_scenarios():
    from agent.models import DayForecast, GammaMarket, WeatherForecast

    weather = WeatherForecast(
        location="TestCity",
        temp_c=38.0, temp_f=100.4, humidity=25,
        wind_speed_kmph=5, wind_dir="S", weather_desc="Clear sky",
        feels_like_c=42.0, pressure_mb=1008, precip_mm=0.0,
        visibility_km=10, uv_index=10,
        forecast_3day=[
            DayForecast(date="2026-06-26", max_temp_c=39.0, min_temp_c=28.0,
                        avg_humidity=30, total_precip_mm=0.0, condition="Sunny"),
            DayForecast(date="2026-06-27", max_temp_c=41.0, min_temp_c=29.0,
                        avg_humidity=25, total_precip_mm=0.0, condition="Sunny"),
            DayForecast(date="2026-06-28", max_temp_c=40.0, min_temp_c=28.0,
                        avg_humidity=28, total_precip_mm=0.0, condition="Sunny"),
        ],
    )

    s1 = GammaMarket(
        id="exact-40c",
        question="Will the highest temperature in TestCity be 40°C on June 27?",
        description="Resolves YES if max temp is exactly 40°C on June 27, 2026.",
        outcome_yes_price=0.08, outcome_no_price=0.92, quality="high",
    )

    s2 = GammaMarket(
        id="range-40-41c",
        question="Will the highest temperature in TestCity be between 40-41°C on June 27?",
        description="Resolves YES if max temp is between 40-41°C on June 27, 2026.",
        outcome_yes_price=0.12, outcome_no_price=0.88, quality="high",
    )

    return weather, s1, s2


async def main():
    from agent.strategy import StrategyEngine
    from common.llm.router import LLMRouter
    from rich.console import Console
    from rich.table import Table

    console = Console()
    weather, scenario1, scenario2 = build_scenarios()

    router = LLMRouter()
    engine = StrategyEngine(router)

    for label, market in [
        ("Scenario 1: exact 40°C (YES=0.08, forecast=41°C)", scenario1),
        ("Scenario 2: range 40-41°C (YES=0.12, forecast=41°C)", scenario2),
    ]:
        console.rule(f"[bold]{label}")
        console.print(f"Running {N_REPEATS} repeats with identical inputs...\n")

        results = []
        for i in range(N_REPEATS):
            signal, raw = await engine.evaluate(weather, market)
            results.append({
                "run": i + 1,
                "action": signal.action,
                "confidence": signal.confidence,
                "size": signal.suggested_size_usd,
                "rationale": signal.rationale,
            })
            console.print(f"  Run {i+1}/{N_REPEATS}: {signal.action} conf={signal.confidence:.2f} size=${signal.suggested_size_usd:.2f}")

        # Build results table
        table = Table(title=f"Results: {market.id}", show_lines=True)
        table.add_column("Run", justify="center", width=4)
        table.add_column("Action", justify="center", width=10)
        table.add_column("Confidence", justify="right", width=11)
        table.add_column("Size ($)", justify="right", width=9)
        table.add_column("Rationale (first 120 chars)", width=60)

        for r in results:
            conf_str = f"{r['confidence']:.2f}"
            size_str = f"{r['size']:.2f}"
            rat_short = r['rationale'][:120] + ("..." if len(r['rationale']) > 120 else "")
            table.add_row(str(r['run']), r['action'], conf_str, size_str, rat_short)

        console.print(table)

        # Statistics
        actions = [r['action'] for r in results]
        confidences = [r['confidence'] for r in results]
        sizes = [r['size'] for r in results]

        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        console.print(f"\n[bold]Action distribution:[/bold] {action_counts}")
        consistent = len(action_counts) == 1
        console.print(f"[bold]Consistent:[/bold] {'YES' if consistent else '[red]NO[/red]'}")

        if confidences:
            mean_conf = statistics.mean(confidences)
            std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            mean_size = statistics.mean(sizes)
            std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
            console.print(f"[bold]Confidence:[/bold] mean={mean_conf:.3f}  std={std_conf:.3f}")
            console.print(f"[bold]Size ($):[/bold]   mean={mean_size:.2f}  std={std_size:.2f}")

        # Print full rationale for inconsistent runs
        if not consistent:
            console.print(f"\n[bold red]Inconsistent actions detected! Full rationales:[/bold red]")
            majority_action = max(action_counts, key=action_counts.get)
            for r in results:
                if r['action'] != majority_action:
                    console.print(f"\n  [red]Run {r['run']} ({r['action']}, conf={r['confidence']:.2f}):[/red]")
                    console.print(f"    {r['rationale']}")

            console.print(f"\n  [green]Sample majority run ({majority_action}):[/green]")
            majority_run = next(r for r in results if r['action'] == majority_action)
            console.print(f"    {majority_run['rationale']}")

        console.print()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
