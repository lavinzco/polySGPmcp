"""Live test: full evaluate() with 5x sampling + aggregation on both scenarios."""
import asyncio
import sys
import io

from dotenv import load_dotenv
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


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
    from agent.risk import RiskManager
    from agent.models import PortfolioState
    from common.llm.router import LLMRouter
    from rich.console import Console
    from rich.table import Table

    console = Console()
    weather, scenario1, scenario2 = build_scenarios()

    router = LLMRouter()
    engine = StrategyEngine(router)  # uses AGENT_STRATEGY_N_REPEATS from .env (5)
    risk = RiskManager()
    portfolio = PortfolioState(total_balance_usd=1000, daily_pnl_usd=0)

    for label, market in [
        ("Scenario 1: exact 40°C (YES=0.08)", scenario1),
        ("Scenario 2: range 40-41°C (YES=0.12)", scenario2),
    ]:
        console.rule(f"[bold]{label}")

        signal, raw_outputs = await engine.evaluate(weather, market)

        # Show individual samples
        table = Table(title=f"Raw Samples ({signal.n_samples} LLM calls)", show_lines=True)
        table.add_column("#", justify="center", width=3)
        table.add_column("Action", justify="center", width=10)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Size", justify="right", width=8)
        table.add_column("Rationale (first 100 chars)", width=55)

        for i, s in enumerate(signal.raw_samples):
            table.add_row(
                str(i + 1), s.action,
                f"{s.confidence:.2f}", f"${s.suggested_size_usd:.0f}",
                s.rationale[:100] + ("..." if len(s.rationale) > 100 else ""),
            )
        console.print(table)

        # Show aggregated result
        console.print(f"\n[bold]Aggregated Signal:[/bold]")
        console.print(f"  action:          {signal.action}")
        console.print(f"  confidence:      {signal.confidence:.2f}")
        console.print(f"  size:            ${signal.suggested_size_usd:.2f}")
        console.print(f"  agreement_ratio: {signal.agreement_ratio:.0%}")
        console.print(f"  quality:         {signal.quality}")
        console.print(f"  n_samples:       {signal.n_samples}")
        console.print(f"  rationale:       {signal.rationale}")

        # Run through risk manager
        filtered = risk.filter(signal, portfolio)
        if filtered:
            console.print(f"\n[bold]After RiskManager:[/bold]")
            console.print(f"  action: {filtered.action}  size: ${filtered.suggested_size_usd:.2f}")
            if filtered.action != "hold" and filtered.suggested_size_usd < signal.suggested_size_usd:
                console.print(f"  (size adjusted from ${signal.suggested_size_usd:.2f})")
        else:
            console.print(f"\n[bold red]RiskManager: BLOCKED[/bold red]")

        console.print()

    console.rule("[bold]DONE")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
