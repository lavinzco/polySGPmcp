"""Compare prompts with and without manual sounding note."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agent.manual_input import DailySoundingNote, InversionStrength
from agent.models import WeatherForecast, GammaMarket, DayForecast
from agent.prompts import build_strategy_prompt


def main():
    weather = WeatherForecast(
        location="Singapore",
        temp_c=31.0, temp_f=87.8, humidity=82,
        wind_speed_kmph=12, wind_dir="SSW", weather_desc="Partly cloudy",
        feels_like_c=36.0, pressure_mb=1009, precip_mm=0.5,
        visibility_km=10, uv_index=11,
        forecast_3day=[
            DayForecast(date="2026-06-28", max_temp_c=33.0, min_temp_c=26.0,
                        avg_humidity=78, total_precip_mm=5.2, condition="Thunderstorms"),
            DayForecast(date="2026-06-29", max_temp_c=32.0, min_temp_c=25.0,
                        avg_humidity=80, total_precip_mm=8.0, condition="Showers"),
            DayForecast(date="2026-06-30", max_temp_c=34.0, min_temp_c=26.0,
                        avg_humidity=75, total_precip_mm=2.0, condition="Partly cloudy"),
        ],
    )
    market = GammaMarket(
        id="2688225",
        question="Will the highest temperature in Singapore be 33°C on June 28?",
        description="Resolves based on Singapore Changi Airport (WSSS) via Wunderground",
        outcome_yes_price=0.30, outcome_no_price=0.70,
        quality="high",
    )

    # === RUN 1: No sounding note ===
    prompt1 = build_strategy_prompt(weather, market, sounding_note=None)
    print("=" * 70)
    print("PROMPT WITHOUT SOUNDING NOTE")
    print("=" * 70)
    print(prompt1)

    # === RUN 2: With sounding note ===
    note = DailySoundingNote(
        target_date="2026-06-28",
        inversion=InversionStrength.STRONG,
        surface_temp_c=26.5,
        note="Windy shows strong low-level inversion below 950hPa, surface heating likely capped",
        submitted_at="2026-06-28T08:15:00+08:00",
    )
    prompt2 = build_strategy_prompt(weather, market, sounding_note=note)
    print("\n" + "=" * 70)
    print("PROMPT WITH SOUNDING NOTE")
    print("=" * 70)
    print(prompt2)

    # === DIFF highlight ===
    lines1 = set(prompt1.splitlines())
    lines2 = set(prompt2.splitlines())
    added = lines2 - lines1
    removed = lines1 - lines2

    print("\n" + "=" * 70)
    print("DIFF: Lines only in WITH-sounding version (+)")
    print("=" * 70)
    for line in sorted(added):
        if line.strip():
            print(f"  + {line}")

    print("\nDIFF: Lines only in WITHOUT-sounding version (-)")
    for line in sorted(removed):
        if line.strip():
            print(f"  - {line}")


if __name__ == "__main__":
    main()
