"""Detailed look at one real sounding + feature extraction prototype."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import numpy as np


def main():
    # Use 48615 (Kota Bharu) 2026-06-26 00Z — confirmed available
    date = datetime(2026, 6, 26, 0)
    station = "48615"

    print(f"Station: {station} (Kota Bharu, Malaysia)")
    print(f"Date:    {date.strftime('%Y-%m-%d %HZ')}")
    print(f"Distance to WSSS (Changi): ~540 km NNE")
    print()

    df = WyomingUpperAir.request_data(date, station)
    print(f"Total levels: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Print full profile up to 500 hPa (we only care about lower troposphere)
    lower = df[df['pressure'] >= 500]
    print("=== Profile (surface to 500 hPa) ===")
    print(lower[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].to_string())
    print()

    # --- Feature extraction ---
    print("=== Extracted Features ===")

    # 1. Surface conditions
    sfc_temp = df['temperature'].iloc[0]
    sfc_dewpoint = df['dewpoint'].iloc[0]
    sfc_pressure = df['pressure'].iloc[0]
    print(f"Surface temp:     {sfc_temp} °C")
    print(f"Surface dewpoint: {sfc_dewpoint} °C")
    print(f"Surface pressure: {sfc_pressure} hPa")

    # 2. 850 hPa temperature
    idx_850 = (df['pressure'] - 850).abs().idxmin()
    t850 = df.loc[idx_850, 'temperature']
    h850 = df.loc[idx_850, 'height']
    p850 = df.loc[idx_850, 'pressure']
    print(f"~850 hPa:         {t850} °C at {h850} m (actual P={p850} hPa)")

    # 3. Lapse rate: surface to 850 hPa
    h_sfc = df['height'].iloc[0]
    dT = sfc_temp - t850
    dH = (h850 - h_sfc) / 1000  # km
    lapse_rate = dT / dH if dH > 0 else 0
    print(f"Lapse rate (sfc→850): {lapse_rate:.1f} °C/km (dry adiabatic ≈ 9.8, moist ≈ 5-6)")

    # 4. Inversion detection (temperature increase with height in lowest 3km)
    low_3km = df[df['height'] <= h_sfc + 3000].copy()
    inversions = []
    temps = low_3km['temperature'].values
    heights = low_3km['height'].values
    pressures = low_3km['pressure'].values
    for i in range(1, len(temps)):
        if temps[i] > temps[i-1]:
            inversions.append({
                'base_h': heights[i-1],
                'top_h': heights[i],
                'base_t': temps[i-1],
                'top_t': temps[i],
                'base_p': pressures[i-1],
                'top_p': pressures[i],
                'strength': temps[i] - temps[i-1],
            })

    if inversions:
        print(f"Low-level inversions: {len(inversions)} found")
        for inv in inversions:
            print(f"  {inv['base_h']:.0f}→{inv['top_h']:.0f}m "
                  f"({inv['base_p']:.0f}→{inv['top_p']:.0f} hPa): "
                  f"{inv['base_t']:.1f}→{inv['top_t']:.1f}°C "
                  f"(+{inv['strength']:.1f}°C)")
    else:
        print("Low-level inversions: NONE (unstable/neutral profile)")

    # 5. Mixed layer depth estimate (height where T first drops below sfc_temp - 3°C)
    ml_depth = None
    for i in range(1, len(temps)):
        if temps[i] < sfc_temp - 3:
            ml_depth = heights[i] - h_sfc
            break
    print(f"Approx mixed layer depth: {ml_depth or 'N/A'} m")

    # Also try Medan (96035) — closer to Singapore
    print("\n" + "=" * 60)
    print("=== Medan (96035) — ~570km W of Singapore ===")
    df2 = WyomingUpperAir.request_data(date, "96035")
    sfc2 = df2['temperature'].iloc[0]
    idx2 = (df2['pressure'] - 850).abs().idxmin()
    t850_2 = df2.loc[idx2, 'temperature']
    h850_2 = df2.loc[idx2, 'height']
    h_sfc2 = df2['height'].iloc[0]
    dT2 = sfc2 - t850_2
    dH2 = (h850_2 - h_sfc2) / 1000
    lr2 = dT2 / dH2 if dH2 > 0 else 0
    print(f"Surface: {sfc2}°C, 850hPa: {t850_2}°C")
    print(f"Lapse rate: {lr2:.1f} °C/km")

    low2 = df2[df2['height'] <= h_sfc2 + 3000]
    t2 = low2['temperature'].values
    h2 = low2['height'].values
    inv2 = any(t2[i] > t2[i-1] for i in range(1, len(t2)))
    print(f"Low-level inversion: {'YES' if inv2 else 'NO'}")


if __name__ == "__main__":
    main()
