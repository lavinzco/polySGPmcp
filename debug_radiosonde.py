"""Diagnose radiosonde station availability near Singapore (WSSS).

Candidate stations in the region:
  48698 - Singapore/Changi (1.37N, 103.98E) — discontinued?
  48694 - Petaling Jaya, Malaysia (3.1N, 101.65E) — ~300km NW
  48657 - Kuantan, Malaysia (3.78N, 103.22E) — ~270km N
  96315 - Medan, Indonesia (3.57N, 98.68E) — ~570km W
  96749 - Jakarta, Indonesia (6.18S, 106.85E) — ~1100km S
  48615 - Kota Bharu, Malaysia (6.17N, 102.28E) — ~540km N

The closest that routinely does 00Z/12Z soundings is typically 48698
(Singapore itself) or 48694 (Petaling Jaya). Let's test both.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir


def try_station(station: str, label: str, date: datetime):
    print(f"\n{'='*60}")
    print(f"Station: {station} ({label})")
    print(f"Date:    {date.strftime('%Y-%m-%d %HZ')}")
    print(f"{'='*60}")
    try:
        df = WyomingUpperAir.request_data(date, station)
        print(f"SUCCESS — {len(df)} levels")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 10 levels:")
        print(df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].head(10).to_string())
        print(f"\nSurface temp: {df['temperature'].iloc[0]} °C")
        print(f"Surface pressure: {df['pressure'].iloc[0]} hPa")

        # Check for 850 hPa level
        p850 = df[df['pressure'] <= 850].head(1)
        if not p850.empty:
            print(f"850 hPa temp: {p850['temperature'].iloc[0]} °C at {p850['height'].iloc[0]} m")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def main():
    # Try recent dates at 00Z (morning Singapore time = 08:00 SGT)
    # and 12Z (evening = 20:00 SGT)
    now = datetime.utcnow()

    stations = [
        ("48698", "Singapore/Changi"),
        ("48694", "Petaling Jaya, Malaysia"),
    ]

    # Try last 3 days at 00Z
    for days_ago in range(0, 4):
        date = (now - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        for station, label in stations:
            try_station(station, label, date)

    # Also try 12Z for most recent day
    date_12z = (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    for station, label in stations:
        try_station(station, label, date_12z)


if __name__ == "__main__":
    main()
