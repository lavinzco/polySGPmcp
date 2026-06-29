"""Extended radiosonde diagnostics — wider station/date search."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir


def try_station(station: str, label: str, date: datetime):
    try:
        df = WyomingUpperAir.request_data(date, station)
        print(f"  OK  {station} ({label}) {date.strftime('%Y-%m-%d %HZ')} — {len(df)} levels, sfc={df['temperature'].iloc[0]}°C")
        return df
    except Exception as e:
        print(f"  FAIL {station} ({label}) {date.strftime('%Y-%m-%d %HZ')} — {e}")
        return None


def main():
    # 1. First verify Wyoming service works at all with a known reliable station
    print("=== Sanity check: US station (72451 = Green Bay, WI) ===")
    yesterday = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    try_station("72451", "Green Bay WI (control)", yesterday)
    try_station("72451", "Green Bay WI (control)", yesterday.replace(hour=12))

    # 2. Try all plausible SE Asia stations
    print("\n=== SE Asia stations, last 7 days at 00Z ===")
    stations = [
        ("48698", "Singapore/Changi"),
        ("48694", "Petaling Jaya MY"),
        ("48657", "Kuantan MY"),
        ("48615", "Kota Bharu MY"),
        ("96035", "Medan ID"),
        ("96749", "Jakarta/Cengkareng ID"),
        ("96315", "Pangkalpinang ID"),
        ("48900", "Sepang MY"),  # KLIA area
        ("96933", "Surabaya ID"),
    ]

    for days_ago in range(1, 8):
        date = (datetime.utcnow() - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"\n--- {date.strftime('%Y-%m-%d')} 00Z ---")
        for sid, label in stations:
            try_station(sid, label, date)

    # 3. Try 12Z for recent days
    print("\n=== 12Z checks (last 3 days) ===")
    for days_ago in range(1, 4):
        date = (datetime.utcnow() - timedelta(days=days_ago)).replace(hour=12, minute=0, second=0, microsecond=0)
        print(f"\n--- {date.strftime('%Y-%m-%d')} 12Z ---")
        for sid, label in stations:
            try_station(sid, label, date)


if __name__ == "__main__":
    main()
