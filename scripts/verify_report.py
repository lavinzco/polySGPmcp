"""Verify daily report works with production decisions data.

Usage (local):
    python scripts/verify_report.py --db data/hermes_decisions.db

Inside Docker container:
    docker compose run --rm hermes python scripts/verify_report.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.calibration.analyze import analyze_decisions
from agent.calibration.daily_report import format_report
from agent.memory import DecisionLog
from datetime import datetime, timezone


def _data_path(filename: str) -> str:
    data_dir = os.environ.get("DATA_DIR", ".")
    return str(Path(data_dir) / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", default=None,
        help="Path to decisions DB (default: $DATA_DIR/hermes_decisions.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else Path(_data_path("hermes_decisions.db"))
    print(f"DB path: {db_path}")
    print(f"DATA_DIR={os.environ.get('DATA_DIR', '(not set)')}")

    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    print(f"DB size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")

    db = DecisionLog(db_path)
    try:
        all_rows = db.get_all_decisions()
        print(f"Total rows in decisions table: {len(all_rows)}")
        if not all_rows:
            print("No data — nothing to verify.")
            return

        cities = set()
        for r in all_rows:
            c = r.get("city", "")
            if c:
                cities.add(c)
        print(f"Cities found: {sorted(cities)}")
        print(f"Unsettled market IDs: {len(db.get_unsettled_market_ids())}")
        print(f"Settlement details: {len(db.get_settlement_details())}")
        print()

        report = analyze_decisions(db)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        text = format_report(report, date_str)
        print(text)
    finally:
        db.close()


if __name__ == "__main__":
    main()
