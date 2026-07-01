"""Verify daily report works with production decisions data.

Usage (local — copy DB from VPS first):
    scp vps:~/weather-mcp/data/hermes_decisions.db ./data/
    python scripts/verify_report.py --db data/hermes_decisions.db

Or inside Docker container:
    docker compose exec hermes python scripts/verify_report.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.calibration.analyze import analyze_decisions
from agent.calibration.daily_report import format_report
from agent.memory import DecisionLog
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/hermes_decisions.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        print("Copy from VPS: scp vps:~/weather-mcp/data/hermes_decisions.db ./data/")
        sys.exit(1)

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
