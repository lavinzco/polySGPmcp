"""Manual daily sounding notes — human-entered atmospheric priors."""
from __future__ import annotations

import argparse
import enum
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pydantic import BaseModel, Field


SGT = timezone(timedelta(hours=8))


class InversionStrength(str, enum.Enum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class DailySoundingNote(BaseModel):
    target_date: str = Field(description="YYYY-MM-DD in SGT")
    inversion: InversionStrength
    surface_temp_c: float | None = None
    note: str = ""
    submitted_at: str = Field(default="", description="ISO timestamp")


def _sgt_today() -> str:
    return datetime.now(SGT).strftime("%Y-%m-%d")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sounding_notes (
            target_date TEXT PRIMARY KEY,
            inversion TEXT NOT NULL,
            surface_temp_c REAL,
            note TEXT NOT NULL DEFAULT '',
            submitted_at TEXT NOT NULL
        )
    """)
    conn.commit()


def submit_sounding_note(
    db_path: str | Path,
    *,
    inversion: InversionStrength,
    surface_temp_c: float | None = None,
    note: str = "",
    target_date: str | None = None,
) -> DailySoundingNote:
    target_date = target_date or _sgt_today()
    now = datetime.now(SGT).isoformat()

    conn = sqlite3.connect(str(db_path))
    _ensure_table(conn)
    conn.execute(
        """INSERT INTO sounding_notes (target_date, inversion, surface_temp_c, note, submitted_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(target_date) DO UPDATE SET
             inversion=excluded.inversion,
             surface_temp_c=excluded.surface_temp_c,
             note=excluded.note,
             submitted_at=excluded.submitted_at""",
        (target_date, inversion.value, surface_temp_c, note, now),
    )
    conn.commit()
    conn.close()

    return DailySoundingNote(
        target_date=target_date,
        inversion=inversion,
        surface_temp_c=surface_temp_c,
        note=note,
        submitted_at=now,
    )


def get_todays_sounding_note(db_path: str | Path) -> DailySoundingNote | None:
    today = _sgt_today()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM sounding_notes WHERE target_date = ?", (today,)
        ).fetchone()
        conn.close()
    except Exception:
        return None

    if row is None:
        return None

    return DailySoundingNote(
        target_date=row["target_date"],
        inversion=InversionStrength(row["inversion"]),
        surface_temp_c=row["surface_temp_c"],
        note=row["note"],
        submitted_at=row["submitted_at"],
    )


def main():
    parser = argparse.ArgumentParser(description="Manual sounding note input")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit today's sounding note")
    submit.add_argument("--inversion", required=True, choices=["strong", "weak", "none"])
    submit.add_argument("--surface-temp", type=float, default=None)
    submit.add_argument("--note", default="")
    submit.add_argument("--db", default="hermes_decisions.db")

    show = sub.add_parser("show", help="Show today's sounding note")
    show.add_argument("--db", default="hermes_decisions.db")

    args = parser.parse_args()

    if args.command == "submit":
        result = submit_sounding_note(
            args.db,
            inversion=InversionStrength(args.inversion),
            surface_temp_c=args.surface_temp,
            note=args.note,
        )
        print(f"Submitted for {result.target_date}:")
        print(f"  Inversion:    {result.inversion.value}")
        print(f"  Surface temp: {result.surface_temp_c or 'N/A'} °C")
        print(f"  Note:         {result.note or '(none)'}")
        print(f"  Submitted at: {result.submitted_at}")

    elif args.command == "show":
        note = get_todays_sounding_note(args.db)
        if note is None:
            print("No sounding note submitted for today.")
        else:
            print(f"Sounding note for {note.target_date}:")
            print(f"  Inversion:    {note.inversion.value}")
            print(f"  Surface temp: {note.surface_temp_c or 'N/A'} °C")
            print(f"  Note:         {note.note or '(none)'}")
            print(f"  Submitted at: {note.submitted_at}")


if __name__ == "__main__":
    main()
