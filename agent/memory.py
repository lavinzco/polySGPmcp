from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.aggregation import AggregatedSignal


class DecisionLog:
    def __init__(self, db_path: str | Path = "hermes_decisions.db"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                eval_date TEXT NOT NULL DEFAULT '',
                market_id TEXT NOT NULL DEFAULT '',
                weather_snapshot_json TEXT NOT NULL,
                market_snapshot_json TEXT NOT NULL,
                llm_raw_outputs_json TEXT NOT NULL,
                raw_samples_json TEXT NOT NULL,
                final_signal_json TEXT NOT NULL,
                risk_decision TEXT NOT NULL,
                agreement_ratio REAL NOT NULL DEFAULT 1.0,
                n_samples INTEGER NOT NULL DEFAULT 1,
                dry_run INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_market_date
            ON decisions (market_id, eval_date)
        """)
        conn.commit()
        self._migrate_columns(conn)

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(decisions)")
        columns = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("eval_date", "TEXT NOT NULL DEFAULT ''"),
            ("market_id", "TEXT NOT NULL DEFAULT ''"),
            ("dry_run", "INTEGER NOT NULL DEFAULT 1"),
        ]
        for col_name, col_def in migrations:
            if col_name not in columns:
                conn.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_def}")
        conn.commit()

    def log_decision(
        self,
        *,
        weather_snapshot: dict,
        market_snapshot: dict,
        llm_raw_outputs: list[str],
        final_signal: AggregatedSignal,
        risk_decision: str,
        dry_run: bool = True,
    ) -> None:
        raw_samples_data = [s.model_dump() for s in final_signal.raw_samples]
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO decisions
               (timestamp, eval_date, market_id,
                weather_snapshot_json, market_snapshot_json,
                llm_raw_outputs_json, raw_samples_json,
                final_signal_json, risk_decision,
                agreement_ratio, n_samples, dry_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                final_signal.market_id,
                json.dumps(weather_snapshot),
                json.dumps(market_snapshot),
                json.dumps(llm_raw_outputs),
                json.dumps(raw_samples_data),
                final_signal.model_dump_json(),
                risk_decision,
                final_signal.agreement_ratio,
                final_signal.n_samples,
                1 if dry_run else 0,
            ),
        )
        conn.commit()

    def was_evaluated_today(self, market_id: str, date_str: str | None = None) -> bool:
        eval_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM decisions WHERE market_id = ? AND eval_date = ? LIMIT 1",
            (market_id, eval_date),
        ).fetchone()
        return row is not None

    def was_evaluated_in_window(self, market_id: str, window_minutes: int) -> bool:
        """Return True if market was evaluated within the last `window_minutes`."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        row = conn.execute(
            "SELECT 1 FROM decisions WHERE market_id = ? AND timestamp >= ? LIMIT 1",
            (market_id, cutoff),
        ).fetchone()
        return row is not None

    def get_recent_decisions(self, n: int = 10) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
