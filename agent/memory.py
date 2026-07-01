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
                city TEXT NOT NULL DEFAULT '',
                weather_snapshot_json TEXT NOT NULL,
                market_snapshot_json TEXT NOT NULL,
                llm_raw_outputs_json TEXT NOT NULL,
                raw_samples_json TEXT NOT NULL,
                final_signal_json TEXT NOT NULL,
                risk_decision TEXT NOT NULL,
                agreement_ratio REAL NOT NULL DEFAULT 1.0,
                n_samples INTEGER NOT NULL DEFAULT 1,
                dry_run INTEGER NOT NULL DEFAULT 1,
                settled INTEGER NOT NULL DEFAULT 0,
                actual_outcome TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_market_date
            ON decisions (market_id, eval_date)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settlement_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                date TEXT NOT NULL,
                gamma_outcome TEXT NOT NULL,
                metar_max_temp_c REAL,
                metar_rounded_c INTEGER,
                threshold_temp_c INTEGER,
                direction TEXT,
                expected_outcome TEXT,
                is_consistent INTEGER,
                note TEXT NOT NULL DEFAULT ''
            )
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
            ("city", "TEXT NOT NULL DEFAULT ''"),
            ("settled", "INTEGER NOT NULL DEFAULT 0"),
            ("actual_outcome", "TEXT"),
        ]
        for col_name, col_def in migrations:
            if col_name not in columns:
                conn.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_def}")
        conn.commit()
        self._backfill_city(conn)

    def _backfill_city(self, conn: sqlite3.Connection) -> None:
        """Backfill city column from weather_snapshot_json for old rows."""
        rows = conn.execute(
            "SELECT id, weather_snapshot_json FROM decisions WHERE city = ''"
        ).fetchall()
        for row in rows:
            try:
                weather = json.loads(row["weather_snapshot_json"])
                city = weather.get("location", "")
            except (json.JSONDecodeError, TypeError):
                city = ""
            if city:
                conn.execute(
                    "UPDATE decisions SET city = ? WHERE id = ?",
                    (city, row["id"]),
                )
        if rows:
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
        city = weather_snapshot.get("location", "")
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO decisions
               (timestamp, eval_date, market_id, city,
                weather_snapshot_json, market_snapshot_json,
                llm_raw_outputs_json, raw_samples_json,
                final_signal_json, risk_decision,
                agreement_ratio, n_samples, dry_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                final_signal.market_id,
                city,
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

    def get_all_decisions(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_unsettled_market_ids(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT market_id FROM decisions WHERE settled = 0"
        ).fetchall()
        return [row["market_id"] for row in rows]

    def settle_market(self, market_id: str, outcome: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE decisions SET settled = 1, actual_outcome = ? "
            "WHERE market_id = ? AND settled = 0",
            (outcome, market_id),
        )
        conn.commit()
        return cur.rowcount

    def get_settled_decisions(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM decisions WHERE settled = 1 ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_settlement_detail(self, check) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO settlement_details
               (timestamp, market_id, date, gamma_outcome,
                metar_max_temp_c, metar_rounded_c, threshold_temp_c,
                direction, expected_outcome, is_consistent, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                check.market_id,
                check.date,
                check.gamma_outcome,
                check.metar_max_temp_c,
                check.metar_rounded_c,
                check.threshold_temp_c,
                check.direction,
                check.expected_outcome,
                int(check.is_consistent) if check.is_consistent is not None else None,
                check.note,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    def get_settlement_details(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM settlement_details ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_discrepancies(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM settlement_details WHERE is_consistent = 0 ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
