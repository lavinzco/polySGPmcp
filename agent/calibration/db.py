"""DEPRECATED: CalibrationDB and calibration_samples table are superseded by
agent.memory.DecisionLog (decisions table + settlement_details table).

This module is retained only for the standalone calibration collector
(agent/calibration/collector.py). The production scheduler path uses
DecisionLog exclusively.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.calibration.models import CalibrationSample


class CalibrationDB:
    def __init__(self, db_path: str | Path = "hermes_calibration.db"):
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
            CREATE TABLE IF NOT EXISTS calibration_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                city TEXT NOT NULL,
                date TEXT NOT NULL,
                threshold_temp REAL NOT NULL,
                threshold_unit TEXT NOT NULL,
                direction TEXT NOT NULL,
                market_yes_price REAL NOT NULL,
                llm_action TEXT NOT NULL,
                llm_confidence REAL NOT NULL,
                llm_rationale TEXT NOT NULL,
                llm_raw_output TEXT NOT NULL,
                weather_snapshot_json TEXT NOT NULL,
                settled INTEGER NOT NULL DEFAULT 0,
                actual_outcome TEXT
            )
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

    def insert_sample(self, sample: CalibrationSample) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO calibration_samples
               (timestamp, market_id, provider_name, model_name, city, date,
                threshold_temp, threshold_unit, direction, market_yes_price,
                llm_action, llm_confidence, llm_rationale, llm_raw_output,
                weather_snapshot_json, settled, actual_outcome)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                sample.market_id,
                sample.provider_name,
                sample.model_name,
                sample.city,
                sample.date,
                sample.threshold_temp,
                sample.threshold_unit,
                sample.direction,
                sample.market_yes_price,
                sample.llm_action,
                sample.llm_confidence,
                sample.llm_rationale,
                sample.llm_raw_output,
                sample.weather_snapshot_json,
                int(sample.settled),
                sample.actual_outcome,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    def get_unsettled_market_ids(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT market_id FROM calibration_samples WHERE settled = 0"
        ).fetchall()
        return [row["market_id"] for row in rows]

    def settle_market(self, market_id: str, outcome: str) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """UPDATE calibration_samples
               SET settled = 1, actual_outcome = ?
               WHERE market_id = ? AND settled = 0""",
            (outcome, market_id),
        )
        conn.commit()
        return cur.rowcount

    def get_all_samples(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM calibration_samples ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_settled_samples(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM calibration_samples WHERE settled = 1 ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_samples_by_provider(self, provider_name: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM calibration_samples WHERE provider_name = ? ORDER BY id",
            (provider_name,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_provider_names(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT provider_name FROM calibration_samples ORDER BY provider_name"
        ).fetchall()
        return [row["provider_name"] for row in rows]

    def insert_settlement_detail(self, check) -> int:
        """Store a METAR cross-validation result. Accepts a METARCheck dataclass."""
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
