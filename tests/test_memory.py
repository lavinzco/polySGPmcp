from __future__ import annotations

import json
from unittest.mock import patch
from datetime import datetime, timezone

import pytest

from agent.aggregation import AggregatedSignal
from agent.memory import DecisionLog
from agent.models import TradeSignal


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_decisions.db"


@pytest.fixture
def decision_log(db_path):
    log = DecisionLog(db_path)
    yield log
    log.close()


def _make_signal(action: str = "buy_yes", market_id: str = "mkt-001") -> AggregatedSignal:
    raw = [
        TradeSignal(
            market_id=market_id, action=action, confidence=0.85,
            suggested_size_usd=25.0, rationale="sample", weather_factors=["wind"],
        )
    ]
    return AggregatedSignal(
        market_id=market_id,
        action=action,
        confidence=0.85,
        suggested_size_usd=25.0,
        rationale="Test signal",
        weather_factors=["wind"],
        agreement_ratio=1.0,
        raw_samples=raw,
        n_samples=1,
    )


class TestDecisionLog:
    def test_log_and_retrieve(self, decision_log):
        decision_log.log_decision(
            weather_snapshot={"location": "Miami", "temp_c": 33},
            market_snapshot={"id": "mkt-001", "question": "Hurricane?"},
            llm_raw_outputs=['{"action":"buy_yes"}'],
            final_signal=_make_signal(),
            risk_decision="approved",
        )

        rows = decision_log.get_recent_decisions(5)
        assert len(rows) == 1
        assert rows[0]["risk_decision"] == "approved"
        assert "Miami" in rows[0]["weather_snapshot_json"]
        assert "buy_yes" in rows[0]["final_signal_json"]
        assert rows[0]["dry_run"] == 1
        assert rows[0]["market_id"] == "mkt-001"
        assert rows[0]["eval_date"] != ""

    def test_multiple_entries_ordered(self, decision_log):
        for i in range(5):
            decision_log.log_decision(
                weather_snapshot={"i": i},
                market_snapshot={"i": i},
                llm_raw_outputs=[f"raw-{i}"],
                final_signal=_make_signal(),
                risk_decision=f"decision-{i}",
            )

        rows = decision_log.get_recent_decisions(3)
        assert len(rows) == 3
        assert rows[0]["risk_decision"] == "decision-4"
        assert rows[2]["risk_decision"] == "decision-2"

    def test_stores_raw_samples(self, decision_log):
        signal = _make_signal()
        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=['{"action":"buy_yes"}', '{"action":"hold"}'],
            final_signal=signal,
            risk_decision="approved",
        )

        rows = decision_log.get_recent_decisions(1)
        raw_outputs = json.loads(rows[0]["llm_raw_outputs_json"])
        assert len(raw_outputs) == 2
        raw_samples = json.loads(rows[0]["raw_samples_json"])
        assert len(raw_samples) == 1
        assert rows[0]["agreement_ratio"] == 1.0
        assert rows[0]["n_samples"] == 1

    def test_stores_timestamp(self, decision_log):
        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=[],
            final_signal=_make_signal(),
            risk_decision="blocked",
        )

        rows = decision_log.get_recent_decisions(1)
        assert "T" in rows[0]["timestamp"]

    def test_dry_run_field(self, decision_log):
        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=[],
            final_signal=_make_signal(),
            risk_decision="approved",
            dry_run=True,
        )
        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=[],
            final_signal=_make_signal(market_id="mkt-002"),
            risk_decision="approved",
            dry_run=False,
        )

        rows = decision_log.get_recent_decisions(2)
        assert rows[0]["dry_run"] == 0  # most recent = mkt-002
        assert rows[1]["dry_run"] == 1

    def test_was_evaluated_today(self, decision_log):
        assert not decision_log.was_evaluated_today("mkt-001")

        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=[],
            final_signal=_make_signal(),
            risk_decision="approved",
        )

        assert decision_log.was_evaluated_today("mkt-001")
        assert not decision_log.was_evaluated_today("mkt-999")

    def test_was_evaluated_today_cross_day(self, decision_log):
        """Records from a different date should not count."""
        decision_log.log_decision(
            weather_snapshot={},
            market_snapshot={},
            llm_raw_outputs=[],
            final_signal=_make_signal(),
            risk_decision="approved",
        )

        assert decision_log.was_evaluated_today("mkt-001")
        assert not decision_log.was_evaluated_today("mkt-001", date_str="2025-01-01")
