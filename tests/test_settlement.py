from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent.aggregation import AggregatedSignal
from agent.calibration.analyze import analyze_decisions
from agent.calibration.daily_report import format_report
from agent.calibration.settlement_tracker import (
    METARCheck,
    SettlementCheckResult,
    _derive_expected_outcome,
    _parse_iem_csv_max_temp,
    _parse_market_date,
    check_settled_markets_singapore,
    fetch_metar_max_temp,
)
from agent.calibration.verified_stations import (
    get_verified_station,
    is_metar_verified,
)
from agent.memory import DecisionLog


# --- Fixtures ---

@pytest.fixture
def decision_db(tmp_path):
    db = DecisionLog(tmp_path / "test_decisions.db")
    yield db
    db.close()


def _make_signal(
    market_id: str = "sg-1",
    action: str = "buy_yes",
    confidence: float = 0.8,
) -> AggregatedSignal:
    return AggregatedSignal(
        market_id=market_id,
        action=action,
        confidence=confidence,
        suggested_size_usd=25.0,
        rationale="Test signal",
        agreement_ratio=1.0,
        n_samples=1,
    )


def _log_sg_decision(
    db: DecisionLog,
    market_id: str = "sg-1",
    action: str = "buy_yes",
    confidence: float = 0.8,
    threshold: float = 33.0,
    direction: str = "at_or_above",
    unit: str = "C",
    city: str = "Singapore",
    date: str = "June 27",
) -> None:
    """Log a decision for a Singapore temperature market."""
    dir_word = {"at_or_above": "higher", "above": "above", "below": "below"}.get(direction, "higher")
    question = (
        f"Will the high temperature in {city} be "
        f"{threshold}°{unit} or {dir_word} on {date}"
    )
    db.log_decision(
        weather_snapshot={"location": city, "temp_c": 33.0},
        market_snapshot={
            "id": market_id,
            "question": question,
            "description": "Temperature market",
            "outcome_yes_price": 0.30,
            "outcome_no_price": 0.70,
        },
        llm_raw_outputs=['{"action":"buy_yes"}'],
        final_signal=_make_signal(market_id=market_id, action=action, confidence=confidence),
        risk_decision="approved",
    )


# --- IEM CSV parsing ---

class TestIEMCSVParsing:
    def test_parse_normal_csv(self):
        csv_text = (
            "station,valid,tmpf\n"
            "WSSS,2026-06-27 00:00,82.40\n"
            "WSSS,2026-06-27 06:00,78.80\n"
            "WSSS,2026-06-27 14:00,91.40\n"
            "WSSS,2026-06-27 18:00,87.80\n"
        )
        max_c = _parse_iem_csv_max_temp(csv_text)
        assert max_c is not None
        assert abs(max_c - 33.0) < 0.1

    def test_parse_with_missing_values(self):
        csv_text = (
            "station,valid,tmpf\n"
            "WSSS,2026-06-27 00:00,M\n"
            "WSSS,2026-06-27 06:00,78.80\n"
            "WSSS,2026-06-27 14:00,M\n"
        )
        max_c = _parse_iem_csv_max_temp(csv_text)
        assert max_c is not None
        assert abs(max_c - 26.0) < 0.1

    def test_parse_empty_csv(self):
        assert _parse_iem_csv_max_temp("station,valid,tmpf\n") is None

    def test_parse_no_header(self):
        assert _parse_iem_csv_max_temp("") is None


# --- Date parsing ---

class TestDateParsing:
    def test_iso_date(self):
        assert _parse_market_date("2026-06-27") == "2026-06-27"

    def test_month_day(self):
        assert _parse_market_date("June 27") == "2026-06-27"

    def test_month_day_year(self):
        assert _parse_market_date("June 27, 2026") == "2026-06-27"

    def test_abbreviated_month(self):
        assert _parse_market_date("Jun 27") == "2026-06-27"

    def test_invalid(self):
        assert _parse_market_date("tomorrow") is None


# --- Outcome derivation ---

class TestDeriveExpectedOutcome:
    def test_above_yes(self):
        assert _derive_expected_outcome(34, 33, "above") == "YES"

    def test_above_no(self):
        assert _derive_expected_outcome(33, 33, "above") == "NO"

    def test_below_yes(self):
        assert _derive_expected_outcome(32, 33, "below") == "YES"

    def test_below_no(self):
        assert _derive_expected_outcome(33, 33, "below") == "NO"

    def test_at_or_above_yes(self):
        assert _derive_expected_outcome(33, 33, "at_or_above") == "YES"

    def test_at_or_above_no(self):
        assert _derive_expected_outcome(32, 33, "at_or_above") == "NO"

    def test_none_threshold(self):
        assert _derive_expected_outcome(33, None, "above") is None

    def test_unknown_direction(self):
        assert _derive_expected_outcome(33, 33, "unknown_dir") is None


# --- Settlement detail storage ---

class TestSettlementDetails:
    def test_insert_and_retrieve(self, decision_db):
        check = METARCheck(
            market_id="sg-1",
            market_question="",
            date="June 27",
            gamma_outcome="YES",
            metar_max_temp_c=33.2,
            metar_rounded_c=33,
            threshold_temp_c=33,
            direction="at_or_above",
            expected_outcome="YES",
            is_consistent=True,
        )
        decision_db.insert_settlement_detail(check)
        details = decision_db.get_settlement_details()
        assert len(details) == 1
        assert details[0]["market_id"] == "sg-1"
        assert details[0]["is_consistent"] == 1

    def test_discrepancy_query(self, decision_db):
        consistent = METARCheck(
            market_id="sg-1", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=33.2, metar_rounded_c=33,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="YES", is_consistent=True,
        )
        discrepant = METARCheck(
            market_id="sg-2", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=32.1, metar_rounded_c=32,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="NO", is_consistent=False,
            note="METAR disagrees",
        )
        decision_db.insert_settlement_detail(consistent)
        decision_db.insert_settlement_detail(discrepant)

        discs = decision_db.get_discrepancies()
        assert len(discs) == 1
        assert discs[0]["market_id"] == "sg-2"


# --- METAR fetch (mocked) ---

@pytest.mark.asyncio
async def test_fetch_metar_max_temp():
    csv_response = (
        "station,valid,tmpf\n"
        "WSSS,2026-06-27 08:00,84.20\n"
        "WSSS,2026-06-27 14:00,91.40\n"
        "WSSS,2026-06-27 20:00,86.00\n"
    )
    with respx.mock:
        respx.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py").mock(
            return_value=httpx.Response(200, text=csv_response)
        )
        async with httpx.AsyncClient() as client:
            max_c = await fetch_metar_max_temp(client, "WSSS", "2026-06-27")

    assert max_c is not None
    assert abs(max_c - 33.0) < 0.1


@pytest.mark.asyncio
async def test_fetch_metar_handles_error():
    with respx.mock:
        respx.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py").mock(
            return_value=httpx.Response(500)
        )
        async with httpx.AsyncClient() as client:
            max_c = await fetch_metar_max_temp(client, "WSSS", "2026-06-27")

    assert max_c is None


# --- Full settlement check (mocked, using decisions table) ---

@pytest.mark.asyncio
async def test_check_settled_singapore_consistent(decision_db):
    # threshold=32 with METAR=33°C → 33>32 → expected YES, Gamma YES → consistent
    _log_sg_decision(decision_db, market_id="sg-100", threshold=32.0)

    gamma_resp = {
        "id": "sg-100",
        "closed": True,
        "outcomePrices": '["1", "0"]',
    }
    metar_csv = (
        "station,valid,tmpf\n"
        "WSSS,2026-06-27 14:00,91.40\n"
    )

    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/markets/sg-100").mock(
            return_value=httpx.Response(200, json=gamma_resp)
        )
        respx.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py").mock(
            return_value=httpx.Response(200, text=metar_csv)
        )
        result = await check_settled_markets_singapore(decision_db)

    assert "sg-100" in result.settled
    assert result.settled["sg-100"] == "YES"
    assert len(result.metar_checks) == 1
    assert result.metar_checks[0].is_consistent is True
    assert len(result.discrepancies) == 0


@pytest.mark.asyncio
async def test_check_settled_singapore_discrepancy(decision_db):
    _log_sg_decision(
        decision_db, market_id="sg-200",
        threshold=34.0, direction="at_or_above",
    )

    gamma_resp = {
        "id": "sg-200",
        "closed": True,
        "outcomePrices": '["1", "0"]',
    }
    metar_csv = (
        "station,valid,tmpf\n"
        "WSSS,2026-06-27 14:00,91.40\n"
    )

    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/markets/sg-200").mock(
            return_value=httpx.Response(200, json=gamma_resp)
        )
        respx.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py").mock(
            return_value=httpx.Response(200, text=metar_csv)
        )
        result = await check_settled_markets_singapore(decision_db)

    assert "sg-200" in result.settled
    assert len(result.discrepancies) == 1
    check = result.discrepancies[0]
    assert check.gamma_outcome == "YES"
    assert check.expected_outcome == "NO"
    assert check.metar_rounded_c == 33


@pytest.mark.asyncio
async def test_check_settled_skips_non_singapore(decision_db):
    _log_sg_decision(
        decision_db, market_id="miami-1",
        city="Miami", threshold=95.0, unit="F", direction="above",
        date="July 1",
    )

    gamma_resp = {
        "id": "miami-1",
        "closed": True,
        "outcomePrices": '["1", "0"]',
    }

    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/markets/miami-1").mock(
            return_value=httpx.Response(200, json=gamma_resp)
        )
        result = await check_settled_markets_singapore(decision_db)

    assert "miami-1" in result.settled
    assert len(result.metar_checks) == 0


# --- Settlement via decisions table ---

class TestSettlementOnDecisions:
    def test_settle_marks_rows(self, decision_db):
        _log_sg_decision(decision_db, market_id="mkt-a")
        _log_sg_decision(decision_db, market_id="mkt-a")
        _log_sg_decision(decision_db, market_id="mkt-b")

        assert set(decision_db.get_unsettled_market_ids()) == {"mkt-a", "mkt-b"}

        count = decision_db.settle_market("mkt-a", "YES")
        assert count == 2

        unsettled = decision_db.get_unsettled_market_ids()
        assert unsettled == ["mkt-b"]

        settled = decision_db.get_settled_decisions()
        assert len(settled) == 2
        assert all(r["actual_outcome"] == "YES" for r in settled)

    def test_city_backfill(self, decision_db):
        _log_sg_decision(decision_db, market_id="sg-bf", city="Singapore")
        rows = decision_db.get_all_decisions()
        assert rows[0]["city"] == "Singapore"

    def test_empty_market_id_excluded(self, decision_db):
        """Old rows with empty market_id (pre-migration) should not appear."""
        conn = decision_db._get_conn()
        conn.execute(
            "INSERT INTO decisions "
            "(timestamp, eval_date, market_id, weather_snapshot_json, "
            "market_snapshot_json, llm_raw_outputs_json, raw_samples_json, "
            "final_signal_json, risk_decision) "
            "VALUES (?, ?, '', '{}', '{}', '[]', '[]', '{}', 'approved')",
            ("2026-06-01T00:00:00", "2026-06-01"),
        )
        conn.commit()
        _log_sg_decision(decision_db, market_id="real-mkt")

        unsettled = decision_db.get_unsettled_market_ids()
        assert unsettled == ["real-mkt"]


# --- Analyze using decisions ---

class TestAnalyzeDecisions:
    def test_action_distribution(self, decision_db):
        _log_sg_decision(decision_db, market_id="m1", action="buy_yes")
        _log_sg_decision(decision_db, market_id="m2", action="hold")
        _log_sg_decision(decision_db, market_id="m3", action="buy_no")

        report = analyze_decisions(decision_db)
        assert report.total_decisions == 3
        assert report.total_markets == 3
        assert report.action_stats.action_counts["buy_yes"] == 1
        assert report.action_stats.action_counts["hold"] == 1
        assert report.action_stats.action_counts["buy_no"] == 1

    def test_city_stats(self, decision_db):
        _log_sg_decision(decision_db, market_id="sg-a", city="Singapore")
        _log_sg_decision(decision_db, market_id="sg-b", city="Singapore")
        decision_db.settle_market("sg-a", "YES")

        report = analyze_decisions(decision_db)
        assert len(report.city_stats) == 1
        cs = report.city_stats[0]
        assert cs.city == "Singapore"
        assert cs.total_samples == 2
        assert cs.settled_samples == 1
        assert cs.correct == 1

    def test_discrepancy_stats_in_report(self, decision_db):
        check = METARCheck(
            market_id="sg-d", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=32.0, metar_rounded_c=32,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="NO", is_consistent=False,
            note="mismatch",
        )
        decision_db.insert_settlement_detail(check)

        report = analyze_decisions(decision_db)
        assert report.discrepancy_stats.total_checks == 1
        assert report.discrepancy_stats.inconsistent == 1

    def test_settled_accuracy(self, decision_db):
        _log_sg_decision(decision_db, market_id="m1", action="buy_yes", confidence=0.85)
        _log_sg_decision(decision_db, market_id="m2", action="buy_no", confidence=0.7)
        decision_db.settle_market("m1", "YES")
        decision_db.settle_market("m2", "YES")

        report = analyze_decisions(decision_db)
        assert report.settled_markets == 2
        assert report.action_stats.overall_accuracy is not None
        assert abs(report.action_stats.overall_accuracy - 0.5) < 0.01

    def test_verified_station_whitelist(self):
        assert is_metar_verified("Singapore") is True
        assert is_metar_verified("singapore") is True
        assert is_metar_verified("Miami") is False

        station = get_verified_station("Singapore")
        assert station is not None
        assert station.metar_station == "WSSS"
        assert station.timezone == "Asia/Singapore"

    def test_report_includes_city_section(self, decision_db):
        _log_sg_decision(decision_db, market_id="sg-r", city="Singapore")
        decision_db.settle_market("sg-r", "YES")

        check = METARCheck(
            market_id="sg-r", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=33.2, metar_rounded_c=33,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="YES", is_consistent=True,
        )
        decision_db.insert_settlement_detail(check)

        report = analyze_decisions(decision_db)
        text = format_report(report, "2026-06-28")

        assert "ACCURACY BY CITY" in text
        assert "Singapore" in text
        assert "METAR CROSS-VALIDATION" in text
        assert "Consistent: 1" in text
