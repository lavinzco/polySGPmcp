from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent.calibration.analyze import analyze_calibration
from agent.calibration.daily_report import format_report
from agent.calibration.db import CalibrationDB
from agent.calibration.models import CalibrationSample
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


# --- Fixtures ---

@pytest.fixture
def cal_db(tmp_path):
    db = CalibrationDB(tmp_path / "test_cal.db")
    yield db
    db.close()


def _make_sg_sample(
    market_id: str = "sg-1",
    action: str = "buy_yes",
    confidence: float = 0.8,
    settled: bool = False,
    outcome: str | None = None,
    threshold: float = 33.0,
    direction: str = "at_or_above",
) -> CalibrationSample:
    return CalibrationSample(
        market_id=market_id,
        provider_name="deepseek-chat",
        model_name="deepseek-chat",
        city="Singapore",
        date="June 27",
        threshold_temp=threshold,
        threshold_unit="C",
        direction=direction,
        market_yes_price=0.30,
        llm_action=action,
        llm_confidence=confidence,
        llm_rationale="test",
        llm_raw_output='{"action":"buy_yes"}',
        weather_snapshot_json="{}",
        settled=settled,
        actual_outcome=outcome,
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
    def test_insert_and_retrieve(self, cal_db):
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
        cal_db.insert_settlement_detail(check)
        details = cal_db.get_settlement_details()
        assert len(details) == 1
        assert details[0]["market_id"] == "sg-1"
        assert details[0]["is_consistent"] == 1

    def test_discrepancy_query(self, cal_db):
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
        cal_db.insert_settlement_detail(consistent)
        cal_db.insert_settlement_detail(discrepant)

        discs = cal_db.get_discrepancies()
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


# --- Full settlement check (mocked) ---

@pytest.mark.asyncio
async def test_check_settled_singapore_consistent(cal_db):
    cal_db.insert_sample(_make_sg_sample(market_id="sg-100"))

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
        result = await check_settled_markets_singapore(cal_db)

    assert "sg-100" in result.settled
    assert result.settled["sg-100"] == "YES"
    assert len(result.metar_checks) == 1
    assert result.metar_checks[0].is_consistent is True
    assert len(result.discrepancies) == 0


@pytest.mark.asyncio
async def test_check_settled_singapore_discrepancy(cal_db):
    cal_db.insert_sample(_make_sg_sample(
        market_id="sg-200", threshold=34.0, direction="at_or_above",
    ))

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
        result = await check_settled_markets_singapore(cal_db)

    assert "sg-200" in result.settled
    assert len(result.discrepancies) == 1
    check = result.discrepancies[0]
    assert check.gamma_outcome == "YES"
    assert check.expected_outcome == "NO"
    assert check.metar_rounded_c == 33


@pytest.mark.asyncio
async def test_check_settled_skips_non_singapore(cal_db):
    sample = CalibrationSample(
        market_id="miami-1",
        provider_name="deepseek-chat",
        model_name="deepseek-chat",
        city="Miami",
        date="July 1",
        threshold_temp=95.0,
        threshold_unit="F",
        direction="above",
        market_yes_price=0.45,
        llm_action="buy_yes",
        llm_confidence=0.8,
        llm_rationale="test",
        llm_raw_output="{}",
        weather_snapshot_json="{}",
    )
    cal_db.insert_sample(sample)

    gamma_resp = {
        "id": "miami-1",
        "closed": True,
        "outcomePrices": '["1", "0"]',
    }

    with respx.mock:
        respx.get("https://gamma-api.polymarket.com/markets/miami-1").mock(
            return_value=httpx.Response(200, json=gamma_resp)
        )
        result = await check_settled_markets_singapore(cal_db)

    assert "miami-1" in result.settled
    assert len(result.metar_checks) == 0


# --- Enhanced analyze ---

class TestEnhancedAnalyze:
    def test_city_stats(self, cal_db):
        cal_db.insert_sample(_make_sg_sample(
            market_id="sg-a", settled=True, outcome="YES",
            action="buy_yes", confidence=0.8,
        ))
        cal_db.insert_sample(_make_sg_sample(
            market_id="sg-b", settled=True, outcome="NO",
            action="buy_yes", confidence=0.7,
        ))

        report = analyze_calibration(cal_db)
        assert len(report.city_stats) == 1
        cs = report.city_stats[0]
        assert cs.city == "Singapore"
        assert cs.settled_samples == 2
        assert cs.correct == 1

    def test_discrepancy_stats_in_report(self, cal_db):
        check = METARCheck(
            market_id="sg-d", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=32.0, metar_rounded_c=32,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="NO", is_consistent=False,
            note="mismatch",
        )
        cal_db.insert_settlement_detail(check)

        report = analyze_calibration(cal_db)
        assert report.discrepancy_stats.total_checks == 1
        assert report.discrepancy_stats.inconsistent == 1

    def test_verified_station_whitelist(self):
        assert is_metar_verified("Singapore") is True
        assert is_metar_verified("singapore") is True
        assert is_metar_verified("Miami") is False
        assert is_metar_verified("New York") is False

        station = get_verified_station("Singapore")
        assert station is not None
        assert station.metar_station == "WSSS"
        assert station.timezone == "Asia/Singapore"

    def test_report_includes_city_section(self, cal_db):
        cal_db.insert_sample(_make_sg_sample(
            market_id="sg-r", settled=True, outcome="YES",
            action="buy_yes",
        ))
        check = METARCheck(
            market_id="sg-r", market_question="", date="June 27",
            gamma_outcome="YES", metar_max_temp_c=33.2, metar_rounded_c=33,
            threshold_temp_c=33, direction="at_or_above",
            expected_outcome="YES", is_consistent=True,
        )
        cal_db.insert_settlement_detail(check)

        report = analyze_calibration(cal_db)
        text = format_report(report, "2026-06-28")

        assert "ACCURACY BY CITY" in text
        assert "Singapore" in text
        assert "METAR CROSS-VALIDATION" in text
        assert "Consistent: 1" in text
