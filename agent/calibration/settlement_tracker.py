from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from agent.calibration.db import CalibrationDB
from agent.calibration.verified_stations import get_verified_station, VerifiedStation
from common.config import settings

logger = logging.getLogger("hermes.calibration.settlement")

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


@dataclass
class METARCheck:
    market_id: str
    market_question: str
    date: str
    gamma_outcome: str
    metar_max_temp_c: float | None
    metar_rounded_c: int | None
    threshold_temp_c: int | None
    direction: str | None
    expected_outcome: str | None
    is_consistent: bool | None
    note: str = ""


@dataclass
class SettlementCheckResult:
    settled: dict[str, str] = field(default_factory=dict)
    metar_checks: list[METARCheck] = field(default_factory=list)

    @property
    def discrepancies(self) -> list[METARCheck]:
        return [c for c in self.metar_checks if c.is_consistent is False]


async def check_settled_markets(
    db: CalibrationDB,
    *,
    gamma_base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, str]:
    """Check unsettled markets against Gamma API, backfill outcomes.
    Returns dict of {market_id: outcome} for newly settled markets.
    """
    base_url = (gamma_base_url or settings.gamma_api_base_url).rstrip("/")
    unsettled_ids = db.get_unsettled_market_ids()
    if not unsettled_ids:
        logger.info("No unsettled markets to check")
        return {}

    logger.info(f"Checking {len(unsettled_ids)} unsettled markets")
    settled: dict[str, str] = {}

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30, headers={"Accept-Encoding": "gzip, deflate"})

    try:
        for market_id in unsettled_ids:
            outcome = await _check_single_market(client, base_url, market_id)
            if outcome is not None:
                count = db.settle_market(market_id, outcome)
                settled[market_id] = outcome
                logger.info(f"  Settled {market_id} → {outcome} ({count} samples updated)")
    finally:
        if own_client:
            await client.aclose()

    logger.info(f"Settled {len(settled)}/{len(unsettled_ids)} markets")
    return settled


async def check_settled_markets_singapore(
    db: CalibrationDB,
    *,
    gamma_base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> SettlementCheckResult:
    """Enhanced settlement check with METAR cross-validation.

    All unsettled markets are checked against the Gamma API for settlement.
    METAR cross-validation only runs for cities in the verified_stations
    whitelist (currently: Singapore/WSSS). Other cities get pure Gamma
    settlement with no discrepancy detection.
    """
    base_url = (gamma_base_url or settings.gamma_api_base_url).rstrip("/")
    unsettled_ids = db.get_unsettled_market_ids()
    result = SettlementCheckResult()

    if not unsettled_ids:
        logger.info("No unsettled markets to check")
        return result

    verified_samples = _get_verified_sample_info(db, unsettled_ids)
    logger.info(
        f"Checking {len(unsettled_ids)} unsettled markets "
        f"({len(verified_samples)} METAR-verified)"
    )

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30, headers={"Accept-Encoding": "gzip, deflate"})

    metar_cache: dict[str, float | None] = {}

    try:
        for market_id in unsettled_ids:
            outcome = await _check_single_market(client, base_url, market_id)
            if outcome is None:
                continue

            count = db.settle_market(market_id, outcome)
            result.settled[market_id] = outcome
            logger.info(f"  Settled {market_id} → {outcome} ({count} samples updated)")

            if market_id in verified_samples:
                info = verified_samples[market_id]
                check = await _cross_validate_metar(
                    client, market_id, outcome, info, metar_cache,
                )
                result.metar_checks.append(check)
                db.insert_settlement_detail(check)

                if check.is_consistent is False:
                    logger.warning(
                        f"  DISCREPANCY {market_id}: Gamma={outcome}, "
                        f"METAR max={check.metar_max_temp_c}°C "
                        f"(rounded {check.metar_rounded_c}°C), "
                        f"expected {check.expected_outcome} — {check.note}"
                    )
    finally:
        if own_client:
            await client.aclose()

    n_disc = len(result.discrepancies)
    logger.info(
        f"Settlement complete: {len(result.settled)} settled, "
        f"{len(result.metar_checks)} METAR-checked, "
        f"{n_disc} discrepancies"
    )
    return result


def _get_verified_sample_info(
    db: CalibrationDB, market_ids: list[str],
) -> dict[str, dict]:
    """Get threshold/date info for markets in METAR-verified cities."""
    all_samples = db.get_all_samples()
    info: dict[str, dict] = {}
    for s in all_samples:
        mid = s["market_id"]
        if mid not in market_ids:
            continue
        city = s.get("city", "")
        station = get_verified_station(city)
        if station is None:
            continue
        info[mid] = {
            "question": s.get("market_id", ""),
            "date": s.get("date", ""),
            "threshold_temp": s.get("threshold_temp"),
            "threshold_unit": s.get("threshold_unit", "C"),
            "direction": s.get("direction", ""),
            "city": city,
            "station": station,
        }
    return info


async def _cross_validate_metar(
    client: httpx.AsyncClient,
    market_id: str,
    gamma_outcome: str,
    sample_info: dict,
    metar_cache: dict[str, float | None],
) -> METARCheck:
    """Cross-validate a market settlement against METAR data for a verified station."""
    date_str = sample_info.get("date", "")
    threshold = sample_info.get("threshold_temp")
    direction = sample_info.get("direction", "")
    unit = sample_info.get("threshold_unit", "C")
    station: VerifiedStation = sample_info["station"]

    threshold_c = _to_celsius(threshold, unit) if threshold is not None else None
    threshold_c_int = round(threshold_c) if threshold_c is not None else None

    iso_date = _parse_market_date(date_str)
    if iso_date is None:
        return METARCheck(
            market_id=market_id,
            market_question=sample_info.get("question", ""),
            date=date_str,
            gamma_outcome=gamma_outcome,
            metar_max_temp_c=None,
            metar_rounded_c=None,
            threshold_temp_c=threshold_c_int,
            direction=direction or None,
            expected_outcome=None,
            is_consistent=None,
            note=f"Could not parse market date: {date_str!r}",
        )

    cache_key = f"{station.metar_station}:{iso_date}"
    if cache_key not in metar_cache:
        metar_cache[cache_key] = await fetch_metar_max_temp(
            client, station.metar_station, iso_date,
            tz=station.timezone,
        )
    metar_max = metar_cache[cache_key]

    if metar_max is None:
        return METARCheck(
            market_id=market_id,
            market_question=sample_info.get("question", ""),
            date=date_str,
            gamma_outcome=gamma_outcome,
            metar_max_temp_c=None,
            metar_rounded_c=None,
            threshold_temp_c=threshold_c_int,
            direction=direction or None,
            expected_outcome=None,
            is_consistent=None,
            note="METAR data unavailable for this date",
        )

    metar_rounded = round(metar_max)

    expected = _derive_expected_outcome(metar_rounded, threshold_c_int, direction)

    is_consistent = (expected == gamma_outcome) if expected else None
    note = ""
    if is_consistent is False:
        note = (
            f"METAR says {metar_max:.1f}°C (→{metar_rounded}°C), "
            f"threshold {threshold_c_int}°C {direction}, "
            f"expected {expected} but Gamma settled {gamma_outcome}"
        )

    return METARCheck(
        market_id=market_id,
        market_question=sample_info.get("question", ""),
        date=date_str,
        gamma_outcome=gamma_outcome,
        metar_max_temp_c=round(metar_max, 1),
        metar_rounded_c=metar_rounded,
        threshold_temp_c=threshold_c_int,
        direction=direction or None,
        expected_outcome=expected,
        is_consistent=is_consistent,
    )


async def fetch_metar_max_temp(
    client: httpx.AsyncClient,
    station: str,
    date_iso: str,
    *,
    tz: str | None = None,
) -> float | None:
    """Fetch METAR observations from IEM ASOS and return max temperature in °C.

    Uses local timezone to match Wunderground's day boundary definition.
    If tz is not provided, looks up the verified station's timezone.
    """
    parts = date_iso.split("-")
    if len(parts) != 3:
        return None

    if tz is None:
        tz = "Etc/UTC"

    try:
        resp = await client.get(
            IEM_ASOS_URL,
            params={
                "station": station,
                "data": "tmpf",
                "year1": parts[0], "month1": parts[1].lstrip("0") or "1",
                "day1": parts[2].lstrip("0") or "1",
                "year2": parts[0], "month2": parts[1].lstrip("0") or "1",
                "day2": parts[2].lstrip("0") or "1",
                "tz": tz,
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": "3",
            },
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(f"IEM ASOS request failed for {station}/{date_iso}: {exc}")
        return None

    return _parse_iem_csv_max_temp(resp.text)


def _parse_iem_csv_max_temp(csv_text: str) -> float | None:
    """Parse IEM ASOS CSV response and return max temperature in °C."""
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if header is None:
        return None

    try:
        tmpf_idx = header.index("tmpf")
    except ValueError:
        return None

    max_c: float | None = None
    for row in reader:
        if len(row) <= tmpf_idx:
            continue
        raw = row[tmpf_idx].strip()
        if raw in ("M", "", "T"):
            continue
        try:
            tmpf = float(raw)
            tmpc = (tmpf - 32) * 5 / 9
            if max_c is None or tmpc > max_c:
                max_c = tmpc
        except ValueError:
            continue

    return max_c


async def _check_single_market(
    client: httpx.AsyncClient,
    base_url: str,
    market_id: str,
) -> str | None:
    """Returns 'YES' or 'NO' if market is closed+resolved, None otherwise."""
    try:
        resp = await client.get(f"{base_url}/markets/{market_id}")
        if resp.status_code == 404:
            logger.warning(f"  Market {market_id} not found (404)")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"  Failed to fetch market {market_id}: {exc}")
        return None

    if not data.get("closed", False):
        return None

    outcome = _extract_outcome(data)
    return outcome


def _extract_outcome(data: dict) -> str | None:
    """Parse Gamma API market response to determine YES/NO outcome."""
    try:
        prices_str = data.get("outcomePrices", "[]")
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        if isinstance(prices, list) and len(prices) >= 2:
            yes_price = float(prices[0])
            no_price = float(prices[1])
            if yes_price >= 0.95:
                return "YES"
            if no_price >= 0.95:
                return "NO"
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        pass

    resolved_to = data.get("resolvedTo")
    if resolved_to is not None:
        return "YES" if str(resolved_to).lower() in ("yes", "1", "true") else "NO"

    return None


def _to_celsius(temp: float, unit: str) -> float:
    if unit.upper() in ("F", "°F"):
        return (temp - 32) * 5 / 9
    return temp


_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_market_date(date_text: str) -> str | None:
    """Parse human-readable date like 'June 27' or '2026-06-27' to ISO format."""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_text):
        return date_text

    m = re.match(r"(\w+)\s+(\d{1,2})(?:,?\s*(\d{4}))?", date_text.strip())
    if m:
        month_str = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else 2026
        month = _MONTH_MAP.get(month_str)
        if month:
            return f"{year}-{month:02d}-{day:02d}"

    return None


def _derive_expected_outcome(
    metar_rounded_c: int,
    threshold_c: int | None,
    direction: str,
) -> str | None:
    """Given METAR max temp and market threshold, what should the outcome be?

    Market types:
    - "above": "Will temp exceed X°C?" → YES if actual > threshold
    - "below": "Will temp be X°C or below?" → YES if actual <= threshold
    - "between": "Will temp be between X-Y°C?" → handled by exact match
    - "exact" or "at": "Will temp be exactly X°C?" → YES if actual == threshold
    """
    if threshold_c is None:
        return None

    direction = direction.lower().strip()

    if direction in ("above", "over", "exceed", "higher"):
        return "YES" if metar_rounded_c > threshold_c else "NO"
    if direction in ("below", "under", "lower"):
        return "YES" if metar_rounded_c < threshold_c else "NO"
    if direction in ("at", "exact", "equal", "at_or_above"):
        return "YES" if metar_rounded_c >= threshold_c else "NO"
    if direction in ("at_or_below",):
        return "YES" if metar_rounded_c <= threshold_c else "NO"

    return None
