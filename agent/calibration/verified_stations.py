"""Whitelist of cities where METAR cross-validation is enabled.

METAR cross-validation compares the Gamma API settlement outcome against
an independently fetched METAR observation from IEM ASOS. This is only
valid when the METAR station and Wunderground's historical page reference
the same underlying data source.

BEFORE ADDING A CITY: verify that:
  1. The city's Polymarket description names a specific Wunderground station
  2. That Wunderground station code matches a real METAR/ASOS station
  3. IEM ASOS (mesonet.agron.iastate.edu) has data for that station
  4. Historical spot-checks confirm METAR max temp == Wunderground max temp
     for at least 5 recent days

Without this verification, a mismatch between METAR and Wunderground
would produce false discrepancy alerts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerifiedStation:
    city_pattern: str
    metar_station: str
    timezone: str
    note: str


VERIFIED_STATIONS: list[VerifiedStation] = [
    VerifiedStation(
        city_pattern="singapore",
        metar_station="WSSS",
        timezone="Asia/Singapore",
        note="Changi Airport — confirmed same source as Wunderground/WSSS",
    ),
]

_LOOKUP: dict[str, VerifiedStation] = {
    v.city_pattern: v for v in VERIFIED_STATIONS
}


def get_verified_station(city: str) -> VerifiedStation | None:
    """Return station info if this city has verified METAR=Wunderground parity."""
    city_lower = city.lower().strip()
    for pattern, station in _LOOKUP.items():
        if pattern in city_lower:
            return station
    return None


def is_metar_verified(city: str) -> bool:
    return get_verified_station(city) is not None
