from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent.memory import DecisionLog

CONFIDENCE_BINS = [
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, 0.6, "0.4-0.6"),
    (0.6, 0.8, "0.6-0.8"),
    (0.8, 1.01, "0.8-1.0"),
]


@dataclass
class BinStats:
    label: str
    total: int = 0
    correct: int = 0
    avg_confidence: float = 0.0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total > 0 else None


@dataclass
class ActionStats:
    total_decisions: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    bins: list[BinStats] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float | None:
        settled_actionable = sum(b.total for b in self.bins)
        if settled_actionable == 0:
            return None
        correct = sum(b.correct for b in self.bins)
        return correct / settled_actionable


@dataclass
class CityStats:
    city: str
    total_samples: int = 0
    settled_samples: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.settled_samples if self.settled_samples > 0 else None


@dataclass
class DiscrepancyStats:
    total_checks: int = 0
    consistent: int = 0
    inconsistent: int = 0
    unknown: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def consistency_rate(self) -> float | None:
        checked = self.consistent + self.inconsistent
        return self.consistent / checked if checked > 0 else None


@dataclass
class CalibrationReport:
    total_decisions: int = 0
    total_markets: int = 0
    settled_markets: int = 0
    unsettled_markets: int = 0
    action_stats: ActionStats = field(default_factory=ActionStats)
    city_stats: list[CityStats] = field(default_factory=list)
    discrepancy_stats: DiscrepancyStats = field(default_factory=DiscrepancyStats)


def _is_prediction_correct(action: str, actual: str) -> bool:
    if action == "buy_yes" and actual == "YES":
        return True
    if action == "buy_no" and actual == "NO":
        return True
    return False


def _parse_signal(row: dict) -> dict:
    """Extract action and confidence from a decision row."""
    try:
        sig = json.loads(row["final_signal_json"])
        return {
            "action": sig.get("action", "hold"),
            "confidence": sig.get("confidence", 0.0),
            "quality": sig.get("quality"),
        }
    except (json.JSONDecodeError, TypeError):
        return {"action": "hold", "confidence": 0.0, "quality": None}


def analyze_decisions(db: DecisionLog) -> CalibrationReport:
    all_rows = db.get_all_decisions()

    market_ids = {r["market_id"] for r in all_rows}
    settled_ids = {r["market_id"] for r in all_rows if r["settled"]}

    action_counts: dict[str, int] = {}
    for r in all_rows:
        sig = _parse_signal(r)
        action = sig["action"]
        action_counts[action] = action_counts.get(action, 0) + 1

    settled_rows = [r for r in all_rows if r["settled"] and r.get("actual_outcome")]
    bins: list[BinStats] = []
    for lo, hi, label in CONFIDENCE_BINS:
        bin_rows = []
        for r in settled_rows:
            sig = _parse_signal(r)
            if sig["action"] != "hold" and lo <= sig["confidence"] < hi:
                bin_rows.append((sig, r))
        correct = sum(
            1 for sig, r in bin_rows
            if _is_prediction_correct(sig["action"], r["actual_outcome"])
        )
        avg_conf = (
            sum(sig["confidence"] for sig, _ in bin_rows) / len(bin_rows)
            if bin_rows else 0.0
        )
        bins.append(BinStats(
            label=label,
            total=len(bin_rows),
            correct=correct,
            avg_confidence=avg_conf,
        ))

    action_stats = ActionStats(
        total_decisions=len(all_rows),
        action_counts=action_counts,
        bins=bins,
    )

    city_stats = _compute_city_stats(all_rows)
    discrepancy_stats = _compute_discrepancy_stats(db)

    return CalibrationReport(
        total_decisions=len(all_rows),
        total_markets=len(market_ids),
        settled_markets=len(settled_ids),
        unsettled_markets=len(market_ids - settled_ids),
        action_stats=action_stats,
        city_stats=city_stats,
        discrepancy_stats=discrepancy_stats,
    )


def _compute_city_stats(all_rows: list[dict]) -> list[CityStats]:
    cities: dict[str, CityStats] = {}
    for r in all_rows:
        city = r.get("city") or "Unknown"
        if city not in cities:
            cities[city] = CityStats(city=city)
        cs = cities[city]
        cs.total_samples += 1
        if r["settled"] and r.get("actual_outcome"):
            cs.settled_samples += 1
            sig = _parse_signal(r)
            if sig["action"] != "hold" and _is_prediction_correct(
                sig["action"], r["actual_outcome"]
            ):
                cs.correct += 1

    return sorted(cities.values(), key=lambda c: c.settled_samples, reverse=True)


def _compute_discrepancy_stats(db: DecisionLog) -> DiscrepancyStats:
    details = db.get_settlement_details()
    stats = DiscrepancyStats(total_checks=len(details))

    for d in details:
        is_consistent = d.get("is_consistent")
        if is_consistent is None:
            stats.unknown += 1
        elif is_consistent:
            stats.consistent += 1
        else:
            stats.inconsistent += 1
            stats.details.append(d)

    return stats
