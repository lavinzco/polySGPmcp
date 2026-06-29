from __future__ import annotations

from dataclasses import dataclass, field

from agent.calibration.db import CalibrationDB

CONFIDENCE_BINS = [
    (0.0, 0.2, "0.0–0.2"),
    (0.2, 0.4, "0.2–0.4"),
    (0.4, 0.6, "0.4–0.6"),
    (0.6, 0.8, "0.6–0.8"),
    (0.8, 1.01, "0.8–1.0"),
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
class ProviderStats:
    name: str
    total_samples: int = 0
    settled_samples: int = 0
    bins: list[BinStats] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float | None:
        if self.settled_samples == 0:
            return None
        correct = sum(b.correct for b in self.bins)
        return correct / self.settled_samples


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
    providers: list[ProviderStats]
    total_markets: int = 0
    settled_markets: int = 0
    unsettled_markets: int = 0
    city_stats: list[CityStats] = field(default_factory=list)
    discrepancy_stats: DiscrepancyStats = field(default_factory=DiscrepancyStats)


def _is_prediction_correct(action: str, actual: str) -> bool:
    if action == "buy_yes" and actual == "YES":
        return True
    if action == "buy_no" and actual == "NO":
        return True
    return False


def analyze_calibration(db: CalibrationDB) -> CalibrationReport:
    provider_names = db.get_provider_names()
    all_samples = db.get_all_samples()

    market_ids = {s["market_id"] for s in all_samples}
    settled_ids = {s["market_id"] for s in all_samples if s["settled"]}

    provider_stats: list[ProviderStats] = []

    for pname in provider_names:
        samples = db.get_samples_by_provider(pname)
        settled = [s for s in samples if s["settled"] and s["actual_outcome"]]

        bins: list[BinStats] = []
        for lo, hi, label in CONFIDENCE_BINS:
            bin_samples = [
                s for s in settled
                if lo <= s["llm_confidence"] < hi and s["llm_action"] != "hold"
            ]
            correct = sum(
                1 for s in bin_samples
                if _is_prediction_correct(s["llm_action"], s["actual_outcome"])
            )
            avg_conf = (
                sum(s["llm_confidence"] for s in bin_samples) / len(bin_samples)
                if bin_samples
                else 0.0
            )
            bins.append(BinStats(
                label=label,
                total=len(bin_samples),
                correct=correct,
                avg_confidence=avg_conf,
            ))

        action_counts: dict[str, int] = {}
        for s in samples:
            action_counts[s["llm_action"]] = action_counts.get(s["llm_action"], 0) + 1

        provider_stats.append(ProviderStats(
            name=pname,
            total_samples=len(samples),
            settled_samples=len(settled),
            bins=bins,
            action_counts=action_counts,
        ))

    city_stats = _compute_city_stats(all_samples)
    discrepancy_stats = _compute_discrepancy_stats(db)

    return CalibrationReport(
        providers=provider_stats,
        total_markets=len(market_ids),
        settled_markets=len(settled_ids),
        unsettled_markets=len(market_ids - settled_ids),
        city_stats=city_stats,
        discrepancy_stats=discrepancy_stats,
    )


def _compute_city_stats(all_samples: list[dict]) -> list[CityStats]:
    """Compute per-city accuracy breakdown from settled samples."""
    cities: dict[str, CityStats] = {}
    for s in all_samples:
        city = s.get("city", "Unknown")
        if city not in cities:
            cities[city] = CityStats(city=city)
        cs = cities[city]
        cs.total_samples += 1
        if s["settled"] and s.get("actual_outcome"):
            cs.settled_samples += 1
            if s["llm_action"] != "hold" and _is_prediction_correct(
                s["llm_action"], s["actual_outcome"]
            ):
                cs.correct += 1

    return sorted(cities.values(), key=lambda c: c.settled_samples, reverse=True)


def _compute_discrepancy_stats(db: CalibrationDB) -> DiscrepancyStats:
    """Compute METAR cross-validation statistics from settlement_details table."""
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
