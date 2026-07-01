from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.calibration.analyze import CalibrationReport, analyze_decisions
from agent.memory import DecisionLog


def format_report(report: CalibrationReport, date_str: str) -> str:
    lines: list[str] = []
    lines.append(f"{'=' * 72}")
    lines.append(f"  HERMES DECISION REPORT — {date_str}")
    lines.append(f"{'=' * 72}")
    lines.append("")
    lines.append(
        f"Decisions: {report.total_decisions}  "
        f"Markets: {report.total_markets}  "
        f"(settled: {report.settled_markets}, "
        f"pending: {report.unsettled_markets})"
    )
    lines.append("")

    if report.total_decisions == 0:
        lines.append("  No decision data yet.")
        return "\n".join(lines)

    # Action distribution
    astats = report.action_stats
    lines.append(f"{'─' * 72}")
    lines.append("  ACTION DISTRIBUTION")
    lines.append("")
    actions_str = ", ".join(
        f"{k}={v}" for k, v in sorted(astats.action_counts.items())
    )
    lines.append(f"  {actions_str}")
    lines.append("")

    # Accuracy (settled only)
    acc = astats.overall_accuracy
    acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
    lines.append(f"  Overall accuracy (settled, non-hold): {acc_str}")
    lines.append("")

    # Confidence calibration table
    lines.append(
        f"  {'Confidence':>12}  {'Samples':>8}  {'Correct':>8}  "
        f"{'Accuracy':>10}  {'Avg Conf':>10}"
    )
    lines.append(
        f"  {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}"
    )
    for b in astats.bins:
        acc_cell = f"{b.accuracy * 100:.1f}%" if b.accuracy is not None else "—"
        conf_cell = f"{b.avg_confidence:.3f}" if b.total > 0 else "—"
        lines.append(
            f"  {b.label:>12}  {b.total:>8}  {b.correct:>8}  "
            f"{acc_cell:>10}  {conf_cell:>10}"
        )
    lines.append("")

    # City breakdown
    if report.city_stats:
        lines.append(f"{'─' * 72}")
        lines.append("  ACCURACY BY CITY (settled non-hold samples)")
        lines.append("")
        lines.append(
            f"  {'City':>20}  {'Total':>6}  {'Settled':>8}  "
            f"{'Correct':>8}  {'Accuracy':>10}"
        )
        lines.append(
            f"  {'─' * 20}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 10}"
        )
        for cs in report.city_stats:
            acc = (
                f"{cs.accuracy * 100:.1f}%"
                if cs.accuracy is not None
                else "—"
            )
            lines.append(
                f"  {cs.city:>20}  {cs.total_samples:>6}  "
                f"{cs.settled_samples:>8}  {cs.correct:>8}  {acc:>10}"
            )
        lines.append("")

    # METAR cross-validation
    ds = report.discrepancy_stats
    if ds.total_checks > 0:
        lines.append(f"{'─' * 72}")
        lines.append("  METAR CROSS-VALIDATION (Singapore)")
        lines.append("")
        rate = ds.consistency_rate
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "—"
        lines.append(
            f"  Total checks: {ds.total_checks}  "
            f"Consistent: {ds.consistent}  "
            f"Inconsistent: {ds.inconsistent}  "
            f"Unknown: {ds.unknown}"
        )
        lines.append(f"  Consistency rate: {rate_str}")

        if ds.details:
            lines.append("")
            lines.append("  Discrepancies:")
            for d in ds.details:
                lines.append(
                    f"    {d.get('market_id', '?')} [{d.get('date', '?')}]: "
                    f"Gamma={d.get('gamma_outcome')}, "
                    f"METAR max={d.get('metar_max_temp_c')}°C "
                    f"(→{d.get('metar_rounded_c')}°C), "
                    f"threshold={d.get('threshold_temp_c')}°C "
                    f"{d.get('direction', '')}"
                )
        lines.append("")

    lines.append(f"{'=' * 72}")
    return "\n".join(lines)


def generate_report(
    db_path: str = "hermes_decisions.db",
    reports_dir: str = "reports",
) -> str:
    db = DecisionLog(db_path)
    try:
        report = analyze_decisions(db)
    finally:
        db.close()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = format_report(report, date_str)

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"calibration_{date_str}.txt"
    out_file.write_text(text, encoding="utf-8")

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hermes daily decision report"
    )
    parser.add_argument("--db", default="hermes_decisions.db")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    text = generate_report(db_path=args.db, reports_dir=args.reports_dir)
    print(text)


if __name__ == "__main__":
    main()
