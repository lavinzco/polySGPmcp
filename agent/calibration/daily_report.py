from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.calibration.analyze import CalibrationReport, analyze_calibration
from agent.calibration.db import CalibrationDB


def format_report(report: CalibrationReport, date_str: str) -> str:
    lines: list[str] = []
    lines.append(f"{'=' * 72}")
    lines.append(f"  HERMES CALIBRATION REPORT — {date_str}")
    lines.append(f"{'=' * 72}")
    lines.append("")
    lines.append(f"Markets tracked: {report.total_markets}  "
                 f"(settled: {report.settled_markets}, "
                 f"pending: {report.unsettled_markets})")
    lines.append("")

    if not report.providers:
        lines.append("  No calibration data yet.")
        return "\n".join(lines)

    # Per-provider breakdown
    for ps in report.providers:
        lines.append(f"{'─' * 72}")
        lines.append(f"  Provider: {ps.name}")
        lines.append(f"  Samples: {ps.total_samples} total, {ps.settled_samples} settled")

        acc = ps.overall_accuracy
        acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
        lines.append(f"  Overall accuracy (settled, non-hold): {acc_str}")
        lines.append("")

        actions_str = ", ".join(f"{k}={v}" for k, v in sorted(ps.action_counts.items()))
        lines.append(f"  Action distribution: {actions_str}")
        lines.append("")

        # Calibration table
        lines.append(f"  {'Confidence':>12}  {'Samples':>8}  {'Correct':>8}  {'Accuracy':>10}  {'Avg Conf':>10}")
        lines.append(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
        for b in ps.bins:
            acc_cell = f"{b.accuracy*100:.1f}%" if b.accuracy is not None else "—"
            conf_cell = f"{b.avg_confidence:.3f}" if b.total > 0 else "—"
            lines.append(
                f"  {b.label:>12}  {b.total:>8}  {b.correct:>8}  {acc_cell:>10}  {conf_cell:>10}"
            )
        lines.append("")

    # Comparison table (if multiple providers)
    if len(report.providers) > 1:
        lines.append(f"{'─' * 72}")
        lines.append("  MODEL COMPARISON (settled non-hold samples)")
        lines.append("")
        header = f"  {'Provider':>20}  {'Samples':>8}  {'Accuracy':>10}  {'Avg Hold%':>10}"
        lines.append(header)
        lines.append(f"  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*10}")
        for ps in report.providers:
            acc = ps.overall_accuracy
            acc_str = f"{acc*100:.1f}%" if acc is not None else "—"
            hold_count = ps.action_counts.get("hold", 0)
            hold_pct = (hold_count / ps.total_samples * 100) if ps.total_samples > 0 else 0
            lines.append(
                f"  {ps.name:>20}  {ps.settled_samples:>8}  {acc_str:>10}  {hold_pct:>9.1f}%"
            )
        lines.append("")

    # City breakdown
    if report.city_stats:
        lines.append(f"{'─' * 72}")
        lines.append("  ACCURACY BY CITY (settled non-hold samples)")
        lines.append("")
        lines.append(f"  {'City':>20}  {'Total':>6}  {'Settled':>8}  {'Correct':>8}  {'Accuracy':>10}")
        lines.append(f"  {'─'*20}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")
        for cs in report.city_stats:
            acc = f"{cs.accuracy*100:.1f}%" if cs.accuracy is not None else "—"
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
        rate_str = f"{rate*100:.1f}%" if rate is not None else "—"
        lines.append(f"  Total checks: {ds.total_checks}  "
                     f"Consistent: {ds.consistent}  "
                     f"Inconsistent: {ds.inconsistent}  "
                     f"Unknown: {ds.unknown}")
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
                    f"threshold={d.get('threshold_temp_c')}°C {d.get('direction', '')}"
                )
        lines.append("")

    lines.append(f"{'=' * 72}")
    return "\n".join(lines)


def generate_report(
    db_path: str = "hermes_calibration.db",
    reports_dir: str = "reports",
) -> str:
    db = CalibrationDB(db_path)
    try:
        report = analyze_calibration(db)
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
    parser = argparse.ArgumentParser(description="Generate Hermes daily calibration report")
    parser.add_argument("--db", default="hermes_calibration.db")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    text = generate_report(db_path=args.db, reports_dir=args.reports_dir)
    print(text)


if __name__ == "__main__":
    main()
