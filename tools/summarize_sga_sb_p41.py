#!/usr/bin/env python3
"""Compare the formal G2-Legacy and G2-Soft five-epoch histories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = ("Dice", "IoU", "mAJI", "mPQ")
LEGACY_E5 = {
    "Dice": 0.819381,
    "IoU": 0.705173,
    "mAJI": 0.636847,
    "mPQ": 0.547004,
}
LEGACY_E45_MEAN = {
    "Dice": 0.819923,
    "IoU": 0.706681,
    "mAJI": 0.644116,
    "mPQ": 0.554363,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_history(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        payload = payload["history"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    history: dict[int, dict[str, object]] = {}
    for row in payload:
        if not isinstance(row, dict) or "epoch" not in row:
            raise ValueError(f"Malformed history row in {path}: {row!r}")
        epoch = int(row["epoch"])
        if epoch in history:
            raise ValueError(f"Duplicate epoch {epoch} in {path}")
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {metric} at epoch {epoch} in {path}")
        history[epoch] = row
    missing = {4, 5} - set(history)
    if missing:
        raise ValueError(f"Missing epochs {sorted(missing)} in {path}")
    return history


def metric_value(history: dict[int, dict[str, object]], epoch: int, metric: str) -> float:
    return float(history[epoch][metric])


def mean_e45(history: dict[int, dict[str, object]], metric: str) -> float:
    return (metric_value(history, 4, metric) + metric_value(history, 5, metric)) / 2.0


def verify_legacy_anchors(history: dict[int, dict[str, object]]) -> None:
    failures = []
    for metric in METRICS:
        actual_e5 = metric_value(history, 5, metric)
        actual_mean = mean_e45(history, metric)
        if round(actual_e5, 6) != round(LEGACY_E5[metric], 6):
            failures.append(f"{metric} E5 actual={actual_e5:.9f} fixed={LEGACY_E5[metric]:.9f}")
        if round(actual_mean, 6) != round(LEGACY_E45_MEAN[metric], 6):
            failures.append(
                f"{metric} E4-5 mean actual={actual_mean:.9f} fixed={LEGACY_E45_MEAN[metric]:.9f}"
            )
    if failures:
        raise ValueError("G2-Legacy history does not match fixed control anchors: " + "; ".join(failures))


def build_rows(
    legacy: dict[int, dict[str, object]],
    soft: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        soft_e5 = metric_value(soft, 5, metric)
        soft_mean = mean_e45(soft, metric)
        legacy_change = metric_value(legacy, 5, metric) - metric_value(legacy, 4, metric)
        soft_change = metric_value(soft, 5, metric) - metric_value(soft, 4, metric)
        rows.append({
            "metric": metric,
            "legacy_e5": LEGACY_E5[metric],
            "soft_e5": soft_e5,
            "delta_e5": soft_e5 - LEGACY_E5[metric],
            "legacy_e4_e5_mean": LEGACY_E45_MEAN[metric],
            "soft_e4_e5_mean": soft_mean,
            "delta_e4_e5_mean": soft_mean - LEGACY_E45_MEAN[metric],
            "legacy_e4_to_e5_change": legacy_change,
            "soft_e4_to_e5_change": soft_change,
            "e4_to_e5_change_difference": soft_change - legacy_change,
        })
    return rows


def evaluate_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {str(row["metric"]): row for row in rows}
    checks = {
        "e5_delta_mpq_ge_0.003": float(lookup["mPQ"]["delta_e5"]) >= 0.003,
        "e5_delta_maji_ge_0": float(lookup["mAJI"]["delta_e5"]) >= 0.0,
        "e45_delta_mpq_gt_0": float(lookup["mPQ"]["delta_e4_e5_mean"]) > 0.0,
        "e45_delta_maji_ge_0": float(lookup["mAJI"]["delta_e4_e5_mean"]) >= 0.0,
        "e5_dice_drop_le_0.003": float(lookup["Dice"]["delta_e5"]) >= -0.003,
        "e5_iou_drop_le_0.003": float(lookup["IoU"]["delta_e5"]) >= -0.003,
        "e45_dice_drop_le_0.003": float(lookup["Dice"]["delta_e4_e5_mean"]) >= -0.003,
        "e45_iou_drop_le_0.003": float(lookup["IoU"]["delta_e4_e5_mean"]) >= -0.003,
    }
    return {"checks": checks, "advance": all(checks.values())}


def fmt(value: object) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_outputs(
    rows: list[dict[str, object]],
    gate: dict[str, object],
    output_csv: Path,
    output_md: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# P4.1 G2-Soft Comparison", "",
        "Primary comparisons use fixed Epoch 5 and Epoch 4-5 mean; best epoch is not used.", "",
        "| Metric | Legacy E5 | Soft E5 | Delta E5 | Legacy E4-5 mean | Soft E4-5 mean | Delta mean | Legacy E4->5 | Soft E4->5 | Change delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[field]) for field in fields) + " |")
    lines += ["", "## Advance gate", ""]
    for name, passed in gate["checks"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines += ["", f"Overall: **{'ADVANCE' if gate['advance'] else 'DO_NOT_ADVANCE'}**", ""]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    legacy = load_history(args.legacy)
    soft = load_history(args.soft)
    verify_legacy_anchors(legacy)
    rows = build_rows(legacy, soft)
    gate = evaluate_gate(rows)
    write_outputs(rows, gate, args.output_csv, args.output_md)
    print(json.dumps({"result": "PASS", "advance": gate["advance"], "checks": gate["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
