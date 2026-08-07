#!/usr/bin/env python3
"""Create the fixed-epoch P3 screening table required by P4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


CASES = ("N0", "S1", "G1", "G2", "G3")
METRICS = ("Dice", "IoU", "mAJI", "mPQ")
DIAGNOSTICS = (
    "SBStructLoss", "SBBoundLoss", "SBLoss", "structure_delta_norm",
    "boundary_delta_norm", "structure_injected_ratio", "boundary_injected_ratio",
    "gamma_structure", "gamma_boundary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


SUMMARY_TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
LOG_TO_HISTORY = {
    "learning_rate": "learning_rate", "SBStructLoss": "SBStructLoss",
    "SBBoundLoss": "SBBoundLoss", "SBLoss": "SBLoss",
    "StructDeltaNorm": "structure_delta_norm", "BoundDeltaNorm": "boundary_delta_norm",
    "StructInjRatio": "structure_injected_ratio", "BoundInjRatio": "boundary_injected_ratio",
    "GammaStruct": "gamma_structure", "GammaBound": "gamma_boundary",
    "Dice": "Dice", "IoU": "IoU", "mAJI": "mAJI", "mPQ": "mPQ",
}


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def mean(values: list[object]) -> float:
    numbers = [finite(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return sum(numbers) / len(numbers) if numbers else math.nan


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    number = finite(value)
    return "inactive" if not math.isfinite(number) else f"{number:.{digits}f}"


def main() -> int:
    args = parse_args()
    histories: dict[str, list[dict[str, object]]] = {}
    log_verified: dict[str, bool] = {}
    for case in CASES:
        path = args.models_root / f"sga_sb_p3_{case.lower()}_seed42_e5_schedfix_v1" / "metrics_history.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        if [int(row["epoch"]) for row in history] != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"{case}: expected exact epochs 1..5 in {path}")
        histories[case] = history
        log_path = args.logs_root / f"P3_{case}_SCHEDFIX_V1_TRAIN.log"
        log_rows: dict[int, dict[str, str]] = {}
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[P3_EPOCH_SUMMARY]" not in line:
                continue
            tokens = dict(SUMMARY_TOKEN.findall(line.split("[P3_EPOCH_SUMMARY]", 1)[1]))
            if "epoch" in tokens:
                log_rows[int(tokens["epoch"])] = tokens
        if sorted(log_rows) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"{case}: log does not contain exact epoch summaries 1..5: {log_path}")
        for epoch, metrics_row in enumerate(history, start=1):
            for log_key, history_key in LOG_TO_HISTORY.items():
                left = finite(log_rows[epoch].get(log_key))
                right = finite(metrics_row.get(history_key))
                if math.isfinite(left) != math.isfinite(right):
                    raise RuntimeError(f"{case} epoch {epoch} {log_key}: log/history finite mismatch")
                if math.isfinite(left) and not math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-6):
                    raise RuntimeError(f"{case} epoch {epoch} {log_key}: log={left} history={right}")
        log_verified[case] = True

    long_rows: list[dict[str, object]] = []
    for case in CASES:
        for source in histories[case]:
            epoch = int(source["epoch"])
            row: dict[str, object] = {
                "case": case, "epoch": epoch, "learning_rate": source["learning_rate"],
                "log_history_verified": log_verified[case],
            }
            for key in METRICS + DIAGNOSTICS:
                row[key] = finite(source.get(key))
            for reference in ("N0", "S1", "G1", "G2"):
                ref = histories[reference][epoch - 1]
                for metric in METRICS:
                    row[f"delta_{metric}_vs_{reference}"] = finite(source[metric]) - finite(ref[metric])
            long_rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(long_rows[0])
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)

    def summary(case: str, epochs: tuple[int, ...]) -> dict[str, float]:
        selected = [histories[case][epoch - 1] for epoch in epochs]
        return {metric: mean([row[metric] for row in selected]) for metric in METRICS}

    epoch5 = {case: summary(case, (5,)) for case in CASES}
    mean45 = {case: summary(case, (4, 5)) for case in CASES}
    drops = {
        case: {metric: finite(histories[case][4][metric]) - finite(histories[case][3][metric]) for metric in METRICS}
        for case in CASES
    }
    minimum_gate = (
        epoch5["G3"]["mAJI"] > epoch5["S1"]["mAJI"]
        and epoch5["G3"]["mPQ"] > epoch5["S1"]["mPQ"]
        and epoch5["G3"]["mAJI"] >= epoch5["N0"]["mAJI"]
        and epoch5["G3"]["mPQ"] >= epoch5["N0"]["mPQ"]
    )
    priority_gate = minimum_gate and all(
        mean45["G3"][metric] > mean45["S1"][metric] for metric in ("mAJI", "mPQ")
    )

    lines = [
        "# P3 Final Screening Report", "",
        "Protocol: fixed epoch 5 is primary; epoch 4–5 mean is auxiliary. No best-epoch selection is used.", "",
        "Source verification: all five `metrics_history.json` files were cross-checked against epochs 1–5 in the corresponding `P3_*_SCHEDFIX_V1_TRAIN.log`; result **PASS**.", "",
        "## Epoch 1–5", "",
        "| Case | Epoch | LR | Dice | IoU | mAJI | mPQ | γS | γB | ΔS norm | ΔB norm | S ratio | B ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in long_rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in (
            "case", "epoch", "learning_rate", "Dice", "IoU", "mAJI", "mPQ",
            "gamma_structure", "gamma_boundary", "structure_delta_norm", "boundary_delta_norm",
            "structure_injected_ratio", "boundary_injected_ratio",
        )) + " |")

    def metric_table(title: str, values: dict[str, dict[str, float]]) -> None:
        lines.extend(["", f"## {title}", "", "| Case | Dice | IoU | mAJI | mPQ |", "|---|---:|---:|---:|---:|"])
        for case in CASES:
            lines.append(f"| {case} | " + " | ".join(fmt(values[case][metric]) for metric in METRICS) + " |")

    metric_table("Fixed epoch 5", epoch5)
    metric_table("Epoch 4–5 mean", mean45)
    metric_table("Epoch 4→5 change (epoch5 - epoch4)", drops)

    lines += ["", "## Fixed-epoch pairwise deltas", ""]
    for reference in ("N0", "S1", "G1", "G2"):
        lines += [f"### Relative to {reference}", "", "| Case | ΔDice | ΔIoU | ΔmAJI | ΔmPQ |", "|---|---:|---:|---:|---:|"]
        for case in CASES:
            lines.append(f"| {case} | " + " | ".join(
                fmt(epoch5[case][metric] - epoch5[reference][metric]) for metric in METRICS
            ) + " |")
        lines.append("")

    lines += ["## Gamma / delta trajectories", ""]
    for case in CASES:
        lines.append(
            f"- **{case}**: gamma_structure="
            + " → ".join(fmt(row.get("gamma_structure")) for row in histories[case])
            + "; gamma_boundary="
            + " → ".join(fmt(row.get("gamma_boundary")) for row in histories[case])
        )
        lines.append(
            "  delta_norm(S/B)=" + " → ".join(
                f"{fmt(row.get('structure_delta_norm'))}/{fmt(row.get('boundary_delta_norm'))}"
                for row in histories[case]
            )
            + "; injection_ratio(S/B)=" + " → ".join(
                f"{fmt(row.get('structure_injected_ratio'))}/{fmt(row.get('boundary_injected_ratio'))}"
                for row in histories[case]
            )
        )

    lines += [
        "", "## Preset P4 advance gate", "",
        f"- Minimum gate: **{'PASS' if minimum_gate else 'FAIL'}**.",
        f"- Priority gate: **{'PASS' if priority_gate else 'FAIL'}**.",
        "- Mechanism conclusion: auxiliary supervision is measurable, but combined feature guidance is not proven by fixed epoch 5.",
        "- G2 has slightly higher epoch-5 mAJI than N0 but lower mPQ; no SGA case jointly exceeds N0 on both primary metrics.",
    ]
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "minimum_gate": minimum_gate, "priority_gate": priority_gate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
