#!/usr/bin/env python3
"""Summarize P3 metrics histories only; performs no model inference."""

import argparse
import csv
import json
import math
from pathlib import Path

CASES = ["N0", "S1", "G1", "G2", "G3"]
RUN_NAMES = {case: f"sga_sb_p3_{case.lower()}_seed42_e5_schedfix_v1" for case in CASES}
METRICS = ["Dice", "IoU", "mAJI", "mPQ"]
INACTIVE_DIAGNOSTICS = {
    "N0": {"final_structure_delta_norm", "final_boundary_delta_norm", "final_gamma_structure", "final_gamma_boundary"},
    "S1": {"final_structure_delta_norm", "final_boundary_delta_norm", "final_gamma_structure", "final_gamma_boundary"},
    "G1": {"final_boundary_delta_norm", "final_gamma_boundary"},
    "G2": {"final_structure_delta_norm", "final_gamma_structure"},
    "G3": set(),
}


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def load_case(models_root, case):
    path = models_root / RUN_NAMES[case] / "metrics_history.json"
    if not path.is_file():
        return {"case": case, "status": "MISSING", "path": str(path)}
    history = json.loads(path.read_text(encoding="utf-8"))
    by_epoch = {int(row["epoch"]): row for row in history}
    if 5 not in by_epoch:
        return {"case": case, "status": "INCOMPLETE", "path": str(path)}
    final = by_epoch[5]
    row = {"case": case, "status": "COMPLETE", "path": str(path)}
    for metric in METRICS:
        row[f"epoch5_{metric}"] = float(final.get(metric, float("nan")))
        row[f"epoch4_5_mean_{metric}"] = mean([by_epoch.get(e, {}).get(metric) for e in (4, 5)])
    for out_key, source in (
        ("final_structure_loss", "SBStructLoss"), ("final_boundary_loss", "SBBoundLoss"),
        ("final_structure_delta_norm", "structure_delta_norm"), ("final_boundary_delta_norm", "boundary_delta_norm"),
        ("final_gamma_structure", "gamma_structure"), ("final_gamma_boundary", "gamma_boundary"),
    ):
        row[out_key] = (
            "inactive"
            if out_key in INACTIVE_DIAGNOSTICS[case]
            else float(final.get(source, float("nan")))
        )
    return row


def fmt(value):
    if value == "inactive":
        return "inactive"
    return "NA" if not isinstance(value, (int, float)) or not math.isfinite(value) else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/hy-tmp/NuSeg")
    parser.add_argument("--output-dir", default="workdir/audits/sga_sb_p3_20260713")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    rows = [load_case(root / "workdir/models", case) for case in CASES]
    fields = ["case"] + [f"epoch5_{m}" for m in METRICS] + [f"epoch4_5_mean_{m}" for m in METRICS] + [
        "final_structure_loss", "final_boundary_loss", "final_structure_delta_norm", "final_boundary_delta_norm",
        "final_gamma_structure", "final_gamma_boundary", "status",
    ]
    with (out / "SGA_SB_P3_SUMMARY.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    lines = ["# SGA-SB P3 Summary", "", "P3 is a single-seed, five-epoch screening experiment; these results are not paper-level evidence.", "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "NA")) if k in ("case", "status") else fmt(row.get(k)) for k in fields) + " |")
    lookup = {row["case"]: row for row in rows}
    lines += ["", "## Fixed-epoch deltas", "", "| Contrast | Dice | IoU | mAJI | mPQ |", "|---|---:|---:|---:|---:|"]
    for left, right in (("S1", "N0"), ("G1", "N0"), ("G2", "N0"), ("G3", "N0"), ("G1", "S1"), ("G2", "S1"), ("G3", "S1")):
        values = []
        for metric in METRICS:
            a, b = lookup[left].get(f"epoch5_{metric}"), lookup[right].get(f"epoch5_{metric}")
            values.append(a - b if isinstance(a, float) and isinstance(b, float) else float("nan"))
        lines.append(f"| {left} - {right} | " + " | ".join(fmt(v) for v in values) + " |")
    g3, s1, n0 = lookup["G3"], lookup["S1"], lookup["N0"]
    if all(row.get("status") == "COMPLETE" for row in rows):
        minimum = g3["epoch5_mAJI"] > s1["epoch5_mAJI"] and g3["epoch5_mPQ"] > s1["epoch5_mPQ"] and g3["epoch5_mAJI"] >= n0["epoch5_mAJI"] and g3["epoch5_mPQ"] >= n0["epoch5_mPQ"]
        priority = minimum and g3["epoch4_5_mean_mAJI"] > s1["epoch4_5_mean_mAJI"] and g3["epoch4_5_mean_mPQ"] > s1["epoch4_5_mean_mPQ"]
        screening = f"Minimum P4 gate: {'PASS' if minimum else 'FAIL'}; priority gate: {'PASS' if priority else 'FAIL'}."
    else:
        screening = "Screening decision unavailable until all five epoch-5 histories are complete."
    lines += ["", "## P4 screening", "", screening]
    (out / "SGA_SB_P3_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'SGA_SB_P3_SUMMARY.md'} and {out / 'SGA_SB_P3_SUMMARY.csv'}")


if __name__ == "__main__":
    main()
