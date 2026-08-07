#!/usr/bin/env python3
"""CPU-only fidelity audit for the formal 64x64 SGA-SB boundary target."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label as connected_components
from skimage.morphology import skeletonize

from build_sga_sb_density_manifest import decode_instance_json, official_boundary64


METHODS = ("adaptive_max_pool2d", "adaptive_avg_pool2d", "nearest", "area")
GROUPS = ("all", "sparse", "medium", "dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--source-size", type=int, default=64)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="CPU smoke only")
    return parser.parse_args()


def align(source: torch.Tensor, method: str, target_size: int) -> torch.Tensor:
    size = (target_size, target_size)
    if method == "adaptive_max_pool2d":
        return F.adaptive_max_pool2d(source, size)
    if method == "adaptive_avg_pool2d":
        return F.adaptive_avg_pool2d(source, size)
    if method == "nearest":
        return F.interpolate(source, size=size, mode="nearest")
    if method == "area":
        return F.interpolate(source, size=size, mode="area")
    raise ValueError(method)


def binary_metrics(source64: np.ndarray, aligned32: np.ndarray) -> dict[str, float]:
    original = source64 > 0.5
    aligned_binary = aligned32 >= 0.5
    upsampled = cv2.resize(
        aligned_binary.astype(np.uint8),
        (source64.shape[1], source64.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    original_ratio = float(original.mean())
    ratio = float(aligned_binary.mean())
    inflation = ratio / original_ratio - 1.0 if original_ratio > 0 else math.nan
    original_cc = int(connected_components(original, connectivity=2).max())
    aligned_cc = int(connected_components(upsampled, connectivity=2).max())
    skeleton_pixels = int(skeletonize(upsampled).sum())
    thickness = float(upsampled.sum() / skeleton_pixels) if skeleton_pixels else math.nan
    original_skeleton_pixels = int(skeletonize(original).sum())
    original_thickness = (
        float(original.sum() / original_skeleton_pixels) if original_skeleton_pixels else math.nan
    )
    intersection = int(np.logical_and(original, upsampled).sum())
    denominator = int(original.sum() + upsampled.sum())
    dice = float((2 * intersection) / denominator) if denominator else 1.0
    if original.any() and upsampled.any():
        source_com = np.argwhere(original).mean(axis=0)
        aligned_com = np.argwhere(upsampled).mean(axis=0)
        com_shift = float(np.linalg.norm(aligned_com - source_com))
    else:
        com_shift = math.nan
    return {
        "original_positive_ratio": original_ratio,
        "positive_ratio": ratio,
        "positive_ratio_inflation": inflation,
        "soft_mass_ratio": float(aligned32.mean()),
        "original_boundary_thickness": original_thickness,
        "boundary_thickness": thickness,
        "boundary_thickness_inflation": (
            thickness / original_thickness - 1.0
            if math.isfinite(thickness) and math.isfinite(original_thickness) and original_thickness > 0
            else math.nan
        ),
        "original_connected_component_count": float(original_cc),
        "connected_component_count": float(aligned_cc),
        "connected_component_drop": float(original_cc - aligned_cc),
        "component_merge_indicator": float(aligned_cc < original_cc),
        "empty_map": float(not aligned_binary.any()),
        "upsampled_dice": dice,
        "center_of_mass_shift": com_shift,
    }


def finite_mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(values)) if values else math.nan


def aggregate(rows: list[dict[str, float]], method: str, group: str) -> dict[str, object]:
    fields = [
        "original_positive_ratio", "positive_ratio", "positive_ratio_inflation",
        "soft_mass_ratio", "original_boundary_thickness", "boundary_thickness",
        "boundary_thickness_inflation", "original_connected_component_count",
        "connected_component_count", "connected_component_drop", "component_merge_indicator",
        "upsampled_dice", "center_of_mass_shift",
    ]
    output: dict[str, object] = {"method": method, "density_group": group, "sample_count": len(rows)}
    output.update({field: finite_mean(rows, field) for field in fields})
    output["empty_map_count"] = int(sum(row["empty_map"] for row in rows))
    return output


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(max(1, min(2, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    cv2.setNumThreads(0)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if args.limit is not None:
        manifest = manifest[: max(args.limit, 0)]
    if not manifest:
        raise SystemExit("Manifest is empty")

    samples: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for item in manifest:
        _, raw_mask = decode_instance_json(Path(item["json_path"]))
        source64 = official_boundary64(raw_mask, args.image_size, args.source_size).astype(np.float32)
        tensor = torch.from_numpy(source64)[None, None]
        for method in METHODS:
            aligned32 = align(tensor, method, args.target_size)[0, 0].numpy()
            values = binary_metrics(source64, aligned32)
            samples[(method, "all")].append(values)
            samples[(method, item["density_group"])].append(values)

    aggregated = [aggregate(samples[(method, group)], method, group) for method in METHODS for group in GROUPS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(aggregated[0])
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregated)

    lookup = {(row["method"], row["density_group"]): row for row in aggregated}
    dense_max = lookup[("adaptive_max_pool2d", "dense")]
    findings = {
        "boundary_thickening": bool(dense_max["boundary_thickness_inflation"] > 0.10),
        "adjacent_boundary_merging": bool(
            dense_max["connected_component_drop"] > 0.10
            or dense_max["component_merge_indicator"] > 0.05
        ),
        "connected_component_reduction": bool(dense_max["connected_component_drop"] > 0.10),
        "positive_ratio_inflation": bool(dense_max["positive_ratio_inflation"] > 0.10),
    }
    findings["aliasing_evidence"] = any(findings.values())
    summary = {
        "schema_version": "sga_sb_boundary_fidelity_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "source_shape": [args.source_size, args.source_size],
        "target_shape": [args.target_size, args.target_size],
        "threshold_for_soft_targets": ">=0.5",
        "geometry_measurement_space": "aligned target nearest-upsampled back to 64x64",
        "sample_count": len(manifest),
        "methods": METHODS,
        "aggregates": aggregated,
        "dense_adaptive_max_findings": findings,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# P4 Boundary Target Fidelity Report", "",
        "The formal 64x64 binary target was reconstructed from GT only. No model or checkpoint was loaded.", "",
        "Geometry is measured after each 32x32 result is thresholded at >=0.5 and nearest-upsampled to 64x64.", "",
        "| Method | Group | N | Pos ratio | Inflation | Thickness | Thickness inflation | Components | Component drop | Empty | Up Dice | COM shift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(
            "| " + " | ".join(fmt(row[key]) for key in (
                "method", "density_group", "sample_count", "positive_ratio",
                "positive_ratio_inflation", "boundary_thickness", "boundary_thickness_inflation",
                "connected_component_count", "connected_component_drop", "empty_map_count",
                "upsampled_dice", "center_of_mass_shift",
            )) + " |"
        )
    lines += ["", "## Dense adaptive-max judgment", ""]
    lines.extend(f"- {key}: **{'YES' if value else 'NO'}**" for key, value in findings.items())
    lines += [
        "", "`adaptive_avg_pool2d` and `area` are expected to be numerically equivalent for the exact 64→32 integer reduction; both are retained as separate protocol rows.",
    ]
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "samples": len(manifest), "dense_max": findings}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
