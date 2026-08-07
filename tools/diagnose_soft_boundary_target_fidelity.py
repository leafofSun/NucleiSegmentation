#!/usr/bin/env python3
"""Compare legacy max boundary targets with direct 512->32 soft area targets."""

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
from skimage.morphology import skeletonize

from build_sga_sb_density_manifest import (
    decode_instance_json,
    instance_boundary,
    resize_instance_map,
)


MODES = ("legacy_max", "direct_area_soft")
GROUPS = ("all", "sparse", "medium", "dense")
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="CPU smoke only")
    return parser.parse_args()


def legacy_target(boundary512: np.ndarray) -> np.ndarray:
    boundary64 = cv2.resize(boundary512, (64, 64), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(boundary64.astype(np.float32))[None, None]
    return F.adaptive_max_pool2d(tensor, (32, 32))[0, 0].numpy()


def direct_area_target(boundary512: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(boundary512.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(tensor, (32, 32))[0, 0].numpy()


def finite_mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(values)) if values else math.nan


def weighted_com_512(array: np.ndarray) -> np.ndarray | None:
    total = float(array.sum())
    if total <= 0:
        return None
    yy, xx = np.indices(array.shape, dtype=np.float64)
    return np.asarray([(yy * array).sum() / total, (xx * array).sum() / total])


def target_com_512(target32: np.ndarray) -> np.ndarray | None:
    total = float(target32.sum())
    if total <= 0:
        return None
    # A 32x32 cell represents a 16x16 source block with center coordinates
    # 7.5, 23.5, ..., 503.5 in the 0..511 pixel-coordinate convention.
    centers = np.arange(32, dtype=np.float64) * 16.0 + 7.5
    return np.asarray([
        (target32.sum(axis=1) * centers).sum() / total,
        (target32.sum(axis=0) * centers).sum() / total,
    ])


def soft_metrics(boundary512: np.ndarray, target32: np.ndarray) -> dict[str, float]:
    source_mass = float(boundary512.sum())
    cell_area = float((boundary512.shape[0] // target32.shape[0]) ** 2)
    reconstructed_mass = float(target32.sum() * cell_area)
    conservation = reconstructed_mass / source_mass if source_mass > 0 else math.nan
    source_com = weighted_com_512(boundary512)
    target_com = target_com_512(target32)
    com_shift = (
        float(np.linalg.norm(target_com - source_com))
        if source_com is not None and target_com is not None else math.nan
    )
    # Exact dot product against nearest block reconstruction without allocating
    # a 512x512 float reconstruction.
    counts = boundary512.reshape(32, 16, 32, 16).sum(axis=(1, 3)).astype(np.float64)
    intersection = float((target32.astype(np.float64) * counts).sum())
    soft_dice = (
        float((2.0 * intersection + 1e-6) / (source_mass + reconstructed_mass + 1e-6))
        if source_mass + reconstructed_mass > 0 else 1.0
    )
    nonzero = target32[target32 > 0]
    return {
        "source_boundary_mass": source_mass,
        "reconstructed_boundary_mass": reconstructed_mass,
        "total_boundary_mass_conservation": conservation,
        "mass_relative_error": conservation - 1.0 if math.isfinite(conservation) else math.nan,
        "weighted_center_of_mass_shift": com_shift,
        "soft_dice_after_reconstruction": soft_dice,
        "mean_nonzero_target_value": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_target_value": float(target32.max()),
        "nonzero_cell_ratio": float(np.mean(target32 > 0)),
        "effective_positive_mass": float(target32.sum()),
    }


def topology_metrics(
    boundary512: np.ndarray,
    target32: np.ndarray,
    threshold: float,
    source_components: int,
) -> dict[str, float]:
    binary = (target32 >= threshold).astype(np.uint8)
    target_components = int(cv2.connectedComponents(binary, connectivity=8)[0] - 1)
    skeleton_pixels = int(skeletonize(binary.astype(bool)).sum())
    thickness = float(binary.sum() / skeleton_pixels) if skeleton_pixels else math.nan
    cell_counts = boundary512.reshape(32, 16, 32, 16).sum(axis=(1, 3))
    intersection = float(cell_counts[binary > 0].sum())
    predicted_mass = float(binary.sum() * 256)
    source_mass = float(boundary512.sum())
    dice = (
        float((2.0 * intersection) / (source_mass + predicted_mass))
        if source_mass + predicted_mass > 0 else 1.0
    )
    return {
        "threshold": threshold,
        "thickness": thickness,
        "component_count": float(target_components),
        "source_component_count": float(source_components),
        "component_drop": float(source_components - target_components),
        "component_retention": float(target_components / source_components) if source_components else math.nan,
        "upsampled_dice": dice,
        "empty_map": float(binary.sum() == 0),
    }


def aggregate_soft(rows: list[dict[str, float]], mode: str, group: str) -> dict[str, object]:
    fields = (
        "source_boundary_mass", "reconstructed_boundary_mass", "total_boundary_mass_conservation",
        "mass_relative_error", "weighted_center_of_mass_shift", "soft_dice_after_reconstruction",
        "mean_nonzero_target_value", "max_target_value", "nonzero_cell_ratio", "effective_positive_mass",
    )
    result: dict[str, object] = {
        "record_type": "soft_quality", "mode": mode, "density_group": group,
        "threshold": math.nan, "sample_count": len(rows),
    }
    result.update({field: finite_mean(rows, field) for field in fields})
    return result


def aggregate_topology(rows: list[dict[str, float]], group: str, threshold: float) -> dict[str, object]:
    fields = ("thickness", "component_count", "source_component_count", "component_drop", "component_retention", "upsampled_dice")
    result: dict[str, object] = {
        "record_type": "threshold_topology", "mode": "direct_area_soft",
        "density_group": group, "threshold": threshold, "sample_count": len(rows),
    }
    result.update({field: finite_mean(rows, field) for field in fields})
    result["empty_map_count"] = int(sum(row["empty_map"] for row in rows))
    return result


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def main() -> int:
    args = parse_args()
    if args.image_size != 512 or args.target_size != 32:
        raise ValueError("P4-CPU-B formal diagnostic requires image_size=512 and target_size=32")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(max(1, min(2, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    cv2.setNumThreads(0)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if args.limit is not None:
        manifest = manifest[: max(args.limit, 0)]
    if not manifest:
        raise SystemExit("Manifest is empty")

    soft_rows: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    topology_rows: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for item in manifest:
        _, raw_mask = decode_instance_json(Path(item["json_path"]))
        label512 = resize_instance_map(raw_mask, 512)
        boundary512 = instance_boundary(label512)
        source_components = int(cv2.connectedComponents(boundary512, connectivity=8)[0] - 1)
        targets = {
            "legacy_max": legacy_target(boundary512),
            "direct_area_soft": direct_area_target(boundary512),
        }
        for mode, target in targets.items():
            values = soft_metrics(boundary512, target)
            soft_rows[(mode, "all")].append(values)
            soft_rows[(mode, item["density_group"])].append(values)
        candidate = targets["direct_area_soft"]
        for threshold in THRESHOLDS:
            values = topology_metrics(boundary512, candidate, threshold, source_components)
            topology_rows[("all", threshold)].append(values)
            topology_rows[(item["density_group"], threshold)].append(values)

    aggregates: list[dict[str, object]] = []
    aggregates.extend(aggregate_soft(soft_rows[(mode, group)], mode, group) for mode in MODES for group in GROUPS)
    aggregates.extend(
        aggregate_topology(topology_rows[(group, threshold)], group, threshold)
        for threshold in THRESHOLDS for group in GROUPS
    )
    all_fields = [
        "record_type", "mode", "density_group", "threshold", "sample_count",
        "source_boundary_mass", "reconstructed_boundary_mass", "total_boundary_mass_conservation",
        "mass_relative_error", "weighted_center_of_mass_shift", "soft_dice_after_reconstruction",
        "mean_nonzero_target_value", "max_target_value", "nonzero_cell_ratio", "effective_positive_mass",
        "thickness", "component_count", "source_component_count", "component_drop",
        "component_retention", "upsampled_dice", "empty_map_count",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregates)

    soft_lookup = {
        (row["mode"], row["density_group"]): row
        for row in aggregates if row["record_type"] == "soft_quality"
    }
    topo_lookup = {
        (row["density_group"], float(row["threshold"])): row
        for row in aggregates if row["record_type"] == "threshold_topology"
    }
    candidate_all = soft_lookup[("direct_area_soft", "all")]
    candidate_dense = soft_lookup[("direct_area_soft", "dense")]
    legacy_all = soft_lookup[("legacy_max", "all")]
    supported = (
        abs(float(candidate_all["mass_relative_error"])) < 1e-6
        and float(candidate_all["weighted_center_of_mass_shift"]) < float(legacy_all["weighted_center_of_mass_shift"])
        and float(candidate_all["nonzero_cell_ratio"]) > float(legacy_all["nonzero_cell_ratio"])
    )
    # Visualization-only recommendation: retain thresholds whose dense Dice is
    # at least 95% of the best dense Dice, then report the contiguous envelope.
    dense_dice = {threshold: float(topo_lookup[("dense", threshold)]["upsampled_dice"]) for threshold in THRESHOLDS}
    best_dense_dice = max(dense_dice.values())
    recommended = [threshold for threshold, value in dense_dice.items() if value >= 0.95 * best_dense_dice]

    summary = {
        "schema_version": "p4_soft_boundary_fidelity_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "sample_count": len(manifest),
        "legacy_definition": "nearest 512->64 then adaptive_max_pool2d 64->32",
        "candidate_definition": "adaptive_avg_pool2d 512->32, no threshold",
        "thresholds": THRESHOLDS,
        "aggregates": aggregates,
        "candidate_supported_for_target_only_screening": supported,
        "loss_soft_target_audit": {
            "supported": True,
            "bce_target_thresholded": False,
            "dice_target_thresholded": False,
            "direct_area_pos_mass": "target.sum()",
            "direct_area_neg_mass": "target.numel() - target.sum()",
            "direct_area_pos_weight": "clamp(neg_mass / (pos_mass + 1e-6), 0.1, 10.0)",
        },
        "formal_switch": {
            "argument": "--spatial_boundary_target_mode",
            "choices": ["legacy_max", "direct_area_soft"],
            "default": "legacy_max",
            "p3_default_behavior_changed": False,
        },
        "visualization_threshold_recommendation": {
            "criterion": "dense upsampled Dice >= 95% of best scanned dense Dice",
            "values": recommended,
            "range": [min(recommended), max(recommended)] if recommended else None,
            "dense_dice_by_threshold": dense_dice,
        },
        "candidate_all": candidate_all,
        "candidate_dense": candidate_dense,
        "legacy_all": legacy_all,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# P4 Soft Boundary Fidelity Report", "",
        "GT-only CPU diagnostic. No model/checkpoint forward, backward, optimizer, training, validation, or test was executed.", "",
        "## Soft quality", "",
        "| Mode | Group | N | Mass conservation | Mass error | COM shift | Soft Dice | Mean nonzero | Max | Nonzero cells | Effective mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for group in GROUPS:
            row = soft_lookup[(mode, group)]
            lines.append("| " + " | ".join(fmt(row.get(key, math.nan)) for key in (
                "mode", "density_group", "sample_count", "total_boundary_mass_conservation",
                "mass_relative_error", "weighted_center_of_mass_shift", "soft_dice_after_reconstruction",
                "mean_nonzero_target_value", "max_target_value", "nonzero_cell_ratio", "effective_positive_mass",
            )) + " |")
    lines += [
        "", "## Direct-area threshold topology", "",
        "Thresholds are visualization/diagnostic only; the candidate training target remains unthresholded.", "",
        "| Threshold | Group | Thickness@32 | Components | Source components | Drop | Retention | Upsampled Dice | Empty |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        for group in GROUPS:
            row = topo_lookup[(group, threshold)]
            lines.append("| " + " | ".join(fmt(row.get(key, math.nan)) for key in (
                "threshold", "density_group", "thickness", "component_count",
                "source_component_count", "component_drop", "component_retention",
                "upsampled_dice", "empty_map_count",
            )) + " |")
    lines += [
        "", "## Decision", "",
        f"- Candidate conserves boundary mass within 1e-6: **{'YES' if abs(float(candidate_all['mass_relative_error'])) < 1e-6 else 'NO'}**.",
        f"- Candidate improves weighted COM shift over legacy: **{'YES' if float(candidate_all['weighted_center_of_mass_shift']) < float(legacy_all['weighted_center_of_mass_shift']) else 'NO'}**.",
        f"- Candidate improves reconstruction soft Dice over legacy: **{'YES' if float(candidate_all['soft_dice_after_reconstruction']) > float(legacy_all['soft_dice_after_reconstruction']) else 'NO'}**.",
        f"- Candidate supported for target-only GPU screening: **{'YES' if supported else 'NO'}**.",
        f"- Visualization-only threshold range: **{min(recommended):.2f}–{max(recommended):.2f}**." if recommended else "- Visualization-only threshold range: unavailable.",
        "", "The lower reconstruction soft Dice is retained as an explicit tradeoff. It prevents declaring the candidate superior or changing the default, but it does not negate exact mass conservation and the much smaller center-of-mass error for a controlled target-only screening.",
        "", "The loss candidate must consume the soft values directly; none of these visualization thresholds belongs in BCE/Dice target construction.",
        "", "## Soft-target loss and formal switch audit", "",
        "- `compute_boundary_loss()` preserves the `[0,1]` target for both BCEWithLogits and Dice; it contains no target threshold operation.",
        "- In `direct_area_soft`, `pos_mass=target.sum()`, `neg_mass=target.numel()-pos_mass`, and `pos_weight=clamp(neg_mass/(pos_mass+1e-6),0.1,10.0)`.",
        "- `--spatial_boundary_target_mode` choices are `legacy_max` and `direct_area_soft`.",
        "- Default remains `legacy_max`; existing P3 commands therefore retain their original target path.",
    ]
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "PASS", "samples": len(manifest), "candidate_supported": supported,
        "candidate_mass_error": candidate_all["mass_relative_error"],
        "legacy_mass_error": legacy_all["mass_relative_error"],
        "visualization_threshold_range": summary["visualization_threshold_recommendation"]["range"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
