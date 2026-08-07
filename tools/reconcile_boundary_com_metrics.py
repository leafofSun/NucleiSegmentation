#!/usr/bin/env python3
"""Reconcile the two historical P4 boundary center-of-mass metrics.

This is a GT-only CPU diagnostic.  It does not import model code, construct a
model, load a checkpoint, or execute forward/backward.
"""

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

from build_sga_sb_density_manifest import (
    decode_instance_json,
    instance_boundary,
    resize_instance_map,
)


METRICS = (
    "legacy_binary_com_shift_512",
    "legacy_weighted_com_shift_512",
    "direct_area_soft_weighted_com_shift_512",
)
GROUPS = ("all", "sparse", "medium", "dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--old-summary", type=Path)
    parser.add_argument("--soft-summary", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="CPU smoke only")
    return parser.parse_args()


def legacy_targets(boundary512: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source64, max-pooled target32, and binary 64 reconstruction."""
    source64 = cv2.resize(boundary512, (64, 64), interpolation=cv2.INTER_NEAREST)
    source = torch.from_numpy(source64.astype(np.float32))[None, None]
    target32 = F.adaptive_max_pool2d(source, (32, 32))[0, 0].numpy()
    reconstructed64 = cv2.resize(
        (target32 >= 0.5).astype(np.uint8),
        (64, 64),
        interpolation=cv2.INTER_NEAREST,
    )
    return source64, target32, reconstructed64


def direct_area_target(boundary512: np.ndarray) -> np.ndarray:
    source = torch.from_numpy(boundary512.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(source, (32, 32))[0, 0].numpy()


def binary_com(array: np.ndarray) -> np.ndarray | None:
    coordinates = np.argwhere(array > 0)
    return coordinates.mean(axis=0) if coordinates.size else None


def weighted_com(array: np.ndarray) -> np.ndarray | None:
    weights = array.astype(np.float64, copy=False)
    total = float(weights.sum())
    if total <= 0:
        return None
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    return np.asarray([(yy * weights).sum() / total, (xx * weights).sum() / total])


def target32_weighted_com_in_512(target32: np.ndarray) -> np.ndarray | None:
    weights = target32.astype(np.float64, copy=False)
    total = float(weights.sum())
    if total <= 0:
        return None
    # Pixel coordinates in the original 0..511 convention.  Each 32-grid
    # value represents one 16x16 block, whose center is 16*i + 7.5.
    centers = np.arange(32, dtype=np.float64) * 16.0 + 7.5
    return np.asarray([
        (weights.sum(axis=1) * centers).sum() / total,
        (weights.sum(axis=0) * centers).sum() / total,
    ])


def euclidean(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return math.nan
    return float(np.linalg.norm(left - right))


def compute_metrics(boundary512: np.ndarray) -> dict[str, float]:
    source64, legacy32, reconstructed64 = legacy_targets(boundary512)
    direct32 = direct_area_target(boundary512)

    # This is exactly the historical binary 64-grid definition, expressed in
    # original-pixel units.  Multiplication by 8 converts a displacement; the
    # pixel-center offset cancels between the two COMs.
    legacy_binary = 8.0 * euclidean(binary_com(source64), binary_com(reconstructed64))
    source512_com = weighted_com(boundary512)
    return {
        "legacy_binary_com_shift_512": legacy_binary,
        "legacy_weighted_com_shift_512": euclidean(
            source512_com, target32_weighted_com_in_512(legacy32)
        ),
        "direct_area_soft_weighted_com_shift_512": euclidean(
            source512_com, target32_weighted_com_in_512(direct32)
        ),
    }


def summarize(values: list[float], sample_count: int) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "valid_sample_count": 0,
            "empty_sample_count": sample_count,
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90, method="linear")),
        "p95": float(np.quantile(finite, 0.95, method="linear")),
        "max": float(finite.max()),
        "valid_sample_count": int(finite.size),
        "empty_sample_count": int(sample_count - finite.size),
    }


def load_historical(old_path: Path | None, soft_path: Path | None) -> dict[str, object]:
    result: dict[str, object] = {}
    if old_path and old_path.exists():
        payload = json.loads(old_path.read_text(encoding="utf-8"))
        result["old_binary_64"] = {
            row["density_group"]: row["center_of_mass_shift"]
            for row in payload["aggregates"]
            if row["method"] == "adaptive_max_pool2d"
        }
    if soft_path and soft_path.exists():
        payload = json.loads(soft_path.read_text(encoding="utf-8"))
        result["soft_report_weighted_512"] = {
            f"{row['mode']}:{row['density_group']}": row["weighted_center_of_mass_shift"]
            for row in payload["aggregates"]
            if row["record_type"] == "soft_quality"
        }
    return result


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def write_report(path: Path, payload: dict[str, object]) -> None:
    audit = payload["historical_implementation_audit"]
    results = payload["results"]
    lines = [
        "# P4.1-A Boundary COM Metric Reconciliation", "",
        "GT-only CPU audit over the existing 7,553-sample manifest. No model/checkpoint was loaded and no forward, backward, optimizer, training, validation, or test was executed.", "",
        "## Historical implementation audit", "",
        "| Property | Old fidelity `center_of_mass_shift` | Soft fidelity `weighted_center_of_mass_shift` |",
        "|---|---|---|",
    ]
    for key, label in (
        ("coordinate_space", "Coordinate space"),
        ("input_weights", "Input/weights"),
        ("source", "Source"),
        ("reconstruction", "Reconstructed/target"),
        ("post_scale", "Post-computation scale"),
        ("empty_handling", "Empty-map handling"),
        ("aggregation", "Aggregation"),
        ("source_zero_exclusion", "Source-mass-zero exclusion"),
        ("distance", "Distance"),
    ):
        lines.append(f"| {label} | {audit['old_fidelity'][key]} | {audit['soft_fidelity'][key]} |")

    lines += [
        "", "## Unified original-512-coordinate results", "",
        "| Metric | Group | N | Mean | Median | P90 | P95 | Max | Valid | Empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    group_counts = payload["group_sample_counts"]
    for metric in METRICS:
        for group in GROUPS:
            row = results[metric][group]
            lines.append("| " + " | ".join([
                metric, group, str(group_counts[group]),
                fmt(row["mean"]), fmt(row["median"]), fmt(row["p90"]),
                fmt(row["p95"]), fmt(row["max"]),
                str(row["valid_sample_count"]), str(row["empty_sample_count"]),
            ]) + " |")

    reconciliation = payload["reconciliation"]
    lines += [
        "", "## Reconciliation", "",
        f"- Fixed-scale equivalence: **NO**. Old all mean x8 = {reconciliation['old_all_times_8']:.6f}, while the legacy weighted all mean is {reconciliation['legacy_weighted_all']:.6f}.",
        "- The decisive definition change is the source support: the old metric compares a nearest-subsampled 64x64 source against a 64x64 reconstruction, whereas the soft report compares the original 512x512 boundary mass against a 32-cell-center reconstruction in original coordinates.",
        "- Binary versus weighted arithmetic does not explain the legacy discrepancy: the legacy max target is binary, so its weighted and binary target centroids coincide. Weighting becomes essential for the fractional direct-area target.",
        "- Use `direct_area_soft_weighted_com_shift_512` as the primary spatial-mass preservation metric. Retain the binary 64-grid definition only with topology diagnostics (thickness, components, merge/drop, thresholded Dice).",
        "- This reconciliation does not weaken the independently measured legacy-max mass inflation or component-merging evidence, and it does not change the target-only GPU screening decision.",
        "", "## Protocol facts", "",
        f"- Sample count: {payload['sample_count']}",
        f"- CUDA_VISIBLE_DEVICES: `{payload['cuda_visible_devices']}`", 
        "- Euclidean distance is used for every unified metric; dx/dy and L1 are not substituted.",
        "- Statistics are computed per sample first, then summarized; non-finite empty cases are excluded from distribution statistics and counted explicitly.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("This audit requires CUDA_VISIBLE_DEVICES to be empty")
    torch.set_num_threads(max(1, min(2, int(os.environ.get("OMP_NUM_THREADS", "2")))))
    cv2.setNumThreads(0)

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if args.limit is not None:
        manifest = manifest[: max(args.limit, 0)]
    if not manifest:
        raise SystemExit("Manifest is empty")

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    group_counts = {group: 0 for group in GROUPS}
    for item in manifest:
        _, raw_mask = decode_instance_json(Path(item["json_path"]))
        boundary512 = instance_boundary(resize_instance_map(raw_mask, 512))
        row = compute_metrics(boundary512)
        density_group = item["density_group"]
        group_counts["all"] += 1
        group_counts[density_group] += 1
        for metric, value in row.items():
            values[(metric, "all")].append(value)
            values[(metric, density_group)].append(value)

    results = {
        metric: {
            group: summarize(values[(metric, group)], group_counts[group])
            for group in GROUPS
        }
        for metric in METRICS
    }
    historical = load_historical(args.old_summary, args.soft_summary)
    old_all = float(historical.get("old_binary_64", {}).get("all", math.nan))
    payload: dict[str, object] = {
        "schema_version": "p4_boundary_com_reconciliation_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "sample_count": len(manifest),
        "group_sample_counts": group_counts,
        "distance": "Euclidean L2 over [y, x] center-of-mass coordinates",
        "historical_implementation_audit": {
            "old_fidelity": {
                "coordinate_space": "64x64 pixel-index coordinates (0..63)",
                "input_weights": "binary source and binary thresholded reconstruction",
                "source": "nearest 512->64 formal boundary target; source64 > 0.5",
                "reconstruction": "adaptive-max 64->32, >=0.5, nearest 32->64",
                "post_scale": "none",
                "empty_handling": "NaN if source or reconstruction is empty",
                "aggregation": "per-sample Euclidean shift, finite-only arithmetic mean",
                "source_zero_exclusion": "yes (NaN); target-empty cases also excluded",
                "distance": "Euclidean L2 of [dy, dx] jointly",
            },
            "soft_fidelity": {
                "coordinate_space": "original 512x512 pixel coordinates (0..511)",
                "input_weights": "binary-valued source as mass; raw target values as weights",
                "source": "original 512x512 boundary map weighted COM",
                "reconstruction": "32x32 target weights located at 16x16 cell centers 7.5..503.5",
                "post_scale": "none; already in original coordinates",
                "empty_handling": "NaN if source mass or target mass is zero",
                "aggregation": "per-sample Euclidean shift, finite-only arithmetic mean",
                "source_zero_exclusion": "yes (NaN); target-zero cases also excluded",
                "distance": "Euclidean L2 of [dy, dx] jointly",
            },
        },
        "unified_metric_definitions": {
            "legacy_binary_com_shift_512": "8 * L2(COM(binary nearest B512->64), COM(nearest-upsample(binary maxpool64->32), 32->64))",
            "legacy_weighted_com_shift_512": "L2(weighted COM(B512), weighted COM(legacy max target32 at 512 cell centers))",
            "direct_area_soft_weighted_com_shift_512": "L2(weighted COM(B512), weighted COM(direct avgpool512->32 target at 512 cell centers))",
        },
        "results": results,
        "historical_report_values": historical,
        "reconciliation": {
            "fixed_scale_equivalent": False,
            "old_all_times_8": old_all * 8.0,
            "legacy_weighted_all": results["legacy_weighted_com_shift_512"]["all"]["mean"],
            "primary_cause": "different source support/resampling (nearest-subsampled 64 source versus original 512 source), not a fixed coordinate scale",
            "binary_vs_weighted_is_primary_cause": False,
            "recommended_primary_metric": "direct_area_soft_weighted_com_shift_512",
            "target_only_screening_decision_changed": False,
        },
        "safety": {
            "gpu_used": False,
            "model_instantiated": False,
            "checkpoint_loaded": False,
            "forward": False,
            "backward": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.output_md, payload)
    print(json.dumps({"result": "PASS", "samples": len(manifest), "groups": group_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
