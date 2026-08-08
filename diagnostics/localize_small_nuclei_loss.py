#!/usr/bin/env python3
"""R0: localize small-nucleus loss using saved artifacts only (CPU-only).

When no raw response tensors are present this tool deliberately degrades to
final-instance coverage.  It never imports a model and cannot run inference or
training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.quantify_conversion_loss import build_instance_map, overlap_matrices  # noqa: E402
from diagnostics.rebuild_index_mapping import (  # noqa: E402
    StreamingRawFoldParquet,
    load_mapping,
    load_rgb_path,
    sha256_file,
    sha256_pixels,
)
from evaluation.recompute_from_npy import load_gt_like_test_py  # noqa: E402


TRAIN_BINS = (
    (1, 10, "[1,10)"),
    (10, 20, "[10,20)"),
    (20, 50, "[20,50)"),
    (50, 100, "[50,100)"),
    (100, 200, "[100,200)"),
    (200, math.inf, "[200,+inf)"),
)
RESPONSE_BINS = (
    (10, 50, "[10,50)"),
    (50, 100, "[50,100)"),
    (200, math.inf, "[200,+inf)"),
)
TENSOR_SUFFIXES = {".npy", ".npz", ".pt", ".pth"}
RAW_KEYWORDS = ("prob", "logit", "heat", "marker", "hv", "raw")


def bin_name(area: int, bins: tuple[tuple[int, float, str], ...]) -> str | None:
    return next((name for low, high, name in bins if low <= area < high), None)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tree_manifest_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\t{path.stat().st_size}\n".encode()
        )
    return digest.hexdigest()


def inventory_raw(roots: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    raw_candidates: list[str] = []
    final_instance_files: list[Path] = []
    for root in roots:
        if not root.exists():
            records.append({"root": str(root), "status": "NOT_FOUND", "tensor_file_count": 0})
            continue
        tensors = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TENSOR_SUFFIXES)
        suffixes = Counter(p.suffix.lower() for p in tensors)
        # Any non-final tensor is conservatively treated as a raw candidate,
        # even when its filename does not contain an expected semantic keyword.
        candidates = [p for p in tensors if not p.stem.endswith("_inst")]
        semantic_candidates = [
            p for p in candidates if any(keyword in p.name.lower() for keyword in RAW_KEYWORDS)
        ]
        finals = [p for p in tensors if p.stem.endswith("_inst")]
        final_instance_files.extend(finals)
        raw_candidates.extend(str(p) for p in candidates)
        records.append(
            {
                "root": str(root),
                "status": "FOUND",
                "tensor_file_count": len(tensors),
                "suffix_counts": dict(sorted(suffixes.items())),
                "raw_candidate_count": len(candidates),
                "semantic_raw_candidate_count": len(semantic_candidates),
                "final_instance_count": len(finals),
            }
        )
    return {
        "status": "RAW_AVAILABLE" if raw_candidates else "RAW_NOT_AVAILABLE",
        "roots": records,
        "raw_candidates": raw_candidates,
        "final_instance_count": len(final_instance_files),
        "final_instance_manifest_sha256": (
            tree_manifest_sha256(final_instance_files, Path("/")) if final_instance_files else "NOT_FOUND"
        ),
    }


def training_distribution(
    raw_paths: list[Path], legacy_train_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pngs = sorted(legacy_train_dir.glob("*.png"))
    jsons = sorted(legacy_train_dir.glob("*.json"))
    if not pngs or len(pngs) != len(jsons):
        raise RuntimeError(f"legacy train PNG/JSON mismatch: {len(pngs)} vs {len(jsons)}")
    legacy_by_hash: dict[str, str] = {}
    for path in pngs:
        digest = sha256_pixels(load_rgb_path(path))
        if digest in legacy_by_hash:
            raise RuntimeError(f"duplicate legacy training image hash: {path} and {legacy_by_hash[digest]}")
        legacy_by_hash[digest] = path.stem

    raw_counts: Counter[str] = Counter()
    retained_raw_counts: Counter[str] = Counter()
    current_counts: Counter[str] = Counter()
    matched_samples = 0
    unmatched_nonempty = 0
    max_abs_diff = 0
    matched_stems: set[str] = set()
    original_instances = 0
    for fold, raw_path in enumerate(raw_paths, 1):
        raw = StreamingRawFoldParquet(raw_path)
        for record in raw:
            image = record.rgb()
            stem = legacy_by_hash.get(sha256_pixels(image))
            masks = record.instance_masks()
            areas = [int(mask.sum()) for mask in masks]
            original_instances += len(masks)
            for area in areas:
                name = bin_name(area, TRAIN_BINS)
                if name is not None:
                    raw_counts[name] += 1
            if stem is None:
                unmatched_nonempty += int(bool(masks))
                continue
            matched_samples += 1
            matched_stems.add(stem)
            legacy_image = load_rgb_path(legacy_train_dir / f"{stem}.png")
            max_abs_diff = max(
                max_abs_diff,
                int(np.max(np.abs(legacy_image.astype(np.int16) - image.astype(np.int16)))),
            )
            reconstructed, _ = load_gt_like_test_py(legacy_train_dir / f"{stem}.json")
            current_areas = np.bincount(reconstructed.ravel())[1:]
            for area in current_areas:
                name = bin_name(int(area), TRAIN_BINS)
                if name is not None:
                    current_counts[name] += 1
            intersections, iou, _, _ = overlap_matrices(masks, reconstructed)
            if iou.size:
                raw_indices, converted_indices = linear_sum_assignment(-iou)
                for raw_index, converted_index in zip(raw_indices, converted_indices, strict=True):
                    if intersections[raw_index, converted_index] > 0:
                        name = bin_name(areas[int(raw_index)], TRAIN_BINS)
                        if name is not None:
                            retained_raw_counts[name] += 1
        print(f"[R0_TRAIN_PROGRESS] fold={fold} samples={len(raw)}", flush=True)
    if matched_stems != set(legacy_by_hash.values()):
        missing = sorted(set(legacy_by_hash.values()) - matched_stems)
        raise RuntimeError(f"{len(missing)} legacy train images did not map to Fold1/2")

    rows: list[dict[str, Any]] = []
    for _, _, name in TRAIN_BINS:
        raw_count = raw_counts[name]
        current_count = current_counts[name]
        retained_count = retained_raw_counts[name]
        rows.append(
            {
                "area_bin": name,
                "original_fold12_instance_count": raw_count,
                "current_training_gt_instance_count_by_current_area": current_count,
                "distribution_count_ratio": current_count / raw_count if raw_count else float("nan"),
                "retained_original_instance_count_unique_overlap": retained_count,
                "retention_rate_unique_overlap": retained_count / raw_count if raw_count else float("nan"),
            }
        )
    return rows, {
        "legacy_train_sample_count": len(pngs),
        "matched_legacy_train_sample_count": matched_samples,
        "raw_fold12_sample_count": sum(len(StreamingRawFoldParquet(p)) for p in raw_paths),
        "raw_fold12_instance_count": original_instances,
        "unmatched_raw_nonempty_sample_count": unmatched_nonempty,
        "image_max_abs_diff": max_abs_diff,
        "legacy_train_input_manifest_sha256": tree_manifest_sha256(pngs + jsons, legacy_train_dir),
    }


def final_response_coverage(
    raw_fold3_path: Path,
    mapping_path: Path,
    prediction_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    by_raw = {int(entry["raw_index"]): str(entry["sample_id"]) for entry in mapping}
    raw = StreamingRawFoldParquet(raw_fold3_path)
    detail: list[dict[str, Any]] = []
    aggregates: dict[str, Counter[str]] = defaultdict(Counter)
    for raw_index, sample_id in sorted(by_raw.items()):
        record = raw[raw_index]
        prediction_path = prediction_dir / f"{sample_id}_inst.npy"
        prediction = np.load(prediction_path, allow_pickle=False).astype(np.int32, copy=False)
        if prediction.shape != (256, 256):
            raise RuntimeError(f"unexpected prediction shape: {prediction_path}")
        pred_areas = np.bincount(prediction.ravel())
        for source_id, mask in enumerate(record.instance_masks(), 1):
            area = int(mask.sum())
            group = bin_name(area, RESPONSE_BINS)
            if group is None:
                continue
            overlaps = np.bincount(prediction[mask], minlength=len(pred_areas))
            overlaps[0] = 0
            pred_id = int(np.argmax(overlaps)) if overlaps.size else 0
            overlap_pixels = int(overlaps[pred_id]) if pred_id else 0
            pred_area = int(pred_areas[pred_id]) if pred_id else 0
            union = area + pred_area - overlap_pixels
            best_iou = overlap_pixels / union if union else 0.0
            covered = overlap_pixels > 0
            independent = best_iou > 0.5
            detail.append(
                {
                    "sample_id": sample_id,
                    "raw_index": raw_index,
                    "source_instance_id": source_id,
                    "area": area,
                    "area_bin": group,
                    "any_prediction_overlap": int(covered),
                    "overlap_pixels": overlap_pixels,
                    "overlap_fraction_of_gt": overlap_pixels / area,
                    "best_prediction_id": pred_id,
                    "best_iou": best_iou,
                    "independent_instance_iou_gt_0_5": int(independent),
                }
            )
            aggregates[group]["count"] += 1
            aggregates[group]["covered"] += int(covered)
            aggregates[group]["independent"] += int(independent)
            aggregates[group]["overlap_pixels"] += overlap_pixels
            aggregates[group]["gt_pixels"] += area
    rows = []
    for _, _, group in RESPONSE_BINS:
        values = aggregates[group]
        rows.append(
            {
                "response_source": "final_instance_map_degraded_RAW_NOT_AVAILABLE",
                "area_bin": group,
                "gt_instance_count": values["count"],
                "any_prediction_overlap_count": values["covered"],
                "coverage_rate": values["covered"] / values["count"] if values["count"] else float("nan"),
                "independent_instance_count_iou_gt_0_5": values["independent"],
                "independent_instance_rate": values["independent"] / values["count"] if values["count"] else float("nan"),
                "gt_pixel_coverage_rate": values["overlap_pixels"] / values["gt_pixels"] if values["gt_pixels"] else float("nan"),
                "prob_mean_median": "UNVERIFIED",
                "prob_max_median": "UNVERIFIED",
                "marker_max_median": "UNVERIFIED",
            }
        )
    prediction_paths = sorted(prediction_dir.glob("*_inst.npy"))
    return detail, rows, {
        "mapped_sample_count": len(mapping),
        "unmapped_fold3_sample_count": len(raw) - len(mapping),
        "prediction_count": len(prediction_paths),
        "prediction_manifest_sha256": tree_manifest_sha256(prediction_paths, prediction_dir),
    }


def write_sha256s(output_dir: Path) -> None:
    sums = output_dir / "SHA256SUMS.txt"
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != sums
    ]
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-fold1", type=Path, required=True)
    parser.add_argument("--raw-fold2", type=Path, required=True)
    parser.add_argument("--raw-fold3", type=Path, required=True)
    parser.add_argument("--legacy-train-dir", type=Path, required=True)
    parser.add_argument("--fold3-mapping", type=Path, required=True)
    parser.add_argument("--visual-pred-dir", type=Path, required=True)
    parser.add_argument("--raw-search-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [args.raw_fold1, args.raw_fold2, args.raw_fold3, args.fold3_mapping]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.legacy_train_dir, args.visual_pred_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_raw(args.raw_search_root)
    if inventory["status"] != "RAW_NOT_AVAILABLE":
        raise RuntimeError(
            "raw response candidates were found; this pre-registered implementation refuses "
            "to guess tensor semantics. Inspect raw_output_inventory.json before a sweep."
        )
    (args.output_dir / "raw_output_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train_rows, train_meta = training_distribution(
        [args.raw_fold1, args.raw_fold2], args.legacy_train_dir
    )
    write_csv(args.output_dir / "train_gt_size_distribution.csv", train_rows)
    detail, response_rows, response_meta = final_response_coverage(
        args.raw_fold3, args.fold3_mapping, args.visual_pred_dir
    )
    write_csv(args.output_dir / "small_nuclei_final_coverage.csv", detail)
    write_csv(args.output_dir / "small_nuclei_response_stats.csv", response_rows)

    first_two = {row["area_bin"]: row for row in train_rows}
    a_confirmed = (
        float(first_two["[1,10)"]["retention_rate_unique_overlap"]) < 0.01
        and float(first_two["[10,20)"]["retention_rate_unique_overlap"]) < 0.10
    )
    summary = {
        "status": "COMPLETE_DEGRADED_RAW_NOT_AVAILABLE",
        "training_started": False,
        "inference_started": False,
        "postprocessing_started": False,
        "raw_status": inventory["status"],
        "attribution": {
            "A_training_signal_loss": {
                "decision": "CONFIRMED" if a_confirmed else "REJECTED",
                "evidence": "Fold1+2 original-to-legacy unique-overlap retention by raw area",
            },
            "B_model_nondetection": {
                "decision": "UNVERIFIED",
                "evidence": "final-map coverage measured, but raw response was not saved",
            },
            "C_postprocessing_deletion": {
                "decision": "UNVERIFIED",
                "evidence": "raw response was not saved; threshold sweep was not run",
            },
        },
        "recoverable_estimate": {
            "postprocessing_only_bpq": "UNVERIFIED",
            "requires_retraining_bpq": "UNVERIFIED",
        },
        "train": train_meta,
        "final_response": response_meta,
        "inputs": {
            "raw_fold1_sha256": sha256_file(args.raw_fold1),
            "raw_fold2_sha256": sha256_file(args.raw_fold2),
            "raw_fold3_sha256": sha256_file(args.raw_fold3),
            "fold3_mapping_sha256": sha256_file(args.fold3_mapping),
        },
        "outputs": {
            "postproc_sweep_results.csv": "NOT_GENERATED_RAW_NOT_AVAILABLE",
            "raw_response_statistics": "UNVERIFIED",
        },
    }
    summary_path = args.output_dir / "r0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_sha256s(args.output_dir)
    print("[R0_RESULT] " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
