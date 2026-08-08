#!/usr/bin/env python3
"""Verify every R1 PanNuke instance-NPZ hard gate on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.build_instance_dataset import (  # noqa: E402
    EXPECTED_INSTANCES,
    EXPECTED_SAMPLES,
    FORMAT_VERSION,
    build_maps,
)
from diagnostics.rebuild_index_mapping import (  # noqa: E402
    StreamingRawFoldParquet,
    load_mapping,
    sha256_file,
    sha256_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    for fold in (1, 2, 3):
        parser.add_argument(f"--fold{fold}-parquet", type=Path, required=True)
    parser.add_argument("--legacy-test-dir", type=Path, required=True)
    parser.add_argument("--fold3-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise RuntimeError(f"unexpected format: {manifest.get('format')}")
    entries = {(int(e["fold"]), int(e["orig_index"])): e for e in manifest["samples"]}
    mapping = load_mapping(args.fold3_mapping)
    mapped_by_raw = {int(e["raw_index"]): e for e in mapping}
    if len(mapping) != 2607 or any(int(e["max_abs_diff"]) != 0 for e in mapping):
        raise RuntimeError("Fold3 legacy mapping hard gate failed")

    gates: dict[str, Any] = {}
    test_image_sha: list[dict[str, Any]] = []
    category_counts: Counter[int] = Counter()
    total_instances = 0
    total_union = 0
    total_intersection = 0
    legacy_max_abs_diff = 0
    legacy_compared = 0
    hole_checks: list[dict[str, Any]] = []

    for fold in (1, 2, 3):
        source = getattr(args, f"fold{fold}_parquet")
        raw = StreamingRawFoldParquet(source)
        if len(raw) != EXPECTED_SAMPLES[fold]:
            raise RuntimeError(f"fold {fold} sample count mismatch")
        fold_instances = 0
        for record in raw:
            key = (fold, record.index)
            if key not in entries:
                raise RuntimeError(f"manifest missing {key}")
            entry = entries[key]
            path = args.dataset_dir / entry["relative_path"]
            if sha256_file(path) != entry["file_sha256"]:
                raise RuntimeError(f"file SHA mismatch: {path}")
            with np.load(path, allow_pickle=False) as value:
                image = value["image"]
                inst_map = value["inst_map"]
                type_map = value["type_map"]
                inst_type = value["inst_type"]
                metadata = {
                    "tissue_id": int(value["tissue_id"]),
                    "tissue_name": str(value["tissue_name"]),
                    "fold": int(value["fold"]),
                    "orig_index": int(value["orig_index"]),
                }
            masks = record.instance_masks()
            expected_inst, expected_type = build_maps(masks, record.categories)
            expected_ids = np.arange(1, len(masks) + 1, dtype=np.int32)
            actual_ids = np.unique(inst_map)
            actual_ids = actual_ids[actual_ids > 0]
            if image.dtype != np.uint8 or image.shape != (256, 256, 3):
                raise RuntimeError(f"image schema failure: {path}")
            if inst_map.dtype != np.int32 or inst_map.shape != (256, 256):
                raise RuntimeError(f"inst_map schema failure: {path}")
            if type_map.dtype != np.uint8 or type_map.shape != (256, 256):
                raise RuntimeError(f"type_map schema failure: {path}")
            if not np.array_equal(actual_ids, expected_ids):
                raise RuntimeError(f"non-continuous/missing IDs: {path}")
            if not np.array_equal(inst_type, np.asarray(record.categories, dtype=np.int32) + 1):
                raise RuntimeError(f"instance category identity failed: {path}")
            if not np.array_equal(inst_map, expected_inst) or not np.array_equal(type_map, expected_type):
                raise RuntimeError(f"pixel map identity failed: {path}")
            if metadata != {
                "tissue_id": record.tissue_id,
                "tissue_name": record.tissue_name,
                "fold": fold,
                "orig_index": record.index,
            }:
                raise RuntimeError(f"metadata identity failed: {path}")
            raw_foreground = np.zeros((256, 256), dtype=bool)
            for mask in masks:
                raw_foreground |= mask
            intersection = int(np.logical_and(raw_foreground, inst_map > 0).sum())
            union = int(np.logical_or(raw_foreground, inst_map > 0).sum())
            total_intersection += intersection
            total_union += union
            fold_instances += len(masks)
            total_instances += len(masks)
            category_counts.update(int(v) + 1 for v in record.categories)

            if fold == 3:
                image_sha = sha256_pixels(image)
                test_image_sha.append({"orig_index": record.index, "image_sha256": image_sha})
                if image_sha != entry["image_sha256"]:
                    raise RuntimeError(f"test image SHA mismatch: {path}")
                expected_legacy = mapped_by_raw.get(record.index)
                if (
                    entry.get("legacy_test_sample_id")
                    != (str(expected_legacy["sample_id"]) if expected_legacy else None)
                    or bool(entry.get("overlaps_legacy_test")) != (expected_legacy is not None)
                    or bool(entry.get("is_new_vs_legacy_test")) != (expected_legacy is None)
                ):
                    raise RuntimeError(f"legacy/new correspondence mismatch: {path}")
                if record.index in mapped_by_raw:
                    sample_id = str(mapped_by_raw[record.index]["sample_id"])
                    legacy_path = args.legacy_test_dir / f"{sample_id}.png"
                    legacy_bgr = cv2.imread(str(legacy_path), cv2.IMREAD_COLOR)
                    if legacy_bgr is None:
                        raise FileNotFoundError(legacy_path)
                    legacy_rgb = cv2.cvtColor(legacy_bgr, cv2.COLOR_BGR2RGB)
                    difference = int(np.max(np.abs(legacy_rgb.astype(np.int16) - image.astype(np.int16))))
                    legacy_max_abs_diff = max(legacy_max_abs_diff, difference)
                    legacy_compared += 1

            if len(hole_checks) < 8:
                for source_id, mask in enumerate(masks, 1):
                    holes = binary_fill_holes(mask) & ~mask
                    if not np.any(holes):
                        continue
                    later_union = np.zeros((256, 256), dtype=bool)
                    for later in masks[source_id:]:
                        later_union |= later
                    checkable = holes & ~later_union
                    if np.any(checkable):
                        retained = bool(np.all(inst_map[checkable] != source_id))
                        hole_checks.append(
                            {
                                "fold": fold,
                                "orig_index": record.index,
                                "instance_id": source_id,
                                "checkable_hole_pixels": int(checkable.sum()),
                                "retained": retained,
                            }
                        )
                        if not retained:
                            raise RuntimeError(f"hole fill regression: {path} instance {source_id}")
                        break
            if (record.index + 1) % 500 == 0:
                print(f"[VERIFY_PROGRESS] fold={fold} samples={record.index + 1}/{len(raw)}", flush=True)
        if fold_instances != EXPECTED_INSTANCES[fold]:
            raise RuntimeError(f"fold {fold} instance total mismatch")

    foreground_iou = 1.0 if total_union == 0 else total_intersection / total_union
    manifest_categories = {int(k): int(v) for k, v in manifest["totals"]["category_distribution"].items()}
    gates = {
        "sample_counts_exact": {str(k): EXPECTED_SAMPLES[k] for k in (1, 2, 3)},
        "instance_counts_exact": {str(k): EXPECTED_INSTANCES[k] for k in (1, 2, 3)},
        "instance_total_exact": total_instances == 189744,
        "legacy_test_compared": legacy_compared,
        "legacy_test_max_abs_diff": legacy_max_abs_diff,
        "new_vs_legacy_test_count": sum(
            int(entry.get("is_new_vs_legacy_test", False))
            for entry in manifest["samples"]
            if int(entry["fold"]) == 3
        ),
        "foreground_iou": foreground_iou,
        "instance_identity_all_samples": True,
        "category_distribution_exact": dict(category_counts) == manifest_categories,
        "continuous_ids_all_samples": True,
        "hole_checks": hole_checks,
        "hole_retention_pass": bool(hole_checks) and all(v["retained"] for v in hole_checks),
    }
    if not (
        gates["instance_total_exact"]
        and legacy_compared == 2607
        and legacy_max_abs_diff == 0
        and gates["new_vs_legacy_test_count"] == 115
        and foreground_iou == 1.0
        and gates["category_distribution_exact"]
        and gates["hole_retention_pass"]
    ):
        raise RuntimeError(f"one or more hard gates failed: {gates}")

    test_sha_path = args.output_dir / "test_image_sha256.json"
    test_sha_path.write_text(json.dumps(test_image_sha, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "training_started": False,
        "inference_started": False,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "gates": gates,
        "test_image_sha256_count": len(test_image_sha),
    }
    summary_path = args.output_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = [
        f"{sha256_file(summary_path)}  verification_summary.json",
        f"{sha256_file(test_sha_path)}  test_image_sha256.json",
    ]
    (args.output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("[VERIFY_RESULT] " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
