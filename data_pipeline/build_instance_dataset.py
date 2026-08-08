#!/usr/bin/env python3
"""Build a lossless, per-instance PanNuke dataset from fixed Parquet folds.

This is a CPU-only data conversion tool.  It has no model, inference, or
training code path and refuses to overwrite a non-empty output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.rebuild_index_mapping import (  # noqa: E402
    StreamingRawFoldParquet,
    sha256_file,
    sha256_pixels,
)


EXPECTED_SAMPLES = {1: 2656, 2: 2523, 3: 2722}
EXPECTED_INSTANCES = {1: 63218, 2: 59872, 3: 66654}
FORMAT_VERSION = "pannuke-instance-npz-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def build_maps(masks: list[np.ndarray], categories: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if len(masks) != len(categories):
        raise ValueError("instance/category length mismatch")
    inst_map = np.zeros((256, 256), dtype=np.int32)
    type_map = np.zeros((256, 256), dtype=np.uint8)
    for instance_id, (mask, category) in enumerate(zip(masks, categories, strict=True), 1):
        if mask.shape != (256, 256) or mask.dtype != np.bool_:
            mask = np.asarray(mask, dtype=bool)
        if not 0 <= int(category) < 5:
            raise ValueError(f"category outside 0..4: {category}")
        # Official PanNuke channel merge semantics: source order is category
        # channel 0..4 then ascending source ID; later instances overwrite.
        inst_map[mask] = instance_id
        type_map[mask] = int(category) + 1
    return inst_map, type_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for fold in (1, 2, 3):
        parser.add_argument(f"--fold{fold}-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {fold: getattr(args, f"fold{fold}_parquet") for fold in (1, 2, 3)}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "source_revision": args.source_revision,
        "merge_rule": (
            "source list order (category channel 0..4, ascending source ID); "
            "later instances overwrite overlap pixels"
        ),
        "split_rule": {"train": [1, 2], "test": [3]},
        "storage": {
            "image": "uint8[256,256,3] RGB",
            "inst_map": "int32[256,256], background=0, IDs=1..N",
            "type_map": "uint8[256,256], background=0, types=1..5",
            "inst_type": "int32[N], types=1..5 in instance-ID order",
        },
        "folds": {},
        "samples": [],
    }
    sha_lines: list[str] = []
    global_categories: Counter[int] = Counter()
    global_tissues: Counter[str] = Counter()

    for fold, source in sources.items():
        raw = StreamingRawFoldParquet(source)
        if len(raw) != EXPECTED_SAMPLES[fold]:
            raise RuntimeError(f"fold {fold} sample count {len(raw)} != {EXPECTED_SAMPLES[fold]}")
        fold_dir = args.output_dir / f"fold{fold}"
        fold_dir.mkdir()
        fold_instances = 0
        fold_empty = 0
        fold_categories: Counter[int] = Counter()
        fold_tissues: Counter[str] = Counter()
        for record in raw:
            image = record.rgb()
            masks = record.instance_masks()
            inst_map, type_map = build_maps(masks, record.categories)
            instance_count = len(masks)
            expected_ids = np.arange(1, instance_count + 1, dtype=np.int32)
            actual_ids = np.unique(inst_map)
            actual_ids = actual_ids[actual_ids > 0]
            if not np.array_equal(actual_ids, expected_ids):
                raise RuntimeError(
                    f"fold {fold} row {record.index}: an instance vanished after overlap resolution; "
                    f"actual={actual_ids.tolist()} expected={expected_ids.tolist()}"
                )
            foreground = np.zeros((256, 256), dtype=bool)
            for mask in masks:
                foreground |= mask
            if not np.array_equal(inst_map > 0, foreground):
                raise RuntimeError(f"fold {fold} row {record.index}: foreground identity failed")

            relative = Path(f"fold{fold}") / f"fold{fold}_{record.index:07d}.npz"
            output = args.output_dir / relative
            payload = {
                "image": image,
                "inst_map": inst_map,
                "type_map": type_map,
                "inst_type": np.asarray(record.categories, dtype=np.int32) + 1,
                "tissue_id": np.asarray(record.tissue_id, dtype=np.int32),
                "tissue_name": np.asarray(record.tissue_name),
                "fold": np.asarray(fold, dtype=np.int32),
                "orig_index": np.asarray(record.index, dtype=np.int32),
            }
            if args.compressed:
                np.savez_compressed(output, **payload)
            else:
                np.savez(output, **payload)
            file_sha = sha256_file(output)
            image_sha = sha256_pixels(image)
            sha_lines.append(f"{file_sha}  {relative.as_posix()}")
            entry = {
                "sample_id": f"fold{fold}_{record.index:07d}",
                "relative_path": relative.as_posix(),
                "fold": fold,
                "orig_index": record.index,
                "split": "train" if fold in (1, 2) else "test",
                "instance_count": instance_count,
                "empty": instance_count == 0,
                "tissue_id": record.tissue_id,
                "tissue_name": record.tissue_name,
                "image_sha256": image_sha,
                "file_sha256": file_sha,
                "file_size": output.stat().st_size,
            }
            manifest["samples"].append(entry)
            fold_instances += instance_count
            fold_empty += int(instance_count == 0)
            fold_categories.update(int(value) + 1 for value in record.categories)
            fold_tissues[record.tissue_name] += 1
            if (record.index + 1) % 250 == 0 or record.index + 1 == len(raw):
                print(
                    f"[BUILD_PROGRESS] fold={fold} samples={record.index + 1}/{len(raw)} "
                    f"instances={fold_instances}",
                    flush=True,
                )

        if fold_instances != EXPECTED_INSTANCES[fold]:
            raise RuntimeError(
                f"fold {fold} instance count {fold_instances} != {EXPECTED_INSTANCES[fold]}"
            )
        manifest["folds"][str(fold)] = {
            "sample_count": len(raw),
            "instance_count": fold_instances,
            "empty_sample_count": fold_empty,
            "category_distribution": {str(k): fold_categories[k] for k in range(1, 6)},
            "tissue_distribution": dict(sorted(fold_tissues.items())),
            "source_path": str(source),
            "source_sha256": sha256_file(source),
        }
        global_categories.update(fold_categories)
        global_tissues.update(fold_tissues)

    manifest["totals"] = {
        "sample_count": sum(EXPECTED_SAMPLES.values()),
        "instance_count": sum(EXPECTED_INSTANCES.values()),
        "empty_sample_count": sum(value["empty_sample_count"] for value in manifest["folds"].values()),
        "category_distribution": {str(k): global_categories[k] for k in range(1, 6)},
        "tissue_distribution": dict(sorted(global_tissues.items())),
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha = sha256_file(manifest_path)
    sha_lines.append(f"{manifest_sha}  dataset_manifest.json")
    sums_path = args.output_dir / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(
        "[BUILD_RESULT] "
        + json.dumps(
            {
                "training_started": False,
                "inference_started": False,
                "sample_count": manifest["totals"]["sample_count"],
                "instance_count": manifest["totals"]["instance_count"],
                "manifest_sha256": manifest_sha,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
