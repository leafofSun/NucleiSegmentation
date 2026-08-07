#!/usr/bin/env python3
"""Build an SGA-SB density manifest from PanNuke GT JSON files only.

This utility never imports NuSeg model code.  Its polygon rasterisation and
label resizing follow DataLoader.UniversalDataset's current PanNuke path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/PanNuke"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--boundary-target-size", type=int, default=64)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="CPU smoke only")
    return parser.parse_args()


def iter_json_files(data_root: Path, splits: Iterable[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for split in splits:
        files.extend((split, path) for path in sorted((data_root / split).glob("*.json")))
    return files


def decode_instance_json(path: Path) -> tuple[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    image_meta = payload.get("image", {}) if isinstance(payload.get("image"), dict) else {}
    height = int(image_meta.get("height", payload.get("height", 256)))
    width = int(image_meta.get("width", payload.get("width", 256)))
    mask = np.zeros((height, width), dtype=np.int32)
    instance_id = 1
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            polygons = (
                [segmentation]
                if all(isinstance(value, (int, float)) for value in segmentation)
                else segmentation
            )
            for polygon in polygons:
                try:
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                except (TypeError, ValueError):
                    continue
                if points.shape[0] < 3:
                    continue
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], int(instance_id))
                instance_id += 1
        elif isinstance(segmentation, dict) and {"counts", "size"} <= set(segmentation):
            try:
                from pycocotools import mask as coco_mask  # type: ignore

                binary = coco_mask.decode(segmentation)
                if binary.ndim == 3:
                    binary = binary.max(axis=2)
                if binary.shape != mask.shape:
                    binary = cv2.resize(
                        binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                    )
                mask[binary > 0] = instance_id
                instance_id += 1
            except (ImportError, TypeError, ValueError):
                continue
    sample_id = str(payload.get("image_id") or path.stem)
    return sample_id, mask


def resize_instance_map(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def instance_boundary(mask: np.ndarray) -> np.ndarray:
    """Exact union of per-instance ``mask - 3x3 erosion(mask)``.

    Edge padding reproduces max_pool2d's ignored outside-of-image padding for
    erosion: touching the image edge alone does not create a boundary.
    """
    padded = np.pad(mask, 1, mode="edge")
    neighbors = [
        padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    local_min = np.minimum.reduce(neighbors)
    local_max = np.maximum.reduce(neighbors)
    interior = (mask > 0) & (local_min == mask) & (local_max == mask)
    return ((mask > 0) & ~interior).astype(np.uint8)


def official_boundary64(mask: np.ndarray, image_size: int = 512, target_size: int = 64) -> np.ndarray:
    resized = resize_instance_map(mask, image_size)
    boundary = instance_boundary(resized)
    return cv2.resize(boundary, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


def sample_record(split: str, path: Path, image_size: int, target_size: int) -> dict[str, object]:
    sample_id, raw_mask = decode_instance_json(path)
    ids = np.unique(raw_mask)
    instance_count = int(np.count_nonzero(ids > 0))
    foreground_ratio = float(np.mean(raw_mask > 0))
    boundary_ratio = float(np.mean(official_boundary64(raw_mask, image_size, target_size) > 0))
    # All PanNuke samples have the same field of view.  Instances per 10k raw
    # pixels is therefore a transparent density/crowding proxy.
    crowding_proxy = float(instance_count * 10000.0 / raw_mask.size)
    return {
        "sample_id": sample_id,
        "split": split,
        "json_path": path.as_posix(),
        "instance_count": instance_count,
        "foreground_ratio": foreground_ratio,
        "boundary_positive_ratio": boundary_ratio,
        "crowding_proxy": crowding_proxy,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    cv2.setNumThreads(0)
    files = iter_json_files(args.data_root, args.splits)
    if args.limit is not None:
        files = files[: max(args.limit, 0)]
    if not files:
        raise SystemExit(f"No GT JSON files found under {args.data_root} for {args.splits}")

    records = [
        sample_record(split, path, args.image_size, args.boundary_target_size)
        for split, path in files
    ]
    proxy = np.asarray([float(row["crowding_proxy"]) for row in records], dtype=np.float64)
    q_low, q_high = np.quantile(proxy, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    for row in records:
        value = float(row["crowding_proxy"])
        row["density_group"] = "sparse" if value <= q_low else "medium" if value <= q_high else "dense"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "split", "json_path", "instance_count", "foreground_ratio",
        "boundary_positive_ratio", "crowding_proxy", "density_group",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    groups: dict[str, dict[str, object]] = {}
    for group in ("sparse", "medium", "dense"):
        subset = [row for row in records if row["density_group"] == group]
        groups[group] = {
            "count": len(subset),
            "instance_count_mean": float(np.mean([row["instance_count"] for row in subset])),
            "foreground_ratio_mean": float(np.mean([row["foreground_ratio"] for row in subset])),
            "boundary_positive_ratio_mean": float(
                np.mean([row["boundary_positive_ratio"] for row in subset])
            ),
            "crowding_proxy_min": float(min(row["crowding_proxy"] for row in subset)),
            "crowding_proxy_max": float(max(row["crowding_proxy"] for row in subset)),
        }
    summary = {
        "schema_version": "sga_sb_density_manifest_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "data_root": args.data_root.as_posix(),
        "splits": args.splits,
        "sample_count": len(records),
        "image_size": args.image_size,
        "boundary_target_size": args.boundary_target_size,
        "crowding_proxy_definition": "instance_count * 10000 / raw_image_pixel_count",
        "quantile_method": "numpy linear",
        "quantile_probabilities": [1.0 / 3.0, 2.0 / 3.0],
        "quantile_thresholds": {"sparse_max": float(q_low), "medium_max": float(q_high)},
        "groups": groups,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "samples": len(records), "thresholds": summary["quantile_thresholds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
