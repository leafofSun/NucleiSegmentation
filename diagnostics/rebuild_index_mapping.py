#!/usr/bin/env python3
"""Rebuild the converted PanNuke test-to-Fold3 index mapping on CPU.

The primary gate is SHA256 over the decoded RGB uint8 pixel array.  This file
also exposes the small raw-Parquet helpers used by the other D1 diagnostics.
It never imports a model and has no training or inference code path.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from bisect import bisect_right
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TISSUE_NAMES = (
    "Adrenal Gland",
    "Bile Duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "Head & Neck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(rgb: np.ndarray) -> str:
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.shape != (256, 256, 3):
        raise ValueError(f"unexpected RGB shape: {array.shape}")
    return hashlib.sha256(array.tobytes()).hexdigest()


def decode_rgb_png(blob: bytes) -> np.ndarray:
    with Image.open(BytesIO(blob)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def decode_binary_png(blob: bytes) -> np.ndarray:
    with Image.open(BytesIO(blob)) as image:
        return np.asarray(image.convert("1"), dtype=np.uint8) > 0


def load_rgb_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to read the retrieved Fold3 Parquet mirror; "
            "set PYTHONPATH to the isolated D1 dependency directory"
        ) from exc
    return parquet


@dataclass(frozen=True)
class RawRecord:
    index: int
    image_bytes: bytes
    instance_bytes: tuple[bytes, ...]
    categories: tuple[int, ...]
    tissue_id: int

    @property
    def tissue_name(self) -> str:
        if not 0 <= self.tissue_id < len(TISSUE_NAMES):
            return f"UNVERIFIED_TISSUE_{self.tissue_id}"
        return TISSUE_NAMES[self.tissue_id]

    def rgb(self) -> np.ndarray:
        return decode_rgb_png(self.image_bytes)

    def instance_masks(self) -> list[np.ndarray]:
        return [decode_binary_png(blob) for blob in self.instance_bytes]


class RawFoldParquet:
    """In-memory view of the retrieved, fixed-version Fold3 Parquet file."""

    def __init__(self, path: Path):
        self.path = path
        parquet = _pyarrow_parquet()
        self._table = parquet.read_table(
            path, columns=["image", "instances", "categories", "tissue"]
        )
        expected = {"image", "instances", "categories", "tissue"}
        if set(self._table.column_names) != expected:
            raise ValueError(f"unexpected Parquet columns: {self._table.column_names}")

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int) -> RawRecord:
        image = self._table["image"][index].as_py()
        instances = self._table["instances"][index].as_py()
        categories = tuple(int(value) for value in self._table["categories"][index].as_py())
        instance_bytes = tuple(item["bytes"] for item in instances)
        if len(instance_bytes) != len(categories):
            raise ValueError(
                f"raw row {index} instance/category mismatch: "
                f"{len(instance_bytes)} != {len(categories)}"
            )
        return RawRecord(
            index=index,
            image_bytes=image["bytes"],
            instance_bytes=instance_bytes,
            categories=categories,
            tissue_id=int(self._table["tissue"][index].as_py()),
        )

    def __iter__(self) -> Iterator[RawRecord]:
        for index in range(len(self)):
            yield self[index]


class StreamingRawFoldParquet:
    """Row-group-cached view for memory-constrained CPU audit environments."""

    def __init__(self, path: Path):
        self.path = path
        parquet = _pyarrow_parquet()
        self._file = parquet.ParquetFile(path)
        self._row_group_starts = [0]
        for group_index in range(self._file.num_row_groups):
            rows = self._file.metadata.row_group(group_index).num_rows
            self._row_group_starts.append(self._row_group_starts[-1] + rows)
        self._cached_group_index: int | None = None
        self._cached_group: Any | None = None

    def __len__(self) -> int:
        return self._row_group_starts[-1]

    def __getitem__(self, index: int) -> RawRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group_index = bisect_right(self._row_group_starts, index) - 1
        if group_index != self._cached_group_index:
            self._cached_group = self._file.read_row_group(
                group_index,
                columns=["image", "instances", "categories", "tissue"],
            )
            self._cached_group_index = group_index
        assert self._cached_group is not None
        local_index = index - self._row_group_starts[group_index]
        image = self._cached_group["image"][local_index].as_py()
        instances = self._cached_group["instances"][local_index].as_py()
        categories = tuple(
            int(value)
            for value in self._cached_group["categories"][local_index].as_py()
        )
        instance_bytes = tuple(item["bytes"] for item in instances)
        if len(instance_bytes) != len(categories):
            raise ValueError(
                f"raw row {index} instance/category mismatch: "
                f"{len(instance_bytes)} != {len(categories)}"
            )
        return RawRecord(
            index=index,
            image_bytes=image["bytes"],
            instance_bytes=instance_bytes,
            categories=categories,
            tissue_id=int(self._cached_group["tissue"][local_index].as_py()),
        )


def load_mapping(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("index_mapping.json must be a top-level list")
    required = {"sample_id", "raw_index", "pixel_sha256", "max_abs_diff"}
    for entry in value:
        if not isinstance(entry, dict) or not required <= set(entry):
            raise ValueError(f"invalid mapping entry: {entry}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-converted-count", type=int, default=2607)
    parser.add_argument("--minimum-match-rate", type=float, default=0.95)
    parser.add_argument("--source-url", default="UNVERIFIED")
    parser.add_argument("--retrieval-url", default="UNVERIFIED")
    parser.add_argument("--source-revision", default="UNVERIFIED")
    parser.add_argument("--source-linked-etag", default="UNVERIFIED")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.raw_parquet.is_file():
        raise FileNotFoundError(args.raw_parquet)
    if not args.converted_dir.is_dir():
        raise FileNotFoundError(args.converted_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_fold = RawFoldParquet(args.raw_parquet)
    raw_by_hash: dict[str, int] = {}
    raw_duplicates: list[dict[str, Any]] = []
    for record in raw_fold:
        pixel_hash = sha256_pixels(record.rgb())
        if pixel_hash in raw_by_hash:
            raw_duplicates.append(
                {
                    "pixel_sha256": pixel_hash,
                    "first_raw_index": raw_by_hash[pixel_hash],
                    "second_raw_index": record.index,
                }
            )
        raw_by_hash[pixel_hash] = record.index

    converted_paths = sorted(args.converted_dir.glob("*.png"))
    if len(converted_paths) != args.expected_converted_count:
        raise RuntimeError(
            f"converted PNG count {len(converted_paths)} != {args.expected_converted_count}"
        )
    converted_by_hash: dict[str, str] = {}
    converted_duplicates: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for path in converted_paths:
        rgb = load_rgb_path(path)
        pixel_hash = sha256_pixels(rgb)
        if pixel_hash in converted_by_hash:
            converted_duplicates.append(
                {
                    "pixel_sha256": pixel_hash,
                    "first_sample_id": converted_by_hash[pixel_hash],
                    "second_sample_id": path.stem,
                }
            )
        converted_by_hash[pixel_hash] = path.stem
        raw_index = raw_by_hash.get(pixel_hash)
        if raw_index is None:
            unmatched.append(
                {
                    "sample_id": path.stem,
                    "pixel_sha256": pixel_hash,
                    "reason": "NO_EXACT_PIXEL_HASH_MATCH",
                    "max_abs_diff": "UNVERIFIED",
                }
            )
            continue
        mapping.append(
            {
                "sample_id": path.stem,
                "raw_index": raw_index,
                "pixel_sha256": pixel_hash,
                "max_abs_diff": 0,
            }
        )

    mapping.sort(key=lambda item: item["sample_id"])
    match_rate = len(mapping) / len(converted_paths) if converted_paths else 0.0
    used_raw = {int(item["raw_index"]) for item in mapping}
    summary = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "raw_parquet": str(args.raw_parquet.resolve()),
        "raw_parquet_sha256": sha256_file(args.raw_parquet),
        "source_url": args.source_url,
        "retrieval_url": args.retrieval_url,
        "source_revision": args.source_revision,
        "source_linked_etag": args.source_linked_etag,
        "raw_sample_count": len(raw_fold),
        "converted_dir": str(args.converted_dir.resolve()),
        "converted_sample_count": len(converted_paths),
        "matched_count": len(mapping),
        "match_rate": match_rate,
        "pixel_exact": not unmatched,
        "raw_duplicate_count": len(raw_duplicates),
        "converted_duplicate_count": len(converted_duplicates),
        "unmatched": unmatched,
        "unused_raw_indices": sorted(set(range(len(raw_fold))) - used_raw),
        "minimum_match_rate_gate": args.minimum_match_rate,
        "gate_pass": (
            match_rate >= args.minimum_match_rate
            and not unmatched
            and not raw_duplicates
            and not converted_duplicates
        ),
        "source_schema": {
            "image": "RGB PNG bytes decoded to uint8[256,256,3]",
            "instances": "list of one-bit PNG masks, one per original instance",
            "categories": "parallel list of original five-class integer labels",
            "tissue": "19-class integer label",
        },
    }
    config = {
        "raw_data_path": str(args.raw_parquet.resolve()),
        "raw_data_sha256": summary["raw_parquet_sha256"],
        "converted_data_path": str(args.converted_dir.resolve()),
        "converted_png_count": len(converted_paths),
        "matched_count": len(mapping),
        "minimum_match_rate": args.minimum_match_rate,
        "training_started": False,
        "inference_rerun": False,
    }
    print("[DIAG_CONFIG] " + json.dumps(config, sort_keys=True), flush=True)
    write_json(args.output_dir / "index_mapping.json", mapping)
    write_json(args.output_dir / "mapping_summary.json", summary)
    write_json(
        args.output_dir / "source_provenance.json",
        {
            "source_url": args.source_url,
            "retrieval_url": args.retrieval_url,
            "source_revision": args.source_revision,
            "source_linked_etag": args.source_linked_etag,
            "downloaded_file": str(args.raw_parquet.resolve()),
            "downloaded_file_sha256": summary["raw_parquet_sha256"],
            "downloaded_file_size_bytes": args.raw_parquet.stat().st_size,
            "raw_sample_count": len(raw_fold),
        },
    )
    write_json(args.output_dir / "raw_duplicate_hashes.json", raw_duplicates)
    write_json(args.output_dir / "converted_duplicate_hashes.json", converted_duplicates)
    print("[MAPPING_RESULT] " + json.dumps(summary, sort_keys=True), flush=True)
    if not summary["gate_pass"]:
        print("[STOP_GATE] image mapping/pixel identity gate failed", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
