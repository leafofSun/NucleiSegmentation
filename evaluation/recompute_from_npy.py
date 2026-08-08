#!/usr/bin/env python3
"""Recompute standard metrics from immutable saved instance-label ``.npy`` files.

This entry point is deliberately CPU-only.  It loads existing predictions and
the JSON ground truth used by ``test.py``; it does not import or construct a
model and contains no inference or training path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics_standard import (  # noqa: E402
    HOVERNET_COMMIT,
    HOVERNET_STATS_UTILS_SHA256,
    PANNuke_METRICS_COMMIT,
    PANNuke_RUN_SHA256,
    PANNuke_UTILS_SHA256,
    aji_kumar_greedy,
    aji_plus,
    binary_dice,
    pq_from_global_counts,
    pq_independent,
    pq_official,
)
from metrics import get_fast_aji as legacy_aji_custom  # noqa: E402
from metrics import get_fast_pq as legacy_pq  # noqa: E402


METHODS = ("visual_baseline", "exp5")
PRED_SUFFIX = "_inst.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_files(root: Path, paths: Iterable[Path]) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        record = {"path": relative, "sha256": file_hash, "size_bytes": size}
        records.append(record)
        digest.update(f"{relative}\t{file_hash}\t{size}\n".encode("utf-8"))
    return digest.hexdigest(), records


def load_gt_like_test_py(json_path: Path) -> tuple[np.ndarray, str]:
    """Decode converted GT exactly as ``test.py:load_filtered_gt`` does."""

    with json_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list) and data:
        data = data[0]
    annotations = data.get("annotations", []) if isinstance(data, dict) else data
    if not annotations:
        raise ValueError(f"no annotations in {json_path}")
    height = data.get("height") if isinstance(data, dict) else None
    width = data.get("width") if isinstance(data, dict) else None
    if height is None or width is None:
        first_segmentation = annotations[0].get("segmentation", {})
        if isinstance(first_segmentation, dict) and "size" in first_segmentation:
            height, width = first_segmentation["size"]
        else:
            height, width = 1000, 1000
    instance_map = np.zeros((int(height), int(width)), dtype=np.int32)
    for index, annotation in enumerate(annotations):
        segmentation = annotation.get("segmentation")
        if not segmentation:
            continue
        if isinstance(segmentation, list):
            for polygon in segmentation:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if points.shape[0] >= 3:
                    points = np.round(points).astype(np.int32)
                    cv2.fillPoly(instance_map, [points], index + 1)
        else:
            raise ValueError(
                f"RLE GT is unsupported without pycocotools and was not expected: {json_path}"
            )
    organ_type = str(data.get("organ_type", "NOT_FOUND"))
    return instance_map, organ_type


def prediction_paths(prediction_dir: Path) -> dict[str, Path]:
    paths = sorted(prediction_dir.glob(f"*{PRED_SUFFIX}"))
    return {path.name[: -len(PRED_SUFFIX)]: path for path in paths}


def gt_paths(gt_dir: Path) -> dict[str, tuple[Path, Path]]:
    images = {path.stem: path for path in sorted(gt_dir.glob("*.png"))}
    jsons = {path.stem: path for path in sorted(gt_dir.glob("*.json"))}
    if images.keys() != jsons.keys():
        raise RuntimeError("GT PNG/JSON keys do not match")
    return {key: (images[key], jsons[key]) for key in sorted(images)}


def validate_prediction(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    prediction = np.load(path, allow_pickle=False)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"unexpected prediction shape at {path}: {prediction.shape} != {expected_shape}"
        )
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError(f"prediction is not an integer instance map: {path} {prediction.dtype}")
    if np.any(prediction < 0):
        raise ValueError(f"negative instance ID in {path}")
    return prediction.astype(np.int32, copy=False)


def check_png_copy(npy_path: Path, prediction: np.ndarray) -> bool:
    png_path = npy_path.with_suffix(".png")
    if not png_path.is_file():
        return False
    png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    return bool(png is not None and np.array_equal(png, prediction))


def mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_method(
    method: str,
    predictions: dict[str, Path],
    ground_truth: dict[str, tuple[Path, Path]],
    output_csv: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if predictions.keys() != ground_truth.keys():
        missing_predictions = sorted(ground_truth.keys() - predictions.keys())
        missing_ground_truth = sorted(predictions.keys() - ground_truth.keys())
        raise RuntimeError(
            f"{method} key mismatch: missing_predictions={missing_predictions[:5]}, "
            f"missing_ground_truth={missing_ground_truth[:5]}"
        )

    rows: list[dict[str, Any]] = []
    png_exact_count = 0
    max_difference = 0.0
    max_difference_sample = None
    for index, sample_id in enumerate(ground_truth):
        _, json_path = ground_truth[sample_id]
        true, organ_type = load_gt_like_test_py(json_path)
        pred_path = predictions[sample_id]
        pred = validate_prediction(pred_path, true.shape)
        png_exact_count += int(check_png_copy(pred_path, pred))

        official = pq_official(true, pred, match_iou=0.5)
        independent = pq_independent(true, pred, match_iou=0.5)
        difference = abs(official.pq - independent.pq)
        if difference > max_difference:
            max_difference = difference
            max_difference_sample = sample_id
        old_pq, old_dq, old_sq = legacy_pq(true, pred, match_iou=0.5)
        row = {
            "index": index,
            "sample_id": sample_id,
            "organ_type": organ_type,
            "prediction_file": pred_path.name,
            "gt_json_file": json_path.name,
            "gt_instance_count": int(len(np.unique(true)) - 1),
            "pred_instance_count": int(len(np.unique(pred)) - 1),
            "tp": official.tp,
            "fp": official.fp,
            "fn": official.fn,
            "matched_iou_sum": official.matched_iou_sum,
            "dq": official.dq,
            "sq": official.sq,
            "bpq_official": official.pq,
            "bpq_independent": independent.pq,
            "bpq_abs_difference": difference,
            "aji_kumar_greedy": aji_kumar_greedy(true, pred),
            "aji_plus_hungarian_iou": aji_plus(true, pred),
            "dice_binary": binary_dice(true, pred),
            "legacy_bpq_img": float(old_pq),
            "legacy_dq": float(old_dq),
            "legacy_sq": float(old_sq),
            "legacy_aji_custom": float(legacy_aji_custom(true, pred)),
        }
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(ground_truth):
            print(f"[PROGRESS] method={method} images={index + 1}/{len(ground_truth)}", flush=True)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gt_nonempty_rows = [row for row in rows if row["gt_instance_count"] > 0]
    totals = {
        "tp": sum(row["tp"] for row in rows),
        "fp": sum(row["fp"] for row in rows),
        "fn": sum(row["fn"] for row in rows),
        "matched_iou_sum": sum(row["matched_iou_sum"] for row in rows),
    }
    global_pq = pq_from_global_counts(**totals)
    organ_types = sorted({row["organ_type"] for row in gt_nonempty_rows})
    tissue_bpq = {
        organ: mean(
            row["bpq_official"]
            for row in gt_nonempty_rows
            if row["organ_type"] == organ
        )
        for organ in organ_types
    }
    summary = {
        "method": method,
        "sample_count": len(rows),
        "gt_nonempty_sample_count": len(gt_nonempty_rows),
        "prediction_empty_sample_count": sum(
            row["pred_instance_count"] == 0 for row in rows
        ),
        "png_exact_npy_count": png_exact_count,
        "bpq_per_image_avg": mean(row["bpq_official"] for row in gt_nonempty_rows),
        "dq_per_image_avg": mean(row["dq"] for row in gt_nonempty_rows),
        "sq_per_image_avg": mean(row["sq"] for row in gt_nonempty_rows),
        "bpq_global": global_pq.pq,
        "dq_global": global_pq.dq,
        "sq_global": global_pq.sq,
        "global_counts": totals,
        "bpq_pannuke_tissue_macro": mean(tissue_bpq.values()),
        "bpq_by_tissue": tissue_bpq,
        "aji_kumar_greedy_per_image_avg": mean(
            row["aji_kumar_greedy"] for row in rows
        ),
        "aji_plus_per_image_avg": mean(
            row["aji_plus_hungarian_iou"] for row in rows
        ),
        "dice_per_image_avg": mean(row["dice_binary"] for row in rows),
        "legacy_bpq_img": mean(row["legacy_bpq_img"] for row in rows),
        "legacy_aji_custom": mean(row["legacy_aji_custom"] for row in rows),
        "official_independent_max_abs_difference": max_difference,
        "official_independent_max_difference_sample": max_difference_sample,
    }
    audit = {
        "prediction_dtype_values": sorted({str(np.load(path, mmap_mode="r").dtype) for path in predictions.values()}),
        "prediction_shapes": sorted(
            {str(tuple(np.load(path, mmap_mode="r").shape)) for path in predictions.values()}
        ),
        "prediction_min": min(int(np.load(path, mmap_mode="r").min()) for path in predictions.values()),
        "prediction_max": max(int(np.load(path, mmap_mode="r").max()) for path in predictions.values()),
        "png_exact_npy_count": png_exact_count,
    }
    return summary, audit


def build_test_manifest(
    ground_truth: dict[str, tuple[Path, Path]], gt_dir: Path
) -> list[dict[str, Any]]:
    records = []
    for index, (sample_id, (image_path, json_path)) in enumerate(ground_truth.items()):
        with json_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        records.append(
            {
                "index": index,
                "sample_id": sample_id,
                "image_file": image_path.relative_to(gt_dir).as_posix(),
                "image_sha256": sha256_file(image_path),
                "gt_json_file": json_path.relative_to(gt_dir).as_posix(),
                "gt_json_sha256": sha256_file(json_path),
                "organ_type": data.get("organ_type", "NOT_FOUND"),
            }
        )
    return records


def write_sha256s(output_dir: Path) -> None:
    output_path = output_dir / "SHA256SUMS.txt"
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path != output_path:
            lines.append(f"{sha256_file(path)}  {path.name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-pred-dir", type=Path, required=True)
    parser.add_argument("--exp5-pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2607)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.visual_pred_dir, args.exp5_pred_dir, args.gt_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = gt_paths(args.gt_dir)
    prediction_sets = {
        "visual_baseline": prediction_paths(args.visual_pred_dir),
        "exp5": prediction_paths(args.exp5_pred_dir),
    }
    if len(ground_truth) != args.expected_count:
        raise RuntimeError(f"GT count {len(ground_truth)} != {args.expected_count}")
    for method, predictions in prediction_sets.items():
        if len(predictions) != args.expected_count:
            raise RuntimeError(f"{method} count {len(predictions)} != {args.expected_count}")

    input_manifest: dict[str, Any] = {}
    gt_hash, gt_records = summarize_files(
        args.gt_dir, [path for pair in ground_truth.values() for path in pair]
    )
    input_manifest["gt"] = {
        "root": str(args.gt_dir.resolve()),
        "content_summary_sha256": gt_hash,
        "files": gt_records,
    }
    for method, predictions in prediction_sets.items():
        prediction_dir = (
            args.visual_pred_dir if method == "visual_baseline" else args.exp5_pred_dir
        )
        all_prediction_files = list(prediction_dir.glob("*.npy")) + list(
            prediction_dir.glob("*.png")
        )
        content_hash, records = summarize_files(prediction_dir, all_prediction_files)
        input_manifest[method] = {
            "root": str(prediction_dir.resolve()),
            "content_summary_sha256": content_hash,
            "files": records,
        }
    write_json(args.output_dir / "input_file_manifest.json", input_manifest)

    metric_module = REPO_ROOT / "evaluation" / "metrics_standard.py"
    entrypoint_module = Path(__file__).resolve()
    legacy_module = REPO_ROOT / "metrics.py"
    config = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "sample_count": len(ground_truth),
        "prediction_inputs": {
            method: {
                "path": manifest["root"],
                "content_summary_sha256": manifest["content_summary_sha256"],
            }
            for method, manifest in input_manifest.items()
            if method != "gt"
        },
        "gt_input": {
            "path": input_manifest["gt"]["root"],
            "content_summary_sha256": gt_hash,
        },
        "primary_metric_sources": {
            "hover_net": {
                "url": "https://github.com/vqdang/hover_net/blob/"
                f"{HOVERNET_COMMIT}/metrics/stats_utils.py",
                "commit": HOVERNET_COMMIT,
                "upstream_file_sha256": HOVERNET_STATS_UTILS_SHA256,
            },
            "pannuke_metrics": {
                "url": "https://github.com/TIA-Lab/PanNuke-metrics",
                "commit": PANNuke_METRICS_COMMIT,
                "utils_sha256": PANNuke_UTILS_SHA256,
                "run_sha256": PANNuke_RUN_SHA256,
            },
        },
        "local_implementations": {
            "entrypoint_path": str(entrypoint_module),
            "entrypoint_sha256": sha256_file(entrypoint_module),
            "standard_path": str(metric_module),
            "standard_sha256": sha256_file(metric_module),
            "legacy_path": str(legacy_module),
            "legacy_sha256": sha256_file(legacy_module),
        },
        "pq_match_rule": "IoU > 0.5",
        "per_image_bpq_empty_policy": "skip GT-empty images, as PanNuke run.py",
        "global_bpq_empty_policy": "accumulate TP/FP/FN; GT-empty predictions contribute FP",
        "other_metric_empty_policy": "both empty=1; exactly one empty=0",
        "boundary_instance_policy": "no special exclusion",
        "metric_side_min_area_filter": None,
    }
    print("[RECOMPUTE_CONFIG] " + canonical_json_bytes(config).decode("utf-8"), flush=True)
    write_json(args.output_dir / "recompute_config.json", config)

    test_manifest = build_test_manifest(ground_truth, args.gt_dir)
    test_manifest_path = args.output_dir / "test_set_manifest.json"
    write_json(test_manifest_path, test_manifest)
    test_manifest_hash = sha256_file(test_manifest_path)

    summaries: dict[str, Any] = {}
    input_audits: dict[str, Any] = {}
    for method in METHODS:
        output_name = (
            "per_image_metrics_visual_baseline.csv"
            if method == "visual_baseline"
            else "per_image_metrics_exp5.csv"
        )
        summary, audit = evaluate_method(
            method,
            prediction_sets[method],
            ground_truth,
            args.output_dir / output_name,
        )
        summaries[method] = summary
        input_audits[method] = audit

    historical = {
        "visual_baseline": {"legacy_bpq_img": 0.6034, "legacy_aji_custom": 0.6270},
        "exp5": {"legacy_bpq_img": 0.6094, "legacy_aji_custom": 0.6361},
    }
    crosscheck = {
        "historical_four_decimal": {},
        "official_vs_independent": {},
    }
    for method in METHODS:
        actual_bpq = summaries[method]["legacy_bpq_img"]
        actual_aji = summaries[method]["legacy_aji_custom"]
        expected = historical[method]
        crosscheck["historical_four_decimal"][method] = {
            "expected": expected,
            "actual": {
                "legacy_bpq_img": actual_bpq,
                "legacy_aji_custom": actual_aji,
            },
            "bpq_matches": round(actual_bpq, 4) == expected["legacy_bpq_img"],
            "aji_matches": round(actual_aji, 4) == expected["legacy_aji_custom"],
        }
        difference = summaries[method]["official_independent_max_abs_difference"]
        crosscheck["official_vs_independent"][method] = {
            "max_abs_difference": difference,
            "passes_lt_1e_6": difference < 1e-6,
        }

    historical_pass = all(
        result["bpq_matches"] and result["aji_matches"]
        for result in crosscheck["historical_four_decimal"].values()
    )
    double_implementation_pass = all(
        result["passes_lt_1e_6"]
        for result in crosscheck["official_vs_independent"].values()
    )
    result = {
        "training_started": False,
        "inference_rerun": False,
        "sample_count": len(ground_truth),
        "test_set_manifest_sha256": test_manifest_hash,
        "input_audit": input_audits,
        "summaries": summaries,
        "crosscheck": crosscheck,
        "gates": {
            "historical_four_decimal_pass": historical_pass,
            "official_independent_lt_1e_6_pass": double_implementation_pass,
        },
    }
    write_json(args.output_dir / "summary.json", result)
    write_json(args.output_dir / "crosscheck.json", crosscheck)
    write_sha256s(args.output_dir)
    print("[RECOMPUTE_RESULT] " + canonical_json_bytes(result).decode("utf-8"), flush=True)
    if not historical_pass:
        print("[STOP_GATE] historical four-decimal reproduction failed", file=sys.stderr)
        return 2
    if not double_implementation_pass:
        print("[STOP_GATE] official/independent PQ cross-check failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
