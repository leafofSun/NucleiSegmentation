#!/usr/bin/env python3
"""Re-evaluate immutable saved predictions against original PanNuke GT on CPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

from diagnostics.quantify_conversion_loss import build_instance_map  # noqa: E402
from diagnostics.rebuild_index_mapping import (  # noqa: E402
    RawFoldParquet,
    load_mapping,
    sha256_file,
    write_json,
)
from evaluation.metrics_standard import (  # noqa: E402
    aji_kumar_greedy,
    pq_independent,
    pq_official,
)
from evaluation.recompute_from_npy import load_gt_like_test_py  # noqa: E402


METHODS = ("visual_baseline", "exp5")
VARIANTS = ("converted", "original_all", "original_area_ge10")
EXPECTED_CONVERTED = {
    "visual_baseline": {
        "bpq_pannuke_tissue_macro": 0.6211684187763323,
        "aji_kumar_greedy_per_image_avg": 0.6172423951583715,
    },
    "exp5": {
        "bpq_pannuke_tissue_macro": 0.6274871308798883,
        "aji_kumar_greedy_per_image_avg": 0.6270204339352187,
    },
}


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_paths(directory: Path) -> dict[str, Path]:
    suffix = "_inst.npy"
    return {
        path.name[: -len(suffix)]: path
        for path in sorted(directory.glob(f"*{suffix}"))
    }


def manifest_sha256(paths: dict[str, Path], root: Path) -> str:
    digest = hashlib.sha256()
    for sample_id, path in sorted(paths.items()):
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\t{path.stat().st_size}\n".encode()
        )
    return digest.hexdigest()


def validate_prediction(path: Path) -> np.ndarray:
    prediction = np.load(path, allow_pickle=False)
    if prediction.shape != (256, 256):
        raise ValueError(f"unexpected prediction shape: {path} {prediction.shape}")
    if not np.issubdtype(prediction.dtype, np.integer) or np.any(prediction < 0):
        raise TypeError(f"prediction must be a non-negative integer instance map: {path}")
    return prediction.astype(np.int32, copy=False)


def metric_row(
    sample_id: str,
    raw_index: int,
    organ_type: str,
    variant: str,
    true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    official = pq_official(true, prediction, match_iou=0.5)
    independent = pq_independent(true, prediction, match_iou=0.5)
    return {
        "sample_id": sample_id,
        "raw_index": raw_index,
        "organ_type": organ_type,
        "gt_variant": variant,
        "gt_instance_count": int(true.max()),
        "pred_instance_count": int(np.unique(prediction).size - 1),
        "tp": official.tp,
        "fp": official.fp,
        "fn": official.fn,
        "matched_iou_sum": official.matched_iou_sum,
        "dq": official.dq,
        "sq": official.sq,
        "bpq_official": official.pq,
        "bpq_independent": independent.pq,
        "bpq_abs_difference": abs(official.pq - independent.pq),
        "aji_kumar_greedy": aji_kumar_greedy(true, prediction),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonempty = [row for row in rows if int(row["gt_instance_count"]) > 0]
    tissues = sorted({str(row["organ_type"]) for row in nonempty})
    by_tissue = {
        tissue: mean(
            [float(row["bpq_official"]) for row in nonempty if row["organ_type"] == tissue]
        )
        for tissue in tissues
    }
    return {
        "sample_count": len(rows),
        "gt_nonempty_sample_count": len(nonempty),
        "bpq_pannuke_tissue_macro": mean(list(by_tissue.values())),
        "bpq_by_tissue": by_tissue,
        "bpq_per_image_avg": mean([float(row["bpq_official"]) for row in nonempty]),
        "aji_kumar_greedy_per_image_avg": mean([float(row["aji_kumar_greedy"]) for row in rows]),
        "official_independent_max_abs_difference": max(float(row["bpq_abs_difference"]) for row in rows),
        "gt_instance_total": sum(int(row["gt_instance_count"]) for row in rows),
        "prediction_instance_total": sum(int(row["pred_instance_count"]) for row in rows),
    }


def preregistered_decision(delta: float) -> str:
    if delta < -0.02:
        return "GT_CONVERSION_INFLATES"
    if delta <= 0.01:
        return "GT_CONVERSION_NEUTRAL"
    return "GT_CONVERSION_PENALIZES"


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
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--visual-pred-dir", type=Path, required=True)
    parser.add_argument("--exp5-pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2607)
    parser.add_argument("--literature-low-bpq", type=float, default=0.6596)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.raw_parquet, args.mapping):
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (args.converted_dir, args.visual_pred_dir, args.exp5_pred_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    if len(mapping) != args.expected_count:
        raise RuntimeError(f"mapping count {len(mapping)} != {args.expected_count}")
    if any(int(entry["max_abs_diff"]) != 0 for entry in mapping):
        raise RuntimeError("image pixels are not exact; D1 stop gate applies")
    raw_fold = RawFoldParquet(args.raw_parquet)
    prediction_sets = {
        "visual_baseline": prediction_paths(args.visual_pred_dir),
        "exp5": prediction_paths(args.exp5_pred_dir),
    }
    expected_ids = {str(entry["sample_id"]) for entry in mapping}
    for method, paths in prediction_sets.items():
        if set(paths) != expected_ids:
            raise RuntimeError(
                f"{method} prediction IDs differ from mapping: "
                f"missing={sorted(expected_ids - set(paths))[:5]} "
                f"extra={sorted(set(paths) - expected_ids)[:5]}"
            )

    config = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "raw_data_path": str(args.raw_parquet.resolve()),
        "raw_data_sha256": sha256_file(args.raw_parquet),
        "converted_data_path": str(args.converted_dir.resolve()),
        "mapping_path": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "matched_count": len(mapping),
        "prediction_inputs": {
            "visual_baseline": {
                "path": str(args.visual_pred_dir.resolve()),
                "npy_manifest_sha256": manifest_sha256(prediction_sets["visual_baseline"], args.visual_pred_dir),
            },
            "exp5": {
                "path": str(args.exp5_pred_dir.resolve()),
                "npy_manifest_sha256": manifest_sha256(prediction_sets["exp5"], args.exp5_pred_dir),
            },
        },
        "original_all_gt": "all verified per-instance binary masks from raw Fold3 mirror",
        "original_area_ge10_gt": "raw instances filtered by decoded binary pixel area >= 10",
        "pq_match_rule": "strict IoU > 0.5",
        "metric_implementation": "evaluation/metrics_standard.py (P0.3 verified HoVer-Net/PanNuke port)",
        "converted_decoder": "evaluation.recompute_from_npy.load_gt_like_test_py; exact port of test.py:880-926",
        "fairness_note": "The model never saw filtered-out small nuclei during training; original-GT scores are for factor attribution, not a model upper bound.",
    }
    print("[DIAG_CONFIG] " + json.dumps(config, sort_keys=True), flush=True)
    write_json(args.output_dir / "reeval_config.json", config)

    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {variant: [] for variant in VARIANTS} for method in METHODS
    }
    for position, entry in enumerate(mapping):
        sample_id = str(entry["sample_id"])
        raw_index = int(entry["raw_index"])
        record = raw_fold[raw_index]
        raw_masks = record.instance_masks()
        original_all, overlap_pixels = build_instance_map(raw_masks, minimum_area=1)
        original_area_ge10, overlap_pixels_area10 = build_instance_map(raw_masks, minimum_area=10)
        if overlap_pixels or overlap_pixels_area10:
            raise RuntimeError(
                f"raw instance masks overlap at raw index {raw_index}; instance-map semantics UNVERIFIED"
            )
        converted, _ = load_gt_like_test_py(args.converted_dir / f"{sample_id}.json")
        ground_truth = {
            "converted": converted,
            "original_all": original_all,
            "original_area_ge10": original_area_ge10,
        }
        for method in METHODS:
            prediction = validate_prediction(prediction_sets[method][sample_id])
            for variant, true in ground_truth.items():
                all_rows[method][variant].append(
                    metric_row(sample_id, raw_index, record.tissue_name, variant, true, prediction)
                )
        if (position + 1) % 100 == 0 or position + 1 == len(mapping):
            print(f"[PROGRESS] images={position + 1}/{len(mapping)}", flush=True)

    output_names = {
        ("visual_baseline", "original_all"): "reeval_original_gt_visual_baseline.csv",
        ("exp5", "original_all"): "reeval_original_gt_exp5.csv",
        ("visual_baseline", "original_area_ge10"): "reeval_original_gt_area_ge10_visual_baseline.csv",
        ("exp5", "original_area_ge10"): "reeval_original_gt_area_ge10_exp5.csv",
        ("visual_baseline", "converted"): "reeval_converted_gt_visual_baseline.csv",
        ("exp5", "converted"): "reeval_converted_gt_exp5.csv",
    }
    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    for method in METHODS:
        for variant in VARIANTS:
            rows = all_rows[method][variant]
            write_csv(args.output_dir / output_names[(method, variant)], rows)
            summaries[method][variant] = summarize_rows(rows)

    decisions: dict[str, Any] = {}
    converted_crosscheck: dict[str, Any] = {}
    for method in METHODS:
        converted = summaries[method]["converted"]
        original = summaries[method]["original_all"]
        area10 = summaries[method]["original_area_ge10"]
        delta = original["bpq_pannuke_tissue_macro"] - converted["bpq_pannuke_tissue_macro"]
        delta_area10 = area10["bpq_pannuke_tissue_macro"] - converted["bpq_pannuke_tissue_macro"]
        decisions[method] = {
            "delta_original_minus_converted": delta,
            "delta_area_ge10_minus_converted": delta_area10,
            "preregistered_decision": preregistered_decision(delta),
        }
        expected = EXPECTED_CONVERTED[method]
        converted_crosscheck[method] = {
            key: {
                "expected": expected[key],
                "actual": converted[key],
                "abs_difference": abs(converted[key] - expected[key]),
                "passes_le_1e_12": abs(converted[key] - expected[key]) <= 1e-12,
            }
            for key in expected
        }

    primary_delta = decisions["visual_baseline"]["delta_original_minus_converted"]
    converted_visual = summaries["visual_baseline"]["converted"]["bpq_pannuke_tissue_macro"]
    original_visual = summaries["visual_baseline"]["original_all"]["bpq_pannuke_tissue_macro"]
    result = {
        "training_started": False,
        "inference_rerun": False,
        "summaries": summaries,
        "decisions": decisions,
        "primary_preregistered_decision": decisions["visual_baseline"]["preregistered_decision"],
        "converted_crosscheck": converted_crosscheck,
        "attribution": {
            "literature_low_bpq": args.literature_low_bpq,
            "converted_visual_bpq": converted_visual,
            "original_visual_bpq": original_visual,
            "reported_gap_literature_minus_converted": args.literature_low_bpq - converted_visual,
            "conversion_effect_raw_minus_converted": primary_delta,
            "data_side_recoverable_bpq_if_positive": max(0.0, primary_delta),
            "gap_literature_minus_original": args.literature_low_bpq - original_visual,
            "remaining_gap_assignment": "UNKNOWN; D1 isolates conversion only and does not assign the remainder to a particular model/training/protocol factor",
        },
        "fairness_note": config["fairness_note"],
        "gates": {
            "converted_p0_3_crosscheck_pass": all(
                item["passes_le_1e_12"]
                for method_result in converted_crosscheck.values()
                for item in method_result.values()
            ),
            "official_independent_lt_1e_6_pass": all(
                summaries[method][variant]["official_independent_max_abs_difference"] < 1e-6
                for method in METHODS
                for variant in VARIANTS
            ),
        },
    }
    write_json(args.output_dir / "reeval_summary.json", result)
    write_sha256s(args.output_dir)
    print("[REEVAL_RESULT] " + json.dumps(result, sort_keys=True), flush=True)
    if not result["gates"]["converted_p0_3_crosscheck_pass"]:
        print("[STOP_GATE] converted-GT P0.3 cross-check failed", file=sys.stderr)
        return 2
    if not result["gates"]["official_independent_lt_1e_6_pass"]:
        print("[STOP_GATE] official/independent PQ cross-check failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
