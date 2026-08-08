#!/usr/bin/env python3
"""R2.1: rebuild historical anchors on lossless PanNuke_v2 GT (CPU-only).

The script reads saved instance predictions and R1 NPZ ground truth.  It has no
model import, inference path, post-processing path, or training path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.decompose_errors import (  # noqa: E402
    area_conditioned_metrics,
    decompose,
)
from evaluation.metrics_standard import (  # noqa: E402
    aji_plus,
    binary_dice,
    pq_from_global_counts,
    pq_official,
    remap_label,
)


PRED_SUFFIX = "_inst.npy"
PROTOCOLS = ("E1", "E2", "E3")
AREA_BINS = (
    (1, 10, "[1,10)"),
    (10, 20, "[10,20)"),
    (20, 50, "[20,50)"),
    (50, 100, "[50,100)"),
    (100, 200, "[100,200)"),
    (200, math.inf, "[200,+inf)"),
)
DENSITY_BINS = (
    (0, 1, "[0,1)"),
    (1, 10, "[1,10)"),
    (10, 25, "[10,25)"),
    (25, 50, "[25,50)"),
    (50, math.inf, "[50,+inf)"),
)
D1_EXPECTED = {
    "visual": {
        "bpq_tissue_macro": 0.593471982510069,
        "aji_kumar_per_image_avg": 0.5839704510548933,
    },
    "exp5": {
        "bpq_tissue_macro": 0.5998214838700427,
        "aji_kumar_per_image_avg": 0.5924488284109662,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def sample_set_sha256(sample_ids: Iterable[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(sample_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def prediction_paths(directory: Path) -> dict[str, Path]:
    return {
        path.name[: -len(PRED_SUFFIX)]: path
        for path in sorted(directory.glob(f"*{PRED_SUFFIX}"))
    }


def prediction_manifest_sha256(paths: dict[str, Path], root: Path) -> str:
    digest = hashlib.sha256()
    for sample_id, path in sorted(paths.items()):
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\t{path.stat().st_size}\n".encode()
        )
    return digest.hexdigest()


def validate_prediction(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape != (256, 256):
        raise ValueError(f"unexpected prediction shape: {path} {value.shape}")
    if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
        raise TypeError(f"prediction must be a non-negative integer map: {path}")
    return remap_label(value.astype(np.int32, copy=False))


def load_gt(path: Path, entry: dict[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as value:
        gt = value["inst_map"].astype(np.int32, copy=False)
        checks = {
            "fold": int(value["fold"]),
            "orig_index": int(value["orig_index"]),
            "tissue_id": int(value["tissue_id"]),
            "tissue_name": str(value["tissue_name"]),
        }
    expected = {
        "fold": int(entry["fold"]),
        "orig_index": int(entry["orig_index"]),
        "tissue_id": int(entry["tissue_id"]),
        "tissue_name": str(entry["tissue_name"]),
    }
    if checks != expected or gt.shape != (256, 256) or gt.dtype != np.int32:
        raise RuntimeError(f"GT schema/metadata mismatch: {path}")
    ids = np.unique(gt)
    ids = ids[ids > 0]
    if not np.array_equal(ids, np.arange(1, int(entry["instance_count"]) + 1)):
        raise RuntimeError(f"GT identity mismatch: {path}")
    return gt


def density_bin(count: int) -> str:
    return next(label for low, high, label in DENSITY_BINS if low <= count < high)


def included(protocol: str, entry: dict[str, Any], prediction_available: bool) -> bool:
    if not prediction_available:
        return False
    gt_nonempty = int(entry["instance_count"]) > 0
    if protocol == "E1":
        return gt_nonempty
    if protocol == "E2":
        return True
    if protocol == "E3":
        return bool(entry["overlaps_legacy_test"]) and gt_nonempty
    raise ValueError(protocol)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NOT_FOUND", "evaluated_sample_count": 0}
    global_pq = pq_from_global_counts(
        sum(int(row["tp"]) for row in rows),
        sum(int(row["fp"]) for row in rows),
        sum(int(row["fn"]) for row in rows),
        sum(float(row["matched_iou_sum"]) for row in rows),
    )
    tissues = sorted({str(row["tissue_name"]) for row in rows})
    tissue_bpq = {
        tissue: mean(float(row["bpq"]) for row in rows if row["tissue_name"] == tissue)
        for tissue in tissues
    }
    return {
        "status": "COMPLETE_AVAILABLE_PREDICTIONS",
        "evaluated_sample_count": len(rows),
        "gt_instance_total": sum(int(row["gt_instance_count"]) for row in rows),
        "pred_instance_total": sum(int(row["pred_instance_count"]) for row in rows),
        "bpq_tissue_macro": mean(tissue_bpq.values()),
        "bpq_by_tissue": tissue_bpq,
        "bpq_per_image_avg": mean(float(row["bpq"]) for row in rows),
        "bpq_global_agg": global_pq.pq,
        "dq_global_agg": global_pq.dq,
        "sq_global_agg": global_pq.sq,
        "aji_kumar_per_image_avg": mean(float(row["aji_kumar"]) for row in rows),
        "aji_plus_per_image_avg": mean(float(row["aji_plus"]) for row in rows),
        "dice_per_image_avg": mean(float(row["dice"]) for row in rows),
        "tp": global_pq.tp,
        "fp": global_pq.fp,
        "fn": global_pq.fn,
        "matched_iou_sum": global_pq.matched_iou_sum,
    }


def protocol_completion_status(protocol: str, evaluated_count: int) -> str:
    if protocol == "E1" and evaluated_count == 2607:
        return "COMPLETE_DENOMINATOR_A; FULL_DENOMINATOR_B_NOT_EVALUATED_1"
    if protocol == "E2" and evaluated_count == 2607:
        return "UNVERIFIED_FULL_E2_MISSING_PREDICTIONS_115"
    if protocol == "E3" and evaluated_count == 2607:
        return "COMPLETE"
    return "UNVERIFIED_UNEXPECTED_SAMPLE_COUNT"


def error_summary(counts: Counter[str]) -> dict[str, Any]:
    fn = counts["pq_fn_total"]
    fp = counts["pq_fp_total"]
    fn_values = {
        "merged": counts["fn_merged"],
        "complex": counts["fn_complex"],
        "true_miss": counts["fn_true_miss"],
        "low_iou": counts["fn_low_iou"],
    }
    fp_values = {
        "split": counts["fp_split"],
        "complex": counts["fp_complex"],
        "spurious": counts["fp_spurious"],
        "low_iou": counts["fp_low_iou"],
    }
    if sum(fn_values.values()) != fn or sum(fp_values.values()) != fp:
        raise AssertionError("V1 FN/FP decomposition sum mismatch")
    return {
        "pq_counts": {"tp": counts["pq_tp_total"], "fp": fp, "fn": fn},
        "fn_decomposition": {**fn_values, "sum": sum(fn_values.values())},
        "fp_decomposition": {**fp_values, "sum": sum(fp_values.values())},
        "r_merge": counts["fn_merged"] / fn if fn else 0.0,
        "r_miss": counts["fn_true_miss"] / fn if fn else 0.0,
    }


def group_strata(
    method: str,
    protocol: str,
    rows: list[dict[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    if dimension == "gt_density":
        keys = [label for _, _, label in DENSITY_BINS]
        field = "density_bin"
    elif dimension == "tissue":
        keys = sorted({str(row["tissue_name"]) for row in rows})
        field = "tissue_name"
    else:
        raise ValueError(dimension)
    output: list[dict[str, Any]] = []
    for key in keys:
        selected = [row for row in rows if row[field] == key]
        summary = summarize_rows(selected)
        output.append(
            {
                "method": method,
                "protocol": protocol,
                "dimension": dimension,
                "bin": key,
                "sample_count": len(selected),
                "gt_instance_count": sum(int(row["gt_instance_count"]) for row in selected),
                "bpq": summary.get("bpq_per_image_avg", "NOT_FOUND"),
                "aji_kumar": summary.get("aji_kumar_per_image_avg", "NOT_FOUND"),
                "dice": summary.get("dice_per_image_avg", "NOT_FOUND"),
                "r_merge": (
                    sum(int(row["fn_merged"]) for row in selected)
                    / sum(int(row["fn"]) for row in selected)
                    if sum(int(row["fn"]) for row in selected)
                    else 0.0
                ),
                "r_miss": (
                    sum(int(row["fn_true_miss"]) for row in selected)
                    / sum(int(row["fn"]) for row in selected)
                    if sum(int(row["fn"]) for row in selected)
                    else 0.0
                ),
                "coverage_rate": "NOT_COMPUTED_FOR_GROUP",
                "metric_scope": "mean per-image standard metric",
            }
        )
    return output


def area_strata_rows(
    method: str,
    protocol: str,
    accumulators: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for _, _, label in AREA_BINS:
        item = accumulators[label]
        fn = item["fn"]
        output.append(
            {
                "method": method,
                "protocol": protocol,
                "dimension": "gt_area",
                "bin": label,
                "sample_count": len(item["bpq_values"]),
                "gt_instance_count": item["gt"],
                "bpq": mean(item["bpq_values"]),
                "aji_kumar": mean(item["aji_values"]),
                "dice": "NOT_APPLICABLE_INSTANCE_STRATUM",
                "r_merge": item["fn_merged"] / fn if fn else 0.0,
                "r_miss": item["fn_true_miss"] / fn if fn else 0.0,
                "coverage_rate": item["covered"] / item["gt"] if item["gt"] else 0.0,
                "metric_scope": (
                    "V1 GT-area-conditioned: predictions assigned by maximum intersection "
                    "to selected GT; zero-overlap predictions excluded"
                ),
            }
        )
    return output


def disposition_row(entry: dict[str, Any]) -> dict[str, Any]:
    empty = int(entry["instance_count"]) == 0
    return {
        "sample_id": entry["sample_id"],
        "orig_index": entry["orig_index"],
        "tissue_name": entry["tissue_name"],
        "gt_instance_count": entry["instance_count"],
        "gt_empty": int(empty),
        "legacy_test_sample_id": entry.get("legacy_test_sample_id") or "NOT_FOUND",
        "E1_disposition": "SKIP_EMPTY_GT" if empty else "NOT_EVALUATED_MISSING_PREDICTION",
        "E2_disposition": "NOT_EVALUATED_MISSING_PREDICTION",
        "E3_disposition": "OUTSIDE_OLD_TEST_INTERSECTION",
    }


def write_sha256s(output_dir: Path) -> None:
    path = output_dir / "SHA256SUMS.txt"
    lines = [
        f"{sha256_file(item)}  {item.name}"
        for item in sorted(output_dir.iterdir())
        if item.is_file() and item != path
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-pred-dir", type=Path, required=True)
    parser.add_argument("--exp5-pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("all", "E1", "E2", "E3"), default="all")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-metrics-sha256", required=True)
    parser.add_argument("--expected-decompose-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.manifest,):
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (args.gt_root, args.visual_pred_dir, args.exp5_pred_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = REPO_ROOT / "evaluation/metrics_standard.py"
    decompose_path = REPO_ROOT / "diagnostics/decompose_errors.py"
    manifest_sha = sha256_file(args.manifest)
    metrics_sha = sha256_file(metrics_path)
    decompose_sha = sha256_file(decompose_path)
    if manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(f"GT manifest SHA mismatch: {manifest_sha}")
    if metrics_sha != args.expected_metrics_sha256:
        raise RuntimeError(f"metrics implementation SHA mismatch: {metrics_sha}")
    if decompose_sha != args.expected_decompose_sha256:
        raise RuntimeError(f"V1 decomposition SHA mismatch: {decompose_sha}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fold3 = sorted(
        (entry for entry in manifest["samples"] if int(entry["fold"]) == 3),
        key=lambda entry: int(entry["orig_index"]),
    )
    if len(fold3) != 2722 or sum(int(e["instance_count"]) for e in fold3) != 66654:
        raise RuntimeError("Fold3 manifest hard gate failed")
    empty_entries = [entry for entry in fold3 if int(entry["instance_count"]) == 0]
    missing_entries = [entry for entry in fold3 if not bool(entry["overlaps_legacy_test"])]
    missing_nonempty = [entry for entry in missing_entries if int(entry["instance_count"]) > 0]
    if len(empty_entries) != 114 or len(missing_entries) != 115 or len(missing_nonempty) != 1:
        raise RuntimeError("missing/empty Fold3 preregistered count gate failed")

    predictions = {
        "visual": prediction_paths(args.visual_pred_dir),
        "exp5": prediction_paths(args.exp5_pred_dir),
    }
    expected_prediction_ids = {
        str(entry["legacy_test_sample_id"])
        for entry in fold3
        if entry.get("legacy_test_sample_id") is not None
    }
    for method, paths in predictions.items():
        if set(paths) != expected_prediction_ids or len(paths) != 2607:
            raise RuntimeError(f"{method} prediction set differs from manifest mapping")
    prediction_hashes = {
        "visual": prediction_manifest_sha256(predictions["visual"], args.visual_pred_dir),
        "exp5": prediction_manifest_sha256(predictions["exp5"], args.exp5_pred_dir),
    }

    selected_protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    expected_counts = {"E1": 2608, "E2": 2722, "E3": 2607}
    actual_counts = {
        protocol: sum(
            included(protocol, entry, entry.get("legacy_test_sample_id") in predictions["visual"])
            for entry in fold3
        )
        for protocol in PROTOCOLS
    }
    config = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "prediction_inputs": {
            "visual": {"path": str(args.visual_pred_dir.resolve()), "manifest_sha256": prediction_hashes["visual"]},
            "exp5": {"path": str(args.exp5_pred_dir.resolve()), "manifest_sha256": prediction_hashes["exp5"]},
        },
        "gt_root": str(args.gt_root.resolve()),
        "gt_manifest": str(args.manifest.resolve()),
        "gt_manifest_sha256": manifest_sha,
        "protocols": list(selected_protocols),
        "expected_sample_counts": expected_counts,
        "actual_evaluated_sample_counts": actual_counts,
        "metric_implementation_sha256": metrics_sha,
        "v1_decomposition_sha256": decompose_sha,
        "missing_prediction_count": len(missing_entries),
        "missing_nonempty_count": len(missing_nonempty),
        "missing_nonempty_sample_ids": [entry["sample_id"] for entry in missing_nonempty],
    }
    print("[R21_CONFIG] " + json.dumps(config, sort_keys=True), flush=True)
    write_json(args.output_dir / "r21_config.json", config)

    per_image: dict[str, list[dict[str, Any]]] = {method: [] for method in predictions}
    evaluated: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {protocol: [] for protocol in PROTOCOLS} for method in predictions
    }
    error_counts: dict[str, dict[str, Counter[str]]] = {
        method: {protocol: Counter() for protocol in PROTOCOLS} for method in predictions
    }
    area_acc: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        method: {
            protocol: {
                label: {
                    "gt": 0,
                    "covered": 0,
                    "fn": 0,
                    "fn_merged": 0,
                    "fn_true_miss": 0,
                    "bpq_values": [],
                    "aji_values": [],
                }
                for _, _, label in AREA_BINS
            }
            for protocol in PROTOCOLS
        }
        for method in predictions
    }

    for position, entry in enumerate(fold3, 1):
        gt_path = args.gt_root / str(entry["relative_path"])
        gt = load_gt(gt_path, entry)
        gt_count = int(entry["instance_count"])
        legacy_id = entry.get("legacy_test_sample_id")
        available = legacy_id is not None
        base = {
            "sample_id": entry["sample_id"],
            "legacy_test_sample_id": legacy_id or "NOT_FOUND",
            "orig_index": entry["orig_index"],
            "tissue_name": entry["tissue_name"],
            "gt_instance_count": gt_count,
            "gt_empty": int(gt_count == 0),
            "density_bin": density_bin(gt_count),
            "prediction_status": "AVAILABLE" if available else "NOT_EVALUATED",
            "included_E1": int(included("E1", entry, available)),
            "included_E2": int(included("E2", entry, available)),
            "included_E3": int(included("E3", entry, available)),
        }
        for method, method_predictions in predictions.items():
            if not available:
                per_image[method].append(
                    {
                        **base,
                        "pred_instance_count": "NOT_EVALUATED",
                        "bpq": "NOT_EVALUATED",
                        "dq": "NOT_EVALUATED",
                        "sq": "NOT_EVALUATED",
                        "aji_kumar": "NOT_EVALUATED",
                        "aji_plus": "NOT_EVALUATED",
                        "dice": "NOT_EVALUATED",
                        "tp": "NOT_EVALUATED",
                        "fp": "NOT_EVALUATED",
                        "fn": "NOT_EVALUATED",
                        "matched_iou_sum": "NOT_EVALUATED",
                        "fn_merged": "NOT_EVALUATED",
                        "fn_complex": "NOT_EVALUATED",
                        "fn_true_miss": "NOT_EVALUATED",
                        "fn_low_iou": "NOT_EVALUATED",
                        "fp_split": "NOT_EVALUATED",
                        "fp_complex": "NOT_EVALUATED",
                        "fp_spurious": "NOT_EVALUATED",
                        "fp_low_iou": "NOT_EVALUATED",
                    }
                )
                continue
            pred = validate_prediction(method_predictions[str(legacy_id)])
            result = decompose(gt, pred)
            pq = pq_official(gt, pred, match_iou=0.5)
            if abs(pq.pq - result.bpq) > 1e-15:
                raise AssertionError("standard PQ and V1 decomposition PQ differ")
            row = {
                **base,
                "pred_instance_count": int(pred.max()),
                "bpq": pq.pq,
                "dq": pq.dq,
                "sq": pq.sq,
                "aji_kumar": result.aji,
                "aji_plus": aji_plus(gt, pred),
                "dice": binary_dice(gt, pred),
                "tp": pq.tp,
                "fp": pq.fp,
                "fn": pq.fn,
                "matched_iou_sum": pq.matched_iou_sum,
                "fn_merged": result.counts["fn_merged"],
                "fn_complex": result.counts["fn_complex"],
                "fn_true_miss": result.counts["fn_true_miss"],
                "fn_low_iou": result.counts["fn_low_iou"],
                "fp_split": result.counts["fp_split"],
                "fp_complex": result.counts["fp_complex"],
                "fp_spurious": result.counts["fp_spurious"],
                "fp_low_iou": result.counts["fp_low_iou"],
            }
            per_image[method].append(row)
            active_protocols = [
                protocol
                for protocol in selected_protocols
                if included(protocol, entry, available)
            ]
            for protocol in active_protocols:
                evaluated[method][protocol].append(row)
                error_counts[method][protocol].update(result.counts)
            for low, high, label in AREA_BINS:
                records = [r for r in result.gt_records if low <= int(r["area"]) < high]
                if not records:
                    continue
                selected_ids = {int(record["gt_id"]) for record in records}
                conditioned = area_conditioned_metrics(result, selected_ids)
                assert conditioned is not None
                fn_records = [record for record in records if record["fn_reason"] is not None]
                covered = sum(
                    bool(result.intersections[int(record["gt_id"]) - 1].sum())
                    for record in records
                )
                for protocol in active_protocols:
                    accumulator = area_acc[method][protocol][label]
                    accumulator["bpq_values"].append(conditioned[0])
                    accumulator["aji_values"].append(conditioned[1])
                    accumulator["gt"] += len(records)
                    accumulator["fn"] += len(fn_records)
                    accumulator["fn_merged"] += sum(record["fn_reason"] == "merged" for record in fn_records)
                    accumulator["fn_true_miss"] += sum(
                        record["fn_reason"] == "true_miss" for record in fn_records
                    )
                    accumulator["covered"] += covered
        if position % 100 == 0 or position == len(fold3):
            print(f"[R21_PROGRESS] samples={position}/{len(fold3)}", flush=True)

    not_evaluated = [disposition_row(entry) for entry in missing_entries]
    write_csv(args.output_dir / "not_evaluated_samples.csv", not_evaluated)
    for method in predictions:
        write_csv(args.output_dir / f"per_image_anchor_{method}.csv", per_image[method])

    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    errors: dict[str, dict[str, Any]] = defaultdict(dict)
    stratified: list[dict[str, Any]] = []
    for method in predictions:
        for protocol in selected_protocols:
            rows = evaluated[method][protocol]
            summaries[method][protocol] = summarize_rows(rows)
            summaries[method][protocol].update(
                {
                    "protocol_expected_sample_count": expected_counts[protocol],
                    "actual_evaluated_sample_count": len(rows),
                    "complete_protocol_status": protocol_completion_status(protocol, len(rows)),
                }
            )
            errors[method][protocol] = error_summary(error_counts[method][protocol])
            stratified.extend(group_strata(method, protocol, rows, "gt_density"))
            stratified.extend(group_strata(method, protocol, rows, "tissue"))
            stratified.extend(area_strata_rows(method, protocol, area_acc[method][protocol]))
    write_csv(args.output_dir / "stratified_anchor.csv", stratified)

    crosscheck: dict[str, Any] = {}
    if "E3" in selected_protocols:
        for method in predictions:
            actual = summaries[method]["E3"]
            expected = D1_EXPECTED[method]
            checks = {
                key: {
                    "expected": value,
                    "actual": actual[key],
                    "abs_difference": abs(actual[key] - value),
                    "passes_lt_1e_6": abs(actual[key] - value) < 1e-6,
                }
                for key, value in expected.items()
            }
            if not all(check["passes_lt_1e_6"] for check in checks.values()):
                raise RuntimeError(f"STOP: E3 D1 cross-check failed for {method}: {checks}")
            crosscheck[method] = checks

    denominator_a_ids = [
        entry["sample_id"]
        for entry in fold3
        if included("E3", entry, entry.get("legacy_test_sample_id") is not None)
    ]
    denominator_b_ids = [entry["sample_id"] for entry in fold3 if int(entry["instance_count"]) > 0]
    e1_visual_anchor = summaries.get("visual", {}).get("E1", {}).get("bpq_tissue_macro", "UNVERIFIED")
    e1_exp5_anchor = summaries.get("exp5", {}).get("E1", {}).get("bpq_tissue_macro", "UNVERIFIED")
    protocol_document = {
        "protocol_id": "pannuke_v2_e1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "inference_rerun": False,
        "primary_protocol": "E1",
        "skip_empty_gt": True,
        "gt_source": "data/PanNuke_v2/fold3",
        "gt_manifest_sha256": manifest_sha,
        "expected_valid_samples_full_test": 2608,
        "anchor_evaluated_samples_denominator_A": len(denominator_a_ids),
        "anchor_denominator_A_sample_ids_sha256": sample_set_sha256(denominator_a_ids),
        "full_test_denominator_B": len(denominator_b_ids),
        "full_test_denominator_B_sample_ids_sha256": sample_set_sha256(denominator_b_ids),
        "not_evaluated_nonempty_samples": [
            {"sample_id": entry["sample_id"], "instance_count": entry["instance_count"]}
            for entry in missing_nonempty
        ],
        "pq_iou_threshold": "strict_greater_than_0.5",
        "aji_variant": "kumar_greedy",
        "tissue_macro": True,
        "min_instance_area_at_eval": 0,
        "area_bins": [label for _, _, label in AREA_BINS],
        "density_bins": [label for _, _, label in DENSITY_BINS],
        "metric_implementation_sha256": metrics_sha,
        "v1_error_decomposition_sha256": decompose_sha,
        "anchor_values_denominator_A": {
            "visual_bpq_tissue_macro": e1_visual_anchor,
            "exp5_bpq_tissue_macro": e1_exp5_anchor,
        },
        "r22_preregistered_criteria": {
            "STRONG_RECOVERY": "delta global tissue-macro bPQ > 0.03 AND bPQ [50,100) > 0.25",
            "MODERATE": "0.01 <= delta global tissue-macro bPQ <= 0.03",
            "WEAK": "delta global tissue-macro bPQ < 0.01",
        },
        "r22_required_comparison": (
            "Primary historical comparison uses denominator A. Also report denominator B for the complete new test."
        ),
        "e2_full_status": (
            "UNVERIFIED: 115 samples have no historical prediction; missing files are not imputed as empty predictions"
        ),
    }
    protocol_path = args.output_dir / "eval_protocol_v2.json"
    write_json(protocol_path, protocol_document)
    protocol_sha = sha256_file(protocol_path)

    for method in predictions:
        payload = {
            "training_started": False,
            "inference_rerun": False,
            "method": method,
            "protocol_id": "pannuke_v2_e1",
            "eval_protocol_v2_sha256": protocol_sha,
            "prediction_manifest_sha256": prediction_hashes[method],
            "gt_manifest_sha256": manifest_sha,
            "metric_implementation_sha256": metrics_sha,
            "protocols": summaries[method],
            "error_decomposition": errors[method],
            "d1_e3_crosscheck": crosscheck.get(method, "NOT_RUN_PROTOCOL_NOT_SELECTED"),
        }
        write_json(args.output_dir / f"anchor_metrics_{method}.json", payload)

    result = {
        "training_started": False,
        "inference_rerun": False,
        "status": "PASS" if crosscheck else "COMPLETE_WITHOUT_E3_CROSSCHECK",
        "protocol_id": "pannuke_v2_e1",
        "eval_protocol_v2_sha256": protocol_sha,
        "actual_evaluated_sample_counts": actual_counts,
        "missing_nonempty_samples": [entry["sample_id"] for entry in missing_nonempty],
        "d1_e3_crosscheck": crosscheck,
    }
    write_json(args.output_dir / "r21_summary.json", result)
    write_sha256s(args.output_dir)
    print("[R21_RESULT] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
