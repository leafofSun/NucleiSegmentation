#!/usr/bin/env python3
"""V1: decompose saved-prediction errors by overlap-graph component on CPU.

This entry point only reads immutable prediction arrays and ground truth.  It
does not import a model and contains no training or inference path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.quantify_conversion_loss import build_instance_map  # noqa: E402
from diagnostics.rebuild_index_mapping import (  # noqa: E402
    StreamingRawFoldParquet,
    load_mapping,
    sha256_file,
    write_json,
)
from evaluation.metrics_standard import (  # noqa: E402
    aji_kumar_greedy,
    pq_official,
    remap_label,
)
from evaluation.recompute_from_npy import load_gt_like_test_py  # noqa: E402


METHODS = ("visual", "exp5")
VARIANTS = ("original", "converted")
PRED_SUFFIX = "_inst.npy"
EXPECTED_CONVERTED_COUNTS = {
    "visual": {"fn": 15609, "fp": 12470},
    "exp5": {"fn": 15754, "fp": 11490},
}
DENSITY_BINS = (
    (1, 10, "[1,10)"),
    (10, 25, "[10,25)"),
    (25, 50, "[25,50)"),
    (50, math.inf, "[50,+inf)"),
)
AREA_BINS = (
    (10, 50, "[10,50)"),
    (50, 100, "[50,100)"),
    (100, 200, "[100,200)"),
    (200, math.inf, "[200,+inf)"),
)


@dataclass(frozen=True)
class GraphComponent:
    component_type: str
    gt_ids: tuple[int, ...]
    pred_ids: tuple[int, ...]
    pq_match_count: int


@dataclass
class Decomposition:
    true: np.ndarray
    pred: np.ndarray
    intersections: np.ndarray
    iou: np.ndarray
    components: list[GraphComponent]
    counts: Counter[str]
    gt_records: list[dict[str, Any]]
    pred_records: list[dict[str, Any]]
    bpq: float
    aji: float
    matched_iou_sum: float


@dataclass
class CompactDecomposition:
    counts: Counter[str]
    gt_records: list[dict[str, Any]]
    pred_instance_count: int
    bpq: float
    aji: float
    area_metrics: dict[str, tuple[float, float] | None]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def manifest_sha256(paths: dict[str, Path], root: Path) -> str:
    digest = hashlib.sha256()
    for sample_id, path in sorted(paths.items()):
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\t{path.stat().st_size}\n".encode()
        )
    return digest.hexdigest()


def prediction_paths(directory: Path) -> dict[str, Path]:
    return {
        path.name[: -len(PRED_SUFFIX)]: path
        for path in sorted(directory.glob(f"*{PRED_SUFFIX}"))
    }


def validate_prediction(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape != (256, 256):
        raise ValueError(f"unexpected prediction shape: {path} {value.shape}")
    if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
        raise TypeError(f"prediction must be a non-negative integer map: {path}")
    return remap_label(value.astype(np.int32, copy=False))


def contingency(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_count = int(true.max())
    pred_count = int(pred.max())
    intersections = np.zeros((gt_count, pred_count), dtype=np.int64)
    overlap = (true > 0) & (pred > 0)
    if np.any(overlap):
        encoded = (true[overlap] - 1) * pred_count + (pred[overlap] - 1)
        intersections = np.bincount(
            encoded, minlength=gt_count * pred_count
        ).reshape(gt_count, pred_count)
    gt_areas = np.bincount(true.ravel(), minlength=gt_count + 1)[1:]
    pred_areas = np.bincount(pred.ravel(), minlength=pred_count + 1)[1:]
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return intersections, iou


def component_type(gt_size: int, pred_size: int, one_iou: float | None) -> str:
    if gt_size == 1 and pred_size == 1:
        return "tp_one_to_one" if float(one_iou) > 0.5 else "weak_tp"
    if gt_size >= 2 and pred_size == 1:
        return "merge"
    if gt_size == 1 and pred_size >= 2:
        return "split"
    if gt_size >= 2 and pred_size >= 2:
        return "complex"
    if gt_size == 1 and pred_size == 0:
        return "fn_true_miss"
    if gt_size == 0 and pred_size == 1:
        return "fp_spurious"
    raise AssertionError(f"unclassifiable component ({gt_size}, {pred_size})")


def graph_components(intersections: np.ndarray, iou: np.ndarray) -> list[GraphComponent]:
    gt_count, pred_count = intersections.shape
    edges = intersections > 0
    gt_adj = [np.flatnonzero(edges[index]).tolist() for index in range(gt_count)]
    pred_adj = [np.flatnonzero(edges[:, index]).tolist() for index in range(pred_count)]
    visited_gt: set[int] = set()
    visited_pred: set[int] = set()
    components: list[GraphComponent] = []

    for seed in range(gt_count):
        if seed in visited_gt:
            continue
        if not gt_adj[seed]:
            visited_gt.add(seed)
            components.append(GraphComponent("fn_true_miss", (seed + 1,), (), 0))
            continue
        queue: deque[tuple[str, int]] = deque([("gt", seed)])
        local_gt: set[int] = set()
        local_pred: set[int] = set()
        while queue:
            side, index = queue.popleft()
            if side == "gt":
                if index in visited_gt:
                    continue
                visited_gt.add(index)
                local_gt.add(index)
                queue.extend(("pred", value) for value in gt_adj[index])
            else:
                if index in visited_pred:
                    continue
                visited_pred.add(index)
                local_pred.add(index)
                queue.extend(("gt", value) for value in pred_adj[index])
        gt_indices = sorted(local_gt)
        pred_indices = sorted(local_pred)
        high_iou = iou[np.ix_(gt_indices, pred_indices)] > 0.5
        if np.any(high_iou.sum(axis=0) > 1) or np.any(high_iou.sum(axis=1) > 1):
            raise AssertionError("strict IoU > 0.5 matches are not unique")
        one_iou = iou[gt_indices[0], pred_indices[0]] if len(gt_indices) == len(pred_indices) == 1 else None
        kind = component_type(len(gt_indices), len(pred_indices), one_iou)
        components.append(
            GraphComponent(
                kind,
                tuple(value + 1 for value in gt_indices),
                tuple(value + 1 for value in pred_indices),
                int(high_iou.sum()),
            )
        )

    for seed in range(pred_count):
        if seed in visited_pred:
            continue
        if pred_adj[seed]:
            raise AssertionError("connected prediction was not visited")
        visited_pred.add(seed)
        components.append(GraphComponent("fp_spurious", (), (seed + 1,), 0))

    if len(visited_gt) != gt_count or len(visited_pred) != pred_count:
        raise AssertionError("component partition does not cover all nodes")
    return components


def decompose(true: np.ndarray, pred: np.ndarray) -> Decomposition:
    true = remap_label(true.astype(np.int32, copy=False))
    pred = remap_label(pred.astype(np.int32, copy=False))
    intersections, iou = contingency(true, pred)
    components = graph_components(intersections, iou)
    official = pq_official(true, pred, match_iou=0.5)
    counts: Counter[str] = Counter()
    gt_records: list[dict[str, Any]] = []
    pred_records: list[dict[str, Any]] = []
    gt_areas = np.bincount(true.ravel(), minlength=int(true.max()) + 1)
    pred_areas = np.bincount(pred.ravel(), minlength=int(pred.max()) + 1)

    for component_index, component in enumerate(components):
        kind = component.component_type
        gt_size = len(component.gt_ids)
        pred_size = len(component.pred_ids)
        counts[f"event_{kind}"] += 1
        counts[f"gt_{kind}"] += gt_size
        counts[f"pred_{kind}"] += pred_size
        if kind == "merge":
            bucket = "5+" if gt_size >= 5 else str(gt_size)
            counts[f"merge_swallow_{bucket}"] += 1
        if kind == "split":
            bucket = "5+" if pred_size >= 5 else str(pred_size)
            counts[f"split_output_{bucket}"] += 1

        matched_gt_ids: set[int] = set()
        matched_pred_ids: set[int] = set()
        if gt_ids := component.gt_ids:
            if pred_ids := component.pred_ids:
                sub = iou[np.ix_([value - 1 for value in gt_ids], [value - 1 for value in pred_ids])]
                rows, columns = np.nonzero(sub > 0.5)
                matched_gt_ids = {gt_ids[index] for index in rows}
                matched_pred_ids = {pred_ids[index] for index in columns}

        for gt_id in component.gt_ids:
            fn_reason: str | None = None
            if gt_id not in matched_gt_ids:
                if kind == "merge":
                    fn_reason = "merged"
                elif kind == "complex":
                    fn_reason = "complex"
                elif kind == "fn_true_miss":
                    fn_reason = "true_miss"
                else:
                    # The preregistered FN list has no FN_split bucket.  An
                    # unmatched GT in a split component has overlap but no
                    # IoU>0.5 match, so it belongs to FN_low_iou.
                    fn_reason = "low_iou"
                counts[f"fn_{fn_reason}"] += 1
            gt_records.append(
                {
                    "gt_id": gt_id,
                    "area": int(gt_areas[gt_id]),
                    "component_index": component_index,
                    "component_type": kind,
                    "pq_matched": gt_id in matched_gt_ids,
                    "fn_reason": fn_reason,
                }
            )

        for pred_id in component.pred_ids:
            fp_reason: str | None = None
            if pred_id not in matched_pred_ids:
                if kind == "split":
                    fp_reason = "split"
                elif kind == "complex":
                    fp_reason = "complex"
                elif kind == "fp_spurious":
                    fp_reason = "spurious"
                else:
                    # The preregistered FP list has no FP_merge bucket.  An
                    # unmatched prediction in a merge/weak component has
                    # overlap but no IoU>0.5 match, hence FP_low_iou.
                    fp_reason = "low_iou"
                counts[f"fp_{fp_reason}"] += 1
            pred_records.append(
                {
                    "pred_id": pred_id,
                    "area": int(pred_areas[pred_id]),
                    "component_index": component_index,
                    "component_type": kind,
                    "pq_matched": pred_id in matched_pred_ids,
                    "fp_reason": fp_reason,
                }
            )

    counts["pq_tp_total"] = official.tp
    counts["pq_fn_total"] = official.fn
    counts["pq_fp_total"] = official.fp
    fn_sum = counts["fn_merged"] + counts["fn_complex"] + counts["fn_true_miss"] + counts["fn_low_iou"]
    fp_sum = counts["fp_split"] + counts["fp_complex"] + counts["fp_spurious"] + counts["fp_low_iou"]
    if fn_sum != official.fn:
        raise AssertionError(f"FN accounting failed: {fn_sum} != {official.fn}")
    if fp_sum != official.fp:
        raise AssertionError(f"FP accounting failed: {fp_sum} != {official.fp}")
    if sum(1 for record in gt_records if record["pq_matched"]) != official.tp:
        raise AssertionError("GT match count differs from PQ TP")
    if sum(1 for record in pred_records if record["pq_matched"]) != official.tp:
        raise AssertionError("prediction match count differs from PQ TP")
    if len(gt_records) != int(true.max()) or len(pred_records) != int(pred.max()):
        raise AssertionError("instance record count mismatch")
    return Decomposition(
        true=true,
        pred=pred,
        intersections=intersections,
        iou=iou,
        components=components,
        counts=counts,
        gt_records=gt_records,
        pred_records=pred_records,
        bpq=official.pq,
        aji=aji_kumar_greedy(true, pred),
        matched_iou_sum=official.matched_iou_sum,
    )


def density_bin(count: int) -> str:
    return next(label for low, high, label in DENSITY_BINS if low <= count < high)


def area_bin(area: int) -> str | None:
    for low, high, label in AREA_BINS:
        if low <= area < high:
            return label
    return None


def boundary_coordinates(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate([contour[:, 0, ::-1] for contour in contours], axis=0)


def pairwise_boundary_distances(instance_map: np.ndarray, gt_ids: tuple[int, ...]) -> list[float]:
    masks = {gt_id: instance_map == gt_id for gt_id in gt_ids}
    boundaries = {gt_id: boundary_coordinates(mask) for gt_id, mask in masks.items()}
    distances: list[float] = []
    for first_index, first_id in enumerate(gt_ids):
        transform = distance_transform_edt(~masks[first_id])
        for second_id in gt_ids[first_index + 1 :]:
            coords = boundaries[second_id]
            if len(coords) == 0:
                continue
            distances.append(float(transform[coords[:, 0], coords[:, 1]].min()))
    return distances


def merge_event_rows(
    sample_id: str,
    raw_index: int,
    tissue: str,
    decomposition: Decomposition,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    areas = {int(record["gt_id"]): int(record["area"]) for record in decomposition.gt_records}
    event_number = 0
    for component_index, component in enumerate(decomposition.components):
        if component.component_type != "merge":
            continue
        event_number += 1
        distances = pairwise_boundary_distances(decomposition.true, component.gt_ids)
        rows.append(
            {
                "sample_id": sample_id,
                "raw_index": raw_index,
                "tissue_type": tissue,
                "event_index_in_image": event_number,
                "component_index": component_index,
                "swallowed_gt_count": len(component.gt_ids),
                "pred_id": component.pred_ids[0],
                "pq_match_count": component.pq_match_count,
                "gt_ids_json": json.dumps(component.gt_ids),
                "gt_areas_json": json.dumps([areas[value] for value in component.gt_ids]),
                "gt_area_sum": sum(areas[value] for value in component.gt_ids),
                "nearest_gt_pair_boundary_distance": min(distances) if distances else "UNVERIFIED",
                "pairwise_boundary_distance_median": float(np.median(distances)) if distances else "UNVERIFIED",
                "pairwise_boundary_distances_json": json.dumps(distances),
            }
        )
    return rows


def flatten_image_row(
    sample_id: str,
    raw_index: int,
    tissue: str,
    raw_gt_count: int,
    converted_gt_count: int,
    result_by_variant: dict[str, Decomposition],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "raw_index": raw_index,
        "tissue_type": tissue,
        "raw_gt_instance_count": raw_gt_count,
        "converted_gt_instance_count": converted_gt_count,
        "density_bin": density_bin(raw_gt_count),
    }
    count_fields = (
        "pq_tp_total", "pq_fn_total", "pq_fp_total",
        "fn_merged", "fn_complex", "fn_true_miss", "fn_low_iou",
        "fp_split", "fp_complex", "fp_spurious", "fp_low_iou",
        "gt_tp_one_to_one", "gt_merge", "gt_split", "gt_complex",
        "gt_fn_true_miss", "gt_weak_tp",
        "event_merge", "event_split", "event_complex",
    )
    for variant, result in result_by_variant.items():
        row[f"{variant}_bpq"] = result.bpq
        row[f"{variant}_aji"] = result.aji
        row[f"{variant}_pred_instance_count"] = int(result.pred.max())
        for field in count_fields:
            row[f"{variant}_{field}"] = result.counts[field]
        denominator = result.counts["pq_fn_total"]
        row[f"{variant}_r_merge"] = result.counts["fn_merged"] / denominator if denominator else 0.0
        row[f"{variant}_r_miss"] = result.counts["fn_true_miss"] / denominator if denominator else 0.0
    return row


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"sample_count": 0, "status": "NOT_FOUND"}
    summed: Counter[str] = Counter()
    for item in items:
        summed.update(item["result"].counts)
    gt_total = sum(len(item["result"].gt_records) for item in items)
    pred_total = sum(item["result"].pred_instance_count for item in items)
    fn_total = summed["pq_fn_total"]
    fp_total = summed["pq_fp_total"]
    tissue_bpq = {
        tissue: mean([item["result"].bpq for item in items if item["tissue"] == tissue])
        for tissue in sorted({item["tissue"] for item in items})
    }
    return {
        "sample_count": len(items),
        "gt_instance_total": gt_total,
        "pred_instance_total": pred_total,
        "gt_instance_categories": {
            "tp_one_to_one": summed["gt_tp_one_to_one"],
            "merge_involved": summed["gt_merge"],
            "split_involved": summed["gt_split"],
            "complex_involved": summed["gt_complex"],
            "fn_true_miss": summed["gt_fn_true_miss"],
            "weak_tp": summed["gt_weak_tp"],
        },
        "gt_instance_category_fractions": {
            "tp_one_to_one": summed["gt_tp_one_to_one"] / gt_total,
            "merge_involved": summed["gt_merge"] / gt_total,
            "split_involved": summed["gt_split"] / gt_total,
            "complex_involved": summed["gt_complex"] / gt_total,
            "fn_true_miss": summed["gt_fn_true_miss"] / gt_total,
            "weak_tp": summed["gt_weak_tp"] / gt_total,
        },
        "event_counts": {
            "merge": summed["event_merge"],
            "split": summed["event_split"],
            "complex": summed["event_complex"],
        },
        "merge_swallowed_gt_distribution": {
            "2": summed["merge_swallow_2"],
            "3": summed["merge_swallow_3"],
            "4": summed["merge_swallow_4"],
            "5+": summed["merge_swallow_5+"],
        },
        "split_output_pred_distribution": {
            "2": summed["split_output_2"],
            "3": summed["split_output_3"],
            "4": summed["split_output_4"],
            "5+": summed["split_output_5+"],
        },
        "pq_counts": {
            "tp": summed["pq_tp_total"],
            "fn": fn_total,
            "fp": fp_total,
        },
        "fn_decomposition": {
            "merged": summed["fn_merged"],
            "complex": summed["fn_complex"],
            "true_miss": summed["fn_true_miss"],
            "low_iou": summed["fn_low_iou"],
            "sum": summed["fn_merged"] + summed["fn_complex"] + summed["fn_true_miss"] + summed["fn_low_iou"],
        },
        "fp_decomposition": {
            "split": summed["fp_split"],
            "complex": summed["fp_complex"],
            "spurious": summed["fp_spurious"],
            "low_iou": summed["fp_low_iou"],
            "sum": summed["fp_split"] + summed["fp_complex"] + summed["fp_spurious"] + summed["fp_low_iou"],
        },
        "r_merge": summed["fn_merged"] / fn_total if fn_total else 0.0,
        "r_miss": summed["fn_true_miss"] / fn_total if fn_total else 0.0,
        "bpq_per_image_avg": mean([item["result"].bpq for item in items]),
        "bpq_by_tissue": tissue_bpq,
        "bpq_tissue_macro": mean(list(tissue_bpq.values())),
        "aji_per_image_avg": mean([item["result"].aji for item in items]),
        "self_checks": {
            "fn_sum_equals_pq_fn": (
                summed["fn_merged"] + summed["fn_complex"] + summed["fn_true_miss"] + summed["fn_low_iou"] == fn_total
            ),
            "fp_sum_equals_pq_fp": (
                summed["fp_split"] + summed["fp_complex"] + summed["fp_spurious"] + summed["fp_low_iou"] == fp_total
            ),
        },
    }


def decision(r_merge: float, r_miss: float) -> str:
    # The preregistered MIXED row explicitly includes a gap < 0.10, so that
    # clause takes precedence if it overlaps a >0.40 dominance clause.
    if abs(r_merge - r_miss) < 0.10:
        return "MIXED"
    if r_merge > 0.40 and r_merge > r_miss:
        return "MERGE_DOMINANT"
    if r_miss > 0.40 and r_miss > r_merge:
        return "MISS_DOMINANT"
    return "MIXED"


def area_conditioned_metrics(
    result: Decomposition, selected_gt_ids: set[int]
) -> tuple[float, float] | None:
    if not selected_gt_ids:
        return None
    true_filtered = np.zeros_like(result.true)
    for gt_id in selected_gt_ids:
        true_filtered[result.true == gt_id] = gt_id

    selected_pred_ids: set[int] = set()
    for pred_id in range(1, int(result.pred.max()) + 1):
        column = result.intersections[:, pred_id - 1]
        if column.sum() == 0:
            continue
        dominant_gt_id = int(np.argmax(column)) + 1
        if dominant_gt_id in selected_gt_ids:
            selected_pred_ids.add(pred_id)
    pred_filtered = np.zeros_like(result.pred)
    for pred_id in selected_pred_ids:
        pred_filtered[result.pred == pred_id] = pred_id
    pq = pq_official(remap_label(true_filtered), remap_label(pred_filtered), match_iou=0.5)
    aji = aji_kumar_greedy(remap_label(true_filtered), remap_label(pred_filtered))
    return pq.pq, aji


def compact_decomposition(result: Decomposition) -> CompactDecomposition:
    conditioned: dict[str, tuple[float, float] | None] = {}
    for low, high, label in AREA_BINS:
        selected = {
            int(record["gt_id"])
            for record in result.gt_records
            if low <= int(record["area"]) < high
        }
        conditioned[label] = area_conditioned_metrics(result, selected)
    return CompactDecomposition(
        counts=Counter(result.counts),
        gt_records=result.gt_records,
        pred_instance_count=len(result.pred_records),
        bpq=result.bpq,
        aji=result.aji,
        area_metrics=conditioned,
    )


def area_strata(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low, high, label in AREA_BINS:
        gt_records = [
            record
            for item in items
            for record in item["result"].gt_records
            if low <= int(record["area"]) < high
        ]
        fn_records = [record for record in gt_records if record["fn_reason"] is not None]
        fn_total = len(fn_records)
        conditioned: list[tuple[float, float]] = []
        for item in items:
            value = item["result"].area_metrics[label]
            if value is not None:
                conditioned.append(value)
        rows.append(
            {
                "dimension": "gt_area",
                "bin": label,
                "sample_count": len(conditioned),
                "gt_instance_count": len(gt_records),
                "pq_fn_total": fn_total,
                "fn_merged": sum(record["fn_reason"] == "merged" for record in fn_records),
                "fn_true_miss": sum(record["fn_reason"] == "true_miss" for record in fn_records),
                "r_merge": sum(record["fn_reason"] == "merged" for record in fn_records) / fn_total if fn_total else 0.0,
                "r_miss": sum(record["fn_reason"] == "true_miss" for record in fn_records) / fn_total if fn_total else 0.0,
                "bpq": mean([value[0] for value in conditioned]),
                "aji": mean([value[1] for value in conditioned]),
                "metric_scope": "GT-area-conditioned; predictions assigned by maximum intersection to a selected GT; zero-overlap predictions excluded",
            }
        )
    return rows


def group_strata(items: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = item["density_bin"] if dimension == "gt_density" else item["tissue"]
        grouped[key].append(item)
    ordering = [label for _, _, label in DENSITY_BINS] if dimension == "gt_density" else sorted(grouped)
    rows: list[dict[str, Any]] = []
    for key in ordering:
        summary = aggregate(grouped.get(key, []))
        if summary.get("status") == "NOT_FOUND":
            continue
        rows.append(
            {
                "dimension": dimension,
                "bin": key,
                "sample_count": summary["sample_count"],
                "gt_instance_count": summary["gt_instance_total"],
                "pq_fn_total": summary["pq_counts"]["fn"],
                "fn_merged": summary["fn_decomposition"]["merged"],
                "fn_true_miss": summary["fn_decomposition"]["true_miss"],
                "r_merge": summary["r_merge"],
                "r_miss": summary["r_miss"],
                "bpq": summary["bpq_per_image_avg"],
                "aji": summary["aji_per_image_avg"],
                "metric_scope": "mean per-image standard metric",
            }
        )
    return rows


def parquet_audit(path: Path, url: str, revision: str) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for Fold availability audit") from exc
    file = parquet.ParquetFile(path)
    instance_total = 0
    for row_group in range(file.num_row_groups):
        column = file.read_row_group(row_group, columns=["categories"])["categories"]
        instance_total += sum(len(value.as_py()) for value in column)
    schema = file.schema_arrow
    logical_schema = {
        "columns": list(schema.names),
        "image": str(schema.field("image").type),
        "instances": str(schema.field("instances").type),
        "categories": str(schema.field("categories").type),
        "tissue": str(schema.field("tissue").type),
    }
    first = file.read_row_group(0, columns=["image", "instances", "categories", "tissue"])
    image_value = first["image"][0].as_py()
    instance_values = first["instances"][0].as_py()
    with Image.open(BytesIO(image_value["bytes"])) as image:
        image_mode = image.mode
        image_size = list(image.size)
    instance_modes: set[str] = set()
    instance_sizes: set[tuple[int, int]] = set()
    for value in instance_values:
        with Image.open(BytesIO(value["bytes"])) as instance:
            instance_modes.add(instance.mode)
            instance_sizes.add(instance.size)
    file_hash = sha256_file(path)
    return {
        "url": url,
        "retrieval_url": url.replace("https://huggingface.co", "https://hf-mirror.com"),
        "revision": revision,
        "available": True,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_hash,
        "source_linked_etag": file_hash,
        "row_count": file.metadata.num_rows,
        "instance_total": instance_total,
        "logical_schema": logical_schema,
        "decoded_schema_probe": {
            "image_mode": image_mode,
            "image_size": image_size,
            "instance_modes": sorted(instance_modes),
            "instance_sizes": [list(value) for value in sorted(instance_sizes)],
            "categories_type": type(first["categories"][0].as_py()).__name__,
            "tissue_type": type(first["tissue"][0].as_py()).__name__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-fold3", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--fold2", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--visual-pred-dir", type=Path, required=True)
    parser.add_argument("--exp5-pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2607)
    parser.add_argument("--source-revision", default="b9a02ac839d0383bdd3d023b56270ff402b1417f")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.raw_fold3, args.fold1, args.fold2, args.mapping):
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (args.converted_dir, args.visual_pred_dir, args.exp5_pred_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    if len(mapping) != args.expected_count or any(int(item["max_abs_diff"]) != 0 for item in mapping):
        raise RuntimeError("D1 exact mapping gate failed")
    expected_ids = {str(item["sample_id"]) for item in mapping}
    prediction_sets = {
        "visual": prediction_paths(args.visual_pred_dir),
        "exp5": prediction_paths(args.exp5_pred_dir),
    }
    for method, paths in prediction_sets.items():
        if set(paths) != expected_ids:
            raise RuntimeError(f"{method} prediction IDs do not match D1 mapping")

    config = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "sample_count": len(mapping),
        "prediction_inputs": {
            "visual": {
                "path": str(args.visual_pred_dir.resolve()),
                "manifest_sha256": manifest_sha256(prediction_sets["visual"], args.visual_pred_dir),
            },
            "exp5": {
                "path": str(args.exp5_pred_dir.resolve()),
                "manifest_sha256": manifest_sha256(prediction_sets["exp5"], args.exp5_pred_dir),
            },
        },
        "gt_source": {
            "raw_fold3_path": str(args.raw_fold3.resolve()),
            "raw_fold3_sha256": sha256_file(args.raw_fold3),
            "raw_fold_loading": "streaming one Parquet row group at a time",
            "converted_path": str(args.converted_dir.resolve()),
            "official_binarize_rule": "channels 0..4 and ascending IDs; later instances overwrite overlap pixels",
        },
        "mapping_path": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "overlap_edge_rule": "IoU > 0 (equivalently intersection pixels > 0)",
        "pq_match_rule": "strict IoU > 0.5",
        "preregistered_thresholds": {
            "dominance": 0.40,
            "mixed_absolute_gap": 0.10,
        },
    }
    print("[V1_CONFIG] " + json.dumps(config, sort_keys=True), flush=True)
    write_json(args.output_dir / "v1_config.json", config)

    raw_fold = StreamingRawFoldParquet(args.raw_fold3)
    per_image_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    analysis_items: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {variant: [] for variant in VARIANTS} for method in METHODS
    }
    merge_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}

    for position, entry in enumerate(mapping):
        sample_id = str(entry["sample_id"])
        raw_index = int(entry["raw_index"])
        record = raw_fold[raw_index]
        original, _ = build_instance_map(record.instance_masks(), minimum_area=1)
        original = remap_label(original)
        converted, _ = load_gt_like_test_py(args.converted_dir / f"{sample_id}.json")
        converted = remap_label(converted)
        raw_gt_count = int(original.max())
        converted_gt_count = int(converted.max())
        image_density_bin = density_bin(raw_gt_count)
        for method in METHODS:
            pred = validate_prediction(prediction_sets[method][sample_id])
            result_by_variant = {
                "original": decompose(original, pred),
                "converted": decompose(converted, pred),
            }
            per_image_rows[method].append(
                flatten_image_row(
                    sample_id,
                    raw_index,
                    record.tissue_name,
                    raw_gt_count,
                    converted_gt_count,
                    result_by_variant,
                )
            )
            for variant, result in result_by_variant.items():
                analysis_items[method][variant].append(
                    {
                        "sample_id": sample_id,
                        "raw_index": raw_index,
                        "tissue": record.tissue_name,
                        "density_bin": image_density_bin,
                        "result": compact_decomposition(result),
                    }
                )
            merge_rows[method].extend(
                merge_event_rows(sample_id, raw_index, record.tissue_name, result_by_variant["original"])
            )
        if (position + 1) % 100 == 0 or position + 1 == len(mapping):
            print(f"[PROGRESS] images={position + 1}/{len(mapping)}", flush=True)

    for method in METHODS:
        write_csv(
            args.output_dir / f"per_image_error_decomposition_{method}.csv",
            per_image_rows[method],
        )
        write_csv(
            args.output_dir / f"merge_events_{method}.csv",
            merge_rows[method],
            [
                "sample_id", "raw_index", "tissue_type", "event_index_in_image",
                "component_index", "swallowed_gt_count", "pred_id", "pq_match_count",
                "gt_ids_json", "gt_areas_json", "gt_area_sum",
                "nearest_gt_pair_boundary_distance", "pairwise_boundary_distance_median",
                "pairwise_boundary_distances_json",
            ],
        )

    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    stratified_rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for method in METHODS:
        for variant in VARIANTS:
            items = analysis_items[method][variant]
            summaries[method][variant] = aggregate(items)
            for row in group_strata(items, "gt_density") + group_strata(items, "tissue") + area_strata(items):
                stratified_rows.append({"method": method, "gt_variant": variant, **row})
        raw_items = analysis_items[method]["original"]
        densities = [len(item["result"].gt_records) for item in raw_items]
        merge_fractions = [
            item["result"].counts["gt_merge"] / len(item["result"].gt_records)
            for item in raw_items
        ]
        rho, p_value = spearmanr(densities, merge_fractions)
        correlations[method] = {
            "spearman_rho_gt_count_vs_merge_gt_fraction": float(rho),
            "p_value": float(p_value),
            "sample_count": len(raw_items),
        }
    write_csv(args.output_dir / "stratified_results.csv", stratified_rows)

    for method in METHODS:
        observed = summaries[method]["converted"]["pq_counts"]
        expected = EXPECTED_CONVERTED_COUNTS[method]
        if observed["fn"] != expected["fn"] or observed["fp"] != expected["fp"]:
            raise AssertionError(
                f"converted-GT P0.3 cross-check failed for {method}: "
                f"observed FN/FP={observed['fn']}/{observed['fp']} "
                f"expected={expected['fn']}/{expected['fp']}"
            )
    if not all(
        summaries[method][variant]["self_checks"][check]
        for method in METHODS
        for variant in VARIANTS
        for check in ("fn_sum_equals_pq_fn", "fp_sum_equals_pq_fp")
    ):
        raise AssertionError("global FN/FP accounting self-check failed")

    primary = summaries["visual"]["original"]
    preregistered = {
        "primary_method": "visual",
        "primary_gt": "original",
        "r_merge": primary["r_merge"],
        "r_miss": primary["r_miss"],
        "absolute_gap": abs(primary["r_merge"] - primary["r_miss"]),
        "decision": decision(primary["r_merge"], primary["r_miss"]),
        "thresholds_unchanged": True,
        "precedence_note": "MIXED gap<0.10 clause applied before dominance clauses because the preregistered rows overlap",
    }

    fold_base = "https://huggingface.co/datasets/RationAI/PanNuke/resolve/b9a02ac/data"
    folds = {
        "fold1": parquet_audit(args.fold1, f"{fold_base}/fold1-00000-of-00001.parquet", args.source_revision),
        "fold2": parquet_audit(args.fold2, f"{fold_base}/fold2-00000-of-00001.parquet", args.source_revision),
        "fold3": parquet_audit(args.raw_fold3, f"{fold_base}/fold3-00000-of-00001.parquet", args.source_revision),
    }
    schema_equal = folds["fold1"]["logical_schema"] == folds["fold2"]["logical_schema"] == folds["fold3"]["logical_schema"]
    instance_total = sum(fold["instance_total"] for fold in folds.values())
    fold_availability = {
        "training_started": False,
        "inference_rerun": False,
        "folds": folds,
        "schema_equal_all_folds": schema_equal,
        "schema_expected_fields_pass": all(
            fold["logical_schema"]["columns"] == ["image", "instances", "categories", "tissue"]
            for fold in folds.values()
        ),
        "decoded_schema_expected_pass": all(
            fold["decoded_schema_probe"]["image_mode"] == "RGB"
            and fold["decoded_schema_probe"]["image_size"] == [256, 256]
            and fold["decoded_schema_probe"]["instance_modes"] == ["1"]
            and fold["decoded_schema_probe"]["instance_sizes"] == [[256, 256]]
            for fold in folds.values()
        ),
        "three_fold_instance_total": instance_total,
        "published_instance_total": 189744,
        "difference_from_published": instance_total - 189744,
        "fold12_available_for_pipeline_rebuild": bool(
            folds["fold1"]["available"] and folds["fold2"]["available"] and schema_equal
        ),
    }
    write_json(args.output_dir / "fold12_availability.json", fold_availability)

    distance_values = {
        method: [
            float(row["nearest_gt_pair_boundary_distance"])
            for row in merge_rows[method]
            if row["nearest_gt_pair_boundary_distance"] != "UNVERIFIED"
        ]
        for method in METHODS
    }
    result = {
        "training_started": False,
        "inference_rerun": False,
        "summaries": summaries,
        "preregistered_result": preregistered,
        "density_correlation": correlations,
        "merge_boundary_distance": {
            method: {
                "event_count_with_distance": len(values),
                "min": min(values) if values else "UNVERIFIED",
                "median": float(np.median(values)) if values else "UNVERIFIED",
                "mean": mean(values),
                "max": max(values) if values else "UNVERIFIED",
                "fraction_lt_3": sum(value < 3 for value in values) / len(values) if values else "UNVERIFIED",
            }
            for method, values in distance_values.items()
        },
        "converted_gt_crosscheck": {
            method: {
                "expected": EXPECTED_CONVERTED_COUNTS[method],
                "observed": {
                    "fn": summaries[method]["converted"]["pq_counts"]["fn"],
                    "fp": summaries[method]["converted"]["pq_counts"]["fp"],
                },
                "pass": True,
            }
            for method in METHODS
        },
        "fold12": fold_availability,
        "methodology_note": "Error attribution only; no method-effectiveness or route recommendation is made.",
    }
    write_json(args.output_dir / "error_summary.json", result)
    print("[V1_RESULT] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
