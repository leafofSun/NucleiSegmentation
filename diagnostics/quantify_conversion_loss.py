#!/usr/bin/env python3
"""Quantify PanNuke conversion losses (D1.2-D1.5), entirely on CPU."""

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

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.rebuild_index_mapping import (  # noqa: E402
    RawFoldParquet,
    load_mapping,
    sha256_file,
    write_json,
)
from evaluation.metrics_standard import aji_kumar_greedy  # noqa: E402
from evaluation.recompute_from_npy import load_gt_like_test_py  # noqa: E402


SIZE_BINS = (
    (1, 5, "[1,5)"),
    (5, 10, "[5,10)"),
    (10, 20, "[10,20)"),
    (20, 50, "[20,50)"),
    (50, 100, "[50,100)"),
    (100, 200, "[100,200)"),
    (200, math.inf, "[200,+inf)"),
)


def summarize(values: list[float | int]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "std": float(finite.std()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def manifest_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\t{sha256_file(path)}\t{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def build_instance_map(masks: list[np.ndarray], minimum_area: int = 1) -> tuple[np.ndarray, int]:
    output = np.zeros((256, 256), dtype=np.int32)
    overlap_pixels = 0
    next_id = 1
    for mask in masks:
        area = int(mask.sum())
        if area < minimum_area:
            continue
        overlap_pixels += int(np.logical_and(mask, output > 0).sum())
        output[mask] = next_id
        next_id += 1
    return output, overlap_pixels


def contour_statistics(mask: np.ndarray) -> tuple[int, int, int]:
    binary = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return 0, 0, 0
    hierarchy = hierarchy[0]
    holes = int(sum(parent >= 0 for parent in hierarchy[:, 3]))
    point_count = int(sum(len(contour) for contour in contours))
    components = int(cv2.connectedComponents(binary, connectivity=8)[0] - 1)
    return holes, point_count, components


def overlap_matrices(
    raw_masks: list[np.ndarray], reconstructed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_count = len(raw_masks)
    conv_count = int(reconstructed.max())
    intersections = np.zeros((raw_count, conv_count), dtype=np.int64)
    raw_areas = np.asarray([int(mask.sum()) for mask in raw_masks], dtype=np.int64)
    conv_areas = np.bincount(reconstructed.ravel(), minlength=conv_count + 1)[1:]
    for raw_index, mask in enumerate(raw_masks):
        counts = np.bincount(reconstructed[mask], minlength=conv_count + 1)
        intersections[raw_index] = counts[1:]
    unions = raw_areas[:, None] + conv_areas[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return intersections, iou, raw_areas, conv_areas


def unique_matches(intersections: np.ndarray, iou: np.ndarray) -> dict[int, tuple[int, float]]:
    if not intersections.size:
        return {}
    raw_indices, conv_indices = linear_sum_assignment(-iou)
    matches: dict[int, tuple[int, float]] = {}
    for raw_index, conv_index in zip(raw_indices, conv_indices, strict=True):
        if intersections[raw_index, conv_index] > 0:
            matches[int(raw_index)] = (int(conv_index), float(iou[raw_index, conv_index]))
    return matches


def converted_json_stats(json_path: Path) -> tuple[list[float], list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", [])
    areas: list[float] = []
    vertices: list[int] = []
    for annotation in annotations:
        areas.append(float(annotation.get("area", float("nan"))))
        polygons = annotation.get("segmentation", [])
        vertices.append(sum(len(polygon) // 2 for polygon in polygons))
    return areas, vertices


def nearest_other_boundary_distances(instance_map: np.ndarray) -> tuple[list[float], int]:
    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for instance_id in range(1, int(instance_map.max()) + 1):
        mask = (instance_map == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        coords = np.concatenate([contour[:, 0, ::-1] for contour in contours], axis=0)
        points.append(coords.astype(np.float64))
        labels.append(np.full(len(coords), instance_id, dtype=np.int32))
    if len(points) < 2:
        return [], 0
    coordinates = np.concatenate(points, axis=0)
    point_labels = np.concatenate(labels, axis=0)
    k = min(128, len(coordinates))
    distances, indices = cKDTree(coordinates).query(coordinates, k=k)
    if k == 1:
        return [], len(points)
    nearest: dict[int, float] = {int(value): math.inf for value in np.unique(point_labels)}
    for row in range(len(coordinates)):
        other = point_labels[indices[row]] != point_labels[row]
        if np.any(other):
            distance = float(distances[row][np.flatnonzero(other)[0]])
            label = int(point_labels[row])
            nearest[label] = min(nearest[label], distance)
    resolved = [value for value in nearest.values() if math.isfinite(value)]
    return resolved, len(nearest) - len(resolved)


def render_size_curve(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "NOT_FOUND: matplotlib"
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x, [row["retention_rate_unique_overlap"] for row in rows], marker="o", label="unique overlap")
    axis.plot(x, [row["retention_rate_iou_gt_0_5"] for row in rows], marker="s", label="IoU > 0.5")
    axis.set_xticks(x, [row["area_bin"] for row in rows], rotation=25, ha="right")
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("retention rate")
    axis.set_xlabel("raw instance area (pixels)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return "CREATED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-matched-count", type=int, default=2607)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.raw_parquet, args.mapping):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.converted_dir.is_dir():
        raise FileNotFoundError(args.converted_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    if len(mapping) != args.expected_matched_count:
        raise RuntimeError(f"mapping count {len(mapping)} != {args.expected_matched_count}")
    if any(int(entry["max_abs_diff"]) != 0 for entry in mapping):
        raise RuntimeError("image pixels are not exact; D1 stop gate applies")
    raw_fold = RawFoldParquet(args.raw_parquet)
    mapped_by_raw = {int(entry["raw_index"]): str(entry["sample_id"]) for entry in mapping}
    if len(mapped_by_raw) != len(mapping):
        raise RuntimeError("mapping is not a raw-index bijection")

    json_paths = [args.converted_dir / f"{entry['sample_id']}.json" for entry in mapping]
    if not all(path.is_file() for path in json_paths):
        raise FileNotFoundError("one or more converted JSON files are missing")
    config = {
        "training_started": False,
        "inference_rerun": False,
        "cpu_only": True,
        "raw_data_path": str(args.raw_parquet.resolve()),
        "raw_data_sha256": sha256_file(args.raw_parquet),
        "converted_data_path": str(args.converted_dir.resolve()),
        "converted_gt_json_manifest_sha256": manifest_sha256(json_paths, args.converted_dir),
        "mapping_path": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "matched_count": len(mapping),
        "raw_instance_semantics": "one binary mask per original class-channel instance, reconstructed from the verified mirror schema",
        "d1_3_n_orig_semantics": "8-connected components after five-class binary foreground merge, exactly as preregistered",
        "size_retention_primary": "one-to-one Hungarian assignment maximizing IoU, retained when intersection > 0",
        "size_retention_secondary": "same one-to-one assignment and strict IoU > 0.5",
        "converted_decoder": "evaluation.recompute_from_npy.load_gt_like_test_py; exact port of test.py:880-926",
    }
    print("[DIAG_CONFIG] " + json.dumps(config, sort_keys=True), flush=True)
    write_json(args.output_dir / "diagnostic_config.json", config)

    removed_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    bin_totals: dict[str, Counter[str]] = {label: Counter() for _, _, label in SIZE_BINS}
    all_annotation_areas: list[float] = []
    all_rendered_areas: list[int] = []
    all_vertex_counts: list[int] = []
    all_boundary_points: list[int] = []
    all_boundary_distances: list[float] = []
    unresolved_boundary_instances = 0
    global_orig_fg = 0
    global_conv_fg = 0
    global_intersection = 0
    global_union = 0
    total_raw_instances = 0
    total_raw_overlap_pixels = 0
    topology = Counter()

    for raw_index, record in enumerate(raw_fold):
        raw_masks = record.instance_masks()
        raw_binary = np.logical_or.reduce(raw_masks) if raw_masks else np.zeros((256, 256), dtype=bool)
        raw_instance_count = len(raw_masks)
        total_raw_instances += raw_instance_count
        if raw_index not in mapped_by_raw:
            contours, _ = cv2.findContours(raw_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            eligible = [contour for contour in contours if cv2.contourArea(contour) >= 10 and len(contour) >= 3]
            if not raw_binary.any():
                reason = "GT_EMPTY"
            elif not eligible:
                reason = "ALL_EXTERNAL_CONTOURS_AREA_LT_10_OR_INVALID"
            else:
                reason = "OTHER_UNVERIFIED"
            removed_rows.append(
                {
                    "raw_index": raw_index,
                    "organ_type": record.tissue_name,
                    "reason": reason,
                    "raw_instance_count": raw_instance_count,
                    "binary_connected_component_count": int(cv2.connectedComponents(raw_binary.astype(np.uint8), connectivity=8)[0] - 1),
                    "foreground_pixels": int(raw_binary.sum()),
                    "eligible_external_contour_count": len(eligible),
                }
            )
            continue

        sample_id = mapped_by_raw[raw_index]
        json_path = args.converted_dir / f"{sample_id}.json"
        reconstructed, converted_organ = load_gt_like_test_py(json_path)
        raw_map, overlap_pixels = build_instance_map(raw_masks)
        total_raw_overlap_pixels += overlap_pixels
        binary_cc_count = int(cv2.connectedComponents(raw_binary.astype(np.uint8), connectivity=8)[0] - 1)
        conv_count = int(reconstructed.max())
        intersections, pair_iou, raw_areas, conv_areas = overlap_matrices(raw_masks, reconstructed)
        matches = unique_matches(intersections, pair_iou)

        for raw_instance_index, area in enumerate(raw_areas):
            label = next(label for low, high, label in SIZE_BINS if low <= area < high)
            bin_totals[label]["raw"] += 1
            if raw_instance_index in matches:
                bin_totals[label]["retained_overlap"] += 1
                if matches[raw_instance_index][1] > 0.5:
                    bin_totals[label]["retained_iou_gt_0_5"] += 1

        merge_conv_count = int(np.sum(np.sum(intersections > 0, axis=0) > 1)) if conv_count else 0
        merge_raw_involved = int(np.sum(np.any(intersections[:, np.sum(intersections > 0, axis=0) > 1] > 0, axis=1))) if merge_conv_count else 0
        topology["converted_instances_overlapping_multiple_raw"] += merge_conv_count
        topology["raw_instances_involved_in_merges"] += merge_raw_involved

        image_holes = 0
        image_holes_filled = 0
        multipart_instances = 0
        multipart_split_across_converted = 0
        for instance_index, mask in enumerate(raw_masks):
            holes, boundary_points, components = contour_statistics(mask)
            all_boundary_points.append(boundary_points)
            image_holes += holes
            if holes:
                hole_pixels = np.logical_and(binary_fill_holes(mask), ~mask)
                match = matches.get(instance_index)
                if match is not None and np.any(np.logical_and(hole_pixels, reconstructed == match[0] + 1)):
                    image_holes_filled += 1
            if components > 1:
                multipart_instances += 1
                overlapping_converted = int(np.sum(intersections[instance_index] > 0))
                if overlapping_converted > 1:
                    multipart_split_across_converted += 1
        topology["raw_holes"] += image_holes
        topology["raw_instances_with_hole_fill_in_reconstruction"] += image_holes_filled
        topology["raw_multipart_instances"] += multipart_instances
        topology["raw_multipart_split_across_converted"] += multipart_split_across_converted

        annotation_areas, vertex_counts = converted_json_stats(json_path)
        all_annotation_areas.extend(annotation_areas)
        all_rendered_areas.extend(int(value) for value in conv_areas)
        all_vertex_counts.extend(vertex_counts)
        distances, unresolved = nearest_other_boundary_distances(reconstructed)
        all_boundary_distances.extend(distances)
        unresolved_boundary_instances += unresolved

        raw_fg = int(raw_binary.sum())
        conv_fg = int((reconstructed > 0).sum())
        fg_intersection = int(np.logical_and(raw_binary, reconstructed > 0).sum())
        fg_union = int(np.logical_or(raw_binary, reconstructed > 0).sum())
        global_orig_fg += raw_fg
        global_conv_fg += conv_fg
        global_intersection += fg_intersection
        global_union += fg_union
        per_image.append(
            {
                "sample_id": sample_id,
                "raw_index": raw_index,
                "raw_organ_type": record.tissue_name,
                "converted_organ_type": converted_organ,
                "raw_channel_instance_count": raw_instance_count,
                "n_orig_binary_connected_components": binary_cc_count,
                "n_conv": conv_count,
                "d1_3_loss": binary_cc_count - conv_count,
                "raw_foreground_pixels": raw_fg,
                "reconstructed_foreground_pixels": conv_fg,
                "foreground_pixel_ratio_conv_over_raw": conv_fg / raw_fg if raw_fg else math.nan,
                "foreground_iou": fg_intersection / fg_union if fg_union else 1.0,
                "aji_kumar_raw_instances_vs_reconstructed": aji_kumar_greedy(raw_map, reconstructed),
                "raw_instances_unique_overlap_retained": len(matches),
                "raw_instances_iou_gt_0_5_retained": sum(value[1] > 0.5 for value in matches.values()),
                "converted_instances_overlapping_multiple_raw": merge_conv_count,
                "raw_instances_involved_in_merges": merge_raw_involved,
                "raw_holes": image_holes,
                "raw_holes_filled_in_reconstruction": image_holes_filled,
                "raw_multipart_instances": multipart_instances,
                "raw_multipart_split_across_converted": multipart_split_across_converted,
                "raw_instance_overlap_pixels": overlap_pixels,
                "converted_annotation_min_contour_area": min(annotation_areas) if annotation_areas else math.nan,
                "converted_rendered_min_pixel_area": int(conv_areas.min()) if conv_areas.size else math.nan,
                "converted_min_polygon_vertices": min(vertex_counts) if vertex_counts else math.nan,
            }
        )
        if len(per_image) % 100 == 0 or len(per_image) == len(mapping):
            print(f"[PROGRESS] matched_images={len(per_image)}/{len(mapping)}", flush=True)

    per_image.sort(key=lambda row: row["sample_id"])
    removed_rows.sort(key=lambda row: row["raw_index"])
    write_csv(args.output_dir / "per_image_conversion_loss.csv", per_image)
    write_csv(
        args.output_dir / "removed_samples.csv",
        removed_rows,
        ["raw_index", "organ_type", "reason", "raw_instance_count", "binary_connected_component_count", "foreground_pixels", "eligible_external_contour_count"],
    )

    size_rows: list[dict[str, Any]] = []
    for low, high, label in SIZE_BINS:
        counts = bin_totals[label]
        raw_count = counts["raw"]
        size_rows.append(
            {
                "area_bin": label,
                "lower_inclusive": low,
                "upper_exclusive": "inf" if math.isinf(high) else int(high),
                "raw_instance_count": raw_count,
                "converted_retained_unique_overlap_count": counts["retained_overlap"],
                "retention_rate_unique_overlap": counts["retained_overlap"] / raw_count if raw_count else math.nan,
                "converted_retained_iou_gt_0_5_count": counts["retained_iou_gt_0_5"],
                "retention_rate_iou_gt_0_5": counts["retained_iou_gt_0_5"] / raw_count if raw_count else math.nan,
            }
        )
    write_csv(args.output_dir / "size_retention_curve.csv", size_rows)
    plot_status = render_size_curve(args.output_dir / "size_retention_curve.png", size_rows)

    d1_losses = [int(row["d1_3_loss"]) for row in per_image]
    removed_organs = Counter(row["organ_type"] for row in removed_rows)
    removed_reasons = Counter(row["reason"] for row in removed_rows)
    matched_raw_instances = sum(int(row["raw_channel_instance_count"]) for row in per_image)
    raw_lt20 = sum(row["raw_instance_count"] for row in size_rows[:3])
    summary = {
        "training_started": False,
        "inference_rerun": False,
        "raw_fold_sample_count": len(raw_fold),
        "converted_sample_count": len(mapping),
        "removed_sample_count": len(removed_rows),
        "removed_reason_distribution": dict(sorted(removed_reasons.items())),
        "removed_organ_distribution": dict(sorted(removed_organs.items())),
        "raw_fold_original_instance_total": total_raw_instances,
        "matched_raw_original_instance_total": matched_raw_instances,
        "d1_3_preregistered": {
            "n_orig_binary_cc_total": sum(int(row["n_orig_binary_connected_components"]) for row in per_image),
            "n_conv_total": sum(int(row["n_conv"]) for row in per_image),
            "absolute_loss": sum(d1_losses),
            "relative_loss_percent": 100.0 * sum(d1_losses) / sum(int(row["n_orig_binary_connected_components"]) for row in per_image),
            "per_image_loss_distribution": summarize(d1_losses),
            "loss_gt_zero_image_fraction": sum(value > 0 for value in d1_losses) / len(d1_losses),
            "loss_gt_10_image_count": sum(value > 10 for value in d1_losses),
        },
        "size": {
            "raw_instances_area_lt20": raw_lt20,
            "raw_instances_area_lt20_fraction": raw_lt20 / matched_raw_instances,
            "contradiction_gate_area_lt20_gt_15_percent": raw_lt20 / matched_raw_instances > 0.15,
            "primary_retention_definition": config["size_retention_primary"],
            "secondary_retention_definition": config["size_retention_secondary"],
            "curve_plot_status": plot_status,
        },
        "pixel": {
            "global_foreground_iou": global_intersection / global_union,
            "global_foreground_pixel_ratio_conv_over_raw": global_conv_fg / global_orig_fg,
            "per_image_foreground_iou": summarize([float(row["foreground_iou"]) for row in per_image]),
            "per_image_foreground_pixel_ratio": summarize([float(row["foreground_pixel_ratio_conv_over_raw"]) for row in per_image]),
            "per_image_aji_kumar": summarize([float(row["aji_kumar_raw_instances_vs_reconstructed"]) for row in per_image]),
        },
        "topology": {
            **dict(topology),
            "raw_instance_overlap_pixels_total": total_raw_overlap_pixels,
            "raw_boundary_point_count_distribution": summarize(all_boundary_points),
            "converted_polygon_vertex_count_distribution": summarize(all_vertex_counts),
            "nearest_other_converted_boundary_distance_distribution": summarize(all_boundary_distances),
            "nearest_other_boundary_unresolved_instance_count_k128": unresolved_boundary_instances,
        },
        "internal_consistency": {
            "converted_annotation_contour_area_distribution": summarize(all_annotation_areas),
            "converted_rendered_pixel_area_distribution": summarize(all_rendered_areas),
            "converted_polygon_vertex_distribution": summarize(all_vertex_counts),
            "minimum_annotation_contour_area_ge_10": min(all_annotation_areas) >= 10,
            "minimum_polygon_vertices_ge_3": min(all_vertex_counts) >= 3,
        },
    }
    write_json(args.output_dir / "conversion_summary.json", summary)
    write_json(args.output_dir / "topology_summary.json", summary["topology"])
    print("[CONVERSION_RESULT] " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
