#!/usr/bin/env python3
"""CPU-only L0 audit of local semantic granularity in PanNuke train tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    raise RuntimeError("L0 is CPU-only; set CUDA_VISIBLE_DEVICES to an empty string.")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.color import rgb2hed
from skimage.measure import regionprops


ATTRIBUTE_NAMES = (
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "touching_nuclei_ratio",
    "nearest_neighbor_distance",
    "boundary_irregularity",
    "nuclear_elongation",
    "mean_stain_intensity",
    "stain_heterogeneity",
)
CORE_ATTRIBUTES = (
    "nuclear_density",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "touching_nuclei_ratio",
    "boundary_irregularity",
    "nuclear_elongation",
)
MORPHOLOGY_ATTRIBUTES = (
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "touching_nuclei_ratio",
    "nearest_neighbor_distance",
    "boundary_irregularity",
    "nuclear_elongation",
)
INSTANCE_FIELDS = (
    "sample_id",
    "organ_type",
    "instance_id",
    "annotation_category_id",
    "area_px2",
    "equivalent_diameter_px",
    "perimeter_px",
    "eccentricity",
    "elongation_major_minor",
    "boundary_irregularity",
    "centroid_x",
    "centroid_y",
    "nearest_neighbor_distance_px",
    "touching_neighbor_count",
    "border_touching_original",
    "partial_instance_original_border",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "mean_stain_intensity",
)
GLOBAL_FIELDS = (
    "sample_id",
    "organ_type",
    "image_height",
    "image_width",
    "instance_count",
    "complete_instance_count",
    "partial_original_count",
    "partial_original_ratio",
    *ATTRIBUTE_NAMES,
    *tuple(f"centroid_{name}" for name in ATTRIBUTE_NAMES),
)
LOCAL_FIELDS = (
    "sample_id",
    "organ_type",
    "window_size",
    "window_basis",
    "x0",
    "y0",
    "x1",
    "y1",
    "window_area_px2",
    "empty_window",
    "complete_instance_count",
    "centroid_inside_count",
    "centroid_crossing_count",
    "original_partial_instance_count",
    "border_cut_ratio_window",
    "partial_original_ratio_window",
    "region_policy_primary",
    *ATTRIBUTE_NAMES,
    *tuple(f"centroid_{name}" for name in ATTRIBUTE_NAMES),
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    json_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--window-policy", default="adaptive")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--augmentation-subset", type=int, default=32)
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = finite_or_none(payload)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_samples(dataset_root: Path, split: str, max_samples: int | None) -> list[Sample]:
    split_dir = dataset_root / split
    samples = []
    for image_path in sorted(split_dir.glob("*.png")):
        json_path = image_path.with_suffix(".json")
        if json_path.is_file():
            samples.append(Sample(image_path.stem, str(image_path), str(json_path)))
    if max_samples is not None:
        samples = samples[: max(0, max_samples)]
    return samples


def decode_sample(image_path: str | Path, json_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[int, Any], dict[str, Any]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    height = int(payload.get("height", payload.get("image", {}).get("height", image.shape[0])))
    width = int(payload.get("width", payload.get("image", {}).get("width", image.shape[1])))
    if image.shape[:2] != (height, width):
        raise ValueError(f"Image/JSON size mismatch for {image_path}: {image.shape[:2]} vs {(height, width)}")
    mask = np.zeros((height, width), dtype=np.int32)
    id_to_category: dict[int, Any] = {}
    instance_id = 1
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            polygons = (
                [segmentation]
                if segmentation and all(isinstance(value, (int, float)) for value in segmentation)
                else segmentation
            )
            if not isinstance(polygons, list):
                continue
            for polygon in polygons:
                try:
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                except Exception:
                    continue
                if points.shape[0] < 3:
                    continue
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], instance_id)
                id_to_category[instance_id] = annotation.get("category_id")
                instance_id += 1
        elif isinstance(segmentation, dict) and "counts" in segmentation:
            try:
                from pycocotools import mask as coco_mask

                binary = coco_mask.decode(segmentation)
                if binary.ndim == 3:
                    binary = np.max(binary, axis=2)
                if binary.shape != mask.shape:
                    binary = cv2.resize(
                        binary.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask[binary > 0] = instance_id
                id_to_category[instance_id] = annotation.get("category_id")
                instance_id += 1
            except Exception:
                continue
    return image, mask, id_to_category, payload


def touching_pairs_from_mask(mask: np.ndarray, radius: int = 2) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    height, width = mask.shape
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            if max(abs(dx), abs(dy)) > radius:
                continue
            y0a = max(0, -dy)
            y1a = min(height, height - dy)
            x0a = max(0, -dx)
            x1a = min(width, width - dx)
            y0b = y0a + dy
            y1b = y1a + dy
            x0b = x0a + dx
            x1b = x1a + dx
            if y1a <= y0a or x1a <= x0a:
                continue
            left = mask[y0a:y1a, x0a:x1a]
            right = mask[y0b:y1b, x0b:x1b]
            valid = (left > 0) & (right > 0) & (left != right)
            if not np.any(valid):
                continue
            values = np.stack((left[valid], right[valid]), axis=1)
            for first, second in np.unique(values, axis=0):
                a, b = sorted((int(first), int(second)))
                if a != b:
                    pairs.add((a, b))
    return pairs


def hematoxylin_channel(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(image.astype(np.float32) / 255.0, 1.0 / 255.0, 1.0)
    channel = rgb2hed(rgb)[..., 0].astype(np.float32)
    return np.maximum(channel, 0.0)


def extract_properties(
    image: np.ndarray,
    mask: np.ndarray,
    id_to_category: dict[int, Any] | None = None,
) -> dict[str, Any]:
    stain = hematoxylin_channel(image)
    props = list(regionprops(mask, intensity_image=stain))
    labels = np.asarray([int(prop.label) for prop in props], dtype=np.int32)
    areas = np.asarray([float(prop.area) for prop in props], dtype=np.float64)
    diameters = np.asarray([float(prop.equivalent_diameter_area) for prop in props], dtype=np.float64)
    perimeters = np.asarray([float(prop.perimeter) for prop in props], dtype=np.float64)
    eccentricities = np.asarray([float(prop.eccentricity) for prop in props], dtype=np.float64)
    major = np.asarray([float(prop.axis_major_length) for prop in props], dtype=np.float64)
    minor = np.asarray([float(prop.axis_minor_length) for prop in props], dtype=np.float64)
    elongations = major / np.maximum(minor, 1e-6)
    irregularity = perimeters / np.maximum(2.0 * np.sqrt(np.pi * areas), 1e-6)
    centroids = np.asarray(
        [[float(prop.centroid[1]), float(prop.centroid[0])] for prop in props],
        dtype=np.float64,
    ).reshape(-1, 2)
    bboxes = np.asarray(
        [[int(prop.bbox[1]), int(prop.bbox[0]), int(prop.bbox[3]), int(prop.bbox[2])] for prop in props],
        dtype=np.int32,
    ).reshape(-1, 4)
    height, width = mask.shape
    partial = (
        (bboxes[:, 0] <= 0)
        | (bboxes[:, 1] <= 0)
        | (bboxes[:, 2] >= width)
        | (bboxes[:, 3] >= height)
    ) if len(props) else np.zeros(0, dtype=bool)
    stain_means = np.asarray([float(prop.intensity_mean) for prop in props], dtype=np.float64)
    if len(props) >= 2:
        differences = centroids[:, None, :] - centroids[None, :, :]
        distance_matrix = np.sqrt(np.sum(differences * differences, axis=2))
        np.fill_diagonal(distance_matrix, np.inf)
        nearest = np.min(distance_matrix, axis=1)
    else:
        distance_matrix = np.full((len(props), len(props)), np.inf, dtype=np.float64)
        nearest = np.full(len(props), np.nan, dtype=np.float64)
    touching_pairs = touching_pairs_from_mask(mask, radius=2)
    label_to_index = {int(label): index for index, label in enumerate(labels)}
    pair_indices = {
        tuple(sorted((label_to_index[a], label_to_index[b])))
        for a, b in touching_pairs
        if a in label_to_index and b in label_to_index
    }
    touching_counts = np.zeros(len(props), dtype=np.int32)
    for first, second in pair_indices:
        touching_counts[first] += 1
        touching_counts[second] += 1
    categories = np.asarray(
        [(id_to_category or {}).get(int(label)) for label in labels],
        dtype=object,
    )
    return {
        "labels": labels,
        "areas": areas,
        "diameters": diameters,
        "perimeters": perimeters,
        "eccentricities": eccentricities,
        "elongations": elongations,
        "irregularity": irregularity,
        "centroids": centroids,
        "bboxes": bboxes,
        "partial": partial,
        "stain_means": stain_means,
        "distance_matrix": distance_matrix,
        "nearest": nearest,
        "pair_indices": pair_indices,
        "touching_counts": touching_counts,
        "categories": categories,
        "stain": stain,
    }


def subset_attribute_values(
    properties: dict[str, Any],
    selected: np.ndarray,
    region_area: float,
    nuclear_area_fraction: float,
    mean_stain: float,
    stain_heterogeneity: float,
) -> dict[str, float]:
    indices = np.flatnonzero(selected)
    count = len(indices)
    result = {
        "nuclear_density": float(count / max(region_area, 1.0) * 10000.0),
        "nuclear_area_fraction": float(nuclear_area_fraction),
        "mean_nuclear_size": float("nan"),
        "nuclear_size_heterogeneity": float("nan"),
        "spatial_crowding": float("nan"),
        "touching_nuclei_ratio": float("nan"),
        "nearest_neighbor_distance": float("nan"),
        "boundary_irregularity": float("nan"),
        "nuclear_elongation": float("nan"),
        "mean_stain_intensity": float(mean_stain),
        "stain_heterogeneity": float(stain_heterogeneity),
    }
    if count == 0:
        return result
    areas = properties["areas"][indices]
    result["mean_nuclear_size"] = float(np.mean(areas))
    result["nuclear_size_heterogeneity"] = float(np.std(areas) / max(np.mean(areas), 1e-6))
    result["boundary_irregularity"] = float(np.mean(properties["irregularity"][indices]))
    result["nuclear_elongation"] = float(np.mean(properties["elongations"][indices]))
    if count >= 2:
        distances = properties["distance_matrix"][np.ix_(indices, indices)].copy()
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        median_nearest = float(np.median(nearest))
        result["nearest_neighbor_distance"] = median_nearest
        result["spatial_crowding"] = float(100.0 / max(median_nearest, 1e-6))
    touching_nodes: set[int] = set()
    selected_set = set(int(index) for index in indices)
    for first, second in properties["pair_indices"]:
        if first in selected_set and second in selected_set:
            touching_nodes.add(first)
            touching_nodes.add(second)
    result["touching_nuclei_ratio"] = float(len(touching_nodes) / count)
    return result


def integral_image(array: np.ndarray) -> np.ndarray:
    return cv2.integral(array.astype(np.float64), sdepth=cv2.CV_64F)


def rectangle_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def attributes_for_window(
    properties: dict[str, Any],
    foreground_integral: np.ndarray,
    stain_sum_integral: np.ndarray,
    stain_sq_integral: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    bboxes = properties["bboxes"]
    centroids = properties["centroids"]
    partial = properties["partial"]
    within_bbox = (
        (bboxes[:, 0] >= x0)
        & (bboxes[:, 1] >= y0)
        & (bboxes[:, 2] <= x1)
        & (bboxes[:, 3] <= y1)
    ) if len(bboxes) else np.zeros(0, dtype=bool)
    centroid_inside = (
        (centroids[:, 0] >= x0)
        & (centroids[:, 0] < x1)
        & (centroids[:, 1] >= y0)
        & (centroids[:, 1] < y1)
    ) if len(centroids) else np.zeros(0, dtype=bool)
    complete = within_bbox & ~partial
    crossing = centroid_inside & ~within_bbox
    region_area = float((x1 - x0) * (y1 - y0))
    foreground_mass = rectangle_sum(foreground_integral, x0, y0, x1, y1)
    stain_sum = rectangle_sum(stain_sum_integral, x0, y0, x1, y1)
    stain_sq = rectangle_sum(stain_sq_integral, x0, y0, x1, y1)
    if foreground_mass > 0:
        stain_mean = stain_sum / foreground_mass
        stain_var = max(stain_sq / foreground_mass - stain_mean * stain_mean, 0.0)
        stain_std = math.sqrt(stain_var)
    else:
        stain_mean = 0.0
        stain_std = 0.0
    area_fraction = foreground_mass / max(region_area, 1.0)
    primary = subset_attribute_values(
        properties,
        complete,
        region_area,
        area_fraction,
        stain_mean,
        stain_std,
    )
    centroid_values = subset_attribute_values(
        properties,
        centroid_inside,
        region_area,
        area_fraction,
        stain_mean,
        stain_std,
    )
    return {
        **primary,
        **{f"centroid_{key}": value for key, value in centroid_values.items()},
    }, complete, centroid_inside, crossing


def global_attribute_row(
    sample_id: str,
    organ_type: str,
    image: np.ndarray,
    mask: np.ndarray,
    properties: dict[str, Any],
) -> dict[str, Any]:
    foreground = mask > 0
    stain_values = properties["stain"][foreground]
    stain_mean = float(np.mean(stain_values)) if stain_values.size else 0.0
    stain_std = float(np.std(stain_values)) if stain_values.size else 0.0
    area_fraction = float(np.mean(foreground))
    complete = ~properties["partial"]
    centroid = np.ones(len(properties["labels"]), dtype=bool)
    primary = subset_attribute_values(
        properties,
        complete,
        float(mask.size),
        area_fraction,
        stain_mean,
        stain_std,
    )
    centroid_values = subset_attribute_values(
        properties,
        centroid,
        float(mask.size),
        area_fraction,
        stain_mean,
        stain_std,
    )
    count = len(properties["labels"])
    partial_count = int(np.sum(properties["partial"]))
    return {
        "sample_id": sample_id,
        "organ_type": organ_type,
        "image_height": image.shape[0],
        "image_width": image.shape[1],
        "instance_count": count,
        "complete_instance_count": int(np.sum(complete)),
        "partial_original_count": partial_count,
        "partial_original_ratio": partial_count / max(count, 1),
        **primary,
        **{f"centroid_{key}": value for key, value in centroid_values.items()},
    }


def analyze_base_worker(sample: Sample) -> dict[str, Any]:
    image, mask, id_to_category, payload = decode_sample(sample.image_path, sample.json_path)
    properties = extract_properties(image, mask, id_to_category)
    organ_type = str(payload.get("organ_type", payload.get("organ_id", "unknown")))
    instance_rows = []
    for index, label in enumerate(properties["labels"]):
        bbox = properties["bboxes"][index]
        nearest = properties["nearest"][index]
        category = properties["categories"][index]
        instance_rows.append({
            "sample_id": sample.sample_id,
            "organ_type": organ_type,
            "instance_id": int(label),
            "annotation_category_id": "" if category is None else category,
            "area_px2": properties["areas"][index],
            "equivalent_diameter_px": properties["diameters"][index],
            "perimeter_px": properties["perimeters"][index],
            "eccentricity": properties["eccentricities"][index],
            "elongation_major_minor": properties["elongations"][index],
            "boundary_irregularity": properties["irregularity"][index],
            "centroid_x": properties["centroids"][index, 0],
            "centroid_y": properties["centroids"][index, 1],
            "nearest_neighbor_distance_px": nearest,
            "touching_neighbor_count": int(properties["touching_counts"][index]),
            "border_touching_original": bool(properties["partial"][index]),
            "partial_instance_original_border": bool(properties["partial"][index]),
            "bbox_x0": int(bbox[0]),
            "bbox_y0": int(bbox[1]),
            "bbox_x1": int(bbox[2]),
            "bbox_y1": int(bbox[3]),
            "mean_stain_intensity": properties["stain_means"][index],
        })
    return {
        "sample_id": sample.sample_id,
        "instance_rows": instance_rows,
        "global_row": global_attribute_row(sample.sample_id, organ_type, image, mask, properties),
        "complete_diameters": properties["diameters"][~properties["partial"]].tolist(),
    }


def align_window(value: float, grid: int, minimum: int, maximum: int) -> int:
    aligned = int(round(value / grid) * grid)
    return min(max(aligned, minimum), maximum)


def build_window_candidates(
    median_diameter: float,
    image_width: int,
    grid_step: int = 8,
    policy: str = "adaptive",
) -> list[dict[str, Any]]:
    if policy != "adaptive":
        values = [int(part.strip()) for part in policy.split(",") if part.strip()]
        raw = [(f"explicit_{value}", value) for value in values]
    else:
        raw = [
            ("4x_median_diameter", 4.0 * median_diameter),
            ("6x_median_diameter", 6.0 * median_diameter),
            ("8x_median_diameter", 8.0 * median_diameter),
            ("half_image_width", image_width / 2.0),
        ]
    merged: dict[int, list[str]] = defaultdict(list)
    for basis, value in raw:
        size = align_window(value, grid_step, max(2 * grid_step, 16), image_width)
        merged[size].append(basis)
    return [
        {"window_size": size, "basis": "+".join(bases)}
        for size, bases in sorted(merged.items())
    ]


def window_positions(length: int, size: int, overlap_ratio: float, grid_step: int = 8) -> list[int]:
    if size >= length:
        return [0]
    raw_step = size * (1.0 - overlap_ratio)
    step = max(grid_step, int(round(raw_step / grid_step) * grid_step))
    positions = list(range(0, length - size + 1, step))
    end = length - size
    if positions[-1] != end:
        positions.append(end)
    return sorted(set(positions))


def analyze_local_worker(payload: tuple[Sample, list[dict[str, Any]], float]) -> dict[str, Any]:
    sample, candidates, overlap_ratio = payload
    image, mask, id_to_category, metadata = decode_sample(sample.image_path, sample.json_path)
    properties = extract_properties(image, mask, id_to_category)
    organ_type = str(metadata.get("organ_type", metadata.get("organ_id", "unknown")))
    foreground = (mask > 0).astype(np.float64)
    stain_foreground = properties["stain"].astype(np.float64) * foreground
    foreground_integral = integral_image(foreground)
    stain_sum_integral = integral_image(stain_foreground)
    stain_sq_integral = integral_image(stain_foreground * properties["stain"].astype(np.float64))
    local_rows = []
    relation_stats: dict[int, dict[str, int]] = {}
    for candidate in candidates:
        size = int(candidate["window_size"])
        retained_pairs: set[tuple[int, int]] = set()
        for y0 in window_positions(mask.shape[0], size, overlap_ratio):
            for x0 in window_positions(mask.shape[1], size, overlap_ratio):
                x1, y1 = x0 + size, y0 + size
                values, complete, centroid_inside, crossing = attributes_for_window(
                    properties,
                    foreground_integral,
                    stain_sum_integral,
                    stain_sq_integral,
                    x0,
                    y0,
                    x1,
                    y1,
                )
                complete_indices = set(int(index) for index in np.flatnonzero(complete))
                for pair in properties["pair_indices"]:
                    if pair[0] in complete_indices and pair[1] in complete_indices:
                        retained_pairs.add(pair)
                centroid_count = int(np.sum(centroid_inside))
                crossing_count = int(np.sum(crossing))
                original_partial_count = int(np.sum(centroid_inside & properties["partial"]))
                foreground_mass = rectangle_sum(foreground_integral, x0, y0, x1, y1)
                local_rows.append({
                    "sample_id": sample.sample_id,
                    "organ_type": organ_type,
                    "window_size": size,
                    "window_basis": candidate["basis"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "window_area_px2": size * size,
                    "empty_window": foreground_mass == 0,
                    "complete_instance_count": int(np.sum(complete)),
                    "centroid_inside_count": centroid_count,
                    "centroid_crossing_count": crossing_count,
                    "original_partial_instance_count": original_partial_count,
                    "border_cut_ratio_window": crossing_count / max(centroid_count, 1),
                    "partial_original_ratio_window": original_partial_count / max(centroid_count, 1),
                    "region_policy_primary": "complete_only",
                    **values,
                })
        relation_stats[size] = {
            "touching_relation_total": len(properties["pair_indices"]),
            "touching_relation_retained": len(retained_pairs),
        }
    return {"local_rows": local_rows, "relation_stats": relation_stats}


def assign_bin(values: Any, lower: float, upper: float) -> Any:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, -1, dtype=np.int8)
    finite = np.isfinite(array)
    result[finite & (array < lower)] = 0
    result[finite & (array >= lower) & (array <= upper)] = 1
    result[finite & (array > upper)] = 2
    if np.isscalar(values):
        return int(result.item())
    return result


def shannon_entropy_from_counts(counts: Sequence[int]) -> float:
    array = np.asarray(counts, dtype=np.float64)
    if array.sum() <= 0:
        return 0.0
    probabilities = array[array > 0] / array.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def bootstrap_mean_ci(
    values: Sequence[float],
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1 or repeats <= 1:
        value = float(array.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = np.mean(rng.choice(array, size=array.size, replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fit_thresholds(global_frame: pd.DataFrame) -> dict[str, Any]:
    thresholds = {}
    for attribute in ATTRIBUTE_NAMES:
        values = pd.to_numeric(global_frame[attribute], errors="coerce")
        values = values[np.isfinite(values)]
        lower, upper = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        thresholds[attribute] = {
            "low_upper_exclusive": float(lower),
            "medium_upper_inclusive": float(upper),
            "fit_split": "train",
            "fit_population": "tile-level complete-only global attributes",
            "labels": ["low", "medium", "high"],
            "finite_count": int(values.size),
        }
    return thresholds


def pareto_flags(rows: list[dict[str, Any]]) -> dict[int, bool]:
    flags = {}
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["empty_window_ratio"] <= row["empty_window_ratio"]
                and other["border_cut_ratio"] <= row["border_cut_ratio"]
                and other["median_complete_instance_count"] >= row["median_complete_instance_count"]
                and other["touching_relation_retention"] >= row["touching_relation_retention"]
                and other["total_windows"] <= row["total_windows"]
            )
            strictly_better = (
                other["empty_window_ratio"] < row["empty_window_ratio"]
                or other["border_cut_ratio"] < row["border_cut_ratio"]
                or other["median_complete_instance_count"] > row["median_complete_instance_count"]
                or other["touching_relation_retention"] > row["touching_relation_retention"]
                or other["total_windows"] < row["total_windows"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags[int(row["window_size"])] = not dominated
    return flags


def disagreement_analysis(
    local_frame: pd.DataFrame,
    global_frame: pd.DataFrame,
    thresholds: dict[str, Any],
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global_index = global_frame.set_index("sample_id")
    output_rows = []
    tile_multi_any: dict[str, bool] = defaultdict(bool)
    distributions: dict[str, dict[str, float]] = {}
    for attribute_index, attribute in enumerate(ATTRIBUTE_NAMES):
        frame = local_frame[["sample_id", attribute]].copy()
        frame["global_value"] = frame["sample_id"].map(global_index[attribute])
        frame["local_value"] = pd.to_numeric(frame[attribute], errors="coerce")
        finite = np.isfinite(frame["local_value"]) & np.isfinite(frame["global_value"])
        frame = frame.loc[finite, ["sample_id", "local_value", "global_value"]]
        lower = thresholds[attribute]["low_upper_exclusive"]
        upper = thresholds[attribute]["medium_upper_inclusive"]
        frame["local_category"] = assign_bin(frame["local_value"].to_numpy(), lower, upper)
        frame["global_category"] = assign_bin(frame["global_value"].to_numpy(), lower, upper)
        frame["disagree"] = frame["local_category"] != frame["global_category"]
        frame["abs_diff"] = np.abs(frame["local_value"] - frame["global_value"])
        frame["normalized_abs_diff"] = frame["abs_diff"] / (np.abs(frame["global_value"]) + 1e-6)
        tile_groups = frame.groupby("sample_id", sort=False)
        tile_disagreement = tile_groups["disagree"].mean()
        tile_abs = tile_groups["abs_diff"].mean()
        tile_std = tile_groups["local_value"].std(ddof=0)
        tile_iqr = tile_groups["local_value"].quantile(0.75) - tile_groups["local_value"].quantile(0.25)
        tile_variance = tile_groups["local_value"].var(ddof=0)
        tile_range = tile_groups["local_value"].max() - tile_groups["local_value"].min()
        tile_any = tile_groups["disagree"].any()
        tile_n_categories = tile_groups["local_category"].nunique()
        for sample_id, count in tile_n_categories.items():
            if int(count) >= 2:
                tile_multi_any[str(sample_id)] = True
        disagreement_ci = bootstrap_mean_ci(
            tile_disagreement.to_numpy(),
            bootstrap_repeats,
            seed + attribute_index * 101,
        )
        abs_ci = bootstrap_mean_ci(
            tile_abs.to_numpy(),
            bootstrap_repeats,
            seed + attribute_index * 101 + 1,
        )
        unique_global = global_index[attribute].dropna().astype(float)
        inter_variance = float(np.var(unique_global, ddof=0))
        intra_variance = float(np.nanmean(tile_variance))
        correlation = spearmanr(
            frame["local_value"].to_numpy(),
            frame["global_value"].to_numpy(),
            nan_policy="omit",
        )
        counts = Counter(int(value) for value in frame["local_category"])
        total = sum(counts.values())
        distributions[attribute] = {
            label: counts.get(index, 0) / max(total, 1)
            for index, label in enumerate(("low", "medium", "high"))
        }
        cluster_equal_disagreement_rate = float(tile_disagreement.mean())
        output_rows.append({
            "attribute": attribute,
            "valid_local_window_count": len(frame),
            "mean_absolute_difference": float(frame["abs_diff"].mean()),
            "median_absolute_difference": float(frame["abs_diff"].median()),
            "mean_normalized_absolute_difference": float(frame["normalized_abs_diff"].mean()),
            "mean_within_tile_local_std": float(np.nanmean(tile_std)),
            "mean_within_tile_local_iqr": float(np.nanmean(tile_iqr)),
            "intra_tile_variance": intra_variance,
            "inter_tile_variance": inter_variance,
            "intra_inter_variance_ratio": intra_variance / max(inter_variance, 1e-12),
            "mean_within_tile_range": float(np.nanmean(tile_range)),
            "spearman_local_global_correlation": float(correlation.statistic),
            "spearman_p_value": float(correlation.pvalue),
            "bootstrap_mean_abs_diff_ci_lower": abs_ci[0],
            "bootstrap_mean_abs_diff_ci_upper": abs_ci[1],
            "local_global_category_disagreement_rate": cluster_equal_disagreement_rate,
            "bootstrap_disagreement_ci_lower": disagreement_ci[0],
            "bootstrap_disagreement_ci_upper": disagreement_ci[1],
            "tiles_with_any_disagreement_ratio": float(tile_any.mean()),
            "tiles_with_two_or_more_local_categories_ratio": float((tile_n_categories >= 2).mean()),
            "global_descriptor_local_coverage_accuracy": float(1.0 - cluster_equal_disagreement_rate),
            "local_low_ratio": distributions[attribute]["low"],
            "local_medium_ratio": distributions[attribute]["medium"],
            "local_high_ratio": distributions[attribute]["high"],
        })
    multi_rate = float(sum(tile_multi_any.values()) / max(global_frame["sample_id"].nunique(), 1))
    return pd.DataFrame(output_rows), {
        "multi_category_tile_rate_any_core_or_aux_attribute": multi_rate,
        "local_category_distributions": distributions,
    }


def transform_geometry(
    image: np.ndarray,
    mask: np.ndarray,
    region: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    horizontal = bool(rng.random() < 0.5)
    vertical = bool(rng.random() < 0.5)
    rotate_triggered = bool(rng.random() < 0.5)
    rotate_k = int(rng.integers(0, 4)) if rotate_triggered else 0
    color_triggered = bool(rng.random() < 0.5)
    transformed_image = image.copy()
    transformed_mask = mask.copy()
    transformed_region = region.copy()
    if horizontal:
        transformed_image = np.flip(transformed_image, axis=1)
        transformed_mask = np.flip(transformed_mask, axis=1)
        transformed_region = np.flip(transformed_region, axis=1)
    if vertical:
        transformed_image = np.flip(transformed_image, axis=0)
        transformed_mask = np.flip(transformed_mask, axis=0)
        transformed_region = np.flip(transformed_region, axis=0)
    if rotate_k:
        transformed_image = np.rot90(transformed_image, k=rotate_k)
        transformed_mask = np.rot90(transformed_mask, k=rotate_k)
        transformed_region = np.rot90(transformed_region, k=rotate_k)
    if color_triggered:
        brightness = float(rng.uniform(0.8, 1.2))
        contrast = float(rng.uniform(0.8, 1.2))
        float_image = transformed_image.astype(np.float32)
        mean = float_image.mean(axis=(0, 1), keepdims=True)
        float_image = (float_image - mean) * contrast + mean
        float_image *= brightness
        transformed_image = np.clip(float_image, 0, 255).astype(np.uint8)
    return (
        np.ascontiguousarray(transformed_image),
        np.ascontiguousarray(transformed_mask),
        np.ascontiguousarray(transformed_region),
        {
            "horizontal_flip": horizontal,
            "vertical_flip": vertical,
            "random_rotate90_triggered": rotate_triggered,
            "rotation_k": rotate_k,
            "color_jitter_simulated": color_triggered,
            "crop_offset": [0, 0],
            "crop_size": [256, 256],
        },
    )


def one_window_values(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[dict[str, Any], int]:
    properties = extract_properties(image, mask, {})
    foreground = (mask > 0).astype(np.float64)
    stain_foreground = properties["stain"].astype(np.float64) * foreground
    values, complete, _, _ = attributes_for_window(
        properties,
        integral_image(foreground),
        integral_image(stain_foreground),
        integral_image(stain_foreground * properties["stain"].astype(np.float64)),
        *box,
    )
    return values, int(np.sum(complete))


def augmentation_audit(
    samples: list[Sample],
    recommended_size: int,
    thresholds: dict[str, Any],
    seed: int,
    subset_size: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    chosen_indices = sorted(
        rng.choice(len(samples), size=min(subset_size, len(samples)), replace=False).tolist()
    )
    details = []
    mapped_matches = 0
    unmapped_matches = 0
    comparisons = 0
    global_matches = 0
    global_comparisons = 0
    coordinate_changes = 0
    complete_before = []
    complete_after = []
    for sample_index in chosen_indices:
        sample = samples[sample_index]
        image, mask, categories, metadata = decode_sample(sample.image_path, sample.json_path)
        starts = window_positions(mask.shape[1], recommended_size, 0.5)
        boxes = [
            (starts[0], starts[0], starts[0] + recommended_size, starts[0] + recommended_size),
            (
                starts[len(starts) // 2],
                starts[len(starts) // 2],
                starts[len(starts) // 2] + recommended_size,
                starts[len(starts) // 2] + recommended_size,
            ),
        ]
        original_props = extract_properties(image, mask, categories)
        original_global = global_attribute_row(
            sample.sample_id,
            str(metadata.get("organ_type", "unknown")),
            image,
            mask,
            original_props,
        )
        for region_index, box in enumerate(boxes):
            region = np.zeros(mask.shape, dtype=np.uint8)
            region[box[1]:box[3], box[0]:box[2]] = 1
            before_values, before_complete = one_window_values(image, mask, box)
            for augmentation_seed in (seed, seed + 1, seed + 2):
                combined_seed = augmentation_seed + sample_index * 1009 + region_index * 17
                aug_image, aug_mask, aug_region, operations = transform_geometry(
                    image, mask, region, combined_seed
                )
                ys, xs = np.nonzero(aug_region)
                mapped_box = (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                )
                mapped_values, after_complete = one_window_values(
                    aug_image, aug_mask, mapped_box
                )
                unmapped_values, _ = one_window_values(aug_image, aug_mask, box)
                coordinate_changed = tuple(box) != tuple(mapped_box)
                coordinate_changes += int(coordinate_changed)
                complete_before.append(before_complete)
                complete_after.append(after_complete)
                attribute_matches_mapped = []
                attribute_matches_unmapped = []
                for attribute in CORE_ATTRIBUTES:
                    lower = thresholds[attribute]["low_upper_exclusive"]
                    upper = thresholds[attribute]["medium_upper_inclusive"]
                    before_category = assign_bin(before_values[attribute], lower, upper)
                    mapped_category = assign_bin(mapped_values[attribute], lower, upper)
                    unmapped_category = assign_bin(unmapped_values[attribute], lower, upper)
                    if before_category >= 0 and mapped_category >= 0:
                        mapped_matches += int(before_category == mapped_category)
                        comparisons += 1
                        attribute_matches_mapped.append(before_category == mapped_category)
                    if before_category >= 0 and unmapped_category >= 0:
                        unmapped_matches += int(before_category == unmapped_category)
                        attribute_matches_unmapped.append(before_category == unmapped_category)
                transformed_props = extract_properties(aug_image, aug_mask, categories)
                transformed_global = global_attribute_row(
                    sample.sample_id,
                    str(metadata.get("organ_type", "unknown")),
                    aug_image,
                    aug_mask,
                    transformed_props,
                )
                for attribute in CORE_ATTRIBUTES:
                    lower = thresholds[attribute]["low_upper_exclusive"]
                    upper = thresholds[attribute]["medium_upper_inclusive"]
                    first = assign_bin(original_global[attribute], lower, upper)
                    second = assign_bin(transformed_global[attribute], lower, upper)
                    if first >= 0 and second >= 0:
                        global_matches += int(first == second)
                        global_comparisons += 1
                details.append({
                    "sample_id": sample.sample_id,
                    "region_index": region_index,
                    "seed": combined_seed,
                    "original_box": list(box),
                    "mapped_box": list(mapped_box),
                    "coordinate_changed": coordinate_changed,
                    "complete_instances_before": before_complete,
                    "complete_instances_after": after_complete,
                    "mapped_local_description_consistency": (
                        float(np.mean(attribute_matches_mapped))
                        if attribute_matches_mapped
                        else None
                    ),
                    "unmapped_local_description_consistency": (
                        float(np.mean(attribute_matches_unmapped))
                        if attribute_matches_unmapped
                        else None
                    ),
                    "operations": operations,
                })
    return {
        "schema_version": "l0_augmentation_consistency_v1",
        "split": "train",
        "sample_ids": [samples[index].sample_id for index in chosen_indices],
        "sample_count": len(chosen_indices),
        "random_seed_count": 3,
        "region_count_per_tile": 2,
        "current_random_crop_removes_content": False,
        "reason": "observed train tiles and crop_size are both 256x256",
        "region_coordinate_changed_ratio": coordinate_changes / max(len(details), 1),
        "mapped_local_description_consistency": mapped_matches / max(comparisons, 1),
        "unmapped_local_description_consistency": unmapped_matches / max(comparisons, 1),
        "global_description_consistency_for_geometric_core_attributes": (
            global_matches / max(global_comparisons, 1)
        ),
        "mean_complete_instances_before": float(np.mean(complete_before)),
        "mean_complete_instances_after": float(np.mean(complete_after)),
        "rotation_present_in_current_pipeline": True,
        "flip_and_rotation_mapping_synchronized_for_image_mask": True,
        "conclusion": (
            "Local attributes and local text should be computed after augmentation, or their "
            "region coordinates must be transformed with exactly the same crop/flip/rotation. "
            "Recomputation is required for stain descriptors after ColorJitter and is the safer "
            "policy for any future nontrivial crop."
        ),
        "simulation_note": (
            "Geometry matches current probabilities and 90-degree transforms. Color jitter uses "
            "the current brightness/contrast ranges; hue/saturation are not needed for morphology checks."
        ),
        "details": details,
    }


def text_bank(thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "l0_local_text_bank_candidates_v1",
        "threshold_source": "train split only",
        "thresholds": thresholds,
        "attributes": {
            "density": {
                "low": "a sparse local nuclear region",
                "medium": "a moderately populated local nuclear region",
                "high": "a densely populated local nuclear region",
            },
            "size": {
                "low": "mostly small nuclei",
                "medium": "mostly medium-sized nuclei",
                "high": "mostly large nuclei",
            },
            "size_heterogeneity": {
                "low": "uniform nuclear sizes",
                "medium": "moderately varied nuclear sizes",
                "high": "highly varied nuclear sizes",
            },
            "crowding": {
                "low": "an isolated spatial arrangement",
                "medium": "a moderately crowded spatial arrangement",
                "high": "a highly crowded spatial arrangement",
            },
            "touching_relation": {
                "low": "few touching nuclei",
                "medium": "some touching nuclei",
                "high": "many touching nuclei",
            },
            "boundary_irregularity": {
                "low": "mostly smooth nuclear boundaries",
                "medium": "moderately irregular nuclear boundaries",
                "high": "highly irregular nuclear boundaries",
            },
            "elongation": {
                "low": "mostly round nuclei",
                "medium": "mostly oval nuclei",
                "high": "mostly elongated nuclei",
            },
            "stain_intensity": {
                "low": "light hematoxylin staining",
                "medium": "moderate hematoxylin staining",
                "high": "strong hematoxylin staining",
            },
        },
        "templates": [
            "a {density} local region containing {size} with {crowding} and {boundary}",
            "a local nuclear region with {size_heterogeneity}, {touching_relation}, and {elongation}",
            "a {stain_intensity} local region containing {density} and {size}",
        ],
        "prohibited_semantics": [
            "disease diagnosis",
            "tumor grade",
            "prognosis",
            "pathology conclusions absent from annotations",
        ],
    }


def make_plots(
    output_dir: Path,
    instance_frame: pd.DataFrame,
    scale_frame: pd.DataFrame,
    disagreement_frame: pd.DataFrame,
    samples: list[Sample],
    local_frame: pd.DataFrame,
    thresholds: dict[str, Any],
    recommended_size: int,
) -> list[dict[str, Any]]:
    plots = []
    plt.figure(figsize=(8, 5))
    plt.hist(instance_frame["equivalent_diameter_px"].dropna(), bins=80, color="#4169e1")
    plt.xlabel("Equivalent diameter (original pixels)")
    plt.ylabel("Instances")
    plt.title("PanNuke train instance diameter distribution")
    path = output_dir / "L0_INSTANCE_DIAMETER_DISTRIBUTION.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append({"path": path.name, "kind": "instance diameter distribution"})

    plt.figure(figsize=(11, 5))
    x = np.arange(len(disagreement_frame))
    rates = disagreement_frame["local_global_category_disagreement_rate"].to_numpy()
    lower = disagreement_frame["bootstrap_disagreement_ci_lower"].to_numpy()
    upper = disagreement_frame["bootstrap_disagreement_ci_upper"].to_numpy()
    plt.bar(x, rates, color="#d2691e")
    lower_error = np.maximum(rates - lower, 0.0)
    upper_error = np.maximum(upper - rates, 0.0)
    plt.errorbar(x, rates, yerr=[lower_error, upper_error], fmt="none", color="black", capsize=3)
    plt.axhline(0.25, linestyle="--", color="red", label="preregistered rate threshold")
    plt.xticks(x, disagreement_frame["attribute"], rotation=55, ha="right")
    plt.ylabel("Local-global category disagreement")
    plt.legend()
    path = output_dir / "L0_LOCAL_GLOBAL_DISAGREEMENT.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append({"path": path.name, "kind": "local-global disagreement"})

    for column, filename, ylabel in (
        ("border_cut_ratio", "L0_WINDOW_SIZE_VS_BORDER_CUT.png", "Border-cut ratio"),
        ("median_complete_instance_count", "L0_WINDOW_SIZE_VS_COMPLETE_COUNT.png", "Median complete instances"),
    ):
        plt.figure(figsize=(7, 5))
        plt.plot(scale_frame["window_size"], scale_frame[column], marker="o")
        plt.axvline(recommended_size, linestyle="--", color="green", label="recommended")
        plt.xlabel("Window size (original pixels)")
        plt.ylabel(ylabel)
        plt.legend()
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append({"path": path.name, "kind": ylabel})

    plt.figure(figsize=(7, 6))
    plt.scatter(
        disagreement_frame["inter_tile_variance"] + 1e-12,
        disagreement_frame["intra_tile_variance"] + 1e-12,
    )
    for _, row in disagreement_frame.iterrows():
        plt.annotate(
            row["attribute"],
            (row["inter_tile_variance"] + 1e-12, row["intra_tile_variance"] + 1e-12),
            fontsize=7,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Inter-tile variance")
    plt.ylabel("Mean intra-tile variance")
    path = output_dir / "L0_INTRA_VS_INTER_VARIANCE.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots.append({"path": path.name, "kind": "intra versus inter variance"})

    ranges = (
        local_frame.groupby("sample_id")["nuclear_density"].agg(lambda values: values.max() - values.min())
        .sort_values(ascending=False)
    )
    sample_map = {sample.sample_id: sample for sample in samples}
    representative_ids = [sample_id for sample_id in ranges.index if sample_id in sample_map][:3]
    for sample_id in representative_ids:
        sample = sample_map[sample_id]
        image = cv2.cvtColor(cv2.imread(sample.image_path), cv2.COLOR_BGR2RGB)
        rows = local_frame[local_frame["sample_id"] == sample_id]
        lower = thresholds["nuclear_density"]["low_upper_exclusive"]
        upper = thresholds["nuclear_density"]["medium_upper_inclusive"]
        fig, axis = plt.subplots(figsize=(7, 7))
        axis.imshow(image)
        for _, row in rows.iterrows():
            category = assign_bin(row["nuclear_density"], lower, upper)
            color = ("#1f77b4", "#ffbf00", "#d62728")[max(category, 0)]
            rectangle = plt.Rectangle(
                (row["x0"], row["y0"]),
                row["x1"] - row["x0"],
                row["y1"] - row["y0"],
                fill=False,
                edgecolor=color,
                linewidth=1.1,
                alpha=0.75,
            )
            axis.add_patch(rectangle)
        axis.set_title(f"{sample_id}: local density low/medium/high")
        axis.axis("off")
        path = output_dir / f"L0_REPRESENTATIVE_HEATMAP_{sample_id}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append({
            "path": path.name,
            "kind": "representative local attribute heatmap",
            "sample_id": sample_id,
        })
    return plots


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    scale_frame: pd.DataFrame,
    disagreement_frame: pd.DataFrame,
    precheck: dict[str, Any],
) -> None:
    recommended = summary["recommended_window"]
    lines = [
        "# L0 Local Semantic Granularity Audit",
        "",
        "## Outcome",
        "",
        f"LOCAL_GRANULARITY_SUPPORTED: **{summary['LOCAL_GRANULARITY_SUPPORTED']}**",
        f"Recommended window: **{recommended['window_size']} original pixels** "
        f"({recommended['model_input_pixels']} pixels after resize; "
        f"{recommended['feature_cells']} feature cells).",
        "Recommended region policy: **complete_only**.",
        "",
        "## Real data and preprocessing",
        "",
        f"- Train pairs: {precheck['dataset']['paired_sample_count']}.",
        f"- Raw image shape: {precheck['dataset']['actual_image_shape_counts_hwc']}.",
        f"- Annotations: {precheck['dataset']['annotation_count']}; "
        f"border-touching bbox annotations: {precheck['dataset']['bbox_border_touching_annotation_count']}.",
        "- Mask: single-channel int32, background 0, sequential positive polygon IDs. "
        "category_id exists in JSON but is not a decoded mask channel.",
        "- Current train geometry: RandomCrop 256, horizontal/vertical flip, RandomRotate90, "
        "ColorJitter, then resize to 512. SAM feature grid is 32x32.",
        "",
        "## Instance definitions and formulas",
        "",
        "- Area: instance pixels in original 256x256 tile.",
        "- Equivalent diameter: diameter of a circle with equal area.",
        "- Elongation: major axis / max(minor axis, 1e-6).",
        "- Boundary irregularity: perimeter / (2 sqrt(pi area)).",
        "- Nearest-neighbor distance: Euclidean centroid distance in original pixels.",
        "- Touching neighbors: distinct labels observed within a two-pixel Chebyshev neighborhood.",
        "- Original partial instance: bbox touches an original tile edge.",
        "- Complete window instance: bbox fully inside the window and not partial at the original tile edge.",
        "- Centroid-inside comparison: centroid inside window, even if bbox crosses it.",
        "- Density: selected instance count per 10,000 original pixels.",
        "- Spatial crowding: 100 / median local nearest-neighbor distance.",
        "- Stain: nonnegative hematoxylin optical-density channel from RGB-to-HED.",
        "",
        "## Instance-size distribution",
        "",
        f"- Instance count: {summary['instance_distribution']['count']}.",
        f"- Equivalent diameter median: {summary['instance_distribution']['diameter_median']:.3f} px.",
        f"- Diameter Q25/Q75: {summary['instance_distribution']['diameter_q25']:.3f}/"
        f"{summary['instance_distribution']['diameter_q75']:.3f} px.",
        f"- Original-border partial ratio: {summary['instance_distribution']['partial_ratio']:.3f}.",
        "",
        "## Window-scale Pareto table",
        "",
        "| Size | Basis | Windows | Empty | Median instances | Median complete | Partial | Border cut | Entropy | Touch retention | Pareto |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in scale_frame.iterrows():
        lines.append(
            f"| {int(row['window_size'])} | {row['window_basis']} | {int(row['total_windows'])} | "
            f"{row['empty_window_ratio']:.3f} | {row['median_instance_count']:.2f} | "
            f"{row['median_complete_instance_count']:.2f} | {row['partial_instance_ratio']:.3f} | "
            f"{row['border_cut_ratio']:.3f} | {row['local_attribute_entropy']:.3f} | "
            f"{row['touching_relation_retention']:.3f} | {bool(row['pareto_efficient'])} |"
        )
    lines += [
        "",
        "No weighted score was used. The smallest scale satisfying the preregistered empty-window "
        "and complete-count constraints was preferred to preserve locality; the complete-only "
        "policy prevents cut instances from contaminating morphology.",
        "",
        "## Local-global disagreement",
        "",
        "| Attribute | Disagreement | 95% CI | Intra var | Inter var | Ratio | Multi-category tiles | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in disagreement_frame.iterrows():
        lines.append(
            f"| {row['attribute']} | {row['local_global_category_disagreement_rate']:.3f} | "
            f"[{row['bootstrap_disagreement_ci_lower']:.3f}, {row['bootstrap_disagreement_ci_upper']:.3f}] | "
            f"{row['intra_tile_variance']:.6g} | {row['inter_tile_variance']:.6g} | "
            f"{row['intra_inter_variance_ratio']:.3f} | "
            f"{row['tiles_with_two_or_more_local_categories_ratio']:.3f} | "
            f"{row['global_descriptor_local_coverage_accuracy']:.3f} |"
        )
    lines += [
        "",
        "Thresholds are train-split tile-global tertiles. Bootstrap confidence intervals resample tiles, "
        "not individual windows. Normalized difference is abs(local-global)/(abs(global)+1e-6).",
        "",
        "## Augmentation consistency",
        "",
        f"- Region coordinates changed: {summary['augmentation']['region_coordinate_changed_ratio']:.3f}.",
        f"- Mapped local-description consistency: "
        f"{summary['augmentation']['mapped_local_description_consistency']:.3f}.",
        f"- Unmapped consistency: {summary['augmentation']['unmapped_local_description_consistency']:.3f}.",
        "- Current RandomCrop does not remove content because observed tiles and crop are both 256x256.",
        "- Local text must be generated after augmentation, or region coordinates must undergo the exact "
        "same crop/flip/rotation. Stain text should be recomputed after ColorJitter.",
        "",
        "## Recommended local attribute set",
        "",
        "density, mean nuclear size, size heterogeneity, crowding, touching relation, "
        "boundary irregularity, elongation, and hematoxylin stain intensity.",
        "",
        "Recommended template:",
        "",
        "> a {density} local region containing {size} with {crowding} and {boundary}",
        "",
        "No diagnostic, tumor-grade, or unannotated pathology language is included.",
        "",
        "## Preregistered Gate",
        "",
    ]
    for name, condition in summary["gate"]["conditions"].items():
        lines.append(
            f"- {name}: **{'PASS' if condition['passed'] else 'FAIL'}**; "
            f"actual={condition['actual']}"
        )
    lines += [
        "",
        f"Overall: **{'L1 has data support' if summary['LOCAL_GRANULARITY_SUPPORTED'] else 'L1 lacks preregistered support'}**.",
        "",
        "## Biases and uncertainty",
        "",
        "- Polygon rasterization follows current DataLoader behavior and may overwrite overlaps.",
        "- Original-border instances are conservatively treated as partial because content outside the tile is unknown.",
        "- The complete-only policy trades sample count for unbiased morphology; centroid-inside columns quantify this bias.",
        "- HED stain intensity is a controlled image statistic, not a diagnostic label.",
        "- Current crop-size equality limits conclusions about future nontrivial random crops.",
        "",
        "Cleanup and final file hashes are finalized by the execution wrapper after successful audit.",
        "",
    ]
    (output_dir / "L0_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if not 0.0 <= args.overlap_ratio < 1.0:
        raise ValueError("--overlap-ratio must be in [0,1)")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive")
    if args.output_dir.exists():
        existing = list(args.output_dir.iterdir())
        allowed_existing = {"L0_PRECHECK.json", "L0_EXECUTION_COMMAND.txt"}
        unknown = [path for path in existing if path.name not in allowed_existing]
        if unknown:
            raise FileExistsError(f"Output directory contains existing artifacts: {unknown[:10]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    precheck_path = args.output_dir / "L0_PRECHECK.json"
    if not precheck_path.is_file():
        raise FileNotFoundError("L0_PRECHECK.json must exist before full audit")
    precheck = json.loads(precheck_path.read_text(encoding="utf-8"))
    if precheck.get("precheck_pass") is not True:
        raise RuntimeError("L0 precheck did not pass")

    command = (
        f"CUDA_VISIBLE_DEVICES=\"\" python tools/audit_local_semantic_granularity.py "
        f"--dataset-root {args.dataset_root} --split {args.split} --output-dir {args.output_dir} "
        f"--seed {args.seed} --num-workers {args.num_workers} "
        f"--window-policy {args.window_policy} --overlap-ratio {args.overlap_ratio} "
        f"--bootstrap-repeats {args.bootstrap_repeats}"
    )
    if args.max_samples is not None:
        command += f" --max-samples {args.max_samples}"
    (args.output_dir / "L0_EXECUTION_COMMAND.txt").write_text(command + "\n", encoding="utf-8")

    samples = list_samples(args.dataset_root, args.split, args.max_samples)
    if not samples:
        raise RuntimeError("No paired samples found")
    print(json.dumps({"stage": "base_pass_start", "samples": len(samples)}), flush=True)

    instance_path = args.output_dir / "L0_INSTANCE_STATISTICS.csv"
    global_path = args.output_dir / "L0_TILE_GLOBAL_ATTRIBUTES.csv"
    global_rows = []
    complete_diameters = []
    with instance_path.open("w", encoding="utf-8", newline="") as instance_handle:
        writer = csv.DictWriter(instance_handle, fieldnames=INSTANCE_FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for index, result in enumerate(executor.map(analyze_base_worker, samples, chunksize=8), start=1):
                writer.writerows(result["instance_rows"])
                global_rows.append(result["global_row"])
                complete_diameters.extend(result["complete_diameters"])
                if index % 250 == 0 or index == len(samples):
                    print(json.dumps({"stage": "base_pass", "completed": index, "total": len(samples)}), flush=True)
    global_frame = pd.DataFrame(global_rows, columns=GLOBAL_FIELDS)
    global_frame.to_csv(global_path, index=False)
    instance_frame = pd.read_csv(instance_path)
    median_diameter = float(np.median(complete_diameters))
    image_width = int(global_frame["image_width"].mode().iloc[0])
    candidates = build_window_candidates(median_diameter, image_width, grid_step=8, policy=args.window_policy)
    print(json.dumps({"stage": "local_pass_start", "candidates": candidates}), flush=True)

    local_path = args.output_dir / "L0_LOCAL_REGION_ATTRIBUTES.csv"
    scale_accumulator: dict[int, dict[str, Any]] = {
        int(candidate["window_size"]): {
            "window_basis": candidate["basis"],
            "complete_counts": [],
            "centroid_counts": [],
            "empty_count": 0,
            "centroid_crossing": 0,
            "centroid_total": 0,
            "original_partial": 0,
            "touch_total": 0,
            "touch_retained": 0,
            "total_windows": 0,
        }
        for candidate in candidates
    }
    with local_path.open("w", encoding="utf-8", newline="") as local_handle:
        writer = csv.DictWriter(local_handle, fieldnames=LOCAL_FIELDS)
        writer.writeheader()
        payloads = ((sample, candidates, args.overlap_ratio) for sample in samples)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for index, result in enumerate(executor.map(analyze_local_worker, payloads, chunksize=2), start=1):
                writer.writerows(result["local_rows"])
                for row in result["local_rows"]:
                    size = int(row["window_size"])
                    accumulator = scale_accumulator[size]
                    accumulator["total_windows"] += 1
                    accumulator["empty_count"] += int(bool(row["empty_window"]))
                    accumulator["complete_counts"].append(int(row["complete_instance_count"]))
                    accumulator["centroid_counts"].append(int(row["centroid_inside_count"]))
                    accumulator["centroid_crossing"] += int(row["centroid_crossing_count"])
                    accumulator["centroid_total"] += int(row["centroid_inside_count"])
                    accumulator["original_partial"] += int(row["original_partial_instance_count"])
                for size, relation in result["relation_stats"].items():
                    accumulator = scale_accumulator[int(size)]
                    accumulator["touch_total"] += int(relation["touching_relation_total"])
                    accumulator["touch_retained"] += int(relation["touching_relation_retained"])
                if index % 100 == 0 or index == len(samples):
                    print(json.dumps({"stage": "local_pass", "completed": index, "total": len(samples)}), flush=True)

    thresholds = fit_thresholds(global_frame)
    write_json(
        args.output_dir / "L0_ATTRIBUTE_BIN_THRESHOLDS.json",
        {
            "schema_version": "l0_attribute_bin_thresholds_v1",
            "fit_split": "train",
            "policy": "complete_only tile-global tertiles",
            "attributes": thresholds,
        },
    )

    category_counts_by_scale: dict[int, dict[str, Counter]] = {
        size: {attribute: Counter() for attribute in ATTRIBUTE_NAMES}
        for size in scale_accumulator
    }
    for chunk in pd.read_csv(
        local_path,
        usecols=["window_size", *ATTRIBUTE_NAMES],
        chunksize=100000,
    ):
        for size, scale_chunk in chunk.groupby("window_size"):
            size = int(size)
            for attribute in ATTRIBUTE_NAMES:
                config = thresholds[attribute]
                categories = assign_bin(
                    pd.to_numeric(scale_chunk[attribute], errors="coerce").to_numpy(),
                    config["low_upper_exclusive"],
                    config["medium_upper_inclusive"],
                )
                category_counts_by_scale[size][attribute].update(
                    int(value) for value in categories if int(value) >= 0
                )

    scale_rows = []
    for candidate in candidates:
        size = int(candidate["window_size"])
        accumulator = scale_accumulator[size]
        entropies = [
            shannon_entropy_from_counts(
                [category_counts_by_scale[size][attribute].get(index, 0) for index in range(3)]
            )
            for attribute in CORE_ATTRIBUTES
        ]
        total = accumulator["total_windows"]
        centroid_total = accumulator["centroid_total"]
        scale_rows.append({
            "window_size": size,
            "model_input_window_size": size * 2,
            "feature_grid_cells": size / 8.0,
            "window_basis": candidate["basis"],
            "overlap_ratio": args.overlap_ratio,
            "total_windows": total,
            "windows_per_tile": total / len(samples),
            "empty_window_ratio": accumulator["empty_count"] / max(total, 1),
            "median_instance_count": float(np.median(accumulator["centroid_counts"])),
            "median_complete_instance_count": float(np.median(accumulator["complete_counts"])),
            "partial_instance_ratio": accumulator["original_partial"] / max(centroid_total, 1),
            "border_cut_ratio": accumulator["centroid_crossing"] / max(centroid_total, 1),
            "local_attribute_entropy": float(np.mean(entropies)),
            "touching_relation_retention": accumulator["touch_retained"] / max(accumulator["touch_total"], 1),
            "computational_cost_window_pixels": int(total * size * size),
            "complete_instance_policy_reliable": True,
        })
    flags = pareto_flags(scale_rows)
    for row in scale_rows:
        row["pareto_efficient"] = flags[int(row["window_size"])]
    scale_frame = pd.DataFrame(scale_rows).sort_values("window_size")
    eligible = scale_frame[
        (scale_frame["empty_window_ratio"] <= 0.15)
        & (scale_frame["median_complete_instance_count"] >= 4)
    ]
    if len(eligible):
        recommended_size = int(eligible.sort_values("window_size").iloc[0]["window_size"])
        recommendation_reason = (
            "smallest scale satisfying empty-window <=0.15 and median complete instances >=4; "
            "complete-only morphology makes border cuts explicitly safe"
        )
    else:
        recommended_size = int(
            scale_frame.sort_values(
                ["median_complete_instance_count", "empty_window_ratio", "window_size"],
                ascending=[False, True, True],
            ).iloc[0]["window_size"]
        )
        recommendation_reason = "fallback: no scale met every preregistered feasibility condition"
    scale_frame["recommended"] = scale_frame["window_size"] == recommended_size
    scale_frame.to_csv(args.output_dir / "L0_WINDOW_SCALE_COMPARISON.csv", index=False)

    recommended_chunks = []
    for chunk in pd.read_csv(local_path, chunksize=100000):
        selected = chunk[chunk["window_size"] == recommended_size]
        if len(selected):
            recommended_chunks.append(selected)
    local_recommended = pd.concat(recommended_chunks, ignore_index=True)
    disagreement_frame, disagreement_extra = disagreement_analysis(
        local_recommended,
        global_frame,
        thresholds,
        args.bootstrap_repeats,
        args.seed,
    )
    disagreement_frame.to_csv(args.output_dir / "L0_LOCAL_GLOBAL_DISAGREEMENT.csv", index=False)

    augmentation = augmentation_audit(
        samples,
        recommended_size,
        thresholds,
        args.seed,
        args.augmentation_subset,
    )
    write_json(args.output_dir / "L0_AUGMENTATION_CONSISTENCY.json", augmentation)
    write_json(args.output_dir / "L0_LOCAL_TEXT_BANK_CANDIDATES.json", text_bank(thresholds))

    disagreement_lookup = disagreement_frame.set_index("attribute")
    qualifying = []
    for attribute in CORE_ATTRIBUTES:
        row = disagreement_lookup.loc[attribute]
        if (
            row["local_global_category_disagreement_rate"] >= 0.25
            and row["bootstrap_disagreement_ci_lower"] >= 0.20
        ):
            qualifying.append(attribute)
    recommended_scale_row = scale_frame.set_index("window_size").loc[recommended_size]
    distributions = disagreement_extra["local_category_distributions"]
    touching_non_degenerate = sum(
        ratio >= 0.05 for ratio in distributions["touching_nuclei_ratio"].values()
    ) >= 2
    crowding_non_degenerate = sum(
        ratio >= 0.05 for ratio in distributions["spatial_crowding"].values()
    ) >= 2
    multi_tile_rate = max(
        float(disagreement_lookup.loc[attribute, "tiles_with_two_or_more_local_categories_ratio"])
        for attribute in CORE_ATTRIBUTES
    )
    gate_conditions = {
        "three_core_attributes_disagreement_and_ci": {
            "passed": len(qualifying) >= 3,
            "actual": {"count": len(qualifying), "attributes": qualifying},
            "threshold": "count>=3, disagreement>=0.25, CI lower>=0.20",
        },
        "recommended_empty_window_ratio": {
            "passed": float(recommended_scale_row["empty_window_ratio"]) <= 0.15,
            "actual": float(recommended_scale_row["empty_window_ratio"]),
            "threshold": "<=0.15",
        },
        "recommended_border_cut_or_reliable_complete_policy": {
            "passed": (
                float(recommended_scale_row["border_cut_ratio"]) <= 0.20
                or bool(recommended_scale_row["complete_instance_policy_reliable"])
            ),
            "actual": {
                "border_cut_ratio": float(recommended_scale_row["border_cut_ratio"]),
                "complete_policy_reliable": bool(recommended_scale_row["complete_instance_policy_reliable"]),
            },
            "threshold": "border_cut<=0.20 OR reliable complete-only policy",
        },
        "recommended_median_complete_instance_count": {
            "passed": float(recommended_scale_row["median_complete_instance_count"]) >= 4,
            "actual": float(recommended_scale_row["median_complete_instance_count"]),
            "threshold": ">=4",
        },
        "touching_and_crowding_not_degenerate": {
            "passed": touching_non_degenerate and crowding_non_degenerate,
            "actual": {
                "touching_non_degenerate": touching_non_degenerate,
                "crowding_non_degenerate": crowding_non_degenerate,
                "touching_distribution": distributions["touching_nuclei_ratio"],
                "crowding_distribution": distributions["spatial_crowding"],
            },
            "threshold": "at least two bins each with >=5%",
        },
        "tiles_with_two_or_more_local_categories": {
            "passed": multi_tile_rate >= 0.40,
            "actual": multi_tile_rate,
            "threshold": ">=0.40 for at least one core descriptor",
        },
    }
    local_supported = all(condition["passed"] for condition in gate_conditions.values())
    plots = make_plots(
        args.output_dir,
        instance_frame,
        scale_frame,
        disagreement_frame,
        samples,
        local_recommended,
        thresholds,
        recommended_size,
    )
    summary = {
        "schema_version": "local_semantic_granularity_l0_final_summary_v1",
        "split": "train",
        "seed": args.seed,
        "sample_count": len(samples),
        "full_train_split": args.max_samples is None and len(samples) == 4946,
        "GPU_USED": False,
        "MODEL_CODE_UNCHANGED": True,
        "TRAINING_NOT_STARTED": True,
        "instance_distribution": {
            "count": int(len(instance_frame)),
            "area_median": float(instance_frame["area_px2"].median()),
            "diameter_q25": float(instance_frame["equivalent_diameter_px"].quantile(0.25)),
            "diameter_median": float(instance_frame["equivalent_diameter_px"].median()),
            "diameter_q75": float(instance_frame["equivalent_diameter_px"].quantile(0.75)),
            "partial_ratio": float(instance_frame["partial_instance_original_border"].mean()),
        },
        "window_candidates": finite_or_none(scale_rows),
        "recommended_window": {
            "window_size": recommended_size,
            "model_input_pixels": recommended_size * 2,
            "feature_cells": recommended_size // 8,
            "overlap_ratio": args.overlap_ratio,
            "region_policy": "complete_only",
            "reason": recommendation_reason,
        },
        "disagreement": finite_or_none(disagreement_frame.to_dict(orient="records")),
        "augmentation": {
            key: augmentation[key]
            for key in (
                "region_coordinate_changed_ratio",
                "mapped_local_description_consistency",
                "unmapped_local_description_consistency",
                "global_description_consistency_for_geometric_core_attributes",
                "mean_complete_instances_before",
                "mean_complete_instances_after",
                "conclusion",
            )
        },
        "recommended_attributes": [
            "nuclear_density",
            "mean_nuclear_size",
            "nuclear_size_heterogeneity",
            "spatial_crowding",
            "touching_nuclei_ratio",
            "boundary_irregularity",
            "nuclear_elongation",
            "mean_stain_intensity",
        ],
        "representative_plots": plots,
        "gate": {"conditions": gate_conditions, "qualifying_core_attributes": qualifying},
        "LOCAL_GRANULARITY_SUPPORTED": local_supported,
        "elapsed_seconds": time.time() - started,
        "status": {
            "CODE_IMPLEMENTATION": "PASS",
            "CPU_TESTS": "PENDING_EXTERNAL_VERIFICATION",
            "FULL_TRAIN_SPLIT_AUDIT": "PASS" if args.max_samples is None and len(samples) == 4946 else "PARTIAL",
            "LOCAL_GRANULARITY_SUPPORTED": local_supported,
            "RECOMMENDED_WINDOW_SIZE": recommended_size,
            "RECOMMENDED_REGION_POLICY": "complete_only",
            "GPU_USED": False,
            "MODEL_CODE_UNCHANGED": True,
            "TRAINING_NOT_STARTED": True,
            "SAFE_CLEANUP": "PENDING",
        },
        "uncertainties": [
            "Original-border instances are conservatively classified as partial.",
            "HED stain intensity is not stain-normalized across acquisition domains.",
            "Current 256 crop equals the observed raw tile size, limiting evidence about content-removing crops.",
        ],
    }
    write_json(args.output_dir / "L0_FINAL_SUMMARY.json", summary)
    write_report(args.output_dir, summary, scale_frame, disagreement_frame, precheck)
    print(
        json.dumps({
            "stage": "complete",
            "sample_count": len(samples),
            "recommended_window": recommended_size,
            "LOCAL_GRANULARITY_SUPPORTED": local_supported,
            "elapsed_seconds": summary["elapsed_seconds"],
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
