#!/usr/bin/env python3
"""
analyze_gt_structure_boundary_attrs.py

Compute 10 GT-derived pathology structure & boundary attributes from PanNuke
instance masks.  Produces three output files:
  1. gt_structure_boundary_attr_stats.json     — global distribution + diagnosis
  2. gt_structure_boundary_attr_samples.jsonl  — per-sample attributes
  3. structure_boundary_prompt_templates.json  — English prompt templates for CONCH

Usage (default, backward-compatible):
    python scripts/analyze_gt_structure_boundary_attrs.py \
        --data_path data/PanNuke/train \
        --out_dir workdir/attr_stats \
        --image_size 512 \
        --min_instance_area 10

Usage (train split — fit thresholds):
    python scripts/analyze_gt_structure_boundary_attrs.py \
        --data_path data/PanNuke \
        --split_name train \
        --out_dir workdir/attr_stats \
        --samples_out workdir/attr_stats/gt_structure_boundary_attr_train.jsonl \
        --fit_thresholds \
        --min_instance_area 10

Usage (test split — apply train thresholds):
    python scripts/analyze_gt_structure_boundary_attrs.py \
        --data_path data/PanNuke \
        --split_name test \
        --out_dir workdir/attr_stats \
        --samples_out workdir/attr_stats/gt_structure_boundary_attr_test.jsonl \
        --apply_thresholds_from workdir/attr_stats/gt_structure_boundary_attr_stats.json \
        --min_instance_area 10

Usage (merge per-split JSONL files):
    python scripts/analyze_gt_structure_boundary_attrs.py \
        --merge_jsonls workdir/attr_stats/gt_structure_boundary_attr_train.jsonl \
                      workdir/attr_stats/gt_structure_boundary_attr_test.jsonl \
        --samples_out workdir/attr_stats/gt_structure_boundary_attr_all.jsonl

Core principles (do NOT modify existing training code):
    - Pathology morphology prior defines attribute semantics.
    - GT instance masks are used ONLY for automatic attribute computation and
      supervision generation.
    - Train-split quantiles discretise low / mid / high; they do NOT determine
      whether an attribute is meaningful.
    - Test split is never used for threshold fitting (no data leakage).
    - Color is excluded because it reflects stain/style, not instance structure.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*minor_axis_length.*")
warnings.filterwarnings("ignore", message=".*major_axis_length.*")
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("⚠️  pycocotools not available — COCO RLE annotations will be skipped.",
          file=sys.stderr)

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available — KDTree for crowding proxy will not work.",
          file=sys.stderr)


# ===================================================================
# 1. Attribute names & frequency-path grouping
# ===================================================================
STRUCTURE_ATTRS = [
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
]

BOUNDARY_ATTRS = [
    "boundary_density",
    "nuclear_irregularity",
    "nuclear_elongation",
    "touching_or_crowding_difficulty",
    "small_nuclei_ratio",
]

ALL_ATTRS = STRUCTURE_ATTRS + BOUNDARY_ATTRS  # length = 10


# ===================================================================
# 2. Mask decoding (reused from DataLoader._decode_mask)
# ===================================================================
def decode_mask(json_path: str) -> np.ndarray:
    """
    Decode instance mask from polygon or COCO-RLE annotations.

    Compatible with the PanNuke-style JSON used in this project.
    Returns int32 instance mask, shape (H, W), 0 = background.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    if isinstance(data_json, list):
        data_json = data_json[0] if len(data_json) > 0 and isinstance(
            data_json[0], dict) else {}

    if "image" in data_json and isinstance(data_json["image"], dict):
        h = int(data_json["image"].get("height",
                                       data_json.get("height", 256)))
        w = int(data_json["image"].get("width",
                                       data_json.get("width", 256)))
    else:
        h = int(data_json.get("height", 256))
        w = int(data_json.get("width", 256))

    mask = np.zeros((h, w), dtype=np.int32)
    inst_id = 1

    annotations = data_json.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        seg = ann.get("segmentation", None)

        if isinstance(seg, list):
            # seg can be [poly] or flat poly
            polygons = (
                [seg] if all(isinstance(x, (int, float)) for x in seg)
                else seg
            )

            for poly in polygons:
                try:
                    pts = np.array(poly, dtype=np.float32).reshape((-1, 2))
                    if pts.shape[0] >= 3:
                        pts = np.round(pts).astype(np.int32)
                        cv2.fillPoly(mask, [pts], inst_id)
                        inst_id += 1
                except Exception:
                    continue

        elif isinstance(seg, dict) and "counts" in seg and "size" in seg:
            if not PYCOCOTOOLS_AVAILABLE:
                print(
                    f"  ⚠️  Skipping RLE annotation in {json_path} — "
                    "pycocotools not available.",
                    file=sys.stderr,
                )
                continue

            try:
                binary_mask = coco_mask.decode(seg)
                if binary_mask.ndim == 3:
                    binary_mask = np.max(binary_mask, axis=2)
                if binary_mask.shape[:2] != mask.shape[:2]:
                    binary_mask = cv2.resize(
                        binary_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask[binary_mask > 0] = inst_id
                inst_id += 1
            except Exception:
                continue

    return mask


# ===================================================================
# 3. Data scanning
# ===================================================================
def scan_data(data_path: str) -> List[Dict[str, str]]:
    """
    Scan *data_path* for PNG images and their paired JSON annotations.
    Returns a list of dicts with keys: img_path, json_path, stem.
    """
    samples = []
    if not os.path.isdir(data_path):
        print(f"❌ data_path not found: {data_path}", file=sys.stderr)
        return samples

    for fname in sorted(os.listdir(data_path)):
        if not fname.lower().endswith(".png"):
            continue
        stem, _ = os.path.splitext(fname)
        img_path = os.path.join(data_path, fname)
        json_path = os.path.join(data_path, stem + ".json")
        if not os.path.isfile(json_path):
            print(f"  ⚠️  No JSON for {fname}, skipping.", file=sys.stderr)
            continue
        samples.append({
            "img_path": img_path,
            "json_path": json_path,
            "stem": stem,
        })

    print(f"📂 Found {len(samples)} image–annotation pairs in {data_path}")
    return samples


# ===================================================================
# 4. Organ extraction from JSON metadata
# ===================================================================
def read_organ(json_path: str) -> str:
    """Read organ_type / organ_id from a PanNuke JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0] if data else {}
        organ = data.get("organ_type", data.get("organ_id", "Unknown"))
        return str(organ)
    except Exception:
        return "Unknown"


# ===================================================================
# 5. Attribute computation helpers
# ===================================================================
def _filter_small_instances(
    inst_mask: np.ndarray, min_area: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Remove connected-components with area < min_area.
    Returns (filtered_mask, kept_instance_ids).
    """
    props = regionprops(inst_mask)
    kept = []
    for p in props:
        if p.label > 0 and p.area >= min_area:
            kept.append(p.label)

    if not kept:
        return np.zeros_like(inst_mask, dtype=np.int32), []

    filtered = np.zeros_like(inst_mask, dtype=np.int32)
    for new_id, label_id in enumerate(kept, start=1):
        filtered[inst_mask == label_id] = new_id

    return filtered, kept


def _compute_centroid_nn_stats(
    centroids: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute mean nearest-neighbour distance and mean radius from centroids.
    centroids: (N, 2) array of (row, col) -- regionprops returns (row, col).
    Returns (mean_nn_dist, mean_radius).
    """
    n = len(centroids)
    if n < 2:
        return 0.0, 0.0

    if SCIPY_AVAILABLE:
        tree = KDTree(centroids)
        dists, _ = tree.query(centroids, k=2)
        nn_dists = dists[:, 1]  # exclude self
    else:
        # fallback brute-force
        nn_dists = []
        for i in range(n):
            best = float("inf")
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(centroids[i] - centroids[j])
                if d < best:
                    best = d
            nn_dists.append(best)
        nn_dists = np.array(nn_dists, dtype=np.float64)

    mean_nn_dist = float(np.mean(nn_dists)) if n > 0 else 0.0
    return mean_nn_dist, 0.0  # radius computed externally


def compute_crowding_proxy(
    props_list: List[Any],
) -> float:
    """
    Compute crowding proxy = 1 / (1 + mean_nn_dist / mean_radius).
    Higher value = more crowded.
    Uses centroid-based nearest-neighbour distance.
    """
    n = len(props_list)
    if n < 2:
        return 0.0

    # regionprops centroid returns (row, col)
    centroids = np.array([(p.centroid[0], p.centroid[1]) for p in props_list],
                         dtype=np.float64)
    areas = np.array([p.area for p in props_list], dtype=np.float64)

    mean_nn_dist, _ = _compute_centroid_nn_stats(centroids)
    mean_radius = float(np.sqrt(np.mean(areas) / np.pi)) if len(areas) > 0 else 1.0
    mean_radius = max(mean_radius, 1.0)

    if mean_nn_dist <= 0 or mean_radius <= 0:
        return 0.0

    return float(1.0 / (1.0 + mean_nn_dist / mean_radius))


def compute_structure_attrs(
    inst_mask: np.ndarray,
    min_area: int = 10,
) -> Dict[str, Any]:
    """
    Compute 5 low-frequency (structure) attributes from instance mask.

    Returns dict with keys matching STRUCTURE_ATTRS.
    Non-computable attributes are set to None.
    """
    h, w = inst_mask.shape[:2]
    crop_area = float(h * w)
    fg_mask = (inst_mask > 0).astype(np.uint8)
    fg_pixels = int(fg_mask.sum())
    result: Dict[str, Any] = {}

    # --- nuclear_density ---
    # Count instances after filtering
    props = regionprops(inst_mask)
    valid_props = [p for p in props if p.label > 0 and p.area >= min_area]
    instance_count = len(valid_props)

    result["nuclear_density"] = instance_count / crop_area if crop_area > 0 else 0.0
    result["instance_count"] = instance_count  # raw field

    # --- nuclear_area_fraction ---
    result["nuclear_area_fraction"] = fg_pixels / crop_area if crop_area > 0 else 0.0

    if instance_count == 0:
        result["mean_nuclear_size"] = None
        result["nuclear_size_heterogeneity"] = None
        result["spatial_crowding"] = None
        return result

    # --- mean_nuclear_size ---
    areas = np.array([p.area for p in valid_props], dtype=np.float64)
    mean_area = float(np.mean(areas))
    result["mean_nuclear_size"] = mean_area

    # --- nuclear_size_heterogeneity (CV of area) ---
    if instance_count >= 2 and mean_area > 1e-8:
        cv_area = float(np.std(areas) / mean_area)
    else:
        cv_area = 0.0
    result["nuclear_size_heterogeneity"] = cv_area

    # --- spatial_crowding ---
    result["spatial_crowding"] = compute_crowding_proxy(valid_props)

    return result


def compute_boundary_attrs(
    inst_mask: np.ndarray,
    min_area: int = 10,
    global_q25: Optional[float] = None,
    boundary_radius: int = 2,
) -> Dict[str, Any]:
    """
    Compute 5 high-frequency (boundary) attributes.

    boundary_density uses per-instance dilation-erosion ring, then merges.
    touching_or_crowding_difficulty is a crowding proxy (not real touching).
    small_nuclei_ratio requires global_q25 from Phase-1.

    Returns dict with keys matching BOUNDARY_ATTRS.
    Non-computable attributes are set to None.
    """
    h, w = inst_mask.shape[:2]
    props = regionprops(inst_mask)
    valid_props = [p for p in props if p.label > 0 and p.area >= min_area]
    instance_count = len(valid_props)
    fg_mask = (inst_mask > 0).astype(np.uint8)
    fg_pixels = int(fg_mask.sum())

    result: Dict[str, Any] = {}

    if instance_count == 0:
        for k in BOUNDARY_ATTRS:
            result[k] = None
        result["touching_or_crowding_difficulty"] = None
        result["small_nuclei_ratio"] = None
        return result

    # --- boundary_density (per-instance ring, then merge) ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * boundary_radius + 1, 2 * boundary_radius + 1),
    )
    total_boundary_pixels = 0
    for p in valid_props:
        one_inst = (inst_mask == p.label).astype(np.uint8)
        dilated = cv2.dilate(one_inst, kernel, iterations=1)
        eroded = cv2.erode(one_inst, kernel, iterations=1)
        ring = (dilated - eroded).clip(0, 1).astype(np.uint8)
        total_boundary_pixels += int(ring.sum())

    if fg_pixels > 0:
        result["boundary_density"] = total_boundary_pixels / fg_pixels
    else:
        result["boundary_density"] = 0.0

    # --- nuclear_irregularity (1 - circularity) ---
    irregularities = []
    for p in valid_props:
        perimeter = p.perimeter
        area = p.area
        if perimeter < 10.0 or area < 1.0:
            # perimeter too small → treat as perfect circle
            irregularities.append(0.0)
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        circularity = min(circularity, 1.0)  # clamp
        irregularities.append(max(0.0, 1.0 - circularity))
    result["nuclear_irregularity"] = float(np.mean(irregularities))

    # --- nuclear_elongation (major / minor axis) ---
    elongations = []
    for p in valid_props:
        # Compat: skimage >= 0.26 renamed minor_axis_length -> axis_minor_length
        _minor = getattr(p, 'axis_minor_length', p.minor_axis_length)
        _major = getattr(p, 'axis_major_length', p.major_axis_length)
        minor = max(_minor, 1.0)  # numerical protection
        major = _major
        if major > 0 and minor > 0:
            elongations.append(major / minor)
    if elongations:
        result["nuclear_elongation"] = float(np.mean(elongations))
    else:
        result["nuclear_elongation"] = 1.0

    # --- touching_or_crowding_difficulty (crowding proxy) ---
    result["touching_or_crowding_difficulty"] = compute_crowding_proxy(valid_props)

    # --- small_nuclei_ratio ---
    if global_q25 is not None and global_q25 > 0:
        areas = np.array([p.area for p in valid_props], dtype=np.float64)
        small_count = int((areas < global_q25).sum())
        result["small_nuclei_ratio"] = small_count / instance_count
    else:
        result["small_nuclei_ratio"] = 0.0

    return result


# ===================================================================
# 6. Discretisation helpers
# ===================================================================
def compute_discretised_label(
    value: Optional[float],
    low_thresh: float,
    high_thresh: float,
) -> int:
    """Return -1 (invalid), 0 (low), 1 (mid), 2 (high)."""
    if value is None:
        return -1
    if value < low_thresh:
        return 0
    if value > high_thresh:
        return 2
    return 1


def compute_thresholds(values: List[float]) -> Tuple[float, float]:
    """
    Compute low→mid and mid→high thresholds using 33% and 66% quantiles.
    If q33 == q66, fall back to median split.
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 1.0
    q33 = float(np.percentile(arr, 33))
    q66 = float(np.percentile(arr, 66))
    if q33 >= q66:
        med = float(np.median(arr))
        q33 = med - 1e-6
        q66 = med + 1e-6
    return q33, q66


# ===================================================================
# 6b. Load thresholds from existing stats JSON
# ===================================================================
def load_thresholds_from_stats(stats_path: str) -> Tuple[Dict[str, Tuple[float, float]], float]:
    """
    Load per-attribute discretization thresholds and global_q25 from a
    previously-saved stats JSON file.

    Returns:
        thresholds: Dict[attr_name -> (low_to_mid, mid_to_high)]
        global_q25: float (25th percentile of instance areas from the split
                    that generated the stats)
    """
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    global_q25 = stats_data.get("global_instance_area_stats", {}).get("global_q25", 0.0)

    thresholds: Dict[str, Tuple[float, float]] = {}
    global_stats = stats_data.get("global_stats", {})
    for attr_name in ALL_ATTRS:
        attr_st = global_stats.get(attr_name, {})
        th = attr_st.get("thresholds", {})
        lo = th.get("low_to_mid", 0.0)
        hi = th.get("mid_to_high", 1.0)
        thresholds[attr_name] = (lo, hi)

    return thresholds, global_q25


# ===================================================================
# 7. Statistics & diagnosis
# ===================================================================
def compute_global_stats(
    attr_name: str,
    values: List[Optional[float]],
    organ_map: Dict[str, List[Optional[float]]],
) -> Dict[str, Any]:
    """Compute mean, std, min, max, quantiles, class distribution etc."""
    # Filter out None (invalid / empty-mask samples)
    valid = [v for v in values if v is not None]
    n_valid = len(valid)
    n_total = len(values)

    stats: Dict[str, Any] = {
        "valid_ratio": n_valid / n_total if n_total > 0 else 0.0,
        "n_valid": n_valid,
        "n_total": n_total,
    }

    if n_valid == 0:
        stats.update({
            "mean": None, "std": None,
            "min": None, "max": None,
            "quantiles": None,
            "thresholds": None,
            "class_distribution": None,
            "unique_value_count": 0,
            "q33_q66_gap": None,
            "near_constant_warning": "All samples are invalid",
            "outlier_warning": None,
            "recommend": False,
            "imbalance_warning": None,
            "organ_wise": {},
        })
        return stats

    arr = np.array(valid, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    q25 = float(np.percentile(arr, 25))
    q33 = float(np.percentile(arr, 33))
    q50 = float(np.percentile(arr, 50))
    q66 = float(np.percentile(arr, 66))
    q75 = float(np.percentile(arr, 75))

    # Discrete thresholds
    low_th, high_th = compute_thresholds(valid)

    # Class distribution
    labels = []
    for v in valid:
        labels.append(compute_discretised_label(v, low_th, high_th))
    label_counts = {0: 0, 1: 0, 2: 0}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    class_dist = {
        "low": label_counts[0] / n_valid,
        "mid": label_counts[1] / n_valid,
        "high": label_counts[2] / n_valid,
    }

    # Unique value count
    unique_vals = len(np.unique(arr))

    # q33–q66 gap relative to range
    range_val = vmax - vmin
    q33_q66_gap = (
        (q66 - q33) / range_val if range_val > 1e-8 else None
    )

    # Near-constant warning
    near_constant = None
    if unique_vals < 10 or std < 1e-6:
        near_constant = (
            f"Only {unique_vals} unique values with std={std:.6f}"
        )

    # Outlier warning (Tukey fences: Q1 - 3*IQR, Q3 + 3*IQR)
    outlier_warning = None
    iqr = q75 - q25
    if iqr > 0:
        lower_fence = q25 - 3.0 * iqr
        upper_fence = q75 + 3.0 * iqr
        outliers = arr[(arr < lower_fence) | (arr > upper_fence)]
        if len(outliers) > max(5, 0.01 * n_valid):
            outlier_warning = (
                f"{len(outliers)}/{n_valid} samples beyond Tukey fences "
                f"[{lower_fence:.4f}, {upper_fence:.4f}]"
            )

    # Imbalance warning
    imbalance = None
    max_frac = max(class_dist.values())
    if max_frac > 0.9:
        imbalance = f"Severe imbalance: class {max(class_dist, key=class_dist.get)} occupies {max_frac:.1%}"
    elif max_frac > 0.7:
        imbalance = f"Moderate imbalance: class {max(class_dist, key=class_dist.get)} occupies {max_frac:.1%}"

    # Recommend?
    recommend = True
    if near_constant is not None:
        recommend = False
    if imbalance is not None and max_frac > 0.9:
        recommend = False

    # Organ-wise stats
    organ_wise: Dict[str, Dict[str, Any]] = {}
    for organ, organ_vals in organ_map.items():
        valid_ov = [v for v in organ_vals if v is not None]
        if valid_ov:
            oa = np.array(valid_ov, dtype=np.float64)
            organ_wise[organ] = {
                "mean": float(np.mean(oa)),
                "std": float(np.std(oa)),
                "count": len(valid_ov),
            }

    stats.update({
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "quantiles": {
            "q25": q25, "q33": q33, "q50": q50, "q66": q66, "q75": q75,
        },
        "thresholds": {
            "low_to_mid": low_th,
            "mid_to_high": high_th,
        },
        "class_distribution": class_dist,
        "unique_value_count": unique_vals,
        "q33_q66_gap": q33_q66_gap,
        "near_constant_warning": near_constant,
        "outlier_warning": outlier_warning,
        "imbalance_warning": imbalance,
        "recommend": recommend,
        "organ_wise": organ_wise,
    })

    return stats


def build_diagnosis(global_stats: Dict[str, Dict]) -> Dict[str, Any]:
    """Build final diagnosis based on per-attribute statistics."""
    healthy = []
    debug_only = []
    low_freq = list(STRUCTURE_ATTRS)
    high_freq = list(BOUNDARY_ATTRS)

    for attr_name in ALL_ATTRS:
        st = global_stats.get(attr_name, {})
        if st.get("recommend", False):
            healthy.append(attr_name)
        else:
            debug_only.append(attr_name)

    # recommended_attrs_v1: healthy attrs except touching_or_crowding_difficulty
    # which is a proxy, not real touching
    recommended = [a for a in healthy if a != "touching_or_crowding_difficulty"]

    return {
        "healthy_attrs_for_v1": healthy,
        "debug_only_imbalanced": debug_only,
        "crowding_proxy_note": (
            "touching_or_crowding_difficulty uses NN-based crowding proxy "
            "(1/(1+mean_nn_dist/mean_radius)), NOT real boundary-touching "
            "detection.  It is highly correlated with spatial_crowding in "
            "this version.  A future version should replace it with "
            "morphological boundary dilation overlap."
        ),
        "low_frequency_path": low_freq,
        "high_frequency_path": high_freq,
        "recommended_attrs_v1": recommended,
    }


# ===================================================================
# 8. Prompt template generation
# ===================================================================
def generate_prompt_templates() -> Dict[str, Any]:
    """Generate English prompt templates for CONCH text encoder."""
    return {
        "version": "v1",
        "intended_encoder": "CONCH text encoder",
        "language": "en",
        "structure_prompts": {
            "nuclear_density": {
                "description": "Normalised nuclear count per pixel — reflects overall nuclear density and low-frequency tissue organisation.",
                "low": "This histopathology patch contains sparsely distributed nuclei with low nuclear density.",
                "mid": "This histopathology patch shows moderately distributed nuclei with moderate density.",
                "high": "This histopathology patch contains densely distributed nuclei with high nuclear density."
            },
            "nuclear_area_fraction": {
                "description": "Fraction of foreground nuclear pixels — reflects nuclear occupancy in the tissue.",
                "low": "This patch shows sparse nuclear coverage with low nuclear area occupancy.",
                "mid": "This patch has moderate nuclear area occupancy.",
                "high": "This patch shows high nuclear area occupancy with extensive foreground coverage."
            },
            "mean_nuclear_size": {
                "description": "Average nuclear area across instances — relates to cell type and nuclear grade.",
                "low": "The nuclei in this patch are predominantly small in size.",
                "mid": "The nuclei in this patch are medium-sized.",
                "high": "The nuclei in this patch are predominantly large in size."
            },
            "nuclear_size_heterogeneity": {
                "description": "Coefficient of variation of nuclear areas — approximates pleomorphism.",
                "low": "The nuclei show uniform size distribution with homogeneous nuclear areas.",
                "mid": "The nuclei show moderately varied sizes.",
                "high": "The nuclei show highly heterogeneous sizes, suggesting pleomorphic nuclear morphology."
            },
            "spatial_crowding": {
                "description": "Nearest-neighbour centroid distance normalised by mean radius — reflects how crowded nuclei are.",
                "low": "The nuclei are well-separated with ample inter-nuclear spacing.",
                "mid": "The nuclei show moderate spatial clustering.",
                "high": "The nuclei are crowded with dense spatial arrangement and minimal inter-nuclear distance."
            }
        },
        "boundary_prompts": {
            "boundary_density": {
                "description": "Per-instance boundary ring pixel ratio — reflects boundary burden for instance separation.",
                "low": "The nuclei have smooth boundaries with low boundary density relative to their area.",
                "mid": "The nuclei show moderate boundary complexity.",
                "high": "The nuclei exhibit complex boundaries with high boundary density, making instance separation challenging."
            },
            "nuclear_irregularity": {
                "description": "1 minus circularity — measures how irregular the nuclear contour is.",
                "low": "The nuclei are regular and round to oval in shape with smooth contours.",
                "mid": "The nuclei show moderately irregular morphology with some contour complexity.",
                "high": "The nuclei exhibit highly irregular morphology with complex, non-circular contours."
            },
            "nuclear_elongation": {
                "description": "Mean major-to-minor axis ratio — captures spindle-shaped or elongated nuclei.",
                "low": "The nuclei are round to oval with low elongation.",
                "mid": "The nuclei show moderate elongation.",
                "high": "The nuclei are elongated, spindle-like, or fusiform in shape."
            },
            "touching_or_crowding_difficulty": {
                "description": "Crowding proxy based on nearest-neighbour centroid distance — reflects close packing that makes boundary separation difficult.",
                "low": "The nuclei are well-isolated with clear, separable boundaries.",
                "mid": "Some nuclei are closely packed but boundaries remain mostly distinguishable.",
                "high": "Many nuclei are tightly clustered, making instance boundaries difficult to separate."
            },
            "small_nuclei_ratio": {
                "description": "Fraction of instances below the global 25th percentile area — reflects small-nucleus prevalence.",
                "low": "Few small nuclei are present in this patch.",
                "mid": "A moderate number of small nuclei are present, requiring careful delineation.",
                "high": "This patch contains many small nuclei requiring fine boundary delineation."
            }
        },
        "combined_prompt_templates": [
            "Cell nuclei segmentation: {structure_context} {boundary_context}",
            "Histopathology patch analysis: {structure_context} Additionally, {boundary_context}",
            "Nuclei instance segmentation guidance — {structure_context} {boundary_context}"
        ],
        "usage_notes": [
            "All prompts are in English for CONCH text encoder compatibility.",
            "Use 'structure_prompts' for low-frequency attribute conditioning.",
            "Use 'boundary_prompts' for high-frequency attribute conditioning.",
            "Combine using one of the 'combined_prompt_templates' patterns.",
            "Do NOT use 'overlapping nuclei' — we do not have overlap annotations.",
            "Preferred terms: crowded, closely packed, touching-like, difficult-to-separate."
        ]
    }


# ===================================================================
# 8b. Merge JSONL files
# ===================================================================
def merge_jsonl_files(input_paths: List[str], output_path: str):
    """
    Merge multiple JSONL files into one, preserving order but deduplicating
    by sample_id (last occurrence wins). Each record must have a 'sample_id' key.
    """
    seen: Dict[str, str] = {}
    order: List[str] = []

    for in_path in input_paths:
        if not os.path.isfile(in_path):
            print(f"⚠️  Input JSONL not found, skipping: {in_path}", file=sys.stderr)
            continue
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("sample_id")
                if sid is None:
                    continue
                if sid not in seen:
                    order.append(sid)
                seen[sid] = line

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sid in order:
            f.write(seen[sid] + "\n")

    print(f"✅ Merged {len(order)} unique samples from {len(input_paths)} files → {output_path}")


# ===================================================================
# 9. Main pipeline
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute GT-derived pathology structure & boundary attributes.",
    )
    parser.add_argument(
        "--data_path", type=str,
        help="Path to PanNuke data directory (root or split dir).",
    )
    parser.add_argument(
        "--split_name", type=str, choices=["train", "test"], default=None,
        help="Split name (train/test). If provided, resolves data_path/split_name "
             "as the data directory.",
    )
    parser.add_argument(
        "--out_dir", type=str,
        help="Output directory for statistics files.",
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Resize dimension (not used for attribute computation; "
             "kept for interface compatibility). Default: 512.",
    )
    parser.add_argument(
        "--min_instance_area", type=int, default=10,
        help="Minimum instance area in pixels. Default: 10.",
    )
    parser.add_argument(
        "--fit_thresholds", action="store_true",
        help="Fit quantile thresholds from this split's data and update "
             "gt_structure_boundary_attr_stats.json.",
    )
    parser.add_argument(
        "--apply_thresholds_from", type=str, default=None,
        help="Path to existing stats JSON (e.g., train-split stats). Loads "
             "its thresholds and global_q25; does NOT update the stats file.",
    )
    parser.add_argument(
        "--samples_out", type=str, default=None,
        help="Explicit output path for samples JSONL. Overrides the default "
             "<out_dir>/gt_structure_boundary_attr_samples.jsonl.",
    )
    parser.add_argument(
        "--merge_jsonls", type=str, nargs="+", default=None,
        help="Merge multiple JSONL files into one (requires --samples_out). "
             "When this flag is used, all other arguments are ignored.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Merge mode: combine existing per-split JSONLs
    # ------------------------------------------------------------------
    if args.merge_jsonls:
        if not args.samples_out:
            print("❌ --merge_jsonls requires --samples_out.", file=sys.stderr)
            sys.exit(1)
        merge_jsonl_files(args.merge_jsonls, args.samples_out)
        return

    # ------------------------------------------------------------------
    # Validate required arguments for non-merge mode
    # ------------------------------------------------------------------
    if not args.data_path or not args.out_dir:
        print("❌ --data_path and --out_dir are required (unless using --merge_jsonls).",
              file=sys.stderr)
        sys.exit(1)

    # Resolve data path with split
    data_path = args.data_path
    split_name = args.split_name
    if split_name:
        data_path = os.path.join(data_path, split_name)

    out_dir = args.out_dir
    min_area = args.min_instance_area

    # Determine operation mode
    fitting_mode = args.fit_thresholds
    apply_from = args.apply_thresholds_from

    # --- Create output directory ---
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"🔬 GT Structure & Boundary Attribute Analysis")
    mode_str = "fitting thresholds" if fitting_mode else (
        "applying thresholds" if apply_from else "standard"
    )
    print(f"   Mode:            {mode_str}")
    print(f"   Split name:      {split_name or 'N/A (using data_path directly)'}")
    print(f"   Data path:       {data_path}")
    print(f"   Output dir:      {out_dir}")
    print(f"   Min instance area: {min_area}")
    if apply_from:
        print(f"   Apply thresholds from: {apply_from}")
    print(f"{'='*70}\n")

    # --- Scan data ---
    samples = scan_data(data_path)
    if not samples:
        print("❌ No samples found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1: Collect global instance areas for global_q25 (or load from file)
    # ------------------------------------------------------------------
    if apply_from:
        # Load thresholds and global_q25 from pre-computed stats
        print(f"\n📊 Loading thresholds and global_q25 from {apply_from} ...")
        thresholds_loaded, global_q25 = load_thresholds_from_stats(apply_from)
        print(f"   Loaded global_q25 = {global_q25:.1f}")
        print(f"   Loaded thresholds for {len(thresholds_loaded)} attributes.")
        # Still need to scan samples for Phase 2, but skip area collection
        all_areas: List[float] = []
        phase1_skipped = 0
        for i, sample in enumerate(samples):
            try:
                inst_mask = decode_mask(sample["json_path"])
            except Exception as e:
                print(f"  ⚠️  Error decoding {sample['stem']}: {e}", file=sys.stderr)
                phase1_skipped += 1
                continue
            # Don't collect areas — use loaded global_q25
        print(f"   Scanned {len(samples)} samples ({phase1_skipped} decode errors).")
    else:
        # Original Phase 1: collect all instance areas
        print("\n📊 Phase 1: Collecting global instance areas for small_nuclei_ratio ...")
        all_areas: List[float] = []
        phase1_skipped = 0
        for i, sample in enumerate(samples):
            try:
                inst_mask = decode_mask(sample["json_path"])
            except Exception as e:
                print(f"  ⚠️  Error decoding {sample['stem']}: {e}", file=sys.stderr)
                phase1_skipped += 1
                continue

            props = regionprops(inst_mask)
            for p in props:
                if p.label > 0 and p.area >= min_area:
                    all_areas.append(float(p.area))

        print(f"   Collected {len(all_areas)} instances from "
              f"{len(samples) - phase1_skipped} samples "
              f"({phase1_skipped} skipped).")

        if len(all_areas) == 0:
            print("❌ No valid instances found. Exiting.", file=sys.stderr)
            sys.exit(1)

        global_area_arr = np.array(all_areas, dtype=np.float64)
        global_q25 = float(np.percentile(global_area_arr, 25))
        global_median = float(np.median(global_area_arr))
        global_mean = float(np.mean(global_area_arr))
        global_std = float(np.std(global_area_arr))

        print(f"   Global instance area stats:")
        print(f"     q25  = {global_q25:.1f}")
        print(f"     median = {global_median:.1f}")
        print(f"     mean = {global_mean:.1f}")
        print(f"     std  = {global_std:.1f}")

    # ------------------------------------------------------------------
    # Phase 2: Compute per-sample attributes
    # ------------------------------------------------------------------
    print("\n📊 Phase 2: Computing per-sample attributes ...")
    all_structure: Dict[str, List[Optional[float]]] = {
        k: [] for k in STRUCTURE_ATTRS
    }
    all_boundary: Dict[str, List[Optional[float]]] = {
        k: [] for k in BOUNDARY_ATTRS
    }
    organ_map: Dict[str, Dict[str, List[Optional[float]]]] = {
        k: defaultdict(list) for k in ALL_ATTRS
    }

    sample_lines: List[str] = []
    empty_count = 0
    error_count = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"   Processing sample {i + 1}/{len(samples)} ...")

        try:
            inst_mask = decode_mask(sample["json_path"])
        except Exception as e:
            print(f"  ⚠️  Error decoding {sample['stem']}: {e}", file=sys.stderr)
            error_count += 1
            continue

        organ = read_organ(sample["json_path"])
        h, w = inst_mask.shape[:2]

        # Filter small instances
        filtered_mask, _ = _filter_small_instances(inst_mask, min_area)
        has_instances = filtered_mask.sum() > 0

        # --- Compute attributes ---
        struct = compute_structure_attrs(filtered_mask, min_area)
        bound = compute_boundary_attrs(
            filtered_mask, min_area, global_q25=global_q25,
        )

        instance_count = struct.get("instance_count", 0)
        if not has_instances:
            empty_count += 1

        # --- Collect for global stats ---
        for k in STRUCTURE_ATTRS:
            all_structure[k].append(struct.get(k))
            organ_map[k][organ].append(struct.get(k))
        for k in BOUNDARY_ATTRS:
            all_boundary[k].append(bound.get(k))
            organ_map[k][organ].append(bound.get(k))

        # --- Build sample record ---
        struct_out = {k: struct.get(k) for k in STRUCTURE_ATTRS}
        bound_out = {k: bound.get(k) for k in BOUNDARY_ATTRS}

        sample_record = {
            "sample_id": sample["stem"],
            "image_path": f"{sample['stem']}.png",
            "organ": organ,
            "has_instances": bool(has_instances),
            "instance_count": int(instance_count),
            "structure_attrs": struct_out,
            "boundary_attrs": bound_out,
        }

        # Add split field when split_name is specified
        if split_name:
            sample_record["split"] = split_name

        sample_lines.append(json.dumps(sample_record))

    print(f"\n   Done. Total: {len(samples)} | "
          f"With instances: {len(samples) - empty_count - error_count} | "
          f"Empty: {empty_count} | Errors: {error_count}")

    # ------------------------------------------------------------------
    # Determine which thresholds to use for discretisation
    # ------------------------------------------------------------------
    print("\n📊 Computing discretised labels ...")

    if apply_from:
        # Use pre-loaded thresholds (from train split) — test must NOT update them
        thresholds = thresholds_loaded
        print(f"   Using loaded thresholds from {apply_from} "
              f"(test split does not update thresholds).")
    elif fitting_mode or not split_name:
        # Compute thresholds from this split's data (train fit or default mode)
        print(f"   Computing thresholds from this split's data ...")
        thresholds: Dict[str, Tuple[float, float]] = {}
        for attr_name in ALL_ATTRS:
            if attr_name in STRUCTURE_ATTRS:
                vals = all_structure[attr_name]
            else:
                vals = all_boundary[attr_name]
            valid = [v for v in vals if v is not None]
            if len(valid) > 0:
                thresholds[attr_name] = compute_thresholds(valid)
            else:
                thresholds[attr_name] = (0.0, 1.0)
    else:
        # Shouldn't happen — treat as default
        print(f"   Computing thresholds from this split's data (fallback) ...")
        thresholds: Dict[str, Tuple[float, float]] = {}
        for attr_name in ALL_ATTRS:
            if attr_name in STRUCTURE_ATTRS:
                vals = all_structure[attr_name]
            else:
                vals = all_boundary[attr_name]
            valid = [v for v in vals if v is not None]
            if len(valid) > 0:
                thresholds[attr_name] = compute_thresholds(valid)
            else:
                thresholds[attr_name] = (0.0, 1.0)

    # Apply discretised labels to samples
    updated_lines: List[str] = []
    for line in sample_lines:
        rec = json.loads(line)
        disc: Dict[str, int] = {}
        for k, v in rec["structure_attrs"].items():
            lo, hi = thresholds[k]
            disc[k] = compute_discretised_label(v, lo, hi)
        for k, v in rec["boundary_attrs"].items():
            lo, hi = thresholds[k]
            disc[k] = compute_discretised_label(v, lo, hi)
        rec["discretized_labels"] = disc
        updated_lines.append(json.dumps(rec))
    sample_lines = updated_lines

    # ------------------------------------------------------------------
    # Write samples JSONL
    # ------------------------------------------------------------------
    if args.samples_out:
        samples_path = args.samples_out
    else:
        samples_path = os.path.join(out_dir, "gt_structure_boundary_attr_samples.jsonl")

    os.makedirs(os.path.dirname(samples_path) or ".", exist_ok=True)
    with open(samples_path, "w", encoding="utf-8") as f:
        for line in sample_lines:
            f.write(line + "\n")
    print(f"✅ Wrote {samples_path}  ({len(sample_lines)} lines)")

    # ------------------------------------------------------------------
    # Global statistics & diagnosis (only when fitting thresholds or default mode)
    # ------------------------------------------------------------------
    if apply_from:
        # Test split: do NOT compute/update stats (no data leakage)
        print(f"\n📊 Skipping global statistics update (--apply_thresholds_from mode: "
              f"test split does not update thresholds or diagnosis).")
    else:
        # Compute global statistics (train fit or default mode)
        print("\n📊 Computing global statistics ...")
        global_stats: Dict[str, Any] = {}
        for attr_name in ALL_ATTRS:
            if attr_name in STRUCTURE_ATTRS:
                vals = all_structure[attr_name]
            else:
                vals = all_boundary[attr_name]

            global_stats[attr_name] = compute_global_stats(
                attr_name, vals,
                {org: organ_map[attr_name][org] for org in organ_map[attr_name]},
            )

        # Build diagnosis
        diagnosis = build_diagnosis(global_stats)

        # ------------------------------------------------------------------
        # Build metadata
        # ------------------------------------------------------------------
        metadata = {
            "source": f"GT mask statistics from {data_path}",
            "split": split_name or "unknown",
            "num_samples": len(samples),
            "empty_samples": empty_count,
            "error_samples": error_count,
            "min_instance_area": min_area,
            "num_attributes": len(ALL_ATTRS),
            "attribute_names": list(ALL_ATTRS),
            "structure_attributes": list(STRUCTURE_ATTRS),
            "boundary_attributes": list(BOUNDARY_ATTRS),
            "frequency_paths": {
                "structure_low_frequency": list(STRUCTURE_ATTRS),
                "boundary_high_frequency": list(BOUNDARY_ATTRS),
            },
        }

        global_instance_area_stats = {
            "total_instances_collected": len(all_areas) if not apply_from else 0,
            "global_q25": global_q25,
        }
        if not apply_from:
            global_instance_area_stats["global_median"] = global_median
            global_instance_area_stats["global_mean"] = global_mean
            global_instance_area_stats["global_std"] = global_std

        # ------------------------------------------------------------------
        # Write stats JSON
        # ------------------------------------------------------------------
        stats_path = os.path.join(out_dir, "gt_structure_boundary_attr_stats.json")
        stats_output = {
            "metadata": metadata,
            "global_instance_area_stats": global_instance_area_stats,
            "global_stats": global_stats,
            "diagnosis": diagnosis,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote {stats_path}")

        # ------------------------------------------------------------------
        # Write prompt templates (only once, in fitting/default mode)
        # ------------------------------------------------------------------
        prompts = generate_prompt_templates()
        prompts_path = os.path.join(out_dir, "structure_boundary_prompt_templates.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote {prompts_path}")

        # ==================================================================
        # Final diagnosis output
        # ==================================================================
        print(f"\n{'='*70}")
        print("📋 DIAGNOSIS REPORT")
        print(f"{'='*70}")

        print("\n── Healthy attributes (recommended for v1 attribute head) ──")
        for a in diagnosis["healthy_attrs_for_v1"]:
            st = global_stats[a]
            cd = st.get("class_distribution", {})
            print(f"  ✅ {a}")
            print(f"     Mean={st['mean']:.4f} | Std={st['std']:.4f} | "
                  f"Valid={st['valid_ratio']:.1%}")
            print(f"     Class dist: low={cd.get('low', 0):.2%}, "
                  f"mid={cd.get('mid', 0):.2%}, high={cd.get('high', 0):.2%}")

        if diagnosis["debug_only_imbalanced"]:
            print("\n── Debug-only attributes (imbalanced / near-constant) ──")
            for a in diagnosis["debug_only_imbalanced"]:
                st = global_stats[a]
                print(f"  ⚠️  {a}")
                if st.get("near_constant_warning"):
                    print(f"     Near-constant: {st['near_constant_warning']}")
                if st.get("imbalance_warning"):
                    print(f"     Imbalance: {st['imbalance_warning']}")

        print("\n── Low-frequency (structure) path ──")
        for a in diagnosis["low_frequency_path"]:
            status = "✅" if a in diagnosis["healthy_attrs_for_v1"] else "⚠️"
            print(f"  {status} {a}")

        print("\n── High-frequency (boundary) path ──")
        for a in diagnosis["high_frequency_path"]:
            status = "✅" if a in diagnosis["healthy_attrs_for_v1"] else "⚠️"
            print(f"  {status} {a}")

        print(f"\n── Recommended attributes for first version attribute head ──")
        for a in diagnosis["recommended_attrs_v1"]:
            print(f"  ✅ {a}")

        print(f"\n── Notes ──")
        print(f"  {diagnosis['crowding_proxy_note']}")
        print(f"\n  Prompt templates written to: {prompts_path}")
        print(f"  Use structure_prompts for low-frequency conditioning.")
        print(f"  Use boundary_prompts for high-frequency conditioning.")
        print(f"  Do NOT use 'overlapping nuclei'; use 'crowded', 'closely packed', "
              f"'touching-like', 'difficult-to-separate'.")

    print(f"\n{'='*70}")
    print("✅ Analysis complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
