import os
import sys

# PNURL_AUDIT: debug print gating (enabled via PNURL_AUDIT_ENABLED=1 env var)
_PNURL_AUDIT = os.environ.get("PNURL_AUDIT_ENABLED", "0") == "1"
_PNURL_AUDIT_DATALOADER_COUNTER = 0


def _dataloader_print(*args, **kwargs):
    """Print only on DDP rank 0 (reads RANK from environment)."""
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


# ── Stage D audit-log gating ──
from training.logging_utils import audit_print
from training.local_region_text_alignment import (
    ATTRIBUTE_NAMES as L1A_LOCAL_ATTRIBUTE_NAMES,
    compute_local_region_targets,
    load_l0_thresholds,
    region_coordinates as l1a_region_coordinates,
)
import cv2
import json
import torch
import numpy as np
import random
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import regionprops
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# === Optional dependencies ===
try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.spatial import KDTree as scipy_KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


# ==============================================================================
# 0. Organ mapping and prompt protocol
# ==============================================================================
ORGAN_TO_ID = {
    # PanNuke 19 organs
    "Adrenal_gland": 0,
    "Bile-duct": 1,
    "Bladder": 2,
    "Breast": 3,
    "Cervix": 4,
    "Colon": 5,
    "Esophagus": 6,
    "HeadNeck": 7,
    "Kidney": 8,
    "Liver": 9,
    "Lung": 10,
    "Ovarian": 11,
    "Pancreatic": 12,
    "Prostate": 13,
    "Skin": 14,
    "Stomach": 15,
    "Testis": 16,
    "Thyroid": 17,
    "Uterus": 18,

    # Extra
    "Brain": 19,
    "Generic": 20,
}

ID_TO_ORGAN = {v: k for k, v in ORGAN_TO_ID.items()}

STRICT_BASE_PROMPT = "Cell nuclei"

VALID_PROMPT_MODES = {
    "base",
    "organ_static",
    "dynamic_gt",
    "dynamic_pred",
}

LEGACY_PROMPT_MODE_ALIASES = {
    # Old non-leaking validation/test name.
    "generic": "organ_static",

    # Old GT-derived prompt mode.
    "dynamic": "dynamic_gt",

    # Old ablation names. These are removed from the formal protocol.
    # They map to dynamic_gt to preserve runnable old scripts, but should not be used
    # for final experiments unless explicitly documented as oracle/debug.
    "attribute_only": "dynamic_gt",
    "morphology_only": "dynamic_gt",
}


def normalize_prompt_mode(prompt_mode: str, default: str = "organ_static") -> Tuple[str, str]:
    """
    Normalize prompt_mode to the formal four-mode protocol.

    Returns:
        canonical_mode:
            One of base / organ_static / dynamic_gt / dynamic_pred.
        raw_mode:
            The original lower-cased user input.
    """
    raw_mode = str(prompt_mode).lower().strip()

    if raw_mode in VALID_PROMPT_MODES:
        return raw_mode, raw_mode

    if raw_mode in LEGACY_PROMPT_MODE_ALIASES:
        return LEGACY_PROMPT_MODE_ALIASES[raw_mode], raw_mode

    return default, raw_mode


def prompt_uses_gt_attributes(prompt_mode: str) -> bool:
    return prompt_mode == "dynamic_gt"


# ==============================================================================
# 1. Attribute configuration
# ==============================================================================
@dataclass
class AttributeConfig:
    AREA_SMALL: float = 250.0
    AREA_LARGE: float = 600.0
    DENSITY_SPARSE: float = 0.05
    DENSITY_DENSE: float = 0.20
    SHAPE_ROUND: float = 0.6
    SHAPE_OVAL: float = 0.85
    ARRANGE_CLUMPED: float = 0.6
    COLOR_BRIGHT: float = 200.0

    @classmethod
    def from_metadata(cls, stats: Optional[Dict[str, Any]]):
        """
        Backward-compatible reader for old medical_knowledge.json.

        medical_knowledge_v2.json mainly uses precomputed attr_labels and prompts,
        so this config is only used for dynamic_gt / fallback crop statistics.
        """
        if not stats:
            return cls()

        return cls(
            AREA_SMALL=stats.get("th_size_small", stats.get("size_q33", 250.0)),
            AREA_LARGE=stats.get("th_size_large", stats.get("size_q66", 600.0)),
            DENSITY_SPARSE=stats.get("th_dens_sparse", 0.05),
            DENSITY_DENSE=stats.get("th_dens_dense", 0.20),
            SHAPE_ROUND=stats.get("th_shape_round", stats.get("shape_q33", 0.6)),
            SHAPE_OVAL=stats.get("th_shape_oval", stats.get("shape_q66", 0.85)),
            ARRANGE_CLUMPED=stats.get("th_arrange_clumped", stats.get("arrangement_q66", 0.6)),
            COLOR_BRIGHT=stats.get("th_color_bright", stats.get("color_q50", 200.0)),
        )


# ==============================================================================
# 2. Prompt construction utilities
# ==============================================================================
def format_organ_name(organ_name: str) -> str:
    """
    Convert PanNuke-style organ names into natural text for VLM encoders.

    Examples:
        Adrenal_gland -> adrenal gland
        Bile-duct     -> bile duct
        HeadNeck      -> head and neck
    """
    if organ_name is None:
        return "generic"

    name = str(organ_name).strip()
    if not name:
        return "generic"

    special_map = {
        "Adrenal_gland": "adrenal gland",
        "Bile-duct": "bile duct",
        "HeadNeck": "head and neck",
        "Ovarian": "ovary",
        "Pancreatic": "pancreas",
        "Generic": "generic",
    }

    if name in special_map:
        return special_map[name]

    name = name.replace("_", " ").replace("-", " ")
    return name.lower()


def _safe_base_prompt(base_prompt: Optional[str]) -> str:
    if base_prompt is None or str(base_prompt).strip() == "":
        return STRICT_BASE_PROMPT
    return str(base_prompt).strip()


def _default_visuals() -> Dict[str, str]:
    return {
        "color": "deep-purple stained",
        "shape": "round",
        "arrangement": "uniformly arranged",
        "size": "medium-sized",
        "density": "moderately distributed",
    }


def _default_attr_labels() -> List[int]:
    # [Color, Shape, Arrangement, Size, Density]
    return [0, 0, 0, 1, 1]


def _build_base_prompts() -> Tuple[str, str, str]:
    return STRICT_BASE_PROMPT, STRICT_BASE_PROMPT, STRICT_BASE_PROMPT


def _build_organ_static_prompts(
    base_prompt: str,
    organ_name: str,
    visuals: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    """
    Build non-crop-leaking organ + attribute-aware prompts.

    Used as fallback or for organ-prior validation/test prompts.
    """
    _ = _safe_base_prompt(base_prompt)
    visuals = visuals or _default_visuals()

    organ_text = format_organ_name(organ_name)

    if organ_text == "generic":
        tissue_prefix = "H&E-stained histopathology patch"
        organ_phrase = "histopathology tissue"
    else:
        tissue_prefix = f"H&E-stained {organ_text} histopathology patch"
        organ_phrase = f"{organ_text} tissue"

    color = visuals.get("color", "deep-purple stained")
    shape = visuals.get("shape", "round")
    arrangement = visuals.get("arrangement", "uniformly arranged")
    size = visuals.get("size", "medium-sized")
    density = visuals.get("density", "moderately distributed")

    text_prompt = f"Cell nuclei in {organ_phrase}."

    attribute_text = (
        f"{tissue_prefix}. "
        f"The nuclei are {color}, {size}, {density}, {arrangement}, "
        f"and {shape} in shape. "
        f"These attribute-aware prompts describe nuclear staining, size, density, "
        f"spatial arrangement, and morphology without using crop-level mask-derived statistics."
    )

    morphology_text = (
        f"{tissue_prefix}. "
        f"Focus on nuclear morphology and boundaries. "
        f"The nuclei tend to be {shape}, {size}, {density}, and {arrangement}. "
        f"Emphasize touching nuclei separation, contour clarity, boundary sharpness, "
        f"and instance-level delineation."
    )

    return text_prompt, attribute_text, morphology_text


def _build_dynamic_gt_prompts(
    base_prompt: str,
    organ_name: str,
    visuals: Dict[str, str],
    task_type: str = "generic",
    text_suffix: str = "",
) -> Tuple[str, str, str]:
    """
    Build GT-derived oracle/training prompts.

    This function uses crop-level attributes estimated from the GT mask.
    It must not be used for normal validation/test.
    """
    _ = _safe_base_prompt(base_prompt)
    organ_text = format_organ_name(organ_name)

    if organ_text == "generic":
        tissue_prefix = "H&E-stained histopathology patch"
        organ_phrase = "histopathology tissue"
    else:
        tissue_prefix = f"H&E-stained {organ_text} histopathology patch"
        organ_phrase = f"{organ_text} tissue"

    color = visuals.get("color", "deep-purple stained")
    shape = visuals.get("shape", "round")
    arrangement = visuals.get("arrangement", "uniformly arranged")
    size = visuals.get("size", "medium-sized")
    density = visuals.get("density", "moderately distributed")

    if task_type != "generic" and text_suffix:
        text_prompt = (
            f"{text_suffix} cell nuclei in {organ_phrase}; "
            f"nuclei are {density} and {arrangement}."
        )
    else:
        text_prompt = (
            f"Cell nuclei in {organ_phrase}; "
            f"nuclei are {density} and {arrangement}."
        )

    attribute_text = (
        f"{tissue_prefix}. "
        f"The cell nuclei are {color}, {shape} in morphology, {size}, "
        f"{density}, and {arrangement}. "
        f"These crop-level attributes describe nuclear staining appearance, tissue context, "
        f"size, density, and spatial arrangement."
    )

    morphology_text = (
        f"{tissue_prefix}. "
        f"Focus on individual nuclear morphology: {shape} contours, {size} nuclei, "
        f"{arrangement}, and {density}. "
        f"Emphasize sharp nuclear boundaries, touching nuclei separation, "
        f"irregular contours, and instance-level delineation."
    )

    return text_prompt, attribute_text, morphology_text


def build_pathology_prompts(
    base_prompt: str,
    organ_name: str,
    visuals: Optional[Dict[str, str]] = None,
    task_type: str = "generic",
    text_suffix: str = "",
    prompt_mode: str = "organ_static",
) -> Tuple[str, str, str]:
    """
    Build three prompts for different branches.

    Formal prompt protocol:
        base:
            Return "Cell nuclei" only. No organ, no attributes.

        organ_static:
            Use organ / tissue context + image-level attribute-aware prompts.
            This is the main PromptNu-style non-crop-leaking mode.

        dynamic_gt:
            Use GT mask-derived crop-level shape / size / density / arrangement / color.
            This is allowed only for training or oracle/debug experiments.

        dynamic_pred:
            Reserved for future predicted-attribute prompts.
            Current implementation uses the same non-crop-leaking attribute-aware prompt
            as organ_static.
    """
    canonical_mode, _ = normalize_prompt_mode(prompt_mode, default="organ_static")

    if canonical_mode == "base":
        return _build_base_prompts()

    if canonical_mode == "organ_static":
        return _build_organ_static_prompts(
            base_prompt=base_prompt,
            organ_name=organ_name,
            visuals=visuals,
        )

    if canonical_mode == "dynamic_pred":
        return _build_organ_static_prompts(
            base_prompt=base_prompt,
            organ_name=organ_name,
            visuals=visuals,
        )

    if canonical_mode == "dynamic_gt":
        visuals = visuals or _default_visuals()
        return _build_dynamic_gt_prompts(
            base_prompt=base_prompt,
            organ_name=organ_name,
            visuals=visuals,
            task_type=task_type,
            text_suffix=text_suffix,
        )

    return _build_organ_static_prompts(
        base_prompt=base_prompt,
        organ_name=organ_name,
        visuals=visuals,
    )


# ==============================================================================
# 3. Knowledge v2 utilities
# ==============================================================================
def _to_int_list(x: Any, default: Optional[List[int]] = None, length: int = 5) -> List[int]:
    default = default or _default_attr_labels()

    if x is None:
        return list(default)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().view(-1).tolist()
    elif isinstance(x, np.ndarray):
        x = x.reshape(-1).tolist()
    elif isinstance(x, (list, tuple)):
        x = list(x)
    else:
        return list(default)

    out = []
    for item in x[:length]:
        try:
            out.append(int(item))
        except Exception:
            out.append(0)

    if len(out) < length:
        out = out + list(default[len(out):length])

    return out


def _entry_is_v2(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False

    required = ["attr_labels", "attribute_text", "morphology_text"]
    return all(k in entry for k in required)


def _normalise_visual_stats_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    stats = entry.get("visual_stats", {}) if isinstance(entry, dict) else {}
    if not isinstance(stats, dict):
        stats = {}

    visuals = _default_visuals()
    for key in ["color", "shape", "arrangement", "size", "density"]:
        value = stats.get(key, None)
        if isinstance(value, str) and value.strip():
            visuals[key] = value.strip()

    return visuals


def _normalise_prompts_from_entry(
    entry: Dict[str, Any],
    organ_name: str,
    visuals: Dict[str, str],
) -> Tuple[str, str, str]:
    """
    Use prompts stored in medical_knowledge_v2.json.
    If any prompt is missing, rebuild a safe one from organ + visuals.
    """
    fallback_text, fallback_attr, fallback_morph = _build_organ_static_prompts(
        base_prompt=STRICT_BASE_PROMPT,
        organ_name=organ_name,
        visuals=visuals,
    )

    text_prompt = entry.get("text_prompt", fallback_text)
    attribute_text = entry.get("attribute_text", fallback_attr)
    morphology_text = entry.get("morphology_text", fallback_morph)

    if not isinstance(text_prompt, str) or not text_prompt.strip():
        text_prompt = fallback_text
    if not isinstance(attribute_text, str) or not attribute_text.strip():
        attribute_text = fallback_attr
    if not isinstance(morphology_text, str) or not morphology_text.strip():
        morphology_text = fallback_morph

    return text_prompt.strip(), attribute_text.strip(), morphology_text.strip()


def _build_from_organ_prior(
    meta: Dict[str, Any],
    organ_name: str,
) -> Tuple[List[int], Dict[str, str], Tuple[str, str, str]]:
    """
    Build non-sample-specific prompts from train-split organ priors.

    This is used by default in val/test to avoid using val/test sample-specific
    GT-derived attributes from medical_knowledge_v2.
    """
    organ_priors = meta.get("organ_priors", {}) if isinstance(meta, dict) else {}
    prior = organ_priors.get(organ_name, None)

    if prior is None:
        prior = organ_priors.get("Generic", None)

    if prior is None:
        labels = _default_attr_labels()
        visuals = _default_visuals()
    else:
        labels = _to_int_list(prior.get("attr_labels", None), default=_default_attr_labels())
        visuals = prior.get("visual_stats", _default_visuals())
        if not isinstance(visuals, dict):
            visuals = _default_visuals()
        merged = _default_visuals()
        for key in ["color", "shape", "arrangement", "size", "density"]:
            if isinstance(visuals.get(key, None), str) and visuals[key].strip():
                merged[key] = visuals[key].strip()
        visuals = merged

    prompts = _build_organ_static_prompts(
        base_prompt=STRICT_BASE_PROMPT,
        organ_name=organ_name,
        visuals=visuals,
    )

    return labels, visuals, prompts


# ==============================================================================
# 4. Physical attribute analyzer for dynamic_gt / fallback
# ==============================================================================
def _estimate_color_from_nuclei(image: np.ndarray, mask: np.ndarray, config: AttributeConfig) -> Tuple[int, str]:
    """
    Estimate coarse nuclear staining intensity from masked pixels.

    Current PNuRL color head has 2 classes:
        0: deep-purple stained
        1: light-purple stained
    """
    if image is None or image.ndim != 3 or mask.sum() == 0:
        return 0, "deep-purple stained"

    nuclei_pixels = image[mask > 0]
    if nuclei_pixels.size == 0:
        return 0, "deep-purple stained"

    mean_intensity = float(nuclei_pixels.mean())

    if mean_intensity >= config.COLOR_BRIGHT:
        return 1, "light-purple stained"

    return 0, "deep-purple stained"


def analyze_physical_attributes(
    image: np.ndarray,
    mask: np.ndarray,
    config: AttributeConfig,
    area_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Estimate physical attributes from an image/mask pair.

    labels order:
        [Color, Shape, Arrangement, Size, Density]
    """
    default_results = {
        "visuals": _default_visuals(),
        "labels": _default_attr_labels(),
        "source": "default",
    }

    if mask is None or mask.sum() == 0:
        return default_results

    mask = (mask > 0).astype(np.uint8)

    # Downsample for fast connected-component analysis.
    analysis_scale = 256.0 / max(mask.shape)
    if analysis_scale < 1.0:
        h, w = mask.shape[:2]
        new_h = max(1, int(h * analysis_scale))
        new_w = max(1, int(w * analysis_scale))
        small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        area_scale_factor = (1.0 / analysis_scale) ** 2
    else:
        small_mask = mask
        area_scale_factor = 1.0

    num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
        small_mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        col_lbl, col_txt = _estimate_color_from_nuclei(image, mask, config)
        out = dict(default_results)
        out["visuals"] = dict(default_results["visuals"])
        out["labels"] = list(default_results["labels"])
        out["visuals"]["color"] = col_txt
        out["labels"][0] = col_lbl
        out["source"] = "physical_empty_or_single"
        return out

    # 1. Size
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) * area_scale_factor
    mean_area = float(np.mean(areas)) if len(areas) > 0 else 0.0

    th_small = config.AREA_SMALL * area_scale
    th_large = config.AREA_LARGE * area_scale

    if mean_area < th_small:
        size_lbl, size_txt = 0, "small-sized"
    elif mean_area > th_large:
        size_lbl, size_txt = 2, "large-sized"
    else:
        size_lbl, size_txt = 1, "medium-sized"

    # 2. Shape
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    aspect_ratios = widths / (heights + 1e-5)
    mean_ar_dev = float(np.mean(np.abs(1.0 - aspect_ratios))) if len(aspect_ratios) > 0 else 0.0

    if mean_ar_dev < 0.3:
        shape_lbl, shape_txt = 0, "round"
    elif mean_ar_dev < 0.6:
        shape_lbl, shape_txt = 1, "oval"
    else:
        shape_lbl, shape_txt = 2, "elongated"

    # 3. Density
    count = int(num_labels - 1)
    sparse_count_th = config.DENSITY_SPARSE * 100.0
    dense_count_th = config.DENSITY_DENSE * 100.0

    if count < sparse_count_th:
        den_lbl, den_txt = 0, "sparsely distributed"
    elif count > dense_count_th:
        den_lbl, den_txt = 2, "densely distributed"
    else:
        den_lbl, den_txt = 1, "moderately distributed"

    # 4. Arrangement
    if count > 5 and SKLEARN_AVAILABLE:
        try:
            pts = centroids[1:].astype(np.float32)
            tree = KDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn_dists = dists[:, 1]
            dist_cv = float(np.std(nn_dists) / (np.mean(nn_dists) + 1e-6))

            if dist_cv > config.ARRANGE_CLUMPED:
                arr_lbl, arr_txt = 1, "disordered/clustered"
            else:
                arr_lbl, arr_txt = 0, "uniformly arranged"
        except Exception:
            arr_lbl, arr_txt = 0, "uniformly arranged"
    else:
        arr_lbl, arr_txt = 0, "uniformly arranged"

    # 5. Color
    col_lbl, col_txt = _estimate_color_from_nuclei(image, mask, config)

    return {
        "visuals": {
            "color": col_txt,
            "shape": shape_txt,
            "arrangement": arr_txt,
            "size": size_txt,
            "density": den_txt,
        },
        "labels": [col_lbl, shape_lbl, arr_lbl, size_lbl, den_lbl],
        "source": "physical",
    }


def generate_adaptive_density(mask: np.ndarray, image_size=(1024, 1024)) -> np.ndarray:
    """
    Generate an adaptive density / point heatmap.

    Args:
        mask:
            Binary nuclei mask, shape [H, W].
        image_size:
            Target output size. Can be int or (H, W).

    Returns:
        heatmap:
            Float32 heatmap, shape [target_h, target_w], roughly in [0, 1].
    """
    if isinstance(image_size, int):
        target_h, target_w = image_size, image_size
    else:
        target_h, target_w = image_size

    target_h, target_w = int(target_h), int(target_w)
    scale = 0.25

    small_h = max(1, int(target_h * scale))
    small_w = max(1, int(target_w * scale))

    mask = (mask > 0).astype(np.uint8)
    small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(small_mask, connectivity=8)
    heatmap = np.zeros((small_h, small_w), dtype=np.float32)

    if num_labels <= 1:
        return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    points = centroids[1:].astype(np.float32)

    if len(points) > 200:
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < small_h and 0 <= x < small_w:
                heatmap[y, x] = 1.0

        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3.0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
    else:
        if SKLEARN_AVAILABLE and len(points) > 1:
            tree = KDTree(points)
            dists, _ = tree.query(points, k=min(4, len(points)))
        else:
            dists = None

        for i, pt in enumerate(points):
            x0, y0 = int(pt[0]), int(pt[1])

            if dists is not None and len(dists[i]) > 1:
                sigma = 0.3 * float(np.mean(dists[i][1:]))
            else:
                sigma = 4.0

            sigma = max(1.0, min(float(sigma), 15.0))

            k_size = int(sigma * 3) * 2 + 1
            kernel = cv2.getGaussianKernel(k_size, sigma)
            kernel = (kernel @ kernel.T).astype(np.float32)

            kh, kw = kernel.shape
            y_min = max(0, y0 - kh // 2)
            y_max = min(small_h, y0 + kh // 2 + 1)
            x_min = max(0, x0 - kw // 2)
            x_max = min(small_w, x0 + kw // 2 + 1)

            ky_min = kh // 2 - (y0 - y_min)
            ky_max = ky_min + (y_max - y_min)
            kx_min = kw // 2 - (x0 - x_min)
            kx_max = kx_min + (x_max - x_min)

            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                kernel[ky_min:ky_max, kx_min:kx_max],
            )

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

    return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def generate_hv_map(inst_mask: np.ndarray) -> np.ndarray:
    """
    Convert instance mask to HoVer-style horizontal/vertical distance map.

    Returns:
        hv_map: [2, H, W]
            channel 0: vertical distance
            channel 1: horizontal distance
            background is 0, nucleus interior is roughly [-1, 1].
    """
    if inst_mask.ndim != 2:
        raise ValueError(f"inst_mask must be 2D, got shape={inst_mask.shape}")

    h, w = inst_mask.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_mask = inst_mask.astype(np.int32)
    props = regionprops(inst_mask)

    for prop in props:
        if prop.label == 0:
            continue

        y_min, x_min, y_max, x_max = prop.bbox
        y_c, x_c = prop.centroid

        if y_max <= y_min or x_max <= x_min:
            continue

        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

        y_den = (y_max - y_min) / 2.0 + 1e-8
        x_den = (x_max - x_min) / 2.0 + 1e-8

        y_dist = ((y_grid - y_c) / y_den).astype(np.float32)
        x_dist = ((x_grid - x_c) / x_den).astype(np.float32)

        y_dist = np.clip(y_dist, -1.0, 1.0)
        x_dist = np.clip(x_dist, -1.0, 1.0)

        inst_bool = inst_mask[y_min:y_max, x_min:x_max] == prop.label

        v_crop = hv_map[0, y_min:y_max, x_min:x_max]
        h_crop = hv_map[1, y_min:y_max, x_min:x_max]
        v_crop[inst_bool] = y_dist[inst_bool]
        h_crop[inst_bool] = x_dist[inst_bool]

    return hv_map



def generate_boundary_uncertainty_targets(
    inst_mask: np.ndarray,
    boundary_radius: int = 4,
    contact_radius: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate foreground / background / boundary / uncertainty targets from an instance mask.

    Args:
        inst_mask:
            Instance mask, shape [H, W], 0=background, >0=instance id.
        boundary_radius:
            Radius used to build per-instance inner/outer boundary ring.
            Use a larger value after resizing small crops to 1024, e.g. 6-8.
        contact_radius:
            Radius used to detect touching / close instances. If None, use boundary_radius.

    Returns:
        Dict with float32 maps in [0, 1], each shape [H, W]:
            fg_target:
                Binary nuclear foreground.
            bg_target:
                Conservative background region away from nuclei and boundary rings.
            boundary_target:
                Per-instance boundary ring, including inner and outer contours.
            uncertain_target:
                Ambiguous boundary/contact region. This includes boundary_target and
                dilated overlap between neighboring instances.
    """
    if inst_mask.ndim != 2:
        raise ValueError(f"inst_mask must be 2D, got shape={inst_mask.shape}")

    inst_mask = inst_mask.astype(np.int32)
    h, w = inst_mask.shape

    boundary_radius = int(max(1, boundary_radius))
    if contact_radius is None:
        contact_radius = boundary_radius
    contact_radius = int(max(1, contact_radius))

    fg = (inst_mask > 0).astype(np.uint8)

    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * boundary_radius + 1, 2 * boundary_radius + 1),
    )
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * contact_radius + 1, 2 * contact_radius + 1),
    )

    boundary = np.zeros((h, w), dtype=np.uint8)
    dilated_owner_count = np.zeros((h, w), dtype=np.uint16)

    inst_ids = np.unique(inst_mask)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        one_inst = (inst_mask == inst_id).astype(np.uint8)
        if one_inst.sum() == 0:
            continue

        dilated = cv2.dilate(one_inst, boundary_kernel, iterations=1)
        eroded = cv2.erode(one_inst, boundary_kernel, iterations=1)

        # Inner + outer instance contour ring.
        ring = (dilated - eroded).clip(0, 1).astype(np.uint8)
        boundary = np.maximum(boundary, ring)

        # Close/touching nuclei cue. If two instance dilations overlap, that region is ambiguous.
        contact_dilated = cv2.dilate(one_inst, contact_kernel, iterations=1)
        dilated_owner_count += contact_dilated.astype(np.uint16)

    contact_region = (dilated_owner_count >= 2).astype(np.uint8)
    uncertain = np.maximum(boundary, contact_region).astype(np.uint8)

    # Conservative background excludes boundary and near-contact regions.
    fg_dilated = cv2.dilate(fg, boundary_kernel, iterations=1)
    bg = ((fg_dilated == 0) & (uncertain == 0)).astype(np.uint8)

    return {
        "fg_target": fg.astype(np.float32),
        "bg_target": bg.astype(np.float32),
        "boundary_target": boundary.astype(np.float32),
        "uncertain_target": uncertain.astype(np.float32),
    }


# ==============================================================================
# 4.4b  Dense Boundary Map Generation (Phase B — MultiLevelAttributeHeads)
# ==============================================================================


def generate_dense_boundary_maps(
    inst_mask: np.ndarray,
    small_nuclei_area_thresh: float = 500.0,
) -> Dict[str, np.ndarray]:
    """
    从实例掩码生成4张密集边界图，用于 DenseBoundaryHead 监督。

    Args:
        inst_mask: [H, W] int32, 0=background, >0=instance id
        small_nuclei_area_thresh: 面积阈值，低于此的核标记为 small_nuclei

    Returns:
        dict with 4 keys, each [H, W] float32 in [0,1]:
            boundary_map:      核边界（内+外轮廓）
            touching_region:   相邻核重叠/接触区域
            small_nuclei:      小核区域（面积 < threshold）
            hv_gradient:       HV 图梯度幅值（归一化到 [0,1]）
    """
    h, w = inst_mask.shape
    inst_mask = inst_mask.astype(np.int32)
    props = regionprops(inst_mask)

    # --- boundary_map: 核边界 (复用 generate_boundary_uncertainty_targets 的逻辑) ---
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary_map = np.zeros((h, w), dtype=np.uint8)
    dilated_owner = np.zeros((h, w), dtype=np.uint16)

    for prop in props:
        if prop.label == 0:
            continue
        one_inst = (inst_mask == prop.label).astype(np.uint8)
        if one_inst.sum() == 0:
            continue
        dilated = cv2.dilate(one_inst, boundary_kernel, iterations=1)
        eroded = cv2.erode(one_inst, boundary_kernel, iterations=1)
        ring = (dilated - eroded).clip(0, 1).astype(np.uint8)
        boundary_map = np.maximum(boundary_map, ring)
        dilated_owner += dilated.astype(np.uint16)

    # --- touching_region: 重叠/接触区域 ---
    touching_region = (dilated_owner >= 2).astype(np.uint8)

    # --- small_nuclei: 面积小于阈值的小核 ---
    small_map = np.zeros((h, w), dtype=np.uint8)
    for prop in props:
        if prop.label == 0:
            continue
        if prop.area < small_nuclei_area_thresh:
            one_inst = (inst_mask == prop.label).astype(np.uint8)
            small_map = np.maximum(small_map, one_inst)

    # --- hv_gradient: 从 HV 图计算梯度幅值 ---
    hv_map = generate_hv_map(inst_mask)
    # 对 HV 两个通道分别计算梯度幅值然后平均
    grad_x_h = cv2.Sobel(hv_map[0], cv2.CV_32F, 1, 0, ksize=3)
    grad_y_h = cv2.Sobel(hv_map[0], cv2.CV_32F, 0, 1, ksize=3)
    grad_x_v = cv2.Sobel(hv_map[1], cv2.CV_32F, 1, 0, ksize=3)
    grad_y_v = cv2.Sobel(hv_map[1], cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_h = np.sqrt(grad_x_h ** 2 + grad_y_h ** 2)
    grad_mag_v = np.sqrt(grad_x_v ** 2 + grad_y_v ** 2)
    hv_gradient = (grad_mag_h + grad_mag_v) * 0.5
    # 归一化到 [0,1]
    max_grad = hv_gradient.max()
    if max_grad > 1e-6:
        hv_gradient = hv_gradient / max_grad
    else:
        hv_gradient = np.zeros_like(hv_gradient)

    return {
        "boundary_map": boundary_map.astype(np.float32),
        "touching_region": touching_region.astype(np.float32),
        "small_nuclei": small_map.astype(np.float32),
        "hv_gradient": hv_gradient.astype(np.float32),
    }


# ==============================================================================
# 4.4c  Instance Morphology Attribute Computation (Phase B)
# ==============================================================================
# 6 instance-level attributes:
#   size, elongation, boundary_irregularity, local_crowding, roundness, solidity
# Each discretized into 3 classes (low/medium/high).


INSTANCE_MORPH_ATTR_NAMES: Tuple[str, ...] = (
    "size",
    "elongation",
    "boundary_irregularity",
    "local_crowding",
    "roundness",
    "solidity",
)
NUM_INSTANCE_MORPH_ATTRS: int = 6


def _discretize_3class(values: np.ndarray, bins: Tuple[float, float]) -> np.ndarray:
    """
    Discretize continuous values into 3 classes (0=low, 1=mid, 2=high).
    bins: (low_thresh, high_thresh)
    """
    result = np.ones_like(values, dtype=np.int64)  # default mid=1
    result[values < bins[0]] = 0  # low
    result[values >= bins[1]] = 2  # high
    return result


def compute_instance_morphology_attrs(
    inst_mask: np.ndarray,
    min_instance_area: int = 8,
    max_instances_per_image: int = 128,
) -> Dict[str, Any]:
    """
    计算每个实例的6个形态属性，同时返回 per-sample 聚合标签和 per-instance 标签。

    Args:
        inst_mask: [H, W] int32, 0=background, >0=instance id
        min_instance_area: Filter out instances with area < this value (default: 8).
        max_instances_per_image: Max instances to keep; if exceeded, sort by area
                                 descending and take top-K (default: 128).

    Returns:
        dict with:
            instance_attr_labels: [6] long tensor, per-sample aggregated labels (0/1/2)
            instance_attr_values: [6] float tensor, per-sample aggregated continuous values
            per_instance_attr_labels: [N, 6] long tensor, per-instance discretized labels
            per_instance_attr_values: [N, 6] float tensor, per-instance continuous values
            per_instance_ids: [N] int64 array, instance IDs in same order as per_instance_attr_labels rows
            per_instance: list of dicts with per-instance details
    """
    from skimage.measure import perimeter

    inst_mask = inst_mask.astype(np.int32)
    props = regionprops(inst_mask)

    # --- Filter by min_instance_area ---
    if min_instance_area > 0:
        props = [p for p in props if p.area >= min_instance_area]

    # --- Truncate by max_instances_per_image ---
    if max_instances_per_image > 0 and len(props) > max_instances_per_image:
        # Sort by area descending, keep top-K
        props = sorted(props, key=lambda p: p.area, reverse=True)[:max_instances_per_image]

    if len(props) == 0:
        # No instances — return mid (1) as default
        return {
            "instance_attr_labels": np.ones(NUM_INSTANCE_MORPH_ATTRS, dtype=np.int64),
            "instance_attr_values": np.zeros(NUM_INSTANCE_MORPH_ATTRS, dtype=np.float32),
            "per_instance_attr_labels": np.zeros((0, NUM_INSTANCE_MORPH_ATTRS), dtype=np.int64),
            "per_instance_attr_values": np.zeros((0, NUM_INSTANCE_MORPH_ATTRS), dtype=np.float32),
            "per_instance_ids": np.zeros((0,), dtype=np.int64),
            "per_instance": [],
        }

    n_inst = len(props)
    areas = np.zeros(n_inst, dtype=np.float32)
    eccentricities = np.zeros(n_inst, dtype=np.float32)
    perimeters = np.zeros(n_inst, dtype=np.float32)
    roundness_vals = np.zeros(n_inst, dtype=np.float32)
    solidities = np.zeros(n_inst, dtype=np.float32)

    centroids = []
    for i, prop in enumerate(props):
        if prop.label == 0:
            continue
        areas[i] = float(prop.area)
        eccentricities[i] = float(prop.eccentricity)
        # perimeter (using skimage measure)
        mask_i = (inst_mask == prop.label).astype(np.uint8)
        perim = perimeter(mask_i)
        perimeters[i] = float(perim)
        # roundness = 4 * pi * area / perimeter^2
        if perim > 0:
            roundness_vals[i] = float(4.0 * np.pi * prop.area / (perim * perim))
        else:
            roundness_vals[i] = 0.0
        # solidity = area / convex_area (area_convex for skimage>=0.25, convex_area for legacy)
        # Use explicit branch to avoid FutureWarning from deprecated property access
        if hasattr(prop, "area_convex"):
            _conv_area = float(prop.area_convex)
        elif hasattr(prop, "convex_area"):
            _conv_area = float(prop.convex_area)
        else:
            _conv_area = 0.0
        if _conv_area > 0:
            solidities[i] = float(prop.area / _conv_area)
        else:
            solidities[i] = 0.0
        centroids.append(prop.centroid)

    # boundary_irregularity = perimeter / (2 * sqrt(pi * area)) — higher = more irregular
    boundary_irregularity = np.zeros(n_inst, dtype=np.float32)
    for i in range(n_inst):
        if areas[i] > 0:
            boundary_irregularity[i] = float(
                perimeters[i] / (2.0 * np.sqrt(np.pi * areas[i]))
            )
        else:
            boundary_irregularity[i] = 1.0

    # local_crowding: mean distance to k-nearest neighbors (k=min(5, n_inst-1))
    local_crowding = np.zeros(n_inst, dtype=np.float32)
    if n_inst >= 2 and len(centroids) == n_inst:
        centroids_arr = np.array(centroids, dtype=np.float32)
        k = min(5, n_inst - 1)
        if SKLEARN_AVAILABLE:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nn.fit(centroids_arr)
            distances, _ = nn.kneighbors(centroids_arr)
            local_crowding = distances[:, 1:].mean(axis=1)  # exclude self
        elif SCIPY_AVAILABLE:
            from scipy.spatial import KDTree
            tree = KDTree(centroids_arr)
            # For each point, query k+1 (including self)
            distances, _ = tree.query(centroids_arr, k=min(k + 1, n_inst))
            local_crowding = distances[:, 1:].mean(axis=1)
        else:
            local_crowding = np.ones(n_inst, dtype=np.float32) * 50.0  # fallback

    # --- Per-instance discretized labels [N, 6] ---
    per_inst_size_labels = _discretize_3class(areas, bins=(500.0, 2000.0))
    per_inst_elong_labels = _discretize_3class(eccentricities, bins=(0.4, 0.8))
    per_inst_irreg_labels = _discretize_3class(boundary_irregularity, bins=(1.2, 2.0))
    per_inst_crowd_labels = _discretize_3class(local_crowding, bins=(30.0, 80.0))
    per_inst_round_labels = _discretize_3class(roundness_vals, bins=(0.3, 0.7))
    per_inst_solid_labels = _discretize_3class(solidities, bins=(0.6, 0.9))

    # Per-instance IDs in same order as props (regionprops iteration order)
    per_instance_ids = np.array([int(prop.label) for prop in props], dtype=np.int64)

    per_instance_attr_labels = np.stack([
        per_inst_size_labels, per_inst_elong_labels, per_inst_irreg_labels,
        per_inst_crowd_labels, per_inst_round_labels, per_inst_solid_labels,
    ], axis=1)  # [N, 6]

    per_instance_attr_values = np.stack([
        areas, eccentricities, boundary_irregularity,
        local_crowding, roundness_vals, solidities,
    ], axis=1)  # [N, 6]

    # --- Aggregate per-sample: weighted by area (larger instances contribute more) ---
    weights = areas / (areas.sum() + 1e-8)

    # Compute weighted means
    avg_size = float(np.average(areas, weights=weights)) if n_inst > 0 else 0.0
    avg_elongation = float(np.average(eccentricities, weights=weights)) if n_inst > 0 else 0.5
    avg_irregularity = float(np.average(boundary_irregularity, weights=weights)) if n_inst > 0 else 1.0
    avg_crowding = float(np.average(local_crowding, weights=weights)) if n_inst > 0 else 50.0
    avg_roundness = float(np.average(roundness_vals, weights=weights)) if n_inst > 0 else 0.5
    avg_solidity = float(np.average(solidities, weights=weights)) if n_inst > 0 else 0.8

    # --- Discretize to 3 classes ---
    size_label = _discretize_3class(
        np.array([avg_size]),
        bins=(500.0, 2000.0),
    )[0]
    elongation_label = _discretize_3class(
        np.array([avg_elongation]),
        bins=(0.4, 0.8),
    )[0]
    irregularity_label = _discretize_3class(
        np.array([avg_irregularity]),
        bins=(1.2, 2.0),
    )[0]
    crowding_label = _discretize_3class(
        np.array([avg_crowding]),
        bins=(30.0, 80.0),
    )[0]
    roundness_label = _discretize_3class(
        np.array([avg_roundness]),
        bins=(0.3, 0.7),
    )[0]
    solidity_label = _discretize_3class(
        np.array([avg_solidity]),
        bins=(0.6, 0.9),
    )[0]

    attr_labels = np.array([
        size_label, elongation_label, irregularity_label,
        crowding_label, roundness_label, solidity_label,
    ], dtype=np.int64)

    attr_values = np.array([
        avg_size, avg_elongation, avg_irregularity,
        avg_crowding, avg_roundness, avg_solidity,
    ], dtype=np.float32)

    per_instance = [
        {
            "instance_id": int(prop.label),
            "area": float(areas[i]),
            "eccentricity": float(eccentricities[i]),
            "perimeter": float(perimeters[i]),
            "boundary_irregularity": float(boundary_irregularity[i]),
            "local_crowding": float(local_crowding[i]),
            "roundness": float(roundness_vals[i]),
            "solidity": float(solidities[i]),
        }
        for i, prop in enumerate(props)
    ]

    return {
        "instance_attr_labels": attr_labels,
        "instance_attr_values": attr_values,
        "per_instance_attr_labels": per_instance_attr_labels,
        "per_instance_attr_values": per_instance_attr_values,
        "per_instance_ids": per_instance_ids,
        "per_instance": per_instance,
    }


# ==============================================================================
# 4.5  Structure & Boundary Attribute Loading (GT-derived pathology attrs)
# ==============================================================================

STRUCTURE_ATTR_NAMES = [
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
]

BOUNDARY_ATTR_NAMES = [
    "boundary_density",
    "nuclear_irregularity",
    "nuclear_elongation",
    "small_nuclei_ratio",
]

# touching_or_crowding_difficulty is excluded from boundary attr head (debug only).

INVALID_ATTR_LABEL = -1

# Labelled class mapping
ATTR_LABEL_MAP = {
    "low": 0,
    "mid": 1,
    "high": 2,
    "invalid": INVALID_ATTR_LABEL,
}


def load_structure_boundary_attrs(attr_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load GT-derived structure & boundary attribute samples from a JSONL file.

    Each line is a JSON object with:
        sample_id      (str) - matching the image stem
        structure_attrs (dict) - raw continuous values
        boundary_attrs  (dict) - raw continuous values
        discretized_labels (dict) - discrete class labels (0=low,1=mid,2=high,-1=invalid)

    Returns:
        Dict mapping sample_id -> record dict with keys:
            structure_attr_values: List[float] (5 elements)
            boundary_attr_values:  List[float] (5 elements, touching_or_crowding_difficulty included raw)
            structure_attr_labels: List[int]   (5 discretized labels)
            boundary_attr_labels:  List[int]   (4 discretized labels, touching_or_crowding_difficulty excluded)
    """
    records: Dict[str, Dict[str, Any]] = {}

    if not os.path.isfile(attr_path):
        _dataloader_print(f"⚠️ [DataLoader] structure_boundary_attr_path not found: {attr_path}")
        return records

    with open(attr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = rec.get("sample_id")
            if not sample_id:
                continue

            structure_raw = rec.get("structure_attrs", {})
            boundary_raw = rec.get("boundary_attrs", {})
            disc = rec.get("discretized_labels", {})

            # Build ordered value lists
            structure_values = [float(structure_raw.get(k, 0.0)) for k in STRUCTURE_ATTR_NAMES]
            boundary_values = [float(boundary_raw.get(k, 0.0)) for k in BOUNDARY_ATTR_NAMES]

            # Build discretised labels; missing -> INVALID_ATTR_LABEL
            structure_labels = [int(disc.get(k, INVALID_ATTR_LABEL)) for k in STRUCTURE_ATTR_NAMES]
            boundary_labels = [int(disc.get(k, INVALID_ATTR_LABEL)) for k in BOUNDARY_ATTR_NAMES]

            records[sample_id] = {
                "structure_attr_values": structure_values,
                "boundary_attr_values": boundary_values,
                "structure_attr_labels": structure_labels,
                "boundary_attr_labels": boundary_labels,
            }

    return records


# ==============================================================================
# 5. Dataset
# ==============================================================================
class UniversalDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        knowledge_path,
        image_size=1024,
        crop_size=256,
        mode="train",
        prompt_mode="organ_static",
        use_structure_boundary_attrs=False,
        structure_boundary_attr_path=None,
        min_instance_area=8,
        max_instances_per_image=128,
        skip_knowledge_loading=False,
        phase="unknown",
        enable_local_region_text_alignment=False,
        local_region_window_size=192,
        local_region_thresholds_path=None,
        local_region_audit=False,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = str(mode).lower().strip()
        self.raw_mode = mode
        self.min_instance_area = min_instance_area
        self.max_instances_per_image = max_instances_per_image
        self.phase = phase
        self.enable_local_region_text_alignment = bool(enable_local_region_text_alignment)
        self.local_region_window_size = int(local_region_window_size)
        self.local_region_audit = bool(local_region_audit)
        self._local_region_audit_count = 0
        self.local_region_thresholds = None
        if self.enable_local_region_text_alignment:
            if self.mode != "train":
                raise ValueError(
                    "GT local-region attributes are train-only and must not be "
                    f"constructed for mode={self.mode!r}"
                )
            if not local_region_thresholds_path:
                raise ValueError("local_region_thresholds_path is required for L1-A")
            self.local_region_thresholds = load_l0_thresholds(local_region_thresholds_path)

        canonical_prompt_mode, raw_prompt_mode = normalize_prompt_mode(prompt_mode, default="organ_static")
        self.requested_prompt_mode = raw_prompt_mode
        self.prompt_mode = canonical_prompt_mode

        if raw_prompt_mode != canonical_prompt_mode:
            _dataloader_print(
                f"⚠️ [DataLoader] prompt_mode='{prompt_mode}' is deprecated or unknown; "
                f"using canonical prompt_mode='{canonical_prompt_mode}'."
            )

        # Hard guard against GT prompt leakage during normal validation/test.
        if self.prompt_mode == "dynamic_gt" and self.mode not in {"train", "oracle", "debug"}:
            _dataloader_print(
                f"⚠️ [DataLoader] prompt_mode='dynamic_gt' is not allowed in mode='{self.mode}'. "
                f"Falling back to 'organ_static' to avoid GT-derived prompt leakage."
            )
            self.prompt_mode = "organ_static"

        self.organ_map = ORGAN_TO_ID

        # Previous code used fixed 20% organ dropout.
        # It caused organ_id=Generic while text still contained organ-specific terms.
        # Default is now 0.0. Enable explicitly only for ablation:
        #   ORGAN_DROPOUT_PROB=0.2
        self.organ_dropout_prob = float(os.environ.get("ORGAN_DROPOUT_PROB", "0.0"))

        # Normal val/test should not consume val/test sample-specific GT-derived v2 attributes.
        # Default:
        #   train/oracle/debug: use sample-level v2 attrs.
        #   val/test: use train-split organ priors from __meta__.
        # For explicit oracle/debug check:
        #   ALLOW_EVAL_SAMPLE_ATTRIBUTES=1
        self.allow_eval_sample_attributes = os.environ.get("ALLOW_EVAL_SAMPLE_ATTRIBUTES", "0") == "1"

        # ------------------------------------------------------------------
        # Phase B (multilevel_attr_warmup): skip knowledge loading entirely.
        # Build sample list by scanning data_root/{mode} for .png files.
        # ------------------------------------------------------------------
        if skip_knowledge_loading:
            self.full_db = {}
            self.meta = {}
            self.is_v2 = False
            self.attr_config = AttributeConfig()

            self.samples = []
            mode_dir = os.path.join(data_root, self.raw_mode)
            if os.path.isdir(mode_dir):
                for fname in sorted(os.listdir(mode_dir)):
                    if not fname.lower().endswith(".png"):
                        continue
                    stem = os.path.splitext(fname)[0]
                    full_img_path = os.path.join(mode_dir, fname)
                    json_path = os.path.join(mode_dir, stem + ".json")
                    if os.path.exists(full_img_path) and os.path.exists(json_path):
                        self.samples.append({
                            "img_path": full_img_path,
                            "json_path": json_path,
                            "data": {},
                            "rel_path": f"{self.raw_mode}/{fname}",
                        })

            _dataloader_print(
                f"📁 [DataLoader] Scanned {mode_dir}: "
                f"Loaded {len(self.samples)} samples (skip_knowledge_loading=True)"
            )
        else:
            _dataloader_print(f"📖 [DataLoader] Loading Knowledge Base: {knowledge_path}")
            with open(knowledge_path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Detect format: standard JSON (single dict) vs JSONL (one JSON object per line).
            raw_stripped = raw.strip()
            is_jsonl = False
            if raw_stripped.startswith("{"):
                try:
                    _test = json.loads(raw_stripped)
                    # Single JSON object – standard format (e.g. medical_knowledge.json)
                    full_db = _test
                except json.JSONDecodeError:
                    is_jsonl = True
            else:
                is_jsonl = True

            if is_jsonl:
                full_db = {}
                for line in raw_stripped.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    _sample_id = entry.get("sample_id", "")
                    _split = entry.get("split", "train")
                    _fname = entry.get("image_path", f"{_sample_id}.png")
                    path_key = f"{_split}/{_fname}"
                    full_db[path_key] = entry

            self.meta = full_db.pop("__meta__", {})
            self.is_v2 = str(self.meta.get("version", "")).lower().startswith("promptnu_freqpath_v2")

            # Attribute config is only for dynamic_gt / fallback.
            if self.meta:
                stats = self.meta.get("stats", None)
                if stats is None:
                    stats = self.meta.get("train_thresholds", {})
                self.attr_config = AttributeConfig.from_metadata(stats)
            else:
                _dataloader_print("⚠️ [DataLoader] Warning: '__meta__' not found. Using default thresholds.")
                self.attr_config = AttributeConfig()

            self.full_db = full_db

            self.samples = []
            skipped = 0

            for rel_path, entry in self.full_db.items():
                if entry.get("split") != self.raw_mode:
                    skipped += 1
                    continue

                if os.path.isabs(rel_path):
                    full_img_path = rel_path
                else:
                    full_img_path = os.path.join(data_root, rel_path)

                full_json_path = self._image_path_to_json_path(full_img_path)

                if os.path.exists(full_img_path) and os.path.exists(full_json_path):
                    self.samples.append(
                        {
                            "img_path": full_img_path,
                            "json_path": full_json_path,
                            "data": entry,
                            "rel_path": rel_path,
                        }
                    )
                else:
                    skipped += 1

            _dataloader_print(
                f"✅ [DataLoader] Mode: {self.raw_mode} | Prompt: {self.prompt_mode} "
                f"(requested: {self.requested_prompt_mode}) | "
                f"V2={self.is_v2} | "
                f"OrganDropout={self.organ_dropout_prob:.3f} | "
                f"AllowEvalSampleAttrs={self.allow_eval_sample_attributes} | "
                f"Loaded: {len(self.samples)} | Skipped: {skipped}"
            )

        # ------------------------------------------------------------------
        # Structure & Boundary Attribute Loading (GT-derived pathology attrs)
        # ------------------------------------------------------------------
        self.use_structure_boundary_attrs = bool(use_structure_boundary_attrs)
        self.structure_boundary_attr_path = structure_boundary_attr_path
        self.structure_boundary_attr_records: Dict[str, Dict[str, Any]] = {}
        self._sb_loaded = False

        if self.use_structure_boundary_attrs:
            sb_path = self.structure_boundary_attr_path or os.path.join(
                os.path.dirname(knowledge_path), "..", "attr_stats",
                "gt_structure_boundary_attr_samples.jsonl",
            )
            self.structure_boundary_attr_records = load_structure_boundary_attrs(sb_path)
            self._sb_loaded = True

            # Match samples
            matched = 0
            missing = 0
            for sample in self.samples:
                rel_path = sample.get("rel_path", "")
                stem = os.path.splitext(os.path.basename(rel_path))[0]
                if stem in self.structure_boundary_attr_records:
                    sample["_sb_sample_id"] = stem
                    matched += 1
                else:
                    sample["_sb_sample_id"] = None
                    missing += 1

            _dataloader_print(
                f"📊 [DataLoader] Structure & Boundary Attrs: "
                f"Loaded={len(self.structure_boundary_attr_records)} | "
                f"Matched={matched} | Missing={missing} | "
                f"Structure={STRUCTURE_ATTR_NAMES} | "
                f"Boundary={BOUNDARY_ATTR_NAMES}"
            )

            if missing > 0:
                _dataloader_print(
                    f"⚠️ [DataLoader] {missing}/{len(self.samples)} samples "
                    f"missing structure/boundary attrs — will return invalid labels."
                )

            # ── [SB_ATTR_DATA_AUDIT] detailed audit (gated behind audit_mode=debug) ──
            if self.use_structure_boundary_attrs and self.structure_boundary_attr_records:
                _first_sb_sample_id = None
                _first_structure_labels = None
                _first_boundary_labels = None
                for sample in self.samples:
                    _sid = sample.get("_sb_sample_id", None)
                    if _sid is not None and _sid in self.structure_boundary_attr_records:
                        _first_sb_sample_id = _sid
                        _rec = self.structure_boundary_attr_records[_sid]
                        _first_structure_labels = _rec.get("structure_attr_labels", [])
                        _first_boundary_labels = _rec.get("boundary_attr_labels", [])
                        break
                _attr_path = self.structure_boundary_attr_path or "N/A"
                _attr_file_exists = os.path.isfile(_attr_path) if _attr_path != "N/A" else False
                audit_print(
                    "SB_ATTR_DATA_AUDIT",
                    f"\n[SB_ATTR_DATA_AUDIT] phase={self.phase} | "
                    f"use_structure_boundary_attrs={self.use_structure_boundary_attrs} | "
                    f"structure_boundary_attr_path={_attr_path} | "
                    f"attr_file_exists={_attr_file_exists} | "
                    f"loaded_attr_records={len(self.structure_boundary_attr_records)} | "
                    f"matched_samples={matched} | "
                    f"missing_samples={missing} | "
                    f"first_sample_id={_first_sb_sample_id} | "
                    f"first_structure_attr_labels={_first_structure_labels} | "
                    f"first_boundary_attr_labels={_first_boundary_labels}",
                )

        self.geometry_transform, self.transform = self._get_transforms()

    @staticmethod
    def _image_path_to_json_path(img_path: str) -> str:
        stem, _ = os.path.splitext(img_path)
        return stem + ".json"

    def _get_transforms(self):
        """
        No PadIfNeeded is used to avoid black borders.
        If an image is smaller than crop_size, it is upscaled in __getitem__.
        """
        if self.mode == "train":
            geometry = A.Compose(
                [
                    A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
            appearance_and_resize = A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=0.5,
                    ),
                    A.Resize(
                        height=self.image_size,
                        width=self.image_size,
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    ToTensorV2(),
                ]
            )
            return geometry, appearance_and_resize

        geometry = A.Compose(
            [A.CenterCrop(width=self.crop_size, height=self.crop_size, p=1.0)]
        )
        appearance_and_resize = A.Compose(
            [
                A.Resize(
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )
        return geometry, appearance_and_resize

    def _decode_mask(self, json_path: str) -> np.ndarray:
        """
        Decode instance mask from polygon or COCO-RLE annotations.

        Compatible with the usual PanNuke-style json used in your project.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        if isinstance(data_json, list):
            data_json = data_json[0] if len(data_json) > 0 and isinstance(data_json[0], dict) else {}

        if "image" in data_json and isinstance(data_json["image"], dict):
            h = int(data_json["image"].get("height", data_json.get("height", 256)))
            w = int(data_json["image"].get("width", data_json.get("width", 256)))
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
                # seg can be [poly] or flat poly.
                polygons = [seg] if all(isinstance(x, (int, float)) for x in seg) else seg

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

    def _sample_dynamic_task(self, mask: np.ndarray, regions=None):
        """
        Reserved for future class-specific or region-specific prompt tasks.
        Current version keeps generic nuclei segmentation.
        """
        active_mask = mask.copy()
        task_type = "generic"
        text_suffix = ""
        return active_mask, task_type, text_suffix

    def _resolve_organ(self, json_data: Dict[str, Any]) -> Tuple[int, str]:
        if "organ_idx" in json_data:
            organ_id = int(json_data["organ_idx"])
            organ_name = ID_TO_ORGAN.get(organ_id, json_data.get("organ_id", "Generic"))
        else:
            organ_name = json_data.get("organ_id", "Generic")
            organ_id = self.organ_map.get(organ_name, 20)

        if organ_name not in self.organ_map:
            organ_name = "Generic"
            organ_id = 20

        return int(organ_id), str(organ_name)

    def _use_sample_v2_attributes(self) -> bool:
        if not self.is_v2:
            return False

        if self.mode in {"train", "oracle", "debug"}:
            return True

        return bool(self.allow_eval_sample_attributes)

    def _get_v2_sample_payload(
        self,
        json_data: Dict[str, Any],
        organ_name: str,
    ) -> Tuple[List[int], Dict[str, str], Tuple[str, str, str], str]:
        labels = _to_int_list(
            json_data.get("attr_labels", None),
            default=_default_attr_labels(),
        )

        visuals = _normalise_visual_stats_from_entry(json_data)

        text_prompt, attribute_text, morphology_text = _normalise_prompts_from_entry(
            entry=json_data,
            organ_name=organ_name,
            visuals=visuals,
        )

        return labels, visuals, (text_prompt, attribute_text, morphology_text), "medical_knowledge_v2_sample"

    def _get_organ_prior_payload(
        self,
        organ_name: str,
    ) -> Tuple[List[int], Dict[str, str], Tuple[str, str, str], str]:
        labels, visuals, prompts = _build_from_organ_prior(
            meta=self.meta,
            organ_name=organ_name,
        )

        return labels, visuals, prompts, "medical_knowledge_v2_organ_prior"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        json_data = item["data"]

        # 1. Image and instance mask
        image = cv2.imread(item["img_path"])
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self._decode_mask(item["json_path"])

        # Safety upscale before crop.
        h, w = image.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            target_h = max(h, self.crop_size)
            target_w = max(w, self.crop_size)

            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 2. Augmentation
        geometric = self.geometry_transform(image=image, mask=mask)
        local_aug_mask_inst = np.asarray(geometric["mask"]).astype(np.int32)
        augmented = self.transform(image=geometric["image"], mask=local_aug_mask_inst)

        # ToTensorV2 keeps uint8 range; SAM preprocess expects 0-255 scale.
        img_tensor = augmented["image"].float()

        aug_mask_inst = augmented["mask"].numpy().astype(np.int32)
        aug_mask = (aug_mask_inst > 0).astype(np.uint8)

        # L1-A labels are recomputed after synchronized crop/flip/rotation and
        # before ColorJitter/model resize. No pre-augmentation label is reused.
        local_region_targets = None
        if self.enable_local_region_text_alignment:
            local_region_targets = compute_local_region_targets(
                local_aug_mask_inst,
                self.local_region_thresholds,
                window_size=self.local_region_window_size,
            )
            if self.local_region_audit and self._local_region_audit_count < 3:
                sample_path = item["img_path"]
                local_coordinates = local_region_targets["coordinates"]
                local_labels = local_region_targets["labels"]
                local_valid = local_region_targets["valid"]
                complete_instance_count = local_region_targets["complete_instance_count"]
                local_region_count = local_region_targets["region_count"]
                _dataloader_print(
                    "\n[L1A_LOCAL_REGION_DATA_AUDIT]\n"
                    f"  sample={os.path.basename(sample_path)}\n"
                    f"  region_coordinates_before={list(l1a_region_coordinates())}\n"
                    f"  region_coordinates_after={list(local_coordinates)}\n"
                    f"  recomputed_label_codes={local_labels.tolist()}\n"
                    f"  valid_attribute_mask={local_valid.tolist()}\n"
                    f"  complete_instance_count={complete_instance_count.tolist()}\n"
                    f"  region_count={local_region_count}\n"
                    f"  attributes={list(L1A_LOCAL_ATTRIBUTE_NAMES)}",
                    flush=True,
                )
                self._local_region_audit_count += 1

        # 3. Crop-level dynamic attributes.
        # Only used by dynamic_gt mode. Not used by normal organ_static.
        area_scale = 1.0
        task_type = "generic"
        text_suffix = ""

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        crop_analysis = analyze_physical_attributes(
            image=img_np,
            mask=aug_mask,
            config=self.attr_config,
            area_scale=area_scale,
        )

        # 4. Organ metadata
        organ_id, organ_name = self._resolve_organ(json_data)

        organ_dropout_applied = False
        if self.mode == "train" and self.organ_dropout_prob > 0.0:
            if random.random() < self.organ_dropout_prob:
                organ_id = 20
                organ_name = "Generic"
                organ_dropout_applied = True

        # 5. Choose prompt attributes and attr label source.
        if self.prompt_mode == "base":
            attr_labels_np = _to_int_list(json_data.get("attr_labels", None), default=_default_attr_labels())
            prompt_visuals = _normalise_visual_stats_from_entry(json_data)
            text_prompt, attribute_text, morphology_text = _build_base_prompts()
            attr_source = "base_prompt_no_text_attribute"

        elif self.prompt_mode == "dynamic_gt":
            # Explicit oracle/debug mode. Normal val/test has been guarded in __init__.
            prompt_visuals = crop_analysis["visuals"]
            attr_labels_np = crop_analysis["labels"]
            text_prompt, attribute_text, morphology_text = build_pathology_prompts(
                base_prompt=STRICT_BASE_PROMPT,
                organ_name=organ_name,
                visuals=prompt_visuals,
                task_type=task_type,
                text_suffix=text_suffix,
                prompt_mode=self.prompt_mode,
            )
            attr_source = "crop_dynamic_gt"

        else:
            # Main formal mode: organ_static / dynamic_pred.
            if organ_dropout_applied:
                labels, prompt_visuals, prompts, attr_source = self._get_organ_prior_payload(
                    organ_name="Generic",
                )
                attr_labels_np = labels
                text_prompt, attribute_text, morphology_text = prompts
                attr_source = attr_source + "_after_organ_dropout"

            elif self._use_sample_v2_attributes() and _entry_is_v2(json_data):
                labels, prompt_visuals, prompts, attr_source = self._get_v2_sample_payload(
                    json_data=json_data,
                    organ_name=organ_name,
                )
                attr_labels_np = labels
                text_prompt, attribute_text, morphology_text = prompts

            elif self.is_v2:
                labels, prompt_visuals, prompts, attr_source = self._get_organ_prior_payload(
                    organ_name=organ_name,
                )
                attr_labels_np = labels
                text_prompt, attribute_text, morphology_text = prompts

            else:
                # Backward-compatible fallback for old medical_knowledge.json.
                # Do not prefer old visual_stats when v2 is available.
                full_binary_mask = (mask > 0).astype(np.uint8)
                full_analysis = analyze_physical_attributes(
                    image=image,
                    mask=full_binary_mask,
                    config=self.attr_config,
                    area_scale=1.0,
                )
                prompt_visuals = full_analysis["visuals"]
                attr_labels_np = full_analysis["labels"]
                text_prompt, attribute_text, morphology_text = build_pathology_prompts(
                    base_prompt=STRICT_BASE_PROMPT,
                    organ_name=organ_name,
                    visuals=prompt_visuals,
                    task_type=task_type,
                    text_suffix=text_suffix,
                    prompt_mode=self.prompt_mode,
                )
                attr_source = "fallback_full_image_physical"

        # 6. Labels / density / HV
        label_tensor = torch.from_numpy(aug_mask).long().unsqueeze(0)
        label_inst_tensor = torch.from_numpy(aug_mask_inst).long().unsqueeze(0)

        gt_heatmap = generate_adaptive_density(
            aug_mask,
            image_size=(self.image_size, self.image_size),
        )
        gt_heatmap_tensor = torch.from_numpy(gt_heatmap).float().unsqueeze(0)

        gt_hv_map = generate_hv_map(aug_mask_inst)
        gt_hv_map_tensor = torch.from_numpy(gt_hv_map).float()

        # Boundary / uncertainty targets for boundary-aware and uncertainty-weighted losses.
        # After crop_size -> image_size resizing, scale the ring radius so that it remains
        # roughly equivalent to a 2-pixel band on the original crop.
        boundary_radius = max(2, int(round(2.0 * float(self.image_size) / float(self.crop_size))))
        structure_targets = generate_boundary_uncertainty_targets(
            inst_mask=aug_mask_inst,
            boundary_radius=boundary_radius,
            contact_radius=boundary_radius,
        )

        fg_target_tensor = torch.from_numpy(structure_targets["fg_target"]).float().unsqueeze(0)
        bg_target_tensor = torch.from_numpy(structure_targets["bg_target"]).float().unsqueeze(0)
        boundary_target_tensor = torch.from_numpy(structure_targets["boundary_target"]).float().unsqueeze(0)
        uncertain_target_tensor = torch.from_numpy(structure_targets["uncertain_target"]).float().unsqueeze(0)

        # ==================================================================
        # 6b. Dense Boundary Maps & Instance Morphology Attrs (Phase B)
        # ==================================================================
        # 4 dense boundary maps [1, H, W] each
        _dense_maps = generate_dense_boundary_maps(aug_mask_inst)
        dense_boundary_map_tensor = torch.from_numpy(_dense_maps["boundary_map"]).float().unsqueeze(0)
        touching_region_tensor = torch.from_numpy(_dense_maps["touching_region"]).float().unsqueeze(0)
        small_nuclei_map_tensor = torch.from_numpy(_dense_maps["small_nuclei"]).float().unsqueeze(0)
        hv_gradient_map_tensor = torch.from_numpy(_dense_maps["hv_gradient"]).float().unsqueeze(0)
        # Instance morphology attributes
        _inst_morph = compute_instance_morphology_attrs(
            aug_mask_inst,
            min_instance_area=self.min_instance_area,
            max_instances_per_image=self.max_instances_per_image,
        )
        instance_attr_labels_tensor = torch.from_numpy(_inst_morph["instance_attr_labels"]).long()  # [6]
        instance_attr_values_tensor = torch.from_numpy(_inst_morph["instance_attr_values"]).float()  # [6]
        # Per-instance morphology labels [N_i, 6]
        per_instance_attr_labels_tensor = torch.from_numpy(_inst_morph["per_instance_attr_labels"]).long()  # [N_i, 6]
        per_instance_attr_values_tensor = torch.from_numpy(_inst_morph["per_instance_attr_values"]).float()  # [N_i, 6]
        # Per-instance IDs [N_i], same order as per_instance_attr_labels rows
        per_instance_ids_tensor = torch.from_numpy(_inst_morph["per_instance_ids"]).long()  # [N_i]

        attr_labels = torch.tensor(attr_labels_np, dtype=torch.long)

        uses_gt_prompt = prompt_uses_gt_attributes(self.prompt_mode)

        # ==================================================================
        # 7. Structure & Boundary Attribute Labels (GT-derived pathology attrs)
        # ==================================================================
        # Default: no structure/boundary attrs available.
        _has_sb = False
        _structure_attr_labels = [INVALID_ATTR_LABEL] * len(STRUCTURE_ATTR_NAMES)
        _boundary_attr_labels = [INVALID_ATTR_LABEL] * len(BOUNDARY_ATTR_NAMES)
        _structure_attr_values = [0.0] * len(STRUCTURE_ATTR_NAMES)
        _boundary_attr_values = [0.0] * len(BOUNDARY_ATTR_NAMES)

        if self._sb_loaded:
            sb_sample_id = item.get("_sb_sample_id")
            if sb_sample_id is not None and sb_sample_id in self.structure_boundary_attr_records:
                sb_rec = self.structure_boundary_attr_records[sb_sample_id]
                _structure_attr_labels = list(sb_rec["structure_attr_labels"])
                _boundary_attr_labels = list(sb_rec["boundary_attr_labels"])
                _structure_attr_values = list(sb_rec["structure_attr_values"])
                _boundary_attr_values = list(sb_rec["boundary_attr_values"])
                _has_sb = True
            # else: keep invalid defaults (missing sample — warning already printed at init)

        # ==================================================================
        # PNURL_AUDIT: Debug print first 3 train samples (Item 1)
        # ==================================================================
        global _PNURL_AUDIT_DATALOADER_COUNTER
        if _PNURL_AUDIT and self.mode == "train" and _PNURL_AUDIT_DATALOADER_COUNTER < 3:
            idx = _PNURL_AUDIT_DATALOADER_COUNTER
            _dataloader_print(f"\n[PNURL_AUDIT_DATALOADER] sample={idx}", flush=True)
            _dataloader_print(f"  image_id={os.path.basename(item['img_path'])}", flush=True)
            _dataloader_print(f"  prompt_mode={self.prompt_mode}", flush=True)
            _dataloader_print(f"  requested_prompt_mode={self.requested_prompt_mode}", flush=True)
            _dataloader_print(f"  organ={organ_name} (id={organ_id})", flush=True)
            _dataloader_print(f"  attr_labels={attr_labels_np}", flush=True)
            _dataloader_print(f"  attr_source={attr_source}", flush=True)
            _dataloader_print(f"  text_prompt={text_prompt}", flush=True)
            _dataloader_print(f"  attribute_text={attribute_text}", flush=True)
            _dataloader_print(f"  morphology_text={morphology_text}", flush=True)
            _PNURL_AUDIT_DATALOADER_COUNTER += 1

        result = {
            "image": img_tensor,

            # Semantic / instance labels
            "label": label_tensor,
            "label_inst": label_inst_tensor,

            # Structure supervision
            "gt_heatmap": gt_heatmap_tensor,
            "gt_hv_map": gt_hv_map_tensor,
            "fg_target": fg_target_tensor,
            "bg_target": bg_target_tensor,
            "boundary_target": boundary_target_tensor,
            "uncertain_target": uncertain_target_tensor,

            # Organ and prompts
            "organ_id": int(organ_id),
            "text_prompt": text_prompt,
            "attribute_text": attribute_text,
            "morphology_text": morphology_text,

            # Attribute labels for PNuRL supervision.
            # v2 train uses sample-level full-image attribute labels.
            # v2 val/test uses train-split organ priors by default to avoid GT-derived prompt leakage.
            "attr_labels": attr_labels,

            # Debug / metadata
            "visual_attributes": prompt_visuals,
            "crop_visual_attributes": crop_analysis["visuals"],
            "metadata_visual_stats": json_data.get("visual_stats", {}),
            "metadata_attr_labels": json_data.get("attr_labels", None),
            "attr_label_source": attr_source,
            "organ_dropout_applied": bool(organ_dropout_applied),
            "name": os.path.basename(item["img_path"]),
            "rel_path": item.get("rel_path", ""),
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type,
            "prompt_mode": self.prompt_mode,
            "requested_prompt_mode": self.requested_prompt_mode,
            "prompt_uses_gt_attributes": bool(uses_gt_prompt),

            # Structure & Boundary Attributes (GT-derived pathology attrs)
            # structure_attr_labels:   [nuclear_density, nuclear_area_fraction, mean_nuclear_size,
            #                           nuclear_size_heterogeneity, spatial_crowding]
            # boundary_attr_labels:   [boundary_density, nuclear_irregularity, nuclear_elongation,
            #                           small_nuclei_ratio]
            # Each label: 0=low, 1=mid, 2=high, -1=invalid
            # structure_attr_values / boundary_attr_values: raw continuous values from GT computation
            # has_structure_boundary_attrs: True if the sample was found in the JSONL
            "structure_attr_labels": torch.tensor(_structure_attr_labels, dtype=torch.long),
            "boundary_attr_labels": torch.tensor(_boundary_attr_labels, dtype=torch.long),
            "structure_attr_values": torch.tensor(_structure_attr_values, dtype=torch.float),
            "boundary_attr_values": torch.tensor(_boundary_attr_values, dtype=torch.float),
            "has_structure_boundary_attrs": _has_sb,

            # Phase B: Dense Boundary Maps (each [1, H, W] float)
            "dense_boundary_map": dense_boundary_map_tensor,
            "dense_touching_region": touching_region_tensor,
            "dense_small_nuclei": small_nuclei_map_tensor,
            "dense_hv_gradient": hv_gradient_map_tensor,
            # Phase B: Instance Morphology Attributes (each [6] long/float)
            "instance_attr_labels": instance_attr_labels_tensor,      # [6] long (0/1/2)
            "instance_attr_values": instance_attr_values_tensor,      # [6] float
            # Phase B: Per-instance morphology labels (variable-length per image)
            "per_instance_attr_labels": per_instance_attr_labels_tensor,  # [N_i, 6] long
            "per_instance_attr_values": per_instance_attr_values_tensor,  # [N_i, 6] float
            # Phase B: Per-instance IDs in same order as attr_labels rows
            "per_instance_ids": per_instance_ids_tensor,  # [N_i] long
        }
        if local_region_targets is not None:
            result["local_region_attr_labels"] = torch.from_numpy(
                local_region_targets["labels"]
            ).long()
            result["local_region_attr_valid"] = torch.from_numpy(
                local_region_targets["valid"]
            ).bool()
            result["local_region_attr_values"] = torch.from_numpy(
                local_region_targets["values"]
            ).float()
            result["local_region_complete_counts"] = torch.from_numpy(
                local_region_targets["complete_instance_count"]
            ).long()
            result["local_region_coordinates"] = torch.tensor(
                local_region_targets["coordinates"], dtype=torch.long
            )
        return result


def stack_dict_batched(batch):
    tensor_dict = {}

    # Keys that contain variable-length per-instance tensors (can't be stacked)
    _per_instance_keys = {"per_instance_attr_labels", "per_instance_attr_values", "per_instance_ids"}

    for key, value in batch[0].items():
        if key in _per_instance_keys:
            # Variable-length per-instance data: keep as list of tensors
            tensor_dict[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, (int, float, str, bool)):
            tensor_dict[key] = [sample[key] for sample in batch]
        else:
            tensor_dict[key] = [sample[key] for sample in batch]

    return tensor_dict
