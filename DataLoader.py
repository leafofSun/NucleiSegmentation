import os
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
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = str(mode).lower().strip()
        self.raw_mode = mode

        canonical_prompt_mode, raw_prompt_mode = normalize_prompt_mode(prompt_mode, default="organ_static")
        self.requested_prompt_mode = raw_prompt_mode
        self.prompt_mode = canonical_prompt_mode

        if raw_prompt_mode != canonical_prompt_mode:
            print(
                f"⚠️ [DataLoader] prompt_mode='{prompt_mode}' is deprecated or unknown; "
                f"using canonical prompt_mode='{canonical_prompt_mode}'."
            )

        # Hard guard against GT prompt leakage during normal validation/test.
        if self.prompt_mode == "dynamic_gt" and self.mode not in {"train", "oracle", "debug"}:
            print(
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

        print(f"📖 [DataLoader] Loading Knowledge Base: {knowledge_path}")
        with open(knowledge_path, "r", encoding="utf-8") as f:
            full_db = json.load(f)

        self.meta = full_db.get("__meta__", {})
        self.is_v2 = str(self.meta.get("version", "")).lower().startswith("promptnu_freqpath_v2")

        # Attribute config is only for dynamic_gt / fallback.
        if "__meta__" in full_db:
            stats = self.meta.get("stats", None)
            if stats is None:
                stats = self.meta.get("train_thresholds", {})
            self.attr_config = AttributeConfig.from_metadata(stats)
        else:
            print("⚠️ [DataLoader] Warning: '__meta__' not found. Using default thresholds.")
            self.attr_config = AttributeConfig()

        self.full_db = {k: v for k, v in full_db.items() if k != "__meta__"}

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

        print(
            f"✅ [DataLoader] Mode: {self.raw_mode} | Prompt: {self.prompt_mode} "
            f"(requested: {self.requested_prompt_mode}) | "
            f"V2={self.is_v2} | "
            f"OrganDropout={self.organ_dropout_prob:.3f} | "
            f"AllowEvalSampleAttrs={self.allow_eval_sample_attributes} | "
            f"Loaded: {len(self.samples)} | Skipped: {skipped}"
        )

        self.transform = self._get_transforms()

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
            return A.Compose(
                [
                    A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
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

        return A.Compose(
            [
                A.CenterCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                A.Resize(
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )

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
        augmented = self.transform(image=image, mask=mask)

        # ToTensorV2 keeps uint8 range; SAM preprocess expects 0-255 scale.
        img_tensor = augmented["image"].float()

        aug_mask_inst = augmented["mask"].numpy().astype(np.int32)
        aug_mask = (aug_mask_inst > 0).astype(np.uint8)

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

        attr_labels = torch.tensor(attr_labels_np, dtype=torch.long)

        uses_gt_prompt = prompt_uses_gt_attributes(self.prompt_mode)

        return {
            "image": img_tensor,

            # Semantic / instance labels
            "label": label_tensor,
            "label_inst": label_inst_tensor,

            # Structure supervision
            "gt_heatmap": gt_heatmap_tensor,
            "gt_hv_map": gt_hv_map_tensor,

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
        }


def stack_dict_batched(batch):
    tensor_dict = {}

    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, (int, float, str, bool)):
            tensor_dict[key] = [sample[key] for sample in batch]
        else:
            tensor_dict[key] = [sample[key] for sample in batch]

    return tensor_dict