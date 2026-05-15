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


# ==============================================================================
# 0. Organ mapping
# ==============================================================================
ORGAN_TO_ID = {
    # PanNuke 19 organs
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3,
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7,
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15,
    "Testis": 16, "Thyroid": 17, "Uterus": 18,

    # Extra
    "Brain": 19, "Generic": 20,
}

ID_TO_ORGAN = {v: k for k, v in ORGAN_TO_ID.items()}

VALID_PROMPT_MODES = {
    "base",
    "generic",
    "organ_static",
    "dynamic",
    "attribute_only",
    "morphology_only",
}


# ==============================================================================
# 1. Dynamic attribute configuration
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
        if not stats:
            return cls()

        return cls(
            AREA_SMALL=stats.get("th_size_small", 250.0),
            AREA_LARGE=stats.get("th_size_large", 600.0),
            DENSITY_SPARSE=stats.get("th_dens_sparse", 0.05),
            DENSITY_DENSE=stats.get("th_dens_dense", 0.20),
            SHAPE_ROUND=stats.get("th_shape_round", 0.6),
            SHAPE_OVAL=stats.get("th_shape_oval", 0.85),
            ARRANGE_CLUMPED=stats.get("th_arrange_clumped", 0.6),
            COLOR_BRIGHT=stats.get("th_color_bright", 200.0),
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
        return "Cell nuclei"
    return str(base_prompt).strip()


def build_pathology_prompts(
    base_prompt: str,
    organ_name: str,
    visuals: Dict[str, str],
    task_type: str = "generic",
    text_suffix: str = "",
    prompt_mode: str = "dynamic",
) -> Tuple[str, str, str]:
    """
    Build three prompts for different branches.

    Returns:
        text_prompt:
            Short prompt for TextGuidedPointGenerator and general text branch.

        attribute_text:
            Low-frequency semantic prompt for PNuRL / CONCH attribute branch.
            It describes organ context, staining, density, size and arrangement.

        morphology_text:
            High-frequency morphology prompt for boundary / morphology branch.
            It describes contour, touching nuclei and instance-level separation.

    prompt_mode:
        base:
            Return base_prompt only. Used for ablation.

        generic:
            Return a non-leaking generic prompt with organ context only.
            This is suitable for training-time quick validation and test-time baseline.

        organ_static:
            Alias of generic. More explicit name.

        dynamic:
            Use crop-level attributes estimated from mask/statistics.
            Suitable for training or oracle/debug ablation.

        attribute_only:
            Use only attribute_text for all text branches.

        morphology_only:
            Use only morphology_text for all text branches.
    """
    prompt_mode = str(prompt_mode).lower().strip()
    if prompt_mode not in VALID_PROMPT_MODES:
        prompt_mode = "dynamic"

    base_prompt = _safe_base_prompt(base_prompt)
    organ_text = format_organ_name(organ_name)

    if organ_text == "generic":
        tissue_prefix = "H&E-stained histopathology patch"
        organ_phrase = "histopathology tissue"
    else:
        tissue_prefix = f"H&E-stained {organ_text} histopathology patch"
        organ_phrase = f"{organ_text} tissue"

    # 1. Strict base ablation: no organ, no attributes.
    if prompt_mode == "base":
        return base_prompt, base_prompt, base_prompt

    # 2. Fair non-leaking text mode for validation/test.
    if prompt_mode in {"generic", "organ_static"}:
        text_prompt = f"{base_prompt} in {organ_phrase}."
        attribute_text = (
            f"{tissue_prefix}. "
            f"The image contains cell nuclei in {organ_phrase}. "
            f"This prompt provides organ context without using crop-level mask-derived attributes."
        )
        morphology_text = (
            f"{tissue_prefix}. "
            f"Focus on nuclear boundaries, touching nuclei, and instance-level delineation."
        )
        return text_prompt, attribute_text, morphology_text

    color = visuals.get("color", "deep-purple stained")
    shape = visuals.get("shape", "round")
    arrangement = visuals.get("arrangement", "uniformly arranged")
    size = visuals.get("size", "medium-sized")
    density = visuals.get("density", "moderately distributed")

    # 3. Dynamic prompt mode.
    if task_type != "generic" and text_suffix:
        text_prompt = (
            f"{text_suffix} cell nuclei in {organ_phrase}; "
            f"nuclei are {density} and {arrangement}."
        )
    else:
        text_prompt = (
            f"{base_prompt} in {organ_phrase}; "
            f"nuclei are {density} and {arrangement}."
        )

    attribute_text = (
        f"{tissue_prefix}. "
        f"The cell nuclei are {color}, {shape} in morphology, {size}, "
        f"{density}, and {arrangement}. "
        f"These attributes describe nuclear staining appearance, tissue context, "
        f"size, density, and spatial arrangement."
    )

    morphology_text = (
        f"{tissue_prefix}. "
        f"Focus on individual nuclear morphology: {shape} contours, {size} nuclei, "
        f"{arrangement}, and {density}. "
        f"Emphasize sharp nuclear boundaries, touching nuclei separation, "
        f"irregular contours, and instance-level delineation."
    )

    # 4. Ablation modes.
    if prompt_mode == "attribute_only":
        return attribute_text, attribute_text, attribute_text

    if prompt_mode == "morphology_only":
        return morphology_text, morphology_text, morphology_text

    return text_prompt, attribute_text, morphology_text


# ==============================================================================
# 3. Physical attribute analyzer
# ==============================================================================
def _estimate_color_from_nuclei(image: np.ndarray, mask: np.ndarray, config: AttributeConfig) -> Tuple[int, str]:
    """
    Estimate coarse nuclear staining intensity from masked pixels.

    Current PNuRL color head has 2 classes:
        0: deep-purple stained
        1: light-purple stained

    This is intentionally coarse and stable.
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
    Estimate physical attributes from current crop.

    labels order:
        [Color, Shape, Arrange, Size, Density]

    label spaces:
        Color:   0 deep-purple, 1 light-purple
        Shape:   0 round, 1 oval, 2 elongated
        Arrange: 0 uniform/isolated, 1 disordered/clustered
        Size:    0 small, 1 medium, 2 large
        Density: 0 sparse, 1 moderate, 2 dense
    """
    default_results = {
        "visuals": {
            "color": "deep-purple stained",
            "shape": "round",
            "arrangement": "uniformly arranged",
            "size": "medium-sized",
            "density": "moderately distributed",
        },
        "labels": [0, 0, 0, 1, 1],
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
        default_results["visuals"]["color"] = col_txt
        default_results["labels"][0] = col_lbl
        return default_results

    # 1. Size
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) * area_scale_factor
    mean_area = float(np.mean(areas)) if len(areas) > 0 else 0.0

    th_small = config.AREA_SMALL * area_scale
    th_large = config.AREA_LARGE * area_scale

    if mean_area < th_small:
        size_lbl, size_txt = 0, "small"
    elif mean_area > th_large:
        size_lbl, size_txt = 2, "large, enlarged"
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
        den_lbl, den_txt = 2, "densely packed"
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
        arr_lbl, arr_txt = 0, "isolated"

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
# 4. Dataset
# ==============================================================================
class UniversalDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        knowledge_path,
        image_size=1024,
        crop_size=256,
        mode="train",
        prompt_mode="dynamic",
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        self.prompt_mode = str(prompt_mode).lower().strip()
        self.organ_map = ORGAN_TO_ID

        if self.prompt_mode not in VALID_PROMPT_MODES:
            print(f"⚠️ [DataLoader] Unknown prompt_mode='{prompt_mode}', fallback to 'dynamic'.")
            self.prompt_mode = "dynamic"

        print(f"📖 [DataLoader] Loading Knowledge Base: {knowledge_path}")
        with open(knowledge_path, "r") as f:
            full_db = json.load(f)

        if "__meta__" in full_db:
            meta = full_db["__meta__"]
            stats = meta.get("stats", {})
            self.attr_config = AttributeConfig.from_metadata(stats)
        else:
            print("⚠️ [DataLoader] Warning: '__meta__' not found. Using default thresholds.")
            self.attr_config = AttributeConfig()

        self.full_db = {k: v for k, v in full_db.items() if k != "__meta__"}

        self.samples = []
        skipped = 0

        for rel_path, entry in self.full_db.items():
            if entry.get("split") != mode:
                skipped += 1
                continue

            if os.path.isabs(rel_path):
                full_img_path = rel_path
            else:
                full_img_path = os.path.join(data_root, rel_path)

            full_json_path = full_img_path.replace(".png", ".json")

            if os.path.exists(full_img_path) and os.path.exists(full_json_path):
                self.samples.append(
                    {
                        "img_path": full_img_path,
                        "json_path": full_json_path,
                        "data": entry,
                    }
                )

        print(
            f"✅ [DataLoader] Mode: {mode} | Prompt: {self.prompt_mode} | "
            f"Loaded: {len(self.samples)} | Skipped: {skipped}"
        )

        self.transform = self._get_transforms()

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
        with open(json_path, "r") as f:
            data_json = json.load(f)

        h, w = data_json.get("height", 256), data_json.get("width", 256)

        mask = np.zeros((h, w), dtype=np.int32)
        inst_id = 1

        for ann in data_json.get("annotations", []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly, dtype=np.float32).reshape((-1, 2))
                if pts.shape[0] >= 3:
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts], inst_id)
                    inst_id += 1

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

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

        # 3. Physical attributes
        area_scale = 1.0
        task_type = "generic"
        text_suffix = ""

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        analysis = analyze_physical_attributes(
            image=img_np,
            mask=aug_mask,
            config=self.attr_config,
            area_scale=area_scale,
        )
        visuals = analysis["visuals"]

        # 4. Organ and prompts
        json_data = item["data"]

        if "organ_idx" in json_data:
            organ_id = int(json_data["organ_idx"])
            organ_name = ID_TO_ORGAN.get(organ_id, json_data.get("organ_id", "Generic"))
        else:
            organ_name = json_data.get("organ_id", "Generic")
            organ_id = self.organ_map.get(organ_name, 20)

        base_prompt = json_data.get("text_prompt", "Cell nuclei")

        text_prompt, attribute_text, morphology_text = build_pathology_prompts(
            base_prompt=base_prompt,
            organ_name=organ_name,
            visuals=visuals,
            task_type=task_type,
            text_suffix=text_suffix,
            prompt_mode=self.prompt_mode,
        )

        # Training-time organ dropout.
        if self.mode == "train" and random.random() < 0.2:
            organ_id = 20

        # 5. Labels / density / HV
        label_tensor = torch.from_numpy(aug_mask).long().unsqueeze(0)
        label_inst_tensor = torch.from_numpy(aug_mask_inst).long().unsqueeze(0)

        gt_heatmap = generate_adaptive_density(
            aug_mask,
            image_size=(self.image_size, self.image_size),
        )
        gt_heatmap_tensor = torch.from_numpy(gt_heatmap).float().unsqueeze(0)

        gt_hv_map = generate_hv_map(aug_mask_inst)
        gt_hv_map_tensor = torch.from_numpy(gt_hv_map).float()

        attr_labels = torch.tensor(analysis["labels"], dtype=torch.long)

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

            # Attribute labels for PNuRL
            "attr_labels": attr_labels,

            # Debug / metadata
            "visual_attributes": visuals,
            "name": os.path.basename(item["img_path"]),
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type,
            "prompt_mode": self.prompt_mode,
        }


def stack_dict_batched(batch):
    tensor_dict = {}

    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, (int, float, str)):
            tensor_dict[key] = [sample[key] for sample in batch]
        else:
            tensor_dict[key] = [sample[key] for sample in batch]

    return tensor_dict