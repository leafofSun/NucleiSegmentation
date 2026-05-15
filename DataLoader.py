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

# === 可选依赖 ===
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


# 🔥 [21类器官映射表]
ORGAN_TO_ID = {
    # --- PanNuke 19 类 ---
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3,
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7,
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15,
    "Testis": 16, "Thyroid": 17, "Uterus": 18,
    # --- 补充 ---
    "Brain": 19, "Generic": 20
}

ID_TO_ORGAN = {v: k for k, v in ORGAN_TO_ID.items()}


# ==============================================================================
# 1. 动态配置类
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
    def from_metadata(cls, stats):
        if not stats:
            return cls()
        return cls(
            AREA_SMALL=stats.get("th_size_small", 250.0),
            AREA_LARGE=stats.get("th_size_large", 600.0),
            DENSITY_SPARSE=stats.get("th_dens_sparse", 0.05),
            DENSITY_DENSE=stats.get("th_dens_dense", 0.20),
        )


# ==============================================================================
# 2. Prompt 文本构造工具
# ==============================================================================
def format_organ_name(organ_name: str) -> str:
    """
    将 PanNuke 风格器官名转换成更适合 VLM 文本编码器的自然语言。
    例如:
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


def build_pathology_prompts(
    base_prompt: str,
    organ_name: str,
    visuals: dict,
    task_type: str = "generic",
    text_suffix: str = "",
    prompt_mode: str = "dynamic",
):
    """
    根据当前 crop 的物理属性生成多种 prompt。

    返回:
        text_prompt:
            主 prompt，给 TextGuidedPointGenerator / 通用文本分支使用。
        attribute_text:
            属性提示，建议给 PNuRL / CONCH 属性语义分支使用。
            重点描述低频语义：染色、器官、核区域整体属性、密度、排列。
        morphology_text:
            形态提示，建议给高频边界 / morphology prompt 分支使用。
            重点描述高频边界：形状、轮廓、粘连、实例分离。
    """
    if base_prompt is None or str(base_prompt).strip() == "":
        base_prompt = "Cell nuclei"

    organ_text = format_organ_name(organ_name)

    color = visuals.get("color", "deep-purple stained")
    shape = visuals.get("shape", "round")
    arrangement = visuals.get("arrangement", "uniformly arranged")
    size = visuals.get("size", "medium-sized")
    density = visuals.get("density", "moderately distributed")

    # 用于消融：完全退化成原始 base prompt
    if prompt_mode == "base":
        return base_prompt, base_prompt, base_prompt

    if organ_text == "generic":
        tissue_prefix = "H&E-stained histopathology patch"
        organ_phrase = "histopathology tissue"
    else:
        tissue_prefix = f"H&E-stained {organ_text} histopathology patch"
        organ_phrase = f"{organ_text} tissue"

    # 主分割 prompt：尽量短，避免文本过长影响文本编码稳定性
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

    # 属性 prompt：偏低频语义，描述整体病理属性
    attribute_text = (
        f"{tissue_prefix}. "
        f"The cell nuclei are {color}, {shape} in morphology, {size}, "
        f"{density}, and {arrangement}. "
        f"These attributes describe nuclear staining appearance, tissue context, "
        f"size, density, and spatial arrangement."
    )

    # 形态 prompt：偏高频边界，给 morphology / boundary 分支使用
    morphology_text = (
        f"{tissue_prefix}. "
        f"Focus on individual nuclear morphology: {shape} contours, {size} nuclei, "
        f"{arrangement}, and {density}. "
        f"Emphasize sharp nuclear boundaries, touching nuclei separation, "
        f"irregular contours, and instance-level delineation."
    )

    # 用于消融：只使用属性 prompt 或只使用形态 prompt
    if prompt_mode == "attribute_only":
        return attribute_text, attribute_text, attribute_text

    if prompt_mode == "morphology_only":
        return morphology_text, morphology_text, morphology_text

    return text_prompt, attribute_text, morphology_text


# ==============================================================================
# 3. 物理属性分析器
# ==============================================================================
def analyze_physical_attributes(image, mask, config: AttributeConfig, area_scale=1.0):
    results = {
        "visuals": {
            "color": "deep-purple stained",
            "shape": "round",
            "arrangement": "uniform",
            "size": "medium-sized",
            "density": "moderately distributed",
        },
        "labels": [0, 0, 0, 1, 1],
    }

    if mask.sum() == 0:
        return results

    # 🔥 [SPEEDUP] 降采样分析
    analysis_scale = 256.0 / max(mask.shape)
    if analysis_scale < 1.0:
        h, w = mask.shape[:2]
        new_h, new_w = int(h * analysis_scale), int(w * analysis_scale)
        small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        area_scale_factor = (1.0 / analysis_scale) ** 2
    else:
        small_mask = mask
        area_scale_factor = 1.0

    # 使用 OpenCV 加速连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        small_mask.astype(np.uint8),
        connectivity=8
    )

    if num_labels <= 1:
        return results

    # 1. Size
    areas = stats[1:, cv2.CC_STAT_AREA] * area_scale_factor
    mean_area = np.mean(areas)

    th_small = config.AREA_SMALL * area_scale
    th_large = config.AREA_LARGE * area_scale

    if mean_area < th_small:
        size_lbl, size_txt = 0, "small"
    elif mean_area > th_large:
        size_lbl, size_txt = 2, "large, enlarged"
    else:
        size_lbl, size_txt = 1, "medium-sized"

    # 2. Shape
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    aspect_ratios = widths.astype(float) / (heights.astype(float) + 1e-5)
    mean_ar = np.mean(np.abs(1.0 - aspect_ratios))

    if mean_ar < 0.3:
        shape_lbl, shape_txt = 0, "round"
    elif mean_ar < 0.6:
        shape_lbl, shape_txt = 1, "oval"
    else:
        shape_lbl, shape_txt = 2, "elongated"

    # 3. Density
    count = num_labels - 1
    if count < config.DENSITY_SPARSE * 100:
        den_lbl, den_txt = 0, "sparsely distributed"
    elif count > config.DENSITY_DENSE * 100:
        den_lbl, den_txt = 2, "densely packed"
    else:
        den_lbl, den_txt = 1, "moderately distributed"

    # 4. Arrangement
    if count > 5 and SKLEARN_AVAILABLE:
        try:
            pts = centroids[1:]
            tree = KDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn_dists = dists[:, 1]
            dist_cv = np.std(nn_dists) / (np.mean(nn_dists) + 1e-6)

            if dist_cv > config.ARRANGE_CLUMPED:
                arr_lbl, arr_txt = 1, "disordered/clustered"
            else:
                arr_lbl, arr_txt = 0, "uniformly arranged"
        except Exception:
            arr_lbl, arr_txt = 0, "uniformly arranged"
    else:
        arr_lbl, arr_txt = 0, "isolated"

    # 当前版本仍固定为 H&E 常见深紫染色；后续可接入真实颜色统计
    col_lbl, col_txt = 0, "deep-purple stained"

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


def generate_adaptive_density(mask, image_size=(1024, 1024)):
    """
    快速自适应密度图。
    输入:
        mask: 二值核区域图，shape=[H, W]
        image_size: 输出热图尺寸
    返回:
        heatmap: shape=[target_h, target_w], float32, range约为[0,1]
    """
    target_h, target_w = image_size
    scale = 0.25
    small_h, small_w = int(target_h * scale), int(target_w * scale)

    small_mask = cv2.resize(mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(small_mask, connectivity=8)

    heatmap = np.zeros((small_h, small_w), dtype=np.float32)

    if num_labels <= 1:
        return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    points = centroids[1:]

    if len(points) > 200:
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < small_h and 0 <= x < small_w:
                heatmap[y, x] = 1.0

        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3.0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    else:
        if SKLEARN_AVAILABLE:
            tree = KDTree(points)
            dists, _ = tree.query(points, k=min(4, len(points)))
        else:
            dists = None

        for i, pt in enumerate(points):
            x0, y0 = int(pt[0]), int(pt[1])

            if dists is not None and len(dists[i]) > 1:
                sigma = 0.3 * np.mean(dists[i][1:])
            else:
                sigma = 4.0

            sigma = max(1.0, min(sigma, 15.0))

            k_size = int(sigma * 3) * 2 + 1
            kernel = cv2.getGaussianKernel(k_size, sigma)
            kernel = kernel @ kernel.T

            kh, kw = kernel.shape
            y_min, y_max = max(0, y0 - kh // 2), min(small_h, y0 + kh // 2 + 1)
            x_min, x_max = max(0, x0 - kw // 2), min(small_w, x0 + kw // 2 + 1)

            ky_min = kh // 2 - (y0 - y_min)
            ky_max = ky_min + (y_max - y_min)
            kx_min = kw // 2 - (x0 - x_min)
            kx_max = kx_min + (x_max - x_min)

            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                kernel[ky_min:ky_max, kx_min:kx_max]
            )

    return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def generate_hv_map(inst_mask: np.ndarray) -> np.ndarray:
    """
    将实例掩膜转化为 HoVer 风格 HV 距离图。
    返回: [2, H, W] 的 numpy 数组。
          0: V direction
          1: H direction
          背景为 0，核内部约在 [-1, 1]
    """
    if inst_mask.ndim != 2:
        raise ValueError(f"inst_mask must be 2D, got shape={inst_mask.shape}")

    h, w = inst_mask.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    props = regionprops(inst_mask.astype(np.int32))
    for prop in props:
        y_min, x_min, y_max, x_max = prop.bbox
        y_c, x_c = prop.centroid

        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

        y_den = (y_max - y_min) / 2.0 + 1e-8
        x_den = (x_max - x_min) / 2.0 + 1e-8

        y_dist = (y_grid - y_c) / y_den
        x_dist = (x_grid - x_c) / x_den

        y_dist = np.clip(y_dist, -1.0, 1.0).astype(np.float32)
        x_dist = np.clip(x_dist, -1.0, 1.0).astype(np.float32)

        inst_bool = inst_mask[y_min:y_max, x_min:x_max] == prop.label

        hv_map[0, y_min:y_max, x_min:x_max][inst_bool] = y_dist[inst_bool]  # V
        hv_map[1, y_min:y_max, x_min:x_max][inst_bool] = x_dist[inst_bool]  # H

    return hv_map


# ==============================================================================
# 4. 通用数据集类
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
        self.prompt_mode = prompt_mode
        self.organ_map = ORGAN_TO_ID

        # 1. 加载知识库
        print(f"📖 [DataLoader] Loading Knowledge Base: {knowledge_path}")
        with open(knowledge_path, "r") as f:
            self.full_db = json.load(f)

        # 2. 提取元数据
        if "__meta__" in self.full_db:
            meta = self.full_db.pop("__meta__")
            stats = meta.get("stats", {})
            self.attr_config = AttributeConfig.from_metadata(stats)
        else:
            print("⚠️ [DataLoader] Warning: '__meta__' not found. Using default thresholds.")
            self.attr_config = AttributeConfig()

        # 3. 构建样本
        self.samples = []
        skipped = 0

        for rel_path, entry in self.full_db.items():
            if entry.get("split") != mode:
                skipped += 1
                continue

            full_img_path = os.path.join(data_root, rel_path)
            full_json_path = full_img_path.replace(".png", ".json")

            if os.path.exists(full_img_path) and os.path.exists(full_json_path):
                self.samples.append({
                    "img_path": full_img_path,
                    "json_path": full_json_path,
                    "data": entry,
                })

        print(f"✅ [DataLoader] Mode: {mode} | Loaded: {len(self.samples)} | Skipped: {skipped}")

        self.transform = self._get_transforms()

    def _get_transforms(self):
        """
        移除 PadIfNeeded，避免黑边。
        如果原图小于 crop_size，会在 __getitem__ 里先放大。
        """
        if self.mode == "train":
            return A.Compose([
                A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.Resize(
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_LINEAR
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.CenterCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                A.Resize(
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_LINEAR
                ),
                ToTensorV2(),
            ])

    def _decode_mask(self, json_path):
        with open(json_path, "r") as f:
            data_json = json.load(f)

        h, w = data_json.get("height", 256), data_json.get("width", 256)

        # 保留实例 ID，避免所有 nuclei 融合为二值图
        mask = np.zeros((h, w), dtype=np.int32)
        inst_id = 1

        for ann in data_json.get("annotations", []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                if pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], inst_id)
                    inst_id += 1

        return mask

    def _sample_dynamic_task(self, mask, regions):
        """
        预留接口：后续可以做某类 nuclei、某种区域、某种 prompt task 的动态采样。
        当前保持 generic。
        """
        active_mask = mask.copy()
        task_type = "generic"
        text_suffix = ""
        return active_mask, task_type, text_suffix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        # 1. Image & Mask
        image = cv2.imread(item["img_path"])
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._decode_mask(item["json_path"])

        # Safety Upscale:
        # 确保输入图片至少有 crop_size 那么大，避免 RandomCrop 产生黑边或报错。
        h, w = image.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            target_h = max(h, self.crop_size)
            target_w = max(w, self.crop_size)

            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 2. Augment
        augmented = self.transform(image=image, mask=mask)

        img_tensor = augmented["image"].float()

        # albumentations 会保持 mask 的整数语义，这里显式转为 int32 作为实例图。
        aug_mask_inst = augmented["mask"].numpy().astype(np.int32)
        aug_mask = (aug_mask_inst > 0).astype(np.uint8)

        # 3. Physics & Dynamic Task
        area_scale = 1.0
        task_type = "generic"
        text_suffix = ""

        # 当前 img_tensor 为 CHW，转回 HWC 供属性分析使用。
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        analysis = analyze_physical_attributes(
            image=img_np,
            mask=aug_mask,
            config=self.attr_config,
            area_scale=area_scale,
        )
        visuals = analysis["visuals"]

        # 4. Organ ID & Prompt
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

        # 训练时随机丢弃器官信息，让模型不要过度依赖 organ id。
        if self.mode == "train" and random.random() < 0.2:
            organ_id = 20

        # 5. Label / Heatmap / HV
        label_tensor = torch.from_numpy(aug_mask).long().unsqueeze(0)

        gt_heatmap = generate_adaptive_density(
            aug_mask,
            image_size=(self.image_size, self.image_size)
        )

        gt_hv_map = generate_hv_map(aug_mask_inst)

        # 6. Return
        return {
            "image": img_tensor,

            # semantic / instance label
            "label": label_tensor,
            "label_inst": torch.from_numpy(aug_mask_inst).long().unsqueeze(0),

            # structure supervision
            "gt_heatmap": torch.from_numpy(gt_heatmap).float().unsqueeze(0),
            "gt_hv_map": torch.from_numpy(gt_hv_map).float(),

            # organ and prompts
            "organ_id": organ_id,
            "text_prompt": text_prompt,
            "attribute_text": attribute_text,
            "morphology_text": morphology_text,

            # attribute labels for PNuRL
            "attr_labels": torch.tensor(analysis["labels"]).long(),

            # debug / metadata
            "visual_attributes": visuals,
            "name": os.path.basename(item["img_path"]),
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type,
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