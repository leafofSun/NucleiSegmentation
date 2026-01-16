import os
import cv2
import json
import torch
import numpy as np
import random
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
from dataclasses import dataclass

# === 可选依赖 ===
try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 🔥 [21类器官映射表]
ORGAN_TO_ID = {
    # --- PanNuke 19 类 ---
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, 
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7, 
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11, 
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15, 
    "Testis": 16, "Thyroid": 17, "Uterus": 18,
    # --- 补充 ---
    "Brain": 19, "Generic": 20, "MoNuSeg": 20
}

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
        if not stats: return cls()
        print(f"📊 [Config] Initializing thresholds from Dataset Statistics...")
        return cls(
            AREA_SMALL=stats.get('th_size_small', 250.0),
            AREA_LARGE=stats.get('th_size_large', 600.0),
            DENSITY_SPARSE=stats.get('th_dens_sparse', 0.05),
            DENSITY_DENSE=stats.get('th_dens_dense', 0.20)
        )

# ==============================================================================
# 2. 物理属性分析器
# ==============================================================================
def analyze_physical_attributes(image, mask, config: AttributeConfig, area_scale=1.0):
    results = {
        "visuals": {"color": "deep-purple stained", "shape": "round", "arrangement": "uniform", "size": "medium", "density": "moderate"},
        "labels": [0, 0, 0, 1, 1] 
    }
    if mask.sum() == 0: return results

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if not regions: return results

    # 1. Size
    th_small = config.AREA_SMALL * area_scale
    th_large = config.AREA_LARGE * area_scale
    areas = np.array([r.area for r in regions])
    mean_area = np.mean(areas)
    if mean_area < th_small: size_lbl, size_txt = 0, "small"
    elif mean_area > th_large: size_lbl, size_txt = 2, "large, enlarged"
    else: size_lbl, size_txt = 1, "medium-sized"

    # 2. Shape
    eccs = np.array([r.eccentricity for r in regions])
    mean_ecc = np.mean(eccs)
    if mean_ecc < config.SHAPE_ROUND: shape_lbl, shape_txt = 0, "round"
    elif mean_ecc < config.SHAPE_OVAL: shape_lbl, shape_txt = 1, "oval"
    else: shape_lbl, shape_txt = 2, "elongated"

    # 3. Density
    count = len(regions)
    if count < config.DENSITY_SPARSE: den_lbl, den_txt = 0, "sparsely distributed"
    elif count > config.DENSITY_DENSE: den_lbl, den_txt = 2, "densely packed"
    else: den_lbl, den_txt = 1, "moderately distributed"

    # 4. Arrangement
    centroids = np.array([r.centroid for r in regions])
    if len(centroids) > 5 and SKLEARN_AVAILABLE:
        try:
            tree = KDTree(centroids)
            dists, _ = tree.query(centroids, k=2)
            nn_dists = dists[:, 1]
            dist_cv = np.std(nn_dists) / (np.mean(nn_dists) + 1e-6)
            if dist_cv > config.ARRANGE_CLUMPED: arr_lbl, arr_txt = 1, "disordered/clustered"
            else: arr_lbl, arr_txt = 0, "uniformly arranged"
        except: arr_lbl, arr_txt = 0, "uniformly arranged"
    else: arr_lbl, arr_txt = 0, "isolated"

    # 5. Color
    col_lbl, col_txt = 0, "deep-purple stained"
    if image is not None:
        masked_pixels = image[mask > 0]
        if masked_pixels.size > 0:
            if np.mean(masked_pixels) > config.COLOR_BRIGHT: col_lbl, col_txt = 1, "pink/light stained"

    return {
        "visuals": {"color": col_txt, "shape": shape_txt, "arrangement": arr_txt, "size": size_txt, "density": den_txt},
        "labels": [col_lbl, shape_lbl, arr_lbl, size_lbl, den_lbl]
    }

def generate_elliptical_heatmap(mask, image_size=(1024, 1024), sigma_scale=0.25):
    """生成椭圆高斯热力图"""
    heatmap = np.zeros(image_size, dtype=np.float32)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    for region in regions:
        if region.area < 5: continue
        y0, x0 = region.centroid
        major, minor = region.axis_major_length, region.axis_minor_length # 🔥 [修复] 使用新版属性名，避免 Warning
        theta = -region.orientation
        
        sigma_x = max(1.0, major * sigma_scale)
        sigma_y = max(1.0, minor * sigma_scale)
        
        # 🔥 [修复关键] 提前计算 a, b, c，防止 NameError
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        a = (cos_t**2)/(2*sigma_x**2) + (sin_t**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (sin_t**2)/(2*sigma_x**2) + (cos_t**2)/(2*sigma_y**2)

        # 优化：只在 Bounding Box 内计算高斯
        bb_size = int(max(major, minor) * 1.5)
        y_min, y_max = max(0, int(y0 - bb_size)), min(image_size[0], int(y0 + bb_size + 1))
        x_min, x_max = max(0, int(x0 - bb_size)), min(image_size[1], int(x0 + bb_size + 1))
        
        if x_max <= x_min or y_max <= y_min: continue

        xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        dx, dy = xx - x0, yy - y0
        
        gaussian = np.exp(-(a*dx**2 + 2*b*dx*dy + c*dy**2))
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)
        
    return heatmap

# ==============================================================================
# 3. 通用数据集类
# ==============================================================================
class UniversalDataset(data.Dataset):
    def __init__(self, 
                 data_root, 
                 knowledge_path, 
                 image_size=1024, 
                 crop_size=256,   
                 mode='train',
                 prompt_mode='dynamic'):
        
        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        self.prompt_mode = prompt_mode
        self.organ_map = ORGAN_TO_ID
        
        # 1. 加载知识库
        print(f"📖 [DataLoader] Loading Knowledge Base: {knowledge_path}")
        with open(knowledge_path, 'r') as f:
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
            if entry.get('split') != mode:
                skipped += 1
                continue
            
            # 🔥 修复路径拼接逻辑：知识库里的 Key 已经是相对路径
            full_img_path = os.path.join(data_root, rel_path)
            full_json_path = full_img_path.replace(".png", ".json")
            
            if os.path.exists(full_img_path) and os.path.exists(full_json_path):
                self.samples.append({
                    "img_path": full_img_path,
                    "json_path": full_json_path,
                    "data": entry 
                })
        
        print(f"✅ [DataLoader] Mode: {mode} | Loaded: {len(self.samples)} | Skipped: {skipped}")
        
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                A.PadIfNeeded(min_height=self.crop_size, min_width=self.crop_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Resize(height=self.image_size, width=self.image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.PadIfNeeded(min_height=self.crop_size, min_width=self.crop_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                ToTensorV2(),
            ])

    def _decode_mask(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        h, w = data.get('height', 256), data.get('width', 256)
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in data.get('annotations', []):
            for poly in ann.get('segmentation', []):
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)
        return mask

    def _sample_dynamic_task(self, mask, regions):
        active_mask = mask.copy()
        task_type = "generic"
        text_suffix = ""

        if self.mode != 'train' or self.prompt_mode != 'dynamic' or len(regions) < 5:
            return active_mask, task_type, text_suffix

        areas = np.array([r.area for r in regions])
        min_a, max_a = np.min(areas), np.max(areas)
        
        if max_a < min_a * 2.0: return active_mask, task_type, text_suffix

        rand_p = random.random()
        if rand_p < 0.25:
            task_type = "large"
            text_suffix = "large, pleomorphic"
            th_high = np.percentile(areas, 67)
            temp_mask = np.zeros_like(mask)
            for r in regions:
                if r.area >= th_high: 
                    y, x = int(r.centroid[0]), int(r.centroid[1])
                    if mask[y, x]: 
                        pass 
            pass 

        return active_mask, task_type, text_suffix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        
        # 1. Images & Masks
        image = cv2.imread(item['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._decode_mask(item['json_path'])
        
        # 2. Augment
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image'].float()
        aug_mask = augmented['mask'].numpy().astype(np.uint8)
        
        # 3. Physics & Dynamic Task
        area_scale = (self.image_size / self.crop_size) ** 2 if self.mode == 'train' else 1.0
        labeled_mask = label(aug_mask)
        regions = regionprops(labeled_mask)
        
        _, task_type, text_suffix = self._sample_dynamic_task(aug_mask, regions)
        img_np = (img_tensor.permute(1, 2, 0).numpy()).astype(np.uint8)
        analysis = analyze_physical_attributes(img_np, aug_mask, self.attr_config, area_scale)
        visuals = analysis['visuals']
        
        # 4. Prompt & Organ ID (安全获取)
        json_data = item['data']
        if 'organ_idx' in json_data:
            organ_id = json_data['organ_idx']
        else:
            organ_name = json_data.get('organ_id', 'Generic')
            organ_id = self.organ_map.get(organ_name, 20) 

        base_prompt = json_data.get('text_prompt', "Cell nuclei")
        
        if task_type != "generic":
            text_prompt = f"{text_suffix} cell nuclei ({visuals['density']}, {visuals['arrangement']})"
        else:
            text_prompt = base_prompt

        # 5. Returns
        label_tensor = torch.from_numpy(aug_mask).long().unsqueeze(0)
        gt_heatmap = generate_elliptical_heatmap(aug_mask, image_size=(self.image_size, self.image_size))
        
        if self.mode == 'train' and random.random() < 0.2:
            organ_id = 20

        return {
            "image": img_tensor,
            "label": label_tensor,
            "gt_heatmap": torch.from_numpy(gt_heatmap).float().unsqueeze(0),
            "organ_id": organ_id,
            "text_prompt": text_prompt,
            "attr_labels": torch.tensor(analysis['labels']).long(),
            "name": os.path.basename(item['img_path']),
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type
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