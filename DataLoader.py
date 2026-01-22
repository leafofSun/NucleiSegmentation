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
        return cls(
            AREA_SMALL=stats.get('th_size_small', 250.0),
            AREA_LARGE=stats.get('th_size_large', 600.0),
            DENSITY_SPARSE=stats.get('th_dens_sparse', 0.05),
            DENSITY_DENSE=stats.get('th_dens_dense', 0.20)
        )

# ==============================================================================
# 2. 物理属性分析器 (🔥 极速优化版)
# ==============================================================================
def analyze_physical_attributes(image, mask, config: AttributeConfig, area_scale=1.0):
    results = {
        "visuals": {"color": "deep-purple stained", "shape": "round", "arrangement": "uniform", "size": "medium", "density": "moderate"},
        "labels": [0, 0, 0, 1, 1] 
    }
    if mask.sum() == 0: return results

    # 🔥 [SPEEDUP] 降采样分析！
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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(small_mask, connectivity=8)
    
    if num_labels <= 1: return results 

    # 1. Size
    areas = stats[1:, cv2.CC_STAT_AREA] * area_scale_factor
    mean_area = np.mean(areas)
    
    th_small = config.AREA_SMALL * area_scale
    th_large = config.AREA_LARGE * area_scale
    
    if mean_area < th_small: size_lbl, size_txt = 0, "small"
    elif mean_area > th_large: size_lbl, size_txt = 2, "large, enlarged"
    else: size_lbl, size_txt = 1, "medium-sized"

    # 2. Shape
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    aspect_ratios = w.astype(float) / (h.astype(float) + 1e-5)
    mean_ar = np.mean(np.abs(1.0 - aspect_ratios))
    if mean_ar < 0.3: shape_lbl, shape_txt = 0, "round"
    elif mean_ar < 0.6: shape_lbl, shape_txt = 1, "oval"
    else: shape_lbl, shape_txt = 2, "elongated"

    # 3. Density
    count = num_labels - 1
    if count < config.DENSITY_SPARSE * 100: den_lbl, den_txt = 0, "sparsely distributed"
    elif count > config.DENSITY_DENSE * 100: den_lbl, den_txt = 2, "densely packed"
    else: den_lbl, den_txt = 1, "moderately distributed"

    # 4. Arrangement
    if count > 5 and SKLEARN_AVAILABLE:
        try:
            pts = centroids[1:]
            tree = KDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn_dists = dists[:, 1]
            dist_cv = np.std(nn_dists) / (np.mean(nn_dists) + 1e-6)
            if dist_cv > config.ARRANGE_CLUMPED: arr_lbl, arr_txt = 1, "disordered/clustered"
            else: arr_lbl, arr_txt = 0, "uniformly arranged"
        except: arr_lbl, arr_txt = 0, "uniformly arranged"
    else: arr_lbl, arr_txt = 0, "isolated"

    col_lbl, col_txt = 0, "deep-purple stained"

    return {
        "visuals": {"color": col_txt, "shape": shape_txt, "arrangement": arr_txt, "size": size_txt, "density": den_txt},
        "labels": [col_lbl, shape_lbl, arr_lbl, size_lbl, den_lbl]
    }

def generate_adaptive_density(mask, image_size=(1024, 1024)):
    """
    🔥 [SPEEDUP] 快速自适应密度图
    """
    target_h, target_w = image_size
    scale = 0.25 
    small_h, small_w = int(target_h * scale), int(target_w * scale)
    
    small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
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
        if heatmap.max() > 0: heatmap /= heatmap.max()
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
            y_min, y_max = max(0, y0 - kh//2), min(small_h, y0 + kh//2 + 1)
            x_min, x_max = max(0, x0 - kw//2), min(small_w, x0 + kw//2 + 1)
            ky_min = kh//2 - (y0 - y_min)
            ky_max = ky_min + (y_max - y_min)
            kx_min = kw//2 - (x0 - x_min)
            kx_max = kx_min + (x_max - x_min)
            
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max], 
                kernel[ky_min:ky_max, kx_min:kx_max]
            )

    return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# ==============================================================================
# 3. 通用数据集类 (🔥 Fix Black Borders)
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
        """
        🔥 [SOTA FIX]: 移除了 PadIfNeeded，保证没有黑边。
        如果图片小于 crop_size，会在 __getitem__ 里先放大，再进这里。
        """
        if self.mode == 'train':
            return A.Compose([
                # 移除 PadIfNeeded！
                # 随机裁剪 (对于大图有用)
                A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                # 数据增强
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2), # 如果追求速度可关
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                # 调整到模型输入尺寸
                A.Resize(height=self.image_size, width=self.image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                # 验证集：CenterCrop -> Resize
                A.CenterCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                A.Resize(height=self.image_size, width=self.image_size, interpolation=cv2.INTER_LINEAR),
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
        return active_mask, task_type, text_suffix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        
        # 1. Images & Masks
        image = cv2.imread(item['img_path'])
        if image is None: 
             image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._decode_mask(item['json_path'])
        
        # 🔥🔥🔥 [关键修复] Safety Upscale: 确保输入图片至少有 crop_size 那么大
        # 如果原图是 256，crop_size 是 512，这里直接把原图拉伸到 512，
        # 这样后面的 RandomCrop 就会取满全图，绝对没有黑边！
        h, w = image.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            target_h = max(h, self.crop_size)
            target_w = max(w, self.crop_size)
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 2. Augment
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image'].float()
        aug_mask = augmented['mask'].numpy().astype(np.uint8)
        
        # 3. Physics & Dynamic Task
        area_scale = 1.0 
        task_type = "generic"; text_suffix = "" 

        img_np = (img_tensor.permute(1, 2, 0).numpy()).astype(np.uint8)
        
        # 计算属性
        analysis = analyze_physical_attributes(img_np, aug_mask, self.attr_config, area_scale)
        visuals = analysis['visuals']
        
        # 4. Prompt & Organ ID
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
        # 生成密度图
        gt_heatmap = generate_adaptive_density(aug_mask, image_size=(self.image_size, self.image_size))
        
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