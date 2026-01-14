import os
import cv2
import json
import torch
import numpy as np
import glob
import random
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Warning: sklearn not available. Arrangement analysis will use simplified method.")

# === 全局 ID 映射表 ===
ORGAN_TO_ID = {
    "Kidney": 0, "Breast": 1, "Prostate": 2, "Lung": 3, 
    "Colon": 4, "Stomach": 5, "Liver": 6, "Bladder": 7, 
    "Brain": 8, "Generic": 9
}

def stack_dict_batched(batch):
    """自定义 Collate Function"""
    tensor_dict = {}
    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, (int, float, str)):
            tensor_dict[key] = [sample[key] for sample in batch]
        else:
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict

# 🔥 [修改] 5维属性分析器 (支持面积缩放)
def analyze_comprehensive_attributes(image, mask, area_scale=1.0):
    """
    计算 PromptNu 定义的 5 个维度的属性。
    🔥 [核心修正] 引入 area_scale，解决 Crop->Resize 导致的面积膨胀问题。
    
    Args:
        area_scale (float): 面积缩放倍数。例如 256->1024 放大时，Scale=16.0。
    """
    # 默认返回值
    default_visuals = {
        "color": "deep-purple stained", "shape": "round", "arrangement": "uniform",
        "size": "medium", "density": "moderate"
    }
    default_labels = [0, 0, 0, 1, 1]

    if mask.sum() == 0:
        return default_visuals, default_labels

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    if not regions:
        return default_visuals, default_labels

    # === 1. 大小 (Size) [0: Small, 1: Medium, 2: Large] ===
    # 原始阈值: Small < 250, Large > 600
    # 🔥 动态调整阈值：乘以面积缩放倍数
    th_small = 250.0 * area_scale
    th_large = 600.0 * area_scale

    areas = np.array([r.area for r in regions])
    mean_area = np.mean(areas)
    
    if mean_area < th_small:
        size_lbl, size_txt = 0, "small"
    elif mean_area > th_large:
        size_lbl, size_txt = 2, "large"
    else:
        size_lbl, size_txt = 1, "medium"

    # === 2. 形状 (Shape) [缩放不变] ===
    eccs = np.array([r.eccentricity for r in regions])
    mean_ecc = np.mean(eccs)
    if mean_ecc < 0.6:
        shape_lbl, shape_txt = 0, "round"
    elif mean_ecc < 0.85:
        shape_lbl, shape_txt = 1, "oval"
    else:
        shape_lbl, shape_txt = 2, "elongated/irregular"

    # === 3. 密度 (Density) [使用覆盖率，缩放不变] ===
    img_area = mask.shape[0] * mask.shape[1]
    coverage = np.sum(areas) / img_area
    if coverage < 0.05:
        den_lbl, den_txt = 0, "sparsely distributed"
    elif coverage > 0.20:
        den_lbl, den_txt = 2, "densely packed"
    else:
        den_lbl, den_txt = 1, "moderately distributed"

    # === 4. 排列 (Arrangement) [相对距离 CV，缩放不变] ===
    centroids = np.array([r.centroid for r in regions])
    if len(centroids) > 5:
        if SKLEARN_AVAILABLE:
            try:
                tree = KDTree(centroids)
                dists, _ = tree.query(centroids, k=2)
                nn_dists = dists[:, 1]
                dist_cv = np.std(nn_dists) / (np.mean(nn_dists) + 1e-6)
                if dist_cv > 0.6:
                    arr_lbl, arr_txt = 1, "disordered/clustered"
                else:
                    arr_lbl, arr_txt = 0, "uniformly arranged"
            except:
                arr_lbl, arr_txt = 0, "uniformly arranged"
        else:
            # 简化方法
            centroid_std = np.std(centroids, axis=0).mean()
            # 这里的 0.3 是个经验相对值，大致稳健
            if centroid_std > np.mean(centroids) * 0.3:
                arr_lbl, arr_txt = 1, "disordered/clustered"
            else:
                arr_lbl, arr_txt = 0, "uniformly arranged"
    else:
        arr_lbl, arr_txt = 0, "isolated"

    # === 5. 颜色 (Color) [缩放不变] ===
    if image is not None and image.size > 0:
        mask_bool = mask > 0
        if mask_bool.sum() > 0:
            if len(image.shape) == 3:
                masked_pixels = image[mask_bool]
                mean_brightness = np.mean(masked_pixels)
                if mean_brightness > 200:
                    col_lbl, col_txt = 1, "pink/light stained"
                else:
                    col_lbl, col_txt = 0, "deep-purple stained"
            else:
                col_lbl, col_txt = 0, "deep-purple stained"
        else:
            col_lbl, col_txt = 0, "deep-purple stained"
    else:
        col_lbl, col_txt = 0, "deep-purple stained"

    visuals = {
        "color": col_txt, "shape": shape_txt, "arrangement": arr_txt,
        "size": size_txt, "density": den_txt
    }
    attr_labels = [col_lbl, shape_lbl, arr_lbl, size_lbl, den_lbl]
    
    return visuals, attr_labels

# 🔥 [新增] 辅助函数：生成椭圆高斯热力图
def generate_elliptical_heatmap(mask, image_size=(1024, 1024), sigma_scale=0.25):
    """
    根据 Mask 生成 Variable Ellipse Gaussian Heatmap.
    """
    heatmap = np.zeros(image_size, dtype=np.float32)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    for region in regions:
        if region.area < 10: continue 
        
        y0, x0 = region.centroid
        orientation = region.orientation
        major = region.major_axis_length
        minor = region.minor_axis_length
        
        sigma_x = max(1.0, major * sigma_scale)
        sigma_y = max(1.0, minor * sigma_scale)
        
        theta = -orientation
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        cos2, sin2 = cos_t**2, sin_t**2
        a = cos2 / (2 * sigma_x**2) + sin2 / (2 * sigma_y**2)
        b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
        c = sin2 / (2 * sigma_x**2) + cos2 / (2 * sigma_y**2)
        
        bb_size = int(max(major, minor) * 1.5)
        y_min, y_max = max(0, int(y0 - bb_size)), min(image_size[0], int(y0 + bb_size + 1))
        x_min, x_max = max(0, int(x0 - bb_size)), min(image_size[1], int(x0 + bb_size + 1))
        
        if x_max <= x_min or y_max <= y_min: continue

        xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        dx = xx - x0
        dy = yy - y0
        
        gaussian = np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)
        
    return heatmap

class TrainingDataset(data.Dataset):
    def __init__(self, 
                 data_dir, 
                 knowledge_path=None, 
                 image_size=1024, # 模型输入尺寸
                 crop_size=256,   # 🔥 [修改] 物理切片尺寸 (默认推荐 256)
                 mode='train',
                 prompt_mode='dynamic'):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.patch_size = crop_size # 重命名为 patch_size 以示区别
        self.mode = mode
        self.organ_to_id = ORGAN_TO_ID
        self.prompt_mode = prompt_mode
        
        # === 1. 加载显式知识库 ===
        self.knowledge_base = {}
        if knowledge_path and os.path.exists(knowledge_path):
            print(f"📖 [DataLoader] Loading Knowledge Base from: {knowledge_path}")
            with open(knowledge_path, 'r') as f:
                self.knowledge_base = json.load(f)
        else:
            if mode == 'train':
                print(f"⚠️ [DataLoader] Warning: Knowledge path not found! Defaults will be used.")

        # === 2. 扫描数据 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        if mode == 'train':
            valid_paths = []
            for p in self.image_paths:
                json_p = os.path.splitext(p)[0] + ".json"
                if not os.path.exists(json_p):
                     json_p = p.rsplit('.', 1)[0] + ".json"
                
                if os.path.exists(json_p):
                    valid_paths.append(p)
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Initialized with {len(self.image_paths)} images (Mode: {mode})")

        # === 3. 增强策略 (核心修正) ===
        if mode == 'train':
            self.transform = A.Compose([
                # 1. 物理裁剪: 先裁出 256x256 (或传入的 crop_size)
                A.PadIfNeeded(min_height=self.patch_size, min_width=self.patch_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(width=self.patch_size, height=self.patch_size, p=1.0),
                
                # 2. 几何增强
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                
                # 3. 🔥 放大回 1024 以适配 SAM 预训练分辨率
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])
        else:
            # 验证/测试集通常不在此 Resize，而是交由 Sliding Window 处理
            # 或者保持原图尺寸输出
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                # 验证集如果不是滑动窗口，可以在这里 resize，但建议在 train.py 用滑动窗口
                # A.Resize(height=image_size, width=image_size), 
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def decode_mask(self, json_path, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            annotations = data.get('annotations', []) if isinstance(data, dict) else data
            
            for ann in annotations:
                if 'segmentation' not in ann: continue
                seg = ann['segmentation']
                
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(mask, [pts], 1)
                elif isinstance(seg, dict):
                    m = coco_mask.decode(seg)
                    mask[m > 0] = 1
        except:
            pass
        return mask

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        
        # 1. 读图
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 2. 读原始 GT Mask
        target_mask = np.zeros((h, w), dtype=np.uint8)
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
             json_path = img_path.rsplit('.', 1)[0] + ".json"
             
        if os.path.exists(json_path):
            target_mask = self.decode_mask(json_path, h, w)

        # 3. 增强 (Crop -> Augment -> Resize)
        augmented = self.transform(image=image, mask=target_mask)
        img_tensor = augmented['image'].float()
        
        # 转回 numpy 进行物理分析
        aug_mask_np = augmented['mask'].numpy() 
        if aug_mask_np.ndim == 3: aug_mask_np = aug_mask_np[0]
        aug_mask_np = aug_mask_np.astype(np.uint8)

        # ============================================================
        # 🔥 计算面积缩放因子 (Area Scale Factor)
        # ============================================================
        if self.mode == 'train':
            # 例如: (1024 / 256)^2 = 16.0
            scale_linear = self.image_size / self.patch_size
            area_scale = scale_linear ** 2
        else:
            area_scale = 1.0

        # ============================================================
        # 🔥 动态任务生成 & 物理过滤
        # ============================================================
        task_type = "generic"
        text_prompt = "Cell nuclei"
        active_mask = aug_mask_np.copy()
        
        kb_entry = self.knowledge_base.get(filename, {})
        organ_name = kb_entry.get("organ_id", "Generic")
        organ_id = self.organ_to_id.get(organ_name, 9)
        attribute_text = kb_entry.get("text_prompt", "Microscopic image of cell nuclei.")

        labeled_mask = label(aug_mask_np)
        regions = regionprops(labeled_mask)

        # 仅在训练模式且有细胞时进行动态采样
        if self.mode == 'train' and self.prompt_mode == 'dynamic' and len(regions) > 5:
            areas = np.array([r.area for r in regions])
            min_a, max_a = np.min(areas), np.max(areas)
            
            is_diverse = (max_a > min_a * 2.0)
            rand_p = random.random()
            
            if rand_p < 0.5 or not is_diverse:
                task_type = "generic"
                text_prompt = "Cell nuclei"
                
            elif rand_p < 0.75:
                # === 找大细胞 (Large) ===
                task_type = "large"
                text_prompt = "Large, pleomorphic tumor nuclei"
                # 动态阈值 (Percentile 自动适应放大后的面积分布)
                th_high = np.percentile(areas, 67)
                active_mask = np.zeros_like(aug_mask_np)
                
                valid_count = 0
                for r in regions:
                    if r.area >= th_high:
                        active_mask[labeled_mask == r.label] = 1
                        valid_count += 1
                
                if valid_count == 0:
                    task_type = "generic"
                    text_prompt = "Cell nuclei"
                    active_mask = aug_mask_np.copy()
            
            else:
                # === 找小细胞 (Small) ===
                task_type = "small"
                text_prompt = "Small, round lymphocyte nuclei"
                th_low = np.percentile(areas, 33)
                active_mask = np.zeros_like(aug_mask_np)
                
                valid_count = 0
                for r in regions:
                    if r.area <= th_low and r.eccentricity < 0.9:
                        active_mask[labeled_mask == r.label] = 1
                        valid_count += 1
                
                if valid_count == 0:
                    task_type = "generic"
                    text_prompt = "Cell nuclei"
                    active_mask = aug_mask_np.copy()

        # ============================================================
        
        # 4. 🔥 物理计算 (传入 area_scale 进行修正)
        img_for_analysis = img_tensor.permute(1, 2, 0).numpy()
        if img_for_analysis.max() <= 1.0:
            img_for_analysis = (img_for_analysis * 255).astype(np.uint8)
        else:
            img_for_analysis = img_for_analysis.astype(np.uint8)
        
        visuals, attr_labels_list = analyze_comprehensive_attributes(
            img_for_analysis,
            active_mask,
            area_scale=area_scale  # 🔥 关键修正
        )
        
        attr_labels_tensor = torch.tensor(attr_labels_list).long()
        
        # 5. 构造融合 Prompt
        full_prompt = (f"Microscopic view of {visuals['density']}, {visuals['size']} nuclei, "
                      f"{visuals['arrangement']}, with {visuals['shape']} features.")
        
        if task_type == "generic":
            text_prompt = full_prompt
        else:
            text_prompt = f"{text_prompt} ({visuals['density']}, {visuals['arrangement']})"
        
        # 6. 封装返回数据
        label_tensor = torch.from_numpy(active_mask).long().unsqueeze(0)
        
        # 生成椭圆热力图 (基于 Resize 后的 Mask)
        gt_heatmap = generate_elliptical_heatmap(active_mask, image_size=(self.image_size, self.image_size))
        heatmap_tensor = torch.from_numpy(gt_heatmap).float().unsqueeze(0)
        
        # Prompt Dropout
        if self.mode == 'train' and random.random() < 0.2:
            organ_id = 9 # Generic

        return {
            "image": img_tensor,
            "label": label_tensor,         # [1, 1024, 1024]
            "gt_heatmap": heatmap_tensor,  # [1, 1024, 1024]
            
            "organ_id": organ_id,
            "attribute_text": attribute_text,
            "text_prompt": text_prompt,
            
            "attr_labels": attr_labels_tensor, # [5] 监督信号
            
            "name": filename,
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type
        }