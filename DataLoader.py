import os
import cv2
import json
import torch
import numpy as np
import random
import glob
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 尝试导入 COCO 工具
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ [DataLoader] pycocotools not installed. SA-1B RLE decoding might fail.")

# === 🔥 核心组件 1: 医学困难负样本池 (Hard Negatives) ===
NEGATIVE_PROMPTS = [
    # Level 1: 最难的干扰 (生物学相似)
    "Red blood cells", "Eosinophilic cytoplasm", "Stromal tissue", 
    "Extracellular matrix", "Collagen fibers", "Adipose tissue cells",
    "Blood vessel lumen",
    # Level 2: 伪影与背景
    "Tissue folds", "Air bubbles", "Glass slide background", "Blurred regions",
    # Level 3: 语义陷阱 & 通用物体
    "Mitochondria", "Golgi apparatus", "A photo of a cat", "A car"
]

def stack_dict_batched(batch):
    """自定义 collate_fn"""
    tensor_dict = {}
    for key, value in batch[0].items():
        if key == 'text_prompt' or key == 'name':
            tensor_dict[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        else:
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict

class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=1024, crop_size=1024, mode='train', 
                 mask_num=1, requires_name=True, 
                 # 🔥 指向统计学文件
                 dynamic_attr_path="data/MoNuSeg_SA1B/dataset_stats.json"):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        
        # === 1. 加载统计学阈值 (PromptNu Logic) ===
        # 默认备用值 (万一文件读不到)
        self.size_thresholds = {"small_upper": 300, "large_lower": 600}
        
        if os.path.exists(dynamic_attr_path):
            print(f"📖 [DataLoader] Loading Statistics from {dynamic_attr_path}...")
            with open(dynamic_attr_path, 'r') as f:
                stats = json.load(f)
                # 读取 PromptNu 计算出的阈值 (Small < Mean, Large > Mean + 2*Std)
                if "thresholds" in stats:
                    self.size_thresholds = stats["thresholds"]
                    print(f"   ✅ Using PromptNu Statistical Thresholds: Small < {self.size_thresholds['small_upper']:.1f}, Large > {self.size_thresholds['large_lower']:.1f}")
        else:
            if mode == 'train':
                print(f"⚠️ [DataLoader] Stats file not found at {dynamic_attr_path}. Using default fallback.")

        # === 2. 扫描文件 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        # 过滤掉 mask 文件
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # SA-1B 格式检查
        valid_paths = []
        for p in self.image_paths:
            json_path = os.path.splitext(p)[0] + ".json"
            if os.path.exists(json_path):
                valid_paths.append(p)
        
        if len(valid_paths) > 0:
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Found {len(self.image_paths)} valid image-json pairs.")
        else:
            print(f"⚠️ [DataLoader] No valid pairs found! SA-1B mode requires local JSONs.")

        # === 3. 增强策略 (关键修改：修复 CropSizeError) ===
        if mode == 'train':
            self.transform = A.Compose([
                # 🔥 第一步：PadIfNeeded
                # 如果原图(1000)小于 crop_size(1024)，先填充边缘，防止 RandomCrop 报错
                # 同时也保证了训练时看到的细胞尺寸与测试时一致 (1:1)
                A.PadIfNeeded(
                    min_height=crop_size, 
                    min_width=crop_size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    mask_value=0
                ),
                # 🔥 第二步：RandomCrop
                # 在 Pad 后的图上随机切 1024 (如果 Pad 到 1024，这就等于全图)
                A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
                
                # 其他增强
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                
                # 最后 Resize (虽然 crop_size=1024=image_size，但这步留着保险)
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                # 测试时：直接 Pad 到 1024，保持原始比例和分辨率
                A.PadIfNeeded(
                    min_height=image_size, 
                    min_width=image_size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    mask_value=0
                ),
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    # === 基于 PromptNu 统计数据的解码 ===
    def decode_sa1b_mask(self, annotations, h, w, size_mode=None):
        """
        解码并根据 dataset_stats.json 里的阈值进行过滤
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        valid_pixel_count = 0
        
        # 从加载的统计数据中获取动态阈值
        small_thresh = self.size_thresholds['small_upper']  # Mean
        large_thresh = self.size_thresholds['large_lower']  # Mean + 2*Std
        
        for ann in annotations:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                m = None
                
                # RLE
                if isinstance(seg, dict) and 'counts' in seg:
                    m = coco_mask.decode(seg)
                # Polygon
                elif isinstance(seg, list):
                    m = np.zeros((h, w), dtype=np.uint8)
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(m, [pts], 1)
                
                if m is not None:
                    # 实时计算面积
                    area = np.sum(m > 0)
                    
                    keep = False
                    if size_mode == 'large':
                        # PromptNu 定义: Area > Mean + 2*Std
                        if area > large_thresh: keep = True 
                    elif size_mode == 'small':
                        # PromptNu 定义: Area < Mean
                        if area < small_thresh: keep = True 
                    else:
                        keep = True # Generic 模式
                    
                    if keep:
                        mask[m > 0] = 1
                        valid_pixel_count += area
                    
        return mask, valid_pixel_count

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 读取 JSON
        annotations = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        annotations = data.get('annotations', [])
                    elif isinstance(data, list):
                        annotations = data
            except:
                pass

        # === 采样逻辑 ===
        rand = random.random()
        text_prompt = "Cell nuclei"
        target_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. 负样本 (10%)
        if self.mode == 'train' and rand < 0.1:
            text_prompt = random.choice(NEGATIVE_PROMPTS)
            target_mask = np.zeros((h, w), dtype=np.uint8)
            
        # 2. 属性训练 (45%)
        elif self.mode == 'train' and rand < 0.55 and len(annotations) > 0:
            if random.random() < 0.5:
                # Large
                text_prompt = random.choice(["Large nuclei", "Tumor nuclei"])
                target_mask, px_count = self.decode_sa1b_mask(annotations, h, w, size_mode='large')
            else:
                # Small
                text_prompt = random.choice(["Small nuclei", "Lymphocyte nuclei"])
                target_mask, px_count = self.decode_sa1b_mask(annotations, h, w, size_mode='small')
            
            # 🔥 兜底机制：如果当前图没有符合统计阈值的细胞（全黑），回退到 Generic
            if target_mask.sum() == 0:
                text_prompt = "Cell nuclei"
                target_mask, _ = self.decode_sa1b_mask(annotations, h, w, size_mode=None)

        # 3. 通用训练 (45%)
        else:
            text_prompt = "Cell nuclei"
            target_mask, _ = self.decode_sa1b_mask(annotations, h, w, size_mode=None)
        
        # 增强 (此时 target_mask 尺寸是原始的，增强后变成 1024)
        augmented = self.transform(image=image, mask=target_mask)
        
        return {
            "image": augmented['image'].float(),
            "label": augmented['mask'].long().unsqueeze(0),
            "text_prompt": text_prompt,
            "name": filename.split('.')[0],
            "original_size": (self.image_size, self.image_size)
        }