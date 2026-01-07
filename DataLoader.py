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
            tensor_dict[key] = torch.stack([sample[key] for sample in batch] )
        else:
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict

class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=1024, crop_size=256, mode='train', 
                 mask_num=1, requires_name=True, 
                 # 🔥 这里指向你生成的那个包含所有图片统计信息的全局 JSON
                 dynamic_attr_path="data/MoNuSeg_SA1B/train_dynamic_instance_attributes.json"):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        
        # === 1. 加载动态属性数据库 (Global Stats) ===
        self.dynamic_attrs = {}
        if os.path.exists(dynamic_attr_path):
            print(f"📖 [DataLoader] Loading Dynamic Attributes from {dynamic_attr_path}...")
            with open(dynamic_attr_path, 'r') as f:
                content = json.load(f)
                # 你的 JSON 结构里，数据是在 "images" 键下
                self.dynamic_attrs = content.get("images", {})
        else:
            if mode == 'train':
                print(f"⚠️ [DataLoader] CRITICAL WARNING: {dynamic_attr_path} not found!")

        # === 2. 扫描文件 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        # 过滤掉 mask 文件
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # SA-1B 格式检查：只保留有对应本地 .json 的图片
        valid_paths = []
        for p in self.image_paths:
            # 假设 image.tif 对应 image.json
            json_path = os.path.splitext(p)[0] + ".json"
            if os.path.exists(json_path):
                valid_paths.append(p)
        
        if len(valid_paths) > 0:
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Found {len(self.image_paths)} valid image-json pairs.")
        else:
            print(f"⚠️ [DataLoader] No valid pairs found! SA-1B mode requires local JSONs.")

        # === 3. 增强策略 ===
        if mode == 'train':
            self.transform = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                # 测试时通常 CenterCrop 或者 Resize
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

   # === 修改 1: 实时计算面积，不再依赖 JSON 字段 ===
    def decode_sa1b_mask(self, annotations, h, w, size_mode=None):
        """
        解码并根据大小过滤
        size_mode: None (全部), 'large', 'small'
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        valid_pixel_count = 0
        
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
                    # 🔥 核心修正：这里实时计算面积！
                    area = np.sum(m > 0)
                    
                    keep = False
                    if size_mode == 'large':
                        if area > 300: keep = True # 阈值
                    elif size_mode == 'small':
                        if area < 150: keep = True # 阈值
                    else:
                        keep = True # Generic 模式，全留
                    
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

        # === 策略调整 ===
        rand = random.random()
        text_prompt = "Cell nuclei"
        target_mask = np.zeros((h, w), dtype=np.uint8)
        
        # ⬇️ 修正点：降低负样本比例，只有 10%
        if self.mode == 'train' and rand < 0.1:
            text_prompt = random.choice(NEGATIVE_PROMPTS)
            target_mask = np.zeros((h, w), dtype=np.uint8)
            
        # ⬇️ 修正点：属性训练 (45%)
        elif self.mode == 'train' and rand < 0.55 and len(annotations) > 0:
            if random.random() < 0.5:
                # Large
                text_prompt = random.choice(["Large nuclei", "Tumor nuclei"])
                target_mask, px_count = self.decode_sa1b_mask(annotations, h, w, size_mode='large')
            else:
                # Small
                text_prompt = random.choice(["Small nuclei", "Lymphocyte nuclei"])
                target_mask, px_count = self.decode_sa1b_mask(annotations, h, w, size_mode='small')
            
            # 🔥 救命机制：如果过滤完发现全是黑的（比如这图里根本没有大细胞）
            # 强制回退到“Generic”模式，不要训练黑Mask！
            if target_mask.sum() == 0:
                text_prompt = "Cell nuclei"
                target_mask, _ = self.decode_sa1b_mask(annotations, h, w, size_mode=None)

        # ⬇️ 修正点：通用训练 (45%) - 提高基础能力权重
        else:
            text_prompt = "Cell nuclei"
            target_mask, _ = self.decode_sa1b_mask(annotations, h, w, size_mode=None)
        
        # 增强
        augmented = self.transform(image=image, mask=target_mask)
        return {
            "image": augmented['image'].float(),
            "label": augmented['mask'].long().unsqueeze(0),
            "text_prompt": text_prompt,
            "name": filename.split('.')[0],
            "original_size": (self.image_size, self.image_size)
        }