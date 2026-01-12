import os
import cv2
import json
import torch
import numpy as np
import glob
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

# === 全局 ID 映射表 (必须与 build_kb.py 保持一致) ===
# 建议放在单独的 config.py 中，这里为了方便直接定义
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

class TrainingDataset(data.Dataset):
    def __init__(self, 
                 data_dir, 
                 knowledge_path=None,  # <--- 🔥 完全参数化，不写死路径
                 image_size=1024, 
                 crop_size=1024, 
                 mode='train'):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        self.organ_to_id = ORGAN_TO_ID
        
        # === 1. 加载显式知识库 (由 train.py 传入) ===
        self.knowledge_base = {}
        if knowledge_path and os.path.exists(knowledge_path):
            print(f"📖 [DataLoader] Loading Knowledge Base from: {knowledge_path}")
            with open(knowledge_path, 'r') as f:
                self.knowledge_base = json.load(f)
        else:
            if mode == 'train':
                print(f"⚠️ [DataLoader] Warning: Knowledge path '{knowledge_path}' not found! Using default prompts.")

        # === 2. 扫描数据 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        # 排除 mask 文件
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # 训练模式下只保留有标注的
        if mode == 'train':
            valid_paths = []
            for p in self.image_paths:
                json_p = os.path.splitext(p)[0] + ".json"
                if os.path.exists(json_p):
                    valid_paths.append(p)
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Initialized with {len(self.image_paths)} images (Mode: {mode})")

        # === 3. 增强策略 ===
        if mode == 'train':
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=cv2.BORDER_CONSTANT, value=0),
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
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def decode_mask(self, json_path, h, w):
        """通用 Mask 解码"""
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            annotations = data.get('annotations', [])
            
            for ann in annotations:
                if 'segmentation' not in ann: continue
                seg = ann['segmentation']
                
                if isinstance(seg, list): # Polygon
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(mask, [pts], 1)
                elif isinstance(seg, dict): # RLE
                    m = coco_mask.decode(seg)
                    mask[m > 0] = 1
        except:
            pass
        return mask

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        json_path = os.path.splitext(img_path)[0] + ".json"
        
        # 1. 读图
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 2. 读 Mask (仅训练/验证需要)
        target_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(json_path):
            target_mask = self.decode_mask(json_path, h, w)

        # 3. 🔥 获取知识 (MP-SAM 核心) 🔥
        # 从知识库中查表，不再硬编码
        kb_entry = self.knowledge_base.get(filename, {})
        
        # [Implicit] Organ ID -> DualPromptLearner
        organ_name = kb_entry.get("organ_id", "Generic")
        organ_id = self.organ_to_id.get(organ_name, 9) 
        
        # [Explicit] Attribute Text -> KIM
        attribute_text = kb_entry.get("text_prompt", "Microscopic image of cell nuclei.")
        
        # 4. 增强
        augmented = self.transform(image=image, mask=target_mask)
        
        return {
            "image": augmented['image'].float(),
            "label": augmented['mask'].long().unsqueeze(0),
            
            # --- 知识流 ---
            "organ_id": organ_id,           # Int
            "attribute_text": attribute_text, # Str
            "text_prompt": "Cell nuclei",   # Str (Base Prompt)
            
            "name": filename,
            "original_size": (self.image_size, self.image_size)
        }