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
# 这些词会让模型学会区分“细胞核”和“长得像细胞核的东西”
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
    def __init__(self, data_dir, image_size=1024, crop_size=256, mode='train', 
                 mask_num=1, requires_name=True, 
                 # 🔥 注意：这里改成加载新的动态属性库
                 dynamic_attr_path="data/MoNuSeg_SA1B/dynamic_instance_attributes.json"):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        
        # === 1. 加载动态属性数据库 ===
        self.dynamic_attrs = {}
        if os.path.exists(dynamic_attr_path):
            print(f"📖 [DataLoader] Loading Dynamic Attributes from {dynamic_attr_path}...")
            with open(dynamic_attr_path, 'r') as f:
                content = json.load(f)
                self.dynamic_attrs = content.get("images", {})
        else:
            if mode == 'train':
                print(f"⚠️ [DataLoader] CRITICAL WARNING: {dynamic_attr_path} not found!")
                print("   Model will NOT learn controllable segmentation without this file.")

        # === 2. 扫描文件 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # 只保留有对应 JSON 的图片
        valid_paths = []
        for p in self.image_paths:
            base, _ = os.path.splitext(p)
            if os.path.exists(base + ".json"):
                valid_paths.append(p)
        
        if len(valid_paths) > 0:
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Found {len(self.image_paths)} valid image-json pairs.")
        else:
            print(f"⚠️ [DataLoader] No valid pairs found! Dynamic GT requires JSONs.")

        # === 3. 增强策略 (训练时必须强力增强) ===
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
                A.CenterCrop(width=crop_size, height=crop_size, p=1.0),
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # === 🔥 核心逻辑 2: 采样策略选择 ===
        # 默认值
        prompt_mode = "Generic" 
        target_tag = None
        text_prompt = "Cell nuclei"
        
        if self.mode == 'train':
            rand = random.random()
            
            # 🌑 策略 A (20%): 负样本 (Negative) -> 训练拒绝能力
            if rand < 0.2:
                prompt_mode = "Negative"
                text_prompt = random.choice(NEGATIVE_PROMPTS)
                
            # 🎯 策略 B (40%): 属性特定 (Attribute) -> 训练筛选能力
            elif rand < 0.6 and filename in self.dynamic_attrs:
                instances = self.dynamic_attrs[filename]
                all_tags = []
                for inst in instances:
                    all_tags.extend(inst.get('tags', []))
                
                if len(all_tags) > 0:
                    prompt_mode = "Attribute"
                    # 随机选一个标签，例如 "Small", "Round"
                    target_tag = random.choice(list(set(all_tags))) 
                    text_prompt = f"{target_tag} cell nuclei"
                else:
                    prompt_mode = "Generic"
            
            # 🌕 策略 C (40%): 通用 (Generic) -> 保持基础能力
            else:
                prompt_mode = "Generic"
                text_prompt = "Cell nuclei"

        # === 🔥 核心逻辑 3: 动态 Mask 构建 (Dynamic Mask Construction) ===
        # 0: 背景, 1: 目标, 255: 忽略 (冲突区域)
        IGNORE_INDEX = 255 
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 负样本模式：Mask 全黑，无需读取 JSON
        if prompt_mode == "Negative":
            pass 
            
        elif os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                anns = data.get('annotations', [])
                if not anns and isinstance(data, list): anns = data
                
                # 确定哪些 ID 是正样本，哪些是忽略样本
                target_ids = set()
                ignore_ids = set()
                
                if prompt_mode == "Attribute" and filename in self.dynamic_attrs:
                    instances_info = self.dynamic_attrs[filename]
                    for inst in instances_info:
                        tags = inst.get('tags', [])
                        if target_tag in tags:
                            target_ids.add(inst['id'])
                        else:
                            # 这是一个细胞，但不是我们要找的 -> 设为忽略
                            ignore_ids.add(inst['id'])
                            
                elif prompt_mode == "Generic":
                    # 通用模式下，所有细胞都是目标
                    target_ids = set(range(len(anns)))

                # 绘制 Mask
                for idx, ann in enumerate(anns):
                    is_target = idx in target_ids
                    is_ignore = idx in ignore_ids
                    
                    if not (is_target or is_ignore): continue
                    
                    if 'segmentation' in ann:
                        seg = ann['segmentation']
                        # RLE decoding
                        if isinstance(seg, dict) and 'counts' in seg:
                            m = coco_mask.decode(seg)
                        # Polygon decoding
                        elif isinstance(seg, list):
                            m = np.zeros((h, w), dtype=np.uint8)
                            for poly in seg:
                                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                                cv2.fillPoly(m, [pts], 1)
                        else:
                            continue
                            
                        # 赋值
                        if is_target:
                            mask[m > 0] = 1
                        elif is_ignore:
                            # 注意：不要覆盖已经是 1 的区域 (防止重叠时覆盖)
                            mask[(m > 0) & (mask == 0)] = IGNORE_INDEX
                                
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")

        # === 4. 增强 ===
        # 注意：插值必须用 nearest，否则 255 会变成 254, 253...
        # albumentations 对 mask 默认就是 nearest，但为了保险起见，我们在外部不手动改
        augmented = self.transform(image=image, mask=mask)
        
        image_tensor = augmented['image'].float()
        
        # Mask 需要保持 long 类型以便 CrossEntropy 使用，或者 float 给 Dice
        # 这里的 mask 包含 0, 1, 255
        mask_tensor = augmented['mask'].long().unsqueeze(0) 
        
        sample = {
            'image': image_tensor,
            'label': mask_tensor,
            'original_size': (self.image_size, self.image_size),
            'name': filename,
            'text_prompt': text_prompt
        }
        
        return sample