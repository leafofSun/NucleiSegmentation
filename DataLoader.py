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

try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ [DataLoader] pycocotools not installed! RLE decoding will FAIL.")

# === 负样本池 ===
NEGATIVE_PROMPTS = [
    "Red blood cells", "Eosinophilic cytoplasm", "Stromal tissue", 
    "Extracellular matrix", "Collagen fibers", "Adipose tissue cells",
    "Blood vessel lumen", "Tissue folds", "Air bubbles", 
    "Glass slide background", "Mitochondria", "A photo of a cat"
]
def stack_dict_batched(batch):
    """自定义 collate_fn"""
    tensor_dict = {}
    for key, value in batch[0].items():
        if key == 'text_prompt' or key == 'name':
            # 字符串列表
            tensor_dict[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            # Tensor 堆叠
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        else:
            # 其他类型作为列表
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict
class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=1024, crop_size=1024, mode='train', 
                 mask_num=1, requires_name=True, 
                 # 🔥 [新增] 加上这个参数以兼容 train.py 的旧调用，虽然我们不用它
                 dynamic_attr_path=None, 
                 # 下面保持不变
                 stats_path="data/MoNuSeg_SA1B/dataset_stats.json",
                 prompts_path="data/MoNuSeg_SA1B/specific_prompts.json"):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        
        # === 1. 加载统计学阈值 (数值) ===
        # 默认备用值
        self.size_thresholds = {"small_upper": 424.3, "large_lower": 731.2}
        
        if os.path.exists(stats_path):
            print(f"📖 [DataLoader] Loading Stats from {stats_path}...")
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                if "thresholds" in stats:
                    self.size_thresholds = stats["thresholds"]
                    print(f"   ✅ Mask Filtering Thresholds: Small < {self.size_thresholds['small_upper']:.1f}, Large > {self.size_thresholds['large_lower']:.1f}")
        
        # === 2. 加载专用文本库 (语义) ===
        self.specific_library = {}
        if os.path.exists(prompts_path):
            print(f"📖 [DataLoader] Loading Specific Prompts from {prompts_path}...")
            with open(prompts_path, 'r') as f:
                self.specific_library = json.load(f)
        else:
            print(f"⚠️ [DataLoader] Specific prompts not found at {prompts_path}. Smart sampling disabled.")

        # === 3. 扫描文件 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        if mode == 'train':
            # 只保留有标注的图
            self.image_paths = [p for p in self.image_paths if os.path.exists(os.path.splitext(p)[0] + ".json")]
            print(f"✅ [DataLoader] Found {len(self.image_paths)} valid training images.")

        # === 4. 增强策略 (Pad + Crop 1024) ===
        if mode == 'train':
            self.transform = A.Compose([
                A.PadIfNeeded(
                    min_height=crop_size, min_width=crop_size, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
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
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def decode_sa1b_mask(self, annotations, h, w, size_mode=None):
        """
        解码并过滤：
        - 符合条件的 -> Label 1
        - 不符合条件的（但确是一个细胞） -> Label 255 (Ignore)
        - 真正的背景 -> Label 0
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 使用加载的统计学阈值
        small_thresh = self.size_thresholds['small_upper']
        large_thresh = self.size_thresholds['large_lower']
        
        for ann in annotations:
            if 'segmentation' not in ann: continue
            
            # 解码单个 Mask
            m = None
            if isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                m = coco_mask.decode(ann['segmentation'])
            elif isinstance(ann['segmentation'], list):
                m = np.zeros((h, w), dtype=np.uint8)
                for poly in ann['segmentation']:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(m, [pts], 1)
            
            if m is not None:
                area = np.sum(m > 0)
                
                # 判断逻辑
                is_target = False
                is_ignore = False
                
                if size_mode == 'large':
                    if area > large_thresh: is_target = True
                    else: is_ignore = True # 是细胞，但太小，忽略
                elif size_mode == 'small':
                    if area < small_thresh: is_target = True
                    else: is_ignore = True # 是细胞，但太大，忽略
                else:
                    is_target = True # Generic 模式全都要
                
                # 赋值 (注意覆盖顺序)
                if is_target:
                    mask[m > 0] = 1
                elif is_ignore:
                    # 只有在还没被标记为 Target 的地方才标记 Ignore (防止重叠覆盖)
                    mask[(m > 0) & (mask == 0)] = 255
                    
        return mask

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        key_name = filename.replace(".tif", "").replace(".png", "") # 用于查表
        json_path = os.path.splitext(img_path)[0] + ".json"
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 1. 读取标注
        annotations = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', []) if isinstance(data, dict) else data
        except: pass

        # 2. 查阅专用文本库
        # 获取该图的专属描述 (如果查不到就用默认值)
        img_info = self.specific_library.get(filename, self.specific_library.get(key_name, {}))
        
        specific_prompt_text = img_info.get("prompt", "Microscopic image of cell nuclei.")
        img_attrs = img_info.get("attributes", {}) # {'size': 'large', ...}
        
        # === 🔥 智能采样逻辑 ===
        rand = random.random()
        text_prompt = "Cell nuclei"
        target_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 🌑 Task A: 负样本 (10%)
        if self.mode == 'train' and rand < 0.1:
            text_prompt = random.choice(NEGATIVE_PROMPTS)
            target_mask = np.zeros((h, w), dtype=np.uint8)
            
        # 🌕 Task B: 属性特定任务 (45%)
        # 根据 specific_prompts.json 里的标签来决定练什么
        elif self.mode == 'train' and rand < 0.55:
            img_size_tag = img_attrs.get("size", "medium")
            
            # 如果这张图本身就是 large，那我们大概率练 Large 任务
            if img_size_tag == "large" and random.random() < 0.8:
                text_prompt = random.choice(["Large nuclei", "Tumor nuclei"])
                target_mask = self.decode_sa1b_mask(annotations, h, w, size_mode='large')
            
            # 如果这张图是 small，大概率练 Small 任务
            elif img_size_tag == "small" and random.random() < 0.8:
                text_prompt = random.choice(["Small nuclei", "Lymphocyte nuclei"])
                target_mask = self.decode_sa1b_mask(annotations, h, w, size_mode='small')
            
            # 否则 (medium 或 没命中概率)，练通用任务
            else:
                text_prompt = "Cell nuclei"
                target_mask = self.decode_sa1b_mask(annotations, h, w, size_mode=None)
            
            # 兜底：如果过滤完Mask是全黑的，强制回退到 Context Task
            if target_mask.sum() == 0:
                text_prompt = specific_prompt_text # 使用生成的长文本
                target_mask = self.decode_sa1b_mask(annotations, h, w, size_mode=None)

        # 🌟 Task C: 上下文感知任务 (45%)
        # 使用生成的长文本："Microscopic image of large, round nuclei..."
        else:
            text_prompt = specific_prompt_text
            target_mask = self.decode_sa1b_mask(annotations, h, w, size_mode=None)
        
        # 增强
        augmented = self.transform(image=image, mask=target_mask)
        return {
            "image": augmented['image'].float(),
            "label": augmented['mask'].long().unsqueeze(0),
            "text_prompt": text_prompt,
            "name": filename.split('.')[0],
            "original_size": (self.image_size, self.image_size)
        }