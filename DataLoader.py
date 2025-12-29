import os
import cv2
import json
import torch
import numpy as np
from torch.utils import data
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 解析 SA-1B 格式必须用 pycocotools
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ [DataLoader] pycocotools not installed. SA-1B RLE decoding might fail.")

def stack_dict_batched(batch):
    """
    自定义 collate_fn，用于处理字典列表
    """
    tensor_dict = {}
    for key, value in batch[0].items():
        if key == 'text_prompt' or key == 'name':
            tensor_dict[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, np.ndarray):
            tensor_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
        else:
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict

class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=1024, crop_size=256, mode='train', point_num=1, mask_num=5, requires_name=True, prompt_path="data/prompt_info.json"):
        """
        Args:
            data_dir: 数据路径
            image_size: 模型输入尺寸 (建议 1024, 适配 SAM ViT)
            crop_size: 从原图裁剪的 Patch 大小 (建议 256, 模拟滑动窗口)
            mode: 'train' 开启增强, 'test' 仅中心裁剪
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.mode = mode
        self.point_num = point_num
        self.mask_num = mask_num
        self.requires_name = requires_name
        
        # === 1. 加载 Prompt JSON ===
        self.prompt_dict = {}
        if os.path.exists(prompt_path):
            print(f"📖 [DataLoader] Loading Prompts from {prompt_path}...")
            with open(prompt_path, 'r') as f:
                self.prompt_dict = json.load(f)
        else:
            # 训练时必须要有 Prompt，否则打印警告
            if mode == 'train':
                print(f"⚠️ [DataLoader] Warning: {prompt_path} not found! Will use default prompts.")

        # === 2. 扫描文件 ===
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            
        # 过滤掉 mask 图片
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # 筛选有对应 JSON 的图片
        valid_paths = []
        for p in self.image_paths:
            base, _ = os.path.splitext(p)
            if os.path.exists(base + ".json"):
                valid_paths.append(p)
        
        if len(valid_paths) > 0:
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Found {len(self.image_paths)} images with matching SA-1B JSONs.")
        else:
            print(f"⚠️ [DataLoader] No JSONs found! Found {len(self.image_paths)} images (assuming masks implied).")

        # === 3. 定义增强策略 (Albumentations) ===
        # 注意：这里不使用 A.Normalize，因为 SAM 模型内部有自己的 preprocess
        # 我们只负责几何变换和 resize，输出 0-255 的 tensor
        
        if mode == 'train':
            self.transform = A.Compose([
                # 空间增强：模拟滑动窗口，保证多样性
                A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 弹性形变：非常适合细胞核这种软组织
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                
                # 颜色增强：病理图像的核心，防止过拟合特定染色风格
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # 可选
                
                # 核心步骤：放大到 1024，激活 ViT 的细节感知能力
                A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
                
                # 转 Tensor (C, H, W)
                ToTensorV2(),
            ])
        else:
            # 验证/测试：保持一致的预处理，但不做随机增强
            # 使用 CenterCrop 取中间一块代表性区域进行验证
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
        
        # 1. 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 2. 读取 Label (SA-1B JSON)
        base_name, _ = os.path.splitext(img_path)
        json_path = base_name + ".json"
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                anns = data.get('annotations', [])
                if not anns and isinstance(data, list): anns = data
                
                for ann in anns:
                    if 'segmentation' in ann:
                        seg = ann['segmentation']
                        # RLE
                        if isinstance(seg, dict) and 'counts' in seg:
                            rle_mask = coco_mask.decode(seg)
                            mask[rle_mask > 0] = 1
                        # Polygon
                        elif isinstance(seg, list):
                            for poly in seg:
                                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                                cv2.fillPoly(mask, [pts], 1)
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")
        else:
            # 回退策略
            mask_path = img_path.replace(".tif", ".png").replace(".jpg", ".png").replace("Images", "Labels")
            if os.path.exists(mask_path):
                 m_temp = cv2.imread(mask_path, 0)
                 if m_temp is not None: mask = (m_temp > 0).astype(np.uint8)

        # 3. 应用增强 (Augmentation & Resize)
        # albumentations 会自动同时处理 image 和 mask
        augmented = self.transform(image=image, mask=mask)
        
        image_tensor = augmented['image'].float() # [3, 1024, 1024], 0-255
        mask_tensor = augmented['mask'].float().unsqueeze(0) # [1, 1024, 1024], 0/1
        
        # 4. 获取 Prompt
        text_prompt = "Cell nuclei"
        if filename in self.prompt_dict:
            info = self.prompt_dict[filename]
            # 优先使用 rich_text
            text_prompt = info.get("rich_text", info.get("target_text", "Cell nuclei"))

        # 5. 构造返回样本
        sample = {
            'image': image_tensor,
            'label': mask_tensor,
            # 这里的 original_size 告诉 SAM 后处理层：现在的特征图对应的是多大的图
            # 因为我们 Resize 到了 1024，所以这里应该是 (1024, 1024)
            # 这样 SAM 生成的 mask 就会和我们的 label 对齐
            'original_size': (self.image_size, self.image_size), 
            'name': filename,
            'text_prompt': text_prompt
        }
        
        return sample