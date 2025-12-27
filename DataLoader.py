import os
import cv2
import json
import torch
import numpy as np
import random
from torch.utils import data
import glob

# 解析 SA-1B 格式必须用 pycocotools
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ [DataLoader] pycocotools not installed. SA-1B RLE decoding might fail.")
    # pip install pycocotools

def stack_dict_batched(batch):
    """
    自定义 collate_fn，用于处理字典列表
    """
    tensor_dict = {}
    for key, value in batch[0].items():
        if key == 'text_prompt':
            tensor_dict[key] = [sample[key] for sample in batch]
        elif key == 'name':
             tensor_dict[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, np.ndarray):
            tensor_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
        else:
            tensor_dict[key] = [sample[key] for sample in batch]
    return tensor_dict

class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', point_num=1, mask_num=5, requires_name=True, prompt_path="data/prompt_info.json"):
        self.data_dir = data_dir
        self.image_size = image_size
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
            print(f"⚠️ [DataLoader] Warning: {prompt_path} not found! Will use default prompts.")

        # === 2. 扫描文件 (SA-1B 格式) ===
        # SA-1B 格式通常是: 图像(.jpg/.png) 和 标注(.json) 同名混在一起，或者分文件夹
        # 这里假设是混在一起或标准结构
        self.image_paths = []
        extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            
        # 过滤掉 mask 图片（如果有的话），因为我们用 JSON
        self.image_paths = [p for p in self.image_paths if "mask" not in p.lower()]
        
        # 检查是否真的有对应的 JSON
        valid_paths = []
        json_count = 0
        for p in self.image_paths:
            base, _ = os.path.splitext(p)
            # 检查同名 JSON
            if os.path.exists(base + ".json"):
                valid_paths.append(p)
                json_count += 1
            # 兼容：有些数据集 JSON 放在 ../labels/ 目录
            # else: 
            #    ... (可根据需要扩展)
        
        # 如果找到了成对的 JSON，就只用这些；否则回退到用所有图片（可能会报错）
        if json_count > 0:
            self.image_paths = valid_paths
            print(f"✅ [DataLoader] Found {len(self.image_paths)} images with matching SA-1B JSONs.")
        else:
            print(f"⚠️ [DataLoader] No JSONs found! Assuming images imply masks (not SA-1B format). Found {len(self.image_paths)} images.")

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
                
                # SA-1B JSON 结构通常包含 'annotations' 列表
                anns = data.get('annotations', [])
                # 兼容：有些格式直接就是 list
                if not anns and isinstance(data, list): anns = data
                
                for ann in anns:
                    if 'segmentation' in ann:
                        seg = ann['segmentation']
                        # 情况 A: RLE 格式 (SA-1B 标准)
                        if isinstance(seg, dict) and 'counts' in seg:
                            rle_mask = coco_mask.decode(seg)
                            mask[rle_mask > 0] = 1
                        # 情况 B: Polygon 格式 (points list)
                        elif isinstance(seg, list):
                            for poly in seg:
                                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                                cv2.fillPoly(mask, [pts], 1)
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")
        else:
            # 回退：如果没有 JSON，尝试找 png mask
            mask_path = img_path.replace(".tif", ".png").replace(".jpg", ".png").replace("Images", "Labels") # 简单猜测
            if os.path.exists(mask_path):
                 m_temp = cv2.imread(mask_path, 0)
                 if m_temp is not None: mask = (m_temp > 0).astype(np.uint8)

        # 3. Resize & Tensor
        # 简单 resize
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()
        label_tensor = (label_tensor > 0).float()

        # 4. Rich Text Prompt
        text_prompt = "Cell nuclei"
        if filename in self.prompt_dict:
            info = self.prompt_dict[filename]
            if "rich_text" in info:
                text_prompt = info["rich_text"]
            elif "target_text" in info:
                text_prompt = info["target_text"]

        sample = {
            'image': image_tensor,
            'label': label_tensor,
            'original_size': (self.image_size, self.image_size), # 修正这里，传 resize 后的尺寸给 SAM 通常更稳定，或者传原始尺寸用于后处理
            'name': filename,
            'text_prompt': text_prompt
        }
        
        return sample