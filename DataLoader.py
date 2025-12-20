import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import random

# ===========================================================================================
# [Tools] Correct stack_dict_batched
# ===========================================================================================
def stack_dict_batched(batch):
    """
    Args:
        batch: list of dicts. 
               例如: [{'image': T1, 'label': L1}, {'image': T2, 'label': L2}]
    Returns:
        out_dict: dict of tensors (batched).
               例如: {'image': stack([T1, T2]), 'label': stack([L1, L2])}
    """
    out_dict = {}
    # 遍历第一个样本的所有键
    for k in batch[0].keys():
        # 收集该 Batch 中所有样本的该键对应的值
        values = [sample[k] for sample in batch]
        
        # 如果是 Tensor，尝试堆叠
        if isinstance(values[0], torch.Tensor):
            try:
                out_dict[k] = torch.stack(values, dim=0)
            except RuntimeError:
                # 万一遇到维度不一致无法堆叠的情况（比如变长 Box），则保留 list
                out_dict[k] = values
        else:
            # 如果不是 Tensor（比如文件名是字符串），保留 list
            out_dict[k] = values
            
    return out_dict

class TrainingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, attribute_info_path=None):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        json_path = os.path.join(data_path, f'image2label_{mode}.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
            
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        self.attribute_info = {}
        if attribute_info_path and os.path.exists(attribute_info_path):
            with open(attribute_info_path, 'r') as f:
                self.attribute_info = json.load(f)

        # --- Path Resolution Logic ---
        base_dir = os.path.basename(os.path.normpath(data_path))
        
        def convert_path(path):
            if isinstance(path, str):
                if path.startswith('data_demo/'):
                    path = path.replace('data_demo/', f'{data_path}/')
                elif path.startswith('cpm17/'):
                    if base_dir == str(data_path):
                        path = os.path.join(data_path, path.replace('cpm17/', ''))
                    else:
                        path = os.path.join(data_path, path)
                return path
            return path

        def resolve_image_path(path: str) -> str:
            base = os.path.basename(path)
            candidates = [
                path,
                os.path.join(data_path, path),
                os.path.join(data_path, 'train', 'Images', base),
                os.path.join(data_path, 'Images', base),
            ]
            for cand in candidates:
                if isinstance(cand, str) and os.path.exists(cand):
                    return cand
            return path

        def resolve_mask_path(path: str) -> str:
            base = os.path.basename(path)
            candidates = [
                path,
                os.path.join(data_path, path),
                os.path.join(data_path, 'train', 'Labels', base),
                os.path.join(data_path, 'Labels', base),
            ]
            for cand in candidates:
                if isinstance(cand, str) and os.path.exists(cand):
                    return cand
            return path
        
        converted_dataset = {convert_path(k): convert_path(v) for k, v in dataset.items()}
        self.image_paths = [resolve_image_path(k) for k in converted_dataset.keys()]
        self.label_paths = []
        for v in converted_dataset.values():
            if isinstance(v, list):
                self.label_paths.append([resolve_mask_path(p) for p in v])
            else:
                self.label_paths.append(resolve_mask_path(v))
        print(f"Training Dataset loaded: {len(self.image_paths)} images.")

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self.label_paths[index]
        if isinstance(label_path, list):
            label_path = label_path[0]
        label = cv2.imread(label_path, -1)
        if label is None:
             return self.__getitem__(random.randint(0, len(self.image_paths) - 1))

        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Initialize Tensors
        point_coords = torch.zeros(1, 2).float()
        point_labels = torch.zeros(1).int()
        
        # 默认值 (无目标情况)
        boxes = torch.tensor([[0, 0, 1, 1]]).float() 

        # =================================================================================
        # [核心修复] Generate Box Prompt & Label
        # 逻辑：必须先选定一个 ID，然后针对这同一个 ID 生成 Box 和 Mask
        # =================================================================================
        obj_ids = np.unique(label)
        obj_ids = obj_ids[obj_ids > 0]
        
        if len(obj_ids) > 0:
            # 1. 只随机选一次 ID！
            target_id = np.random.choice(obj_ids)
            
            # 2. 根据这个 ID 生成 Box
            y_indices, x_indices = np.where(label == target_id)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            H, W = label.shape
            # 添加随机扰动 (Jitter) 模拟不完美的提示
            x_min = max(0, x_min - np.random.randint(0, 5))
            x_max = min(W, x_max + np.random.randint(0, 5))
            y_min = max(0, y_min - np.random.randint(0, 5))
            y_max = min(H, y_max + np.random.randint(0, 5))
            
            boxes = torch.tensor([[x_min, y_min, x_max, y_max]]).float()
            
            # 3. 根据同一个 ID 生成 Label (Single Instance Binary Mask)
            # 这一点至关重要：告诉模型“只分割框住的这个细胞”
            label = (label == target_id).astype(np.float32)
        else:
            # 如果没有细胞，Label 全黑
            label = np.zeros_like(label).astype(np.float32)
            
        # =================================================================================
        
        image = (image - self.pixel_mean) / self.pixel_std
        image = torch.tensor(image).permute(2, 0, 1) 
        label = torch.tensor(label).unsqueeze(0) 

        attribute_prompts = []
        if self.attribute_info:
            filename = os.path.basename(image_path)
            info = self.attribute_info.get(filename)
            if info and "prompt" in info:
                attribute_prompts.append(info["prompt"])
        if not attribute_prompts:
            attribute_prompts.append("Medical image")

        sample = {
            "image": image,
            "label": label,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "boxes": boxes,
            "original_size": original_size,
            "image_path": image_path,
            "attribute_prompts": attribute_prompts,
        }
        
        if self.requires_name:
            sample["name"] = os.path.basename(image_path)
            
        return sample

    def __len__(self):
        return len(self.image_paths)

class TestingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None, attribute_info_path=None):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        json_path = os.path.join(data_path, f'image2label_{mode}.json')
        if not os.path.exists(json_path):
             raise FileNotFoundError(f"JSON not found: {json_path}")
             
        with open(json_path, 'r') as f:
            dataset = json.load(f)

        self.attribute_info = {}
        if attribute_info_path and os.path.exists(attribute_info_path):
            with open(attribute_info_path, 'r') as f:
                self.attribute_info = json.load(f)

        def resolve_image_path(path: str) -> str:
            base = os.path.basename(path)
            candidates = [
                path,
                os.path.join(data_path, path),
                os.path.join(data_path, 'test', 'Images', base),
                os.path.join(data_path, 'Images', base),
            ]
            for cand in candidates:
                if isinstance(cand, str) and os.path.exists(cand):
                    return cand
            return path
            
        def resolve_mask_path(path: str) -> str:
            base = os.path.basename(path)
            candidates = [
                path,
                os.path.join(data_path, path),
                os.path.join(data_path, 'test', 'Labels', base),
                os.path.join(data_path, 'Labels', base),
            ]
            for cand in candidates:
                if isinstance(cand, str) and os.path.exists(cand):
                    return cand
            return path
        
        self.image_paths = [resolve_image_path(k) for k in dataset.keys()]
        self.label_paths = [resolve_mask_path(v) for v in dataset.values()]
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
             raise ValueError(f"Cannot read image: {image_path}")
        
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self.label_paths[index]
        label = cv2.imread(label_path, -1)
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # =======================================================
        # [测试集逻辑] 生成所有细胞的 Box，供评估使用
        # =======================================================
        boxes_list = []
        obj_ids = np.unique(label)
        
        for obj_id in obj_ids:
            if obj_id == 0: continue # 跳过背景
            
            y_indices, x_indices = np.where(label == obj_id)
            if len(y_indices) == 0: continue

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            boxes_list.append([x_min, y_min, x_max, y_max])
            
        if len(boxes_list) == 0:
            boxes_tensor = torch.tensor([[0, 0, 1, 1]]).float()
        else:
            boxes_tensor = torch.tensor(boxes_list).float()

        point_coords = torch.zeros(1, 2).float()
        point_labels = torch.zeros(1).int()
        # =======================================================

        image = (image - self.pixel_mean) / self.pixel_std
        image = torch.tensor(image).permute(2, 0, 1)
        
        # 测试集保留原始 Label (Instance ID)，用于计算 mPQ 等指标
        label_tensor = torch.tensor(label).unsqueeze(0).float()

        attribute_prompts = []
        if self.attribute_info:
            filename = os.path.basename(image_path)
            info = self.attribute_info.get(filename)
            if info and "prompt" in info:
                attribute_prompts.append(info["prompt"])
        if not attribute_prompts:
            attribute_prompts.append("Medical image")

        sample = {
            "image": image,
            "label": label_tensor,
            "ori_label": label_tensor, # 关键 Key，供 test.py 使用
            "boxes": boxes_tensor,     # 关键 Key，供 test.py 使用
            "point_coords": point_coords,
            "point_labels": point_labels,
            "original_size": original_size,
            "image_path": image_path,
            "attribute_prompts": attribute_prompts
        }
        
        if self.requires_name:
            sample["name"] = os.path.basename(image_path)
            
        return sample

    def __len__(self):
        return len(self.image_paths)