import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import random
import glob

# 尝试导入 pycocotools
try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

# ===========================================================================================
# [Tools] Correct stack_dict_batched
# ===========================================================================================
def stack_dict_batched(batch):
    out_dict = {}
    for k in batch[0].keys():
        values = [sample[k] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                out_dict[k] = torch.stack(values, dim=0)
            except RuntimeError:
                out_dict[k] = values
        else:
            out_dict[k] = values
    return out_dict

# ===========================================================================================
# Training Dataset (双模式支持)
# ===========================================================================================
class TrainingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, attribute_info_path=None):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        # 移除 DataLoader 中的 Normalization 参数，交给模型处理
        # self.pixel_mean = ... (Removed)
        # self.pixel_std = ... (Removed)
        self.data_path = data_path
        
        self.attribute_info = {}
        if attribute_info_path and os.path.exists(attribute_info_path):
            with open(attribute_info_path, 'r') as f:
                self.attribute_info = json.load(f)

        # 1. 优先检查旧版 image2label
        json_map_path = os.path.join(data_path, f'image2label_{mode}.json')
        
        if os.path.exists(json_map_path):
            self.dataset_type = 'legacy'
            print(f"✅ [Training] Found map file: {os.path.basename(json_map_path)}")
            with open(json_map_path, 'r') as f:
                dataset = json.load(f)
            self.image_paths, self.label_paths = self._parse_legacy_paths(dataset, data_path)
        else:
            # 2. 否则进入 SA-1B 模式
            search_path = os.path.join(data_path, mode) if os.path.exists(os.path.join(data_path, mode)) else data_path
            self.json_files = sorted(glob.glob(os.path.join(search_path, '**', '*.json'), recursive=True))
            self.json_files = [f for f in self.json_files if "image2label" not in os.path.basename(f)]
            
            if len(self.json_files) > 0:
                self.dataset_type = 'sa1b'
                print(f"✅ [Training] Found {len(self.json_files)} SA-1B JSONs in {search_path}")
            else:
                raise FileNotFoundError(f"❌ No training data found in {data_path}")

    def _parse_legacy_paths(self, dataset, data_path):
        img_paths, lbl_paths = [], []
        for k, v in dataset.items():
            img_p = os.path.join(data_path, k) if not os.path.exists(k) else k
            if isinstance(v, list): lbl_p = [os.path.join(data_path, i) if not os.path.exists(i) else i for i in v]
            else: lbl_p = os.path.join(data_path, v) if not os.path.exists(v) else v
            img_paths.append(img_p)
            lbl_paths.append(lbl_p)
        return img_paths, lbl_paths

    def __len__(self):
        return len(self.image_paths) if self.dataset_type == 'legacy' else len(self.json_files)

    def __getitem__(self, index):
        if self.dataset_type == 'legacy':
            return self._getitem_legacy(index)
        else:
            return self._getitem_sa1b(index)

    def _getitem_legacy(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_path = self.label_paths[index]
        if isinstance(label_path, list): label_path = label_path[0]
        label = cv2.imread(label_path, -1)
        
        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        obj_ids = np.unique(label)
        obj_ids = obj_ids[obj_ids > 0]
        boxes = torch.tensor([[0, 0, 1, 1]]).float()
        if len(obj_ids) > 0:
            target_id = np.random.choice(obj_ids)
            ys, xs = np.where(label == target_id)
            x1, x2, y1, y2 = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
            boxes = torch.tensor([[x1, y1, x2, y2]]).float()
            label = (label == target_id).astype(np.float32)
        else:
            label = np.zeros_like(label).astype(np.float32)
            
        return self._pack(image, label, boxes, original_size, image_path)

    def _getitem_sa1b(self, index):
        json_path = self.json_files[index]
        base_path = os.path.splitext(json_path)[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif']:
            if os.path.exists(base_path + ext): img_path = base_path + ext; break
        
        if not img_path: return self.__getitem__(random.randint(0, len(self)-1))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        with open(json_path, 'r') as f: data = json.load(f)
        anns = data.get('annotations', [])
        
        label = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        boxes = torch.tensor([[0, 0, 1, 1]]).float()
        
        if len(anns) > 0:
            ann = random.choice(anns)
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                sx, sy = self.image_size/ori_w, self.image_size/ori_h
                boxes = torch.tensor([[x*sx, y*sy, (x+w)*sx, (y+h)*sy]]).float()
            
            if 'segmentation' in ann and mask_utils:
                try:
                    mask = mask_utils.decode(ann['segmentation'])
                    label = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                except:
                    x1, y1, x2, y2 = boxes[0].int().tolist()
                    label[y1:y2, x1:x2] = 1.0
            elif 'bbox' in ann:
                x1, y1, x2, y2 = boxes[0].int().tolist()
                label[y1:y2, x1:x2] = 1.0

        return self._pack(image, label, boxes, (ori_h, ori_w), img_path)

    def _pack(self, image, label, boxes, ori_size, path):
        # [核心修复] 移除这里的归一化！SAM 模型内部会自己做。
        # image = (image - self.pixel_mean) / self.pixel_std  <-- 删除这行
        
        # 保持 0-255 范围，仅转为 Float Tensor
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        if not isinstance(label, torch.Tensor): label = torch.tensor(label).float().unsqueeze(0)
        
        sample = {
            "image": image, "label": label, "boxes": boxes, 
            "point_coords": torch.zeros(1,2), "point_labels": torch.zeros(1),
            "original_size": ori_size, "image_path": path,
            "attribute_prompts": ["Medical image"]
        }
        if self.requires_name: sample["name"] = os.path.basename(path)
        return sample


# ===========================================================================================
# Testing Dataset (关键修正版)
# ===========================================================================================
class TestingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None, attribute_info_path=None):
        self.image_size = image_size
        self.requires_name = requires_name
        # 移除 Testing 中的 Normalization 参数
        self.data_path = data_path
        
        # 1. 尝试寻找 Legacy JSON
        json_path = os.path.join(data_path, f'image2label_{mode}.json')
        
        if os.path.exists(json_path):
            self.dataset_type = 'legacy'
            print(f"✅ [Testing] Found legacy map: {json_path}")
            with open(json_path, 'r') as f: dataset = json.load(f)
            self.image_paths = [os.path.join(data_path, k) if not os.path.exists(k) else k for k in dataset.keys()]
            self.label_paths = [os.path.join(data_path, v) if not os.path.exists(v) else v for v in dataset.values()]
        else:
            # 2. 尝试 SA-1B 模式
            search_path = os.path.join(data_path, mode)
            if not os.path.exists(search_path): 
                search_path = data_path
            
            self.json_files = sorted(glob.glob(os.path.join(search_path, '**', '*.json'), recursive=True))
            self.json_files = [f for f in self.json_files if "image2label" not in f]
            
            if len(self.json_files) > 0:
                self.dataset_type = 'sa1b'
                print(f"✅ [Testing] Found {len(self.json_files)} SA-1B JSONs in {search_path}")
            else:
                self.dataset_type = 'legacy_folder'
                img_dir = os.path.join(data_path, 'Images')
                if not os.path.exists(img_dir): img_dir = data_path
                self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.tif')))
                self.label_paths = [] 

    def __len__(self):
        if self.dataset_type == 'sa1b': return len(self.json_files)
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.dataset_type == 'sa1b':
            return self._getitem_sa1b(index)
        else:
            return self._getitem_legacy(index)

    def _getitem_sa1b(self, index):
        json_path = self.json_files[index]
        base_path = os.path.splitext(json_path)[0]
        
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif']:
            if os.path.exists(base_path + ext): img_path = base_path + ext; break
        
        if not img_path: return self.__getitem__(random.randint(0, len(self) - 1))
        
        image = cv2.imread(img_path)
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        try:
            with open(json_path, 'r') as f: data = json.load(f)
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))

        anns = data.get('annotations', [])
        gt_mask = np.zeros(original_size, dtype=np.int32)
        boxes_list = []
        
        for idx, ann in enumerate(anns):
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                boxes_list.append([x, y, x+w, y+h])

            if 'segmentation' in ann and mask_utils:
                try:
                    mask = mask_utils.decode(ann['segmentation'])
                    gt_mask[mask > 0] = (idx + 1)
                except:
                    pass
        
        label_resized = cv2.resize(gt_mask.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if len(boxes_list) == 0:
            boxes_tensor = torch.tensor([[0,0,1,1]]).float()
        else:
            boxes_tensor = torch.tensor(boxes_list).float()
            sx, sy = self.image_size/original_size[1], self.image_size/original_size[0]
            boxes_tensor[:, 0::2] *= sx
            boxes_tensor[:, 1::2] *= sy

        return self._pack(image_resized, label_resized, boxes_tensor, original_size, img_path)

    def _getitem_legacy(self, index):
        img_path = self.image_paths[index]
        lbl_path = self.label_paths[index] if index < len(self.label_paths) else None
        
        image = cv2.imread(img_path)
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        if lbl_path and os.path.exists(lbl_path):
            label = cv2.imread(lbl_path, -1)
            label_resized = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            label_resized = np.zeros((self.image_size, self.image_size))
            
        boxes_list = []
        if np.max(label_resized) > 0:
            ids = np.unique(label_resized)
            for i in ids:
                if i==0: continue
                ys, xs = np.where(label_resized == i)
                boxes_list.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
        
        boxes_tensor = torch.tensor(boxes_list).float() if boxes_list else torch.tensor([[0,0,1,1]]).float()
        return self._pack(image_resized, label_resized, boxes_tensor, original_size, img_path)

    def _pack(self, image, label, boxes, ori_size, path):
        # [核心修复] 移除这里的归一化！
        # 保持 0-255，让 SAM 模型内部去处理
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        label_tensor = torch.tensor(label).unsqueeze(0).float()
        
        sample = {
            "image": image,
            "label": label_tensor,
            "ori_label": label_tensor,
            "boxes": boxes, 
            "point_coords": torch.zeros(1,2),
            "point_labels": torch.zeros(1),
            "original_size": ori_size,
            "image_path": path,
            "attribute_prompts": ["Medical image"]
        }
        if self.requires_name: sample["name"] = os.path.basename(path)
        return sample