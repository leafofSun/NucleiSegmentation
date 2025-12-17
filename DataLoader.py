import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_transforms, get_boxes_from_mask, init_point_sampling
import json
import random


class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None, attribute_info_path=None):
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        
        # ... (Prompt list loading logic remains same) ...
        if prompt_path is None:
            self.prompt_list = {}
            self.prompt_list_type = None
        else:
            loaded_data = json.load(open(prompt_path, "r"))
            if isinstance(loaded_data, dict):
                self.prompt_list = loaded_data
                self.prompt_list_type = "dict"
            else:
                self.prompt_list = {}
                self.prompt_list_type = None

        self.requires_name = requires_name
        self.point_num = point_num
        
        # Load attribute info
        self.attribute_info = {}
        if attribute_info_path is None:
            attribute_info_path = os.path.join(data_path, f'attribute_info_{mode}.json')
        if os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
            except: pass

        if not os.path.isabs(data_path):
            current_dir = os.path.abspath(os.getcwd())
            if 'SAM-Med2D-main' in data_path and 'SAM-Med2D-main' in current_dir:
                parts = data_path.split('SAM-Med2D-main/')
                if len(parts) > 1:
                    data_path = parts[-1]
        
        json_file_path = os.path.join(data_path, f'image2label_{mode}.json')
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Data file not found: {json_file_path}")
        json_file = open(json_file_path, "r")
        dataset = json.load(json_file)
        
        def convert_path(path):
            if isinstance(path, str) and path.startswith('data_demo/'):
                return path.replace('data_demo/', f'{data_path}/')
            elif isinstance(path, str) and path.startswith('cpm17/'):
                return path.replace('cpm17/', f'{data_path}/')
            return path
        
        self.image_paths = []
        self.mask_paths_list = []
        for image_path, mask_list in dataset.items():
            image_path = convert_path(image_path)
            if isinstance(mask_list, list):
                mask_paths = [convert_path(mask_path) for mask_path in mask_list]
                self.image_paths.append(image_path)
                self.mask_paths_list.append(mask_paths)
            else:
                mask_path = convert_path(mask_list)
                self.image_paths.append(image_path)
                self.mask_paths_list.append([mask_path])
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        image_input = {}
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise ValueError(f"Cannot read image: {self.image_paths[index]}")
        
        # === [Fix 1] BGR -> RGB ===
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = (image - self.pixel_mean) / self.pixel_std

        # Read Masks
        mask_paths = self.mask_paths_list[index]
        ori_np_mask_instance = None
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, 0)
            if mask is None: continue
            mask = mask.astype(np.float32)
            if ori_np_mask_instance is None:
                ori_np_mask_instance = mask.copy()
            else:
                if mask.shape != ori_np_mask_instance.shape:
                    mask = cv2.resize(mask, (ori_np_mask_instance.shape[1], ori_np_mask_instance.shape[0]), interpolation=cv2.INTER_NEAREST)
                current_max_id = ori_np_mask_instance.max()
                if current_max_id > 0 and mask.max() > 0:
                    mask[mask > 0] += current_max_id
                ori_np_mask_instance = np.maximum(ori_np_mask_instance, mask)
        
        if ori_np_mask_instance is None:
             # Fallback if no masks found
             h, w = image.shape[:2]
             ori_np_mask_instance = np.zeros((h, w), dtype=np.float32)

        h, w = ori_np_mask_instance.shape
        target_size = self.image_size

        # === [Fix 2] Resize Longest Side (Crucial for SAM performance) ===
        scale = target_size * 1.0 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(ori_np_mask_instance, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # === Padding ===
        pad_h = max(target_size - new_h, 0)
        pad_w = max(target_size - new_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        mask_instance_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).float()
        
        # === Generate Boxes from Padded Mask ===
        if self.prompt_path is None:
            if mask_instance_padded.max() > 0:
                from skimage.measure import label, regionprops
                if mask_instance_padded.max() > 1:
                    label_img = mask_instance_padded.astype(int)
                else:
                    label_img = label(mask_instance_padded)
                
                regions = regionprops(label_img)
                real_boxes = [tuple(region.bbox) for region in regions]
                
                boxes_coord = []
                for box in real_boxes:
                    y0, x0, y1, x1 = box
                    boxes_coord.append([x0, y0, x1, y1])
                
                if len(boxes_coord) == 0:
                    boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                else:
                    boxes = torch.tensor(boxes_coord, dtype=torch.float)
                
                binary_mask_padded = (mask_instance_padded > 0).astype(np.float32)
                point_coords, point_labels = init_point_sampling(binary_mask_padded, self.point_num)
            else:
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                point_coords = torch.tensor([[[0, 0]]], dtype=torch.float)
                point_labels = torch.tensor([[0]], dtype=torch.int)
        else:
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
            point_coords = torch.tensor([[[0, 0]]], dtype=torch.float)
            point_labels = torch.tensor([[0]], dtype=torch.int)

        image_input["image"] = image_tensor
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["label"] = torch.tensor(mask_instance_padded > 0).float().unsqueeze(0)
        image_input["original_size"] = (h, w) # Keep original size for post-processing
        if mask_paths:
            image_input["label_path"] = '/'.join(mask_paths[0].split('/')[:-1])

        # === [Fix 3] Ensure ori_label is always returned ===
        # Even if return_ori_mask is False, for validation/testing we usually need it for metrics
        image_input["ori_label"] = torch.tensor(ori_np_mask_instance).unsqueeze(0)
     
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
        
        # PNuRL Key Matching
        if self.attribute_info:
             file_stem = image_name.split('.')[0]
             attr_info = None
             for key in [file_stem, image_name, self.image_paths[index]]:
                 if key in self.attribute_info:
                     attr_info = self.attribute_info[key]
                     break
             if attr_info and 'attribute_prompts' in attr_info:
                 image_input['attribute_prompts'] = attr_info['attribute_prompts']
        
        return image_input

    def __len__(self):
        return len(self.image_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, attribute_info_path=None):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        
        self.attribute_info = {}
        if attribute_info_path is None:
            attribute_info_path = os.path.join(data_dir, f'attribute_info_{mode}.json')
        if os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
            except: pass

        def convert_path(path):
            if isinstance(path, str):
                if path.startswith('data_demo/'):
                    return path.replace('data_demo/', f'{data_dir}/')
                elif path.startswith('cpm17/'):
                    base_dir = str(data_dir).rstrip('/cpm17').rstrip('\\cpm17')
                    if base_dir == str(data_dir):
                        return os.path.join(data_dir, path.replace('cpm17/', ''))
                    else:
                        return os.path.join(base_dir, path)
                return path
            elif isinstance(path, list):
                return [convert_path(p) for p in path]
            return path
        
        converted_dataset = {convert_path(k): convert_path(v) for k, v in dataset.items()}
        self.image_paths = list(converted_dataset.keys())
        self.label_paths = list(converted_dataset.values())
    
    def __getitem__(self, index):
        image_input = {}
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise ValueError(f"Cannot read image: {self.image_paths[index]}")
        
        # 1. BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - self.pixel_mean) / self.pixel_std
        
        # === [核心修改] 先 Resize 到 1024 (与测试一致) ===
        h, w = image.shape[:2]
        target_size = self.image_size # 训练时传入 1024
        
        # 计算缩放比例 (Resize Longest Side)
        scale = target_size * 1.0 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 放大图像
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. 准备 Masks
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        
        # 更新后的尺寸用于 transforms
        # 注意：这里我们不需要 Padding，因为后面会 RandomCrop
        # 但 RandomCrop 的输入尺寸必须大于 crop_size
        transforms = get_transforms(self.image_size, new_h, new_w, mode='train') 
        
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask is None: continue
            pre_mask = pre_mask.astype(np.float32)
            
            # === [核心修改] Mask 也要跟着 Resize ===
            pre_mask = cv2.resize(pre_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # 3. 随机裁剪 (现在是在放大的图上裁，细胞核变大了！)
            for retry_count in range(10):
                augments = transforms(image=image, mask=pre_mask)
                image_tensor = augments['image']
                mask_instance_crop = augments['mask']
                # ... (后续获取 mask_instance_crop_np 代码不变) ...
                if isinstance(mask_instance_crop, torch.Tensor):
                    mask_instance_crop_np = mask_instance_crop.cpu().numpy()
                else:
                    mask_instance_crop_np = mask_instance_crop
                if mask_instance_crop_np.max() > 0:
                    break
            
            # ... (后续生成 Box/Point 的逻辑完全不变) ...
            from skimage.measure import label, regionprops
            if mask_instance_crop_np.max() > 1:
                label_img = mask_instance_crop_np.astype(int)
            else:
                label_img = label(mask_instance_crop_np)
                
            regions = regionprops(label_img)
            
            if len(regions) > 0:
                target_region = max(regions, key=lambda x: x.area)
                y0, x0, y1, x1 = target_region.bbox
                h_box, w_box = y1-y0, x1-x0
                noise = 0.1
                y0 += random.randint(int(-h_box*noise), int(h_box*noise))
                x0 += random.randint(int(-w_box*noise), int(w_box*noise))
                y1 += random.randint(int(-h_box*noise), int(h_box*noise))
                x1 += random.randint(int(-w_box*noise), int(w_box*noise))
                boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float)
                
                target_id = target_region.label
                target_mask = (label_img == target_id).astype(np.float32)
                point_coords, point_label = init_point_sampling(target_mask, self.point_num)
            else:
                target_mask = np.zeros_like(mask_instance_crop_np, dtype=np.float32)
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                point_coords = torch.zeros((self.point_num, 2), dtype=torch.float)
                point_label = torch.zeros((self.point_num,), dtype=torch.int)

            masks_list.append(torch.tensor(target_mask).long())
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        # ... (后续 stack 和 PNuRL Key 匹配代码不变) ...
        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1).float()
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        
        if self.attribute_info:
            attr_data = None
            file_stem = image_name.split('.')[0]
            for key in [file_stem, image_name, self.image_paths[index]]:
                if key in self.attribute_info:
                    attr_data = self.attribute_info[key]
                    break
            # ... (赋值 attribute_prompts 代码不变) ...
            if attr_data is not None:
                if isinstance(attr_data, dict):
                    if "attribute_prompts" in attr_data:
                        image_input["attribute_prompts"] = attr_data["attribute_prompts"]
                    if "attribute_labels" in attr_data:
                        attr_labels = attr_data["attribute_labels"]
                        if isinstance(attr_labels, list):
                            attr_labels_tensors = []
                            for label in attr_labels:
                                if isinstance(label, list):
                                    attr_labels_tensors.append(torch.tensor(label, dtype=torch.float32))
                                elif isinstance(label, torch.Tensor):
                                    attr_labels_tensors.append(label)
                            if len(attr_labels_tensors) > 0:
                                image_input["attribute_labels"] = attr_labels_tensors
                elif isinstance(attr_data, list):
                    if len(attr_data) > 0 and isinstance(attr_data[0], str):
                        image_input["attribute_prompts"] = attr_data
        
        if self.requires_name:
            image_input["name"] = image_name
            
        return image_input

    def __len__(self):
        return len(self.image_paths)

# ... (stack_dict_batched remains same) ...
def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            if k == 'text_prompts' or k == 'attribute_prompts' or k == 'attribute_labels' or k == 'global_labels':
                out_dict[k] = v
            else:
                out_dict[k] = v
        else:
            if k == 'point_coords' or k == 'point_labels':
                if v.dim() >= 3:
                    out_dict[k] = v.reshape(-1, *v.shape[2:])
                else:
                    out_dict[k] = v
            else:
                out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict
