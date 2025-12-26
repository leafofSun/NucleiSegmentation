import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import json
import random
# 确保安装: pip install albumentations
import albumentations as A 

def stack_dict_batched(batched_input):
    out_dict = {}
    for k, v in batched_input[0].items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = torch.stack([x[k] for x in batched_input], dim=0)
        else:
            out_dict[k] = [x[k] for x in batched_input]
    return out_dict

class TrainingDataset(data.Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', point_num=1, mask_num=5, requires_name=True, attribute_info_path=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        self.point_num = point_num
        self.mask_num = mask_num
        self.requires_name = requires_name
        
        self.image_paths = []
        self.label_paths = []
        
        # === 1. 文件扫描逻辑 (递归查找) ===
        target_dir = os.path.join(data_dir, mode)
        if not os.path.exists(target_dir):
            target_dir = data_dir 
            
        print(f"🔍 [DataLoader] Scanning in: {target_dir} ...")
        
        json_files = []
        png_mask_files = []
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.json') and 'attribute' not in file:
                    json_files.append(os.path.join(root, file))
                elif file.endswith('.png') and ('mask' in file.lower() or 'label' in file.lower() or 'mask' in root.lower()):
                    png_mask_files.append(os.path.join(root, file))

        # 匹配逻辑
        if len(json_files) > 0:
            print(f"✅ Found {len(json_files)} JSON labels. Matching images...")
            for json_path in json_files:
                base_path = os.path.splitext(json_path)[0]
                # 尝试同目录查找
                found = False
                for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                    if os.path.exists(base_path + ext):
                        self.image_paths.append(base_path + ext)
                        self.label_paths.append(json_path)
                        found = True
                        break
                # 尝试跨目录查找 (../images/)
                if not found:
                    parent = os.path.dirname(os.path.dirname(json_path))
                    fname = os.path.basename(base_path)
                    # 假设图片在 images 文件夹
                    for ext in ['.jpg', '.png', '.tif']:
                        possible_img = os.path.join(parent, 'images', fname + ext)
                        if os.path.exists(possible_img):
                            self.image_paths.append(possible_img)
                            self.label_paths.append(json_path)
                            break
                            
        elif len(png_mask_files) > 0:
            print(f"✅ Found {len(png_mask_files)} PNG masks. Matching images...")
            for mask_path in png_mask_files:
                try:
                    # 简单替换规则：把路径里的 masks 换成 images
                    img_path = mask_path.replace('masks', 'images').replace('labels', 'images').replace('Masks', 'Images')
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(mask_path)
                    else:
                        # 尝试换后缀
                        base_img = os.path.splitext(img_path)[0]
                        for ext in ['.jpg', '.tif', '.png']:
                            if os.path.exists(base_img + ext):
                                self.image_paths.append(base_img + ext)
                                self.label_paths.append(mask_path)
                                break
                except:
                    continue

        if len(self.image_paths) == 0:
            print(f"❌ Error: No valid pairs found in {target_dir}")
        else:
            print(f"🎉 DataLoader ready: {len(self.image_paths)} samples paired.")

        # === 2. 增强策略 (Albumentations) ===
        if self.mode == 'train':
            self.transform = A.Compose([
                # 安全填充：防止图比256小
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                # 随机裁剪：保留高清细节
                A.RandomCrop(width=image_size, height=image_size),
                # 翻转旋转
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 颜色增强
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            ], is_check_shapes=False)
        else:
            # 验证集：使用 CenterCrop 保证有内容且不缩放
            # 如果用 Resize 会导致 Dice 虚低
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.CenterCrop(height=image_size, width=image_size)
            ], is_check_shapes=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取 Mask
        mask = None
        if label_path.endswith('.json'):
            try:
                with open(label_path, 'r') as f:
                    data = json.load(f)
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                anns = data.get('annotations', [])
                if not anns and isinstance(data, list): anns = data
                
                for ann in anns:
                    if 'segmentation' in ann:
                        seg = ann['segmentation']
                        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                            for poly in seg:
                                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                                cv2.fillPoly(mask, [pts], 1)
            except:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(label_path, 0)
            
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        mask = (mask > 0).astype(np.uint8)

        # === 3. [关键] 带重试机制的增强 ===
        # 目的：确保切出来的图里有细胞 (前景)，防止模型只学到全黑背景
        
        max_retries = 10
        image_aug = None
        mask_aug = None
        
        for _ in range(max_retries):
            # 应用变换
            try:
                augmented = self.transform(image=image, mask=mask)
                temp_img = augmented['image']
                temp_mask = augmented['mask']
            except Exception as e:
                # 极端情况回退到 Resize
                resizer = A.Resize(height=self.image_size, width=self.image_size)
                augmented = resizer(image=image, mask=mask)
                temp_img = augmented['image']
                temp_mask = augmented['mask']

            # 判定条件：
            # 1. 如果是验证集，不需要重试，直接接受 CenterCrop
            # 2. 如果是训练集，且 mask 里有东西 (sum > 0)，接受
            # 3. 如果 mask 全黑，重试下一刀
            if self.mode != 'train' or temp_mask.sum() > 0:
                image_aug = temp_img
                mask_aug = temp_mask
                break
        
        # 如果重试 10 次还是全黑 (说明原图可能就没什么细胞)，也只能接受了
        if image_aug is None: 
            image_aug = temp_img
            mask_aug = temp_mask

        # 转 Tensor
        image_tensor = torch.tensor(image_aug).permute(2, 0, 1).float() 
        label_tensor = torch.tensor(mask_aug).unsqueeze(0).float()

        sample = {
            'image': image_tensor,
            'label': label_tensor,
            'original_size': (self.image_size, self.image_size) 
        }
        
        if self.requires_name:
            sample['name'] = os.path.basename(img_path)
            
        return sample