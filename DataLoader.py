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
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
            attribute_info_path (str, optional): Path to JSON file containing attribute_prompts for PNuRL. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        # 加载 prompt_list，可能是字典或列表格式
        if prompt_path is None:
            self.prompt_list = {}
            self.prompt_list_type = None
        else:
            loaded_data = json.load(open(prompt_path, "r"))
            if isinstance(loaded_data, dict):
                # 字典格式：{"image_name": {"boxes": [...], "point_coords": [...], ...}}
                self.prompt_list = loaded_data
                self.prompt_list_type = "dict"
            elif isinstance(loaded_data, list):
                # 列表格式：[{"id": "image_name", "boxes": [...], ...}, ...]
                # 转换为字典格式以便快速查找
                self.prompt_list = {}
                self.prompt_list_type = "list"
                for item in loaded_data:
                    # 支持多种可能的键名
                    if "id" in item:
                        # id 可能是列表或字符串
                        ids = item["id"] if isinstance(item["id"], list) else [item["id"]]
                        for img_id in ids:
                            # 移除可能的扩展名
                            img_id = img_id.split('.')[0] if '.' in img_id else img_id
                            self.prompt_list[img_id] = item
                    elif "image_name" in item:
                        img_id = item["image_name"]
                        img_id = img_id.split('.')[0] if '.' in img_id else img_id
                        self.prompt_list[img_id] = item
            else:
                self.prompt_list = {}
                self.prompt_list_type = None
                print(f"警告: prompt_path 文件格式不支持: {type(loaded_data)}")
        self.requires_name = requires_name
        self.point_num = point_num
        
        # 加载属性信息（用于PNuRL）
        self.attribute_info = {}
        if attribute_info_path is None:
            attribute_info_path = os.path.join(data_path, f'attribute_info_{mode}.json')
        if os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
                print(f"加载测试属性信息从: {attribute_info_path}")
            except Exception as e:
                print(f"警告: 无法加载属性信息文件 {attribute_info_path}: {e}")
        elif attribute_info_path and os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
                print(f"加载测试属性信息从: {attribute_info_path}")
            except Exception as e:
                print(f"警告: 无法加载属性信息文件 {attribute_info_path}: {e}")

        # 处理 data_path：如果是相对路径且包含项目名，需要修正
        # 例如：如果 data_path 是 "SAM-Med2D-main/data/cpm17"，但当前目录已经是 SAM-Med2D-main
        # 则需要去掉重复的 "SAM-Med2D-main" 前缀
        if not os.path.isabs(data_path):
            # 获取当前工作目录的绝对路径
            current_dir = os.path.abspath(os.getcwd())
            # 如果 data_path 以项目名开头，但当前目录已经包含项目名，则去掉重复部分
            if 'SAM-Med2D-main' in data_path and 'SAM-Med2D-main' in current_dir:
                # 找到 data_path 中 "SAM-Med2D-main" 之后的部分
                parts = data_path.split('SAM-Med2D-main/')
                if len(parts) > 1:
                    data_path = parts[-1]  # 取最后一部分，例如 "data/cpm17"
        
        json_file_path = os.path.join(data_path, f'image2label_{mode}.json')
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(
                f"找不到数据文件: {json_file_path}\n"
                f"请检查 data_path 是否正确。当前 data_path: {data_path}\n"
                f"当前工作目录: {os.getcwd()}\n"
                f"尝试的完整路径: {os.path.abspath(json_file_path)}"
            )
        json_file = open(json_file_path, "r")
        dataset = json.load(json_file)
        
        # 将 JSON 中的路径从 data_demo 转换为实际的数据目录路径
        def convert_path(path):
            if isinstance(path, str) and path.startswith('data_demo/'):
                return path.replace('data_demo/', f'{data_path}/')
            elif isinstance(path, str) and path.startswith('cpm17/'):
                # 处理cpm17路径，确保使用绝对路径
                return path.replace('cpm17/', f'{data_path}/')
            return path
        
        # image2label格式：{"image_path": [mask_path1, mask_path2, ...]}
        # 对于测试，按图片分组，每张图片的所有掩码合并成一个mask进行评估
        self.image_paths = []
        self.mask_paths_list = []  # 每张图片对应的所有mask路径列表
        for image_path, mask_list in dataset.items():
            image_path = convert_path(image_path)
            # 如果mask_list是列表，保存所有mask路径
            if isinstance(mask_list, list):
                mask_paths = [convert_path(mask_path) for mask_path in mask_list]
                self.image_paths.append(image_path)
                self.mask_paths_list.append(mask_paths)
            else:
                # 如果mask_list是单个路径（向后兼容）
                mask_path = convert_path(mask_list)
                self.image_paths.append(image_path)
                self.mask_paths_list.append([mask_path])
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        image_input = {}
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise ValueError(f"无法读取图像文件: {self.image_paths[index]}")
        
        image = (image - self.pixel_mean) / self.pixel_std

        # 1. 读取并合并实例 Mask
        mask_paths = self.mask_paths_list[index]
        ori_np_mask_instance = None
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise ValueError(f"无法读取: {mask_path}")
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
            raise ValueError(f"没有找到有效的掩码文件")
        
        h, w = ori_np_mask_instance.shape
        target_size = self.image_size  # 1024

        # === 【关键修复】手动 Padding (替代 albumentations) ===
        # 使用与 test.py 中 postprocess_masks 完全一致的计算逻辑
        pad_h = max(target_size - h, 0)
        pad_w = max(target_size - w, 0)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # 使用 OpenCV 进行补零填充
        image_padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, 
                                          cv2.BORDER_CONSTANT, value=0)
        mask_instance_padded = cv2.copyMakeBorder(ori_np_mask_instance, pad_top, pad_bottom, pad_left, pad_right, 
                                                  cv2.BORDER_CONSTANT, value=0)

        # 转换为 Tensor: HWC -> CHW
        image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).float()
        
        # 3. 基于 Padded Mask 生成 Box
        if self.prompt_path is None:
            if mask_instance_padded.max() > 0:
                # 传入 Padded Mask
                boxes = get_boxes_from_mask(mask_instance_padded, box_num=500, max_pixel=0)
                
                binary_mask_padded = (mask_instance_padded > 0).astype(np.float32)
                point_coords, point_labels = init_point_sampling(binary_mask_padded, self.point_num)
            else:
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                point_coords = torch.tensor([[[0, 0]]], dtype=torch.float)
                point_labels = torch.tensor([[0]], dtype=torch.int)
        else:
            # 如果使用 json prompt，这里需要非常小心，暂不处理 padding 偏移
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
            point_coords = torch.tensor([[[0, 0]]], dtype=torch.float)
            point_labels = torch.tensor([[0]], dtype=torch.int)

        # 4. 组装输入
        image_input["image"] = image_tensor
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["label"] = torch.tensor(mask_instance_padded > 0).float().unsqueeze(0)
        image_input["original_size"] = (h, w)
        if mask_paths:
            image_input["label_path"] = '/'.join(mask_paths[0].split('/')[:-1])

        # 5. 返回原始尺寸的实例 Mask (用于 metrics 计算)
        if self.return_ori_mask:
            image_input["ori_label"] = torch.tensor(ori_np_mask_instance).unsqueeze(0)
     
        if self.requires_name:
            image_input["name"] = self.image_paths[index].split('/')[-1]
        
        if self.attribute_info:
            label_key = self.image_paths[index]
            if label_key not in self.attribute_info:
                label_key = image_input.get("name", "")
            if label_key in self.attribute_info:
                attr_info = self.attribute_info[label_key]
                if 'attribute_prompts' in attr_info:
                    image_input['attribute_prompts'] = attr_info['attribute_prompts']
        
        return image_input

    def __len__(self):
        return len(self.image_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, 
                 attribute_info_path=None):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
            attribute_info_path (str, optional): Path to JSON file containing attribute_prompts and attribute_labels for PNuRL. 
                If None, will try to load from data_dir/attribute_info_{mode}.json. Defaults to None.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        
        # 加载属性信息（用于PNuRL）
        self.attribute_info = {}
        if attribute_info_path is None:
            attribute_info_path = os.path.join(data_dir, f'attribute_info_{mode}.json')
        if os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
                print(f"加载属性信息从: {attribute_info_path}")
            except Exception as e:
                print(f"警告: 无法加载属性信息文件 {attribute_info_path}: {e}")
                self.attribute_info = {}
        elif attribute_info_path and os.path.exists(attribute_info_path):
            try:
                self.attribute_info = json.load(open(attribute_info_path, "r"))
                print(f"加载属性信息从: {attribute_info_path}")
            except Exception as e:
                print(f"警告: 无法加载属性信息文件 {attribute_info_path}: {e}")
                self.attribute_info = {}
        # 将 JSON 中的路径转换为实际的数据目录路径
        # JSON 中的路径格式可能是 "data_demo/..." 或 "cpm17/..."，需要转换为绝对路径
        def convert_path(path):
            if isinstance(path, str):
                # 处理 data_demo/ 格式
                if path.startswith('data_demo/'):
                    return path.replace('data_demo/', f'{data_dir}/')
                # 处理 cpm17/ 格式
                elif path.startswith('cpm17/'):
                    # data_dir 应该是 data/cpm17，所以需要去掉末尾的 cpm17
                    base_dir = str(data_dir).rstrip('/cpm17').rstrip('\\cpm17')
                    if base_dir == str(data_dir):
                        # 如果data_dir就是cpm17目录，直接使用
                        return os.path.join(data_dir, path.replace('cpm17/', ''))
                    else:
                        return os.path.join(base_dir, path)
                # 如果已经是绝对路径或相对路径，直接返回
                return path
            elif isinstance(path, list):
                return [convert_path(p) for p in path]
            return path
        
        converted_dataset = {convert_path(k): convert_path(v) for k, v in dataset.items()}
        self.image_paths = list(converted_dataset.keys())
        self.label_paths = list(converted_dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise ValueError(f"无法读取图像文件: {self.image_paths[index]}")
        
        image = (image - self.pixel_mean) / self.pixel_std
        h, w, _ = image.shape
        # 训练模式：使用随机裁剪（模仿 PromptNu）
        transforms = get_transforms(self.image_size, h, w, mode='train')
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        
        # 随机选择 mask_num 个样本进行训练
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask is None:
                raise ValueError(f"无法读取掩码文件: {m}")
            pre_mask = pre_mask.astype(np.float32)

            # 1. 随机裁剪并尝试找到有细胞的区域
            for retry_count in range(10):
                augments = transforms(image=image, mask=pre_mask)
                image_tensor = augments['image']
                mask_instance_crop = augments['mask'] # 裁剪后的实例图
                
                # 转换为 numpy
                if isinstance(mask_instance_crop, torch.Tensor):
                    mask_instance_crop_np = mask_instance_crop.cpu().numpy()
                else:
                    mask_instance_crop_np = mask_instance_crop
                
                if mask_instance_crop_np.max() > 0:
                    break
            
            # 2. 【核心修复】分离目标实例
            # 我们不能简单地用 mask > 0 作为标签，必须选出一个特定的实例
            from skimage.measure import label, regionprops
            
            # 获取所有连通域
            if mask_instance_crop_np.max() > 1:
                label_img = mask_instance_crop_np.astype(int)
            else:
                label_img = label(mask_instance_crop_np)
                
            regions = regionprops(label_img)
            
            if len(regions) > 0:
                # 策略：随机选一个实例，或者选最大的
                # 这里为了稳定，选最大的那个实例作为本次的训练目标
                target_region = max(regions, key=lambda x: x.area)
                
                # A. 生成该实例的 Box
                y0, x0, y1, x1 = target_region.bbox
                # 添加一点噪声模拟真实提示
                h_box, w_box = y1-y0, x1-x0
                noise = 0.1
                y0 += random.randint(int(-h_box*noise), int(h_box*noise))
                x0 += random.randint(int(-w_box*noise), int(w_box*noise))
                y1 += random.randint(int(-h_box*noise), int(h_box*noise))
                x1 += random.randint(int(-w_box*noise), int(w_box*noise))
                boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float)
                
                # B. 【关键】生成只包含该实例的 Mask
                # 只有 ID 等于目标实例 ID 的像素才设为 1，其他设为 0
                target_id = target_region.label
                target_mask = (label_img == target_id).astype(np.float32)
                
                # Point 同理，基于 target_mask 生成
                point_coords, point_label = init_point_sampling(target_mask, self.point_num)
                # 确保 point_coords 形状为 [num_points, 2]（SAM 期望的格式，不需要额外的 batch 维度）
                # init_point_sampling 返回的格式：
                # - point_num == 1: [1, 2] 和 [1]
                # - point_num > 1: [num_points, 2] 和 [num_points]
                # 这些格式已经是 SAM 期望的格式，不需要修改
                
            else:
                # 兜底：没有实例
                target_mask = np.zeros_like(mask_instance_crop_np, dtype=np.float32)
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                # 确保形状与正常情况一致：[num_points, 2] 和 [num_points]（SAM 期望的格式）
                point_coords = torch.zeros((self.point_num, 2), dtype=torch.float)
                point_label = torch.zeros((self.point_num,), dtype=torch.int)

            # 3. 存入列表
            masks_list.append(torch.tensor(target_mask).long()) # 现在 mask 只有这一个细胞了！
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        # label 增加 channel 维度: [mask_num, 1, H, W]
        image_input["label"] = mask.unsqueeze(1).float()
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        # 添加属性信息（用于PNuRL，如果可用）
        image_name = self.image_paths[index].split('/')[-1]
        image_key = self.image_paths[index]  # 使用完整路径作为key
        
        # 尝试从attribute_info中获取属性信息
        if self.attribute_info:
            # 尝试多种key格式
            attr_data = None
            for key_format in [image_key, image_name, os.path.basename(image_key)]:
                if key_format in self.attribute_info:
                    attr_data = self.attribute_info[key_format]
                    break
            
            if attr_data is not None:
                # 支持PNuRL格式: {"attribute_prompts": [...], "attribute_labels": [[...], [...], ...]}
                if isinstance(attr_data, dict):
                    # 属性提示词（5个属性：颜色、形状、排列、大小、分布）
                    if "attribute_prompts" in attr_data:
                        image_input["attribute_prompts"] = attr_data["attribute_prompts"]
                    
                    # 属性标签（5个属性的标签列表）
                    if "attribute_labels" in attr_data:
                        attr_labels = attr_data["attribute_labels"]
                        # 转换为tensor列表
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
                    # 如果直接是列表，假设是attribute_prompts
                    if len(attr_data) > 0 and isinstance(attr_data[0], str):
                        image_input["attribute_prompts"] = attr_data
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            # 对于文本提示、属性提示和全局标签，保持列表格式
            if k == 'text_prompts' or k == 'attribute_prompts':
                # 如果每个样本都有文本提示列表，合并或保持
                out_dict[k] = v
            elif k == 'attribute_labels':
                # attribute_labels 是列表的列表，每个元素是一个属性的标签列表
                # 保持列表格式，不进行stack
                out_dict[k] = v
            elif k == 'global_labels':
                # 如果是全局标签列表，尝试stack成tensor
                if len(v) > 0 and isinstance(v[0], torch.Tensor):
                    try:
                        out_dict[k] = torch.stack(v, dim=0)
                    except:
                        out_dict[k] = v
                else:
                    out_dict[k] = v
            else:
                out_dict[k] = v
        else:
            # 特殊处理 point_coords 和 point_labels
            # 它们的形状应该是 [batch_size, mask_num, num_points, 2] 和 [batch_size, mask_num, num_points]
            # 需要 reshape 成 [batch_size * mask_num, num_points, 2] 和 [batch_size * mask_num, num_points]
            if k == 'point_coords' or k == 'point_labels':
                # 如果已经是正确的形状 [batch_size, mask_num, num_points, ...]，直接 reshape
                if v.dim() >= 3:
                    out_dict[k] = v.reshape(-1, *v.shape[2:])
                else:
                    out_dict[k] = v
            else:
                out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data/data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

