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
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
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

        json_file = open(os.path.join(data_path, f'image2label_{mode}.json'), "r")
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
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise ValueError(f"无法读取图像文件: {self.image_paths[index]}")
        
        image = (image - self.pixel_mean) / self.pixel_std

        # 读取该图片的所有mask并合并
        mask_paths = self.mask_paths_list[index]
        ori_np_mask = None
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise ValueError(f"无法读取掩码文件: {mask_path}")
            
            if mask.max() == 255:
                mask = mask / 255
            
            # 合并所有mask（使用逻辑或操作）
            if ori_np_mask is None:
                ori_np_mask = mask.copy()
            else:
                # 确保尺寸一致
                if mask.shape != ori_np_mask.shape:
                    mask = cv2.resize(mask, (ori_np_mask.shape[1], ori_np_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                ori_np_mask = np.maximum(ori_np_mask, mask)  # 合并mask（逻辑或）
        
        if ori_np_mask is None:
            raise ValueError(f"没有找到有效的掩码文件")

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. Image: {self.image_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, max_pixel = 0)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            # 使用第一张图片的路径作为prompt key
            prompt_key = self.image_paths[index].split('/')[-1]
            # 移除可能的扩展名
            prompt_key_no_ext = prompt_key.split('.')[0] if '.' in prompt_key else prompt_key
            
            # 尝试查找对应的 prompt 数据
            prompt_data = None
            if prompt_key_no_ext in self.prompt_list:
                prompt_data = self.prompt_list[prompt_key_no_ext]
            elif prompt_key in self.prompt_list:
                prompt_data = self.prompt_list[prompt_key]
            
            # 检查是否有 boxes、point_coords、point_labels 字段
            if prompt_data is not None and "boxes" in prompt_data and "point_coords" in prompt_data and "point_labels" in prompt_data:
                # 使用 JSON 文件中的 prompt 数据
                boxes = torch.as_tensor(prompt_data["boxes"], dtype=torch.float)
                point_coords = torch.as_tensor(prompt_data["point_coords"], dtype=torch.float)
                point_labels = torch.as_tensor(prompt_data["point_labels"], dtype=torch.int)
            else:
                # JSON 文件中没有 prompt 数据，从 mask 生成
                # 这通常发生在 JSON 文件只包含属性信息（如 global_label）时
                boxes = get_boxes_from_mask(mask, max_pixel = 0)
                point_coords, point_labels = init_point_sampling(mask, self.point_num)
                if prompt_data is None:
                    print(f"警告: 在 prompt_list 中未找到图片 {prompt_key} 的 prompt 数据，将从 mask 生成")
                else:
                    print(f"警告: 图片 {prompt_key} 的 prompt 数据缺少必要字段（boxes/point_coords/point_labels），将从 mask 生成")

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        # 使用第一张mask的路径作为label_path
        if mask_paths:
            image_input["label_path"] = '/'.join(mask_paths[0].split('/')[:-1])

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
        
        # 添加属性信息（如果可用，用于PNuRL）
        if self.attribute_info:
            # 使用图片路径或图片名作为key查找属性信息
            label_key = self.image_paths[index]
            # 也尝试使用文件名作为key
            if label_key not in self.attribute_info:
                label_key = image_name
            
            if label_key in self.attribute_info:
                attr_info = self.attribute_info[label_key]
                # 测试时只需要attribute_prompts，不需要attribute_labels
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
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask is None:
                raise ValueError(f"无法读取掩码文件: {m}")
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
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
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data/data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

