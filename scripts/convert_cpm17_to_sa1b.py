#!/usr/bin/env python3
"""
将CPM17数据集转换为SAM-Med2D模型可以接受的格式，并划分为训练集和验证集。

输出格式：
  output_dir/
    images/          # 图像文件（PNG格式）
    masks/           # 掩码文件（PNG格式，每个实例一个mask）
    image2label_train.json   # 训练集映射：image -> [mask1, mask2, ...]
    image2label_val.json     # 验证集映射：image -> [mask1, mask2, ...]

使用方法:
  python scripts/convert_cpm17_to_sa1b.py --input-dir data/cpm17/train --output-dir data/cpm17 --train-ratio 0.8
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import scipy.io
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="将CPM17数据集转换为SAM-Med2D格式并划分train/val")
    parser.add_argument("--input-dir", type=str, default="data/cpm17/train",
                       help="CPM17训练数据目录（包含Images和Labels文件夹）")
    parser.add_argument("--output-dir", type=str, default="data/cpm17",
                       help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="训练集比例（默认0.8，即80%训练，20%验证）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，用于可重复的数据划分")
    parser.add_argument("--force", action='store_true',
                       help="强制重新生成所有文件（覆盖已存在的文件）")
    return parser.parse_args()


def load_mat_mask(mat_path):
    """
    从.mat文件中加载实例分割mask。
    
    Args:
        mat_path: .mat文件路径
        
    Returns:
        inst_map: numpy数组，包含实例ID的mask
    """
    try:
        mat_data = scipy.io.loadmat(mat_path)
        if 'inst_map' in mat_data:
            inst_map = mat_data['inst_map']
            # 确保是numpy数组
            if isinstance(inst_map, np.ndarray):
                return inst_map.astype(np.uint16)
        # 尝试其他可能的键名
        for key in ['instance_map', 'label', 'mask']:
            if key in mat_data:
                inst_map = mat_data[key]
                if isinstance(inst_map, np.ndarray):
                    return inst_map.astype(np.uint16)
        raise ValueError(f"无法在{mat_path}中找到实例mask数据")
    except Exception as e:
        raise ValueError(f"加载.mat文件失败 {mat_path}: {str(e)}")


def extract_instances_from_mask(inst_map):
    """
    从实例分割mask中提取每个实例的二进制mask。
    
    Args:
        inst_map: 包含实例ID的mask数组
        
    Returns:
        masks: 列表，每个元素是一个二进制mask（numpy数组，0和1）
    """
    masks = []
    unique_ids = np.unique(inst_map)
    # 排除背景（ID=0）
    unique_ids = unique_ids[unique_ids > 0]
    
    for inst_id in unique_ids:
        # 创建二进制mask：当前实例为1，其他为0
        binary_mask = (inst_map == inst_id).astype(np.uint8)
        masks.append(binary_mask)
    
    return masks


def convert_cpm17_dataset(input_dir, output_dir, train_ratio=0.8, seed=42, force=False):
    """
    转换CPM17数据集并划分为训练集和验证集。
    
    Args:
        input_dir: 输入目录（包含Images和Labels文件夹）
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
        force: 是否强制重新生成
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    images_dir = input_dir / 'Images'
    labels_dir = input_dir / 'Labels'
    out_images_dir = output_dir / 'images'
    out_masks_dir = output_dir / 'masks'
    
    # 创建输出目录
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        raise ValueError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"标签目录不存在: {labels_dir}")
    
    # 获取所有图像文件
    image_files = sorted([f for f in images_dir.iterdir() 
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
    
    if len(image_files) == 0:
        raise ValueError(f"在{images_dir}中未找到图像文件")
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 构建图像到标签的映射
    image_to_masks = {}
    all_image_names = []
    
    print("正在转换图像和掩码...")
    for img_path in tqdm(image_files):
        base_name = img_path.stem
        all_image_names.append(base_name)
        
        # 对应的.mat文件
        mat_path = labels_dir / f"{base_name}.mat"
        
        if not mat_path.exists():
            print(f"警告: 未找到对应的标签文件 {mat_path}，跳过图像 {img_path.name}")
            continue
        
        # 读取图像以获取尺寸
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
        except Exception as e:
            print(f"警告: 读取图像失败 {img_path}: {str(e)}，跳过")
            continue
        
        # 保存图像为PNG（如果不存在或强制重新生成）
        out_img_path = out_images_dir / f"{base_name}.png"
        if not out_img_path.exists() or force:
            Image.fromarray(img).save(out_img_path)
        
        # 读取.mat文件并提取实例
        try:
            inst_map = load_mat_mask(mat_path)
            # 确保mask尺寸与图像一致
            if inst_map.shape != (h, w):
                print(f"警告: mask尺寸 {inst_map.shape} 与图像尺寸 {(h, w)} 不匹配，调整mask尺寸")
                inst_map = cv2.resize(inst_map, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 提取每个实例的二进制mask
            instance_masks = extract_instances_from_mask(inst_map)
        except Exception as e:
            print(f"警告: 处理标签文件失败 {mat_path}: {str(e)}，跳过")
            continue
        
        if len(instance_masks) == 0:
            print(f"警告: 图像 {base_name} 没有找到任何实例，跳过")
            continue
        
        # 保存每个实例的mask
        mask_paths = []
        for i, mask in enumerate(instance_masks):
            mask_name = f"{base_name}_mask_{i:04d}.png"
            out_mask_path = out_masks_dir / mask_name
            
            # 保存mask（如果不存在或强制重新生成）
            if not out_mask_path.exists() or force:
                # 保存为二进制PNG（0和255）
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(out_mask_path)
            
            # 使用相对路径（相对于data目录）
            # 格式：cpm17/masks/xxx.png 或 data/cpm17/masks/xxx.png
            mask_rel_path = f"cpm17/masks/{mask_name}"
            mask_paths.append(mask_rel_path)
        
        # 图像相对路径
        img_rel_path = f"cpm17/images/{base_name}.png"
        image_to_masks[img_rel_path] = mask_paths
    
    print(f"成功转换 {len(image_to_masks)} 个图像")
    
    # 划分训练集和验证集
    print(f"正在划分数据集（训练集比例: {train_ratio}）...")
    image_keys = list(image_to_masks.keys())
    
    if len(image_keys) < 2:
        print("警告: 图像数量太少，无法划分数据集，全部用作训练集")
        train_keys = image_keys
        val_keys = []
    else:
        # 使用随机种子确保可重复性
        random.seed(seed)
        np.random.seed(seed)
        
        # 随机打乱
        shuffled_keys = image_keys.copy()
        random.shuffle(shuffled_keys)
        
        # 计算划分点
        split_idx = int(len(shuffled_keys) * train_ratio)
        train_keys = shuffled_keys[:split_idx]
        val_keys = shuffled_keys[split_idx:]
    
    # 构建训练集和验证集的映射
    train_image2label = {k: image_to_masks[k] for k in train_keys}
    test_image2label = {k: image_to_masks[k] for k in val_keys}
    
    # 保存JSON文件
    train_json_path = output_dir / 'image2label_train.json'
    val_json_path = output_dir / 'image2label_test.json'
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_image2label, f, indent=4, ensure_ascii=False)
    
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_image2label, f, indent=4, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"训练集: {len(train_image2label)} 个图像")
    print(f"验证集: {len(test_image2label)} 个图像")
    print(f"输出目录: {output_dir}")
    print(f"训练集JSON: {train_json_path}")
    print(f"验证集JSON: {test_json_path}")


def main():
    args = parse_args()
    convert_cpm17_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        force=args.force
    )


if __name__ == '__main__':
    main()

