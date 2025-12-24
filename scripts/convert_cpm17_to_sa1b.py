import os
import json
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm
from pycocotools import mask as mask_utils
try:
    from scipy.io import loadmat
except ImportError:
    print("⚠️ 警告: 需要安装 scipy 来读取 .mat 文件: pip install scipy")
    loadmat = None

# ==============================================================================
# 工具函数
# ==============================================================================
def binary_mask_to_rle(binary_mask):
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def find_images_and_masks_cpm17(root_dir):
    """
    针对 CPM17 数据格式：
    1. 找到所有原图 (Images/image_xx.png)
    2. 找到所有标签 (Labels/image_xx.mat)
    3. 进行配对
    """
    image_map = {}  # {'image_00': 'path/to/image_00.png'}
    mask_map = {}   # {'image_00': 'path/to/image_00.mat'}
    
    # 查找 Images 目录下的图片
    images_dir = os.path.join(root_dir, 'Images')
    if os.path.exists(images_dir):
        for fname in os.listdir(images_dir):
            if (fname.startswith('image_') or fname.startswith('Image_')) and \
               (fname.endswith('.png') or fname.endswith('.tif')):
                stem = os.path.splitext(fname)[0]
                image_map[stem] = os.path.join(images_dir, fname)
    
    # 查找 Labels 目录下的 .mat 文件
    labels_dir = os.path.join(root_dir, 'Labels')
    if os.path.exists(labels_dir):
        for fname in os.listdir(labels_dir):
            if fname.endswith('.mat'):
                stem = os.path.splitext(fname)[0]
                mask_map[stem] = os.path.join(labels_dir, fname)
    
    print(f"   -> 找到 {len(image_map)} 张原图")
    print(f"   -> 找到 {len(mask_map)} 个标签文件")
    
    return image_map, mask_map

def load_mat_instances(mat_path):
    """
    从 .mat 文件中加载实例分割图，并提取所有实例
    返回: list of binary masks (每个实例一个)
    """
    if loadmat is None:
        raise ImportError("需要安装 scipy: pip install scipy")
    
    try:
        data = loadmat(mat_path)
        inst_map = data.get('inst_map', None)
        
        if inst_map is None:
            # 尝试查找其他可能的键
            keys = [k for k in data.keys() if not k.startswith('__')]
            if keys:
                inst_map = data[keys[0]]
            else:
                return []
        
        # 提取所有唯一的实例 ID（排除背景 0）
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        instances = []
        for inst_id in unique_ids:
            binary_mask = (inst_map == inst_id).astype(np.uint8)
            instances.append(binary_mask)
        
        return instances
    except Exception as e:
        print(f"⚠️ 读取 {mat_path} 时出错: {e}")
        return []

def convert_cpm17_recursive(src_root, dst_root):
    print(f"🚀 开始转换 CPM17 -> {dst_root}")
    
    if loadmat is None:
        print("❌ 错误: 需要安装 scipy 来读取 .mat 文件")
        print("   请运行: pip install scipy")
        return
    
    # 分别处理 train 和 test
    for split in ['train', 'test']:
        split_src = os.path.join(src_root, split)
        if not os.path.exists(split_src):
            print(f"⚠️ 跳过 {split}: 路径不存在")
            continue
            
        # 目标路径
        split_dst = os.path.join(dst_root, split)
        os.makedirs(split_dst, exist_ok=True)
        
        print(f"\n📁 处理 {split} 数据集...")
        # === 核心：查找图片和标签 ===
        img_dict, mask_dict = find_images_and_masks_cpm17(split_src)
        
        success_count = 0
        skip_count = 0
        
        # 开始转换
        for stem, img_path in tqdm(img_dict.items(), desc=f"转换 {split}"):
            # 检查是否有对应的 .mat 标签文件
            if stem not in mask_dict:
                skip_count += 1
                continue
            
            mat_path = mask_dict[stem]
            
            # 读取原图
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 无法读取图片: {img_path}")
                continue
            h, w = img.shape[:2]
            
            # 从 .mat 文件中加载所有实例
            instances = load_mat_instances(mat_path)
            if len(instances) == 0:
                skip_count += 1
                continue
            
            annotations = []
            
            # 处理每个实例
            for inst_idx, binary_mask in enumerate(instances):
                # 确保掩码尺寸与图片一致
                if binary_mask.shape[0] != h or binary_mask.shape[1] != w:
                    # 调整掩码尺寸
                    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (binary_mask > 0).astype(np.uint8)
                
                # 提取坐标
                y_inds, x_inds = np.where(binary_mask > 0)
                if len(y_inds) < 3:  # 至少需要3个像素点
                    continue
                
                x1, x2 = int(np.min(x_inds)), int(np.max(x_inds))
                y1, y2 = int(np.min(y_inds)), int(np.max(y_inds))
                
                # 计算面积和 RLE
                rle = binary_mask_to_rle(binary_mask)
                area = int(mask_utils.area(rle))
                
                # 写入标注
                annotations.append({
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "segmentation": rle,
                    "area": area,
                    "iscrowd": 0,
                    "category_id": 1
                })
            
            if len(annotations) > 0:
                # 1. 复制图片到目标文件夹
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(split_dst, filename))
                
                # 2. 保存 JSON
                json_dict = {
                    "image": {"file_name": filename, "height": h, "width": w, "id": stem},
                    "annotations": annotations
                }
                with open(os.path.join(split_dst, stem + ".json"), 'w') as f:
                    json.dump(json_dict, f, indent=2)
                
                success_count += 1
        
        print(f"✅ {split} 转换完成: 成功生成 {success_count} 个样本, 跳过 {skip_count} 个样本")

if __name__ == "__main__":
    SRC_DIR = "data/cpm17"
    DST_DIR = "data/cpm17_SA1B"
    
    # 清空旧数据
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
        
    convert_cpm17_recursive(SRC_DIR, DST_DIR)