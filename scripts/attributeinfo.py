import json
import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm

# === 配置路径 ===
DATA_ROOT = "data/MoNuSeg"
TEST_JSON_PATH = os.path.join(DATA_ROOT, "image2label_test.json")
OUTPUT_JSON_PATH = os.path.join(DATA_ROOT, "attribute_info_test.json")

# === 属性映射规则 (根据 CPM17 数据集特性调整) ===
# 1. 颜色 (Color) - 通常 H&E 染色都是深紫色
COLORS = ["deep purple", "light pink", "purple"]
# 2. 形状 (Shape) - 细胞核通常是椭圆
SHAPES = ["elliptical/oval", "round", "irregular", "spindle", "elongated"]
# 3. 排列 (Arrangement)
ARRANGEMENTS = ["scattered", "clustered", "linear", "regular"]
# 4. 大小 (Size) - 根据平均像素面积判断
SIZES = ["small", "medium", "large"]
# 5. 分布 (Density) - 根据细胞数量判断
DENSITIES = ["sparsely distributed", "moderately dense", "densely packed"]

def get_attributes_from_mask(mask, image=None):
    """
    输入: Mask (numpy array)
    输出: 属性文本列表 ["color", "shape", "arrange", "size", "density"]
    """
    # 确保是实例 Mask
    if mask.max() <= 1:
        label_img = label(mask)
    else:
        label_img = mask.astype(int)
        
    regions = regionprops(label_img)
    num_cells = len(regions)
    
    # --- 1. 统计属性计算 ---
    if num_cells > 0:
        avg_area = np.mean([r.area for r in regions])
    else:
        avg_area = 0
        
    # --- 2. 规则映射 (Thresholds) ---
    
    # A. 大小 (Size)
    # CPM17 是 500x500 或 1024x1024，这里假设原图尺度
    if avg_area < 250:
        size_prompt = "small"
        size_label = [1, 0, 0]
    elif avg_area < 550:
        size_prompt = "medium"
        size_label = [0, 1, 0]
    else:
        size_prompt = "large"
        size_label = [0, 0, 1]
        
    # B. 密度 (Density)
    if num_cells < 30:
        density_prompt = "sparsely distributed"
        density_label = [1, 0, 0]
    elif num_cells < 100:
        density_prompt = "moderately dense"
        density_label = [0, 1, 0]
    else:
        density_prompt = "densely packed"
        density_label = [0, 0, 1]
        
    # C. 其他属性 (简化处理，设为默认高频词)
    # 颜色：默认为 Deep Purple (CPM17特性)
    color_prompt = "deep purple"
    color_label = [1, 0, 0] 
    
    # 形状：默认为 Elliptical/Oval
    shape_prompt = "elliptical/oval"
    shape_label = [1, 0, 0, 0, 0]
    
    # 排列：默认为 Scattered
    arrange_prompt = "scattered"
    arrange_label = [1, 0, 0, 0]
    
    # --- 3. 组装输出 ---
    # 必须严格按照 [Color, Shape, Arrangement, Size, Distribution] 顺序
    prompts = [color_prompt, shape_prompt, arrange_prompt, size_prompt, density_prompt]
    labels = [color_label, shape_label, arrange_label, size_label, density_label]
    
    return prompts, labels

def main():
    if not os.path.exists(TEST_JSON_PATH):
        print(f"Error: 找不到测试集文件 {TEST_JSON_PATH}")
        return

    print(f"正在读取测试集索引: {TEST_JSON_PATH}")
    with open(TEST_JSON_PATH, 'r') as f:
        test_data = json.load(f)
    
    attribute_info = {}
    
    # 遍历每张图片
    for img_path, mask_list in tqdm(test_data.items(), desc="Generating Attributes"):
        # 处理路径 (去除 data_demo 前缀等，根据你的 DataLoader 逻辑调整)
        # 这里假设 mask_path 是相对于项目根目录的
        if isinstance(mask_list, list):
            mask_path = mask_list[0] # 取第一个mask路径，或者你需要合并所有mask
        else:
            mask_path = mask_list
            
        # 修正路径 (适配 data/cpm17)
        if mask_path.startswith('MoNuSeg/'):
            full_mask_path = os.path.join(DATA_ROOT, mask_path.replace('MoNuSeg/', ''))
        else:
            full_mask_path = mask_path
            
        if not os.path.exists(full_mask_path):
            print(f"Warning: Mask not found: {full_mask_path}")
            continue
            
        # 读取 Mask
        mask = cv2.imread(full_mask_path, 0)
        if mask is None:
            continue
            
        # 生成属性
        prompts, labels = get_attributes_from_mask(mask)
        
        # 提取 Key (文件名，不带后缀)
        # 例如: "data/cpm17/test/image_01.png" -> "image_01.png" (根据 DataLoader 逻辑，key 是文件名)
        key_name = os.path.basename(img_path) 
        
        attribute_info[key_name] = {
            "attribute_prompts": prompts,
            "attribute_labels": labels
        }
        
    # 保存结果
    print(f"正在保存属性信息到: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(attribute_info, f, indent=2)
    print("完成！")

if __name__ == "__main__":
    main()