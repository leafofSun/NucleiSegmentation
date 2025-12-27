import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from skimage import measure

# 尝试导入 pycocotools，这是解析 SA-1B RLE 的标准工具
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ 请安装 pycocotools: pip install pycocotools")
    exit()

# === 配置路径 ===
# 指向包含图片和对应 json 的文件夹
DATA_ROOT = "data/MoNuSeg_SA1B/train" 
OUTPUT_JSON = "data/MoNuSeg_SA1B/attribute_info_train.json"

def decode_sa1b_mask(json_path, shape=None):
    """
    从 SA-1B 格式的 JSON 中解析出二值 Mask
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # SA-1B 标注通常在 'annotations' 列表里
    anns = data.get('annotations', [])
    if not anns and isinstance(data, list): anns = data # 兼容直接是 list 的情况
    
    # 如果不知道图像尺寸，尝试从 json 或第一条标注推断，或者由外部传入
    # 这里我们建立一个全黑底图
    if shape is None:
        # 尝试读取同名图片获取尺寸
        img_path = json_path.replace(".json", ".tif") 
        if not os.path.exists(img_path):
             img_path = json_path.replace(".json", ".png")
        if os.path.exists(img_path):
            temp_img = cv2.imread(img_path)
            h, w = temp_img.shape[:2]
        else:
            # 兜底：如果找不到图，默认 1000x1000 (MoNuSeg标准)
            h, w = 1000, 1000
    else:
        h, w = shape

    full_mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        if 'segmentation' in ann:
            seg = ann['segmentation']
            # 情况 A: RLE 格式 (SA-1B 标准)
            if isinstance(seg, dict) and 'counts' in seg:
                rle_mask = coco_mask.decode(seg)
                full_mask[rle_mask > 0] = 1
            # 情况 B: Polygon 格式 (点列表)
            elif isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(full_mask, [pts], 1)
    
    return full_mask

def analyze_mask(mask):
    """
    对 Mask 进行连通域分析，提取 PromptNu 所需的 5 大属性
    """
    # 连通域标记
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    if len(props) == 0:
        return None

    # --- 1. Size (大小) ---
    areas = [p.area for p in props]
    mean_area = np.mean(areas)
    
    size_tags = []
    # 阈值可微调
    if mean_area < 250: size_tags.append("small")
    elif mean_area < 650: size_tags.append("medium")
    else: size_tags.append("large")
        
    # --- 2. Density (密度) ---
    h, w = mask.shape
    foreground_ratio = np.sum(mask) / (h * w)
    count = len(props)
    
    density_tags = []
    if foreground_ratio > 0.20 or count > 400:
        density_tags.append("densely packed")
    elif foreground_ratio > 0.05 or count > 100:
        density_tags.append("moderately dense")
    else:
        density_tags.append("sparsely distributed")

    # --- 3. Shape (形状) ---
    eccentricities = [p.eccentricity for p in props]
    mean_ecc = np.mean(eccentricities)
    
    shape_tags = []
    if mean_ecc > 0.85:
        shape_tags.extend(["elongated", "spindle-shaped"])
    elif mean_ecc < 0.5:
        shape_tags.extend(["round", "spherical"])
    else:
        shape_tags.append("elliptical/oval")
        
    solidities = [p.solidity for p in props]
    if np.mean(solidities) < 0.85:
        shape_tags.append("irregular")

    # --- 4. Arrange (排列) ---
    arrange_tags = ["scattered"]
    if "densely packed" in density_tags:
        arrange_tags.append("clustered")
        
    # --- 5. Color (颜色) ---
    color_tags = ["deep purple"] # H&E 固定

    # === 构造 Rich Text ===
    # 类似于: "Deep purple small elliptical/oval nuclei, densely packed"
    rich_text = f"{color_tags[0]} {size_tags[0]} "
    if len(shape_tags) > 0: rich_text += f"{shape_tags[0]} "
    rich_text += "nuclei"
    if "densely packed" in density_tags: rich_text += ", densely packed"
    elif "sparsely distributed" in density_tags: rich_text += ", scattered"

    return {
        "color": list(set(color_tags)),
        "size": list(set(size_tags)),
        "density": list(set(density_tags)),
        "arrange": list(set(arrange_tags)),
        "shape": list(set(shape_tags)),
        "rich_text": rich_text,
        "target_text": rich_text # 兼容旧代码 Key
    }

def main():
    # 扫描所有 .json 文件 (排除掉我们自己生成的 attribute json)
    json_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.json"), recursive=True)
    
    # 过滤掉非 GT 的 json (比如生成的 prompt json)
    json_files = [f for f in json_files if "attribute_info" not in f and "global_label" not in f]
    
    print(f"🔍 Found {len(json_files)} SA-1B JSON files. Analyzing...")
    
    prompt_dict = {} # 用于保存结果的字典
    
    for json_path in tqdm(json_files):
        filename = os.path.basename(json_path)
        # 假设图片名和json名一致 (e.g., img.tif, img.json)
        # 或者是 img.tif 对应 img.json
        # 我们用图片文件名作为 Key
        img_name = filename.replace(".json", ".tif") 
        # 如果您的数据集中是 .png，请改为 .png
        if not os.path.exists(os.path.join(os.path.dirname(json_path), img_name)):
             img_name = filename.replace(".json", ".png")

        # 1. 解码 Mask
        try:
            mask = decode_sa1b_mask(json_path)
        except Exception as e:
            print(f"❌ Error decoding {filename}: {e}")
            continue
            
        # 2. 分析属性
        if np.sum(mask) == 0:
            continue # 空 Mask 跳过
            
        attrs = analyze_mask(mask)
        
        # 3. 记录
        # 添加 PromptNu 风格的 attribute_prompts 字段
        all_prompts = []
        for k in ["color", "size", "shape", "density", "arrange"]:
            all_prompts.extend(attrs[k])
        attrs["attribute_prompts"] = all_prompts
        
        # 以图片文件名 (xxx.tif) 为 Key
        prompt_dict[img_name] = attrs

    # 保存为 Dict 格式 (供 DataLoader 使用)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(prompt_dict, f, indent=4)
        
    print(f"✅ Saved attributes to {OUTPUT_JSON}")
    print(f"   Total processed: {len(prompt_dict)}")

if __name__ == "__main__":
    main()