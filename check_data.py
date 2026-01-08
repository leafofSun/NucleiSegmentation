import os
import json
import numpy as np
import glob
import cv2
import math
from tqdm import tqdm

# 尝试导入 pycocotools (处理 SA-1B RLE 格式)
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("⚠️ [Warning] pycocotools not found. RLE decoding might fail.")

def calculate_roundness(area, perimeter):
    """计算圆度: 4 * pi * Area / Perimeter^2 (1.0 为完美圆)"""
    if perimeter == 0: return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def get_polygon_props(segmentation):
    """从 Polygon 格式获取面积和周长"""
    area = 0
    perimeter = 0
    # SA-1B Polygon 是 [[x1, y1, x2, y2, ...]]
    for poly in segmentation:
        pts = np.array(poly).reshape(-1, 2).astype(np.float32)
        area += cv2.contourArea(pts)
        perimeter += cv2.arcLength(pts, True)
    return area, perimeter

def get_rle_props(segmentation):
    """从 RLE 格式获取面积 (周长计算较复杂，暂略或用近似)"""
    if 'counts' in segmentation:
        mask = coco_mask.decode(segmentation)
        area = np.sum(mask)
        # 从 mask 提取轮廓算周长
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0
        for cnt in contours:
            perimeter += cv2.arcLength(cnt, True)
        return area, perimeter
    return 0, 0

def generate_prompt_library(data_root, output_path="data/MoNuSeg_SA1B/specific_prompts.json"):
    print(f"🚀 Scanning dataset in: {data_root}")
    
    json_files = glob.glob(os.path.join(data_root, "**/*.json"), recursive=True)
    # 过滤掉非标注文件 (比如我们自己生成的 stats.json)
    json_files = [f for f in json_files if "attributes" not in f and "prompts" not in f and "stats" not in f]
    
    print(f"📂 Found {len(json_files)} annotation files.")

    # === 第一步：全局统计 (Global Statistics) ===
    print("\n[Step 1] Analyzing Global Statistics (PromptNu Method)...")
    
    all_areas = []
    all_roundness = []
    img_counts = []
    
    # 暂存每张图的原始数据，避免读两次文件
    img_cache = {} 

    for json_file in tqdm(json_files):
        filename = os.path.basename(json_file).replace(".json", "")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            anns = data.get('annotations', []) if isinstance(data, dict) else data
            
            img_areas = []
            img_roundness = []
            
            for ann in anns:
                area, perimeter = 0, 0
                
                # 1. 优先使用预计算的 area
                if 'area' in ann:
                    area = ann['area']
                
                # 2. 如果没有，或者需要算周长，则解码 Segmentation
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, list): # Polygon
                        a, p = get_polygon_props(seg)
                        if area == 0: area = a
                        perimeter = p
                    elif isinstance(seg, dict): # RLE
                        a, p = get_rle_props(seg)
                        if area == 0: area = a
                        perimeter = p
                
                if area > 10: # 忽略极小噪点
                    img_areas.append(area)
                    all_areas.append(area)
                    if perimeter > 0:
                        r = calculate_roundness(area, perimeter)
                        img_roundness.append(r)
                        all_roundness.append(r)

            count = len(img_areas)
            img_counts.append(count)
            
            img_cache[filename] = {
                "areas": img_areas,
                "roundness": img_roundness,
                "count": count
            }
            
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")

    # === 计算 PromptNu 阈值 ===
    # 论文 III.B: Mean 和 Mean + 2*Std
    np_areas = np.array(all_areas)
    np_counts = np.array(img_counts)
    np_round = np.array(all_roundness)

    mean_area = np.mean(np_areas)
    std_area = np.std(np_areas)
    
    mean_count = np.mean(np_counts)
    std_count = np.std(np_counts)
    
    mean_round = np.mean(np_round)

    print("\n📊 Dataset Statistics:")
    print(f"  Nuclei Size (Area): Mean={mean_area:.1f}, Std={std_area:.1f}")
    print(f"  Nuclei Count/Img:   Mean={mean_count:.1f}, Std={std_count:.1f}")
    print(f"  Roundness:          Mean={mean_round:.2f}")

    # 定义阈值 (Thresholds)
    THRESHOLDS = {
        "size": {
            "small_limit": mean_area, # 小于均值 = Small
            "large_limit": mean_area + std_area # 大于均值+1倍方差 = Large (论文是2倍，但病理图通常1倍更合理，可调)
        },
        "density": {
            "sparse_limit": max(1, mean_count - std_count),
            "dense_limit": mean_count + std_count
        },
        "shape": {
            "round_limit": 0.85, # 圆度 > 0.85 算圆
            "irregular_limit": 0.60 # 圆度 < 0.60 算不规则
        }
    }
    
    print(f"⚙️  Thresholds: Large > {THRESHOLDS['size']['large_limit']:.1f}, Dense > {THRESHOLDS['density']['dense_limit']:.1f}")

    # === 第二步：生成专用文本 (Prompt Generation) ===
    print("\n[Step 2] Generating Specific Prompts...")
    
    prompt_library = {}
    
    for filename, stats in img_cache.items():
        if stats["count"] == 0: continue
        
        # 1. Size Attribute
        avg_area = np.mean(stats["areas"])
        if avg_area > THRESHOLDS["size"]["large_limit"]:
            size_desc = "large"
        elif avg_area < THRESHOLDS["size"]["small_limit"]:
            size_desc = "small"
        else:
            size_desc = "medium"
            
        # 2. Density Attribute
        cnt = stats["count"]
        if cnt > THRESHOLDS["density"]["dense_limit"]:
            density_desc = "densely packed"
        elif cnt < THRESHOLDS["density"]["sparse_limit"]:
            density_desc = "sparsely distributed"
        else:
            density_desc = "moderately distributed"
            
        # 3. Shape Attribute
        avg_rnd = np.mean(stats["roundness"]) if stats["roundness"] else 0
        if avg_rnd > THRESHOLDS["shape"]["round_limit"]:
            shape_desc = "round"
        elif avg_rnd < THRESHOLDS["shape"]["irregular_limit"]:
            shape_desc = "irregular"
        else:
            shape_desc = "elliptical"

        # === 核心：PromptNu 风格的句子构建 ===
        # Template: "Microscopic image of [Size], [Shape] nuclei, [Density]."
        specific_prompt = f"Microscopic image of {size_desc}, {shape_desc} nuclei, {density_desc}."
        
        prompt_library[filename] = {
            "prompt": specific_prompt,
            "attributes": {
                "size": size_desc,
                "shape": shape_desc,
                "density": density_desc
            },
            "stats": {
                "avg_area": float(avg_area),
                "count": int(cnt),
                "avg_roundness": float(avg_rnd)
            }
        }

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(prompt_library, f, indent=4)
        
    print(f"\n✅ Generated Specific Prompts for {len(prompt_library)} images.")
    print(f"💾 Saved to: {output_path}")
    print(f"📝 Example: {list(prompt_library.values())[0]['prompt']}")

if __name__ == "__main__":
    # 🔥 修改为你的数据路径
    DATA_PATH = "data/MoNuSeg_SA1B" 
    generate_prompt_library(DATA_PATH)