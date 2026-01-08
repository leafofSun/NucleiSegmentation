import os
import json
import numpy as np
import glob
from tqdm import tqdm
try:
    from pycocotools import mask as coco_mask
except ImportError:
    print("Please install pycocotools: pip install pycocotools")

def calculate_dataset_stats(data_dir):
    print(f"🔍 Scanning dataset in: {data_dir} ...")
    
    # 查找所有 json 文件
    json_paths = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    
    all_areas = []
    
    print(f"Found {len(json_paths)} json files. Calculating areas...")
    
    for json_path in tqdm(json_paths):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 兼容 SA-1B 格式 (dict with 'annotations' key or list)
            annotations = data.get('annotations', []) if isinstance(data, dict) else data
            
            for ann in annotations:
                area = 0
                # 优先使用预计算的 area 字段
                if 'area' in ann:
                    area = ann['area']
                # 如果没有，从 segmentation 计算
                elif 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, dict) and 'counts' in seg: # RLE
                        area = int(coco_mask.area(seg))
                    elif isinstance(seg, list): # Polygon
                        # Polygon area calculation implies decoding or shapely, 
                        # for simplicity we assume area field exists or skip complex poly calc here
                        pass 
                
                if area > 0:
                    all_areas.append(area)
                    
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue

    if not all_areas:
        print("❌ No nuclei found! Check your data path or json format.")
        return None

    # === 核心统计学计算 ===
    all_areas = np.array(all_areas)
    mean_area = np.mean(all_areas)
    std_area = np.std(all_areas)
    
    # PromptNu 论文定义:
    # Small < Mean
    # Large > Mean + 2 * Std
    
    stats = {
        "count": len(all_areas),
        "mean_area": float(mean_area),
        "std_area": float(std_area),
        "thresholds": {
            "small_upper": float(mean_area),               # 小于均值算 Small
            "large_lower": float(mean_area + 2 * std_area) # 大于 均值+2倍方差 算 Large
        }
    }
    
    print("\n✅ Statistics Calculated:")
    print(f"   Total Nuclei: {stats['count']}")
    print(f"   Mean Area (μ): {stats['mean_area']:.2f}")
    print(f"   Std Dev (σ):   {stats['std_area']:.2f}")
    print(f"   --------------------------------")
    print(f"   [Small]  < {stats['thresholds']['small_upper']:.2f} pixels")
    print(f"   [Medium] {stats['thresholds']['small_upper']:.2f} ~ {stats['thresholds']['large_lower']:.2f} pixels")
    print(f"   [Large]  > {stats['thresholds']['large_lower']:.2f} pixels")
    
    # 保存到文件
    save_path = os.path.join(data_dir, "dataset_stats.json")
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n💾 Stats saved to: {save_path}")
    return stats

if __name__ == "__main__":
    # 请修改为您实际的数据路径
    DATA_PATH = "data/MoNuSeg_SA1B" 
    calculate_dataset_stats(DATA_PATH)