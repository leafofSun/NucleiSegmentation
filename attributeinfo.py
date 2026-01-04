import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from pycocotools import mask as coco_mask
from sklearn.cluster import KMeans

# === 配置路径 ===
DATA_ROOT = "data/MoNuSeg_SA1B/test" 
OUTPUT_JSON = "data/MoNuSeg_SA1B/test_dynamic_instance_attributes.json"

def calculate_morphology(binary_mask):
    """
    计算单个实例的形态学特征
    """
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    cnt = contours[0]
    
    # 1. Area
    area = cv2.contourArea(cnt)
    if area < 10: return None 

    # 2. Roundness
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: return None
    roundness = (4 * np.pi * area) / (perimeter ** 2)

    # 3. Aspect Ratio
    if len(cnt) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        if ma == 0: aspect_ratio = 1.0
        else: aspect_ratio = MA / ma
    else:
        aspect_ratio = 1.0

    # 4. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: solidity = 1.0
    else: solidity = area / hull_area

    return {
        "area": float(area),
        "roundness": float(roundness),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity)
    }

def main():
    print(f"🚀 Starting Instance-level Attribute Mining from {DATA_ROOT}...")
    
    json_files = glob.glob(os.path.join(DATA_ROOT, "*.json"))
    all_data = {}
    
    global_stats = {
        "area": [],
        "roundness": [],
        "aspect_ratio": [],
        "solidity": []
    }

    # === 步骤 1: 遍历所有文件收集属性 ===
    for json_file in tqdm(json_files):
        filename = os.path.basename(json_file).replace(".json", ".tif")
        # 兼容性检查
        if not os.path.exists(os.path.join(DATA_ROOT, filename)):
            filename = filename.replace(".tif", ".png")
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        anns = data.get('annotations', [])
        if not anns and isinstance(data, list): anns = data
        
        image_instances = []
        
        for idx, ann in enumerate(anns):
            if 'segmentation' not in ann: continue
            
            seg = ann['segmentation']
            mask = None
            
            # Case A: RLE (常见)
            if isinstance(seg, dict) and 'counts' in seg:
                mask = coco_mask.decode(seg)
            
            # Case B: Polygon (补全逻辑)
            elif isinstance(seg, list):
                # 尝试通过 bbox 获取尺寸，如果没有 bbox 则估算
                bbox = ann.get('bbox', [0, 0, 1024, 1024]) # x, y, w, h
                h = int(bbox[1] + bbox[3])
                w = int(bbox[0] + bbox[2])
                # 稍微给大一点防止越界，或者如果有原图尺寸最好
                h = max(h, 1024) 
                w = max(w, 1024)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [pts], 1)
            
            if mask is not None:
                feats = calculate_morphology(mask)
                if feats:
                    instance_info = {
                        "id": idx, 
                        "bbox": ann.get('bbox', []),
                        "attrs": feats
                    }
                    image_instances.append(instance_info)
                    for k, v in feats.items():
                        global_stats[k].append(v)
        
        all_data[filename] = image_instances

    # === 步骤 2: 计算全局阈值 (修正缩进：必须在循环外！) ===
    cluster_centers = {} 
    print("\n📊 Calculating Adaptive Thresholds via K-Means...")
    
    for k, v in global_stats.items():
        if len(v) < 3: continue 
        
        data = np.array(v).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(data)
        
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        # 计算分界线
        thresh_low = (centers[0] + centers[1]) / 2
        thresh_high = (centers[1] + centers[2]) / 2
        
        cluster_centers[k] = {"low": float(thresh_low), "high": float(thresh_high)}
        print(f"  - {k}: Low < {thresh_low:.2f} | High > {thresh_high:.2f} (Centers: {[round(c,2) for c in centers]})")

    # === 步骤 3: 打标签 (修正缩进：必须在循环外！) ===
    print("\n🏷️ Tagging instances based on K-Means boundaries...")
    final_output = {
        "meta": {
            "method": "K-Means Adaptive Quantization (K=3)",
            "thresholds": cluster_centers,
            "description": "Auto-generated instance attributes for Dynamic GT Masking"
        },
        "images": {}
    }

    for fname, instances in all_data.items():
        tagged_instances = []
        for inst in instances:
            attrs = inst['attrs']
            tags = []
            
            # Size
            if 'area' in cluster_centers:
                if attrs['area'] < cluster_centers['area']['low']: tags.append("Small")
                elif attrs['area'] > cluster_centers['area']['high']: tags.append("Large")
                else: tags.append("Medium")
            
            # Shape
            if 'roundness' in cluster_centers:
                if attrs['roundness'] > cluster_centers['roundness']['high']: tags.append("Round")
                elif attrs['roundness'] < cluster_centers['roundness']['low']: tags.append("Irregular")
                else: tags.append("Semi-round")
            
            # Elongation
            if 'aspect_ratio' in cluster_centers:
                if attrs['aspect_ratio'] < cluster_centers['aspect_ratio']['low']: tags.append("Elongated")
            
            inst['tags'] = tags
            tagged_instances.append(inst)
            
        final_output['images'][fname] = tagged_instances

    # === 保存 ===
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"✅ Done! Saved attribute database to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()