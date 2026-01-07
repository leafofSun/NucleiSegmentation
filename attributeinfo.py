import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from pycocotools import mask as coco_mask

# ==========================================
# ⚙️ 配置区域
# ==========================================
# 建议先运行 "test" 模式来修复您的评估尺子
MODE = "test"  

if MODE == "test":
    DATA_ROOT = "data/MoNuSeg_SA1B/test"
    OUTPUT_JSON = "data/MoNuSeg_SA1B/test_dynamic_instance_attributes.json"
else:
    DATA_ROOT = "data/MoNuSeg_SA1B/train"
    OUTPUT_JSON = "data/MoNuSeg_SA1B/train_dynamic_instance_attributes.json"

# ==========================================
# 🏥 器官映射表 (Source: NCI GDC & MoNuSeg Official)
# ==========================================

# 1. 训练集映射 (来自您的 PDF: Training Patient Organ information)
TRAIN_ORGAN_MAP = {
    # Breast
    "TCGA-A7-A13E": "Breast", "TCGA-A7-A13F": "Breast", "TCGA-AR-A1AK": "Breast",
    "TCGA-AR-A1AS": "Breast", "TCGA-E2-A1B5": "Breast", "TCGA-E2-A14V": "Breast",
    # Kidney
    "TCGA-B0-5711": "Kidney", "TCGA-HE-7128": "Kidney", "TCGA-HE-7129": "Kidney",
    "TCGA-HE-7130": "Kidney", "TCGA-B0-5710": "Kidney", "TCGA-B0-5698": "Kidney",
    # Liver (原发灶不明或肺癌肝转移，但组织纹理为肝脏)
    "TCGA-18-5592": "Liver", "TCGA-38-6178": "Liver", "TCGA-49-4488": "Liver",
    "TCGA-50-5931": "Liver", "TCGA-21-5784": "Liver", "TCGA-21-5786": "Liver",
    # Prostate
    "TCGA-G9-6336": "Prostate", "TCGA-G9-6348": "Prostate", "TCGA-G9-6356": "Prostate",
    "TCGA-G9-6363": "Prostate", "TCGA-CH-5767": "Prostate", "TCGA-G9-6362": "Prostate",
    # Bladder
    "TCGA-DK-A216": "Bladder", "TCGA-G2-A2EK": "Bladder",
    # Colon
    "TCGA-AY-A8YK": "Colon", "TCGA-NH-A8F7": "Colon",
    # Stomach
    "TCGA-KB-A93J": "Stomach", "TCGA-RD-A8N9": "Stomach"
}

# 2. 测试集映射 (经过 NCI GDC 核对)
TEST_ORGAN_MAP = {
    "TCGA-2Z-A9J9": "Kidney",    
    "TCGA-69-7764": "Lung",   
    "TCGA-GL-A4EM": "Kidney",    
    "TCGA-EJ-A46H": "Prostate",  
    "TCGA-FG-A4MU": "Brain",  
    "TCGA-AO-A0J2": "Breast",    
    "TCGA-AC-A2FO": "Breast",    
    "TCGA-CU-A0YN": "Bladder",  
    "TCGA-A6-2675": "Colorectal", 
    "TCGA-A6-2680": "Colorectal", 
    "TCGA-A6-5662": "Colorectal", 
    "TCGA-HC-7209": "Prostate", 
    "TCGA-44-2665": "Lung",      
    "TCGA-HT-8564": "Brain"       
}

# 根据模式选择字典
CURRENT_MAP = TEST_ORGAN_MAP if MODE == "test" else TRAIN_ORGAN_MAP

# ==========================================
# 📊 统计学阈值配置 (PromptNu Style)
# ==========================================
STATS_CONFIG = {
    # 面积：Large 必须显著大于均值 (Mean + 1.0 * Std)
    "area":        {"alpha_low": 0.5, "alpha_high": 1.0}, 
    "roundness":   {"alpha_low": 1.0, "alpha_high": 0.5},
    "aspect_ratio":{"alpha_low": 0.0, "alpha_high": 1.0},
    "solidity":    {"alpha_low": 1.0, "alpha_high": 0.0}
}

def get_organ(filename):
    # 模糊匹配文件名中的ID
    for k, v in CURRENT_MAP.items():
        if k in filename: return v
    return "Unknown"

def calculate_morphology(binary_mask):
    """计算准确的形态学特征 (修正版)"""
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    if area < 10: return None 

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: return None
    roundness = (4 * np.pi * area) / (perimeter ** 2)

    # 修正长宽比：确保 >= 1.0
    if len(cnt) >= 5:
        (x, y), (d1, d2), angle = cv2.fitEllipse(cnt)
        major, minor = max(d1, d2), min(d1, d2)
        aspect_ratio = major / minor if minor > 0 else 1.0
    else:
        aspect_ratio = 1.0

    return {"area": float(area), "roundness": float(roundness), "aspect_ratio": float(aspect_ratio)}

def main():
    print(f"🚀 Generating Attributes for [{MODE.upper()}] Set...")
    print(f"📂 Reading from: {DATA_ROOT}")
    print(f"📚 Using Organ Map with {len(CURRENT_MAP)} entries (Source: NCI GDC).")

    json_files = glob.glob(os.path.join(DATA_ROOT, "*.json"))
    all_data = {}
    global_stats = {"area": [], "roundness": [], "aspect_ratio": []}

    # === Step 1: 提取特征 ===
    for json_file in tqdm(json_files):
        fname = os.path.basename(json_file).replace(".json", "")
        organ = get_organ(fname)
        
        with open(json_file, 'r') as f: data = json.load(f)
        anns = data.get('annotations', data)
        
        img_instances = []
        for idx, ann in enumerate(anns):
            if 'segmentation' not in ann: continue
            
            # Mask Decoding
            mask = None
            seg = ann['segmentation']
            if isinstance(seg, dict) and 'counts' in seg:
                mask = coco_mask.decode(seg)
            elif isinstance(seg, list):
                # 如果是多边形，创建画布
                bbox = ann.get('bbox', [0, 0, 1024, 1024])
                h = max(int(bbox[1]+bbox[3]), 1024)
                w = max(int(bbox[0]+bbox[2]), 1024)
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [pts], 1)
            
            if mask is not None:
                feats = calculate_morphology(mask)
                if feats:
                    img_instances.append({
                        "id": idx, "organ": organ, 
                        "bbox": ann.get('bbox', []), "attrs": feats
                    })
                    for k, v in feats.items(): global_stats[k].append(v)
        
        all_data[os.path.basename(json_file).replace(".json", ".tif")] = img_instances

    # === Step 2: 计算全局阈值 (Mean +/- Std) ===
    thresholds = {}
    print("\n📊 Global Statistics (PromptNu Style):")
    for k, v in global_stats.items():
        if not v: continue
        arr = np.array(v)
        mean, std = np.mean(arr), np.std(arr)
        
        low = max(0, mean - STATS_CONFIG[k]["alpha_low"] * std)
        high = mean + STATS_CONFIG[k]["alpha_high"] * std
        
        thresholds[k] = {"low": low, "high": high, "mean": mean}
        print(f"  - {k.ljust(12)}: Mean={mean:.2f} | Small < {low:.2f} | Large > {high:.2f}")

    # === Step 3: 打标签 ===
    final_output = {"meta": {"mode": MODE, "thresholds": thresholds}, "images": {}}
    for fname, instances in all_data.items():
        processed_insts = []
        for inst in instances:
            tags = []
            attrs = inst['attrs']
            
            # Size Tagging
            if attrs['area'] < thresholds['area']['low']: tags.append("Small")
            elif attrs['area'] > thresholds['area']['high']: tags.append("Large")
            else: tags.append("Medium")
            
            # Organ Tagging
            if inst['organ'] != "Unknown": tags.append(f"{inst['organ']}-like")
            
            # Shape Tagging
            if 'roundness' in thresholds:
                 if attrs['roundness'] < thresholds['roundness']['low']: tags.append("Irregular")
                 elif attrs['roundness'] > 0.85: tags.append("Round")
            
            inst['tags'] = tags
            processed_insts.append(inst)
        final_output['images'][fname] = processed_insts

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"✅ Done! Saved attributes to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()