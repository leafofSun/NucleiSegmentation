import os
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
from skimage import measure

# === 配置区域 ===
# 请确保这里指向您存放 .tif 和 .json 的目录
DATA_ROOT = "data/MoNuSeg_SA1B/train" 
OUTPUT_JSON = "data/MoNuSeg_SA1B/medical_knowledge.json"

# === 1. MoNuSeg 精确器官映射 (TCGA Mapping) ===
# 基于您提供的 PDF 文档，并修正了其中将 Lung 误标为 Liver 的问题
TCGA_MAP = {
    # --- Training Set (30 images) ---
    # Kidney (肾) - Renal Cell Carcinoma
    "TCGA-B0": "Kidney", "TCGA-HE": "Kidney", "TCGA-2Z": "Kidney", 
    # Breast (乳腺) - Invasive Carcinoma
    "TCGA-A7": "Breast", "TCGA-AR": "Breast", "TCGA-E2": "Breast", "TCGA-AO": "Breast",
    # Prostate (前列腺) - Adenocarcinoma
    "TCGA-G9": "Prostate", "TCGA-CH": "Prostate", "TCGA-EJ": "Prostate",
    # Lung (肺) - [关键修正: PDF中标为Liver但实际是肺癌]
    "TCGA-18": "Lung", "TCGA-38": "Lung", "TCGA-49": "Lung", "TCGA-50": "Lung", "TCGA-21": "Lung",
    # Colon (结肠) - Adenocarcinoma
    "TCGA-A6": "Colon", "TCGA-CM": "Colon", "TCGA-NH": "Colon", 
    
    # --- Test Set (涵盖更多器官) ---
    "TCGA-AY": "Stomach", "TCGA-KB": "Stomach", "TCGA-RD": "Stomach",
    "TCGA-IZ": "Liver", "TCGA-MH": "Liver",
    "TCGA-DK": "Bladder", "TCGA-ZF": "Bladder",
    "TCGA-HT": "Brain", "TCGA-CS": "Brain",
}

# === 2. 显式病理先验库 (The "Doctor's Rules") ===
# 将器官信息转化为具体的细胞学描述
ORGAN_KNOWLEDGE = {
    "Kidney": {
        "context": "Renal tissue",
        "cell_desc": "Epithelial cells of proximal tubules",
        "structure": "tubular structure"
    },
    "Breast": {
        "context": "Mammary tissue",
        "cell_desc": "Ductal epithelial cells",
        "structure": "ductal lobular units"
    },
    "Prostate": {
        "context": "Prostatic tissue",
        "cell_desc": "Glandular epithelial cells",
        "structure": "acinar glands"
    },
    "Lung": {
        "context": "Pulmonary tissue",
        "cell_desc": "Pneumocytes and macrophages",
        "structure": "alveolar architecture"
    },
    "Colon": {
        "context": "Colonic mucosa",
        "cell_desc": "Columnar epithelial cells",
        "structure": "glandular crypts"
    },
    "Stomach": {
        "context": "Gastric mucosa",
        "cell_desc": "Glandular cells",
        "structure": "gastric pits"
    },
    "Liver": {
        "context": "Hepatic tissue",
        "cell_desc": "Hepatocytes",
        "structure": "hepatic cords"
    },
    "Bladder": {
        "context": "Urothelial tissue",
        "cell_desc": "Transitional epithelial cells",
        "structure": "urothelium layers"
    },
    "Brain": {
        "context": "Brain tissue",
        "cell_desc": "Glial cells and neurons",
        "structure": "neuropil background"
    },
    "Generic": {
        "context": "Histopathology tissue",
        "cell_desc": "Nuclei",
        "structure": "cellular region"
    }
}

def get_organ_from_filename(filename):
    """从文件名解析器官"""
    for code, organ in TCGA_MAP.items():
        if code in filename:
            return organ
    
    # 兜底：如果文件名本身包含器官名
    for organ in ORGAN_KNOWLEDGE.keys():
        if organ.lower() in filename.lower():
            return organ
    return "Generic"

def decode_mask_from_json(json_path, shape_hint=(1000, 1000)):
    """从 SA-1B JSON 读取 Mask"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 尝试获取真实尺寸
        h, w = shape_hint
        if "image" in data:
            h = data["image"].get("height", h)
            w = data["image"].get("width", w)

        mask = np.zeros((h, w), dtype=np.uint8)
        anns = data.get('annotations', [])
        
        # 简单解码 Polygon 或 RLE
        import pycocotools.mask as coco_mask
        for ann in anns:
            if 'segmentation' not in ann: continue
            seg = ann['segmentation']
            if isinstance(seg, dict): # RLE
                m = coco_mask.decode(seg)
                mask[m > 0] = 1
            elif isinstance(seg, list): # Polygon
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        return mask
    except:
        # 如果出错（比如没装 pycocotools），返回空 Mask，不影响流程
        return np.zeros(shape_hint, dtype=np.uint8)

def analyze_visuals(mask):
    """PromptNu 视觉属性提取"""
    if mask.sum() == 0:
        return {"size": "medium", "shape": "round", "density": "moderate"}
        
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    if not props: return {"size": "medium", "shape": "round", "density": "moderate"}
    
    # 1. Size
    mean_area = np.mean([p.area for p in props])
    if mean_area < 250: size = "small"
    elif mean_area > 650: size = "large, enlarged"
    else: size = "medium-sized"
    
    # 2. Shape (Eccentricity)
    mean_ecc = np.mean([p.eccentricity for p in props])
    if mean_ecc > 0.8: shape = "elongated, spindle-shaped"
    elif mean_ecc < 0.4: shape = "round, spherical"
    else: shape = "oval"
    
    # 3. Density
    density_val = len(props) / (mask.shape[0] * mask.shape[1])
    if density_val > 0.003: density = "densely packed"
    elif density_val < 0.0005: density = "sparsely distributed"
    else: density = "moderately distributed"
    
    return {"size": size, "shape": shape, "density": density}

def construct_text_prompt(organ, visuals):
    """
    核心逻辑：融合 [器官上下文] + [视觉特征]
    """
    kb = ORGAN_KNOWLEDGE.get(organ, ORGAN_KNOWLEDGE["Generic"])
    
    # === 推理层 (Explicit Rules) ===
    cell_desc = kb['cell_desc']
    adj = ""
    
    # 规则 1: 肿瘤特征 (大且不规则)
    if organ in ["Breast", "Kidney", "Lung", "Colon"] and "enlarged" in visuals['size']:
        cell_desc = "Pleomorphic Tumor Nuclei"
        adj = "hyperchromatic"
    # 规则 2: 淋巴细胞特征 (小且圆)
    elif "small" in visuals['size'] and "round" in visuals['shape']:
        cell_desc = "Lymphocytes"
        adj = "darkly stained"
    # 规则 3: 前列腺/结肠 (密集腺体)
    elif organ in ["Prostate", "Colon"] and "dense" in visuals['density']:
        cell_desc = "Glandular Epithelial Nuclei"
        adj = "basally oriented"

    # 生成最终句子
    text = (f"Microscopic view of {adj} {visuals['size']} {cell_desc} with {visuals['shape']} features, "
            f"{visuals['density']} in {kb['context']} featuring {kb['structure']}.")
    
    return " ".join(text.split())

def main():
    # 扫描 .json 文件
    json_files = glob.glob(os.path.join(DATA_ROOT, "*.json"))
    # 排除非数据 json
    json_files = [f for f in json_files if "knowledge" not in f and "attribute" not in f]
    
    print(f"🚀 Building Explicit Knowledge Base from {len(json_files)} samples...")
    
    kb_database = {}
    
    for json_path in tqdm(json_files):
        # 对应图片文件名 (.tif)
        filename = os.path.basename(json_path).replace(".json", ".tif")
        
        # A. 确定器官 (MoNuSeg 官方映射 + PDF修正)
        organ = get_organ_from_filename(filename)
        
        # B. 提取视觉特征 (PromptNu 算法)
        mask = decode_mask_from_json(json_path)
        visuals = analyze_visuals(mask)
        
        # C. 生成显式知识文本 (KIM Input)
        prompt = construct_text_prompt(organ, visuals)
        
        # D. 存入库
        kb_database[filename] = {
            "organ_id": organ,         # -> DualLearner
            "text_prompt": prompt,     # -> KIM (Explicit Refiner)
            "visual_stats": visuals
        }
        
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(kb_database, f, indent=4)
        
    print(f"✅ Knowledge Base saved to: {OUTPUT_JSON}")
    print("   Ready for training!")

if __name__ == "__main__":
    main()