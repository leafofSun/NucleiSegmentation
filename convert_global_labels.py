import os
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
from skimage import measure

# ==============================================================================
# ⚙️ 配置区域 (Configuration)
# ==============================================================================
# 请确保这里指向您存放 .json 文件的目录 (例如 MoNuSeg 的 Training 或 Test 目录)
DATA_ROOT = "data/MoNuSeg_SA1B/train" 
OUTPUT_JSON = "data/MoNuSeg_SA1B/medical_knowledge.json"

# ==============================================================================
# 🧠 医学知识映射 (Medical Knowledge Mappings)
# ==============================================================================

# 1. MoNuSeg 精确器官映射 (TCGA Mapping)
TCGA_MAP = {
    # --- Training Set ---
    "TCGA-B0": "Kidney", "TCGA-HE": "Kidney", "TCGA-2Z": "Kidney", 
    "TCGA-A7": "Breast", "TCGA-AR": "Breast", "TCGA-E2": "Breast", "TCGA-AO": "Breast",
    "TCGA-G9": "Prostate", "TCGA-CH": "Prostate", "TCGA-EJ": "Prostate",
    "TCGA-18": "Lung", "TCGA-38": "Lung", "TCGA-49": "Lung", "TCGA-50": "Lung", "TCGA-21": "Lung",
    "TCGA-A6": "Colon", "TCGA-CM": "Colon", "TCGA-NH": "Colon", 
    # --- Test Set ---
    "TCGA-AY": "Stomach", "TCGA-KB": "Stomach", "TCGA-RD": "Stomach",
    "TCGA-IZ": "Liver", "TCGA-MH": "Liver",
    "TCGA-DK": "Bladder", "TCGA-ZF": "Bladder",
    "TCGA-HT": "Brain", "TCGA-CS": "Brain",
}

# 2. 显式病理先验库
ORGAN_KNOWLEDGE = {
    "Kidney": {"context": "Renal tissue", "cell_desc": "Epithelial cells of proximal tubules", "structure": "tubular structure"},
    "Breast": {"context": "Mammary tissue", "cell_desc": "Ductal epithelial cells", "structure": "ductal lobular units"},
    "Prostate": {"context": "Prostatic tissue", "cell_desc": "Glandular epithelial cells", "structure": "acinar glands"},
    "Lung": {"context": "Pulmonary tissue", "cell_desc": "Pneumocytes and macrophages", "structure": "alveolar architecture"},
    "Colon": {"context": "Colonic mucosa", "cell_desc": "Columnar epithelial cells", "structure": "glandular crypts"},
    "Stomach": {"context": "Gastric mucosa", "cell_desc": "Glandular cells", "structure": "gastric pits"},
    "Liver": {"context": "Hepatic tissue", "cell_desc": "Hepatocytes", "structure": "hepatic cords"},
    "Bladder": {"context": "Urothelial tissue", "cell_desc": "Transitional epithelial cells", "structure": "urothelium layers"},
    "Brain": {"context": "Brain tissue", "cell_desc": "Glial cells and neurons", "structure": "neuropil background"},
    "Generic": {"context": "Histopathology tissue", "cell_desc": "Nuclei", "structure": "cellular region"}
}

def get_organ_from_filename(filename):
    """根据文件名解析器官类型"""
    for code, organ in TCGA_MAP.items():
        if code in filename: return organ
    for organ in ORGAN_KNOWLEDGE.keys():
        if organ.lower() in filename.lower(): return organ
    return "Generic"

# ==============================================================================
# 🛠️ 核心工具函数 (Core Utilities)
# ==============================================================================

def decode_instance_mask_from_json(json_path, shape_hint=(1000, 1000)):
    """
    🔥 [核心修复] 生成 Instance Mask (int32)，每个细胞一个独立 ID。
    解决旧版 'Binary Mask' 导致的细胞粘连、标准差爆炸问题。
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        h, w = shape_hint
        if "image" in data:
            h = data["image"].get("height", h)
            w = data["image"].get("width", w)

        # 使用 int32 存储 ID (支持 >255 个细胞)
        instance_mask = np.zeros((h, w), dtype=np.int32)
        anns = data.get('annotations', [])
        
        # 尝试使用 pycocotools 加速 RLE 解码
        try:
            import pycocotools.mask as coco_mask
            has_coco = True
        except ImportError:
            has_coco = False

        current_id = 1 
        
        for ann in anns:
            if 'segmentation' not in ann: continue
            seg = ann['segmentation']
            
            single_obj_mask = None
            
            if isinstance(seg, dict) and has_coco: # RLE 格式
                single_obj_mask = coco_mask.decode(seg)
            elif isinstance(seg, list): # Polygon 格式
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    # 注意：坐标可能需要取整
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp_mask, [pts], 1)
                single_obj_mask = temp_mask

            # 将当前细胞以 Unique ID 填入主 Mask
            if single_obj_mask is not None:
                # 即使像素重叠，也会覆盖为新的 ID，从而在逻辑上分离它们
                instance_mask[single_obj_mask > 0] = current_id
                current_id += 1
                
        return instance_mask
    except Exception as e:
        # print(f"Error decoding {json_path}: {e}")
        return np.zeros(shape_hint, dtype=np.int32)

def get_dataset_statistics(json_files):
    """
    🔥 [Pass 1] 全局扫描：计算分位数统计 (Percentiles)。
    相比 Mean/Std，分位数对长尾分布和剩余的粘连噪声更鲁棒。
    """
    print("📊 Phase 1: Analyzing dataset statistics (Global Pass - Instance Level)...")
    
    all_areas = []
    nuclei_counts = []
    
    for json_path in tqdm(json_files):
        # 使用实例掩码解码！
        instance_mask = decode_instance_mask_from_json(json_path)
        
        # 如果 Mask 为空，跳过
        if instance_mask.max() == 0: continue
            
        # measure.regionprops 可以正确区分不同的 ID
        props = measure.regionprops(instance_mask)
        
        nuclei_counts.append(len(props))
        for p in props:
            # 安全过滤：忽略 >10000 像素的极端异常值（可能是标注错误）
            if p.area < 10000:
                all_areas.append(p.area)
            
    all_areas = np.array(all_areas)
    nuclei_counts = np.array(nuclei_counts)
    
    if len(all_areas) == 0:
        return None

    # 使用分位数定义阈值
    # Small: 最小的 33%
    # Large: 最大的 33% (Top 33%)
    stats = {
        "size_th_small": np.percentile(all_areas, 33),
        "size_th_large": np.percentile(all_areas, 67),
        
        "dense_th_sparse": np.percentile(nuclei_counts, 33),
        "dense_th_dense": np.percentile(nuclei_counts, 67),

        # 仅供参考的均值
        "size_mean": np.mean(all_areas)
    }
    
    print(f"\n📈 Robust Statistics Report (Percentiles):")
    print(f"   [Size] Mean: {stats['size_mean']:.1f} px (Instance-based)")
    print(f"   [Size Thresholds] Small < {stats['size_th_small']:.1f} | Large > {stats['size_th_large']:.1f}")
    print(f"   [Density Thresholds] Sparse < {stats['dense_th_sparse']:.1f} | Dense > {stats['dense_th_dense']:.1f}\n")
    
    return stats

def analyze_visuals_dynamic(mask, stats):
    """
    🔥 [Pass 2] 动态判定：根据全局阈值判定当前图片的属性
    """
    # mask 必须是 Instance Mask
    if mask.max() == 0 or stats is None:
        return {"size": "medium-sized", "shape": "round", "density": "moderate"}
        
    props = measure.regionprops(mask)
    if not props: 
        return {"size": "medium-sized", "shape": "round", "density": "moderate"}
    
    # 1. Size (对比全局阈值)
    current_mean_area = np.mean([p.area for p in props])
    
    if current_mean_area > stats['size_th_large']:
        size_desc = "large, enlarged"
    elif current_mean_area < stats['size_th_small']:
        size_desc = "small"
    else:
        size_desc = "medium-sized"
    
    # 2. Density (对比全局阈值)
    count = len(props)
    if count > stats['dense_th_dense']:
        density_desc = "densely packed"
    elif count < stats['dense_th_sparse']:
        density_desc = "sparsely distributed"
    else:
        density_desc = "moderately distributed"
        
    # 3. Shape (使用偏心率近似形状)
    mean_ecc = np.mean([p.eccentricity for p in props])
    if mean_ecc > 0.8: shape_desc = "elongated, spindle-shaped"
    elif mean_ecc < 0.6: shape_desc = "round, spherical"
    else: shape_desc = "oval"
    
    return {"size": size_desc, "shape": shape_desc, "density": density_desc}

def construct_text_prompt(organ, visuals):
    """构建最终的文本提示，融合语义与视觉特征"""
    kb = ORGAN_KNOWLEDGE.get(organ, ORGAN_KNOWLEDGE["Generic"])
    cell_desc = kb['cell_desc']
    adj = ""
    
    # 规则 1: 恶性肿瘤特征
    if organ in ["Breast", "Kidney", "Lung", "Colon"] and "enlarged" in visuals['size']:
        cell_desc = "Pleomorphic Tumor Nuclei"
        adj = "hyperchromatic"
    # 规则 2: 淋巴细胞特征
    elif "small" in visuals['size'] and "round" in visuals['shape']:
        cell_desc = "Lymphocytes"
        adj = "darkly stained"
    # 规则 3: 腺体特征
    elif organ in ["Prostate", "Colon"] and "dense" in visuals['density']:
        cell_desc = "Glandular Epithelial Nuclei"
        adj = "basally oriented"

    text = (f"Microscopic view of {adj} {visuals['size']} {cell_desc} with {visuals['shape']} features, "
            f"{visuals['density']} in {kb['context']} featuring {kb['structure']}.")
    
    return " ".join(text.split())

# ==============================================================================
# 🚀 主程序 (Main Execution)
# ==============================================================================
def main():
    # 1. 扫描文件
    json_files = glob.glob(os.path.join(DATA_ROOT, "*.json"))
    # 排除非数据 json
    json_files = [f for f in json_files if "knowledge" not in f and "attribute" not in f]
    
    if not json_files:
        print(f"❌ No .json files found in {DATA_ROOT}")
        return

    # 2. 第一遍扫描: 获取全局统计信息 (Pass 1)
    dataset_stats = get_dataset_statistics(json_files)
    
    print(f"🚀 Phase 2: Generating Knowledge Base...")
    kb_database = {}
    
    # 3. 第二遍扫描: 生成具体描述 (Pass 2)
    for json_path in tqdm(json_files):
        filename = os.path.basename(json_path).replace(".json", ".tif")
        organ = get_organ_from_filename(filename)
        
        # ⚠️ 必须使用 decode_instance_mask_from_json 以保持逻辑一致
        instance_mask = decode_instance_mask_from_json(json_path)
        
        visuals = analyze_visuals_dynamic(instance_mask, dataset_stats)
        prompt = construct_text_prompt(organ, visuals)
        
        kb_database[filename] = {
            "organ_id": organ,
            "text_prompt": prompt,
            "visual_stats": visuals
        }
        
    # 4. 保存结果
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(kb_database, f, indent=4)
        
    print(f"✅ Knowledge Base saved to: {OUTPUT_JSON}")
    print("   Ready for data-driven training!")

if __name__ == "__main__":
    main()