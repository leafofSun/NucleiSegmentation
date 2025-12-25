import json
import os
import glob
import numpy as np
from tqdm import tqdm

# === 1. 配置路径 ===
# 指向您转换后的 SA-1B 格式数据根目录
# 基于脚本所在目录计算项目根目录（scripts 的父目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/MoNuSeg_SA1B")
OUTPUT_JSON_PATH = os.path.join(DATA_ROOT, "attribute_info_train.json")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")

# === 2. 属性定义 ===
ATTR_CATEGORIES = {
    "color": ["deep purple", "light pink", "purple"], 
    "shape": ["elliptical/oval", "round", "irregular", "spindle", "elongated"], 
    "arrange": ["scattered", "clustered", "linear", "regular"], 
    "size": ["small", "medium", "large"], 
    "density": ["sparsely distributed", "moderately dense", "densely packed"],
    "organ": ["breast", "kidney", "liver", "prostate", "bladder", "colon", "stomach"]
}

# === 3. MoNuSeg 器官映射表 (保持不变) ===
MONUSEG_MAP = {
    "TCGA-A7": {"organ": "breast", "target": "breast invasive carcinoma nuclei", "shape": "irregular", "arrange": "clustered"},
    "TCGA-AR": {"organ": "breast", "target": "breast invasive carcinoma nuclei", "shape": "irregular", "arrange": "clustered"},
    "TCGA-E2": {"organ": "breast", "target": "breast invasive carcinoma nuclei", "shape": "irregular", "arrange": "clustered"},
    "TCGA-B0": {"organ": "kidney", "target": "kidney renal clear cell carcinoma nuclei", "shape": "round", "arrange": "clustered"},
    "TCGA-HE": {"organ": "kidney", "target": "kidney renal papillary cell carcinoma nuclei", "shape": "round", "arrange": "regular"},
    "TCGA-18": {"organ": "liver", "target": "lung squamous cell carcinoma nuclei in liver tissue", "shape": "round", "arrange": "scattered"},
    "TCGA-38": {"organ": "liver", "target": "lung adenocarcinoma nuclei in liver tissue", "shape": "round", "arrange": "scattered"},
    "TCGA-49": {"organ": "liver", "target": "lung adenocarcinoma nuclei in liver tissue", "shape": "round", "arrange": "scattered"},
    "TCGA-50": {"organ": "liver", "target": "lung adenocarcinoma nuclei in liver tissue", "shape": "round", "arrange": "scattered"},
    "TCGA-21": {"organ": "liver", "target": "lung squamous cell carcinoma nuclei in liver tissue", "shape": "irregular", "arrange": "scattered"},
    "TCGA-G9": {"organ": "prostate", "target": "prostate adenocarcinoma nuclei", "shape": "round", "arrange": "clustered"},
    "TCGA-CH": {"organ": "prostate", "target": "prostate adenocarcinoma nuclei", "shape": "round", "arrange": "clustered"},
    "TCGA-DK": {"organ": "bladder", "target": "bladder urothelial carcinoma nuclei", "shape": "irregular", "arrange": "scattered"},
    "TCGA-G2": {"organ": "bladder", "target": "bladder urothelial carcinoma nuclei", "shape": "irregular", "arrange": "scattered"},
    "TCGA-AY": {"organ": "colon", "target": "colon adenocarcinoma nuclei", "shape": "elongated", "arrange": "regular"},
    "TCGA-NH": {"organ": "colon", "target": "colon adenocarcinoma nuclei", "shape": "elongated", "arrange": "regular"},
    "TCGA-KB": {"organ": "stomach", "target": "stomach adenocarcinoma nuclei", "shape": "irregular", "arrange": "scattered"},
    "TCGA-RD": {"organ": "stomach", "target": "stomach adenocarcinoma nuclei", "shape": "irregular", "arrange": "scattered"}
}

def get_one_hot(value, category_list):
    label = [0] * len(category_list)
    if value in category_list:
        idx = category_list.index(value)
        label[idx] = 1
    else:
        label[0] = 1 
    return label

def analyze_json_stats(json_data):
    """
    直接从 SA-1B JSON 中读取统计信息
    不需要解码 RLE，直接用 'area' 字段，速度极快
    """
    annotations = json_data.get('annotations', [])
    num_cells = len(annotations)
    
    if num_cells == 0:
        return "small", "sparsely distributed"

    # 1. 计算平均大小 (Size)
    # convert_monuseg_to_sa1b.py 已经计算了 'area'
    areas = [ann['area'] for ann in annotations if 'area' in ann]
    avg_area = np.mean(areas) if areas else 0
    
    if avg_area < 250: size = "small"
    elif avg_area < 550: size = "medium"
    else: size = "large"

    # 2. 计算密度 (Density)
    # 需要知道原图尺寸来计算比例
    img_h = json_data['image']['height']
    img_w = json_data['image']['width']
    
    # 归一化到 256x256 的密度标准
    area_ratio = (img_h * img_w) / (256 * 256)
    
    if num_cells < 20 * area_ratio: density = "sparsely distributed"
    elif num_cells < 60 * area_ratio: density = "moderately dense"
    else: density = "densely packed"
    
    return size, density

def main():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: 找不到训练集目录: {TRAIN_DIR}")
        print("   请修改脚本中的 DATA_ROOT 为您 convert_monuseg_to_sa1b.py 输出的目录")
        return

    # 查找所有 JSON 文件 (SA-1B 格式的核心)
    json_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.json")))
    
    print(f"🚀 开始处理 SA-1B 格式数据...")
    print(f"   数据目录: {TRAIN_DIR}")
    print(f"   找到 JSON 文件数: {len(json_files)}")

    attribute_info = {}

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading {json_path}: {e}")
            continue

        filename = data['image']['file_name'] # 从 JSON 中获取准确的图片文件名
        
        # A. 动态计算属性 (基于 JSON 中的标注数据)
        size_val, density_val = analyze_json_stats(data)
        
        # B. 静态属性映射 (基于文件名 TCGA ID)
        color_val = "deep purple"
        shape_val = "elliptical/oval"
        arrange_val = "scattered"
        organ_val = "breast"
        target_val = "cell nuclei"
        
        for prefix, info in MONUSEG_MAP.items():
            if prefix in filename:
                shape_val = info["shape"]
                arrange_val = info["arrange"]
                organ_val = info["organ"]
                target_val = info["target"]
                break
        
        # C. 构造输出
        prompts = [color_val, shape_val, arrange_val, size_val, density_val, organ_val, target_val]
        
        labels = [
            get_one_hot(color_val, ATTR_CATEGORIES["color"]),
            get_one_hot(shape_val, ATTR_CATEGORIES["shape"]),
            get_one_hot(arrange_val, ATTR_CATEGORIES["arrange"]),
            get_one_hot(size_val, ATTR_CATEGORIES["size"]),
            get_one_hot(density_val, ATTR_CATEGORIES["density"]),
            get_one_hot(organ_val, ATTR_CATEGORIES["organ"])
        ]
        
        # 以图片文件名作为 Key
        attribute_info[filename] = {
            "attribute_prompts": prompts,
            "attribute_labels": labels,
            "target_text": target_val
        }

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(attribute_info, f, indent=4)
        
    print(f"✅ 完成！SA-1B 格式属性文件已生成: {OUTPUT_JSON_PATH}")
    if attribute_info:
        k = list(attribute_info.keys())[0]
        print(f"   示例 ({k}): {attribute_info[k]['attribute_prompts']}")

if __name__ == "__main__":
    main()