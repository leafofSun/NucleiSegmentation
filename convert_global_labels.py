import os
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
from skimage import measure
from typing import Dict, List

# ==============================================================================
# 1. 扩展医学先验库 (保持不变)
# ==============================================================================
DEFAULT_ORGAN_KNOWLEDGE = {
    # --- PanNuke 19 类 ---
    "Adrenal_gland": {"context": "Adrenal tissue", "desc": "Adrenocortical cells"},
    "Bile-duct": {"context": "Biliary tissue", "desc": "Cholangiocytes"},
    "Bladder": {"context": "Urothelial tissue", "desc": "Transitional epithelial cells"},
    "Breast": {"context": "Mammary tissue", "desc": "Ductal epithelial cells"},
    "Cervix": {"context": "Cervical tissue", "desc": "Squamous epithelial cells"},
    "Colon": {"context": "Colonic mucosa", "desc": "Columnar epithelial cells"},
    "Esophagus": {"context": "Esophageal tissue", "desc": "Squamous cells"},
    "HeadNeck": {"context": "Head and Neck tissue", "desc": "Squamous epithelial cells"},
    "Kidney": {"context": "Renal tissue", "desc": "Tubular epithelial cells"},
    "Liver": {"context": "Hepatic tissue", "desc": "Hepatocytes"},
    "Lung": {"context": "Pulmonary tissue", "desc": "Pneumocytes and macrophages"},
    "Ovarian": {"context": "Ovarian tissue", "desc": "Stromal and epithelial cells"},
    "Pancreatic": {"context": "Pancreatic tissue", "desc": "Acinar cells"},
    "Prostate": {"context": "Prostatic tissue", "desc": "Glandular epithelial cells"},
    "Skin": {"context": "Cutaneous tissue", "desc": "Keratinocytes"},
    "Stomach": {"context": "Gastric mucosa", "desc": "Glandular cells"},
    "Testis": {"context": "Testicular tissue", "desc": "Germ cells"},
    "Thyroid": {"context": "Thyroid tissue", "desc": "Follicular cells"},
    "Uterus": {"context": "Uterine tissue", "desc": "Endometrial cells"},
    # --- MoNuSeg 补充 ---
    "Brain": {"context": "Brain tissue", "desc": "Glial cells and neurons"},
    # --- 通用兜底 ---
    "Generic": {"context": "Histopathology tissue", "desc": "Nuclei"}
}

# ==============================================================================
# 2. 统计分析器 (保持不变)
# ==============================================================================
class DatasetAnalyzer:
    def __init__(self):
        self.areas = []
        self.counts = []
        self.stats = {}

    def update(self, mask: np.ndarray):
        props = measure.regionprops(mask)
        count = len(props)
        self.counts.append(count)
        for p in props:
            if 10 < p.area < 10000:
                self.areas.append(p.area)

    def compute_global_stats(self):
        if not self.areas:
            print("⚠️ Warning: No valid nuclei found for stats.")
            return

        areas_np = np.array(self.areas)
        counts_np = np.array(self.counts)

        mu_size, std_size = np.mean(areas_np), np.std(areas_np)
        mu_dens, std_dens = np.mean(counts_np), np.std(counts_np)

        # 严格遵循 Paper 逻辑
        self.stats = {
            "size_mean": float(mu_size),
            "th_size_large": float(mu_size + 2 * std_size),
            "th_size_small": float(max(0, mu_size - 0.5 * std_size)),
            "th_dens_sparse": float(max(0, mu_dens - 1.0 * std_dens)),
            "th_dens_dense": float(mu_dens + 1.0 * std_dens)
        }
        
        print("\n📊 [PromptNu Full-Dataset Statistics Report]")
        print(f"   Nuclei Size (px): Mean={mu_size:.1f}, Std={std_size:.1f}")
        print(f"   -> Small < {self.stats['th_size_small']:.1f} | Large > {self.stats['th_size_large']:.1f}")
        print(f"   Nuclei Count/Img: Mean={mu_dens:.1f}, Std={std_dens:.1f}")
        print(f"   -> Sparse < {self.stats['th_dens_sparse']:.1f} | Dense > {self.stats['th_dens_dense']:.1f}\n")

    def analyze_single_image(self, mask: np.ndarray) -> Dict[str, str]:
        if not self.stats or mask.max() == 0:
            return {"size": "medium", "density": "moderate", "shape": "round"}

        props = measure.regionprops(mask)
        if not props:
            return {"size": "medium", "density": "moderate", "shape": "round"}

        local_mean_area = np.mean([p.area for p in props])
        if local_mean_area < self.stats['th_size_small']: size_txt = "small"
        elif local_mean_area > self.stats['th_size_large']: size_txt = "large, enlarged"
        else: size_txt = "medium-sized"

        count = len(props)
        if count < self.stats['th_dens_sparse']: dens_txt = "sparsely distributed"
        elif count > self.stats['th_dens_dense']: dens_txt = "densely packed"
        else: dens_txt = "moderately distributed"

        mean_ecc = np.mean([p.eccentricity for p in props])
        if mean_ecc < 0.6: shape_txt = "round"
        elif mean_ecc > 0.85: shape_txt = "elongated"
        else: shape_txt = "oval"

        return {"size": size_txt, "density": dens_txt, "shape": shape_txt}

# ==============================================================================
# 3. 知识生成器 (修正版)
# ==============================================================================
class KnowledgeGenerator:
    def __init__(self, data_root, output_path):
        self.data_root = data_root
        self.output_path = output_path
        self.analyzer = DatasetAnalyzer()
        self.kb_database = {}

    def _decode_mask(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            h = data.get("height", 256)
            w = data.get("width", 256)
            mask = np.zeros((h, w), dtype=np.int32)
            organ_type = data.get("organ_type", "Generic")
            
            for i, ann in enumerate(data.get("annotations", [])):
                for poly in ann.get("segmentation", []):
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], i + 1)
            
            return mask, organ_type
        except Exception as e:
            # print(f"Error reading {json_path}: {e}")
            return None, "Generic"

    def run(self):
        # 递归扫描 train 和 test
        search_path = os.path.join(self.data_root, "**", "*.json")
        all_files = glob.glob(search_path, recursive=True)
        all_files = [f for f in all_files if "knowledge" not in f]
        
        if not all_files:
            print(f"❌ No JSON files found in {self.data_root}")
            return

        print(f"🚀 Found {len(all_files)} samples. Starting FULL analysis...")

        # === Phase 1: Global Statistics ===
        for fpath in tqdm(all_files, desc="Stats Analysis"):
            mask, _ = self._decode_mask(fpath)
            if mask is not None:
                self.analyzer.update(mask)
        
        self.analyzer.compute_global_stats()

        # === Phase 2: Generate Prompts ===
        for fpath in tqdm(all_files, desc="Prompt Gen"):
            mask, organ_raw = self._decode_mask(fpath)
            if mask is None: continue

            organ_key = "Generic"
            for known_organ in DEFAULT_ORGAN_KNOWLEDGE.keys():
                if known_organ.lower() in organ_raw.lower():
                    organ_key = known_organ
                    break
            
            med_info = DEFAULT_ORGAN_KNOWLEDGE.get(organ_key, DEFAULT_ORGAN_KNOWLEDGE["Generic"])
            visuals = self.analyzer.analyze_single_image(mask)
            
            prompt = (f"Microscopic view of {visuals['density']}, {visuals['size']} {med_info['desc']} "
                      f"with {visuals['shape']} features, in {med_info['context']}.")
            prompt = " ".join(prompt.split())
            
            # 🔥 [修正 1] 使用相对路径作为 Key
            # 例如: "train/sa_00001.png"
            # 这样 DataLoader 拼接 data_root + key 才能找到文件
            rel_path = os.path.relpath(fpath, self.data_root) # e.g. train/sa_001.json
            img_key = rel_path.replace(".json", ".png")       # e.g. train/sa_001.png
            
            self.kb_database[img_key] = {
                "organ_id": organ_key,
                "text_prompt": prompt,
                "visual_stats": visuals,
                "split": "train" if "train" in fpath else "test"
            }

        # 🔥 [修正 2] 写入元数据 __meta__
        # 这对于 DataLoader 动态配置至关重要！
        self.kb_database["__meta__"] = {
            "stats": self.analyzer.stats
        }

        # === Save ===
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.kb_database, f, indent=4)
        
        print(f"✅ Success! Knowledge base saved to: {self.output_path}")

# ==============================================================================
# 4. 执行入口
# ==============================================================================
if __name__ == "__main__":
    # 配置你的数据集根目录 (SA-1B 格式的根目录，包含 train/ 和 test/ 文件夹)
    ROOT_DIR = "data/PanNuke_SA1B" 
    SAVE_PATH = os.path.join(ROOT_DIR, "medical_knowledge.json")
    
    generator = KnowledgeGenerator(ROOT_DIR, SAVE_PATH)
    generator.run()