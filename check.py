import os
import json
import cv2
import torch
from PIL import Image
import clip
from tqdm import tqdm
from collections import defaultdict

# 复制自 test.py
ID_TO_ORGAN = {
    0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder", 3: "Breast", 
    4: "Cervix", 5: "Colon", 6: "Esophagus", 7: "HeadNeck", 
    8: "Kidney", 9: "Liver", 10: "Lung", 11: "Ovarian", 
    12: "Pancreatic", 13: "Prostate", 14: "Skin", 15: "Stomach", 
    16: "Testis", 17: "Thyroid", 18: "Uterus", 19: "Brain", 20: "Generic"
}

class OrganPredictor:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.valid_ids = sorted(list(ID_TO_ORGAN.keys())) 
        self.organs = [ID_TO_ORGAN[i] for i in self.valid_ids]
        with torch.no_grad():
            self.text_features = self.model.encode_text(clip.tokenize([f"histology {org}" for org in self.organs]).to(device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_cv2):
        img_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        img_in = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img_in)
            feat /= feat.norm(dim=-1, keepdim=True)
            idx = (feat @ self.text_features.T).argmax().item()
        return self.organs[idx], self.valid_ids[idx]

def main():
    data_root = "data/PanNuke"
    knowledge_path = "data/PanNuke/medical_knowledge.json"
    
    print(f"📖 加载知识库: {knowledge_path}...")
    with open(knowledge_path, 'r') as f:
        full_db = json.load(f)
        
    if "__meta__" in full_db:
        full_db.pop("__meta__")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 初始化 CLIP OrganPredictor...")
    predictor = OrganPredictor(device)
    
    correct = 0
    total = 0
    organ_correct = defaultdict(int)
    organ_total = defaultdict(int)

    # 仅遍历测试集 (test split)
    test_items = {k: v for k, v in full_db.items() if v.get('split') == 'test'}
    print(f"📦 发现 {len(test_items)} 张测试集图像。")

    for rel_path, entry in tqdm(test_items.items(), desc="评估 CLIP 准确率"):
        img_path = os.path.join(data_root, rel_path)
        if not os.path.exists(img_path):
            continue
            
        # 获取真实的器官标签
        gt_organ_name = entry.get('organ_id', 'Generic')
        
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
            
        # CLIP 预测
        pred_organ_name, _ = predictor.predict(image_bgr)
        
        total += 1
        organ_total[gt_organ_name] += 1
        
        if pred_organ_name == gt_organ_name:
            correct += 1
            organ_correct[gt_organ_name] += 1

    if total == 0:
        print("❌ 未找到任何测试图像，请检查 data_root 路径。")
        return

    overall_acc = correct / total * 100
    print("\n" + "="*55)
    print(f"🌟 整体 CLIP Zero-Shot 准确率: {overall_acc:.2f}% ({correct}/{total})")
    print("="*55)
    print("📊 各器官分类准确率 (Per-Organ Accuracy):")
    
    for org in sorted(organ_total.keys()):
        acc = organ_correct[org] / organ_total[org] * 100
        print(f" - {org.ljust(15)}: {acc:>6.2f}%  ({organ_correct[org]}/{organ_total[org]})")

if __name__ == '__main__':
    main()