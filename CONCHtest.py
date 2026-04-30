import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

# ==========================================
# 1. 基础配置
# ==========================================
HF_TOKEN = ""  # ⚠️ 填入你的 Token
TEST_DIR = "/root/shared-nvme/NuSeg/data/PanNuke/test" # 你的测试集图片路径
JSON_PATH = "/root/shared-nvme/NuSeg/data/PanNuke/medical_knowledge.json" # 你的大 JSON 文件路径
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ 核心修复：建立临床标准名与 PanNuke 标注名的严格映射字典
# Key 是 CONCH 预测时用的标准 Prompt 名字
# Value 是你的 json 文件 (medical_knowledge.json) 中实际存储的名字
ORGAN_MAPPING = {
    "Adrenal Gland": "Adrenal_gland",
    "Bile Duct": "Bile-duct",
    "Bladder": "Bladder",
    "Breast": "Breast",
    "Cervix": "Cervix",
    "Colon": "Colon",
    "Esophagus": "Esophagus",
    "Head and Neck": "HeadNeck",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Ovarian": "Ovarian",
    "Pancreatic": "Pancreatic",
    "Prostate": "Prostate",
    "Skin": "Skin",
    "Stomach": "Stomach",
    "Testis": "Testis",
    "Thyroid": "Thyroid",
    "Uterus": "Uterus"
}

# 提取用于预测的标准临床名字列表
CLINICAL_ORGANS = list(ORGAN_MAPPING.keys())

# 构建专门针对 CONCH 的临床 Prompt
prompts = [f"H&E stained histopathology image of {organ.lower()} tissue" for organ in CLINICAL_ORGANS]

# ==========================================
# 2. 加载模型与大 JSON 文件
# ==========================================
print("Loading medical knowledge JSON...")
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        medical_knowledge = json.load(f)
except Exception as e:
    print(f"❌ 读取 JSON 文件失败，请检查路径: {e}")
    exit()

print("Loading CONCH Model...")
model, preprocess = create_model_from_pretrained(
    'conch_ViT-B-16', 
    "hf_hub:MahmoodLab/conch", 
    hf_auth_token=HF_TOKEN
)
model.to(device)
model.eval()
tokenizer = get_tokenizer()

# ==========================================
# 3. 编码文本 Prompt (直接使用 Tokenizer)
# ==========================================
print("Encoding Text Prompts...")
with torch.no_grad():
    tokenized_output = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    text_tokens = tokenized_output["input_ids"].to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ==========================================
# 4. 遍历测试集进行验证
# ==========================================
png_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
print(f"Found {len(png_files)} images in {TEST_DIR}")

correct_count = 0
total_count = 0
organ_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for img_name in tqdm(png_files, desc="Evaluating"):
    img_path = os.path.join(TEST_DIR, img_name)
    dict_key = f"test/{img_name}"
    
    if dict_key not in medical_knowledge:
        continue
        
    # 从 JSON 获取真实的标签名 (例如 'Adrenal_gland')
    gt_organ_raw = medical_knowledge[dict_key].get("organ_id")
    if not gt_organ_raw:
        continue
            
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
        continue
        
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor, proj_contrast=True, normalize=True)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_features.T
        
        # 获取最大概率的索引 (对应 CLINICAL_ORGANS 的索引)
        pred_idx = logits.argmax(dim=-1).item()
        
        # ⚠️ 核心修复：将预测的临床名字转换回 PanNuke 的专属标签名
        predicted_clinical_name = CLINICAL_ORGANS[pred_idx]
        pred_organ_mapped = ORGAN_MAPPING[predicted_clinical_name]
        
    # 5. 统计正确率 (现在是精确匹配)
    total_count += 1
    # 统一转换为小写进行比对，防止大小写差异
    is_correct = (pred_organ_mapped.lower() == str(gt_organ_raw).lower())
    
    organ_stats[gt_organ_raw]["total"] += 1
    if is_correct:
        correct_count += 1
        organ_stats[gt_organ_raw]["correct"] += 1

# ==========================================
# 6. 打印最终报告
# ==========================================
print("\n" + "="*50)
print("🎯 CONCH Zero-Shot Evaluation Report (PanNuke) - BUG FIXED")
print("="*50)
if total_count > 0:
    print(f"Overall Accuracy: {correct_count / total_count:.2%} ({correct_count}/{total_count})")
    print("-" * 50)
    print("Accuracy per Organ (Sorted):")
    sorted_stats = sorted(organ_stats.items(), key=lambda x: (x[1]["correct"]/x[1]["total"] if x[1]["total"]>0 else 0), reverse=True)
    for organ, stats in sorted_stats:
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f" - {organ.ljust(15)}: {acc:.2%} ({stats['correct']}/{stats['total']})")
else:
    print("⚠️ No valid evaluations completed. Please check your JSON keys and paths.")