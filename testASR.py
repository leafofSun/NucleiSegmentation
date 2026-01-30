import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
import os
import json
import torch.nn.functional as F

# === 🔧 配置区域 (严格保留您的原始设置) ===
# 1. 您的模型权重路径
CHECKPOINT_PATH = "workdir/models/MP_SAM_Final_Clean_Run/best_model.pth" 

# 2. 您的测试图片路径
IMAGE_PATH = "data/PanNuke/test/sa_0007543.png" 

# 3. 设备
DEVICE = "cuda"

# === 🧠 21 类器官映射 ===
ORGAN_TO_ID = {
    "Kidney": 0, "Breast": 1, "Prostate": 2, "Lung": 3, 
    "Colon": 4, "Stomach": 5, "Liver": 6, "Bladder": 7, 
    "Brain": 8, "Thyroid": 9, "Testis": 10, "Skin": 11, 
    "Ovary": 12, "Uterus": 13, "Pancreas": 14, "Adrenal_gland": 15,
    "Esophagus": 16, "Gallbladder": 17, "Larynx": 18, "Parotid_gland": 19,
    "Generic": 20, "Other": 20
}

def load_data_from_json(img_path):
    """根据图片路径寻找并解析 SA-1B 格式的 JSON"""
    json_path = os.path.splitext(img_path)[0] + ".json"
    
    if not os.path.exists(json_path):
        candidate = img_path.replace("/Images/", "/Labels/").replace(".png", ".json")
        if os.path.exists(candidate):
            json_path = candidate

    if not os.path.exists(json_path):
        print(f"⚠️ 未找到 JSON 文件，将使用默认器官 'Generic'")
        return None, "Generic"

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        h = data.get('height', 256)
        w = data.get('width', 256)
        organ_type = data.get('organ_type', 'Generic')
        
        mask = np.zeros((h, w), dtype=np.uint8)
        annotations = data.get('annotations', [])
        
        for ann in annotations:
            seg = ann.get('segmentation', [])
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)
                
        return mask, organ_type

    except Exception as e:
        print(f"❌ JSON 解析错误: {e}")
        return None, "Generic"

# === 🪝 Hook 函数 ===
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    # 1. 准备参数 (Args)
    args = type('Args', (), {
        'image_size': 512, 
        'crop_size': 512, 
        'num_organs': 21,
        'encoder_adapter': True, 
        'sam_checkpoint': None, 
        'checkpoint': None,     # 🔥 已修复
        'clip_model': "ViT-B/16" 
    })()
    
    print(f"🚀 开始验证: {IMAGE_PATH}")

    # 2. 加载数据与真值
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ 图片读取失败，请检查路径")
        return

    gt_mask, organ_name = load_data_from_json(IMAGE_PATH)
    
    # 获取 ID
    organ_id = ORGAN_TO_ID.get(organ_name, 20)
    print(f"📋 识别器官: {organ_name} -> ID: {organ_id}")
    
    # 预处理
    image_input = cv2.resize(image, (512, 512))
    img_tensor = torch.from_numpy(image_input).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    
    if gt_mask is not None:
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # 3. 构建模型
    vanilla_sam = sam_model_registry["vit_b"](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        num_organs=21, 
        clip_model_name="ViT-B/16"
    ).to(DEVICE)
    
    # 加载权重
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt, strict=False)
        print("✅ 权重加载成功")
    else:
        print(f"❌ 权重未找到: {CHECKPOINT_PATH}")
        return
        
    model.eval()

    # 4. 注册 ASR 探针
    if hasattr(model.mask_decoder, 'asr_upscale_1'):
        model.mask_decoder.asr_upscale_1.gate.register_forward_hook(get_activation('asr_gate'))
        print("🪝 已挂载 ASR Gate 探针")

    # 5. 构造 Input
    input_sample = [{
        'image': img_tensor.squeeze(0),
        'original_size': (512, 512),
        'text_prompt': f"{organ_name} cell nuclei", 
        'organ_id': organ_id,
        'attribute_text': f"{organ_name} cell nuclei"
    }]

    # 6. 推理
    with torch.no_grad():
        outputs = model(input_sample, multimask_output=True)
    
    # 7. 提取结果
    res = outputs[0]
    best_idx = torch.argmax(res['iou_predictions'])
    pred_mask = res['masks'][0, best_idx].cpu().numpy()
    
    # (A) SAR Density Map
    density_map = res.get('density_map', torch.zeros(1, 512, 512)).squeeze().cpu().numpy()
    
    # 🔥 (B) Heatmap Logits (点提示) - [修复维度问题]
    heatmap_raw = res.get('heatmap_logits', torch.zeros(1, 2, 256, 256)).detach()
    heatmap_vis = F.interpolate(heatmap_raw, size=(512, 512), mode='bilinear', align_corners=False).sigmoid().cpu().numpy()
    heatmap_vis = heatmap_vis.squeeze() # 可能变成 [2, 512, 512]
    
    if heatmap_vis.ndim == 3 and heatmap_vis.shape[0] == 2:
        print("🔍 提取双通道热力图中的 Positive 通道")
        heatmap_vis = heatmap_vis[0, :, :] # 取第一个通道 (Cell nuclei)
    
    # (C) ASR Gate Map
    gate_map = activation.get('asr_gate', torch.zeros(1, 1, 512, 512))
    gate_map = gate_map.mean(dim=0).squeeze().cpu().numpy() 
    gate_map = cv2.resize(gate_map, (512, 512))

    # 8. 绘图 (六联图)
    plt.figure(figsize=(30, 5))
    
    titles = ["Image", "Ground Truth", "Heatmap (Points)", "Density (Clouds)", "ASR Gate", "Prediction"]
    images_list = [
        cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB),
        gt_mask if gt_mask is not None else np.zeros((512,512)),
        heatmap_vis, 
        density_map,
        gate_map,
        pred_mask > 0
    ]
    cmaps = [None, 'gray', 'inferno', 'jet', 'magma', 'gray']

    for i in range(6):
        plt.subplot(1, 6, i+1)
        # 安全获取最大值
        max_val = images_list[i].max() if images_list[i] is not None else 0
        plt.title(f"{titles[i]}\n(Max: {max_val:.2f})")
        
        img_show = images_list[i]
        # 归一化显示
        if i in [2, 3, 4] and max_val > 0:
            img_show = (img_show - img_show.min()) / (max_val - img_show.min() + 1e-6)
            
        plt.imshow(img_show, cmap=cmaps[i])
        plt.axis('off')

    save_path = "verification_final.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📸 验证完成！结果已保存至: {save_path}")

if __name__ == "__main__":
    main()