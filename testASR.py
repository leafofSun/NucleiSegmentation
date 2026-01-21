import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
import os
import json

# === 🔧 配置区域 ===
# 1. 您的模型权重路径
CHECKPOINT_PATH = "workdir/models/MP_SAM_End2End/best_model.pth" 

# 2. 您的测试图片路径 (请确保对应的 .json 文件在同一目录下，或者在 labels 目录下)
IMAGE_PATH = "data/PanNuke_SA1B/test/sa_0007543.png" 

# 3. 设备
DEVICE = "cuda"

# === 🧠 核心函数：加载 SA-1B 格式真值 ===
def load_data_from_json(img_path):
    """
    根据图片路径寻找并解析 SA-1B 格式的 JSON
    返回: (gt_mask, organ_name)
    """
    # 1. 推断 JSON 路径
    # 策略 A: 同名 json (sa_0007543.png -> sa_0007543.json)
    json_path = os.path.splitext(img_path)[0] + ".json"
    
    # 策略 B: 如果不在同级目录，可能在上一级的 labels 目录 (根据您的数据集结构调整)
    if not os.path.exists(json_path):
        # 尝试把路径中的 /Images/ 替换为 /Labels/ 或 /Jsons/
        # 这是一个常见的猜测，如果您的 json 和 png 都在同一个文件夹，上面的策略 A 就够了
        candidate = img_path.replace("/Images/", "/Labels/").replace(".png", ".json")
        if os.path.exists(candidate):
            json_path = candidate

    if not os.path.exists(json_path):
        print(f"⚠️ 未找到 JSON 文件: {json_path}")
        return None, "Generic"

    # 2. 解析 JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 获取图片尺寸
        h = data.get('height', 256)
        w = data.get('width', 256)
        
        # 获取器官类型 (自动推断 Prompt!)
        organ_type = data.get('organ_type', 'Generic')
        
        # 绘制 Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        annotations = data.get('annotations', [])
        
        for ann in annotations:
            seg = ann.get('segmentation', [])
            # 处理 Polygon 格式: [[x1, y1, x2, y2, ...]]
            for poly in seg:
                # 将扁平列表转为 (N, 2) 坐标点
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)
                
        return mask, organ_type

    except Exception as e:
        print(f"❌ JSON 解析错误: {e}")
        return None, "Generic"

# === 🪝 Hook 函数：偷窥 ASR 内部 ===
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    # 1. 准备参数 (Args)
    # 确保这里的 image_size 与训练时一致
    args = type('Args', (), {
        'image_size': 512, 
        'crop_size': 256, 
        'num_organs': 21,
        'encoder_adapter': True, 
        'sam_checkpoint': None, 
        'checkpoint': None
    })()
    
    print(f"🚀 开始验证: {IMAGE_PATH}")

    # 2. 加载数据与真值
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ 图片读取失败，请检查路径")
        return

    # 获取 GT 和 器官类型
    gt_mask, organ_name = load_data_from_json(IMAGE_PATH)
    print(f"📋 自动识别器官: {organ_name}")
    
    # 预处理图片
    image_input = cv2.resize(image, (512, 512))
    img_tensor = torch.from_numpy(image_input).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
    
    # 预处理 GT (用于可视化)
    if gt_mask is not None:
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # 3. 构建模型
    vanilla_sam = sam_model_registry["vit_b"](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        num_organs=21
    ).to(DEVICE)
    
    # 加载权重
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt, strict=False)
        print("✅ 权重加载成功")
    else:
        print(f"❌ 权重未找到: {CHECKPOINT_PATH}")
        return
        
    model.eval()

    # 4. 注册 ASR 探针
    # 注意: 路径可能根据您的代码微调，通常是 model.mask_decoder.asr_upscale_1.gate[1]
    if hasattr(model.mask_decoder, 'asr_upscale_1'):
        model.mask_decoder.asr_upscale_1.gate[1].register_forward_hook(get_activation('asr_gate'))
        print("🪝 已挂载 ASR Gate 探针")

    # 5. 构造 Input (自动使用 JSON 里的器官名)
    # 简单的 Organ ID 映射 (根据您的 ID_TO_ORGAN 字典)
    ORGAN_TO_ID = {
        "Kidney": 0, "Breast": 1, "Prostate": 2, "Lung": 3, 
        "Colon": 4, "Stomach": 5, "Liver": 6, "Bladder": 7, 
        "Brain": 8, "Generic": 9
    }
    organ_id = ORGAN_TO_ID.get(organ_name, 9) # 默认为 Generic

    input_sample = [{
        'image': img_tensor.squeeze(0),
        'original_size': (512, 512),
        'text_prompt': f"{organ_name} cell nuclei", # 自动生成 Prompt
        'organ_id': organ_id,
        'attribute_text': f"{organ_name} cell nuclei"
    }]

    # 6. 推理
    with torch.no_grad():
        outputs = model(input_sample, multimask_output=True)
    
    # 7. 提取结果
    res = outputs[0]
    # 取最高分的 Mask
    best_idx = torch.argmax(res['iou_predictions'])
    pred_mask = res['masks'][0, best_idx].cpu().numpy()
    
    # 提取 SAR Density Map
    density_map = res.get('density_map', torch.zeros(1, 512, 512)).squeeze().cpu().numpy()
    
    # 提取 ASR Gate Map
    gate_map = activation.get('asr_gate', torch.zeros(1, 1, 512, 512))
    gate_map = gate_map.mean(dim=0).squeeze().cpu().numpy() # [H, W]
    gate_map = cv2.resize(gate_map, (512, 512))

    # 8. 绘图 (五联图：原图 - GT - 预测 - 语义 - 门控)
    plt.figure(figsize=(25, 5))
    
    # A. 原图
    plt.subplot(1, 5, 1)
    plt.title(f"Image ({organ_name})")
    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # B. Ground Truth (真值)
    plt.subplot(1, 5, 2)
    plt.title("Ground Truth")
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap='gray')
    else:
        plt.text(0.5, 0.5, "GT Not Found", ha='center')
    plt.axis('off')

    # C. Prediction (预测)
    plt.subplot(1, 5, 3)
    plt.title("Prediction")
    plt.imshow(pred_mask > 0, cmap='gray')
    plt.axis('off')

    # D. SAR Density (你的“学霸”指挥官)
    plt.subplot(1, 5, 4)
    plt.title("SAR Density (Semantic)")
    plt.imshow(density_map, cmap='jet') # 红=高密度, 蓝=低密度
    plt.axis('off')

    # E. ASR Gate (你的“学渣”执行器)
    plt.subplot(1, 5, 5)
    plt.title("ASR Gate (Current State)")
    plt.imshow(gate_map, cmap='magma') # 亮=高频注入, 暗=平滑
    plt.axis('off')

    save_path = "verification_final.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📸 验证完成！结果已保存至: {save_path}")
    print("👉 请重点对比 [Ground Truth] vs [ASR Gate] 以及 [SAR Density] vs [ASR Gate]")

if __name__ == "__main__":
    main()