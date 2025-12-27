import torch
import cv2
import os
import numpy as np
import argparse
from segment_anything import sam_model_registry
# 假设 TextSam 在 segment_anything.modeling.sam 中
from segment_anything.modeling.sam import TextSam
import matplotlib.pyplot as plt
import torch.nn.functional as F

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="workdir/models/text-guided-sam/best_model.pth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a test image")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument("--sam_checkpoint", type=str, default=None) 
    cmd_args = parser.parse_args()

    # 1. 加载模型
    print(f"Loading model from {cmd_args.model_path}...")
    vanilla_sam = sam_model_registry["vit_b"](cmd_args) 
    
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(cmd_args.device)
    
    try:
        checkpoint = torch.load(cmd_args.model_path, map_location=cmd_args.device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()

    # 2. 读取图片
    if not os.path.exists(cmd_args.image_path):
        print(f"Error: Image not found at {cmd_args.image_path}")
        return
        
    image = cv2.imread(cmd_args.image_path)
    if image is None:
        print(f"Error: Failed to read image {cmd_args.image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CenterCrop
    h, w = image_rgb.shape[:2]
    crop_size = cmd_args.image_size
    
    if h < crop_size or w < crop_size:
        image_crop = cv2.resize(image_rgb, (crop_size, crop_size))
    else:
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        image_crop = image_rgb[start_h:start_h+crop_size, start_w:start_w+crop_size]

    # 转 Tensor
    img_tensor = torch.tensor(image_crop).permute(2, 0, 1).float().unsqueeze(0).to(cmd_args.device)
    model_input = [{'image': img_tensor[0], 'original_size': (crop_size, crop_size)}]

    # 3. 预测
    print("Running inference...")
    with torch.no_grad():
        outputs = model(model_input, multimask_output=True)
    
    # 4. 解析结果
    out = outputs[0]
    
    # [核心修复] 处理维度
    # Masks Shape: [1, 3, 256, 256] -> Squeeze Batch Dim -> [3, 256, 256]
    masks = out['masks']
    if masks.ndim == 4:
        masks = masks.squeeze(0)
        
    # Scores Shape: [3] (或者 [1, 3])
    scores = out['iou_predictions']
    if scores.ndim == 2:
        scores = scores.squeeze(0)
    
    print(f"🔍 Processed Shapes -> Masks: {masks.shape}, Scores: {scores.shape}")

    # 取出分数最高的索引
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    
    # 根据索引取 Mask: [3, 256, 256] -> [256, 256]
    mask_logits = masks[best_idx]
    
    # Sigmoid + Threshold
    prob_map = torch.sigmoid(mask_logits).cpu().numpy()
    mask_binary = (prob_map > 0.5).astype(np.uint8) 
    
    # Heatmap
    if 'heatmap_logits' in out:
        # heatmap_logits 可能也是 [1, 1, 64, 64]
        heatmap_tensor = out['heatmap_logits']
        if heatmap_tensor.ndim == 4:
            heatmap_tensor = heatmap_tensor.squeeze(0) # [1, 64, 64]
        heatmap = torch.sigmoid(heatmap_tensor[0]).cpu().numpy()
    else:
        heatmap = np.zeros_like(prob_map)

    # 5. 绘图
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_crop)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Text-Guided Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(prob_map, cmap='gray')
    plt.title(f"Prob Map (Score: {best_score:.3f})")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(image_crop)
    show_mask(mask_binary, plt.gca())
    plt.title("Final Overlay")
    plt.axis('off')
    
    save_name = "debug_result.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"✅ Result saved to {save_name}. Please check it!")

if __name__ == "__main__":
    main()