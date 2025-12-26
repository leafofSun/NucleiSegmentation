import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import json
import logging
from collections import defaultdict

# === Imports ===
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import TestingDataset
from utils import save_masks
from metrics import SegMetrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="test data path") 
    parser.add_argument("--metrics", nargs='+', default=['mDice', 'mAJI', 'mPQ'], help="metrics to calc")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained .pth checkpoint")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter in image encoder")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="pre-trained sam weights")
    parser.add_argument("--auto_prompt", action='store_true', default=True, help="使用模型自动生成的提示")
    parser.add_argument("--save_pred", action='store_true', help="保存预测结果图片")
    args = parser.parse_args()
    return args

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                device_input[key] = value.to(device)
            elif isinstance(value, list):
                device_input[key] = value
            else:
                device_input[key] = value
    return device_input

def main(args):
    print('*'*50)
    print(f"🚀 Running Inference: {args.run_name}")
    print('*'*50)

    # 1. 构建模型
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    
    # 2. 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. 数据加载
    test_dataset = TestingDataset(
        data_path=args.data_path, 
        image_size=args.image_size, 
        mode='test', 
        requires_name=True, 
        point_num=1, 
        return_ori_mask=True,
        attribute_info_path=None 
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    all_metrics = defaultdict(list)

    # 4. 推理循环
    pbar = tqdm(test_loader, desc="Inference")
    
    # 定义 ImageNet 反归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(args.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(args.device)

    for i, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        img_tensor = batched_input["image"][0] # [3, H, W]

        # === 核心修复：智能反归一化 ===
        # 如果最大值很小（比如 < 5.0），说明被归一化过了，必须还原回 [0, 255]
        if img_tensor.max() < 5.0:
            # 反归一化: x * std + mean
            img_tensor = img_tensor * std + mean
            # 缩放到 0-255
            img_tensor = torch.clamp(img_tensor * 255.0, 0, 255)
        
        # 调试打印 (只打印第一张确认修复)
        if i == 0:
            print(f"\n[Fixed Input] Range: Min={img_tensor.min():.2f}, Max={img_tensor.max():.2f} (Should be 0-255)")

        ori_labels = batched_input["ori_label"] 
        gt_mask = ori_labels.cpu().numpy() 
        while gt_mask.ndim > 2: gt_mask = gt_mask[0] # 强力降维
            
        img_name = batched_input['name'][0]
        original_size = batched_input["original_size"] 

        sam_input = [{
            'image': img_tensor, # 传入修复后的 Tensor
            'original_size': (original_size[0].item(), original_size[1].item())
        }]

        with torch.no_grad():
            if args.auto_prompt:
                outputs = model(sam_input, multimask_output=True)
                out = outputs[0]
                
                if out['masks'].ndim == 4:
                    pred_logits = out['masks'][0, 0, :, :] 
                else:
                    pred_logits = out['masks'][0, :, :]
                
                heatmap = out['heatmap_logits'] 
                
                # 调试打印输出分布
                if i == 0:
                    prob = torch.sigmoid(pred_logits)
                    print(f"[Fixed Output] Pred Max Prob: {prob.max():.4f}, Heatmap Max: {heatmap[0].max():.4f}")

            else:
                continue

        # 后处理
        pred_prob = torch.sigmoid(pred_logits).cpu().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.uint8)
        
        if pred_binary.shape != gt_mask.shape:
            pred_binary = cv2.resize(pred_binary, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 计算指标
        try:
            batch_res = SegMetrics(
                pred_binary[np.newaxis, ...], 
                gt_mask[np.newaxis, ...], 
                args.metrics
            )
            for k, v in batch_res.items():
                all_metrics[k].append(v)
        except Exception as e:
            print(f"Error metrics: {e}")
            continue
            
        # 可视化
        if args.save_pred:
            save_dir = os.path.join(args.work_dir, args.run_name, "viz_results")
            os.makedirs(save_dir, exist_ok=True)
            
            # 使用修复后的 img_tensor 来做图，这样颜色才正常
            vis_img = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            vis_img = cv2.resize(vis_img, (gt_mask.shape[1], gt_mask.shape[0])) 
            
            contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2) 
            
            gt_bin_vis = (gt_mask > 0).astype(np.uint8)
            contours_gt, _ = cv2.findContours(gt_bin_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours_gt, -1, (0, 0, 255), 1)

            if 'heatmap' in locals():
                hm = heatmap[0].cpu().numpy() 
                hm = cv2.resize(hm, (gt_mask.shape[1], gt_mask.shape[0]))
                hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
                hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                final_vis = np.hstack([vis_img, hm_color])
            else:
                final_vis = vis_img
                
            save_name = os.path.splitext(img_name)[0] + ".jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), final_vis)

    # 5. 打印结果
    print("\n" + "="*40)
    print(f"📊 Evaluation Report: {args.run_name}")
    print("="*40)
    for k, v in all_metrics.items():
        if len(v) > 0:
            print(f"{k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)