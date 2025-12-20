import torch
import argparse
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# 引入模块
from segment_anything import sam_model_registry
from prompt_generator import AutoBoxGenerator
from DataLoader import TestingDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/MoNuSeg_Processed')
    # 填入你的 Baseline SAM
    parser.add_argument('--sam_checkpoint', type=str, default='workdir/models/monuseg_train/best_epoch129_loss0.1031_sam.pth')
    # 填入你的 Box Generator
    parser.add_argument('--box_checkpoint', type=str, default='workdir/models/auto_box_head/box_head_epoch50.pth')
    parser.add_argument('--image_size', type=int, default=1024)
    # 稍微调高一点阈值，或者依靠代码里的 Top-K 限制
    parser.add_argument('--conf_thresh', type=float, default=0.05) 
    
    # SAM-Med2D 必要参数
    parser.add_argument('--encoder_adapter', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 SAM
    print("Loading SAM...")
    sam = sam_model_registry['vit_b'](args=args)
    sam.to(device)
    sam.eval()

    # 2. 加载 Generator
    print("Loading AutoBoxGenerator...")
    box_generator = AutoBoxGenerator(embed_dim=256).to(device)
    ckpt = torch.load(args.box_checkpoint, map_location=device)
    box_generator.load_state_dict(ckpt)
    box_generator.eval()

    # 3. 数据集
    dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Start Auto-Prompt Inference (Thresh: {args.conf_thresh})...")
    
    dice_scores = []
    
    # === [关键设置] SAM 推理时的 Batch Size ===
    # 每次只处理 64 个框，防止爆显存
    SAM_INFERENCE_BATCH_SIZE = 64 

    for batch in tqdm(dataloader):
        image = batch['image'].to(device).float()
        gt_mask = batch['label'].to(device)
        
        with torch.no_grad():
            # A. 提取特征
            image_embedding = sam.image_encoder(image) 
            
            # B. 自动生成 Box
            pred_heatmap, pred_wh = box_generator(image_embedding)
            
            # C. 解析 Box (Top-K 策略 + 局部极大值)
            heatmap = pred_heatmap[0, 0]
            
            # 简单 NMS: 3x3 MaxPool
            hmax = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1,padding=1)
            keep = (hmax == heatmap.unsqueeze(0).unsqueeze(0)).float()
            heatmap_peaks = heatmap * keep.squeeze()
            
            # 阈值过滤
            pos_indices = torch.nonzero(heatmap_peaks > args.conf_thresh)
            
            # [安全锁] 如果检测出太多(例如超过500个)，强制截断，取置信度最高的500个
            # 防止阈值设太低导致几千个框
            if len(pos_indices) > 500:
                scores = heatmap_peaks[pos_indices[:, 0], pos_indices[:, 1]]
                _, topk_idx = torch.topk(scores, 500)
                pos_indices = pos_indices[topk_idx]
            
            pred_boxes = []
            if len(pos_indices) > 0:
                for idx in pos_indices:
                    y_grid, x_grid = idx[0].item(), idx[1].item()
                    w_norm = pred_wh[0, 0, y_grid, x_grid].item()
                    h_norm = pred_wh[0, 1, y_grid, x_grid].item()
                    
                    stride = 16
                    cx = (x_grid + 0.5) * stride
                    cy = (y_grid + 0.5) * stride
                    w = w_norm * stride
                    h = h_norm * stride
                    pred_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                
                boxes_tensor = torch.tensor(pred_boxes, device=device).float()
            else:
                # 没检测到，跳过 SAM，直接给全黑 Mask
                dice_scores.append(0.0)
                continue

            # D. 分批运行 SAM (关键修改！！！)
            # -----------------------------------------------------------------
            final_mask_list = []
            
            # 将 boxes_tensor 切分成小块
            box_batches = torch.split(boxes_tensor, SAM_INFERENCE_BATCH_SIZE)
            
            for box_batch in box_batches:
                # Prompt Encoder
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_batch,
                    masks=None,
                )
                
                # Mask Decoder
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding, # 这里的 image_embedding 会自动广播，不占额外内存
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, 
                )
                
                # Upscale
                upscaled_masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=(args.image_size, args.image_size),
                    original_size=(args.image_size, args.image_size),
                )
                
                # 存下来
                final_mask_list.append(upscaled_masks)
                
            # 合并所有 Batch 的结果
            if len(final_mask_list) > 0:
                all_masks = torch.cat(final_mask_list, dim=0) # [Total_Boxes, 1, 1024, 1024]
                binary_masks = (all_masks > 0.0).float()
                # 融合 Mask (取最大值)
                final_mask, _ = torch.max(binary_masks, dim=0) # [1, 1024, 1024]
            else:
                final_mask = torch.zeros((1, args.image_size, args.image_size), device=device)

            # E. 计算 Dice
            gt_binary = (gt_mask > 0).float()
            intersection = (final_mask * gt_binary).sum()
            union = final_mask.sum() + gt_binary.sum()
            dice = (2. * intersection) / (union + 1e-7)
            dice_scores.append(dice.item())

    print(f"Auto-Prompt mDice: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    main()