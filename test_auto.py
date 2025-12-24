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
    parser.add_argument('--data_path', type=str, default='data/cpm17_SA1B')
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--box_checkpoint', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--conf_thresh', type=float, default=0.3) 
    parser.add_argument('--encoder_adapter', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print("Loading SAM...")
    sam = sam_model_registry['vit_b'](args=args)
    sam.to(device)
    sam.eval()

    print("Loading AutoBoxGenerator...")
    box_generator = AutoBoxGenerator(embed_dim=256).to(device)
    ckpt = torch.load(args.box_checkpoint, map_location=device)
    box_generator.load_state_dict(ckpt)
    box_generator.eval()

    # 2. 数据集
    dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Start Inference (Thresh: {args.conf_thresh})...")
    
    dice_scores = []
    SAM_INFERENCE_BATCH_SIZE = 64 
    save_count = 0

    for batch in tqdm(dataloader):
        try:
            image = batch['image'].to(device).float()
            gt_mask = batch['label'].to(device) 
            
            with torch.no_grad():
                # A. 提取特征
                image_embedding = sam.image_encoder(image) 
                pred_heatmap, pred_wh = box_generator(image_embedding)
                
                heatmap = pred_heatmap[0, 0]
                
                # B. NMS 找峰值
                hmax = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
                keep = (hmax == heatmap.unsqueeze(0).unsqueeze(0)).float()
                heatmap_peaks = heatmap * keep.squeeze()
                
                pos_indices = torch.nonzero(heatmap_peaks > args.conf_thresh)
                
                # 限制最大检测数
                if len(pos_indices) > 500:
                    scores = heatmap_peaks[pos_indices[:, 0], pos_indices[:, 1]]
                    _, topk_idx = torch.topk(scores, 500)
                    pos_indices = pos_indices[topk_idx]
                
                pred_boxes = []
                if len(pos_indices) > 0:
                    for idx in pos_indices:
                        # 标准坐标读取
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
                    dice_scores.append(0.0)
                    continue

                # C. SAM 分割
                final_mask_list = []
                box_batches = torch.split(boxes_tensor, SAM_INFERENCE_BATCH_SIZE)
                
                for box_batch in box_batches:
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None, boxes=box_batch, masks=None,
                    )
                    low_res_masks, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False, 
                    )
                    upscaled_masks = sam.postprocess_masks(
                        low_res_masks,
                        input_size=(args.image_size, args.image_size),
                        original_size=(args.image_size, args.image_size),
                    )
                    final_mask_list.append(upscaled_masks)
                    
                if len(final_mask_list) > 0:
                    all_masks = torch.cat(final_mask_list, dim=0) 
                    binary_masks = (all_masks > 0.0).float()
                    final_mask, _ = torch.max(binary_masks, dim=0) 
                else:
                    final_mask = torch.zeros((1, args.image_size, args.image_size), device=device)

                # D. Dice 计算
                gt_binary = (gt_mask > 0).float().squeeze()
                pred_binary = final_mask.squeeze()
                
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum()
                
                if union > 0:
                    dice = (2. * intersection) / (union + 1e-7)
                    dice_val = dice.item()
                else:
                    dice_val = 1.0 if gt_binary.sum() == 0 and pred_binary.sum() == 0 else 0.0
                
                dice_scores.append(dice_val)
                
                # E. 可视化
                if save_count < 5:
                    img_vis = image[0].permute(1, 2, 0).cpu().numpy()
                    img_vis = (img_vis * np.array([57.375, 57.12, 58.395]) + np.array([103.53, 116.28, 123.675]))
                    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8).copy()
                    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

                    # 画 GT (绿色)
                    if 'boxes' in batch:
                         for gt_box in batch['boxes'][0]:
                            if gt_box.sum() == 0: continue
                            x1, y1, x2, y2 = map(int, gt_box.tolist())
                            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 画 Pred (红色)
                    for box in pred_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    cv2.putText(img_vis, f"Dice: {dice_val:.4f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    os.makedirs("workdir/vis_auto_box", exist_ok=True)
                    
                    # 修复的那一行：
                    cv2.imwrite(f"workdir/vis_auto_box/final_vis_{save_count}.jpg", img_vis)
                    save_count += 1

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    if len(dice_scores) > 0:
        print(f"\n✅ Auto-Prompt mDice (Real Mask): {np.mean(dice_scores):.4f}")
    else:
        print("\n❌ No predictions.")

if __name__ == "__main__":
    main()