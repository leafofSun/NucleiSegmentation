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
    parser.add_argument('--box_checkpoint', type=str, default='workdir/models/auto_box_head_dense/box_head_epoch50.pth')
    parser.add_argument('--image_size', type=int, default=1024)
    
    # 建议阈值：0.1 ~ 0.3
    parser.add_argument('--conf_thresh', type=float, default=0.1) 
    
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
    SAM_INFERENCE_BATCH_SIZE = 64 

    # 计数器，用于控制保存图片的数量
    save_count = 0

    for batch in tqdm(dataloader):
        image = batch['image'].to(device).float()
        gt_mask = batch['label'].to(device)
        
        with torch.no_grad():
            # A. 提取特征
            image_embedding = sam.image_encoder(image) 
            
            # B. 自动生成 Box
            pred_heatmap, pred_wh = box_generator(image_embedding)
            
            # C. 解析 Box
            heatmap = pred_heatmap[0, 0]
            
            # 3x3 MaxPool NMS (Stride=1)
            hmax = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
            keep = (hmax == heatmap.unsqueeze(0).unsqueeze(0)).float()
            heatmap_peaks = heatmap * keep.squeeze()
            
            # 阈值过滤
            pos_indices = torch.nonzero(heatmap_peaks > args.conf_thresh)
            
            # 截断过多框
            if len(pos_indices) > 500:
                scores = heatmap_peaks[pos_indices[:, 0], pos_indices[:, 1]]
                _, topk_idx = torch.topk(scores, 500)
                pos_indices = pos_indices[topk_idx]
            
            pred_boxes = []
            if len(pos_indices) > 0:
                for idx in pos_indices:
                    y_grid, x_grid = idx[0].item(), idx[1].item()
                    
                    # === 💡 实验性策略: 强制固定大小 ===
                    # 如果这一步分数涨了，说明中心点找对了，是WH回归没练好
                    # w_norm = pred_wh[0, 0, y_grid, x_grid].item()
                    # h_norm = pred_wh[0, 1, y_grid, x_grid].item()
                    
                    # 强制设为 32 像素 (MoNuSeg 细胞平均大小)
                    w_fixed = 32.0 / 16.0 # 16是stride
                    h_fixed = 32.0 / 16.0 
                    
                    stride = 16
                    cx = (x_grid + 0.5) * stride
                    cy = (y_grid + 0.5) * stride
                    
                    # 使用固定宽高
                    w = w_fixed * stride
                    h = h_fixed * stride
                    
                    pred_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                
                boxes_tensor = torch.tensor(pred_boxes, device=device).float()
            else:
                dice_scores.append(0.0)
                continue

            # D. 分批运行 SAM
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

            # E. 计算 Dice (先算 Dice，后面文件名要用)
            gt_binary = (gt_mask > 0).float()
            intersection = (final_mask * gt_binary).sum()
            union = final_mask.sum() + gt_binary.sum()
            dice = (2. * intersection) / (union + 1e-7)
            dice_scores.append(dice.item())
            
            # F. 可视化保存 (只保存前 10 张)
            if save_count < 10:
                img_vis = image[0].permute(1, 2, 0).cpu().numpy()
                # 反归一化
                img_vis = (img_vis * np.array([57.375, 57.12, 58.395]) + np.array([103.53, 116.28, 123.675]))
                img_vis = img_vis.astype(np.uint8).copy()
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

                # 画预测红框
                if len(pred_boxes) > 0:
                    for box in pred_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                os.makedirs("workdir/vis_auto_box", exist_ok=True)
                save_name = f"workdir/vis_auto_box/vis_{save_count}_dice{dice.item():.2f}.jpg"
                cv2.imwrite(save_name, img_vis)
                save_count += 1

    print(f"Auto-Prompt mDice (Fixed Size 32x32): {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    main()