import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
from torch.cuda import amp
import random

# === [Imports] ===
from segment_anything import sam_model_registry
# 确保这里导入的是我们在 segment_anything/modeling/sam.py 中修改后的 TextSam 类
from segment_anything.modeling.sam import TextSam 
# 确保 prompt_generator.py 就在根目录下，且包含 point_guidance_loss
from prompt_generator import point_guidance_loss 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam", help="run model name")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="train data path") 
    parser.add_argument("--attribute_info_path", type=str, default="data/MoNuSeg_SA1B/attribute_info_train.json", help="json path")
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    # 注意：这里的 checkpoint 最好是官方的 sam_vit_b_01ec64.pth 或 sam-med2d 预训练权重
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="save model interval (epochs)")
    parser.add_argument("--use_amp", action='store_true', help="use amp")
    
    # SAM-Med2D 特定参数
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter in image encoder")
    
    args = parser.parse_args()
    return args

# =========================================================================================
# 🛠️ [Helper] Generate Gaussian Heatmap from Mask
# =========================================================================================
def generate_gaussian_target(masks, shape=(64, 64), sigma=1.0, device='cuda'):
    """
    为前景（细胞核）生成高斯热力图
    Input: masks [B, 1, H, W] (原始尺寸或 256x256)
    Output: heatmap [B, 1, 64, 64]
    """
    B = len(masks)
    H_feat, W_feat = shape
    targets = torch.zeros(B, 1, H_feat, W_feat, device=device)
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H_feat, device=device), 
        torch.arange(W_feat, device=device), 
        indexing='ij'
    )
    
    for i in range(B):
        mask = masks[i, 0] # [H, W]
        if mask.sum() == 0: continue 
        
        # 1. 计算质心
        # 注意：这里假设每个 Instance 已经被合并成一个 Binary Mask
        # 如果是 Instance Segmentation，最好先用 measure.label 分离
        # 这里为了简便，我们直接对 Mask > 0 的区域计算
        # (更严谨的做法是：如果 DataLoader 提供了 instance mask，应该对每个 instance 生成高斯，然后取 max)
        
        # 简单处理：将 Mask 缩放到 64x64，然后对每个像素点做高斯模糊（或者是 Distance Transform）
        # 但最标准的 CenterNet 做法是对每个 Instance 中心画高斯。
        # 既然 MoNuSeg 是密集预测，我们采用 "Downsampled Mask" 作为基础监督可能会更稳健。
        # -----------------------------------------------------------
        # 方案 A: 仅在质心画高斯 (适合稀疏目标，如检测)
        # 方案 B: 使用 Distance Transform 或 缩放后的 Mask (适合密集分割)
        # 这里我们采用方案 A 的变体：对所有前景点计算高斯可能会太慢。
        # 建议：直接将 Mask 缩放到 64x64 作为 "Dense Heatmap Target"
        # -----------------------------------------------------------
        
        # 重新实现：直接缩放 Mask，然后做一点平滑 (可选)
        # 这种方式对于形状不规则的细胞核更鲁棒
        mask_float = mask.float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        resized_mask = F.interpolate(mask_float, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        targets[i] = resized_mask.squeeze(0)
        
    return targets

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler=None):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        # 1. Prepare Data
        images = batched_input['image'].to(args.device)     # [B, 3, 256, 256]
        labels = batched_input['label'].to(args.device)     # [B, 1, 256, 256]
        
        if labels.ndim == 3:
            labels = labels.unsqueeze(1) # Ensure [B, 1, H, W]

        # 2. Format Input for SAM
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size)
            })
            
        # 3. (删除) 原来的 GT Heatmap 生成代码已移除，改为在 Loss 计算中动态生成

        optimizer.zero_grad()

        # 4. Loss Computation Function
        def compute_loss(outputs):
            loss_batch_sum = 0
            loss_m_val_sum = 0
            loss_h_val_sum = 0
            
            for i, out in enumerate(outputs):
                # --- A. Segmentation Mask Loss ---
                # 处理 Mask 维度 [1, C, H, W] 或 [C, H, W]
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, 0, :, :] # [H, W]
                else:
                    pred_mask = out['masks'][0, :, :] # [H, W]
                
                gt_mask = labels[i].float() # [1, H, W]

                # 确保 Mask 尺寸对齐 (GT -> Pred)
                if pred_mask.shape != gt_mask.shape[-2:]:
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(0), 
                        size=pred_mask.shape, 
                        mode='nearest'
                    ).squeeze(0)

                # IoU Prediction
                pred_iou = out['iou_predictions'][0]
                
                # Mask Loss
                loss_m = criterion(
                    pred_mask.unsqueeze(0).unsqueeze(0), 
                    gt_mask.unsqueeze(0), 
                    pred_iou.unsqueeze(0)
                )
                
                # --- B. Heatmap Loss (动态尺寸适配) ---
                pred_heatmap = out['heatmap_logits'] # shape 可能是 [2, 16, 16] 或 [2, 64, 64]
                
                # 获取模型实际输出的特征图尺寸
                current_feat_size = pred_heatmap.shape[-2:] # (H_feat, W_feat)
                
                # 实时生成匹配尺寸的 GT
                # labels[i] 是 [1, 256, 256] -> 插值到 [1, 16, 16]
                with torch.no_grad():
                    # 前景 GT
                    gt_nuclei = F.interpolate(
                        labels[i].float().unsqueeze(0), # [1, 1, 256, 256]
                        size=current_feat_size, 
                        mode='nearest' # 或 'bilinear'
                    ).squeeze(0) # [1, 16, 16]
                    
                    # 背景 GT
                    gt_background = 1.0 - gt_nuclei

                # 计算 Loss
                # Channel 0 vs Nuclei
                loss_h_pos = point_guidance_loss(pred_heatmap[0:1].unsqueeze(0), gt_nuclei.unsqueeze(0))
                # Channel 1 vs Background
                loss_h_neg = point_guidance_loss(pred_heatmap[1:2].unsqueeze(0), gt_background.unsqueeze(0))
                
                loss_h = loss_h_pos + loss_h_neg
                
                # Total Loss (Mask Loss 为主，Heatmap 为辅)
                loss_i = loss_m + 0.5 * loss_h
                
                loss_batch_sum += loss_i
                loss_m_val_sum += loss_m.item()
                loss_h_val_sum += loss_h.item()
            
            return loss_batch_sum / len(images), loss_m_val_sum, loss_h_val_sum

        # 5. Forward & Backward
        if args.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(sam_input, multimask_output=True)
                loss_batch, loss_m_val, loss_h_val = compute_loss(outputs)

            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sam_input, multimask_output=True)
            loss_batch, loss_m_val, loss_h_val = compute_loss(outputs)
            
            loss_batch.backward()
            optimizer.step()

        # Log
        losses.append(loss_batch.item())
        mask_losses.append(loss_m_val / len(images))
        heatmap_losses.append(loss_h_val / len(images))
        
        pbar.set_postfix(
            loss=f"{loss_batch.item():.4f}", 
            msk=f"{(loss_m_val / len(images)):.4f}", 
            ht=f"{(loss_h_val / len(images)):.4f}"
        )

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        # 1. Prepare Data
        images = batched_input['image'].to(args.device)     # [B, 3, 256, 256]
        labels = batched_input['label'].to(args.device)     # [B, 1, 256, 256] or [B, 256, 256]
        
        if labels.ndim == 3:
            labels = labels.unsqueeze(1) # Ensure [B, 1, H, W]

        # 2. Format Input for SAM
        # TextSam 会自动在内部生成 [Nuclei, Background] 的文本特征
        # 所以这里的 text_prompts 实际上不会被 TextSam 用到，但为了保持接口兼容，我们传个空的或者默认的
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size) # 告诉 SAM 最终还原回什么尺寸
            })
            
        # 3. Generate GT Heatmaps (用于监督 TextGuidedPointGenerator)
        with torch.no_grad():
            # A. 前景 (Nuclei) GT: 64x64 的高斯图或缩放图
            # 建议：对于密集分割，直接用 Downsample 的 Mask 效果更好
            gt_nuclei_map = F.interpolate(labels.float(), size=(64, 64), mode='nearest') # [B, 1, 64, 64]
            
            # B. 背景 (Background) GT: 1 - Nuclei
            gt_background_map = 1.0 - gt_nuclei_map # [B, 1, 64, 64]

        optimizer.zero_grad()

        # 4. Loss Computation Function
        def compute_loss(outputs):
            loss_batch_sum = 0
            loss_m_val_sum = 0
            loss_h_val_sum = 0
            
            for i, out in enumerate(outputs):
                # --- A. Segmentation Mask Loss ---
                # out['masks'] shape: [1, C, H, W] 
                # 我们取第 0 个通道 (对应第 0 个 Mask)
                # 这种写法兼容 [1, C, H, W] 和 [C, H, W]
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, 0, :, :] # [H, W]
                else:
                    pred_mask = out['masks'][0, :, :] # [H, W]
                
                gt_mask = labels[i].float() # [1, H, W]

                # 确保 GT 和 Pred 尺寸对齐
                # pred_mask 是 [H, W]，gt_mask 是 [1, H, W]
                if pred_mask.shape != gt_mask.shape[-2:]:
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(0), # [1, 1, H, W]
                        size=pred_mask.shape, 
                        mode='nearest'
                    ).squeeze(0) # [1, H, W]

                # --- B. IoU Prediction ---
                # ❌ 错误原因：out['iou_predictions'] 是 1D Tensor [C]，检查 shape[1] 会报错
                # ✅ 修复：直接取第 0 个值即可
                pred_iou = out['iou_predictions'][0]
                
                # --- C. Calc Loss ---
                # criterion 需要 [B, 1, H, W] 格式
                loss_m = criterion(
                    pred_mask.unsqueeze(0).unsqueeze(0), # [1, 1, H, W]
                    gt_mask.unsqueeze(0),                # [1, 1, H, W] (gt_mask已经是[1,H,W])
                    pred_iou.unsqueeze(0)                # [1]
                )
                
                # --- D. Heatmap Loss (双向监督) ---
                pred_heatmap = out['heatmap_logits'] # [2, 64, 64]
                
                # 前景 Loss (Channel 0 vs GT Nuclei)
                loss_h_pos = point_guidance_loss(pred_heatmap[0:1].unsqueeze(0), gt_nuclei_map[i].unsqueeze(0))
                
                # 背景 Loss (Channel 1 vs GT Background)
                loss_h_neg = point_guidance_loss(pred_heatmap[1:2].unsqueeze(0), gt_background_map[i].unsqueeze(0))
                
                loss_h = loss_h_pos + loss_h_neg
                
                # Total Loss
                loss_i = loss_m + 0.5 * loss_h
                
                loss_batch_sum += loss_i
                loss_m_val_sum += loss_m.item()
                loss_h_val_sum += loss_h.item()
            
            return loss_batch_sum / len(images), loss_m_val_sum, loss_h_val_sum
        # 5. Forward & Backward
        if args.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(sam_input, multimask_output=True)
                loss_batch, loss_m_val, loss_h_val = compute_loss(outputs)

            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sam_input, multimask_output=True)
            loss_batch, loss_m_val, loss_h_val = compute_loss(outputs)
            
            loss_batch.backward()
            optimizer.step()

        # Log
        losses.append(loss_batch.item())
        mask_losses.append(loss_m_val / len(images))
        heatmap_losses.append(loss_h_val / len(images))
        
        pbar.set_postfix(
            loss=f"{loss_batch.item():.4f}", 
            msk=f"{(loss_m_val / len(images)):.4f}", 
            ht=f"{(loss_h_val / len(images)):.4f}"
        )

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)

def main(args):
    # Init Loggers
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    
    logger.info(f"Args: {args}")

    # 1. Dataset
    train_dataset = TrainingDataset(
        args.data_path, 
        image_size=args.image_size, 
        mode='train', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=True, 
        attribute_info_path=args.attribute_info_path 
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=stack_dict_batched,
        pin_memory=True
    )
    logger.info(f"Train data size: {len(train_dataset)}")

    # 2. Build TextSam Model
    logger.info("Building TextSam model...")
    
    # === A. 初始化基础 SAM (SAM-Med2D) ===
    # 这里使用 sam_model_registry 加载基础结构
    # args 包含了 encoder_adapter，如果是 sam-med2d 的代码库，它会自动处理 Adapter
    vanilla_sam = sam_model_registry[args.model_type](args)

    # === B. 加载预训练权重 ===
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        logger.info(f"Loading checkpoint from {args.sam_checkpoint}...")
        try:
            checkpoint = torch.load(args.sam_checkpoint, map_location=args.device,weights_only=False)
            # 兼容性处理：有些权重在 'model' key 下，有些直接是 dict
            state_dict = checkpoint.get("model", checkpoint)
            
            # 使用 strict=False，因为 args.encoder_adapter 可能会改变模型结构
            keys = vanilla_sam.load_state_dict(state_dict, strict=False)
            logger.info(f"Checkpoint loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Continuing with random initialization for missing parts...")
    else:
        logger.warning(f"Checkpoint path {args.sam_checkpoint} does not exist. Training from scratch (not recommended).")

    # === C. 构建 TextSam 并迁移权重 ===
    # 我们将 vanilla_sam 的 encoder/decoder 传给 TextSam
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        pixel_mean=vanilla_sam.pixel_mean,
        pixel_std=vanilla_sam.pixel_std,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    
    # 释放 vanilla_sam 节省内存
    del vanilla_sam 

    # 3. Optimizer
    # 这里的学习率设置很重要：
    # Mask Decoder 需要微调
    # Prompt Generator 是全新层，建议 LR 大一点 (x10)
    # Image Encoder 如果使用了 Adapter，也需要放入优化器
    
    params_to_optimize = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 10}
    ]
    
    # 如果启用了 Adapter，Image Encoder 也是可训练的
    if args.encoder_adapter:
        # 假设 Image Encoder 里有 adapter 相关的参数
        # 简单的做法是把 image_encoder 加入，但只训练 requires_grad=True 的部分
        params_to_optimize.append({'params': filter(lambda p: p.requires_grad, model.image_encoder.parameters()), 'lr': args.lr})

    optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)

    criterion = FocalDiceloss_IoULoss()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 4. Training Loop
    best_loss = 1e10
    start_epoch = 0
    
    # Resume Logic
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']

    logger.info("Start Training...")
    for epoch in range(start_epoch, args.epochs):
        avg_loss, avg_msk, avg_ht = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} (Mask: {avg_msk:.4f}, Heatmap: {avg_ht:.4f})")
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss
            }, save_path)
            logger.info(f"Saved Best Model (Loss: {best_loss:.4f})")
            
        # Regular Save
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1
            }, save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)