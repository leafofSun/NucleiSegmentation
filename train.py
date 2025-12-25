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
from segment_anything.modeling.sam import TextSam 
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
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="save model interval (epochs)")
    parser.add_argument("--use_amp", action='store_true', help="use amp")
    
    # === [新增修复] 添加 encoder_adapter 参数 ===
    # SAM-Med2D 需要知道是否使用 adapter。默认设为 True 以启用微调层。
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter in image encoder")
    
    args = parser.parse_args()
    return args

# =========================================================================================
# 🛠️ [Helper] Generate Gaussian Heatmap from Mask
# =========================================================================================
def generate_gaussian_target(masks, shape=(64, 64), sigma=1.0, device='cuda'):
    """
    Input: masks [B, 1, H, W]
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
        
        # Calculate centroid
        y_indices, x_indices = torch.where(mask > 0)
        center_y = y_indices.float().mean()
        center_x = x_indices.float().mean()
        
        # Map coordinates: Original -> 64x64
        scale_h = H_feat / mask.shape[0]
        scale_w = W_feat / mask.shape[1]
        
        feat_cy = center_y * scale_h
        feat_cx = center_x * scale_w
        
        # Generate Gaussian
        gaussian = torch.exp(-((x_grid - feat_cx)**2 + (y_grid - feat_cy)**2) / (2 * sigma**2))
        targets[i, 0] = gaussian
        
    return targets

# 将此函数完整替换 train.py 中的同名函数
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler=None):
    model.train()
    pbar = tqdm(train_loader)
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        # 1. Prepare Data
        images = batched_input['image'].to(args.device)     
        labels = batched_input['label'].to(args.device)     
        
        # Handle text prompts
        raw_prompts = batched_input.get('attribute_prompts', [])
        text_prompts_list = []
        for p_item in raw_prompts:
            if isinstance(p_item, (list, tuple)):
                text_prompts_list.append([str(x) for x in p_item])
            else:
                text_prompts_list.append(["cell"]) 

        # 2. Format Input
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'text_prompts': text_prompts_list[i], 
                'original_size': (args.image_size, args.image_size)
            })
            
        # 3. GT Heatmaps
        with torch.no_grad():
            # Generate heatmaps (usually 64x64)
            gt_heatmaps = generate_gaussian_target(labels, shape=(64, 64), device=args.device)

        optimizer.zero_grad()

        # 4. Forward Pass Definition
        def compute_loss(outputs):
            loss_batch_sum = 0
            loss_m_val_sum = 0
            loss_h_val_sum = 0
            
            for i, out in enumerate(outputs):
                # --- A. Mask Processing ---
                # 获取预测 Mask (通常是 256x256)
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, 0, :, :] 
                else:
                    pred_mask = out['masks'][0, :, :]    
                
                # 获取真实 Label (可能是 1024x1024)
                gt_mask = labels[i, 0, :, :]
                
                # === [FIX] 强制对齐尺寸 ===
                # 如果尺寸不一致，将 GT 缩放到 Pred 的大小
                if pred_mask.shape != gt_mask.shape:
                    # 插值需要 4D 输入: [1, 1, H, W]
                    target_size = pred_mask.shape[-2:] # (256, 256)
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0).float(), 
                        size=target_size, 
                        mode='nearest' # Mask 使用最近邻插值保持 0/1
                    ).squeeze(0).squeeze(0) # 变回 [H, W]

                # --- B. IoU Prediction ---
                pred_iou = out['iou_predictions'][0]
                
                # --- C. Calculate Mask Loss ---
                loss_m = criterion(
                    pred_mask.unsqueeze(0).unsqueeze(0), 
                    gt_mask.unsqueeze(0).unsqueeze(0), 
                    pred_iou
                )
                
                # --- D. Calculate Heatmap Loss ---
                pred_heatmap = out['heatmap_logits']
                target_heatmap = gt_heatmaps[i].unsqueeze(0)
                loss_h = point_guidance_loss(pred_heatmap, target_heatmap)
                
                # Total Loss
                loss_i = loss_m + 1.0 * loss_h
                
                loss_batch_sum += loss_i
                loss_m_val_sum += loss_m.item()
                loss_h_val_sum += loss_h.item()
            
            return loss_batch_sum / len(images), loss_m_val_sum, loss_h_val_sum

        # 5. Execution (AMP or FP32)
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
            loss=loss_batch.item(), 
            msk=mask_losses[-1], 
            ht=heatmap_losses[-1]
        )

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)
    model.train()
    pbar = tqdm(train_loader)
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        # 1. Prepare Data
        images = batched_input['image'].to(args.device)     
        labels = batched_input['label'].to(args.device)     
        
        # Handle text prompts
        raw_prompts = batched_input.get('attribute_prompts', [])
        text_prompts_list = []
        for p_item in raw_prompts:
            if isinstance(p_item, (list, tuple)):
                text_prompts_list.append([str(x) for x in p_item])
            else:
                text_prompts_list.append(["cell"]) 

        # 2. Format Input
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'text_prompts': text_prompts_list[i], 
                'original_size': (args.image_size, args.image_size)
            })
            
        # 3. GT Heatmaps
        with torch.no_grad():
            gt_heatmaps = generate_gaussian_target(labels, shape=(64, 64), device=args.device)

        optimizer.zero_grad()

        # 4. Forward Pass
        # 定义内部处理函数以复用代码
        def compute_loss(outputs):
            loss_batch_sum = 0
            loss_m_val_sum = 0
            loss_h_val_sum = 0
            
            for i, out in enumerate(outputs):
                # --- Mask Processing ---
                pred_mask = out['masks'][:, 0, :, :]
                gt_mask = labels[i, 0, :, :]
                
                # [FIX] Extract IoU prediction
                pred_iou = out['iou_predictions'][ 0]
                
                # [FIX] Pass pred_iou to criterion
                loss_m = criterion(
                    pred_mask.unsqueeze(0).unsqueeze(0), 
                    gt_mask.unsqueeze(0).unsqueeze(0), 
                    pred_iou
                )
                
                # --- Heatmap Processing ---
                pred_heatmap = out['heatmap_logits']
                target_heatmap = gt_heatmaps[i].unsqueeze(0)
                loss_h = point_guidance_loss(pred_heatmap, target_heatmap)
                
                # Total Loss
                loss_i = loss_m + 1.0 * loss_h
                
                loss_batch_sum += loss_i
                loss_m_val_sum += loss_m.item()
                loss_h_val_sum += loss_h.item()
            
            return loss_batch_sum / len(images), loss_m_val_sum, loss_h_val_sum

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
            loss=loss_batch.item(), 
            msk=mask_losses[-1], 
            ht=heatmap_losses[-1]
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
        collate_fn=stack_dict_batched
    )
    logger.info(f"Train data size: {len(train_dataset)}")

    # 2. Build TextSam Model
    logger.info("Building TextSam model...")
    
    # === 模型初始化 ===
    # 步骤 A: 初始化原生 SAM 模型 (传入 args，此时 args 包含 encoder_adapter)
    vanilla_sam = sam_model_registry[args.model_type](args)

    # 步骤 B: 手动加载权重
    if args.sam_checkpoint:
        print(f"Loading checkpoint from {args.sam_checkpoint}...")
        try:
            checkpoint = torch.load(args.sam_checkpoint, map_location=args.device,weights_only=False)
            # 兼容性处理
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # strict=False 防止版本差异报错
            vanilla_sam.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Continuing with random initialization for missing parts...")

    # 步骤 C: 将组件转移到 TextSam
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
    
    del vanilla_sam # 释放内存

    # 3. Optimizer
    optimizer = optim.AdamW([
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 10} 
    ], weight_decay=1e-4)

    criterion = FocalDiceloss_IoULoss()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 4. Training Loop
    best_loss = 1e10
    start_epoch = 0
    
    # Resume Logic
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        avg_loss, avg_msk, avg_ht = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        duration = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} (Mask: {avg_msk:.4f}, Heatmap: {avg_ht:.4f}) | Time: {duration:.1f}s")
        
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
            torch.save({'model': model.state_dict(), 'epoch': epoch+1}, save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)