import argparse
import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Native AMP support
try:
    from torch.amp import autocast, GradScaler 
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# === 项目模块导入 ===
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import TrainingDataset, stack_dict_batched

# 🔥 核心工具导入 (Loss 和 Logger)
from utils import FocalDiceloss_IoULoss, point_guidance_loss, get_logger

# 🔥 高性能指标导入 (AJI, PQ, DQ, SQ)
from metrics import SegMetrics

# ==================================================================================================
# 1. 参数配置 (Configuration)
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM: Explicit-Implicit Dual-Stream Training")
    
    # --- 基础环境 ---
    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam_monuseg_final", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    
    # --- 数据路径 (完全解耦) ---
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/MoNuSeg_SA1B/medical_knowledge.json", 
                        help="Path to the generated Explicit Knowledge Base JSON")
    
    # --- 图像参数 ---
    parser.add_argument("--image_size", type=int, default=1024, help="Input image resolution")
    parser.add_argument("--crop_size", type=int, default=1024, help="Random crop size during training")
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    # --- 模型配置 ---
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Path to original/medsam checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model version for Text Encoder")
    parser.add_argument("--num_organs", type=int, default=10, help="Number of organ categories for DualPromptLearner")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Use Adapters in Image Encoder")

    # --- 训练超参 ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision")
    
    # --- Loss 权重 ---
    parser.add_argument("--mask_weight", type=float, default=2.0, help="Weight for Segmentation Loss (Focal+Dice)")
    parser.add_argument("--heatmap_weight", type=float, default=1.0, help="Weight for Auto-Prompt Heatmap Loss")

    # --- 验证指标 (PromptNu 标准) ---
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ', 'mDQ', 'mSQ'], 
                        help="Metrics to evaluate: dice, iou, mAJI, mPQ, mDQ, mSQ")

    return parser.parse_args()

# ==================================================================================================
# 2. 辅助函数 (Utils)
# ==================================================================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def resize_pos_embed(state_dict, model_state_dict):
    """调整 SAM 位置编码尺寸"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                if 'pos_embed' in k:
                    v = v.permute(0, 3, 1, 2)
                    v = F.interpolate(v, size=model_state_dict[k].shape[1:3], mode='bicubic', align_corners=False)
                    v = v.permute(0, 2, 3, 1)
                elif 'rel_pos' in k:
                    v = v.unsqueeze(0).permute(0, 2, 1)
                    target_len = model_state_dict[k].shape[0]
                    v = F.interpolate(v, size=target_len, mode='linear', align_corners=False)
                    v = v.permute(0, 2, 1).squeeze(0)
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# ==================================================================================================
# 3. 训练逻辑 (Train Loop)
# ==================================================================================================
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']
        
        optimizer.zero_grad()

        # === 🔥 [核心] 构建 MP-SAM 数据流 (Knowledge Injection) 🔥 ===
        model_input = []
        organ_ids = batched_input.get('organ_id', None) # 隐式流
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images)) # 显式流
        base_texts = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))

        for i in range(len(images)):
            model_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size),
                # 注入 MP-SAM 字段
                'organ_id': organ_ids[i] if organ_ids is not None else 9, # 9=Generic fallback
                'attribute_text': attr_texts[i],
                'text_prompt': base_texts[i]
            })

        # === Forward Pass (AMP) ===
        with autocast('cuda', enabled=args.use_amp):
            # TextSam.forward 内部会自动分发 organ_id 和 attribute_text 到对应模块
            outputs = model(model_input, multimask_output=True)
            
            loss_batch = 0
            loss_m_accum = 0
            loss_h_accum = 0
            
            for i, out in enumerate(outputs):
                # --- A. Mask Loss Calculation ---
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2: iou_preds = iou_preds.squeeze(0)
                
                # 选取 IoU 预测最高的 Mask 计算 Loss
                best_idx = torch.argmax(iou_preds).item()
                
                # 处理 Mask 维度 [1, 3, H, W] or [3, H, W]
                pred_mask = out['masks'][best_idx, :, :] if out['masks'].ndim==3 else out['masks'][0, best_idx]
                pred_iou = iou_preds[best_idx]
                
                gt_mask = labels[i].squeeze(0).float() # [H, W]
                
                # 尺寸对齐
                if pred_mask.shape != gt_mask.shape:
                     gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                # 🔥 使用 utils.py 中的 Loss 计算 (自动处理 ignore_index=255)
                loss_m, _ = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                # --- B. Heatmap Loss Calculation (Auto-Point Supervision) ---
                pred_heatmap = out['heatmap_logits'] # [1, H_feat, W_feat]
                
                with torch.no_grad():
                    # 将 GT 缩放到 Feature Map 尺寸生成监督信号
                    target_mask = labels[i].float().unsqueeze(0)
                    gt_nuclei = F.interpolate(target_mask, size=pred_heatmap.shape[-2:], mode='nearest').squeeze(0)
                    gt_nuclei[gt_nuclei==255] = 0 # 忽略区域设为背景
                
                # 🔥 使用 utils.py 中的 point_guidance_loss
                loss_h = point_guidance_loss(pred_heatmap, gt_nuclei.unsqueeze(0))
                
                # --- C. Weighted Sum ---
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h
                
                loss_batch += loss_i
                loss_m_accum += loss_m.item()
                loss_h_accum += loss_h.item()
            
            final_loss = loss_batch / len(images)

        # === Backward ===
        if scaler:
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            final_loss.backward()
            optimizer.step()

        losses.append(final_loss.item())
        mask_losses.append(loss_m_accum / len(images))
        heatmap_losses.append(loss_h_accum / len(images))
        
        # 显示当前 prompt
        prompt_preview = attr_texts[0][:15] + ".." if len(attr_texts[0]) > 15 else attr_texts[0]
        pbar.set_postfix(Loss=f"{final_loss.item():.3f}", Prompt=prompt_preview)

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)

# ==================================================================================================
# 4. 验证逻辑 (Val Loop)
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    
    # 动态初始化所有指标列表
    val_results = {k: [] for k in args.metrics} 
    
    pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val")
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label'].cpu().numpy()
        
        # 构建验证输入
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        
        for i in range(len(images)):
            model_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size),
                'organ_id': organ_ids[i] if organ_ids is not None else 9,
                'attribute_text': attr_texts[i],
                'text_prompt': "Cell nuclei"
            })
            
        outputs = model(model_input, multimask_output=True)
        
        for i, out in enumerate(outputs):
            iou_preds = out['iou_predictions']
            best_idx = torch.argmax(iou_preds).item()
            
            # 获取预测 Mask (0/1)
            # 注意：Metrics 中会自动处理二值->实例 (label)，所以这里只需给概率图或二值图
            pred_logits = out['masks'][0, best_idx]
            pred_mask = (torch.sigmoid(pred_logits).cpu().numpy() > 0.5).astype(np.uint8)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            
            # 处理 Ignore 区域 (Metrics 假设 0 为背景)
            gt_valid = gt.copy()
            gt_valid[gt == 255] = 0
            
            # 🔥 [关键] 计算 SegMetrics (含 AJI, PQ, DQ, SQ)
            # metrics.py 会自动判断是否需要转实例
            res = SegMetrics(pred_mask, gt_valid, args.metrics)
            
            for k in args.metrics:
                if k in res:
                    val_results[k].append(res[k])
        
        # 实时显示 AJI，因为它是最重要的实例指标
        if 'mAJI' in args.metrics and len(val_results['mAJI']) > 0:
            pbar.set_postfix(AJI=f"{val_results['mAJI'][-1]:.3f}")
                
    # 计算平均值
    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    return avg_results

# ==================================================================================================
# 5. 主程序 (Main)
# ==================================================================================================
def main(args):
    setup_seed(args.seed)
    
    # --- 日志设置 ---
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{timestamp}.log"))
    logger.info(f"🚀 [Start] MP-SAM Training | Device: {args.device}")
    logger.info(f"📁 Data: {args.data_path}")
    logger.info(f"🧠 Knowledge Base: {args.knowledge_path}")

    # --- 数据加载 ---
    train_dataset = TrainingDataset(
        os.path.join(args.data_path, "train"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, crop_size=args.crop_size, mode='train'
    )
    val_dataset = TrainingDataset(
        os.path.join(args.data_path, "test"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, crop_size=args.crop_size, mode='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=stack_dict_batched, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=stack_dict_batched, pin_memory=True)
    
    logger.info(f"📊 Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # --- 模型构建 ---
    logger.info(f"🏗️ Building TextSam (Organs={args.num_organs})...")
    args.checkpoint = args.sam_checkpoint
    # 1. 加载 Vanilla SAM
    vanilla_sam = sam_model_registry[args.model_type](args)
    if os.path.exists(args.sam_checkpoint):
        logger.info(f"📥 Loading checkpoint: {args.sam_checkpoint}")
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location='cpu',weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            state_dict = resize_pos_embed(state_dict, vanilla_sam.state_dict())
            vanilla_sam.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint loading failed: {e}. Using random init.")
    
    # 2. 构建 MP-SAM
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name=args.clip_model,
        num_organs=args.num_organs
    ).to(args.device)
    
    del vanilla_sam

    # --- Adapter 初始化 ---
    if args.encoder_adapter:
        logger.info("🧹 Resetting Adapter weights to Zero...")
        for n, p in model.image_encoder.named_parameters():
            if "Adapter" in n and "weight" in n:
                torch.nn.init.zeros_(p)

    # --- 优化器配置 (自动模块发现) ---
    params = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 5}
    ]
    
    # 自动发现 DualPromptLearner
    if hasattr(model, 'prompt_learner'):
        logger.info(f"✨ Optimizing DualPromptLearner (Implicit Stream)")
        params.append({'params': model.prompt_learner.parameters(), 'lr': args.lr})
        
    # 自动发现 KIM/PNuRL
    if hasattr(model, 'pnurl'): 
        logger.info(f"✨ Optimizing KIM/PNuRL (Explicit Stream)")
        params.append({'params': model.pnurl.parameters(), 'lr': args.lr})
    elif hasattr(model, 'kim'):
        logger.info(f"✨ Optimizing KIM (Explicit Stream)")
        params.append({'params': model.kim.parameters(), 'lr': args.lr})

    # Adapter
    adapter_params = [p for n, p in model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
    if adapter_params:
        logger.info(f"✨ Optimizing Adapters ({len(adapter_params)} tensors)")
        params.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    
    # 🔥 初始化 Loss (Utils)
    criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)
    
    scaler = GradScaler() if args.use_amp else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # --- 训练循环 ---
    best_dice = 0.0
    best_aji = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 1. Train
        loss, m_loss, h_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        # 2. Val
        val_res = validate_one_epoch(args, model, val_loader, epoch)
        
        dice = val_res.get('dice', 0.0)
        aji = val_res.get('mAJI', 0.0)
        pq = val_res.get('mPQ', 0.0)
        
        # 3. Log
        logger.info(
            f"Ep {epoch+1}/{args.epochs} | "
            f"Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}) | "
            f"Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}"
        )
        
        # 4. Save Best (通常 AJI 是实例分割的核心指标，可以按 AJI 或 Dice 保存)
        if aji > best_aji:
            best_aji = aji
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_model.pth"))
            logger.info(f"⭐ Best Model Saved (AJI: {best_aji:.4f})")
        
        # 5. Scheduler Step
        scheduler.step()
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))

    total_time = time.time() - start_time
    logger.info(f"🏁 Training Finished. Total time: {datetime.timedelta(seconds=int(total_time))}")

if __name__ == '__main__':
    args = parse_args()
    main(args)