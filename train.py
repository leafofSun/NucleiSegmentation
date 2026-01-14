import argparse
import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import logging
import math

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

# 🔥 核心工具导入
from utils import FocalDiceloss_IoULoss, point_guidance_loss, get_logger

# 🔥 高性能指标导入
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
    
    # --- 数据路径 ---
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/MoNuSeg_SA1B/medical_knowledge.json", 
                        help="Path to the generated Explicit Knowledge Base JSON")
    
    # --- 图像参数 ---
    parser.add_argument("--image_size", type=int, default=1024, help="SAM input resolution (Target Size)")
    parser.add_argument("--crop_size", type=int, default=256, help="Physical Patch Size (Source Size)") # 🔥 256->1024
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
    parser.add_argument("--mask_weight", type=float, default=2.0, help="Weight for Segmentation Loss")
    parser.add_argument("--heatmap_weight", type=float, default=1.0, help="Weight for Auto-Prompt Heatmap Loss")
    parser.add_argument("--attr_weight", type=float, default=0.1, help="Weight for Attribute Classification Loss")

    # --- 验证指标 ---
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], 
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
    """调整 SAM 位置编码尺寸 (以防 checkpoint 尺寸不匹配)"""
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

# 🔥 [新增] 滑动窗口推理 (解决验证集尺度不匹配问题)
def sliding_window_inference(model, image, patch_size=256, target_size=1024, stride=256, device='cuda'):
    """
    1. 切割: image (H, W) -> patch_size (256)
    2. 放大: 256 -> 1024 (适配训练时的 Scale)
    3. 预测: SAM Inference
    4. 缩小: 1024 -> 256
    5. 拼接: 还原到 (H, W)
    """
    C, H, W = image.shape
    
    # 初始化全图概率图
    full_prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # 计算步长
    h_steps = math.ceil((H - patch_size) / stride) + 1
    w_steps = math.ceil((W - patch_size) / stride) + 1

    for h_idx in range(h_steps):
        for w_idx in range(w_steps):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + patch_size, H)
            x2 = min(x1 + patch_size, W)
            
            # 修正边缘：如果最后一步超出边界，就往回退，保证 patch 大小固定为 256
            if y2 - y1 < patch_size: y1 = max(0, y2 - patch_size)
            if x2 - x1 < patch_size: x1 = max(0, x2 - patch_size)
            
            # 1. Crop Patch [3, 256, 256]
            patch = image[:, y1:y1+patch_size, x1:x1+patch_size]
            
            # 2. Resize to 1024 (Model Input)
            # 必须使用 bilinear 插值，且 unsqueeze 增加 batch 维度
            patch_1024 = F.interpolate(
                patch.unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 3. Model Predict
            # 构造验证时的 Prompt (使用通用 Prompt，因为验证时我们不知道具体属性)
            model_input = [{
                'image': patch_1024.squeeze(0), 
                'original_size': (target_size, target_size),
                'text_prompt': "Cell nuclei",
                'organ_id': 9, # Generic
                'attribute_text': "Cell nuclei" 
            }]
            
            with torch.no_grad():
                out = model(model_input, multimask_output=True)
                iou_preds = out[0]['iou_predictions']
                best_idx = torch.argmax(iou_preds).item()
                # 获取 Logits [1024, 1024]
                pred_logits_1024 = out[0]['masks'][0, best_idx]
            
            # 4. Resize back to 256
            pred_logits_256 = F.interpolate(
                pred_logits_1024.unsqueeze(0).unsqueeze(0), 
                size=(patch_size, patch_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            # 转概率
            pred_prob_256 = torch.sigmoid(pred_logits_256)

            # 5. Stitch (累加)
            full_prob_map[y1:y1+patch_size, x1:x1+patch_size] += pred_prob_256
            count_map[y1:y1+patch_size, x1:x1+patch_size] += 1

    # 取平均处理重叠区域
    full_prob_map /= torch.clamp(count_map, min=1.0)
    return full_prob_map

# ==================================================================================================
# 3. 训练逻辑 (Train Loop)
# ==================================================================================================
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    attr_losses = []
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']
        
        optimizer.zero_grad()

        # === 构建 MP-SAM 数据流 ===
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        base_texts = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))
        attr_labels = batched_input.get('attr_labels', None)

        for i in range(len(images)):
            model_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size),
                'organ_id': organ_ids[i] if organ_ids is not None else 9,
                'attribute_text': attr_texts[i],
                'text_prompt': base_texts[i],
                'attr_labels': attr_labels[i] if attr_labels is not None else None
            })

        # === Forward Pass ===
        with autocast('cuda', enabled=args.use_amp):
            outputs = model(model_input, multimask_output=True)
            
            loss_batch = 0
            loss_m_accum = 0
            loss_h_accum = 0
            loss_attr_accum = 0
            
            for i, out in enumerate(outputs):
                # A. Mask Loss
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2: iou_preds = iou_preds.squeeze(0)
                best_idx = torch.argmax(iou_preds).item()
                
                pred_mask = out['masks'][best_idx, :, :] if out['masks'].ndim==3 else out['masks'][0, best_idx]
                pred_iou = iou_preds[best_idx]
                gt_mask = labels[i].squeeze(0).float()
                
                # 尺寸对齐 (防患未然)
                if pred_mask.shape != gt_mask.shape:
                      gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                loss_m, _ = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                # B. Heatmap Loss
                pred_heatmap = out['heatmap_logits']
                with torch.no_grad():
                    target_mask = labels[i].float().unsqueeze(0)
                    gt_nuclei = F.interpolate(target_mask, size=pred_heatmap.shape[-2:], mode='nearest').squeeze(0)
                    gt_nuclei[gt_nuclei==255] = 0
                
                loss_h = point_guidance_loss(pred_heatmap, gt_nuclei.unsqueeze(0))
                
                # C. Attribute Loss
                loss_attr = out.get('pnurl_loss', None)
                if loss_attr is None or not isinstance(loss_attr, torch.Tensor):
                    loss_attr = torch.tensor(0.0, device=loss_m.device, requires_grad=True)
                elif loss_attr.dim() > 0:
                    loss_attr = loss_attr.mean()
                
                # D. Sum
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h + args.attr_weight * loss_attr
                
                loss_batch += loss_i
                loss_m_accum += loss_m.item()
                loss_h_accum += loss_h.item()
                loss_attr_accum += loss_attr.item()
            
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
        attr_losses.append(loss_attr_accum / len(images))
        
        prompt_preview = attr_texts[0][:15] + ".." if len(attr_texts[0]) > 15 else attr_texts[0]
        pbar.set_postfix(Loss=f"{final_loss.item():.3f}", Prompt=prompt_preview)

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses), np.mean(attr_losses)

# ==================================================================================================
# 4. 验证逻辑 (Val Loop - 修复版)
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    val_results = {k: [] for k in args.metrics} 
    
    pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val")
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image'] # [B, 3, H, W] 注意这里是原图尺寸(约1000x1000)
        labels = batched_input['label'].cpu().numpy()
        
        for i in range(len(images)):
            # 🔥 [关键修复] 使用滑动窗口推理
            # patch_size=256 (与训练时的 crop_size 一致)
            # target_size=1024 (模型输入尺寸)
            prob_map = sliding_window_inference(
                model, images[i], 
                patch_size=args.crop_size, 
                target_size=args.image_size, 
                stride=args.crop_size, # 不重叠步长，追求速度
                device=args.device
            )
            
            # 转为二值 Mask (Threshold 0.5)
            pred_mask = (prob_map.cpu().numpy() > 0.5).astype(np.uint8)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            
            gt_valid = gt.copy()
            gt_valid[gt == 255] = 0
            
            # 计算指标
            res = SegMetrics(pred_mask, gt_valid, args.metrics)
            
            for k in args.metrics:
                if k in res: val_results[k].append(res[k])
        
        if 'mAJI' in args.metrics and len(val_results['mAJI']) > 0:
            pbar.set_postfix(AJI=f"{val_results['mAJI'][-1]:.3f}")
                
    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    return avg_results

# ==================================================================================================
# 5. 主程序 (Main)
# ==================================================================================================
def main(args):
    setup_seed(args.seed)
    
    # --- 日志 ---
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{timestamp}.log"))
    
    logger.info(f"🚀 [Start] MP-SAM (Scale: {args.crop_size}->{args.image_size})")

    # --- 数据加载 ---
    # 训练集: crop=256, prompt_mode=dynamic
    train_dataset = TrainingDataset(
        os.path.join(args.data_path, "train"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, 
        crop_size=args.crop_size, # 256
        mode='train',
        prompt_mode='dynamic'
    )
    # 验证集: prompt_mode=generic (验证时不需要动态任务)
    val_dataset = TrainingDataset(
        os.path.join(args.data_path, "test"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, 
        crop_size=args.crop_size, # 256 (用于滑动窗口参数)
        mode='test',
        prompt_mode='generic'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=stack_dict_batched, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=stack_dict_batched, pin_memory=True)
    
    logger.info(f"📊 Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # --- 模型构建 ---
    vanilla_sam = sam_model_registry[args.model_type](args)
    if os.path.exists(args.sam_checkpoint):
        logger.info(f"📥 Loading checkpoint: {args.sam_checkpoint}")
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location='cpu')
            state_dict = ckpt.get("model", ckpt)
            state_dict = resize_pos_embed(state_dict, vanilla_sam.state_dict())
            vanilla_sam.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint loading failed: {e}")
    
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name=args.clip_model,
        num_organs=args.num_organs
    ).to(args.device)
    
    del vanilla_sam

    # --- Adapter Reset ---
    if args.encoder_adapter:
        for n, p in model.image_encoder.named_parameters():
            if "Adapter" in n and "weight" in n:
                torch.nn.init.zeros_(p)

    # --- 优化器 ---
    params = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 5}
    ]
    if hasattr(model, 'prompt_learner'):
        params.append({'params': model.prompt_learner.parameters(), 'lr': args.lr})
    if hasattr(model, 'pnurl'): 
        params.append({'params': model.pnurl.parameters(), 'lr': args.lr})
    elif hasattr(model, 'kim'):
        params.append({'params': model.kim.parameters(), 'lr': args.lr})
        
    adapter_params = [p for n, p in model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
    if adapter_params:
        params.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)
    scaler = GradScaler() if args.use_amp else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # --- Loop ---
    best_aji = 0.0
    
    for epoch in range(args.epochs):
        loss, m_loss, h_loss, a_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        val_res = validate_one_epoch(args, model, val_loader, epoch)
        
        dice = val_res.get('dice', 0.0)
        aji = val_res.get('mAJI', 0.0)
        pq = val_res.get('mPQ', 0.0)
        
        logger.info(
            f"Ep {epoch+1}/{args.epochs} | "
            f"Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | "
            f"Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}"
        )
        
        if aji > best_aji:
            best_aji = aji
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_model.pth"))
            logger.info(f"⭐ Best Model Saved (AJI: {best_aji:.4f})")
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    args = parse_args()
    main(args)