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
from utils import FocalDiceloss_IoULoss, point_guidance_loss, get_logger, physical_semantic_consistency_loss

# 🔥 高性能指标导入
from metrics import SegMetrics

# ==================================================================================================
# 1. 参数配置 (Configuration)
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM: Explicit-Implicit Dual-Stream Training")
    
    # --- 基础环境 ---
    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # --- 数据路径 ---
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/MoNuSeg_SA1B/medical_knowledge.json", 
                        help="Path to the generated Explicit Knowledge Base JSON")
    
    # --- 图像参数 ---
    parser.add_argument("--image_size", type=int, default=1024, help="SAM input resolution (Target Size)")
    parser.add_argument("--crop_size", type=int, default=256, help="Physical Patch Size (Source Size)") 
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    # --- 模型配置 ---
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Path to original/medsam checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model version for Text Encoder")
    parser.add_argument("--num_organs", type=int, default=21, help="Number of organ categories for DualPromptLearner (including Generic)")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Use Adapters in Image Encoder")

    # --- 训练超参 ---
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision")
    
    # --- Loss 权重 ---
    parser.add_argument("--mask_weight", type=float, default=2.0, help="Weight for Segmentation Loss")
    parser.add_argument("--heatmap_weight", type=float, default=1.0, help="Weight for Auto-Prompt Heatmap Loss")
    parser.add_argument("--attr_weight", type=float, default=0.1, help="Weight for Attribute Classification Loss")
    parser.add_argument("--consistency_weight", type=float, default=0.5, help="Weight for Physical-Semantic Consistency Loss")
    parser.add_argument("--consistency_warmup_epochs", type=int, default=20, help="Warm-up epochs before applying consistency loss")

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

# 滑动窗口推理
def sliding_window_inference(model, image, organ_id, patch_size=256, target_size=1024, stride=256, device='cuda'):
    C, H, W = image.shape
    
    full_prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    h_steps = math.ceil((H - patch_size) / stride) + 1
    w_steps = math.ceil((W - patch_size) / stride) + 1

    for h_idx in range(h_steps):
        for w_idx in range(w_steps):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + patch_size, H)
            x2 = min(x1 + patch_size, W)
            
            if y2 - y1 < patch_size: y1 = max(0, y2 - patch_size)
            if x2 - x1 < patch_size: x1 = max(0, x2 - patch_size)
            
            patch = image[:, y1:y1+patch_size, x1:x1+patch_size]
            
            patch_1024 = F.interpolate(
                patch.unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            model_input = [{
                'image': patch_1024.squeeze(0), 
                'original_size': (target_size, target_size),
                'text_prompt': "Cell nuclei",
                'organ_id': organ_id, # 使用传入的真实 organ_id
                'attribute_text': "Cell nuclei" 
            }]
            
            with torch.no_grad():
                out = model(model_input, multimask_output=True)
                iou_preds = out[0]['iou_predictions']
                best_idx = torch.argmax(iou_preds).item()
                pred_logits_1024 = out[0]['masks'][0, best_idx]
            
            pred_logits_256 = F.interpolate(
                pred_logits_1024.unsqueeze(0).unsqueeze(0), 
                size=(patch_size, patch_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            pred_prob_256 = torch.sigmoid(pred_logits_256)

            full_prob_map[y1:y1+patch_size, x1:x1+patch_size] += pred_prob_256
            count_map[y1:y1+patch_size, x1:x1+patch_size] += 1

    full_prob_map /= torch.clamp(count_map, min=1.0)
    return full_prob_map

# ==================================================================================================
# 3. 训练逻辑
# ==================================================================================================
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    attr_losses = []
    
    optimizer.zero_grad() # 初始化
    
    for batch_idx, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']
        
        # === 构建 MP-SAM 数据流 ===
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        base_texts = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))
        attr_labels = batched_input.get('attr_labels', None)

        for i in range(len(images)):
            # 兼容处理 organ_id (如果是 Tensor 则转 int)
            curr_id = 9
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val

            model_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size),
                'organ_id': curr_id,
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
            loss_consistency_accum = 0
            
            # 收集所有样本的峰值计数和密度 logits（用于批量计算一致性 loss）
            peak_counts_list = []
            density_logits_list = []
            
            for i, out in enumerate(outputs):
                # A. Mask Loss
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2: iou_preds = iou_preds.squeeze(0)
                best_idx = torch.argmax(iou_preds).item()
                
                pred_mask = out['masks'][best_idx, :, :] if out['masks'].ndim==3 else out['masks'][0, best_idx]
                pred_iou = iou_preds[best_idx]
                gt_mask = labels[i].squeeze(0).float()
                
                if pred_mask.shape != gt_mask.shape:
                      gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                loss_m, _ = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                # B. Heatmap Loss
                pred_heatmap = out['heatmap_logits']  # [1, C, H, W]
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
                
                # D. 收集一致性 Loss 所需的数据
                # 从热力图中统计峰值数量
                with torch.no_grad():
                    peak_count = model.prompt_generator.count_peaks_from_heatmap(pred_heatmap, threshold=0.3)  # [1]
                    peak_counts_list.append(peak_count)
                
                # 从属性 logits 中提取密度 logits
                attr_logits = out.get('attr_logits', None)
                if attr_logits is not None and 'density' in attr_logits:
                    density_logits = attr_logits['density']  # [1, 3]
                    density_logits_list.append(density_logits)
                else:
                    # 如果没有，创建一个零 logits（不会产生梯度）
                    density_logits = torch.zeros((1, 3), device=loss_m.device, requires_grad=False)
                    density_logits_list.append(density_logits)
                
                # E. Sum (暂时不包含一致性 loss，稍后批量计算)
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h + args.attr_weight * loss_attr
                
                loss_batch += loss_i
                loss_m_accum += loss_m.item()
                loss_h_accum += loss_h.item()
                loss_attr_accum += loss_attr.item()
            
            # F. 计算一致性 Loss (批量计算，仅在 warm-up 后)
            loss_consistency = torch.tensor(0.0, device=loss_m.device, requires_grad=True)
            loss_consistency_accum = 0.0
            if epoch >= args.consistency_warmup_epochs and len(peak_counts_list) > 0:
                # 合并所有样本
                peak_counts_batch = torch.cat(peak_counts_list, dim=0)  # [B]
                density_logits_batch = torch.cat(density_logits_list, dim=0)  # [B, 3]
                
                # 计算一致性 loss
                loss_consistency = physical_semantic_consistency_loss(
                    peak_counts=peak_counts_batch,
                    density_logits=density_logits_batch,
                    margin_low=10.0,
                    margin_high=30.0,
                    temperature=1.0
                )
                
                # 添加到总 loss
                loss_batch += args.consistency_weight * loss_consistency * len(images)
                loss_consistency_accum = loss_consistency.item()
            
            final_loss = loss_batch / len(images)
            
            # Loss 归一化 (梯度累积)
            final_loss = final_loss / args.accumulation_steps

        # === Backward ===
        if scaler:
            scaler.scale(final_loss).backward()
        else:
            final_loss.backward()
            
        # Step (每 accumulation_steps 次更新一次)
        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # 记录
        current_loss_val = final_loss.item() * args.accumulation_steps
        losses.append(current_loss_val)
        mask_losses.append(loss_m_accum / len(images))
        heatmap_losses.append(loss_h_accum / len(images))
        attr_losses.append(loss_attr_accum / len(images))
        
        # 显示一致性 loss（仅在 warm-up 后）
        consistency_info = ""
        if epoch >= args.consistency_warmup_epochs:
            consistency_info = f" | Consist: {loss_consistency_accum:.3f}"
        
        prompt_preview = attr_texts[0][:15] + ".." if len(attr_texts[0]) > 15 else attr_texts[0]
        pbar.set_postfix(Loss=f"{current_loss_val:.3f}", Prompt=prompt_preview, Consist=consistency_info)

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses), np.mean(attr_losses)

# ==================================================================================================
# 4. 验证逻辑
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    val_results = {k: [] for k in args.metrics} 
    
    pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val")
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image'] 
        labels = batched_input['label'].cpu().numpy()
        organ_ids = batched_input.get('organ_id', None)
        
        for i in range(len(images)):
            # 🔥 [修复点] 安全获取 organ_id
            curr_organ_id = 9
            if organ_ids is not None:
                val = organ_ids[i]
                # 如果是 tensor, 取 .item(), 如果是 int, 直接用
                curr_organ_id = val.item() if isinstance(val, torch.Tensor) else val
            
            # 传入 organ_id
            prob_map = sliding_window_inference(
                model, images[i], 
                organ_id=curr_organ_id, 
                patch_size=args.crop_size, 
                target_size=args.image_size, 
                stride=args.crop_size, 
                device=args.device
            )
            
            pred_mask = (prob_map.cpu().numpy() > 0.5).astype(np.uint8)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            gt_valid = gt.copy()
            gt_valid[gt == 255] = 0
            
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
    
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{timestamp}.log"))
    
    logger.info(f"🚀 [Start] MP-SAM (Scale: {args.crop_size}->{args.image_size})")

    # 数据加载
    train_dataset = TrainingDataset(
        os.path.join(args.data_path, "train"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, 
        crop_size=args.crop_size, 
        mode='train',
        prompt_mode='dynamic'
    )
    val_dataset = TrainingDataset(
        os.path.join(args.data_path, "test"),
        knowledge_path=args.knowledge_path,
        image_size=args.image_size, 
        crop_size=args.crop_size, 
        mode='test',
        prompt_mode='generic'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=stack_dict_batched, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=stack_dict_batched, pin_memory=True)
    
    logger.info(f"📊 Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # 模型构建
    args.checkpoint = args.sam_checkpoint
    vanilla_sam = sam_model_registry[args.model_type](args)
    if os.path.exists(args.sam_checkpoint):
        logger.info(f"📥 Loading checkpoint: {args.sam_checkpoint}")
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location='cpu',weights_only=False)
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

    if args.encoder_adapter:
        for n, p in model.image_encoder.named_parameters():
            if "Adapter" in n and "weight" in n:
                torch.nn.init.zeros_(p)

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

    best_aji = 0.0
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        # 1. Train
        loss, m_loss, h_loss, a_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        # 2. Val (每10轮或最后)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            val_res = validate_one_epoch(args, model, val_loader, epoch)
            
            dice = val_res.get('dice', 0.0)
            aji = val_res.get('mAJI', 0.0)
            pq = val_res.get('mPQ', 0.0)
            
            logger.info(
                f"Ep {epoch+1}/{args.epochs} | "
                f"Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | "
                f"Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}"
            )
            
            # 3. Smart Save
            saved = False
            
            if aji > best_aji:
                best_aji = aji
                best_dice = max(best_dice, dice)
                torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_model.pth"))
                logger.info(f"⭐ New Best AJI! ({best_aji:.4f}) -> Model Saved")
                saved = True
            
            elif (best_aji - aji) < 0.002 and dice > (best_dice + 0.005):
                best_dice = dice
                torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_dice_model.pth"))
                logger.info(f"✨ High Dice Model Found! (Dice: {dice:.4f}, AJI: {aji:.4f}) -> Saved as best_dice_model.pth")
                saved = True
            
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))
            
        else:
            logger.info(
                f"Ep {epoch+1}/{args.epochs} | "
                f"Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | "
                f"Skipping Val"
            )

        scheduler.step()

    logger.info(f"🏁 Training Finished. Best AJI: {best_aji:.4f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)