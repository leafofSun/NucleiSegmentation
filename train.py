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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Native AMP support
try:
    from torch.amp import autocast, GradScaler 
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# === Project Module Imports ===
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import UniversalDataset, stack_dict_batched

# 🔥 Core Tools
from utils import FocalDiceloss_IoULoss, point_guidance_loss, get_logger, physical_semantic_consistency_loss, density_map_loss

# 🔥 Metrics
from metrics import SegMetrics

# ==================================================================================================
# 1. Configuration
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM: Explicit-Implicit Dual-Stream Training")
    
    # ... (Environment args unchanged)
    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam_pannuke_full", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # ... (Data/Image args unchanged)
    parser.add_argument("--data_path", type=str, default="data/PanNuke_SA1B", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke_SA1B/medical_knowledge.json", help="Path to KB")
    parser.add_argument("--image_size", type=int, default=512, help="SAM input resolution (Target Size)")
    parser.add_argument("--crop_size", type=int, default=256, help="Physical Patch Size (Source Size)") 
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    # ... (Model/Training args unchanged)
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Path to checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model version")
    parser.add_argument("--num_organs", type=int, default=21, help="Number of organ categories")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Use Adapters")

    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU") 
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Min LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use AMP")
    
    # ... (Loss/Metrics args unchanged)
    parser.add_argument("--mask_weight", type=float, default=10.0, help="Weight for Segmentation Loss")
    parser.add_argument("--heatmap_weight", type=float, default=1.0, help="Weight for Heatmap Loss")
    parser.add_argument("--attr_weight", type=float, default=0.1, help="Weight for Attribute Loss") 
    parser.add_argument("--consistency_weight", type=float, default=1.0, help="Weight for Consistency Loss")
    parser.add_argument("--consistency_warmup_epochs", type=int, default=20, help="Warm-up epochs")
    parser.add_argument("--density_map_weight", type=float, default=2.0, help="Weight for Density Map Loss")

    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], help="Metrics to evaluate")

    return parser.parse_args()

# ==================================================================================================
# 2. Utils
# ==================================================================================================
# ... (setup_seed, to_device, resize_pos_embed unchanged)
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
    # (保持原样)
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

# 🔥 [核心修改]：生成高斯权重图
def get_gaussian_weight_map(patch_size, device):
    """生成一个中间高、边缘低的 2D 高斯权重图"""
    coords = torch.arange(patch_size, dtype=torch.float32, device=device)
    center = patch_size / 2.0
    sigma = patch_size / 4.0  # 控制高斯的宽度
    
    # 1D 高斯
    gauss = torch.exp(-(coords - center)**2 / (2 * sigma**2))
    
    # 2D 高斯 (外积)
    weight_map = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    
    # 归一化到 [0, 1] 并不是必须的，因为后面会除以 count_map，但保持数值范围一致较好
    weight_map = weight_map / weight_map.max()
    return weight_map

# 🔥 [核心修改]：重叠滑动窗口推理
def sliding_window_inference(model, image, organ_id, patch_size=256, target_size=512, stride=None, device='cuda'):
    C, H, W = image.shape
    
    # 如果没有指定 stride，默认使用 50% 重叠 (解决边界切割问题)
    if stride is None:
        stride = patch_size // 2
    
    full_prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)
    
    # 生成高斯权重图，用于消除边界伪影
    weight_map = get_gaussian_weight_map(patch_size, device)

    h_steps = math.ceil((H - patch_size) / stride) + 1
    w_steps = math.ceil((W - patch_size) / stride) + 1

    for h_idx in range(h_steps):
        for w_idx in range(w_steps):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + patch_size, H)
            x2 = min(x1 + patch_size, W)
            
            # 处理边界情况 (Last Patch)
            if y2 - y1 < patch_size: y1 = max(0, y2 - patch_size)
            if x2 - x1 < patch_size: x1 = max(0, x2 - patch_size)
            
            patch = image[:, y1:y1+patch_size, x1:x1+patch_size]
            
            patch_resized = F.interpolate(
                patch.unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            model_input = [{
                'image': patch_resized.squeeze(0), 
                'original_size': (target_size, target_size),
                'text_prompt': "Cell nuclei",
                'organ_id': organ_id,
                'attribute_text': "Cell nuclei" 
            }]
            
            with torch.no_grad():
                out = model(model_input, multimask_output=True)
                iou_preds = out[0]['iou_predictions']
                best_idx = torch.argmax(iou_preds).item()
                pred_logits_target = out[0]['masks'][0, best_idx]
            
            pred_logits_256 = F.interpolate(
                pred_logits_target.unsqueeze(0).unsqueeze(0), 
                size=(patch_size, patch_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            pred_prob_256 = torch.sigmoid(pred_logits_256)

            # 🔥 [核心逻辑] 加权累加
            # 预测值 * 权重 (中心权重大，边缘权重小)
            full_prob_map[y1:y1+patch_size, x1:x1+patch_size] += pred_prob_256 * weight_map
            # 计数器也加权重，而不是简单 +1
            count_map[y1:y1+patch_size, x1:x1+patch_size] += weight_map

    # 加权平均
    full_prob_map /= torch.clamp(count_map, min=1e-5)
    return full_prob_map

# ==================================================================================================
# 3. Training Logic (包含所有之前的修复)
# ==================================================================================================
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, writer, rank):
    model.train()
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
    else:
        pbar = train_loader
        
    losses = []
    mask_losses = []
    heatmap_losses = []
    attr_losses = []
    
    optimizer.zero_grad()
    
    for batch_idx, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']
        
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        base_texts = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))
        attr_labels = batched_input.get('attr_labels', None)

        for i in range(len(images)):
            curr_id = 20
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

        with autocast('cuda', enabled=args.use_amp):
            outputs = model(model_input, multimask_output=True)
            
            loss_batch = 0
            loss_m_accum = 0
            loss_h_accum = 0
            loss_attr_accum = 0
            
            peak_counts_list = []
            density_logits_list = []
            density_map_list = []
            pred_mask_list = []
            
            for i, out in enumerate(outputs):
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2: iou_preds = iou_preds.squeeze(0)
                best_idx = torch.argmax(iou_preds).item()
                
                pred_mask = out['masks'][best_idx, :, :] if out['masks'].ndim==3 else out['masks'][0, best_idx]
                pred_iou = iou_preds[best_idx]
                gt_mask = labels[i].squeeze(0).float()
                
                if pred_mask.shape != gt_mask.shape:
                      gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                loss_m, _ = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                pred_heatmap = out['heatmap_logits'] 
                with torch.no_grad():
                    target_mask = labels[i].float().unsqueeze(0)
                    gt_nuclei = F.interpolate(target_mask, size=pred_heatmap.shape[-2:], mode='nearest').squeeze(0)
                    gt_nuclei[gt_nuclei==255] = 0
                loss_h = point_guidance_loss(pred_heatmap, gt_nuclei.unsqueeze(0))
                
                loss_attr = out.get('pnurl_loss', None)
                if loss_attr is None or not isinstance(loss_attr, torch.Tensor):
                    loss_attr = torch.tensor(0.0, device=loss_m.device, requires_grad=True)
                elif loss_attr.dim() > 0:
                    loss_attr = loss_attr.mean()
                
                attr_logits = out.get('attr_logits', None)
                if attr_logits is not None and 'density' in attr_logits:
                    all_density_logits = attr_logits['density'] 
                    if all_density_logits.shape[0] == len(images): 
                        curr_density_logits = all_density_logits[i].unsqueeze(0) 
                    else:
                        curr_density_logits = all_density_logits
                    density_logits_list.append(curr_density_logits)
                    
                    prompt_gen = model.module.prompt_generator if hasattr(model, 'module') else model.prompt_generator
                    with torch.no_grad():
                        peak_count = prompt_gen.count_peaks_from_heatmap(pred_heatmap, threshold=0.3)
                        if isinstance(peak_count, torch.Tensor):
                            if peak_count.dim() == 0:
                                peak_count = peak_count.unsqueeze(0) 
                        else:
                            peak_count = torch.tensor([peak_count], device=curr_density_logits.device)
                        peak_counts_list.append(peak_count)
                
                # 🔥 [核心修复] 收集密度图和预测 mask（自动补零逻辑）
                pred_mask_logits = pred_mask.unsqueeze(0).unsqueeze(0)
                pred_mask_list.append(pred_mask_logits)
                
                density_map_i = out.get('density_map', None)
                if density_map_i is not None:
                    density_map_list.append(density_map_i.unsqueeze(0))
                else:
                    B_zero, C_zero, H_zero, W_zero = pred_mask_logits.shape
                    zero_map = torch.zeros((B_zero, 1, H_zero, W_zero), device=pred_mask.device)
                    density_map_list.append(zero_map)
                
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h + args.attr_weight * loss_attr
                loss_batch += loss_i
                loss_m_accum += loss_m.item()
                loss_h_accum += loss_h.item()
                loss_attr_accum += loss_attr.item()
            
            loss_consistency = torch.tensor(0.0, device=loss_m.device, requires_grad=True)
            loss_consistency_accum = 0.0
            if epoch >= args.consistency_warmup_epochs and len(peak_counts_list) > 0:
                peak_counts_batch = torch.cat(peak_counts_list, dim=0)
                density_logits_batch = torch.cat(density_logits_list, dim=0)
                loss_consistency = physical_semantic_consistency_loss(
                    peak_counts=peak_counts_batch,
                    density_logits=density_logits_batch,
                    margin_low=10.0,
                    margin_high=30.0,
                    temperature=1.0
                )
                loss_batch += args.consistency_weight * loss_consistency * len(images)
                loss_consistency_accum = loss_consistency.item()
            
            loss_density_map = torch.tensor(0.0, device=loss_m.device, requires_grad=True)
            loss_density_map_accum = 0.0
            if len(density_map_list) > 0:
                pred_density_batch = torch.cat(density_map_list, dim=0)
                pred_mask_batch = torch.cat(pred_mask_list, dim=0)
                gt_mask_batch = []
                for i in range(len(images)):
                    gt_mask_i = labels[i].squeeze(0).float()
                    if gt_mask_i.dim() == 2:
                        gt_mask_i = gt_mask_i.unsqueeze(0)
                    gt_mask_batch.append(gt_mask_i.unsqueeze(0))
                gt_mask_batch = torch.cat(gt_mask_batch, dim=0)
                
                enable_iou = (epoch >= args.consistency_warmup_epochs)
                loss_density_map, loss_density_mse, loss_density_iou = density_map_loss(
                    pred_density_map=pred_density_batch,
                    gt_mask=gt_mask_batch,
                    pred_mask=pred_mask_batch,
                    mse_weight=1.0,
                    iou_weight=0.5,
                    enable_iou=enable_iou
                )
                loss_batch += args.density_map_weight * loss_density_map * len(images)
                loss_density_map_accum = loss_density_map.item()
            
            final_loss = loss_batch / len(images)
            final_loss = final_loss / args.accumulation_steps

        if scaler:
            scaler.scale(final_loss).backward()
        else:
            final_loss.backward()
            
        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        current_loss_val = final_loss.item() * args.accumulation_steps
        losses.append(current_loss_val)
        mask_losses.append(loss_m_accum / len(images))
        heatmap_losses.append(loss_h_accum / len(images))
        attr_losses.append(loss_attr_accum / len(images))
        
        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_Total', current_loss_val, global_step)
            writer.add_scalar('Train/Loss_Mask', loss_m_accum / len(images), global_step)
            writer.add_scalar('Train/Loss_Heatmap', loss_h_accum / len(images), global_step)
            writer.add_scalar('Train/Loss_Attr', loss_attr_accum / len(images), global_step)
            if epoch >= args.consistency_warmup_epochs:
                writer.add_scalar('Train/Loss_Consistency', loss_consistency_accum, global_step)
            if loss_density_map_accum > 0:
                writer.add_scalar('Train/Loss_DensityMap_Total', loss_density_map_accum, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
        
        if rank == 0:
            consistency_info = ""
            if epoch >= args.consistency_warmup_epochs:
                consistency_info = f" | Consist: {loss_consistency_accum:.3f}"
            pbar.set_postfix(Loss=f"{current_loss_val:.3f}", Consist=consistency_info)

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses), np.mean(attr_losses)

# ==================================================================================================
# 4. Validation Logic
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch, writer, rank):
    model.eval()
    val_results = {k: [] for k in args.metrics}
    visualize_done = False
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val")
    else:
        pbar = val_loader
    
    eval_model = model.module if hasattr(model, 'module') else model
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image'] 
        labels = batched_input['label'].cpu().numpy()
        organ_ids = batched_input.get('organ_id', None)
        
        for i in range(len(images)):
            curr_organ_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_organ_id = val.item() if isinstance(val, torch.Tensor) else val
            
            # 🔥 [核心修改] 验证推理时，启用重叠滑动 (stride = crop_size // 2)
            prob_map = sliding_window_inference(
                eval_model, images[i], 
                organ_id=curr_organ_id, 
                patch_size=args.crop_size, 
                target_size=args.image_size, 
                stride=args.crop_size // 2,  # 🔥 关键：50% 重叠
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
                
            if rank == 0 and writer is not None and not visualize_done:
                img_vis = images[i].cpu().numpy().transpose(1, 2, 0)
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
                gt_vis = (gt_valid * 255).astype(np.uint8)
                pred_vis = (pred_mask * 255).astype(np.uint8)
                writer.add_image(f'Val_Viz/Image', torch.tensor(img_vis.transpose(2, 0, 1)), epoch)
                writer.add_image(f'Val_Viz/GT', torch.tensor(gt_vis).unsqueeze(0), epoch)
                writer.add_image(f'Val_Viz/Pred', torch.tensor(pred_vis).unsqueeze(0), epoch)
                visualize_done = True
        
        if rank == 0 and 'mAJI' in args.metrics and len(val_results['mAJI']) > 0:
            pbar.set_postfix(AJI=f"{val_results['mAJI'][-1]:.3f}")
                
    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    if rank == 0 and writer is not None:
        for metric_name, metric_value in avg_results.items():
            writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
        main_score = avg_results.get('mAJI', 0) * 0.6 + avg_results.get('dice', 0) * 0.4
        writer.add_scalar('Val/Weighted_Score', main_score, epoch)
        
    return avg_results

# ... (Main function unchanged)
def main(args):
    # (Same as before...)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        args.device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if world_size > 1:
        dist.barrier()

    setup_seed(args.seed + rank) 
    
    platform_tb_roots = ["/home/pod", "/root/shared-nvme/tensorboard/logs"]
    platform_tb_root = None
    for tb_path in platform_tb_roots:
        if os.path.exists(tb_path):
            platform_tb_root = tb_path
            break
    if platform_tb_root is None:
        platform_tb_root = os.path.join(args.work_dir, "runs")
        if rank == 0:
            print(f"⚠️ 警告：未找到平台默认日志目录，将使用本地目录: {platform_tb_root}")
    
    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        run_log_dir = os.path.join(platform_tb_root, f"{args.run_name}_{timestamp}")
        text_log_dir = os.path.join(args.work_dir, "logs")
        model_save_dir = os.path.join(args.work_dir, "models", args.run_name)
        os.makedirs(run_log_dir, exist_ok=True)
        os.makedirs(text_log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        logger = get_logger(os.path.join(text_log_dir, f"{args.run_name}_{timestamp}.log"))
        writer = SummaryWriter(log_dir=run_log_dir, flush_secs=60)
        logger.info(f"🚀 [Start] MP-SAM (Scale: {args.crop_size}->{args.image_size})")
        logger.info(f"   GPUs: {world_size}, Batch/GPU: {args.batch_size}")
        logger.info(f"📊 TensorBoard logs saved to: {run_log_dir}")
    else:
        logger = None
        writer = None

    train_dataset = UniversalDataset(
        data_root=args.data_path, knowledge_path=args.knowledge_path,
        image_size=args.image_size, crop_size=args.crop_size, mode='train', prompt_mode='dynamic'
    )
    val_dataset = UniversalDataset(
        data_root=args.data_path, knowledge_path=args.knowledge_path,
        image_size=args.image_size, crop_size=args.crop_size, mode='test', prompt_mode='generic'
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              num_workers=8, collate_fn=stack_dict_batched, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=stack_dict_batched, pin_memory=True, sampler=val_sampler)
    
    if rank == 0:
        logger.info(f"📊 Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    args.checkpoint = args.sam_checkpoint
    vanilla_sam = sam_model_registry[args.model_type](args)
    
    if os.path.exists(args.sam_checkpoint):
        if rank == 0: logger.info(f"📥 Loading checkpoint: {args.sam_checkpoint}")
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            state_dict = resize_pos_embed(state_dict, vanilla_sam.state_dict())
            vanilla_sam.load_state_dict(state_dict, strict=False)
        except Exception as e:
            if rank == 0: logger.warning(f"⚠️ Checkpoint loading failed: {e}")
    
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder, clip_model_name=args.clip_model, num_organs=args.num_organs
    ).to(args.device)
    
    del vanilla_sam

    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if args.encoder_adapter:
        raw_model = model.module if world_size > 1 else model
        for n, p in raw_model.image_encoder.named_parameters():
            if "Adapter" in n and "weight" in n:
                torch.nn.init.zeros_(p)

    raw_model = model.module if world_size > 1 else model
    params = [
        {'params': raw_model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': raw_model.prompt_generator.parameters(), 'lr': args.lr * 5}
    ]
    if hasattr(raw_model, 'prompt_learner'):
        params.append({'params': raw_model.prompt_learner.parameters(), 'lr': args.lr})
    if hasattr(raw_model, 'pnurl'): 
        params.append({'params': raw_model.pnurl.parameters(), 'lr': args.lr})
        
    adapter_params = [p for n, p in raw_model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
    if adapter_params:
        params.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)
    scaler = GradScaler() if args.use_amp else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_aji = 0.0
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
            
        loss, m_loss, h_loss, a_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, writer, rank)
        val_res = validate_one_epoch(args, model, val_loader, epoch, writer, rank)
        
        if rank == 0:
            dice = val_res.get('dice', 0.0)
            aji = val_res.get('mAJI', 0.0)
            pq = val_res.get('mPQ', 0.0)
            
            logger.info(
                f"Ep {epoch+1}/{args.epochs} | "
                f"Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | "
                f"Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}"
            )
            
            # 保存 latest_model.pth (覆盖式，每个 epoch 都更新)
            latest_model_path = os.path.join(args.work_dir, "models", args.run_name, "latest_model.pth")
            torch.save(raw_model.state_dict(), latest_model_path)
            
            if aji > best_aji:
                best_aji = aji
                best_dice = max(best_dice, dice)
                best_model_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
                torch.save(raw_model.state_dict(), best_model_path)
                logger.info(f"⭐ New Best AJI! ({best_aji:.4f}) -> Model Saved")

        scheduler.step()

    if rank == 0:
        logger.info(f"🏁 Training Finished. Best AJI: {best_aji:.4f}")
        if writer is not None:
            writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)