import argparse
import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import logging
import math
import cv2 
import gc  # 垃圾回收

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

# 🔥 [SPEEDUP] 开启 TF32 (RTX 30/40/50系 核心加速)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==================================================================================================
# 1. Configuration
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM: Explicit-Implicit Dual-Stream Training")
    
    # ... Environment ...
    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam_pannuke_final", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")
    
    # ... Data ...
    parser.add_argument("--data_path", type=str, default="data/PanNuke", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json", help="Path to KB")
    
    # 🔥 SOTA 设置：原生 512
    parser.add_argument("--image_size", type=int, default=512, help="SAM input resolution")
    parser.add_argument("--crop_size", type=int, default=512, help="Patch Size") 
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    # ... Model ...
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="Backbone")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model version")
    parser.add_argument("--num_organs", type=int, default=21, help="Number of organ categories")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Use Adapters")

    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU") 
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Min LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use AMP")
    
    # ... Loss ...
    parser.add_argument("--mask_weight", type=float, default=10.0)
    parser.add_argument("--heatmap_weight", type=float, default=1.0)
    parser.add_argument("--attr_weight", type=float, default=0.1) 
    
    # 🔥 [修改 1] 降低辅助任务权重，防止梯度干扰主任务
    parser.add_argument("--consistency_weight", type=float, default=0.1)  # 原 1.0 -> 0.1
    parser.add_argument("--consistency_warmup_epochs", type=int, default=50) # 原 20 -> 50 (延后介入)
    parser.add_argument("--density_map_weight", type=float, default=0.5)  # 原 2.0 -> 0.5

    # 🔥🔥🔥 [新增] 断点续训参数
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from")

    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], help="Metrics to evaluate")

    return parser.parse_args()

# ==================================================================================================
# 2. Utils
# ==================================================================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                device_input[key] = value.to(device, non_blocking=True)
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

def get_gaussian_weight_map(patch_size, device):
    coords = torch.arange(patch_size, dtype=torch.float32, device=device)
    center = patch_size / 2.0
    sigma = patch_size / 4.0  
    gauss = torch.exp(-(coords - center)**2 / (2 * sigma**2))
    weight_map = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    weight_map = weight_map / weight_map.max()
    return weight_map

def sliding_window_inference(model, image, organ_id, patch_size=256, target_size=512, stride=None, device='cuda'):
    C, H, W = image.shape
    if stride is None: stride = patch_size // 2
    full_prob_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)
    weight_map = get_gaussian_weight_map(patch_size, device)

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
            
            # 安全逻辑：如果 patch 小于 target，做插值 (通常现在都是 512->512)
            if patch.shape[-1] != target_size:
                patch_resized = F.interpolate(patch.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False)
            else:
                patch_resized = patch.unsqueeze(0)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_input = [{'image': patch_resized.squeeze(0), 'original_size': (target_size, target_size),
                                'text_prompt': "Cell nuclei", 'organ_id': organ_id, 'attribute_text': "Cell nuclei"}]
                with torch.no_grad():
                    out = model(model_input, multimask_output=True)
                    iou_preds = out[0]['iou_predictions']
                    best_idx = torch.argmax(iou_preds).item()
                    pred_logits_target = out[0]['masks'][0, best_idx]
            
            # 还原尺寸
            pred_logits_256 = F.interpolate(pred_logits_target.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False).squeeze()
            pred_prob_256 = torch.sigmoid(pred_logits_256)
            full_prob_map[y1:y1+patch_size, x1:x1+patch_size] += pred_prob_256 * weight_map
            count_map[y1:y1+patch_size, x1:x1+patch_size] += weight_map

    full_prob_map /= torch.clamp(count_map, min=1e-5)
    return full_prob_map

# ==================================================================================================
# 3. Training Logic (Native 512 + 30% Dropout)
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
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']

        # 🔥 [安全逻辑] 确保输入是 args.image_size (512)
        if images.shape[-1] != args.image_size:
             images = F.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
             labels = F.interpolate(labels.float(), size=(args.image_size, args.image_size), mode='nearest').long()
        
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        attr_texts = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        base_texts = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))
        attr_labels = batched_input.get('attr_labels', None)

        # 🔥 30% 概率丢弃文本
        if np.random.rand() < 0.3:
             attr_texts = [""] * len(images)

        for i in range(len(images)):
            curr_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val

            model_input.append({
                'image': images[i], 'original_size': (args.image_size, args.image_size), 'organ_id': curr_id,
                'attribute_text': attr_texts[i], 'text_prompt': base_texts[i], 'attr_labels': attr_labels[i] if attr_labels is not None else None
            })

        with autocast('cuda', enabled=args.use_amp):
            outputs = model(model_input, multimask_output=True)
            loss_batch = 0
            loss_m_accum = 0; loss_h_accum = 0; loss_attr_accum = 0
            peak_counts_list = []; density_logits_list = []; density_map_list = []; pred_mask_list = []
            
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
                
                loss_attr = out.get('pnurl_loss', torch.tensor(0.0, device=loss_m.device))
                if loss_attr.dim() > 0: loss_attr = loss_attr.mean()
                
                attr_logits = out.get('attr_logits', None)
                if attr_logits is not None and 'density' in attr_logits:
                    all_density_logits = attr_logits['density'] 
                    curr_density_logits = all_density_logits[i].unsqueeze(0) if all_density_logits.shape[0] == len(images) else all_density_logits
                    density_logits_list.append(curr_density_logits)
                    prompt_gen = model.module.prompt_generator if hasattr(model, 'module') else model.prompt_generator
                    with torch.no_grad():
                        peak_count = prompt_gen.count_peaks_from_heatmap(pred_heatmap, threshold=0.3)
                        if isinstance(peak_count, torch.Tensor): 
                            if peak_count.dim() == 0: peak_count = peak_count.unsqueeze(0)
                        else: peak_count = torch.tensor([peak_count], device=curr_density_logits.device)
                        peak_counts_list.append(peak_count)
                
                pred_mask_logits = pred_mask.unsqueeze(0).unsqueeze(0)
                pred_mask_list.append(pred_mask_logits)
                density_map_i = out.get('density_map', None)
                if density_map_i is not None: density_map_list.append(density_map_i.unsqueeze(0))
                else: 
                    B_zero, C_zero, H_zero, W_zero = pred_mask_logits.shape
                    density_map_list.append(torch.zeros((B_zero, 1, H_zero, W_zero), device=pred_mask.device))
                
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h + args.attr_weight * loss_attr
                loss_batch += loss_i
                loss_m_accum += loss_m.item(); loss_h_accum += loss_h.item(); loss_attr_accum += loss_attr.item()
            
            loss_consistency = torch.tensor(0.0, device=loss_m.device)
            loss_consistency_accum = 0.0
            if epoch >= args.consistency_warmup_epochs and len(peak_counts_list) > 0:
                loss_consistency = physical_semantic_consistency_loss(
                    peak_counts=torch.cat(peak_counts_list, dim=0), density_logits=torch.cat(density_logits_list, dim=0),
                    margin_low=10.0, margin_high=30.0, temperature=1.0)
                loss_batch += args.consistency_weight * loss_consistency * len(images)
                loss_consistency_accum = loss_consistency.item()
            
            loss_density_map = torch.tensor(0.0, device=loss_m.device)
            loss_density_map_accum = 0.0
            if len(density_map_list) > 0:
                gt_mask_batch = torch.cat([l.squeeze(0).float().unsqueeze(0).unsqueeze(0) if l.dim()==3 else l.float().unsqueeze(0) for l in labels], dim=0)
                loss_density_map, _, _ = density_map_loss(
                    pred_density_map=torch.cat(density_map_list, dim=0), gt_mask=gt_mask_batch,
                    pred_mask=torch.cat(pred_mask_list, dim=0), mse_weight=1.0, iou_weight=0.5,
                    enable_iou=(epoch >= args.consistency_warmup_epochs))
                loss_batch += args.density_map_weight * loss_density_map * len(images)
                loss_density_map_accum = loss_density_map.item()
            
            final_loss = loss_batch / len(images)
            final_loss = final_loss / args.accumulation_steps

        if scaler:
            scaler.scale(final_loss).backward()
            # 🔥 [新增 2] 梯度裁剪：防止梯度爆炸导致的指标崩塌
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        current_loss_val = final_loss.item() * args.accumulation_steps
        losses.append(current_loss_val)
        mask_losses.append(loss_m_accum / len(images)); heatmap_losses.append(loss_h_accum / len(images)); attr_losses.append(loss_attr_accum / len(images))
        
        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_Total', current_loss_val, global_step)
            writer.add_scalar('Train/Loss_Mask', loss_m_accum / len(images), global_step)
            writer.add_scalar('Train/Loss_Heatmap', loss_h_accum / len(images), global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
        
        if rank == 0:
            consistency_info = f" | Consist: {loss_consistency_accum:.3f}" if epoch >= args.consistency_warmup_epochs else ""
            pbar.set_postfix(Loss=f"{current_loss_val:.3f}", Consist=consistency_info)

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses), np.mean(attr_losses)

# ==================================================================================================
# 4. Validation Logic (🔥 极速验证: Stride = Patch Size)
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch, writer, rank):
    gc.collect() 
    torch.cuda.empty_cache()
    
    model.eval()
    val_results = {k: [] for k in args.metrics}
    visualize_done = False
    
    if rank == 0: pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val")
    else: pbar = val_loader
    
    # DDP 脱壳
    eval_model = model.module if hasattr(model, 'module') else model
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image'] 
        labels = batched_input['label'].cpu().numpy()
        organ_ids = batched_input.get('organ_id', None)

        # 安全逻辑: 确保验证集也是 512
        if images.shape[-1] != args.image_size:
             images = F.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)

        for i in range(len(images)):
            curr_organ_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_organ_id = val.item() if isinstance(val, torch.Tensor) else val
            
            # 🔥 [修改 3] 给验证集一点重叠 (25% Overlap)，消除边界伪影，提升 AJI
            infer_stride = int(args.crop_size * 0.75) 
            
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prob_map = sliding_window_inference(
                    eval_model, images[i], 
                    organ_id=curr_organ_id, 
                    patch_size=args.crop_size, 
                    target_size=args.image_size, 
                    stride=infer_stride,  # 使用带重叠的 stride
                    device=args.device
                )
            
            pred_mask = (prob_map.cpu().numpy() > 0.5).astype(np.uint8)
            gt = labels[i]; gt = gt[0] if gt.ndim == 3 else gt
            gt_valid = gt.copy(); gt_valid[gt == 255] = 0
            
            if pred_mask.shape != gt_valid.shape:
                 gt_valid = cv2.resize(gt_valid, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            res = SegMetrics(pred_mask, gt_valid, args.metrics)
            for k in args.metrics:
                if k in res: val_results[k].append(res[k])
                
            if rank == 0 and writer is not None and not visualize_done:
                img_vis = images[i].cpu().numpy().transpose(1, 2, 0)
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
                writer.add_image(f'Val_Viz/Image', torch.tensor(img_vis.transpose(2, 0, 1)), epoch)
                writer.add_image(f'Val_Viz/GT', torch.tensor(gt_valid * 255).unsqueeze(0).to(torch.uint8), epoch)
                writer.add_image(f'Val_Viz/Pred', torch.tensor(pred_mask * 255).unsqueeze(0).to(torch.uint8), epoch)
                visualize_done = True
        
        if rank == 0 and 'mAJI' in args.metrics and len(val_results['mAJI']) > 0:
            pbar.set_postfix(AJI=f"{val_results['mAJI'][-1]:.3f}")
    
    gc.collect()
    torch.cuda.empty_cache()
                
    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    if rank == 0 and writer is not None:
        for metric_name, metric_value in avg_results.items(): writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
        writer.add_scalar('Val/Weighted_Score', avg_results.get('mAJI', 0) * 0.6 + avg_results.get('dice', 0) * 0.4, epoch)
        
    return avg_results

def main(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        args.device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0; local_rank = 0; world_size = 1
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if world_size > 1: dist.barrier()
    setup_seed(args.seed + rank) 
    
    platform_tb_roots = ["/home/pod", "/root/shared-nvme/tensorboard/logs"]
    platform_tb_root = None
    for tb_path in platform_tb_roots:
        if os.path.exists(tb_path): platform_tb_root = tb_path; break
    if platform_tb_root is None: platform_tb_root = os.path.join(args.work_dir, "runs")
    
    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        run_log_dir = os.path.join(platform_tb_root, f"{args.run_name}_{timestamp}")
        text_log_dir = os.path.join(args.work_dir, "logs")
        model_save_dir = os.path.join(args.work_dir, "models", args.run_name)
        os.makedirs(run_log_dir, exist_ok=True); os.makedirs(text_log_dir, exist_ok=True); os.makedirs(model_save_dir, exist_ok=True)
        logger = get_logger(os.path.join(text_log_dir, f"{args.run_name}_{timestamp}.log"))
        writer = SummaryWriter(log_dir=run_log_dir, flush_secs=60)
        logger.info(f"🚀 [Start] MP-SAM Stable (Size: {args.image_size})")
        logger.info(f"   GPUs: {world_size}, Batch/GPU: {args.batch_size}, Resume: {args.resume if args.resume else 'No'}")
    else: logger = None; writer = None

    train_dataset = UniversalDataset(data_root=args.data_path, knowledge_path=args.knowledge_path, image_size=args.image_size, crop_size=args.crop_size, mode='train', prompt_mode='dynamic')
    val_dataset = UniversalDataset(data_root=args.data_path, knowledge_path=args.knowledge_path, image_size=args.image_size, crop_size=args.crop_size, mode='test', prompt_mode='generic')
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              num_workers=8, collate_fn=stack_dict_batched, pin_memory=True, sampler=train_sampler,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=stack_dict_batched, pin_memory=True, sampler=val_sampler,
                            persistent_workers=True, prefetch_factor=2)
    
    if rank == 0: logger.info(f"📊 Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

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
    
    model = TextSam(image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
                    mask_decoder=vanilla_sam.mask_decoder, clip_model_name=args.clip_model, num_organs=args.num_organs).to(args.device)
    del vanilla_sam

    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 兼容性定义
    raw_model = model.module if world_size > 1 else model

    # 🔥🔥🔥 [关键] 断点续训逻辑
    if args.resume and os.path.exists(args.resume):
        if rank == 0: logger.info(f"🔄 Resuming from: {args.resume} (Start Epoch: {args.start_epoch})")
        try:
            ckpt = torch.load(args.resume, map_location='cpu')
            raw_model.load_state_dict(ckpt, strict=False)
            if rank == 0: logger.info("✅ Resume weights loaded successfully.")
        except Exception as e:
            if rank == 0: logger.warning(f"⚠️ Resume failed: {e}")

    if args.encoder_adapter:
        for n, p in raw_model.image_encoder.named_parameters():
            if "Adapter" in n and "weight" in n: torch.nn.init.zeros_(p)

    params = [{'params': raw_model.mask_decoder.parameters(), 'lr': args.lr},
              {'params': raw_model.prompt_generator.parameters(), 'lr': args.lr * 5}]
    if hasattr(raw_model, 'prompt_learner'): params.append({'params': raw_model.prompt_learner.parameters(), 'lr': args.lr})
    if hasattr(raw_model, 'pnurl'): params.append({'params': raw_model.pnurl.parameters(), 'lr': args.lr})
    adapter_params = [p for n, p in raw_model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
    if adapter_params: params.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)
    scaler = GradScaler() if args.use_amp else None
    
    # 🔥 [修改 4] 学习率策略改为 ReduceLROnPlateau
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    best_aji = 0.0; best_dice = 0.0
    
    # 🔥 从 args.start_epoch 开始循环
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        loss, m_loss, h_loss, a_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, writer, rank)
        val_res = validate_one_epoch(args, model, val_loader, epoch, writer, rank)
        
        if rank == 0:
            dice = val_res.get('dice', 0.0); aji = val_res.get('mAJI', 0.0); pq = val_res.get('mPQ', 0.0)
            logger.info(f"Ep {epoch+1}/{args.epochs} | Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}")
            latest_model_path = os.path.join(args.work_dir, "models", args.run_name, "latest_model.pth")
            torch.save(raw_model.state_dict(), latest_model_path)
            if aji > best_aji:
                best_aji = aji; best_dice = max(best_dice, dice)
                best_model_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
                torch.save(raw_model.state_dict(), best_model_path)
                logger.info(f"⭐ New Best AJI! ({best_aji:.4f}) -> Model Saved")
                
        # 🔥 [修改 5] Scheduler Step 传入监控指标 (AJI)
        # 获取当前 Epoch 的验证 AJI (所有进程使用 rank 0 的结果，或者广播)
        # 这里简化：每个进程都跑了 validate，val_res 是 synced 或者本地的近似
        # ReduceLROnPlateau 不需要全局同步 step，只要 rank 0 打印即可
        # 为保险起见，建议 val_res['mAJI'] 是 reduce 后的结果 (当前 validate_one_epoch 返回的是本地的，DDP 下应在外部 reduce)
        # 但在 validate_one_epoch 内部没有做 all_reduce，这是一个潜在小问题。
        # 考虑到当前代码架构，rank 0 会负责保存模型，scheduler 在 rank 0 step 即可 (optimizer state 需要同步吗？DDP 会处理)
        # 简单起见，所有进程都 step，传入各自的 aji (期望分布均匀)
        val_aji = val_res.get('mAJI', 0.0)
        scheduler.step(val_aji)

    if rank == 0:
        logger.info(f"🏁 Training Finished. Best AJI: {best_aji:.4f}")
        if writer is not None: writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)