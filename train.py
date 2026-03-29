import argparse
import os
import time
import datetime
import random
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm
import logging
import math
import cv2 
import gc  
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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
from metrics import SegMetrics

# HoVer-style HV supervision
from hover_loss import msge_loss, generate_hv_map_from_inst

# 后处理依赖
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label as skimage_label

# 🔥 [SPEEDUP] 开启 TF32 (RTX 30/40/50系 核心加速)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==================================================================================================
# 1. Configuration
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM: Explicit-Implicit Dual-Stream Training")
    
    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam_pannuke_final", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")
    
    parser.add_argument("--data_path", type=str, default="data/PanNuke", help="Root directory of dataset")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json", help="Path to KB")
    
    parser.add_argument("--image_size", type=int, default=512, help="SAM input resolution")
    parser.add_argument("--crop_size", type=int, default=256, help="Patch Size") 
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="Backbone")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model version")
    parser.add_argument("--num_organs", type=int, default=21, help="Number of organ categories")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Use Adapters")
    parser.add_argument("--hv_weight", type=float, default=1.0, help="Weight for HoVer HV supervision loss")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for multi-modal fusion")

    parser.add_argument("--use_pnurl", action='store_true', default=False, help="启用 PNuRL 属性预测分支")
    parser.add_argument("--use_coop", action='store_true', default=False, help="启用 CoOp 可学习文本提示")
    parser.add_argument("--use_sgot", action='store_true', default=False, help="启用 SG-OT 语义引导最优传输模块")
    parser.add_argument("--use_asr", action='store_true', default=False, help="启用 ASR 自适应谱细化上采样模块")
    parser.add_argument("--prompt_mode", type=str, default="dynamic", choices=["generic", "dynamic"], help="训练时的文本提示生成模式")

    parser.add_argument("--sg_epsilon", type=float, default=0.05, help="Sinkhorn熵正则化系数")
    parser.add_argument("--sg_iters", type=int, default=3, help="Sinkhorn迭代次数")

    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU") 
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Min LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use AMP")
    
    parser.add_argument("--mask_weight", type=float, default=10.0)
    parser.add_argument("--heatmap_weight", type=float, default=1.0)
    parser.add_argument("--attr_weight", type=float, default=0.1) 
    
    parser.add_argument("--consistency_weight", type=float, default=0.1) 
    parser.add_argument("--consistency_warmup_epochs", type=int, default=50) 
    parser.add_argument("--density_map_weight", type=float, default=0.5) 

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from")

    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], help="Metrics to evaluate")

    return parser.parse_args()

# ==================================================================================================
# 2. Utils & Post-Processing
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

def hover_post_process(prob_map, hv_map, prob_thresh=0.45, marker_thresh=0.4, min_marker_size=10):
    mask = prob_map > prob_thresh
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    diff_v = np.gradient(v_map, axis=0) 
    diff_h = np.gradient(h_map, axis=1) 
    
    sobel_mag = np.sqrt(diff_v**2 + diff_h**2)
    
    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    
    marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    markers = skimage_label(marker_map).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)

# ==================================================================================================
# 3. Training Logic
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
    hv_losses = []
    attr_losses = []  
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input['image']
        labels = batched_input['label']

        if images.shape[-1] != args.image_size:
             images = F.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
             labels = F.interpolate(labels.float(), size=(args.image_size, args.image_size), mode='nearest').long()
        
        model_input = []
        organ_ids = batched_input.get('organ_id', None)
        
        attr_labels = batched_input.get('attr_labels', None)
        dynamic_text = batched_input.get('text_prompt', ["Cell nuclei"] * len(images))
        dynamic_attr_text = batched_input.get('attribute_text', ["Cell nuclei"] * len(images))
        
        for i in range(len(images)):
            curr_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val

            model_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size),
                'organ_id': curr_id,
                'attribute_text': dynamic_attr_text[i],
                'text_prompt': dynamic_text[i],
                'attr_labels': attr_labels[i] if attr_labels is not None else None, 
            })

        with autocast('cuda', enabled=args.use_amp):
            outputs = model(model_input, multimask_output=True)
            loss_batch = 0
            loss_m_accum = 0.0
            loss_h_accum = 0.0
            loss_hv_accum = 0.0
            loss_attr_accum = 0.0
            
            for i, out in enumerate(outputs):
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2: iou_preds = iou_preds.squeeze(0)
                best_idx = torch.argmax(iou_preds).item()
                pred_mask = out['masks'][best_idx, :, :] if out['masks'].ndim==3 else out['masks'][0, best_idx]
                pred_iou = iou_preds[best_idx]
                gt_mask = labels[i].squeeze(0).float()
                gt_mask = (gt_mask > 0).float()
                
                if pred_mask.shape != gt_mask.shape:
                      gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                # 1. Mask Loss
                loss_m, _ = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                # 2. Heatmap Loss
                pred_heatmap = out['heatmap_logits'] 
                with torch.no_grad():
                    target_mask = labels[i].float().unsqueeze(0)
                    gt_nuclei = F.interpolate(target_mask, size=pred_heatmap.shape[-2:], mode='nearest').squeeze(0)
                    gt_nuclei[gt_nuclei==255] = 0
                loss_h = point_guidance_loss(pred_heatmap, gt_nuclei.unsqueeze(0))

                # 3. HV Distance Map Loss
                loss_hv = torch.tensor(0.0, device=loss_m.device)
                pred_hv = out.get('hv_logits', None)
                if pred_hv is not None:
                    if pred_hv.dim() == 3: pred_hv = pred_hv.unsqueeze(0)
                    pred_hv = torch.tanh(pred_hv)

                    with torch.no_grad():
                        gt_hv_map_batch = batched_input.get("gt_hv_map", None)
                        if gt_hv_map_batch is not None:
                            gt_hv_full = gt_hv_map_batch[i].to(pred_hv.device)
                            if gt_hv_full.dim() == 3: gt_hv_full = gt_hv_full.unsqueeze(0)
                        else:
                            inst_batch = batched_input.get("label_inst", None)
                            if inst_batch is None: gt_hv_full = None
                            else:
                                inst_map = inst_batch[i].squeeze(0).long()
                                inst_map = inst_map.clone()
                                inst_map[inst_map == 255] = 0
                                gt_hv_full = generate_hv_map_from_inst(inst_map).unsqueeze(0)

                        if gt_hv_full is not None:
                            gt_hv = F.interpolate(gt_hv_full.float(), size=pred_hv.shape[-2:], mode='bilinear', align_corners=False)
                            inst_batch = batched_input.get("label_inst", None)
                            if inst_batch is not None:
                                focus_full = (inst_batch[i].squeeze(0) > 0).float().unsqueeze(0).unsqueeze(0)
                            else:
                                focus_full = (labels[i].squeeze(0) > 0).float().unsqueeze(0).unsqueeze(0)
                            focus = F.interpolate(focus_full, size=pred_hv.shape[-2:], mode='nearest').squeeze(1)

                    if pred_hv is not None and gt_hv_full is not None:
                        focus_exp = focus.unsqueeze(1)
                        mse_map = F.mse_loss(pred_hv.float(), gt_hv.float(), reduction='none')
                        loss_hv_mse = (mse_map * focus_exp).sum() / (focus_exp.sum() * pred_hv.shape[1] + 1e-8)
                        loss_hv_grad = msge_loss(gt_hv, pred_hv, focus)
                        loss_hv = loss_hv_mse + 2.0 * loss_hv_grad
                        
                # 4. PNuRL 属性预测损失与拓展损失
                loss_attr = out.get('pnurl_loss', torch.tensor(0.0, device=loss_m.device))
                loss_c = out.get('consistency_loss', torch.tensor(0.0, device=loss_m.device))
                loss_d = out.get('density_loss', torch.tensor(0.0, device=loss_m.device))
                
                # 终极 Loss 大一统融合
                loss_i = (
                    args.mask_weight * loss_m
                    + args.heatmap_weight * loss_h
                    + getattr(args, "hv_weight", 1.0) * loss_hv
                    + getattr(args, "attr_weight", 0.1) * loss_attr
                    + getattr(args, "consistency_weight", 0.1) * loss_c
                    + getattr(args, "density_map_weight", 0.5) * loss_d
                )
                
                loss_batch += loss_i
                loss_m_accum += loss_m.item()
                loss_h_accum += loss_h.item()
                loss_hv_accum += loss_hv.item()
                loss_attr_accum += loss_attr.item()
            
            final_loss = loss_batch / len(images)
            final_loss = final_loss / args.accumulation_steps

        is_accumulating = (batch_idx + 1) % args.accumulation_steps != 0 and (batch_idx + 1) != len(train_loader)

        # 🔥🔥🔥 终极全覆盖防死锁补丁：绑定计算图中所有可学习的参数，保证反向传播路径 100% 对称 🔥🔥🔥
        dummy_loss = 0.0
        for p in model.parameters():
            if p.requires_grad:
                dummy_loss = dummy_loss + p.sum() * 0.0
        final_loss = final_loss + dummy_loss

        # 正常回传，由于有 dummy_loss 兜底，绝不会出现未激活参数卡死的问题
        if scaler:
            scaler.scale(final_loss).backward()
        else:
            final_loss.backward()

        if not is_accumulating:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        current_loss_val = final_loss.item() * args.accumulation_steps
        mean_m = loss_m_accum / len(images)
        mean_h = loss_h_accum / len(images)
        mean_hv = loss_hv_accum / len(images)
        mean_a = loss_attr_accum / len(images)  

        losses.append(current_loss_val)
        mask_losses.append(mean_m)
        heatmap_losses.append(mean_h)
        hv_losses.append(mean_hv)
        attr_losses.append(mean_a)
        
        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_Total', current_loss_val, global_step)
            writer.add_scalar('Train/Loss_Mask', mean_m, global_step)
            writer.add_scalar('Train/Loss_Heatmap', mean_h, global_step)
            writer.add_scalar('Train/Loss_HV', mean_hv, global_step)
            writer.add_scalar('Train/Loss_Attr', mean_a, global_step) 
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
        
        if rank == 0:
            pbar.set_postfix(
                L=f"{current_loss_val:.3f}",
                M=f"{mean_m:.3f}",
                H=f"{mean_h:.3f}",
                A=f"{mean_a:.3f}" 
            )

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses), np.mean(attr_losses)

# ==================================================================================================
# 4. Validation Logic
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch, writer, rank):
    gc.collect() 
    torch.cuda.empty_cache()
    
    model.eval()
    val_results = {k: [] for k in args.metrics}
    visualize_done = False
    
    total_val_batches = len(val_loader)
    limit_batches = int(total_val_batches * 0.4)
    if limit_batches < 1: limit_batches = 1
    
    if rank == 0: pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Val (40%)", total=limit_batches)
    else: pbar = val_loader
    
    eval_model = model.module if hasattr(model, 'module') else model
    
    for batch, batched_input in enumerate(pbar):
        if batch >= limit_batches:
            break

        batched_input = to_device(batched_input, args.device)
        images = batched_input['image'] 
        inst_labels = batched_input.get('label_inst', batched_input['label']).cpu().numpy()
        organ_ids = batched_input.get('organ_id', None)

        if images.shape[-1] != args.image_size:
             images = F.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)

        for i in range(len(images)):
            curr_organ_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_organ_id = val.item() if isinstance(val, torch.Tensor) else val
            
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_input = [{
                    'image': images[i],
                    'original_size': (args.image_size, args.image_size),
                    'text_prompt': "Cell nuclei",
                    'organ_id': curr_organ_id,
                    'attribute_text': "Cell nuclei",
                }]
                out = eval_model(model_input, multimask_output=True)
                iou_preds = out[0]['iou_predictions']
                if iou_preds.ndim == 2:
                    iou_preds = iou_preds.squeeze(0)
                best_idx = torch.argmax(iou_preds).item()
                pred_logits = out[0]['masks'][0, best_idx]
                prob_map = torch.sigmoid(pred_logits)

                hv_logits = out[0].get('hv_logits', None)
                if hv_logits is not None:
                    hv_map = torch.tanh(hv_logits)
                    is_expanded = False
                    if hv_map.dim() == 3:
                        hv_map = hv_map.unsqueeze(0)
                        is_expanded = True
                    target_hw = prob_map.shape[-2:]
                    hv_map = F.interpolate(hv_map.float(), size=target_hw, mode='bilinear', align_corners=False)
                    if is_expanded: hv_map = hv_map.squeeze(0)
                    elif hv_map.dim() == 4 and hv_map.shape[0] == 1: hv_map = hv_map.squeeze(0)
                else:
                    target_hw = prob_map.shape[-2:]
                    hv_map = torch.zeros((2, target_hw[0], target_hw[1]), device=args.device)
            
            prob_np = prob_map.float().cpu().numpy()
            hv_np = hv_map.float().cpu().numpy()

            pred_mask = hover_post_process(prob_np, hv_np, prob_thresh=0.45, marker_thresh=0.4, min_marker_size=10)
            
            if pred_mask.max() == 0:
                pred_mask = skimage_label(prob_np > 0.5).astype(np.int32)

            gt = inst_labels[i]; gt = gt[0] if gt.ndim == 3 else gt
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
                pred_binary = (pred_mask > 0).astype(np.uint8)
                writer.add_image(f'Val_Viz/Pred', torch.tensor(pred_binary * 255).unsqueeze(0).to(torch.uint8), epoch)
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
        logger.info(f"🚀 [Start] MP-SAM Stable (Size: {args.image_size})")
        logger.info(f"   GPUs: {world_size}, Batch/GPU: {args.batch_size}, Resume: {args.resume if args.resume else 'No'}")
        writer = SummaryWriter(log_dir=run_log_dir, flush_secs=60)
    else: logger = None; writer = None

    try:
        train_dataset = UniversalDataset(
            data_root=args.data_path,
            knowledge_path=args.knowledge_path,
            image_size=args.image_size,
            crop_size=args.crop_size,
            mode='train',
            prompt_mode=args.prompt_mode,
        )
        val_dataset = UniversalDataset(
            data_root=args.data_path,
            knowledge_path=args.knowledge_path,
            image_size=args.image_size,
            crop_size=args.crop_size,
            mode='test',
            prompt_mode='generic',
        )
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=True) if world_size > 1 else None
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                                  num_workers=8, collate_fn=stack_dict_batched, pin_memory=True, sampler=train_sampler,
                                  persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), 
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

                key_mapping = {
                    "mask_decoder.output_upscaling.0.weight": "mask_decoder.asr_upscale_1.structure_upsample.0.weight",
                    "mask_decoder.output_upscaling.0.bias": "mask_decoder.asr_upscale_1.structure_upsample.0.bias",
                    "mask_decoder.output_upscaling.1.weight": "mask_decoder.asr_upscale_1.structure_upsample.1.weight",
                    "mask_decoder.output_upscaling.1.bias": "mask_decoder.asr_upscale_1.structure_upsample.1.bias",
                    "mask_decoder.output_upscaling.3.weight": "mask_decoder.asr_upscale_2.structure_upsample.0.weight",
                    "mask_decoder.output_upscaling.3.bias": "mask_decoder.asr_upscale_2.structure_upsample.0.bias",
                }
                mapped_state_dict = dict(state_dict)
                mapped_count = 0
                for old_key, new_key in key_mapping.items():
                    if old_key in state_dict:
                        mapped_state_dict[new_key] = state_dict[old_key]
                        mapped_count += 1

                vanilla_sam.load_state_dict(mapped_state_dict, strict=False)
                if rank == 0:
                    logger.info(f"✅ ASRBlock upscaling weights mapped: {mapped_count}/{len(key_mapping)}")
            except Exception as e:
                if rank == 0: logger.warning(f"⚠️ Checkpoint loading failed: {e}")
        
        model = TextSam(
            image_encoder=vanilla_sam.image_encoder,
            prompt_encoder=vanilla_sam.prompt_encoder,
            mask_decoder=vanilla_sam.mask_decoder,
            clip_model_name=args.clip_model,
            num_organs=args.num_organs,
            num_heads=args.num_heads,
            sg_epsilon=args.sg_epsilon,
            sg_iters=args.sg_iters,
            use_pnurl=args.use_pnurl,
            use_coop=args.use_coop,
            use_sgot=args.use_sgot,
            use_asr=args.use_asr,
        ).to(args.device)
        del vanilla_sam

        if world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # 🔥🔥🔥 终极修复：因为 dummy_loss 绑定了所有参数，DDP 不再需要耗时的 find_unused_parameters 扫描，直接关闭即可！
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        raw_model = model.module if world_size > 1 else model

        if args.encoder_adapter:
            for n, p in raw_model.image_encoder.named_parameters():
                if "Adapter" in n and "weight" in n: torch.nn.init.zeros_(p)

        params = [{'params': raw_model.mask_decoder.parameters(), 'lr': args.lr},
                  {'params': raw_model.prompt_generator.parameters(), 'lr': args.lr * 5}]
        if hasattr(raw_model, 'basic_hv_head'):
            params.append({'params': raw_model.basic_hv_head.parameters(), 'lr': args.lr})
        if getattr(raw_model, 'use_asr', False):
            cnn_lr = args.lr * 0.1
            params.append({'params': raw_model.cnn_stage0.parameters(), 'lr': cnn_lr})
            params.append({'params': raw_model.cnn_stage1.parameters(), 'lr': cnn_lr})
            params.append({'params': raw_model.cnn_stage2.parameters(), 'lr': cnn_lr})          
        if getattr(raw_model, 'use_coop_prompt', getattr(raw_model, 'use_coop', True)) and hasattr(raw_model, 'prompt_learner'):
            params.append({'params': raw_model.prompt_learner.parameters(), 'lr': args.lr})
        if getattr(raw_model, 'use_pnurl', True) and hasattr(raw_model, 'pnurl'):
            params.append({'params': raw_model.pnurl.parameters(), 'lr': args.lr})
        if getattr(raw_model, 'use_sgot', True) and hasattr(raw_model, 'sg_ot'):
            params.append({'params': raw_model.sg_ot.parameters(), 'lr': args.lr * 5})
            
        adapter_params = [p for n, p in raw_model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
        if adapter_params: params.append({'params': adapter_params, 'lr': args.lr})

        optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
        criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)
        scaler = GradScaler() if args.use_amp else None
        
        warmup_epochs = 15 # 🔥 [调优]：给复杂多模态组件更多的预热磨合时间！
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=getattr(args, "min_lr", 1e-6),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        best_aji = 0.0; best_dice = 0.0
        
        if args.resume and os.path.exists(args.resume):
            if rank == 0: logger.info(f"🔄 Resuming checkpoint from: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
                
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                state_dict = resize_pos_embed(state_dict, raw_model.state_dict())
                raw_model.load_state_dict(state_dict, strict=False)
                
                if 'model' in checkpoint:
                    if 'optimizer' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    if 'scheduler' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    if 'scaler' in checkpoint and scaler is not None and checkpoint['scaler'] is not None:
                        scaler.load_state_dict(checkpoint['scaler'])
                    if 'epoch' in checkpoint:
                        args.start_epoch = checkpoint['epoch'] + 1  
                    if 'best_aji' in checkpoint:
                        best_aji = checkpoint['best_aji']
                    if 'best_dice' in checkpoint:
                        best_dice = checkpoint['best_dice']
                        
                if rank == 0: logger.info(f"✅ Resume success! Auto-resuming from Epoch {args.start_epoch} (Best AJI was {best_aji:.4f}).")
            except Exception as e:
                if rank == 0: logger.warning(f"⚠️ Resume failed: {e}")

        for epoch in range(args.start_epoch, args.epochs):
            if train_sampler: train_sampler.set_epoch(epoch)
            if val_sampler: val_sampler.set_epoch(epoch)
            
            loss, m_loss, h_loss, a_loss = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, writer, rank)
            val_res = validate_one_epoch(args, model, val_loader, epoch, writer, rank)
            
            if rank == 0:
                dice = val_res.get('dice', 0.0); aji = val_res.get('mAJI', 0.0); pq = val_res.get('mPQ', 0.0)
                logger.info(f"Ep {epoch+1}/{args.epochs} | Loss: {loss:.4f} (M:{m_loss:.3f}, H:{h_loss:.3f}, A:{a_loss:.3f}) | Dice: {dice:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}")
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'best_aji': best_aji,
                    'best_dice': best_dice
                }
                
                latest_model_path = os.path.join(args.work_dir, "models", args.run_name, "latest_model.pth")
                torch.save(checkpoint_dict, latest_model_path)
                
                if aji > best_aji:
                    best_aji = aji; best_dice = max(best_dice, dice)
                    checkpoint_dict['best_aji'] = best_aji
                    checkpoint_dict['best_dice'] = best_dice
                    best_model_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
                    torch.save(checkpoint_dict, best_model_path)
                    logger.info(f"New Best AJI! ({best_aji:.4f}) -> Model Saved")
                    
            if dist.is_initialized():
                dist.barrier()
            scheduler.step()

        if rank == 0:
            logger.info(f"🏁 Training Finished. Best AJI: {best_aji:.4f}")
            if writer is not None: writer.close()
            
    except Exception as e:
        if rank == 0 and logger is not None:
            logger.error("\n" + "="*50)
            logger.error(f"❌ 训练发生致命错误中止 (Fatal Error): {str(e)}")
            logger.error(f"🔍 完整错误堆栈 (Traceback):\n{traceback.format_exc()}")
            logger.error("="*50 + "\n")
        raise e  
        
    finally:
        if rank == 0 and writer is not None: 
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    main(args)