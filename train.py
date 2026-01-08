import argparse
import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Native AMP
try:
    from torch.amp import autocast, GradScaler 
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger

try:
    from prompt_generator import point_guidance_loss
except ImportError:
    def point_guidance_loss(pred_heatmap, gt_nuclei):
        return F.mse_loss(torch.sigmoid(pred_heatmap), gt_nuclei)

from metrics import SegMetrics

def parse_args():
    parser = argparse.ArgumentParser()
    
    # === 基础配置 ===
    parser.add_argument("--work_dir", type=str, default="workdir", help="Work directory for logs and checkpoints")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-coop", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda')
    
    # === 数据相关 ===
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="Root path to data")
    # 🔥 [修改] 分离两个关键库的路径，不再硬编码
    parser.add_argument("--stats_path", type=str, default="data/MoNuSeg_SA1B/dataset_stats.json", help="Path to statistical thresholds (dataset_stats.json)")
    parser.add_argument("--prompts_path", type=str, default="data/MoNuSeg_SA1B/specific_prompts.json", help="Path to specific prompts library (specific_prompts.json)")
    
    parser.add_argument("--image_size", type=int, default=1024, help="Input image size")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size during training (Keep 1024 for consistency)")
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks")
    
    # === 模型配置 ===
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM model type (vit_b, vit_l, vit_h)")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="Path to pretrained SAM checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model type")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="Enable adapter in image encoder")
    
    # === 优化器与训练 ===
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate") 
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="LR scheduler")
    parser.add_argument("--use_amp", action='store_true', default=False, help="Use Automatic Mixed Precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # === 🔥 [新增] Loss 权重配置 ===
    parser.add_argument("--mask_weight", type=float, default=2.0, help="Weight for mask loss (Focal+Dice+IoU)")
    parser.add_argument("--heatmap_weight", type=float, default=1.0, help="Weight for heatmap guidance loss (Default 0.0 to avoid conflict)")
    
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ', 'mDQ', 'mSQ'], help="Evaluation metrics")
    
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key in ['image', 'label']:
                device_input[key] = value.float().to(device)
            elif isinstance(value, (list, torch.Size)):
                device_input[key] = value
            else:
                try:
                    device_input[key] = value.to(device)
                except:
                    device_input[key] = value
        else:
            device_input[key] = value
    return device_input

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        labels = batched_input['label'] 
        
        optimizer.zero_grad()

        # === Data Formatting ===
        model_input = []
        images = batched_input['image']
        
        if 'text_prompt' in batched_input:
            texts = batched_input['text_prompt']
        else:
            texts = ["Cell nuclei" for _ in range(len(images))]

        for i in range(len(images)):
            sample = {
                'image': images[i],
                'text_prompt': texts[i],
                'original_size': (args.image_size, args.image_size)
            }
            if 'original_size' in batched_input:
                sample['original_size'] = batched_input['original_size'][i]
            model_input.append(sample)

        with autocast('cuda', enabled=args.use_amp):
            outputs = model(model_input, multimask_output=True)
            
            loss_batch = 0
            loss_m_val = 0
            loss_h_val = 0
            
            for i, out in enumerate(outputs):
                iou_preds = out['iou_predictions']
                if iou_preds.ndim == 2:
                    iou_preds = iou_preds.squeeze(0)
                
                best_idx = torch.argmax(iou_preds).item()
                
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, best_idx, :, :] 
                else:
                    pred_mask = out['masks'][best_idx, :, :]
                
                pred_iou = iou_preds[best_idx]
                gt_mask = labels[i].squeeze(0).float()
                
                # Resize GT if needed
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode='nearest').squeeze()

                # === 1. Mask Loss ===
                loss_m, loss_dict = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0), pred_iou.unsqueeze(0))
                
                # === 2. Heatmap Loss ===
                pred_heatmap = out['heatmap_logits'] 
                current_feat_size = pred_heatmap.shape[-2:] 
                
                with torch.no_grad():
                    # Resize GT to feature map size for heatmap supervision
                    gt_nuclei = F.interpolate(labels[i].float().unsqueeze(0), size=current_feat_size, mode='nearest').squeeze(0)
                    valid_mask = (gt_nuclei != 255).float()
                    gt_nuclei[gt_nuclei == 255] = 0 
                    gt_background = 1.0 - gt_nuclei

                loss_h_pos = point_guidance_loss(pred_heatmap[0:1].unsqueeze(0), gt_nuclei.unsqueeze(0))
                loss_h_neg = point_guidance_loss(pred_heatmap[1:2].unsqueeze(0), gt_background.unsqueeze(0))
                loss_h = loss_h_pos + loss_h_neg
                
                # 🔥 [修改] 使用参数控制权重
                loss_i = args.mask_weight * loss_m + args.heatmap_weight * loss_h
                
                loss_batch += loss_i
                loss_m_val += loss_m.item()
                loss_h_val += loss_h.item()
            
            final_loss = loss_batch / len(labels)

        if scaler is not None:
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            final_loss.backward()
            optimizer.step()

        losses.append(final_loss.item())
        mask_losses.append(loss_m_val / len(labels))
        heatmap_losses.append(loss_h_val / len(labels))
        
        pbar.set_postfix(L=f"{final_loss.item():.3f}", Prompt=texts[0][:15])

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)

@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    
    # Positive Metrics (Generic Task)
    val_results = {k: [] for k in args.metrics}
    obj_dices = []
    
    # Negative Metrics (Rejection Task)
    neg_responses = [] 
    
    pbar = tqdm(val_loader, desc=f"Val Ep {epoch+1}")
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        labels = batched_input['label'].cpu().numpy() 
        images = batched_input['image']
        
        # 🟢 Positive Validation
        model_input_pos = []
        texts_pos = ["Cell nuclei" for _ in range(len(images))]
        
        for i in range(len(images)):
            model_input_pos.append({
                'image': images[i],
                'text_prompt': texts_pos[i],
                'original_size': (args.image_size, args.image_size)
            })
            
        outputs_pos = model(model_input_pos, multimask_output=True)
        
        for i, out in enumerate(outputs_pos):
            best_idx = torch.argmax(out['iou_predictions']).item()
            pred_mask = (torch.sigmoid(out['masks'][0, best_idx]).cpu().numpy() > 0.5).astype(np.uint8)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            
            res = SegMetrics(pred_mask, gt, args.metrics)
            if gt.sum() > 0: obj_dices.append(res['dice'])
            
            for k, v in res.items():
                if k in val_results: val_results[k].append(v)

        # 🔴 Negative Validation
        model_input_neg = []
        texts_neg = ["A photo of a cat" for _ in range(len(images))] 
        
        for i in range(len(images)):
            model_input_neg.append({
                'image': images[i],
                'text_prompt': texts_neg[i], 
                'original_size': (args.image_size, args.image_size)
            })
            
        outputs_neg = model(model_input_neg, multimask_output=True)
        
        for i, out in enumerate(outputs_neg):
            best_idx = torch.argmax(out['iou_predictions']).item()
            pred_prob = torch.sigmoid(out['masks'][0, best_idx])
            pred_mask_neg = (pred_prob > 0.5).float()
            neg_responses.append(pred_mask_neg.mean().item())

    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    avg_results['obj_dice'] = np.mean(obj_dices) if len(obj_dices) > 0 else 0.0
    avg_neg_response = np.mean(neg_responses) if len(neg_responses) > 0 else 0.0
        
    return avg_results, avg_neg_response

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

def main(args):
    setup_seed(args.seed)
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    logger.info(f"Args: {args}")

    train_root = os.path.join(args.data_path, "train")
    val_root = os.path.join(args.data_path, "test")
    
    if not os.path.exists(train_root): train_root = args.data_path
    if not os.path.exists(val_root): val_root = args.data_path

    # === Dataset ===
    # 🔥 [修改] 传递参数 stats_path 和 prompts_path
    train_dataset = TrainingDataset(
        train_root, 
        image_size=args.image_size,
        crop_size=args.crop_size, 
        mode='train', 
        mask_num=args.mask_num, 
        requires_name=True,
        stats_path=args.stats_path,
        prompts_path=args.prompts_path
    )
    
    val_dataset = TrainingDataset(
        val_root, 
        image_size=args.image_size,
        crop_size=args.crop_size, 
        mode='test', 
        mask_num=args.mask_num, 
        requires_name=True,
        stats_path=args.stats_path,
        prompts_path=args.prompts_path
    )
    logger.info(f"Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=stack_dict_batched)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=stack_dict_batched)

    # === Model ===
    logger.info("Building TextSam...")
    args.checkpoint = args.sam_checkpoint
    vanilla_sam = sam_model_registry[args.model_type](args)
    
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location=args.device, weights_only=False)
        except:
            ckpt = torch.load(args.sam_checkpoint, map_location=args.device)
        state_dict = ckpt.get("model", ckpt)
        print("🔄 Interpolating position embeddings...")
        state_dict = resize_pos_embed(state_dict, vanilla_sam.state_dict())
        vanilla_sam.load_state_dict(state_dict, strict=False)
        logger.info("✅ Loaded & Resized SAM checkpoint.")

    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name=args.clip_model, # 🔥 [修改] 使用参数
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    del vanilla_sam

    # === Reset Adapters ===
    print("\n🧹 Enforcing Zero Initialization for Adapters...")
    for name, param in model.image_encoder.named_parameters():
        if "Adapter" in name and ("spatial.2.weight" in name or "channel.2.weight" in name):
            torch.nn.init.zeros_(param)
    print("✅ Adapters Reset.")

    # --- Optimizer ---
    params_to_optimize = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 5} 
    ]
    
    # 🔥 [修改] 自动检测 PromptLearner (CoOp) 是否存在并加入优化器
    if hasattr(model, 'prompt_learner'):
        print(f"✨ Adding CoOp PromptLearner to optimizer (lr={args.lr})")
        params_to_optimize.append({'params': model.prompt_learner.parameters(), 'lr': args.lr})
        
    if args.encoder_adapter:
        adapter_params = [p for n, p in model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
        params_to_optimize.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params_to_optimize, weight_decay=args.weight_decay)
    criterion = FocalDiceloss_IoULoss(ignore_index=255)
    scaler = GradScaler() if args.use_amp else None

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_dice = 0.0
    
    logger.info("Start Dynamic Text-Guided Training...")
    
    for epoch in range(args.epochs):
        avg_loss, avg_msk, avg_ht = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        val_metrics, val_neg_resp = validate_one_epoch(args, model, val_loader, epoch)
        
        dice_score = val_metrics.get('dice', 0.0)
        
        logger.info(
            f"Ep {epoch+1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Dice: {dice_score:.4f} | "
            f"IoU: {val_metrics.get('iou', 0.0):.4f} | "
            f"AJI: {val_metrics.get('mAJI', 0.0):.4f} | "
            f"NegResp: {val_neg_resp:.4f}"
        )
        
        if scheduler is not None:
            scheduler.step()
            
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_model.pth"))
            logger.info(f"⭐ Saved Best Model (Dice: {best_dice:.4f})")
            
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    args = parse_args()
    main(args)