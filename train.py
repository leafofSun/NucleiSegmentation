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
# 假设 TextSam 类定义在 segment_anything.modeling.sam 中
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
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    # 建议修改 run_name 以区分之前的实验
    parser.add_argument("--run_name", type=str, default="text-guided-sam-rich", help="run model name")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B", help="train data path")
    parser.add_argument("--prompt_path", type=str, default="data/MoNuSeg_SA1B/attribute_info_train.json", help="Path to the prompt JSON file")
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate") # 建议 1e-4
    parser.add_argument("--min_lr", type=float, default=1e-6, help="min learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    # 建议使用原始权重重新开始训练
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--use_amp", action='store_true', default=False, help="use amp")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="lr scheduler")
    
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
                    # 对于字符串列表等无法 .to(device) 的数据，保持原样
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

        # === 🔥 [核心修改] 注入文本 Prompt ===
        model_input = []
        images = batched_input['image']
        
        # 获取 batch 里的文本列表
        # 如果 DataLoader 没有返回 text_prompt，则使用默认值
        if 'text_prompt' in batched_input:
            texts = batched_input['text_prompt']
        else:
            texts = ["Cell nuclei" for _ in range(len(images))]

        for i in range(len(images)):
            sample = {
                'image': images[i],
                'text_prompt': texts[i]  # 显式注入文本!
            }
            if 'original_size' in batched_input:
                sample['original_size'] = batched_input['original_size'][i]
            model_input.append(sample)

        with autocast('cuda', enabled=args.use_amp):
            # 传入 model_input，TextSam 内部会处理 text_prompt
            outputs = model(model_input, multimask_output=True)
            
            # --- Loss 计算 ---
            loss_batch = 0
            loss_m_val = 0
            loss_h_val = 0
            
            for i, out in enumerate(outputs):
                # 1. Mask Loss
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, 0, :, :] 
                else:
                    pred_mask = out['masks'][0, :, :]
                
                gt_mask = labels[i].float()
                if pred_mask.shape != gt_mask.shape[-2:]:
                    gt_mask = F.interpolate(gt_mask.unsqueeze(0), size=pred_mask.shape[-2:], mode='nearest').squeeze(0)

                pred_iou = out['iou_predictions'][0]
                
                # criterion 要求 [B, C, H, W]
                loss_m = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0), pred_iou.unsqueeze(0))
                
                # 2. Heatmap Loss
                pred_heatmap = out['heatmap_logits'] 
                current_feat_size = pred_heatmap.shape[-2:] 
                
                with torch.no_grad():
                    gt_nuclei = F.interpolate(labels[i].float().unsqueeze(0), size=current_feat_size, mode='nearest').squeeze(0)
                    gt_background = 1.0 - gt_nuclei

                loss_h_pos = point_guidance_loss(pred_heatmap[0:1].unsqueeze(0), gt_nuclei.unsqueeze(0))
                loss_h_neg = point_guidance_loss(pred_heatmap[1:2].unsqueeze(0), gt_background.unsqueeze(0))
                loss_h = loss_h_pos + loss_h_neg
                
                # === 权重调整 ===
                # 强迫 Decoder 输出 (2.0)，辅助 Heatmap (0.1)
                loss_i = 2.0 * loss_m + 0.1 * loss_h
                
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
        
        pbar.set_postfix(L=f"{final_loss.item():.3f}", M=f"{np.mean(mask_losses):.3f}", H=f"{np.mean(heatmap_losses):.3f}")

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)

@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    val_results = {k: [] for k in args.metrics}
    obj_dices = [] # 记录有前景物体的 Dice
    
    pbar = tqdm(val_loader, desc=f"Val Ep {epoch+1}")
    
    for batch, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        labels = batched_input['label'].cpu().numpy()
        
        # === 同样需要注入文本 ===
        model_input = []
        images = batched_input['image']
        
        if 'text_prompt' in batched_input:
            texts = batched_input['text_prompt']
        else:
            texts = ["Cell nuclei" for _ in range(len(images))]
            
        for i in range(len(images)):
            sample = {
                'image': images[i],
                'text_prompt': texts[i]
            }
            if 'original_size' in batched_input:
                sample['original_size'] = batched_input['original_size'][i]
            model_input.append(sample)
            
        outputs = model(model_input, multimask_output=True)
        
        for i, out in enumerate(outputs):
            if out['masks'].ndim == 4:
                logit = out['masks'][0, 0, :, :]
            else:
                logit = out['masks'][0, :, :]
            
            prob = torch.sigmoid(logit).cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.uint8)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            
            # 计算指标
            res = SegMetrics(pred_mask, gt, args.metrics)
            
            # 如果 GT 有东西，记录 Obj Dice
            if gt.sum() > 0:
                obj_dices.append(res['dice'])
            
            for k, v in res.items():
                if k in val_results:
                    val_results[k].append(v)
            
    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    
    if len(obj_dices) > 0:
        avg_results['obj_dice'] = np.mean(obj_dices)
    else:
        avg_results['obj_dice'] = 0.0
        
    return avg_results

def main(args):
    setup_seed(args.seed)
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    logger.info(f"Args: {args}")

    # === 🔥 [关键修改] 显式分离训练集和测试集路径 ===
    # 假设您的目录结构是 data/MoNuSeg_SA1B/train 和 data/MoNuSeg_SA1B/test
    train_root = os.path.join(args.data_path, "train")
    val_root = os.path.join(args.data_path, "test")
    
    if not os.path.exists(train_root):
        logger.warning(f"⚠️ Train dir {train_root} not found! Fallback to {args.data_path}")
        train_root = args.data_path
    if not os.path.exists(val_root):
        logger.warning(f"⚠️ Test dir {val_root} not found! Fallback to {args.data_path}")
        val_root = args.data_path

    # === Dataset ===
    # 1. 训练集: 读取 Rich Prompt (JSON)
    train_dataset = TrainingDataset(
        train_root,  # <--- 指向 train 文件夹
        image_size=args.image_size, 
        mode='train', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=True,
        prompt_path=args.prompt_path # 这里加载 attribute_info_train.json
    )
    
    # 2. 验证集: 指向 test 文件夹 (严格隔离)
    # 注意: 这里的 prompt_path 虽然传了，但因为 JSON 里只有训练集文件名，
    # DataLoader 查不到测试图的 Key，会自动回退到 "Cell nuclei"，这正是我们想要的 Zero-shot 验证！
    val_dataset = TrainingDataset(
        val_root,    # <--- 指向 test 文件夹
        image_size=args.image_size, 
        mode='test', # 验证模式 (无随机增强)
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=True,
        prompt_path=args.prompt_path 
    )
    logger.info(f"Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)}")

    # === DataLoader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=stack_dict_batched
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=stack_dict_batched
    )
    # === Model ===
    logger.info("Building TextSam...")
    vanilla_sam = sam_model_registry[args.model_type](args)
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location=args.device, weights_only=False)
        except:
            ckpt = torch.load(args.sam_checkpoint, map_location=args.device)
        state_dict = ckpt.get("model", ckpt)
        vanilla_sam.load_state_dict(state_dict, strict=False)
        logger.info("Loaded vanilla SAM checkpoint.")

    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    del vanilla_sam

    # === Adapter Zero Init ===
    print("\n🧹 [Force Reset] Enforcing Zero Initialization for Adapters...")
    reset_cnt = 0
    for name, param in model.image_encoder.named_parameters():
        if "Adapter" in name and ("spatial.2.weight" in name or "channel.2.weight" in name):
            torch.nn.init.zeros_(param)
            reset_cnt += 1
    print(f"✅ Reset {reset_cnt} Adapter weights.")

    # --- Optimizer & Scheduler ---
    params_to_optimize = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 5} 
    ]
    if args.encoder_adapter:
        adapter_params = [p for n, p in model.image_encoder.named_parameters() if "Adapter" in n and p.requires_grad]
        params_to_optimize.append({'params': adapter_params, 'lr': args.lr})

    optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)
    criterion = FocalDiceloss_IoULoss()
    scaler = GradScaler() if args.use_amp else None

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_dice = 0.0
    
    logger.info("Start Text-Guided Training (Rich Prompts)...")
    for epoch in range(args.epochs):
        avg_loss, avg_msk, avg_ht = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        val_metrics = validate_one_epoch(args, model, val_loader, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 安全获取指标
        m_aji = val_metrics.get('mAJI', 0.0)
        m_pq = val_metrics.get('mPQ', 0.0)
        obj_dice = val_metrics.get('obj_dice', 0.0)
        
        logger.info(f"Ep {epoch+1} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f} (M:{avg_msk:.3f} H:{avg_ht:.3f}) | Val Dice: {val_metrics['dice']:.4f} | Obj Dice: {obj_dice:.4f} | AJI: {m_aji:.4f} | PQ: {m_pq:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, "best_model.pth"))
            logger.info(f"⭐ Saved Best Model (Dice: {best_dice:.4f})")
            
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    args = parse_args()
    main(args)