import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import datetime
import logging
from collections import defaultdict

# === Imports ===
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from prompt_generator import point_guidance_loss 
from metrics import SegMetrics

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
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="save model interval (epochs)")
    parser.add_argument("--use_amp", action='store_true', help="use amp")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter in image encoder")
    
    args = parser.parse_args()
    return args

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler=None):
    model.train()
    pbar = tqdm(train_loader, desc=f"Train Ep {epoch}")
    
    losses = []
    mask_losses = []
    heatmap_losses = []
    
    for batch, batched_input in enumerate(pbar):
        # 1. Prepare Data
        images = batched_input['image'].to(args.device)     # [B, 3, 256, 256]
        labels = batched_input['label'].to(args.device)     # [B, 1, 256, 256]
        
        if labels.ndim == 3:
            labels = labels.unsqueeze(1) 

        # 2. Format Input
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size)
            })
            
        optimizer.zero_grad()

        # 3. Forward & Loss
        def compute_loss(outputs):
            loss_batch_sum = 0
            loss_m_val_sum = 0
            loss_h_val_sum = 0
            
            for i, out in enumerate(outputs):
                # A. Segmentation Mask Loss
                if out['masks'].ndim == 4:
                    pred_mask = out['masks'][0, 0, :, :] 
                else:
                    pred_mask = out['masks'][0, :, :]
                
                gt_mask = labels[i].float()
                if pred_mask.shape != gt_mask.shape[-2:]:
                    gt_mask = F.interpolate(gt_mask.unsqueeze(0), size=pred_mask.shape[-2:], mode='nearest').squeeze(0)

                pred_iou = out['iou_predictions'][0]
                
                loss_m = criterion(pred_mask.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0), pred_iou.unsqueeze(0))
                
                # B. Heatmap Loss
                pred_heatmap = out['heatmap_logits'] 
                current_feat_size = pred_heatmap.shape[-2:] 
                
                with torch.no_grad():
                    gt_nuclei = F.interpolate(labels[i].float().unsqueeze(0), size=current_feat_size, mode='nearest').squeeze(0)
                    gt_background = 1.0 - gt_nuclei

                loss_h_pos = point_guidance_loss(pred_heatmap[0:1].unsqueeze(0), gt_nuclei.unsqueeze(0))
                loss_h_neg = point_guidance_loss(pred_heatmap[1:2].unsqueeze(0), gt_background.unsqueeze(0))
                loss_h = loss_h_pos + loss_h_neg
                
                # Total Loss
                loss_i = loss_m + 0.5 * loss_h
                
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

        losses.append(loss_batch.item())
        mask_losses.append(loss_m_val / len(images))
        heatmap_losses.append(loss_h_val / len(images))
        
        pbar.set_postfix(L=f"{loss_batch.item():.3f}", M=f"{(loss_m_val/len(images)):.3f}", H=f"{(loss_h_val/len(images)):.3f}")

    return np.mean(losses), np.mean(mask_losses), np.mean(heatmap_losses)

@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch):
    model.eval()
    val_metrics = defaultdict(list)
    
    pbar = tqdm(val_loader, desc=f"Val Ep {epoch}")
    
    for batch, batched_input in enumerate(pbar):
        images = batched_input['image'].to(args.device)
        labels = batched_input['label'].detach().cpu().numpy() # GT
        
        sam_input = []
        for i in range(len(images)):
            sam_input.append({
                'image': images[i],
                'original_size': (args.image_size, args.image_size)
            })
            
        outputs = model(sam_input, multimask_output=True)
        
        pred_list = []
        gt_list = []
        
        for i, out in enumerate(outputs):
            if out['masks'].ndim == 4:
                logit = out['masks'][0, 0, :, :]
            else:
                logit = out['masks'][0, :, :]
            
            prob = torch.sigmoid(logit).cpu().numpy()
            pred_mask = (prob > 0.5).astype(np.uint8)
            pred_list.append(pred_mask)
            
            gt = labels[i]
            if gt.ndim == 3: gt = gt[0]
            gt_list.append(gt)

        pred_batch = np.array(pred_list) 
        gt_batch = np.array(gt_list)     
        
        try:
            batch_res = SegMetrics(pred_batch, gt_batch, args.metrics)
            for k, v in batch_res.items():
                val_metrics[k].append(v)
        except Exception as e:
            pass 
            
    avg_results = {k: np.mean(v) for k, v in val_metrics.items() if len(v) > 0}
    return avg_results

def main(args):
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    logger = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    logger.info(f"Args: {args}")

    # 1. Dataset (Full Data)
    train_dataset = TrainingDataset(
        args.data_path, 
        image_size=args.image_size, 
        mode='train', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=True, 
        attribute_info_path=args.attribute_info_path 
    )
    
    val_dataset = TrainingDataset(
        args.data_path, 
        image_size=args.image_size, 
        mode='test', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=True, 
        attribute_info_path=args.attribute_info_path 
    )
    
    logger.info(f"Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=stack_dict_batched, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=stack_dict_batched, pin_memory=True)

    # 2. Model
    logger.info("Building TextSam...")
    vanilla_sam = sam_model_registry[args.model_type](args)
    
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        try:
            ckpt = torch.load(args.sam_checkpoint, map_location=args.device, weights_only=False)
        except TypeError:
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

    # 3. Optimizer
    params_to_optimize = [
        {'params': model.mask_decoder.parameters(), 'lr': args.lr},
        {'params': model.prompt_generator.parameters(), 'lr': args.lr * 10}
    ]
    if args.encoder_adapter:
        params_to_optimize.append({'params': filter(lambda p: p.requires_grad, model.image_encoder.parameters()), 'lr': args.lr})

    optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)
    criterion = FocalDiceloss_IoULoss()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 4. Loop
    best_aji = 0.0
    start_epoch = 0
    
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        except:
            ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    logger.info("Start Training...")
    for epoch in range(start_epoch, args.epochs):
        avg_loss, avg_msk, avg_ht = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler)
        
        # Validation
        val_metrics = validate_one_epoch(args, model, val_loader, epoch)
        
        d = val_metrics.get('dice', 0)
        aji = val_metrics.get('mAJI', 0)
        pq = val_metrics.get('mPQ', 0)
        
        logger.info(f"Ep {epoch+1} | Loss: {avg_loss:.4f} (M:{avg_msk:.3f} H:{avg_ht:.3f}) | Val Dice: {d:.4f} | AJI: {aji:.4f} | PQ: {pq:.4f}")
        
        # Save Best
        if aji > best_aji:
            best_aji = aji
            save_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_aji': best_aji,
                'metrics': val_metrics
            }, save_path)
            logger.info(f"⭐ Saved Best Model (AJI: {best_aji:.4f})")
            
        if (epoch + 1) % args.save_interval == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch+1}, 
                       os.path.join(args.work_dir, "models", args.run_name, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    args = parse_args()
    main(args)