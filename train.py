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
from torch.cuda.amp import autocast, GradScaler  # 使用原生混合精度

# 假设这些模块在您的本地路径中可用
from segment_anything import sam_model_registry
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default="cosine", help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", action='store_true', default=True, help="output multimask")
    parser.add_argument("--encoder_adapter", action='store_true', default=True, help="use adapter")
    parser.add_argument("--use_amp", action='store_true', default=False, help="use amp")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

def setup_seed(seed):
    """设置随机种子以保证可复现性"""
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
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def set_grad_state(model, mode='adapter'):
    """
    根据训练阶段控制梯度的冻结与解冻
    mode='adapter': 只训练 ImageEncoder 中的 Adapter
    mode='decoder': 冻结 ImageEncoder，训练 PromptEncoder 和 MaskDecoder
    """
    if mode == 'adapter':
        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        # 此时通常不需要调整 decoder，视具体算法逻辑而定，保持原逻辑不显式冻结decoder
    
    elif mode == 'decoder':
        # 冻结整个 ImageEncoder
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

def run_forward_pass(args, model, batched_input, image_embeddings):
    """封装 Forward Pass 逻辑"""
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    # Prompt Encoder
    with torch.set_grad_enabled(True): # 确保 Prompt Encoder 计算图被追踪
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    # Mask Decoder
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        iou_predictions = max_values.unsqueeze(1)
        low_res_masks = low_res_masks[torch.arange(low_res_masks.size(0)), max_indexs].unsqueeze(1)

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    train_loader_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)

    for batch, batched_input in enumerate(train_loader_bar):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        labels = batched_input["label"]

        # ====================================================
        # Step 1: Initial Prompt (Box or Point) - Training Adapter
        # ====================================================
        if random.random() > 0.5:
            batched_input["point_coords"] = None
            prompt_type = "boxes"
        else:
            batched_input["boxes"] = None
            prompt_type = "point"

        # 设置梯度：只训练 Adapter
        set_grad_state(model, mode='adapter')

        optimizer.zero_grad()
        
        # 准备 Image Embeddings (支持 Mixed Precision)
        with autocast(enabled=args.use_amp):
            image_embeddings_orig = model.image_encoder(batched_input["image"])
            
            # Repeat embeddings for mask_num logic
            B, _, _, _ = image_embeddings_orig.shape
            image_embeddings = image_embeddings_orig.repeat_interleave(args.mask_num, dim=0)
            
            masks, low_res_masks, iou_predictions = run_forward_pass(args, model, batched_input, image_embeddings)
            loss = criterion(masks, labels, iou_predictions)

        # Backward Step 1
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad() # 清空梯度准备 Step 2

        # Log metrics occasionally
        if (batch + 1) % 50 == 0:
            print(f'\nEpoch: {epoch+1}, Batch: {batch+1}, First {prompt_type} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # ====================================================
        # Step 2: Iterative Refinement - Training Decoder
        # ====================================================
        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        # 这里的 image_embeddings 不需要梯度，因为它是在 Step 1 计算的，现在我们要冻结 Encoder
        image_embeddings = image_embeddings.detach()

        # 设置梯度：冻结 Encoder，训练其他
        set_grad_state(model, mode='decoder')

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        
        for iter_idx in range(args.iter_point):
            # 在特定迭代次数重置 Prompt，增加鲁棒性
            if iter_idx == init_mask_num or iter_idx == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            optimizer.zero_grad()
            
            with autocast(enabled=args.use_amp):
                masks, low_res_masks, iou_predictions = run_forward_pass(args, model, batched_input, image_embeddings)
                loss = criterion(masks, labels, iou_predictions)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 准备下一次迭代的 Prompt
            if iter_idx != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

            # Log metrics occasionally
            if (batch + 1) % 50 == 0:
                if iter_idx == init_mask_num or iter_idx == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # ====================================================
        # Logging & Saving
        # ====================================================
        if (batch + 1) % 200 == 0:
            print(f"Epoch:{epoch+1}, Iteration:{batch+1}, Loss:{loss.item():.6f}")
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, save_path)

        train_losses.append(loss.item())
        train_loader_bar.set_postfix(train_loss=f"{loss.item():.4f}")

        # Metrics accumulation
        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics

def main(args):
    setup_seed(args.seed)
    
    # 目录准备
    os.makedirs(os.path.join(args.work_dir, "models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)
    
    # 日志记录器
    log_path = os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log")
    logger = get_logger(log_path)

    # 模型加载
    print(f"******* Creating model: {args.model_type}")
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    # 混合精度 Scaler (Native PyTorch)
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("******* Mixed precision with torch.cuda.amp")

    # 学习率调度器
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
        print('******* Use MultiStepLR')

    # 断点续训
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"******* Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(f"******* Resume file not found: {args.resume}")

    # 数据加载
    train_dataset = TrainingDataset(
        args.data_path, 
        image_size=args.image_size, 
        mode='train', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=False
    )
    # 启用 pin_memory 以加速 Host 到 Device 的传输
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    print(f'******* Train data size: {len(train_dataset)}')

    best_loss = float('inf')
    num_batches = len(train_loader)

    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_losses, train_iter_metrics = train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion, scaler
        )

        if scheduler is not None:
            scheduler.step()

        # 计算 Epoch 平均指标
        avg_metrics = [metric / num_batches for metric in train_iter_metrics]
        metrics_dict = {args.metrics[i]: f'{avg_metrics[i]:.4f}' for i in range(len(avg_metrics))}
        average_loss = np.mean(train_losses)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch: {epoch + 1}, LR: {current_lr:.6f}, Train Loss: {average_loss:.4f}, Metrics: {metrics_dict}")

        # 保存最佳模型
        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam_best.pth")
            # 保存时统一转为 float32，避免半精度兼容性问题
            state = {
                'model': model.float().state_dict() if args.use_amp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }
            torch.save(state, save_path)
            # 如果是 AMP 模式，保存完需要切回 half 吗？SAM 通常内部处理得很好，但为了保险起见，
            # 这里不需要显式切回 model.half()，因为 autocast 会自动处理 forward 过程。
            # 只有当模型权重本身被手动 cast 成 half 时才需要。PyTorch Native AMP 保持权重为 float32。

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)