from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
from torch.cuda import amp
import random
import torch.serialization
torch.serialization.add_safe_globals([torch.optim.Adam])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data/data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['mDice', 'mAJI', 'mPQ', 'mDQ', 'mSQ'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    # PNuRL相关参数
    parser.add_argument("--use_pnurl", action='store_true', help="启用PNuRL训练（使用属性提示词增强图像特征）")
    parser.add_argument("--pnurl_clip_path", type=str, default="ViT-B/16", help="CLIP模型路径（用于PNuRL文本编码）")
    parser.add_argument("--pnurl_num_classes", type=str, default="3,5,4,3,3", help="PNuRL每个属性的类别数量，格式：颜色,形状,排列,大小,分布（默认：3,5,4,3,3）")
    parser.add_argument("--pnurl_loss_weight", type=float, default=1.0, help="PNuRL属性损失的权重")
    parser.add_argument("--attribute_info_path", type=str, default=None, help="属性信息文件路径（如果不在data_dir中）")
    args = parser.parse_args()
    
    # 解析PNuRL类别数量
    if args.use_pnurl:
        try:
            args.pnurl_num_classes = [int(x.strip()) for x in args.pnurl_num_classes.split(',')]
            if len(args.pnurl_num_classes) != 5:
                raise ValueError("PNuRL类别数量必须是5个（颜色、形状、排列、大小、分布）")
        except Exception as e:
            print(f"警告: 解析PNuRL类别数量失败: {e}，使用默认值 [3, 5, 4, 3, 3]")
            args.pnurl_num_classes = [3, 5, 4, 3, 3]
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif key == 'attribute_prompts':
                # 属性提示保持为列表，但需展开 DataLoader 默认 collate 产生的 tuple
                if isinstance(value, (list, tuple)):
                    flattened_prompts = []
                    for item in value:
                        if isinstance(item, (list, tuple)):
                            # DataLoader 对单样本会给出 ('prompt',)，多样本则 ('p1','p2',...)
                            if len(item) > 0:
                                flattened_prompts.append(item[0])
                        else:
                            flattened_prompts.append(item)
                    device_input[key] = flattened_prompts
                else:
                    device_input[key] = value
            elif key == 'attribute_labels':
                # 属性标签是tensor列表，需要移到device
                if isinstance(value, list):
                    device_input[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                else:
                    device_input[key] = value
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False, text_embeddings = None):
    """
    Args:
        args: 训练参数
        batched_input: 批次输入数据
        model: SAM 模型
        image_embeddings: 图像嵌入特征
        decoder_iter: 是否为 decoder 迭代模式
        text_embeddings: 文本嵌入（来自 PNuRL 的 learnable_context），shape: [B, feat_dim] 或 [B, 1, feat_dim]
    """
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
                text_embeddings=text_embeddings,  # 传递文本嵌入
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embeddings=text_embeddings,  # 传递文本嵌入
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        # 获取图像特征（如果使用PNuRL，PNuRL会在forward中处理）
        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())
        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])
        
        # 如果使用PNuRL，PNuRL会对ViT图像特征进行加权
        pnurl_context = None  # 初始化，确保后续可以使用
        if args.use_pnurl:
            # PNuRL处理：使用属性提示词对ViT特征进行加权
            attribute_prompts = batched_input.get("attribute_prompts", None)
            attribute_labels = batched_input.get("attribute_labels", None)
            return_loss = True if attribute_labels is not None else False
            
            # PNuRL返回：加权后的ViT特征、可学习上下文、损失、logits
            weighted_image_embeddings, pnurl_context, pnurl_loss, _ = model.pnurl(
                image_features=image_embeddings,
                attribute_prompts=attribute_prompts,
                attribute_labels=attribute_labels,
                return_loss=return_loss,
            )
            # 使用加权后的ViT特征替代原始特征（关键：让PNuRL的加权生效）
            image_embeddings = weighted_image_embeddings
        else:
            pnurl_loss = None
        
        # 处理mask_num：每个图像对应多个mask
        B, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(B):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
        
        # 如果使用PNuRL，需要将 pnurl_context 也扩展到相同的 batch 大小
        # pnurl_context shape: [B, feat_dim] -> [B * mask_num, feat_dim]
        text_embeddings_for_prompt = None
        if args.use_pnurl and pnurl_context is not None:
            B_original = pnurl_context.shape[0]  # 原始 batch size
            text_embeddings_repeat = []
            for i in range(B_original):
                text_embed = pnurl_context[i]  # [feat_dim]
                text_embed = text_embed.repeat(args.mask_num, 1)  # [mask_num, feat_dim]
                text_embeddings_repeat.append(text_embed)
            text_embeddings_for_prompt = torch.cat(text_embeddings_repeat, dim=0)  # [B * mask_num, feat_dim]
            # 添加一个维度以匹配 sparse_embeddings 的格式 [B * mask_num, 1, feat_dim]
            text_embeddings_for_prompt = text_embeddings_for_prompt.unsqueeze(1)
        
        masks, low_res_masks, iou_predictions = prompt_and_decoder(
            args, batched_input, model, image_embeddings, 
            decoder_iter=False, 
            text_embeddings=text_embeddings_for_prompt  # 传递 PNuRL 的文本提示
        )
        
        loss = criterion(masks, labels, iou_predictions)
        if pnurl_loss is not None:
            loss = loss + args.pnurl_loss_weight * pnurl_loss
        
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)
        else:
            loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch+1) % 50 == 0:
            # 将 Logits 转为 0/1 二值图
            masks_binary = (masks > 0.0).float()
            batch_metrics = SegMetrics(masks_binary, labels, args.metrics)
            if isinstance(batch_metrics, dict):
                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in batch_metrics.items()])
            else:
                metrics_str = str(batch_metrics)
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {metrics_str}')

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)
    
        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            # 在迭代过程中，也需要传递文本嵌入（如果使用PNuRL）
            # 注意：在迭代过程中，image_embeddings 已经被扩展过了，所以 text_embeddings_for_prompt 也需要匹配
            if args.use_amp:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, 
                    decoder_iter=True,
                    text_embeddings=text_embeddings_for_prompt  # 传递 PNuRL 的文本提示
                )
                loss = criterion(masks, labels, iou_predictions)
                with amp.scale_loss(loss,  optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, 
                    decoder_iter=True,
                    text_embeddings=text_embeddings_for_prompt  # 传递 PNuRL 的文本提示
                )
                loss = criterion(masks, labels, iou_predictions)
                loss.backward(retain_graph=True)
                
            optimizer.step()
            optimizer.zero_grad()
          
            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)
       
            if int(batch+1) % 50 == 0:
                # 将 Logits 转为 0/1 二值图
                masks_binary = (masks > 0.0).float()
                batch_metrics = SegMetrics(masks_binary, labels, args.metrics)
                if isinstance(batch_metrics, dict):
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in batch_metrics.items()])
                else:
                    metrics_str = str(batch_metrics)
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {metrics_str}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: {metrics_str}')

        if int(batch+1) % 200 == 0:
            print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        # 将 Logits 转为 0/1 二值图
        masks_binary = (masks > 0.0).float()
        train_batch_metrics = SegMetrics(masks_binary, labels, args.metrics)
        # SegMetrics返回字典，需要转换为列表格式
        if isinstance(train_batch_metrics, dict):
            # 处理metric名称映射：mDice -> dice（因为metrics.py中mDice返回的键是'dice'）
            train_batch_metrics = [
                train_batch_metrics.get('dice' if metric == 'mDice' else metric, 0.0) 
                for metric in args.metrics
            ]
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics



def main(args):
    # 如果启用PNuRL，设置相关参数
    if args.use_pnurl:
        args.use_pnurl = True
        args.pnurl_config = {
            'clip_model_path': args.pnurl_clip_path,
            'num_classes_per_attr': args.pnurl_num_classes,
            'attr_loss_weight': args.pnurl_loss_weight,
        }
        print(f"启用PNuRL训练")
        print(f"  - CLIP模型路径: {args.pnurl_clip_path}")
        print(f"  - 属性类别数: {args.pnurl_num_classes}")
        print(f"  - 损失权重: {args.pnurl_loss_weight}")
    
    model = sam_model_registry[args.model_type](args).to(args.device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')

    start_epoch = 0
    if args.resume is not None:
        print(f"*******恢复训练从: {args.resume}")
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 从保存的epoch继续训练
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"*******从epoch {start_epoch}继续训练")
            
            # 恢复最佳loss和epoch
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
                print(f"*******最佳loss: {best_loss:.4f}")
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
                print(f"*******最佳epoch: {best_epoch}")
            
            # 恢复学习率调度器状态
            if args.lr_scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"*******恢复学习率调度器状态")
            
            print(f"*******成功加载checkpoint")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(
        args.data_path, 
        image_size=args.image_size, 
        mode='train', 
        point_num=1, 
        mask_num=args.mask_num, 
        requires_name=False,
        attribute_info_path=args.attribute_info_path
    )
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    # 初始化最佳loss和epoch（如果resume会覆盖这些值）
    best_loss = 1e10
    best_epoch = 0
    l = len(train_loader)
    save_interval = 50  # 每50个epoch保存一次最好的结果

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

        # 更新最佳loss
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch + 1
            print(f"✓ 新的最佳loss: {best_loss:.4f} (epoch {best_epoch})")

        # 每50个epoch保存一次最好的结果
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"best_epoch{best_epoch}_loss{best_loss:.4f}_sam.pth")
            state = {
                'model': model.float().state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'train_metrics': train_metrics
            }
            # 保存学习率调度器状态
            if args.lr_scheduler is not None:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, save_path)
            print(f"✓ 保存最佳模型到: {save_path}")
            print(f"  - 最佳epoch: {best_epoch}, 最佳loss: {best_loss:.4f}")
            if args.use_amp:
                model = model.half()
        
        # 最后一个epoch也保存
        if epoch + 1 == args.epochs:
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"best_epoch{best_epoch}_loss{best_loss:.4f}_sam.pth")
            state = {
                'model': model.float().state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'train_metrics': train_metrics
            }
            # 保存学习率调度器状态
            if args.lr_scheduler is not None:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, save_path)
            print(f"✓ 训练完成，保存最终最佳模型到: {save_path}")
            if args.use_amp:
                model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


