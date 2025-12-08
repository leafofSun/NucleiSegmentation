from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['mDice', 'mAJI', 'mPQ', 'mDQ', 'mSQ'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    # PNuRL相关参数
    parser.add_argument("--use_pnurl", action='store_true', help="启用PNuRL（使用属性提示词增强图像特征）")
    parser.add_argument("--pnurl_clip_path", type=str, default=None, help="CLIP模型路径（用于PNuRL文本编码）")
    parser.add_argument("--pnurl_num_classes", type=str, default="3,5,4,3,3", help="PNuRL每个属性的类别数量，格式：颜色,形状,排列,大小,分布（默认：3,5,4,3,3）")
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
    
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif key == 'attribute_prompts':
                # 属性提示保持为列表，不需要移到device
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


def postprocess_masks(low_res_masks, image_size, original_size):
    # 处理original_size可能是tensor或tuple的情况
    if isinstance(original_size, (list, tuple)) and len(original_size) == 2:
        if isinstance(original_size[0], torch.Tensor):
            ori_h = original_size[0].item() if original_size[0].numel() == 1 else int(original_size[0])
            ori_w = original_size[1].item() if original_size[1].numel() == 1 else int(original_size[1])
        else:
            ori_h, ori_w = original_size
    else:
        ori_h, ori_w = original_size
    
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, (ori_h, ori_w), mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings, text_embeddings=None):
    """
    Args:
        args: 测试参数
        batched_input: 批次输入数据
        ddp_model: SAM 模型
        image_embeddings: 图像嵌入特征
        text_embeddings: 文本嵌入（来自 PNuRL 的 learnable_context），shape: [B, feat_dim] 或 [B, 1, feat_dim]
    """
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embeddings=text_embeddings,  # 传递文本嵌入
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
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


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    # 如果启用PNuRL，设置相关参数
    if args.use_pnurl:
        args.pnurl_config = {
            'clip_model_path': args.pnurl_clip_path,
            'num_classes_per_attr': args.pnurl_num_classes,
            'attr_loss_weight': 1.0,  # 测试时不需要损失权重，但需要设置
        }
        print(f"启用PNuRL测试")
        print(f"  - CLIP模型路径: {args.pnurl_clip_path}")
        print(f"  - 属性类别数: {args.pnurl_num_classes}")
    else:
        args.pnurl_config = None
    
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args).to(args.device) 
    
    # # 调试：检查模型是否加载了checkpoint
    # if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
    #     print(f"\n[调试] 检查checkpoint加载:")
    #     print(f"  Checkpoint路径: {args.sam_checkpoint}")
    #     # 检查模型参数是否已更新（不是随机初始化）
    #     first_param = next(iter(model.parameters()))
    #     print(f"  模型第一个参数统计: mean={first_param.data.mean().item():.6f}, std={first_param.data.std().item():.6f}")
    #     # 检查mask_decoder的输出层参数
    #     if hasattr(model, 'mask_decoder'):
    #         output_hypernetworks = model.mask_decoder.output_hypernetworks_mlps
    #         if len(output_hypernetworks) > 0:
    #             first_mlp_param = next(iter(output_hypernetworks[0].parameters()))
    #             print(f"  Mask decoder输出层参数: mean={first_mlp_param.data.mean().item():.6f}, std={first_mlp_param.data.std().item():.6f}")

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(
        data_path=args.data_path, 
        image_size=args.image_size, 
        mode='test', 
        requires_name=True, 
        point_num=args.point_num, 
        return_ori_mask=True, 
        prompt_path=args.prompt_path,
        attribute_info_path=args.attribute_info_path
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        # 确保original_size是正确的格式（tuple或list）
        if isinstance(original_size, (list, tuple)) and len(original_size) == 2:
            if isinstance(original_size[0], torch.Tensor):
                # 如果是tensor，转换为int
                original_size = (int(original_size[0].item() if original_size[0].numel() == 1 else original_size[0]), 
                                int(original_size[1].item() if original_size[1].numel() == 1 else original_size[1]))
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])
            
            # 如果使用PNuRL，对图像特征进行加权
            pnurl_context = None
            if args.use_pnurl:
                attribute_prompts = batched_input.get("attribute_prompts", None)
                # 测试时不需要计算损失，所以不传入attribute_labels
                if attribute_prompts is not None:
                    weighted_image_embeddings, pnurl_context, _, _ = model.pnurl(
                        image_features=image_embeddings,
                        attribute_prompts=attribute_prompts,
                        attribute_labels=None,
                        return_loss=False,
                    )
                    # 使用加权后的ViT特征
                    image_embeddings = weighted_image_embeddings

        # 处理 pnurl_context 为 text_embeddings 格式
        text_embeddings_input = None
        if pnurl_context is not None:
            # pnurl_context shape: [B, feat_dim] -> [B, 1, feat_dim] 以匹配 sparse_embeddings 格式
            text_embeddings_input = pnurl_context.unsqueeze(1)
        
        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings, 
                text_embeddings=text_embeddings_input  # 传递 PNuRL 的文本提示
            )
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter_idx in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings,
                    text_embeddings=text_embeddings_input  # 传递 PNuRL 的文本提示
                )
                if iter_idx != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        # 后处理 masks：将低分辨率 masks 调整到原始图像尺寸
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        # 确保ori_labels的形状与masks匹配：[B, 1, H, W]
        # ori_labels从DataLoader返回时是[1, H, W]，需要添加batch和channel维度
        if len(ori_labels.shape) == 3:
            # [1, H, W] -> [1, 1, H, W]
            ori_labels = ori_labels.unsqueeze(0)
        elif len(ori_labels.shape) == 2:
            # [H, W] -> [1, 1, H, W]
            ori_labels = ori_labels.unsqueeze(0).unsqueeze(0)
        
        # 确保 masks 和 ori_labels 的空间尺寸匹配
        if masks.shape[2:] != ori_labels.shape[2:]:
            # 将 ori_labels 调整到 masks 的尺寸
            ori_labels = F.interpolate(ori_labels, size=masks.shape[2:], mode='nearest')
        
        # ================== [开始] 标签值域检查和修复 ==================
        # 调试：检查标签值域（只打印第一张图）
        if i == 0:
            print(f"\n[DEBUG] 标签值域检查: Image {img_name}")
            print(f"  - ori_labels Min: {ori_labels.min().item():.4f}, Max: {ori_labels.max().item():.4f}")
            print(f"  - masks (Pred) Min: {masks.min().item():.4f}, Max: {masks.max().item():.4f}")
        
        # 强制二值化修复：如果标签值大于 1，强制转为 0/1
        # 这是因为 CPM17 等数据集可能使用 Instance Mask（每个细胞核有不同的像素值）
        if ori_labels.max() > 1:
            if i == 0:
                print(f"  >>> 检测到标签值 > 1 (Max={ori_labels.max().item():.2f})，正在执行强制二值化 (Label > 0 -> 1)...")
            ori_labels = (ori_labels > 0).float()
            if i == 0:
                print(f"  >>> 二值化后: Min={ori_labels.min().item():.4f}, Max={ori_labels.max().item():.4f}")
        # ================== [结束] 标签值域检查和修复 ==================
        
        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        # SegMetrics返回字典，需要转换为列表
        if isinstance(test_batch_metrics, dict):
            # 按照args.metrics的顺序提取值，处理键名映射（dice/mDice统一为dice）
            metric_values = []
            for metric in args.metrics:
                # 处理dice和mDice的键名映射
                if metric == 'mDice' or metric == 'dice':
                    value = test_batch_metrics.get('dice', test_batch_metrics.get('mDice', 0.0))
                else:
                    value = test_batch_metrics.get(metric, 0.0)
                metric_values.append(value)
            test_batch_metrics = metric_values
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir,f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
