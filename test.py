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

    # 修复：处理多个框的情况
    # 如果 boxes 的形状是 [1, N, 4]，需要 reshape 为 [N, 4] 以便 prompt_encoder 正确处理
    boxes = batched_input.get("boxes", None)
    num_boxes = 1  # 默认 batch size
    if boxes is not None:
        if len(boxes.shape) == 3 and boxes.shape[0] == 1:
            # [1, N, 4] -> [N, 4]
            num_boxes = boxes.shape[1]
            boxes = boxes.squeeze(0)
        elif len(boxes.shape) == 2:
            # [N, 4] 已经是正确格式
            num_boxes = boxes.shape[0]
        else:
            # 其他情况，尝试 flatten
            boxes = boxes.reshape(-1, 4)
            num_boxes = boxes.shape[0]
    
    # 如果有多于1个框，需要扩展 image_embeddings 以匹配 batch size
    if num_boxes > 1:
        # image_embeddings: [1, C, H, W] -> [N, C, H, W]
        image_embeddings = image_embeddings.repeat(num_boxes, 1, 1, 1)
        # text_embeddings 也需要扩展（如果存在）
        if text_embeddings is not None:
            if len(text_embeddings.shape) == 2:
                # [1, feat_dim] -> [N, feat_dim]
                text_embeddings = text_embeddings.repeat(num_boxes, 1)
            elif len(text_embeddings.shape) == 3:
                # [1, 1, feat_dim] -> [N, 1, feat_dim]
                text_embeddings = text_embeddings.repeat(num_boxes, 1, 1)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=boxes,  # 使用处理后的 boxes
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
        
        # ================== [修复] 构建实例图 (Instance Map) 而不是简单合并 ==================
        # 原逻辑: masks, _ = torch.max(masks, dim=0, keepdim=True) (导致粘连)
        # 新逻辑: 为每个 box 的预测分配唯一 ID，构建实例索引图
        
        # 保存原始 logits 用于 loss 计算
        masks_for_loss = masks.clone()
        if masks.shape[0] > 1:
            # 对于 loss 计算，使用 max 合并（保持原有逻辑）
            masks_for_loss, _ = torch.max(masks_for_loss, dim=0, keepdim=True)
        
        N, _, H, W = masks.shape
        
        if N > 1:
            # 构建实例图：为每个预测分配唯一 ID
            full_instance_map = torch.zeros((H, W), dtype=torch.int32, device=masks.device)
            
            # 将 logits 转为概率
            probs = torch.sigmoid(masks)
            
            # 逐个叠加实例 (Painter's Algorithm)
            # 注意：如果有重叠，后一个会覆盖前一个
            # 更高级的做法是比较重叠区域的置信度，但对于细胞核，这种简单做法通常够用
            for idx in range(N):
                # 获取第 idx 个实例的 mask (阈值 0.5)
                # shape: [1, H, W]
                single_mask = probs[idx] > 0.5
                
                # 赋值 ID (idx + 1)，因为 0 是背景
                # 只在 mask 为 True 的地方赋值
                full_instance_map[single_mask.squeeze()] = idx + 1
            
            # 调整维度以适配后续代码 [H, W] -> [1, 1, H, W]
            # 注意：现在 masks 里面存的是 int32 的 ID，不再是 logits
            masks = full_instance_map.unsqueeze(0).unsqueeze(0).float()
            
            # iou_predictions 也取最大值或者平均值，仅用于 loss 计算
            if iou_predictions is not None and iou_predictions.shape[0] > 1:
                iou_predictions, _ = torch.max(iou_predictions, dim=0, keepdim=True)
        else:
            # 如果只有一个预测，仍然需要转换为实例图格式（ID=1）
            probs = torch.sigmoid(masks)
            single_mask = (probs > 0.5).squeeze()
            full_instance_map = torch.zeros((H, W), dtype=torch.int32, device=masks.device)
            full_instance_map[single_mask] = 1
            masks = full_instance_map.unsqueeze(0).unsqueeze(0).float()
        # ================== [结束修复] ==================
        
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)
        # ================== [结束修复] ==================

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
        
        # 【修复】不再强制二值化，保持实例信息用于 Metrics 计算
        # ori_labels 应该是实例图（如果 DataLoader 正确返回了 ori_label）
        # 如果 ori_labels 是实例图（max > 1），我们需要保持它用于 AJI/PQ 计算
        # 但 Loss 计算需要二值图，所以创建一个二值副本
        if ori_labels.max() > 1:
            if i == 0:
                print(f"  >>> 检测到实例图 (Max={ori_labels.max().item():.2f})，保持实例信息用于 Metrics 计算")
            # 创建二值副本用于 Loss 计算
            ori_labels_binary = (ori_labels > 0).float()
        else:
            # 已经是二值图
            ori_labels_binary = ori_labels
        # ================== [结束] 标签值域检查和修复 ==================
        
        # 计算 Loss 用二值
        loss = criterion(masks_for_loss, ori_labels_binary, iou_predictions)
        test_loss.append(loss.item())

        # ================== [开始修复] 数据对齐与清洗 ==================
        # 1. 重新映射 ori_labels 的 ID，使其从 1 开始连续排列
        #    解决 Max=11730 的问题，把它变成 1...N
        #    注意：ori_labels 是 tensor [1, 1, H, W]
        gt_numpy = ori_labels.squeeze().cpu().numpy().astype(int)
        
        # 使用 skimage.segmentation.relabel_sequential 重新编号
        from skimage.segmentation import relabel_sequential
        gt_relabelled, _, _ = relabel_sequential(gt_numpy)
        
        # 转回 Tensor
        ori_labels_clean = torch.tensor(gt_relabelled).unsqueeze(0).unsqueeze(0).to(args.device).float()
        
        # 2. 确保 Pred 也是干净的实例图
        #    masks 已经是我们自己构建的实例图，通常从 1 开始，问题不大
        #    但为了保险，也可以转一下类型
        masks_clean = masks.round().float()
        
        # ================== [修改 SegMetrics 调用] ==================
        # 我们现在有两个版本的 GT：
        # 1. ori_labels_clean: 实例图 (0, 1, 2...) -> 用于 AJI, PQ
        # 2. ori_labels_binary: 二值图 (0, 1) -> 用于 Dice
        
        # 临时方案：直接在这里计算 Dice，绕过 SegMetrics 的潜在 Bug
        # 手动计算 Batch Dice (Binary)
        pred_bin = (masks_clean > 0).float()
        gt_bin = (ori_labels_clean > 0).float()
        inter = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum()
        dice_val = 2.0 * inter / (union + 1e-7)
        
        # 调用 SegMetrics 计算 AJI/PQ (传入清洗后的实例图)
        # 注意：metrics.py 里的 AJI/PQ 计算能处理实例图输入
        metrics_results = SegMetrics(masks_clean, ori_labels_clean, args.metrics)
        
        # 覆盖 Dice 结果
        if isinstance(metrics_results, dict):
            metrics_results['dice'] = dice_val.item()
            metrics_results['mDice'] = dice_val.item()  # 双保险
            
            # 将结果转为列表，适配后续代码
            metric_values = []
            for metric in args.metrics:
                if metric in ['mDice', 'dice']:
                    metric_values.append(dice_val.item())
                else:
                    # 获取其他指标 (AJI, PQ)
                    val = metrics_results.get(metric, 0.0)
                    metric_values.append(val)
            test_batch_metrics = metric_values
        else:
            # 如果 SegMetrics 返回的是列表，可能需要手动替换第一个元素
            # 但根据你的代码，它应该返回字典
            test_batch_metrics = metrics_results
        
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
        # ================== [结束修复] ==================

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
