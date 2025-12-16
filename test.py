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
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
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
    parser.add_argument("--pnurl_clip_path", type=str, default="ViT-B/16", help="CLIP模型路径（用于PNuRL文本编码）")
    parser.add_argument("--pnurl_num_classes", type=str, default="3,5,4,3,3", help="PNuRL每个属性的类别数量")
    parser.add_argument("--attribute_info_path", type=str, default="data/cpm17/attribute_info_test.json", help="属性信息文件路径")
    args = parser.parse_args()
    
    # 解析PNuRL类别数量
    if args.use_pnurl:
        try:
            if isinstance(args.pnurl_num_classes, str):
                args.pnurl_num_classes = [int(x.strip()) for x in args.pnurl_num_classes.split(',')]
            if len(args.pnurl_num_classes) != 5:
                # 默认回退
                args.pnurl_num_classes = [3, 5, 4, 3, 3]
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
                # 属性提示保持为列表，但需展开 DataLoader 默认 collate 产生的 tuple
                if isinstance(value, (list, tuple)):
                    flattened_prompts = []
                    for item in value:
                        if isinstance(item, (list, tuple)):
                            if len(item) > 0:
                                flattened_prompts.append(item[0])
                        else:
                            flattened_prompts.append(item)
                    device_input[key] = flattened_prompts
                else:
                    device_input[key] = value
            elif key == 'attribute_labels':
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
    """
    后处理：Padding -> 1024 -> Crop Padding -> Resize Back to Original
    """
    # ... (获取 ori_h, ori_w 代码不变) ...
    if isinstance(original_size, (list, tuple)) and len(original_size) == 2:
        if isinstance(original_size[0], torch.Tensor):
            ori_h = original_size[0].item() if original_size[0].numel() == 1 else int(original_size[0])
            ori_w = original_size[1].item() if original_size[1].numel() == 1 else int(original_size[1])
        else:
            ori_h, ori_w = original_size
    else:
        ori_h, ori_w = image_size, image_size

    # 1. 上采样到 1024 (模型输出)
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    
    # 2. 计算缩放后的尺寸 (与 DataLoader 逻辑对应)
    scale = image_size * 1.0 / max(ori_h, ori_w)
    new_h, new_w = int(ori_h * scale), int(ori_w * scale)
    
    # 3. 计算 Padding (与 DataLoader 逻辑对应)
    pad_h = max(image_size - new_h, 0)
    pad_w = max(image_size - new_w, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    
    # 4. 裁剪 Padding (Crop)
    masks = masks[..., pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    
    # 5. 反向缩放回原始尺寸 (Resize Back)
    masks = F.interpolate(masks, (ori_h, ori_w), mode="bilinear", align_corners=False)
    
    pad = (pad_top, pad_left)
    return masks, pad

def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings, text_embeddings=None):
    """
    Args:
        args: 测试参数
        batched_input: 批次输入数据
        ddp_model: SAM 模型
        image_embeddings: 图像嵌入特征
        text_embeddings: 文本嵌入（来自 PNuRL 的 learnable_context）
    """
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    # 处理 boxes
    boxes = batched_input.get("boxes", None)
    num_boxes = 1
    
    if boxes is not None:
        if len(boxes.shape) == 3 and boxes.shape[0] == 1:
            # [1, N, 4] -> [N, 4]
            num_boxes = boxes.shape[1]
            boxes = boxes.squeeze(0)
        elif len(boxes.shape) == 2:
            num_boxes = boxes.shape[0]
        else:
            boxes = boxes.reshape(-1, 4)
            num_boxes = boxes.shape[0]
            
    # 【核心修复】这里绝对不要加 max_boxes = 50 的限制！
    # 让模型处理所有框，否则 AJI/Dice 会因漏检而极低。
    
    # 扩展 image_embeddings 以匹配 batch size (num_boxes)
    if num_boxes > 1:
        # [1, C, H, W] -> [N, C, H, W]
        image_embeddings = image_embeddings.repeat(num_boxes, 1, 1, 1)
        # 扩展 text_embeddings
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
            boxes=boxes,
            masks=batched_input.get("mask_inputs", None),
            text_embeddings=text_embeddings,
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        # 选择置信度最高的 mask
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    
    # 这里暂时不需要插值到 image_size，留给 postprocess_masks 处理
    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


def main(args):
    # PNuRL 配置
    if args.use_pnurl:
        args.pnurl_config = {
            'clip_model_path': args.pnurl_clip_path,
            'num_classes_per_attr': args.pnurl_num_classes,
            'attr_loss_weight': 1.0,
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

    # 加载模型
    model = sam_model_registry[args.model_type](args).to(args.device) 
    
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
            
            # PNuRL: 文本特征注入
            pnurl_context = None
            if args.use_pnurl:
                attribute_prompts = batched_input.get("attribute_prompts", None)
                if attribute_prompts is not None:
                    weighted_image_embeddings, pnurl_context, _, _ = model.pnurl(
                        image_features=image_embeddings,
                        attribute_prompts=attribute_prompts,
                        attribute_labels=None,
                        return_loss=False,
                    )
                    image_embeddings = weighted_image_embeddings

            # 准备 text_embeddings
            text_embeddings_input = None
            if pnurl_context is not None:
                text_embeddings_input = pnurl_context.unsqueeze(1)
            
            if args.boxes_prompt:
                save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
                batched_input["point_coords"], batched_input["point_labels"] = None, None
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, 
                    text_embeddings=text_embeddings_input
                )
                points_show = None
            else:
                save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
                batched_input["boxes"] = None
                point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
                for iter_idx in range(args.iter_point):
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(
                        args, batched_input, model, image_embeddings,
                        text_embeddings=text_embeddings_input
                    )
                    if iter_idx != args.iter_point-1:
                        batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                        batched_input = to_device(batched_input, args.device)
                        point_coords.append(batched_input["point_coords"])
                        point_labels.append(batched_input["point_labels"])
                        batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                        batched_input["point_labels"] = torch.concat(point_labels, dim=1)
                points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        # === [核心修复] 后处理：裁剪 Padding ===
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        
        # === [核心修复] 构建实例分割图 (防止 mask 粘连) ===
        # masks shape: [N, 1, H, W]
        N, _, H, W = masks.shape
        
        # 准备 Loss 计算用的 merged mask
        masks_for_loss = masks.clone()
        if N > 1:
            masks_for_loss, _ = torch.max(masks_for_loss, dim=0, keepdim=True)
        
        if N > 1:
            # 这里的 N 等于 box 的数量 (例如 46)
            full_instance_map = torch.zeros((H, W), dtype=torch.int32, device=masks.device)
            probs = torch.sigmoid(masks)
            
            # Painter's algorithm: 依次叠加
            for idx in range(N):
                single_mask = probs[idx] > 0.5
                full_instance_map[single_mask.squeeze()] = idx + 1
            
            # 转回 float tensor [1, 1, H, W] 用于 metrics
            masks = full_instance_map.unsqueeze(0).unsqueeze(0).float()
            
            if iou_predictions is not None and iou_predictions.shape[0] > 1:
                iou_predictions, _ = torch.max(iou_predictions, dim=0, keepdim=True)
        else:
            probs = torch.sigmoid(masks)
            full_instance_map = (probs > 0.5).int()
            masks = full_instance_map
        
        if args.save_pred:
            boxes_for_vis = None
            if batched_input.get("boxes", None) is not None:
                boxes_for_vis = batched_input["boxes"]
                if boxes_for_vis.dim() == 3:
                    boxes_for_vis = boxes_for_vis.squeeze(0)
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, boxes_for_vis, points_show)

        # 处理 GT 标签维度
        if len(ori_labels.shape) == 3:
            ori_labels = ori_labels.unsqueeze(0)
        elif len(ori_labels.shape) == 2:
            ori_labels = ori_labels.unsqueeze(0).unsqueeze(0)
        
        # 确保尺寸匹配 (防止除法取整误差导致的 1px 差异)
        if masks.shape[2:] != ori_labels.shape[2:]:
            ori_labels = F.interpolate(ori_labels, size=masks.shape[2:], mode='nearest')
        
        # 重排 GT ID
        from skimage.segmentation import relabel_sequential
        gt_numpy = ori_labels.squeeze().cpu().numpy().astype(int)
        gt_relabelled, _, _ = relabel_sequential(gt_numpy)
        ori_labels = torch.tensor(gt_relabelled).unsqueeze(0).unsqueeze(0).to(args.device).float()
        
        # 标签检查
        if i == 0:
            print(f"\n[DEBUG] 标签值域: GT Max={ori_labels.max().item()}, Pred Max={masks.max().item()}")

        # Loss 用二值
        ori_labels_binary = (ori_labels > 0).float()
        loss = criterion(masks_for_loss, ori_labels_binary, iou_predictions)
        test_loss.append(loss.item())

        # Metrics 计算 (使用重排后的实例图)
        pred_numpy = masks.squeeze().cpu().numpy().astype(int)
        pred_relabelled, _, _ = relabel_sequential(pred_numpy)
        masks_clean = torch.tensor(pred_relabelled).unsqueeze(0).unsqueeze(0).to(args.device).float()
        
        # 计算指标
        # 临时手动计算 Dice 以确保准确性
        pred_bin = (masks_clean > 0).float()
        gt_bin = (ori_labels > 0).float()
        inter = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum()
        dice_val = 2.0 * inter / (union + 1e-7)
        
        metrics_results = SegMetrics(masks_clean, ori_labels, args.metrics)
        
        # 整理结果
        metric_values = []
        for metric in args.metrics:
            if metric in ['mDice', 'dice']:
                metric_values.append(dice_val.item())
            else:
                metric_values.append(metrics_results.get(metric, 0.0))
        
        test_batch_metrics = [float('{:.4f}'.format(m)) for m in metric_values]

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