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
import json
from pycocotools import mask as mask_utils
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from PIL import Image

# === Project Module Imports ===
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from metrics import SegMetrics

# 后处理库
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as skimage_label

# ==================================================================================================
#  器官映射表 (对齐 DataLoader.py)
# ==================================================================================================
ORGAN_TO_ID = {
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, 
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7, 
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11, 
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15, 
    "Testis": 16, "Thyroid": 17, "Uterus": 18, "Brain": 19, "Generic": 20
}

# ==================================================================================================
#  权重插值函数 
# ==================================================================================================
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

# ==================================================================================================
# 1. 核心后处理
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.35, marker_thresh=0.4, min_marker_size=250):
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
    
    # 修复 Warning: min_size -> max_size 兼容 skimage 新版本
    try:
        marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    except TypeError:
        marker_map = remove_small_objects(marker_map, max_size=min_marker_size)
        
    markers = skimage_label(marker_map).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)

# ==================================================================================================
# 2. 单次批量推理
# ==================================================================================================
# def tta_inference_1x_batch(model, image_rgb, organ_id, args):
#     device = args.device
#     input_size = (args.image_size, args.image_size)
    
#     img_t = cv2.resize(image_rgb, input_size)
#     img_tensor = torch.from_numpy(img_t).permute(2, 0, 1).float().to(device)
    
#     with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#         # 🔥🔥🔥 修复致命 Bug 2：补充 'attribute_text' 以及 'attr_labels' 兜底！
#         input_sample = [{
#             'image': img_tensor, 
#             'original_size': input_size, 
#             'organ_id': organ_id, 
#             'text_prompt': "Cell nuclei",
#             'attribute_text': "Cell nuclei",  # 必须传入，否则 CLIP 抽空字符
#             'attr_labels': None
#         }]
        
#         out = model(input_sample, multimask_output=True)[0]
#         best_idx = torch.argmax(out['iou_predictions']).item()
        
#         prob = torch.sigmoid(out['masks'][0, best_idx]) 
#         hv_raw = out.get('hv_logits')
        
#         if hv_raw is not None:
#             if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
#             hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
#             hv = torch.tanh(hv) 
#         else:
#             hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

#         prob_np = prob.cpu().float().numpy()
#         hv_np = hv.cpu().float().numpy()
        
#     return prob_np, hv_np, out.get('attr_logits', {})
# ==================================================================================================
# 2. 8-fold TTA 批量推理 (包含完整的空间还原与 HoVer 向量逆转)
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, args):
    device = args.device
    input_size = (args.image_size, args.image_size)
    
    # 定义 8 种空间变换: (flip_code, rot_k)
    # flip_code: None(不翻转), 1(水平), 0(垂直), -1(水平+垂直)
    # rot_k: 0(不旋转), 1(逆时针旋转90度)
    transforms = [
        (None, 0), (1, 0), (0, 0), (-1, 0), 
        (None, 1), (1, 1), (0, 1), (-1, 1)  
    ]
    
    # 1. 准备 8 张增强后的图像
    img_list = []
    for f_code, r_k in transforms:
        img_t = image_rgb.copy()
        # 翻转
        if f_code is not None: 
            img_t = cv2.flip(img_t, f_code)
        # 旋转
        if r_k > 0: 
            img_t = np.rot90(img_t, k=r_k)
        
        img_t = cv2.resize(img_t, input_size)
        img_list.append(torch.from_numpy(img_t).permute(2, 0, 1).float())
    
    batch_img = torch.stack(img_list).to(device)
    
    all_probs = []
    all_hvs = []
    first_attr_logits = {}

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # 逐个推理以防 OOM，你也可以改成一次性 batch 推理
        for i in range(len(transforms)):
            # 🔥 必须带上 PNuRL 需要的医学语义文本
            input_sample = [{
                'image': batch_img[i], 
                'original_size': input_size, 
                'organ_id': organ_id, 
                'text_prompt': "Cell nuclei",
                'attribute_text': "Cell nuclei",  
                'attr_labels': None
            }]
            
            out = model(input_sample, multimask_output=True)[0]
            best_idx = torch.argmax(out['iou_predictions']).item()
            
            # 提取概率图和 HoVer 图
            prob = torch.sigmoid(out['masks'][0, best_idx]) 
            hv_raw = out.get('hv_logits')
            
            if hv_raw is not None:
                if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
                hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
                hv = torch.tanh(hv) 
            else:
                hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

            # 保留第一张图（原图）的属性预测，用于动态 min_size 计算
            if i == 0:
                first_attr_logits = out.get('attr_logits', {})

            # 2. 空间逆变换 (恢复到原始视角)
            f_code, r_k = transforms[i]
            
            # 逆向旋转 90 度
            if r_k == 1:
                prob = torch.rot90(prob, k=-1, dims=[0, 1])
                hv = torch.rot90(hv, k=-1, dims=[1, 2])
                # 🔥 旋转后，HoVer 的 X/Y 轴方向发生改变
                v_new, h_new = hv[1].clone(), -hv[0].clone()
                hv[0], hv[1] = v_new, h_new

            # 逆向翻转
            if f_code == 1: 
                # 水平翻转：X 轴反向
                prob = torch.flip(prob, [1])
                hv = torch.flip(hv, [2])
                hv[1] = -hv[1] 
            elif f_code == 0: 
                # 垂直翻转：Y 轴反向
                prob = torch.flip(prob, [0])
                hv = torch.flip(hv, [1])
                hv[0] = -hv[0] 
            elif f_code == -1: 
                # 水平+垂直翻转
                prob = torch.flip(prob, [0, 1])
                hv = torch.flip(hv, [1, 2])
                hv = -hv

            all_probs.append(prob)
            all_hvs.append(hv)

    # 3. 对 8 次预测结果求平均
    avg_prob = torch.stack(all_probs).mean(0).cpu().float().numpy()
    avg_hv = torch.stack(all_hvs).mean(0).cpu().float().numpy()
    
    return avg_prob, avg_hv, first_attr_logits
# ==================================================================================================
# 3. 滑动窗口推理
# ==================================================================================================
def get_gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel.astype(np.float32)

def sliding_window_inference(model, image_rgb, organ_id, args, patch_size=256, overlap=0.75):
    h, w = image_rgb.shape[:2]
    stride = int(patch_size * (1 - overlap))
    
    pad_h = 0 if h % stride == 0 else stride - (h % stride)
    pad_w = 0 if w % stride == 0 else stride - (w % stride)
    pad_h = max(pad_h, patch_size - h) if h < patch_size else pad_h
    pad_w = max(pad_w, patch_size - w) if w < patch_size else pad_w
    
    padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pad_h_full, pad_w_full = padded_img.shape[:2]
    
    canvas_prob = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    canvas_hv = np.zeros((2, pad_h_full, pad_w_full), dtype=np.float32)
    canvas_weight = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    
    weight_mask = get_gaussian_kernel(patch_size, sigma=0.5)
    accumulated_size_logits = None
    
    for y in range(0, pad_h_full - patch_size + 1, stride):
        for x in range(0, pad_w_full - patch_size + 1, stride):
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            
            prob_512, hv_512, attr_logits = tta_inference_8x_batch(model, patch, organ_id, args)
            
            prob_256 = cv2.resize(prob_512, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            hv_v_256 = cv2.resize(hv_512[0], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            hv_h_256 = cv2.resize(hv_512[1], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            
            canvas_prob[y:y+patch_size, x:x+patch_size] += prob_256 * weight_mask
            canvas_hv[0, y:y+patch_size, x:x+patch_size] += hv_v_256 * weight_mask
            canvas_hv[1, y:y+patch_size, x:x+patch_size] += hv_h_256 * weight_mask
            canvas_weight[y:y+patch_size, x:x+patch_size] += weight_mask
            
            if 'size' in attr_logits:
                if accumulated_size_logits is None:
                    accumulated_size_logits = attr_logits['size'].detach().cpu().clone()
                else:
                    accumulated_size_logits += attr_logits['size'].detach().cpu()
                    
    canvas_prob /= (canvas_weight + 1e-8)
    canvas_hv /= (canvas_weight + 1e-8)
    
    final_prob = canvas_prob[:h, :w]
    final_hv = canvas_hv[:, :h, :w]
    
    dynamic_min_size = 30
    if accumulated_size_logits is not None:
        pred_size_idx = torch.argmax(accumulated_size_logits, dim=1).item()
        dynamic_min_size = {0: 15, 1: 30, 2: 60}.get(pred_size_idx, 30)
        
    return final_prob, final_hv, dynamic_min_size

# ==================================================================================================
# 4. GT 解析工具
# ==================================================================================================
def load_filtered_gt(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path): 
        return None
        
    try:
        with open(json_path, 'r') as f: 
            data = json.load(f)
            
        if isinstance(data, list):
            data = data[0]
            
        annotations = data.get('annotations', []) if isinstance(data, dict) else data
        
        if not annotations:
            return None

        h, w = None, None
        if isinstance(data, dict):
            h = data.get('height')
            w = data.get('width')
            
        if h is None or w is None:
            first_seg = annotations[0].get('segmentation', {})
            if isinstance(first_seg, dict) and 'size' in first_seg:
                h, w = first_seg['size']
            else:
                h, w = 1000, 1000
                
        instance_map = np.zeros((h, w), dtype=np.int32)
        
        for idx, ann in enumerate(annotations):
            seg = ann.get('segmentation')
            if not seg: continue
                
            if isinstance(seg, list):
                for poly in seg:
                    poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(instance_map, [poly_np], idx + 1)
            elif isinstance(seg, dict) and 'counts' in seg:
                binary_mask = mask_utils.decode(seg)
                instance_map[binary_mask > 0] = idx + 1
                
        return instance_map
    except Exception as e:
        return None

# ==================================================================================================
# 5. Main & Args
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM Inference & Testing")
    
    # 基础路径
    parser.add_argument("--data_path", type=str, required=True, help="Path to testing data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--save_pred", action='store_true', help="Whether to save prediction images")
    
    # 模型架构维度与开关 (必须与 train.py 严格对齐)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--num_organs", type=int, default=21)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--sg_epsilon", type=float, default=0.05)
    parser.add_argument("--sg_iters", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 🔥 修复：补回漏掉的 encoder_adapter 架构参数
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    
    # 消融实验开关
    parser.add_argument("--use_pnurl", action='store_true', default=False)
    parser.add_argument("--use_coop", action='store_true', default=False)
    parser.add_argument("--use_sgot", action='store_true', default=False)
    parser.add_argument("--use_asr", action='store_true', default=False)
    
    # 推理阈值
    parser.add_argument("--prob_thresh", type=float, default=0.35)
    parser.add_argument("--marker_thresh", type=float, default=0.40)
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    
    return parser.parse_args()

def main(args):
    # 🔥🔥🔥 修复致命 Bug 4：防止 build_sam.py 在加载纯净网络时误吃微调后的权重崩溃
    fine_tuned_ckpt = args.checkpoint
    args.checkpoint = None 
    vanilla_sam = sam_model_registry[args.model_type](args)
    args.checkpoint = fine_tuned_ckpt # 恢复
    
    # 🔥🔥🔥 修复致命 Bug 3：向 TextSam 完整传入所有的必要参数，防止使用默认值导致维度不匹配
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
        use_asr=args.use_asr
    ).to(args.device)
    
    print(f"📥 Loading Checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    state_dict = resize_pos_embed(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_metrics = defaultdict(list)
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.lower().endswith(('.png', '.tif'))]
    
    save_dir = os.path.join("workdir", "eval_tta8x", "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="Sliding Window Inference"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        organ_id = 20
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list): data = data[0]
                    organ_name = data.get('organ_id', 'Generic')
                    organ_id = ORGAN_TO_ID.get(organ_name, data.get('organ_idx', 20))
            except Exception:
                pass
        
        # 🚀 滑动窗口推理
        prob, hv, dynamic_min_size = sliding_window_inference(model, image_rgb, organ_id, args, patch_size=256, overlap=0.5)
        
        # 🚀 在大图上统一执行后处理
        pred_mask = hover_post_process(prob, hv, args.prob_thresh, args.marker_thresh, min_marker_size=dynamic_min_size)
        
        if pred_mask.max() == 0:
            pred_mask = skimage_label(prob > 0.5).astype(np.int32)
        
        gt_mask = load_filtered_gt(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)
        else:
            print(f"❌ Missing GT: {img_path}") 

    print("\n" + "🌟" * 15 + "\n📊 Final Results (Sliding Window + Oracle Organ ID):")
    for k, v in all_metrics.items(): print(f"{k:>10}: {np.mean(v):.4f}")

if __name__ == '__main__':
    main(parse_args())