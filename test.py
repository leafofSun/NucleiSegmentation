import argparse
import os
import torch
import numpy as np
import cv2
import json
from pycocotools import mask as mask_utils
from tqdm import tqdm
from collections import defaultdict
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from metrics import SegMetrics
import torch.nn.functional as F
from PIL import Image

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
# 1. 核心后处理：np.gradient
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
    
    marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    markers = skimage_label(marker_map).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)

# ==================================================================================================
# 2. 单次批量推理 (关闭 TTA，保留原函数名防止报错)
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, args):
    device = args.device
    input_size = (args.image_size, args.image_size)
    
    # 1. 仅仅做 Resize，没有任何翻转和旋转
    img_t = cv2.resize(image_rgb, input_size)
    img_tensor = torch.from_numpy(img_t).permute(2, 0, 1).float().to(device)
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        input_sample = [{'image': img_tensor, 'original_size': input_size, 'organ_id': organ_id, 'text_prompt': "Cell nuclei"}]
        
        # 2. 只做这一次前向传播
        out = model(input_sample, multimask_output=True)[0]
        best_idx = torch.argmax(out['iou_predictions']).item()
        
        prob = torch.sigmoid(out['masks'][0, best_idx]) 
        hv_raw = out.get('hv_logits')
        
        if hv_raw is not None:
            if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
            hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
            hv = torch.tanh(hv) 
        else:
            hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

        # 3. 直接转为 numpy 返回，不需要求平均了
        prob_np = prob.cpu().float().numpy()
        hv_np = hv.cpu().float().numpy()
        
    return prob_np, hv_np, out.get('attr_logits', {})
# ==================================================================================================
# 2. 8-fold TTA 批量推理 (GPU 物理坐标对齐版)
# ==================================================================================================
# def tta_inference_8x_batch(model, image_rgb, organ_id, args):
#     device = args.device
#     input_size = (args.image_size, args.image_size)
    
#     transforms = [
#         (None, 0), (1, 0), (0, 0), (-1, 0), 
#         (None, 1), (1, 1), (0, 1), (-1, 1)  
#     ]
    
#     img_list = []
#     for f_code, r_k in transforms:
#         img_t = image_rgb.copy()
#         if f_code is not None: img_t = cv2.flip(img_t, f_code)
#         if r_k > 0: img_t = np.rot90(img_t, k=r_k)
#         img_t = cv2.resize(img_t, input_size)
#         img_list.append(torch.from_numpy(img_t).permute(2, 0, 1).float())
    
#     batch_img = torch.stack(img_list).to(device)
#     all_probs = []
#     all_hvs = []
    
#     with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#         for i in range(len(transforms)):
#             input_sample = [{'image': batch_img[i], 'original_size': input_size, 'organ_id': organ_id, 'text_prompt': "Cell nuclei"}]
#             out = model(input_sample, multimask_output=True)[0]
#             best_idx = torch.argmax(out['iou_predictions']).item()
            
#             prob = torch.sigmoid(out['masks'][0, best_idx]) 
#             hv_raw = out.get('hv_logits')
#             if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
#             hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
#             hv = torch.tanh(hv) 

#             f_code, r_k = transforms[i]
            
#             if r_k == 1:
#                 prob = torch.rot90(prob, k=-1, dims=[0, 1])
#                 hv = torch.rot90(hv, k=-1, dims=[1, 2])
#                 v_new, h_new = hv[1].clone(), -hv[0].clone()
#                 hv[0], hv[1] = v_new, h_new

#             if f_code == 1: 
#                 prob = torch.flip(prob, [1]); hv = torch.flip(hv, [2]); hv[1] = -hv[1]
#             elif f_code == 0: 
#                 prob = torch.flip(prob, [0]); hv = torch.flip(hv, [1]); hv[0] = -hv[0]
#             elif f_code == -1: 
#                 prob = torch.flip(prob, [0, 1]); hv = torch.flip(hv, [1, 2]); hv = -hv
            
#             all_probs.append(prob)
#             all_hvs.append(hv)

#         avg_prob = torch.stack(all_probs).mean(0).cpu().float().numpy()
#         avg_hv = torch.stack(all_hvs).mean(0).cpu().float().numpy()
        
#     return avg_prob, avg_hv, out.get('attr_logits', {})

# ==================================================================================================
# 3. 滑动窗口推理 (Sliding Window Inference + Gaussian Weighting)
# ==================================================================================================
def get_gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel.astype(np.float32)

def sliding_window_inference(model, image_rgb, organ_id, args, patch_size=256, overlap=0.5):
    h, w = image_rgb.shape[:2]
    stride = int(patch_size * (1 - overlap))
    
    # 1. 计算 Padding (确保能被完整切割)
    pad_h = 0 if h % stride == 0 else stride - (h % stride)
    pad_w = 0 if w % stride == 0 else stride - (w % stride)
    pad_h = max(pad_h, patch_size - h) if h < patch_size else pad_h
    pad_w = max(pad_w, patch_size - w) if w < patch_size else pad_w
    
    padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pad_h_full, pad_w_full = padded_img.shape[:2]
    
    # 2. 初始化拼接画布
    canvas_prob = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    canvas_hv = np.zeros((2, pad_h_full, pad_w_full), dtype=np.float32)
    canvas_weight = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    
    weight_mask = get_gaussian_kernel(patch_size, sigma=0.5)
    accumulated_size_logits = None
    
    # 3. 滑动推理
    for y in range(0, pad_h_full - patch_size + 1, stride):
        for x in range(0, pad_w_full - patch_size + 1, stride):
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            
            # patch 本身是 256x256，送入 tta 会被 resize 到 args.image_size (512x512) 推理
            prob_512, hv_512, attr_logits = tta_inference_8x_batch(model, patch, organ_id, args)
            
            # 将 512x512 的输出放缩回 256x256 以拼接到画布
            prob_256 = cv2.resize(prob_512, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            hv_v_256 = cv2.resize(hv_512[0], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            hv_h_256 = cv2.resize(hv_512[1], (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            
            canvas_prob[y:y+patch_size, x:x+patch_size] += prob_256 * weight_mask
            canvas_hv[0, y:y+patch_size, x:x+patch_size] += hv_v_256 * weight_mask
            canvas_hv[1, y:y+patch_size, x:x+patch_size] += hv_h_256 * weight_mask
            canvas_weight[y:y+patch_size, x:x+patch_size] += weight_mask
            
            # 累加属性 logit，用于整图的自适应阈值
            if 'size' in attr_logits:
                if accumulated_size_logits is None:
                    accumulated_size_logits = attr_logits['size'].detach().cpu().clone()
                else:
                    accumulated_size_logits += attr_logits['size'].detach().cpu()
                    
    # 4. 加权归一化与裁剪
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
            print(f"⚠️ 解析警告: {json_path} 中未找到 annotations 列表。")
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
                
        if instance_map.max() == 0:
            print(f"⚠️ 解析警告: {json_path} 已读取，但实例全为空。")
            
        return instance_map
    except Exception as e:
        print(f"❌ 解析错误 {json_path}: {str(e)}")
        return None

# ==================================================================================================
# 5. Main
# ==================================================================================================
def main(args):
    vanilla_sam = sam_model_registry[args.model_type](args)
    
    # 🔥 [修复] 显式传入消融开关状态，防止默认开启随机权重的模块
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder, 
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder, 
        num_organs=21,
        use_pnurl=args.use_pnurl,
        use_coop=args.use_coop,
        use_sgot=args.use_sgot,
        use_asr=args.use_asr
    ).to(args.device)
    
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    state_dict = resize_pos_embed(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_metrics = defaultdict(list)
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.lower().endswith(('.png', '.tif'))]
    
    save_dir = os.path.join("workdir", "eval_tta8x", "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="Sliding Window 8x-TTA Inference"):
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
            # 🔥 不再破坏 Ground Truth 结构！只确保预测图的维度与 GT 严格对齐
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)
        else:
            print(f"❌ Missing GT: {img_path}") 

    print("\n" + "🌟" * 15 + "\n📊 Final Results (Sliding Window + Oracle Organ ID + 8x TTA):")
    for k, v in all_metrics.items(): print(f"{k:>10}: {np.mean(v):.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--prob_thresh", type=float, default=0.35)
    parser.add_argument("--marker_thresh", type=float, default=0.40)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    
    # 消融开关，默认关闭，与纯视觉基线对齐
    parser.add_argument("--use_pnurl", action='store_true', default=False)
    parser.add_argument("--use_coop", action='store_true', default=False)
    parser.add_argument("--use_sgot", action='store_true', default=False)
    parser.add_argument("--use_asr", action='store_true', default=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())