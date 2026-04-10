import argparse
import os
import math
import cv2 
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from collections import defaultdict
import multiprocessing as mp

# === 核心防死锁机制 (针对多核 CPU) ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from metrics import SegMetrics

from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as skimage_label

ORGAN_TO_ID = {
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, 
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7, 
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11, 
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15, 
    "Testis": 16, "Thyroid": 17, "Uterus": 18, "Brain": 19, "Generic": 20
}

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
# 1. 核心后处理 (严格回滚到产生 0.6639 的原版代码)
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.40, marker_thresh=0.45, min_marker_size=12):
    mask = prob_map > prob_thresh
    mask = binary_fill_holes(mask)
    
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    diff_v = np.gradient(v_map, axis=0) 
    diff_h = np.gradient(h_map, axis=1) 
    sobel_mag = np.sqrt(diff_v**2 + diff_h**2)
    
    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    
    try:
        marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    except TypeError:
        marker_map = remove_small_objects(marker_map, max_size=min_marker_size)
        
    markers = skimage_label(marker_map).astype(np.int32)
    inst_map = watershed(-prob_map, markers, mask=mask)
    
    try:
        inst_map = remove_small_objects(inst_map, min_size=15)
    except TypeError:
        inst_map = remove_small_objects(inst_map, max_size=15)
        
    return inst_map.astype(np.int32)

# ==================================================================================================
# 2. 8-fold TTA 批量推理
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, args):
    device = args.device
    input_size = (args.image_size, args.image_size)
    
    transforms = [
        (None, 0), (1, 0), (0, 0), (-1, 0), 
        (None, 1), (1, 1), (0, 1), (-1, 1)  
    ]
    
    img_list = []
    for f_code, r_k in transforms:
        img_t = image_rgb.copy()
        if f_code is not None: img_t = cv2.flip(img_t, f_code)
        if r_k > 0: img_t = np.rot90(img_t, k=r_k)
        img_t = cv2.resize(img_t, input_size)
        img_list.append(torch.from_numpy(img_t).permute(2, 0, 1).float())
    
    batch_img = torch.stack(img_list).to(device)
    
    all_probs = []
    all_hvs = []
    first_attr_logits = None

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        input_samples = []
        for i in range(len(transforms)):
            input_samples.append({
                'image': batch_img[i], 
                'original_size': input_size, 
                'organ_id': organ_id, 
                'text_prompt': "Cell nuclei",
                'attribute_text': "Cell nuclei",  
                'attr_labels': None
            })
        
        outputs = model(input_samples, multimask_output=True)
        
        for i in range(len(transforms)):
            out = outputs[i]
            best_idx = torch.argmax(out['iou_predictions']).item()
            prob = torch.sigmoid(out['masks'][0, best_idx]) 
            hv_raw = out.get('hv_logits')
            
            if hv_raw is not None:
                if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
                hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
                # hv = torch.tanh(hv*0.6) 
            else:
                hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

            if i == 0:
                first_attr_logits = out.get('attr_logits', {})

            f_code, r_k = transforms[i]
            
            if r_k == 1:
                prob = torch.rot90(prob, k=-1, dims=[0, 1])
                hv = torch.rot90(hv, k=-1, dims=[1, 2])
                v_new, h_new = hv[1].clone(), -hv[0].clone()
                hv[0], hv[1] = v_new, h_new

            if f_code == 1: 
                prob = torch.flip(prob, [1])
                hv = torch.flip(hv, [2])
                hv[1] = -hv[1] 
            elif f_code == 0: 
                prob = torch.flip(prob, [0])
                hv = torch.flip(hv, [1])
                hv[0] = -hv[0] 
            elif f_code == -1: 
                prob = torch.flip(prob, [0, 1])
                hv = torch.flip(hv, [1, 2])
                hv = -hv

            all_probs.append(prob)
            all_hvs.append(hv)

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

def sliding_window_inference(model, image_rgb, organ_id, args, patch_size=256, overlap=0.8):
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
    
    # 恢复原版的 0.33 sigma
    weight_mask = get_gaussian_kernel(patch_size, sigma=0.33)
    accumulated_size_logits = None
    
    for y in range(0, pad_h_full - patch_size + 1, stride):
        for x in range(0, pad_w_full - patch_size + 1, stride):
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            
            prob_512, hv_512, attr_logits = tta_inference_8x_batch(model, patch, organ_id, args)
            
            # 严格恢复原版的 INTER_LINEAR 双线性插值降采样逻辑
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
    
    # 严格恢复原版的 256 尺度自适应阈值逻辑
    dynamic_min_size = 12
    if accumulated_size_logits is not None:
        if accumulated_size_logits.ndim > 1:
            mean_logits = accumulated_size_logits.mean(dim=0)
        else:
            mean_logits = accumulated_size_logits
        pred_size_idx = torch.argmax(mean_logits).item()
        dynamic_min_size = {0: 12, 1: 25, 2: 38}.get(pred_size_idx, 12)
        
    return final_prob, final_hv, dynamic_min_size

def load_filtered_gt(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path): return None
        
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if isinstance(data, list): data = data[0]
        annotations = data.get('annotations', []) if isinstance(data, dict) else data
        if not annotations: return None

        h, w = None, None
        if isinstance(data, dict): h, w = data.get('height'), data.get('width')
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
# 🔥 4. 并行 Worker (双卡分配)
# ==================================================================================================
def process_chunk(worker_id, image_files_chunk, args):
    os.environ["OMP_NUM_THREADS"] = "1"
    
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id % max(1, num_gpus)
    device = torch.device(f'cuda:{gpu_id}')
    args.device = device
    
    fine_tuned_ckpt = args.checkpoint
    args.checkpoint = None 
    vanilla_sam = sam_model_registry[args.model_type](args)
    args.checkpoint = fine_tuned_ckpt
    
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder, clip_model_name=args.clip_model,
        num_organs=args.num_organs, num_heads=args.num_heads, sg_epsilon=args.sg_epsilon, sg_iters=args.sg_iters,
        use_pnurl=args.use_pnurl, use_coop=args.use_coop, use_sgot=args.use_sgot, use_asr=args.use_asr
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    state_dict = resize_pos_embed(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    chunk_metrics = defaultdict(list)
    pbar = tqdm(image_files_chunk, desc=f"Worker {worker_id} (GPU {gpu_id})", position=worker_id, leave=False)
    
    for img_path in pbar:
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
            except: pass
        
        prob, hv, dynamic_min_size = sliding_window_inference(model, image_rgb, organ_id, args, patch_size=256, overlap=0.8)
        
        pred_mask = hover_post_process(prob, hv, args.prob_thresh, args.marker_thresh, min_marker_size=dynamic_min_size)
        
        if pred_mask.max() == 0:
            fallback_mask = prob > args.prob_thresh
            fallback_mask = binary_fill_holes(fallback_mask)
            pred_mask = skimage_label(fallback_mask).astype(np.int32)
            try:
                pred_mask = remove_small_objects(pred_mask, min_size=15)
            except TypeError:
                pred_mask = remove_small_objects(pred_mask, max_size=15)
        
        gt_mask = load_filtered_gt(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): chunk_metrics[k].append(v)
            
    return dict(chunk_metrics)

def parse_args():
    parser = argparse.ArgumentParser(description="MP-SAM Inference & Testing")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--num_organs", type=int, default=21)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--sg_epsilon", type=float, default=0.05)
    parser.add_argument("--sg_iters", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument("--use_pnurl", action='store_true', default=False)
    parser.add_argument("--use_coop", action='store_true', default=False)
    parser.add_argument("--use_sgot", action='store_true', default=False)
    parser.add_argument("--use_asr", action='store_true', default=False)
    
    parser.add_argument("--prob_thresh", type=float, default=0.40)
    parser.add_argument("--marker_thresh", type=float, default=0.45)
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    
    return parser.parse_args()

def main(args):
    mp.set_start_method('spawn', force=True)
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.lower().endswith(('.png', '.tif'))]
    
    num_gpus = torch.cuda.device_count()
    workers_per_gpu = 2
    num_workers = min(num_gpus * workers_per_gpu, len(image_files)) 
    
    chunk_size = math.ceil(len(image_files) / num_workers)
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]
    
    print(f"\n🚀 System Detected {num_gpus} GPUs. Launching {len(chunks)} parallel Workers!")
    print(f"🔥 Activating Fast Mathematical Reversion Pipeline...")
    
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append((i, chunk, args))
        
    all_metrics = defaultdict(list)
    
    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(process_chunk, tasks)
        
    for res in results:
        for k, v in res.items():
            all_metrics[k].extend(v)

    print("\n" + "🌟" * 15 + "\n📊 Final Accelerated Results:")
    for k, v in all_metrics.items(): 
        print(f"{k:>10}: {np.mean(v):.4f}")

if __name__ == '__main__':
    main(parse_args())