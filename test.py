import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from metrics import SegMetrics
import torch.nn.functional as F

# 后处理库
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, opening, disk
from scipy import ndimage

# GT 解析
try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

# === 🔥 [核心] MoNuSeg 测试集器官映射表 (Hardcoded) ===
ORGAN_MAP = {
    "TCGA-2Z-A9J9": "Prostate", "TCGA-44-2665": "Kidney", 
    "TCGA-69-7764": "Kidney", "TCGA-A6-2675": "Colorectal",
    "TCGA-A6-2680": "Colorectal", "TCGA-A6-5662": "Lung",
    "TCGA-AC-A2FO": "Lung", "TCGA-AO-A0J2": "Breast",
    "TCGA-CU-A0YN": "Bladder", "TCGA-EJ-A46H": "Prostate",
    "TCGA-FG-A4MU": "Prostate", "TCGA-GL-A4EM": "Kidney",
    "TCGA-HC-7209": "Lung", "TCGA-HT-8564": "Brain"
}

def get_organ_prompt(filename):
    """根据文件名自动返回最精准的 Organ Prompt"""
    for key, val in ORGAN_MAP.items():
        if key in filename:
            return f"{val} cell nuclei"
    return "Cell nuclei"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-rich") 
    
    # 🔥 [关键设置]
    # patch_size: 滑动窗口大小 (从原图切多大)，建议 256
    parser.add_argument("--patch_size", type=int, default=256, help="Sliding window crop size")
    # image_size: 模型输入大小 (必须与训练时一致，即 1024)
    parser.add_argument("--image_size", type=int, default=1024, help="Model input resolution")
    
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B/test") 
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--use_watershed", action='store_true', default=True)
    return parser.parse_args()

def load_gt_mask(img_path):
    """全能型 GT 加载函数"""
    import json
    import os
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        pass

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    dir_name = os.path.dirname(img_path)
    
    # 读取图片尺寸
    temp_img = cv2.imread(img_path)
    if temp_img is None: return None
    h, w = temp_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. 尝试 JSON
    json_path = os.path.splitext(img_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            anns = data.get('annotations', [])
            if not anns and isinstance(data, list): anns = data
            found_ann = False
            for ann in anns:
                if 'segmentation' in ann:
                    found_ann = True
                    seg = ann['segmentation']
                    if isinstance(seg, dict) and 'counts' in seg: 
                        rle_mask = coco_mask.decode(seg)
                        mask[rle_mask > 0] = 1
                    elif isinstance(seg, list):
                        for poly in seg:
                            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                            cv2.fillPoly(mask, [pts], 1)
            if found_ann: return mask
        except Exception as e:
            print(f"⚠️ Error parsing JSON {json_path}: {e}")

    # 2. 尝试 PNG/TIF
    candidates = [
        os.path.join(dir_name, base_name + ".png"),
        os.path.join(dir_name, base_name + ".tif"),
        os.path.join(dir_name, base_name + "_mask.png"),
        img_path.replace("Images", "Labels").replace(".tif", ".png"),
        img_path.replace("test", "test/Labels").replace(".tif", ".png"),
        img_path.replace(".tif", ".png")
    ]
    
    for p in candidates:
        if os.path.exists(p):
            m = cv2.imread(p, 0)
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 0).astype(np.uint8)
    return None

def postprocess_watershed(prob_map, thresh=0.35, min_distance=3):
    """分水岭后处理"""
    binary_mask = prob_map > thresh
    binary_mask = opening(binary_mask, disk(1))
    distance = ndimage.distance_transform_edt(binary_mask)
    # min_distance 越小，切分越细致
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    final_mask = remove_small_objects(labels, min_size=15)
    return (final_mask > 0).astype(np.uint8)

def sliding_window_inference(model, image, device, patch_size=256, image_size=1024, stride=128, text_prompt="Cell nuclei", filename=None):
    """
    🔥 显微镜模式推理逻辑：
    Input(256) -> Resize(1024) -> Model(1024) -> Output(1024) -> Resize(256) -> Stitch
    """
    h, w = image.shape[:2]
    # Padding
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad = image_pad.shape[:2]
    
    prob_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    y_steps = list(range(0, h_pad - patch_size + 1, stride))
    if (h_pad - patch_size) % stride != 0: y_steps.append(h_pad - patch_size)
    x_steps = list(range(0, w_pad - patch_size + 1, stride))
    if (w_pad - patch_size) % stride != 0: x_steps.append(w_pad - patch_size)
    
    model.eval()
    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                # 1. 切片 (256x256)
                patch = image_pad[y:y+patch_size, x:x+patch_size, :]
                
                # 2. 🔥 放大 (Resize 256 -> 1024)
                # 这里的 interpolation 必须用线性插值，保证图像平滑
                patch_large = cv2.resize(patch, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                
                img_tensor = torch.from_numpy(patch_large).permute(2, 0, 1).float().to(device)
                
                input_sample = [{
                    'image': img_tensor,
                    # 告诉模型：现在的图是 1024 的。SAM 会输出 1024 的 Mask。
                    'original_size': (image_size, image_size), 
                    'text_prompt': text_prompt
                }]
                
                # 3. 推理 (输出也是 1024x1024)
                outputs = model(input_sample, multimask_output=True)
                out = outputs[0]
                
                # 可选：保存 Heatmap 调试
                if filename and y==0 and x==0:
                    heatmap = out['heatmap_logits']
                    fg_map = torch.sigmoid(heatmap[0, 0]).cpu().numpy()
                    fg_map_vis = (fg_map * 255).astype(np.uint8)
                    cv2.imwrite(f"workdir/debug_heatmap_{filename}.png", fg_map_vis)
                
                # 获取 Mask logits (1024x1024)
                scores = out['iou_predictions'].squeeze()
                best_idx = torch.argmax(scores).item()
                logits_large = out['masks'][0, best_idx, :, :] 
                
                # 4. 🔥 缩小 (Resize 1024 -> 256)
                # 把高分辨率的预测结果还原回原图切片尺寸
                logits_large = logits_large.unsqueeze(0).unsqueeze(0) # [1, 1, 1024, 1024]
                logits_small = F.interpolate(logits_large, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                prob_small = torch.sigmoid(logits_small).squeeze().cpu().numpy() # [256, 256]
                
                # 5. 累加
                prob_map_full[y:y+patch_size, x:x+patch_size] += prob_small
                count_map_full[y:y+patch_size, x:x+patch_size] += 1.0
                
    count_map_full[count_map_full == 0] = 1.0
    avg_prob = prob_map_full / count_map_full
    return avg_prob[:h, :w]

def main(args):
    print('*'*60)
    print(f"🚀 Running Inference: {args.run_name}")
    print(f"   Input Patch (Crop): {args.patch_size} -> Model Input (Zoom): {args.image_size}")
    print(f"   Strategy: Microscope Mode (Zoom-in & Stitch)")
    print(f"   Watershed: {args.use_watershed}")
    print('*'*60)

    # ❌ 绝对不要写 args.image_size = args.patch_size，否则前功尽弃

    # Model
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    
    # Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"🔄 Loading checkpoint: {args.checkpoint}")
        # 这里已经是 1024 的权重了，不需要再做 resize_pos_embed
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint Loaded Successfully.")
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    # Data Scan
    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    print(f"📂 Found {len(image_files)} test images.")
    all_metrics = defaultdict(list)
    save_dir = os.path.join(args.work_dir, args.run_name, "viz_final")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    # Inference Loop
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取 Prompt
        prompt_text = get_organ_prompt(filename)
        # prompt_text = "A photo of a cat" # 👈 测试用：如果想测鲁棒性，取消这行注释
        
        # 推理
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size, # 256
            image_size=args.image_size, # 1024 (关键！)
            stride=args.stride,
            text_prompt=prompt_text,
            filename=filename
        )
        
        # 后处理
        if args.use_watershed:
            # 这里的参数可以根据 1024 训练后的表现微调
            # 如果发现粘连还是多，把 min_distance 改成 4 或 5
            pred_mask = postprocess_watershed(pred_prob, thresh=0.4, min_distance=3)
        else:
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # 指标计算
        gt_mask = load_gt_mask(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                all_metrics[k].append(v)
        
        # 可视化
        if args.save_pred:
            vis = image.copy()
            cnts, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            cv2.putText(vis, f"{prompt_text[:15]}...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(save_dir, filename.replace('.tif','.jpg')), vis)

    print("\n" + "="*40)
    print(f"📊 Final Results (Microscope Mode):")
    for k, v in all_metrics.items():
        if len(v) > 0:
            print(f"{k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)