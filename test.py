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
# 只要文件名包含 Key，就自动使用对应的 Prompt
ORGAN_MAP = {
    "TCGA-2Z-A9J9": "Prostate", "TCGA-44-2665": "Kidney", 
    "TCGA-69-7764": "Kidney", "TCGA-A6-2675": "Colorectal",
    "TCGA-A6-2680": "Colorectal", "TCGA-A6-5662": "Lung",
    "TCGA-AC-A2FO": "Lung", "TCGA-AO-A0J2": "Breast",
    "TCGA-CU-A0YN": "Bladder", "TCGA-EJ-A46H": "Prostate",
    "TCGA-FG-A4MU": "Prostate", "TCGA-GL-A4EM": "Kidney",
    "TCGA-HC-7209": "Lung", "TCGA-HT-8564": "Brain"
}

def get_smart_prompt(filename):
    """根据文件名自动返回最精准的 Organ Prompt"""
    organ = "tissue"
    for key, val in ORGAN_MAP.items():
        if key in filename:
            organ = val
            break
            
    # 构造 Rich Text
    # 这里的形容词是我们根据病理经验加的，强迫模型关注形态
    prompt = f"Deep purple {organ} cell nuclei, densely packed, H&E stained"
    return prompt, organ

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-rich") 
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B/test") 
    parser.add_argument("--prompt_path", type=str, default=None, help="Deprecated. We use hardcoded map.")
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--use_watershed", action='store_true', default=True)
    return parser.parse_args()

def load_gt_mask(img_path):
    """
    全能型 GT 加载函数：支持 JSON, PNG, _mask, Labels 目录等多种变体
    """
    import json # <--- 🔥 [关键修复] 强制在函数内引入 json 模块
    import os
    import cv2
    import numpy as np
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        pass

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    dir_name = os.path.dirname(img_path)
    
    # 1. 尝试同名 SA-1B JSON
    json_path = os.path.splitext(img_path)[0] + ".json"
    
    # 尝试读取图片获取尺寸
    temp_img = cv2.imread(img_path)
    if temp_img is None: 
        # print(f"⚠️ Image not found: {img_path}")
        return None
    h, w = temp_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # === 策略 A: 读取 JSON ===
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f) # 现在这里绝对不会报 name 'json' is not defined 了
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
            if found_ann: 
                # print(f"✅ Loaded GT from JSON: {json_path}")
                return mask
        except Exception as e:
            print(f"⚠️ Error parsing JSON {json_path}: {e}")

    # === 策略 B: 读取 PNG/TIF Mask ===
    # MoNuSeg 常见的 Mask 存放位置
    candidates = [
        # 1. 同目录下同名
        os.path.join(dir_name, base_name + ".png"),
        os.path.join(dir_name, base_name + ".tif"),
        # 2. 同目录下加后缀
        os.path.join(dir_name, base_name + "_mask.png"),
        os.path.join(dir_name, base_name + "_label.png"),
        # 3. 父目录下的 Labels/BinaryMask 文件夹
        img_path.replace("Images", "Labels").replace(".tif", ".png"),
        img_path.replace("test", "test/Labels").replace(".tif", ".png"),
        # 4. 暴力替换扩展名
        img_path.replace(".tif", ".png"),
        img_path.replace(".tif", "_mask.png")
    ]
    
    for p in candidates:
        if os.path.exists(p):
            m = cv2.imread(p, 0) # 读取灰度
            if m is not None:
                # print(f"✅ Loaded GT from PNG: {p}")
                # 确保尺寸一致
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 0).astype(np.uint8)
    
    # print(f"❌ No GT found for {base_name}")
    return None

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, opening, disk
from scipy import ndimage

def postprocess_watershed(prob_map, thresh=0.35, min_distance=3):
    """
    适配 TextSam 的距离变换分水岭
    """
    # 1. 激进的二值化：只要有 35% 把握就认为是前景，先召回再切分
    binary_mask = prob_map > thresh
    binary_mask = opening(binary_mask, disk(1))
    # 2. 稍微腐蚀一点点，断开极其细微的粘连
    # binary_mask = opening(binary_mask, disk(1)) 
    
    # 3. 计算距离场：越靠近细胞中心，值越大
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # 4. 寻找山峰 (种子点)
    # min_distance=3 是关键！允许两个细胞核中心距离只有3像素
    # 这能解决 MoNuSeg 中那种极其拥挤的细胞粘连
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    # 5. 执行分水岭：让水从 markers 开始流，填满 binary_mask
    labels = watershed(-distance, markers, mask=binary_mask)
    
    # 6. 去除噪点
    final_mask = remove_small_objects(labels, min_size=15)
    
    return (final_mask > 0).astype(np.uint8)

def sliding_window_inference(model, image, device, patch_size=256, stride=128, text_prompt="Cell nuclei"):
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
                patch = image_pad[y:y+patch_size, x:x+patch_size, :]
                img_tensor = torch.from_numpy(patch).permute(2, 0, 1).float().to(device)
                
                input_sample = [{
                    'image': img_tensor,
                    'original_size': (patch_size, patch_size),
                    'text_prompt': text_prompt
                }]
                
                outputs = model(input_sample, multimask_output=True)
                out = outputs[0]
                
                scores = out['iou_predictions'].squeeze()
                best_idx = torch.argmax(scores).item()
                logits = out['masks'][0, best_idx, :, :]
                prob = torch.sigmoid(logits).cpu().numpy()
                
                prob_map_full[y:y+patch_size, x:x+patch_size] += prob
                count_map_full[y:y+patch_size, x:x+patch_size] += 1.0
                
    count_map_full[count_map_full == 0] = 1.0
    avg_prob = prob_map_full / count_map_full
    return avg_prob[:h, :w]

def main(args):
    print('*'*60)
    print(f"🚀 Running Inference: {args.run_name}")
    print(f"   Patch: {args.patch_size} | Watershed: {args.use_watershed}")
    print(f"   Prompt Strategy: Hardcoded Organ Mapping (Robust)")
    print('*'*60)

    args.image_size = args.patch_size 
    args.sam_checkpoint = None 

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
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ Checkpoint Loaded.")
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
        
        # 🔥 [关键] 强制使用 Organ Prompt
        # prompt_text, organ_name = get_smart_prompt(filename)
        prompt_text = "Cell nuclei"  # <--- 强制回退到通用提示
        organ_name = "Generic"
        # 打印出来确认一下！
        # tqdm.write(f"Processing {filename} -> Organ: {organ_name} | Prompt: {prompt_text[:30]}...")
        
        # Inference
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size, 
            stride=args.stride,
            text_prompt=prompt_text # 传入精准文本
        )
        
        # Post-process
        if args.use_watershed:
            # 调整参数：thresh=0.4 (捕获更多), min_distance=5 (切得更细)
            pred_mask = postprocess_watershed(pred_prob, thresh=0.4, min_distance=5)
        else:
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # Metrics
        gt_mask = load_gt_mask(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                all_metrics[k].append(v)
        
        # Viz
        if args.save_pred:
            vis = image.copy()
            cnts, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            cv2.putText(vis, f"{organ_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(save_dir, filename.replace('.tif','.jpg')), vis)

    print("\n" + "="*40)
    print(f"📊 Final Results (Watershed+, RichPrompt):")
    for k, v in all_metrics.items():
        if len(v) > 0:
            print(f"{k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    main(parse_args())