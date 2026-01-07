import argparse
import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from collections import defaultdict
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from metrics import SegMetrics
import torch.nn.functional as F

# 后处理
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, opening, disk
from scipy import ndimage
# 形态学
from skimage.measure import label, regionprops

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

# === MoNuSeg 器官映射表 ===
ORGAN_MAP = {
    "TCGA-2Z-A9J9": "Prostate", "TCGA-44-2665": "Kidney", 
    "TCGA-69-7764": "Kidney", "TCGA-A6-2675": "Colorectal",
    "TCGA-A6-2680": "Colorectal", "TCGA-A6-5662": "Lung",
    "TCGA-AC-A2FO": "Lung", "TCGA-AO-A0J2": "Breast",
    "TCGA-CU-A0YN": "Bladder", "TCGA-EJ-A46H": "Prostate",
    "TCGA-FG-A4MU": "Prostate", "TCGA-GL-A4EM": "Kidney",
    "TCGA-HC-7209": "Lung", "TCGA-HT-8564": "Brain"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-dynamic-final") 
    
    parser.add_argument("--text_prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--test_attr_path", type=str, default="data/MoNuSeg_SA1B/test_dynamic_attributes.json")
    
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/MoNuSeg_SA1B/test") 
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--use_watershed", action='store_true', default=True)
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    
    return parser.parse_args()

def analyze_predictions(pred_mask):
    """
    计算预测 Mask 中所有实例的形态学统计数据
    """
    labeled = label(pred_mask)
    regions = regionprops(labeled)
    
    if not regions:
        return 0.0, 0.0, 0
    
    areas = []
    roundnesses = []
    
    for r in regions:
        area = r.area
        perimeter = r.perimeter
        if perimeter == 0: roundness = 0
        else: roundness = (4 * np.pi * area) / (perimeter ** 2)
        
        areas.append(area)
        roundnesses.append(roundness)
        
    return np.mean(areas), np.mean(roundnesses), len(regions)

def postprocess_watershed(prob_map, thresh=0.4, min_distance=3):
    binary_mask = prob_map > thresh
    binary_mask = opening(binary_mask, disk(1))
    distance = ndimage.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    final_mask = remove_small_objects(labels, min_size=15)
    return (final_mask > 0).astype(np.uint8)

# 🔥 [再次确认] 这里没有 print，并且 filename=None 参数已补上
def sliding_window_inference(model, image, device, patch_size, image_size, stride, text_prompt, filename=None):
    h, w = image.shape[:2]
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
                patch_large = cv2.resize(patch, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(patch_large).permute(2, 0, 1).float().to(device)
                
                input_sample = [{
                    'image': img_tensor,
                    'original_size': (image_size, image_size), 
                    'text_prompt': text_prompt 
                }]
                
                # 🔥 这里绝对没有 print
                outputs = model(input_sample, multimask_output=True)
                out = outputs[0]
                
                scores = out['iou_predictions'].squeeze()
                best_idx = torch.argmax(scores).item()
                logits_large = out['masks'][0, best_idx, :, :] 
                
                logits_large = logits_large.unsqueeze(0).unsqueeze(0)
                logits_small = F.interpolate(logits_large, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                prob_small = torch.sigmoid(logits_small).squeeze().cpu().numpy()
                
                prob_map_full[y:y+patch_size, x:x+patch_size] += prob_small
                count_map_full[y:y+patch_size, x:x+patch_size] += 1.0
                
    count_map_full[count_map_full == 0] = 1.0
    avg_prob = prob_map_full / count_map_full
    return avg_prob[:h, :w]

def load_filtered_gt(img_path, attr_data, target_tag=None):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    filename_key = os.path.basename(img_path) 
    if filename_key not in attr_data:
        filename_key = filename_key.replace(".tif", ".json").replace(".png", ".json")
        if filename_key not in attr_data:
            for k in attr_data.keys():
                if base_name in k:
                    filename_key = k
                    break
    
    temp_img = cv2.imread(img_path)
    if temp_img is None: return None
    h, w = temp_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    instances = attr_data.get(filename_key, [])
    
    if target_tag is None or target_tag in ["Generic", "Cell nuclei", "Auto_Organ"]:
        valid_ids = set([inst['id'] for inst in instances])
    else:
        valid_ids = set()
        for inst in instances:
            tags = [t.lower() for t in inst.get('tags', [])]
            search_key = target_tag.split()[0].lower() 
            if search_key in tags:
                valid_ids.add(inst['id'])
    
    json_path = os.path.splitext(img_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            anns = data.get('annotations', [])
            if not anns and isinstance(data, list): anns = data
            
            for idx, ann in enumerate(anns):
                if idx in valid_ids and 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, dict) and 'counts' in seg: 
                        rle_mask = coco_mask.decode(seg)
                        mask[rle_mask > 0] = 1
                    elif isinstance(seg, list):
                        for poly in seg:
                            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                            cv2.fillPoly(mask, [pts], 1)
            return mask
        except: return None
    return None

def main(args):
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ).to(args.device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    attr_data = {}
    if os.path.exists(args.test_attr_path):
        print(f"📖 Loading Test Attributes from {args.test_attr_path}...")
        with open(args.test_attr_path, 'r') as f:
            content = json.load(f)
            attr_data = content.get("images", {})
    else:
        print(f"⚠️ Warning: Attribute file {args.test_attr_path} not found.")

    current_tag = "Generic"
    if args.text_prompt:
        lower_p = args.text_prompt.lower()
        if "small" in lower_p: current_tag = "Small"
        elif "large" in lower_p: current_tag = "Large"
        elif "round" in lower_p: current_tag = "Round"
        elif "cat" in lower_p: current_tag = "Negative"
    
    print(f"🎯 Evaluation Mode: {current_tag} (Filtered GT & Avg Area)")

    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    all_metrics = defaultdict(list)
    prompt_stats = {
        "avg_area": [],
        "avg_roundness": [],
        "count": []
    }
    
    prompt_tag = args.text_prompt.replace(" ", "_") if args.text_prompt else "Auto_Organ"
    save_dir = os.path.join(args.work_dir, args.run_name, f"viz_{prompt_tag}")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    print('*'*60)
    print(f"🚀 Running Inference: {args.run_name}")
    print(f"   Prompt Mode: {prompt_tag}")
    print('*'*60)

    # 🔥 使用 tqdm 迭代
    pbar = tqdm(image_files)
    for img_path in pbar:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if args.text_prompt:
            prompt_text = args.text_prompt
        else:
            for key, val in ORGAN_MAP.items():
                if key in filename:
                    prompt_text = f"{val} cell nuclei"
                    break
            else:
                prompt_text = "Cell nuclei"
        
        # 🔥 [关键] 只在这里打印一次，并且使用 pbar.write 防止进度条错乱
        pbar.write(f"🖼️  Processing: {filename} | 💬 Prompt: '{prompt_text}'")
        
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size,
            image_size=args.image_size,
            stride=args.stride,
            text_prompt=prompt_text, 
            filename=filename
        )
        
        # if args.use_watershed:
        if False:
            pred_mask = postprocess_watershed(pred_prob, thresh=0.4, min_distance=3)
        else:
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # 1. 无参考统计
        p_area, p_round, p_count = analyze_predictions(pred_mask)
        if p_count > 0:
            prompt_stats["avg_area"].append(p_area)
            prompt_stats["avg_roundness"].append(p_round)
            prompt_stats["count"].append(p_count)

        # 2. 动态 GT 过滤 (Dice)
        gt_mask = load_filtered_gt(img_path, attr_data, target_tag=current_tag)
        
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                all_metrics[k].append(v)
        
        if args.save_pred:
            vis = image.copy()
            cnts_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts_pred, -1, (0, 255, 0), 2)
            
            if gt_mask is not None:
                cnts_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts_gt, -1, (0, 0, 255), 1)

            cv2.putText(vis, f"{prompt_text} | Area:{int(p_area)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(save_dir, filename.replace('.tif','.jpg')), vis)

    print("\n" + "="*40)
    print(f"📊 Final Results ({prompt_tag} vs {current_tag}_GT):")
    for k, v in all_metrics.items():
        if len(v) > 0:
            print(f"  {k:>10}: {np.mean(v):.4f}")
            
    print("-" * 20)
    mean_area = np.mean(prompt_stats["avg_area"]) if prompt_stats["avg_area"] else 0
    mean_round = np.mean(prompt_stats["avg_roundness"]) if prompt_stats["avg_roundness"] else 0
    
    print(f"🧬 Morphological Stats (Reference-Free):")
    print(f"  Avg Pred Area : {mean_area:.2f} pixels")
    print(f"  Avg Roundness : {mean_round:.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)