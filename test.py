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
from PIL import Image
import clip

# 后处理库
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, opening, disk
from scipy import ndimage
from skimage.measure import label

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

# HoVer-style HV post-process (watershed from HV gradients)
def hover_post_process(prob_map, hv_map, prob_thresh=0.45, marker_thresh=0.4, min_marker_size=10):
    """
    prob_map: [H, W] float, nucleus probability
    hv_map:   [2, H, W] float, (V,H) distances (0: V, 1: H)
    """
    mask = prob_map > prob_thresh
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    # Sobel 梯度幅值，使用绝对阈值而非按 max 归一化
    sobel_h = cv2.Sobel(h_map, cv2.CV_32F, 1, 0, ksize=5)
    sobel_v = cv2.Sobel(v_map, cv2.CV_32F, 0, 1, ksize=5)
    sobel_mag = np.sqrt(sobel_h * sobel_h + sobel_v * sobel_v)

    grad_thr = 0.5
    marker_map = (sobel_mag < grad_thr) & (prob_map > prob_thresh)
    marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    markers = label(marker_map).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)

# 器官 ID 配置
ID_TO_ORGAN = {
    0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder", 3: "Breast", 
    4: "Cervix", 5: "Colon", 6: "Esophagus", 7: "HeadNeck", 
    8: "Kidney", 9: "Liver", 10: "Lung", 11: "Ovarian", 
    12: "Pancreatic", 13: "Prostate", 14: "Skin", 15: "Stomach", 
    16: "Testis", 17: "Thyroid", 18: "Uterus",
    19: "Brain", 20: "Generic"
}
ORGAN_TO_ID = {v.lower().replace("-", "_"): k for k, v in ID_TO_ORGAN.items()}
ORGAN_TO_ID.update({"bile_duct": 1, "head_neck": 7, "adrenal gland": 0})

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

class OrganPredictor:
    def __init__(self, device):
        self.device = device
        print("🧠 Loading CLIP for Organ Diagnosis...")
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        self.valid_ids = sorted(list(ID_TO_ORGAN.keys())) 
        self.organs = [ID_TO_ORGAN[i] for i in self.valid_ids]
        self.templates = [f"A histology image of {org} tissue." for org in self.organs]
        with torch.no_grad():
            text_inputs = clip.tokenize(self.templates).to(device)
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_cv2):
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        image_input = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
        return self.organs[indices.item()], self.valid_ids[indices.item()], values.item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="universal_test") 
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/PanNuke/test") 
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json", help="Path to global attributes JSON")
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--use_watershed", action='store_true', default=True)
    parser.add_argument("--use_hover_watershed", action='store_true', default=False, help="Use HoVer HV-gradient watershed post-process")
    parser.add_argument("--encoder_adapter", action='store_true')
    return parser.parse_args()

# 🔥 高斯权重图缓存，避免每次推理重复计算
GAUSSIAN_MAP_CACHE = {}

def get_gaussian_map(patch_size):
    """越靠近 Patch 中心的预测越可信，越靠近边缘的权重越低，消除拼接缝隙"""
    if patch_size in GAUSSIAN_MAP_CACHE:
        return GAUSSIAN_MAP_CACHE[patch_size]
    center = patch_size / 2.0
    sigma = patch_size / 4.0
    y_grid, x_grid = np.ogrid[-center:patch_size - center, -center:patch_size - center]
    g = np.exp(-(x_grid * x_grid + y_grid * y_grid) / (2 * sigma * sigma)).astype(np.float32)
    GAUSSIAN_MAP_CACHE[patch_size] = g
    return g

# 🔥 神级后处理：高斯平滑 + 动态接收自适应参数
def post_process_watershed(prob_map, threshold=0.45, min_size=30, min_distance=5):
    # sigma=0.3 保留边缘，平滑内部
    prob_map_smoothed = ndimage.gaussian_filter(prob_map, sigma=0.3) 
    binary_mask = prob_map_smoothed > threshold
    if not np.any(binary_mask): return np.zeros_like(binary_mask, dtype=np.int32)
    
    # 🔥 OpenCV 优化的距离变换，比 scipy 快数倍
    distance = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
    local_maxi = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    
    if len(local_maxi) == 0:
        return label(binary_mask).astype(np.int32)
        
    local_maxi_mask = np.zeros_like(prob_map, dtype=bool)
    local_maxi_mask[tuple(local_maxi.T)] = True
    markers = label(local_maxi_mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    
    if min_size > 0:
        labels = remove_small_objects(labels, min_size=min_size)
        labels = label(labels > 0)
    return labels.astype(np.int32)

def sliding_window_inference(model, image, device, patch_size, image_size, stride, 
                             text_prompt, organ_id, attribute_text=None, filename=None):
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad = image_pad.shape[:2]
    
    prob_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    hv_map_full = np.zeros((2, h_pad, w_pad), dtype=np.float32)  # (V,H)
    count_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    real_stride = min(stride, patch_size)
    y_steps = list(range(0, h_pad - patch_size + 1, real_stride))
    if (h_pad - patch_size) % real_stride != 0: y_steps.append(h_pad - patch_size)
    x_steps = list(range(0, w_pad - patch_size + 1, real_stride))
    if (w_pad - patch_size) % real_stride != 0: x_steps.append(w_pad - patch_size)
    
    if attribute_text is None: attribute_text = ""

    # 🔥 高斯权重图：中心可信度高，边缘权重低，消除拼接缝隙
    weight_map = get_gaussian_map(patch_size)

    model.eval()
    for y in y_steps:
        for x in x_steps:
            patch = image_pad[y:y+patch_size, x:x+patch_size, :]
            if patch_size != image_size:
                patch_input = cv2.resize(patch, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            else:
                patch_input = patch
            
            img_tensor = torch.from_numpy(patch_input).permute(2, 0, 1).float().to(device)
            
            input_sample = [{
                'image': img_tensor,
                'original_size': (image_size, image_size), 
                'text_prompt': text_prompt, 
                'organ_id': organ_id, 
                'attribute_text': attribute_text 
            }]
            
            # 🔥 AMP + Inference Mode：5090 满血推理
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_sample, multimask_output=True)
                out = outputs[0]
                best_idx = torch.argmax(out['iou_predictions']).item()
                logits = out['masks'][0, best_idx, :, :]
                hv_logits = out.get('hv_logits', None)  # [1,2,h,w] or None
                if hv_logits is not None:
                    hv_logits = torch.tanh(hv_logits)
            
            logits = logits.unsqueeze(0).unsqueeze(0) 
            if patch_size != image_size:
                logits_orig = F.interpolate(logits, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            else:
                logits_orig = logits
            
            prob_patch = torch.sigmoid(logits_orig).squeeze().cpu().float().numpy()

            hv_patch = None
            if hv_logits is not None:
                # hv at feature resolution -> patch_size
                if hv_logits.dim() == 3:
                    hv_logits = hv_logits.unsqueeze(0)
                hv_up = F.interpolate(hv_logits.float(), size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                hv_patch = hv_up.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [2, ps, ps]

            # 🔥 高斯加权拼接：消除 Patch 边缘突变
            prob_map_full[y:y+patch_size, x:x+patch_size] += prob_patch * weight_map
            if hv_patch is not None:
                hv_map_full[:, y:y+patch_size, x:x+patch_size] += hv_patch * weight_map[None, ...]
            count_map_full[y:y+patch_size, x:x+patch_size] += weight_map
                
    count_map_full[count_map_full == 0] = 1.0
    avg_prob = prob_map_full / count_map_full
    avg_hv = hv_map_full / count_map_full[None, ...]
    return avg_prob[:h, :w], avg_hv[:, :h, :w]

def get_organ_from_json(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path): json_path = img_path.rsplit('.', 1)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            organ = data.get('organ_type', data.get('metadata', {}).get('organ_type'))
            if organ:
                key = organ.lower().replace("-", "_")
                if key in ORGAN_TO_ID: return ID_TO_ORGAN[ORGAN_TO_ID[key]], ORGAN_TO_ID[key]
                for k_name, v_id in ORGAN_TO_ID.items():
                    if k_name in key or key in k_name: return ID_TO_ORGAN[v_id], v_id
        except: pass
    return None, None

# 🔥 完美兼容 Polygon(PanNuke) 和 RLE(MoNuSeg) 的解析
def load_filtered_gt(img_path, attr_data, target_tag=None):
    mask_path_png = img_path.replace('.tif', '_mask.png').replace('.png', '_mask.png')
    if os.path.exists(mask_path_png):
        return cv2.imread(mask_path_png, cv2.IMREAD_UNCHANGED).astype(np.int32)
        
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        possible_json = img_path.rsplit('.', 1)[0] + ".json"
        if os.path.exists(possible_json): json_path = possible_json
        else: return None

    try:
        with open(json_path, 'r') as f: data = json.load(f)
        anns = data.get('annotations', []) if isinstance(data, dict) else data
        if not anns: return None
        
        temp_img = cv2.imread(img_path)
        if temp_img is None: return None
        h, w = temp_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        
        for idx, ann in enumerate(anns):
            seg = ann.get('segmentation', [])
            inst_id = idx + 1
            
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [pts], inst_id)
            elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                try:
                    from pycocotools import mask as coco_mask
                    binary_mask = coco_mask.decode(seg)
                    mask[binary_mask > 0] = inst_id
                except ImportError:
                    pass
        return mask
    except: return None

def main(args):
    vanilla_sam = sam_model_registry[args.model_type](args)
    vanilla_sam.image_encoder.img_size = args.image_size 

    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512, embed_dim=256, num_organs=21 
    ).to(args.device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = resize_pos_embed(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"❌ Checkpoint not found")
        return

    organ_predictor = OrganPredictor(args.device)
    
    # 🔥🔥🔥 1. 加载全局知识库与统计常数 (Oracle Knowledge) 🔥🔥🔥
    global_knowledge = {}
    dataset_stats = {}
    if os.path.exists(args.knowledge_path):
        print(f"📖 Loading Global Knowledge Base from {args.knowledge_path} ...")
        try:
            with open(args.knowledge_path, 'r') as f:
                global_knowledge = json.load(f)
            
            if "__meta__" in global_knowledge and "stats" in global_knowledge["__meta__"]:
                dataset_stats = global_knowledge["__meta__"]["stats"]
                print(f"📊 Extracted Global Stats: {dataset_stats}")
            else:
                print("⚠️ No __meta__.stats found. Will fallback to Universal Baseline.")
                
            print(f"✅ Loaded {len(global_knowledge)} knowledge entries.")
        except Exception as e:
            print(f"⚠️ Failed to parse knowledge base: {e}")
    else:
        print(f"⚠️ Warning: Knowledge Base not found at {args.knowledge_path}. Using Universal Baseline.")

    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    all_metrics = defaultdict(list)
    save_dir = os.path.join(args.work_dir, args.run_name, "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    print('*'*60)
    print(f"🚀 SCIENTIFIC ADAPTIVE INFERENCE: Window={args.patch_size} -> Model={args.image_size}")
    print('*'*60)

    pbar = tqdm(image_files)
    for img_path in pbar:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_organ, gt_id = get_organ_from_json(img_path)
        if gt_organ:
            pred_organ = gt_organ
            current_organ_id = gt_id
            log_msg = f"📂 JSON: {pred_organ}"
        else:
            pred_organ, pred_id, conf = organ_predictor.predict(image)
            current_organ_id = pred_id
            log_msg = f"🧠 AI: {pred_organ} ({conf:.1%})"

        class_prompt_text = args.text_prompt if args.text_prompt else f"{pred_organ} cell nuclei"
        if args.text_prompt: current_organ_id = 20

        # 🔥🔥🔥 2. 查表获取当前图像的 Oracle 物理属性标签 🔥🔥🔥
        gt_size_tag, gt_density_tag = "DEFAULT", "DEFAULT"
        
        rel_path = os.path.relpath(img_path, os.path.dirname(os.path.dirname(args.data_path)))
        # 尝试多种可能的 Key 匹配方式
        match_keys = [rel_path, f"test/{filename}", filename, os.path.join("test", filename)]
        
        for key in match_keys:
            if key in global_knowledge:
                stats = global_knowledge[key].get("visual_stats", {})
                gt_size_tag = stats.get("size", "DEFAULT")
                gt_density_tag = stats.get("density", "DEFAULT")
                break
                
        # 🔥🔥🔥 3. 数学级自适应参数推导 (Scientific Adaptive Mapping) 🔥🔥🔥
        UNIVERSAL_BASE_SIZE = 30
        UNIVERSAL_BASE_DIST = 5
        dynamic_min_size = UNIVERSAL_BASE_SIZE
        dynamic_min_dist = UNIVERSAL_BASE_DIST

        if dataset_stats:
            # 轨道一：Data-Driven Prior Injection
            th_size_small = dataset_stats.get("th_size_small", 250)
            
            # Size 逻辑
            if gt_size_tag == "large-sized":
                dynamic_min_size = int(th_size_small * 0.25)
            elif gt_size_tag == "small-sized":
                dynamic_min_size = max(15, int(th_size_small * 0.08))
            else:
                dynamic_min_size = max(30, int(th_size_small * 0.12))
                
            # Density 逻辑
            if gt_density_tag == "densely distributed":
                dynamic_min_dist = 3
            elif gt_density_tag == "sparsely distributed":
                dynamic_min_dist = 8
            else:
                dynamic_min_dist = 5
        else:
            # 轨道二：Fallback 降级策略 (基于通用基线的缩放)
            if gt_size_tag == "large-sized": dynamic_min_size = int(UNIVERSAL_BASE_SIZE * 2.0)
            elif gt_size_tag == "small-sized": dynamic_min_size = int(UNIVERSAL_BASE_SIZE * 0.5)
            
            if gt_density_tag == "densely distributed": dynamic_min_dist = max(3, UNIVERSAL_BASE_DIST - 2)
            elif gt_density_tag == "sparsely distributed": dynamic_min_dist = UNIVERSAL_BASE_DIST + 3

        pbar.write(f"🖼️  {filename} | Oracle: [{gt_size_tag}, {gt_density_tag}] ➔ Adaptive: min_size={dynamic_min_size}, min_dist={dynamic_min_dist}")

        # TTA 推理 (同时返回 prob + hv)
        prob_orig, hv_orig = sliding_window_inference(
            model, image_rgb, args.device, args.patch_size, args.image_size, args.stride,
            class_prompt_text, current_organ_id, "", filename
        )

        # Horizontal flip: spatial flip + H 通道取反
        img_h = cv2.flip(image_rgb, 1)
        prob_h, hv_h = sliding_window_inference(
            model, img_h, args.device, args.patch_size, args.image_size, args.stride,
            class_prompt_text, current_organ_id, "", filename
        )
        prob_h = cv2.flip(prob_h, 1)
        hv_h = np.flip(hv_h, axis=2).copy()
        hv_h[1] = -hv_h[1]

        # Vertical flip: spatial flip + V 通道取反
        img_v = cv2.flip(image_rgb, 0)
        prob_v, hv_v = sliding_window_inference(
            model, img_v, args.device, args.patch_size, args.image_size, args.stride,
            class_prompt_text, current_organ_id, "", filename
        )
        prob_v = cv2.flip(prob_v, 0)
        hv_v = np.flip(hv_v, axis=1).copy()
        hv_v[0] = -hv_v[0]
        
        target_shape = prob_orig.shape
        def safe_crop2d(arr, shape): return arr[: shape[0], : shape[1]]
        def safe_crop3d(arr, shape): return arr[:, : shape[0], : shape[1]]
        prob_h = safe_crop2d(prob_h, target_shape)
        prob_v = safe_crop2d(prob_v, target_shape)
        hv_h = safe_crop3d(hv_h, target_shape)
        hv_v = safe_crop3d(hv_v, target_shape)

        pred_prob = (prob_orig + prob_h + prob_v) / 3.0
        pred_hv = (hv_orig + hv_h + hv_v) / 3.0

        # 🔥 4. 后处理：可控消融 (HoVer-Watershed vs 传统距离分水岭)
        if args.use_watershed:
            if args.use_hover_watershed:
                pred_mask = hover_post_process(
                    pred_prob,
                    pred_hv,
                    prob_thresh=0.45,
                    marker_thresh=0.4,
                    min_marker_size=max(10, int(dynamic_min_size * 0.25)),
                )
                # 如果 marker 太少导致失败，fallback 到传统分水岭
                if pred_mask.max() == 0:
                    pred_mask = post_process_watershed(
                        pred_prob,
                        threshold=0.45,
                        min_size=dynamic_min_size,
                        min_distance=dynamic_min_dist,
                    )
            else:
                pred_mask = post_process_watershed(
                    pred_prob,
                    threshold=0.45,
                    min_size=dynamic_min_size,
                    min_distance=dynamic_min_dist,
                )
        else:
            pred_mask = label(pred_prob > 0.45).astype(np.int32)
        
        gt_mask = load_filtered_gt(img_path, {}, target_tag="Auto_Organ")
        
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.int32), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)
        
        if args.save_pred:
            vis = image.copy()
            cnts_pred, _ = cv2.findContours((pred_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts_pred, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, filename), vis)

    print("\n" + "="*40)
    print(f"📊 Final Results:")
    for k, v in all_metrics.items():
        if len(v) > 0: print(f"  {k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)