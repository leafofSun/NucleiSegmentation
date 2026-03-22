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
import clip

# 后处理库
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

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
# 1. 核心后处理：官方 21x21 Sobel 版 HoVer-Watershed
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.5, marker_thresh=0.4, min_marker_size=10):
    mask = prob_map > prob_thresh
    if not np.any(mask): return np.zeros_like(mask, dtype=np.int32)

    h_dir = cv2.normalize(hv_map[1], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(hv_map[0], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    sobelh = 1 - cv2.normalize(cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobelv = 1 - cv2.normalize(cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = (overall - (1 - mask)).clip(0) 

    dist = -cv2.GaussianBlur((1.0 - overall) * mask, (3, 3), 0)
    marker = (overall >= marker_thresh) & mask
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    markers = label(marker)
    markers = remove_small_objects(markers, min_size=min_marker_size)

    return watershed(dist, markers=markers, mask=mask).astype(np.int32)

# ==================================================================================================
# 2. 8-fold TTA 批量推理 (GPU 物理坐标对齐版)
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, args):
    device = args.device
    input_size = (args.image_size, args.image_size)
    transforms = [(None, 0)]
    
    # transforms = [
    #     (None, 0), (1, 0), (0, 0), (-1, 0), 
    #     (None, 1), (1, 1), (0, 1), (-1, 1)  
    # ]
    
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
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in range(len(transforms)):
            input_sample = [{'image': batch_img[i], 'original_size': input_size, 'organ_id': organ_id, 'text_prompt': "Cell nuclei"}]
            out = model(input_sample, multimask_output=True)[0]
            best_idx = torch.argmax(out['iou_predictions']).item()
            
            prob = torch.sigmoid(out['masks'][0, best_idx]) 
            hv_raw = out.get('hv_logits')
            if hv_raw.dim() == 3: hv_raw = hv_raw.unsqueeze(0)
            hv = F.interpolate(hv_raw.float(), size=input_size, mode='bilinear', align_corners=False).squeeze(0)
            hv = torch.tanh(hv) 

            f_code, r_k = transforms[i]
            
            if r_k == 1:
                prob = torch.rot90(prob, k=-1, dims=[0, 1])
                hv = torch.rot90(hv, k=-1, dims=[1, 2])
                v_new, h_new = hv[1].clone(), -hv[0].clone()
                hv[0], hv[1] = v_new, h_new

            if f_code == 1: 
                prob = torch.flip(prob, [1]); hv = torch.flip(hv, [2]); hv[1] = -hv[1]
            elif f_code == 0: 
                prob = torch.flip(prob, [0]); hv = torch.flip(hv, [1]); hv[0] = -hv[0]
            elif f_code == -1: 
                prob = torch.flip(prob, [0, 1]); hv = torch.flip(hv, [1, 2]); hv = -hv
            
            all_probs.append(prob)
            all_hvs.append(hv)

        avg_prob = torch.stack(all_probs).mean(0).cpu().float().numpy()
        avg_hv = torch.stack(all_hvs).mean(0).cpu().float().numpy()
        
    return avg_prob, avg_hv, out.get('attr_logits', {})

# ==================================================================================================
# 3. 工具函数
# ==================================================================================================
def load_filtered_gt(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if isinstance(data, list): data = data[0]
        h, w = data.get('height', 512), data.get('width', 512)
        instance_map = np.zeros((h, w), dtype=np.int32)
        for idx, ann in enumerate(data.get('annotations', [])):
            seg = ann.get('segmentation')
            if isinstance(seg, list):
                for poly in seg:
                    poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(instance_map, [poly_np], idx + 1)
        return instance_map
    except: return None

class OrganPredictor:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.valid_ids = sorted(list(ID_TO_ORGAN.keys())) 
        self.organs = [ID_TO_ORGAN[i] for i in self.valid_ids]
        with torch.no_grad():
            self.text_features = self.model.encode_text(clip.tokenize([f"histology {org}" for org in self.organs]).to(device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_cv2):
        img_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        img_in = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img_in)
            feat /= feat.norm(dim=-1, keepdim=True)
            idx = (feat @ self.text_features.T).argmax().item()
        return self.organs[idx], self.valid_ids[idx]

ID_TO_ORGAN = {0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder", 3: "Breast", 4: "Cervix", 5: "Colon", 6: "Esophagus", 7: "HeadNeck", 8: "Kidney", 9: "Liver", 10: "Lung", 11: "Ovarian", 12: "Pancreatic", 13: "Prostate", 14: "Skin", 15: "Stomach", 16: "Testis", 17: "Thyroid", 18: "Uterus", 19: "Brain", 20: "Generic"}

# ==================================================================================================
# 4. Main
# ==================================================================================================
def main(args):
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
                    mask_decoder=vanilla_sam.mask_decoder, num_organs=21).to(args.device)
    
    # 🌟 修复加载权重的逻辑
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    state_dict = resize_pos_embed(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=False)  # <--- 修改了这里
    model.eval()

    predictor = OrganPredictor(args.device)
    all_metrics = defaultdict(list)
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.lower().endswith(('.png', '.tif'))]
    
    save_dir = os.path.join("workdir", "eval_tta8x", "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="8x-TTA Inference"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        
        # 🚀 修复 BGR 转 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        _, organ_id = predictor.predict(image_bgr)
        
        # 传递 rgb 图像给模型
        prob, hv, attr_logits = tta_inference_8x_batch(model, image_rgb, organ_id, args)
        
        dynamic_min_size = 30
        if 'size' in attr_logits:
            pred_size_idx = torch.argmax(attr_logits['size'], dim=1).item()
            dynamic_min_size = {0: 15, 1: 30, 2: 60}.get(pred_size_idx, 30)
            
        pred_mask = hover_post_process(prob, hv, args.prob_thresh, args.marker_thresh, min_marker_size=dynamic_min_size)
        
        gt_mask = load_filtered_gt(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.int32), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)
        else:
            print(f"❌ Missing GT: {img_path}") 

    print("\n" + "🌟" * 15 + "\n📊 Final Results (8x TTA):")
    for k, v in all_metrics.items(): print(f"{k:>10}: {np.mean(v):.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--prob_thresh", type=float, default=0.50)
    parser.add_argument("--marker_thresh", type=float, default=0.55)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())