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
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
import scipy.ndimage as ndimage

# ==================================================================================================
# 1. 核心后处理：官方对齐版 HoVer-Watershed
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.5, marker_thresh=0.4, min_marker_size=10):
    mask = prob_map > prob_thresh
    if not np.any(mask): return np.zeros_like(mask, dtype=np.int32)

    # 尺寸对齐保护
    if hv_map.shape[1:] != prob_map.shape:
        hv_map = np.array([cv2.resize(hv_map[i], (prob_map.shape[1], prob_map.shape[0])) for i in range(2)])

    # 归一化与 21x21 大核梯度计算 (真正释放 HoVer 威力)
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

# 器官 ID 配置
ID_TO_ORGAN = {
    0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder", 3: "Breast", 
    4: "Cervix", 5: "Colon", 6: "Esophagus", 7: "HeadNeck", 
    8: "Kidney", 9: "Liver", 10: "Lung", 11: "Ovarian", 
    12: "Pancreatic", 13: "Prostate", 14: "Skin", 15: "Stomach", 
    16: "Testis", 17: "Thyroid", 18: "Uterus",
    19: "Brain", 20: "Generic"
}

class OrganPredictor:
    def __init__(self, device):
        self.device = device
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
            indices = similarity[0].argmax().item()
        return self.organs[indices], self.valid_ids[indices]

def load_filtered_gt(img_path):
    mask_path_png = img_path.replace('.tif', '_mask.png').replace('.png', '_mask.png')
    if os.path.exists(mask_path_png):
        return cv2.imread(mask_path_png, cv2.IMREAD_UNCHANGED).astype(np.int32)
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--prob_thresh", type=float, default=0.50)
    parser.add_argument("--marker_thresh", type=float, default=0.54)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    return parser.parse_args()

# ==================================================================================================
# 2. 8-fold TTA 全图推理
# ==================================================================================================
def tta_inference_8x(model, image_rgb, organ_id, args):
    device = args.device
    input_size = (args.image_size, args.image_size)
    transforms = [(None, 0), (1, 0), (0, 0), (1, 2), (None, 1), (1, 1), (0, 1), (-1, 1)]
    
    accum_prob = None
    accum_hv = None

    for f_code, r_k in transforms:
        img_trans = image_rgb.copy()
        if f_code is not None: img_trans = cv2.flip(img_trans, f_code)
        if r_k > 0: img_trans = np.rot90(img_trans, k=r_k)
        
        img_t = torch.from_numpy(cv2.resize(img_trans, input_size)).permute(2, 0, 1).float().to(device)
        # 内部构造 input_sample，不再依赖外部 text_prompt
        input_sample = [{'image': img_t, 'original_size': input_size, 'organ_id': organ_id, 'text_prompt': "Cell nuclei"}]
        
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(input_sample, multimask_output=True)[0]
            best_idx = torch.argmax(out['iou_predictions']).item()
            prob = torch.sigmoid(out['masks'][0, best_idx]).float().cpu().numpy()
            
            hv_logits = out.get('hv_logits')
            if hv_logits.dim() == 3: hv_logits = hv_logits.unsqueeze(0)
            hv_rescaled = F.interpolate(hv_logits.float(), size=input_size, mode='bilinear', align_corners=False)
            hv = torch.tanh(hv_rescaled).squeeze(0).cpu().numpy()

        # 逆变换还原 (物理矢量对齐)
        if r_k > 0:
            prob = np.rot90(prob, k=-r_k)
            hv = np.rot90(hv, k=-r_k, axes=(1, 2))
            if r_k % 2 != 0: hv[0], hv[1] = hv[1].copy(), hv[0].copy()
        
        if f_code is not None:
            prob = cv2.flip(prob, f_code)
            hv = np.array([cv2.flip(hv[0], f_code), cv2.flip(hv[1], f_code)])
            if f_code == 1: hv[1] = -hv[1]
            elif f_code == 0: hv[0] = -hv[0]
            elif f_code == -1: hv = -hv

        if accum_prob is None:
            accum_prob, accum_hv = prob, hv
        else:
            accum_prob += prob
            accum_hv += hv

    return accum_prob / 8.0, accum_hv / 8.0, out.get('attr_logits', {})

def main(args):
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
                    mask_decoder=vanilla_sam.mask_decoder, num_organs=21).to(args.device)
    
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    predictor = OrganPredictor(args.device)
    all_metrics = defaultdict(list)
    image_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.lower().endswith(('.png', '.tif'))]
    
    save_dir = os.path.join("workdir", "eval_tta", "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="Inference"):
        image = cv2.imread(img_path)
        if image is None: continue
        
        # 1. 自动识别器官
        _, organ_id = predictor.predict(image)
        
        # 2. 8倍 TTA 推理
        prob, hv, attr_logits = tta_inference_8x(model, image, organ_id, args)
        
        # 3. 动态属性自适应 (基于模型内部 PNuRL 分支)
        dynamic_min_size = 30
        if 'size' in attr_logits:
            pred_size_idx = torch.argmax(attr_logits['size'], dim=1).item()
            dynamic_min_size = {0: 15, 1: 30, 2: 60}.get(pred_size_idx, 30)
            
        # 4. 终极后处理
        pred_mask = hover_post_process(prob, hv, args.prob_thresh, args.marker_thresh, min_marker_size=dynamic_min_size)
        
        # 5. 性能评估
        gt_mask = load_filtered_gt(img_path)
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask.astype(np.int32), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)

        if args.save_pred:
            vis = image.copy()
            cnts, _ = cv2.findContours((pred_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), vis)

    print("\n" + "="*40 + "\n📊 Final Results:")
    for k, v in all_metrics.items(): print(f"{k:>10}: {np.mean(v):.4f}")

if __name__ == '__main__':
    main(parse_args())