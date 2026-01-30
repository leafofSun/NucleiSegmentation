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
from skimage.measure import label, regionprops

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

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

# 🔥 权重适配 (必须保留，用于适配不同分辨率的权重文件)
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
        best_list_idx = indices.item()
        confidence = values.item()
        return self.organs[best_list_idx], self.valid_ids[best_list_idx], confidence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="universal_test") 
    parser.add_argument("--text_prompt", type=str, default=None)
    
    # 🔥🔥🔥 核心定义 🔥🔥🔥
    # patch_size: 在原图上切多大的窗口 (例如 PanNuke 原图 256，WSI 可能 1024)
    parser.add_argument("--patch_size", type=int, default=256, help="Window size to crop from original image")
    # image_size: 缩放成多大喂给模型 (例如模型是 512 训练的)
    parser.add_argument("--image_size", type=int, default=512, help="Input size expected by the model")
    # stride: 滑窗步长 (重叠控制)
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride")
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data/PanNuke/test") 
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou', 'mAJI', 'mPQ'])
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument("--use_watershed", action='store_true', default=True)
    parser.add_argument("--encoder_adapter", action='store_true', default=True)
    return parser.parse_args()

def post_process_watershed(prob_map, threshold=0.5, min_size=20, min_distance=5):
    binary_mask = prob_map > threshold
    if not np.any(binary_mask): return np.zeros_like(binary_mask, dtype=np.int32)
    distance = ndimage.distance_transform_edt(binary_mask)
    local_maxi = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    local_maxi_mask = np.zeros_like(prob_map, dtype=bool)
    local_maxi_mask[tuple(local_maxi.T)] = True
    markers = label(local_maxi_mask)
    labels = watershed(-distance, markers, mask=binary_mask)
    if min_size > 0:
        labels = remove_small_objects(labels, min_size=min_size)
        labels = label(labels > 0)
    return labels.astype(np.int32)

# 🔥🔥🔥 通用滑窗推理函数 🔥🔥🔥
# 逻辑：原图 -> Padding -> 切Patch -> Resize(image_size) -> 预测 -> ResizeBack(patch_size) -> 拼回原图
def sliding_window_inference(model, image, device, patch_size, image_size, stride, 
                             text_prompt, organ_id, attribute_text=None, filename=None):
    h, w = image.shape[:2]
    
    # 1. Padding 保证能被切尽 (基于 patch_size)
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad = image_pad.shape[:2]
    
    prob_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map_full = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    # 2. 生成滑窗坐标
    # 注意：步长不能大于 patch_size
    real_stride = min(stride, patch_size)
    
    y_steps = list(range(0, h_pad - patch_size + 1, real_stride))
    if (h_pad - patch_size) % real_stride != 0: y_steps.append(h_pad - patch_size)
    x_steps = list(range(0, w_pad - patch_size + 1, real_stride))
    if (w_pad - patch_size) % real_stride != 0: x_steps.append(w_pad - patch_size)
    
    if attribute_text is None: attribute_text = ""

    model.eval()
    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                # A. 裁剪 Patch (物理尺寸)
                patch = image_pad[y:y+patch_size, x:x+patch_size, :]
                
                # B. Resize 到模型输入尺寸 (image_size)
                # 例如：PanNuke 256 -> 512 (放大)
                # 例如：WSI 1024 -> 512 (缩小)
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
                
                # C. 预测
                outputs = model(input_sample, multimask_output=True)
                out = outputs[0]
                best_idx = torch.argmax(out['iou_predictions']).item()
                logits = out['masks'][0, best_idx, :, :] # [image_size, image_size]
                
                # D. Resize Back 到物理尺寸 (patch_size)
                # 这步很关键，保证拼回去的时候像素是对齐的
                logits = logits.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
                if patch_size != image_size:
                    logits_orig = F.interpolate(logits, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                else:
                    logits_orig = logits
                
                prob_patch = torch.sigmoid(logits_orig).squeeze().cpu().numpy()
                
                # E. 累加到全图
                prob_map_full[y:y+patch_size, x:x+patch_size] += prob_patch
                count_map_full[y:y+patch_size, x:x+patch_size] += 1.0
                
    count_map_full[count_map_full == 0] = 1.0
    avg_prob = prob_map_full / count_map_full
    
    # 3. 裁剪回原始图片尺寸
    return avg_prob[:h, :w]

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

def load_filtered_gt(img_path, attr_data, target_tag=None):
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
        return mask
    except: return None

def main(args):
    # 初始化 SAM (注意这里 img_size 要设为模型实际大小)
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
        # 权重自适应 (无论权重是 256 还是 512，都适配当前 args.image_size)
        state_dict = resize_pos_embed(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"❌ Checkpoint not found")
        return

    organ_predictor = OrganPredictor(args.device)
    attr_data = {} 
    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    all_metrics = defaultdict(list)
    save_dir = os.path.join(args.work_dir, args.run_name, "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    print('*'*60)
    print(f"🚀 Inference Config: Window={args.patch_size} -> Model={args.image_size}")
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

        if args.text_prompt:
             class_prompt_text = args.text_prompt
             current_organ_id = 20
             log_msg = f"👤 Override: '{class_prompt_text}'"
        else:
             class_prompt_text = f"{pred_organ} cell nuclei"

        attribute_prompt_text = "" 
        pbar.write(f"🖼️  {filename} | {log_msg} -> Class: '{class_prompt_text}' | Attr: [AUTO]")

        # 🔥 通用推理
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size,  # 切多大 (如 256)
            image_size=args.image_size,  # 喂多大 (如 512)
            stride=args.stride,          # 步长 (如 128)
            text_prompt=class_prompt_text, organ_id=current_organ_id, 
            attribute_text=attribute_prompt_text, filename=filename
        )
        
        if args.use_watershed:
            pred_mask = post_process_watershed(pred_prob, threshold=0.5)
        else:
            pred_mask = label(pred_prob > 0.5).astype(np.int32)
        
        gt_mask = load_filtered_gt(img_path, attr_data, target_tag="Auto_Organ")
        
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