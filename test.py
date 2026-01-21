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

# 后处理
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, opening, disk
from scipy import ndimage
from skimage.measure import label, regionprops

# 确保在文件开头导入了 mask
try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

# 🔥 [配置] 字典必须是 ID -> Name (Int -> Str)
ID_TO_ORGAN = {
    # --- PanNuke 19 类 ---
    0: "Adrenal_gland", 1: "Bile-duct", 2: "Bladder", 3: "Breast", 
    4: "Cervix", 5: "Colon", 6: "Esophagus", 7: "HeadNeck", 
    8: "Kidney", 9: "Liver", 10: "Lung", 11: "Ovarian", 
    12: "Pancreatic", 13: "Prostate", 14: "Skin", 15: "Stomach", 
    16: "Testis", 17: "Thyroid", 18: "Uterus",
    # --- 补充 ---
    19: "Brain", 20: "Generic"
}

# 🔥 [配置] 反向字典：Name -> ID (忽略大小写)
ORGAN_TO_ID = {v.lower().replace("-", "_"): k for k, v in ID_TO_ORGAN.items()}
# 补充一些常见的变体映射
ORGAN_TO_ID.update({
    "bile_duct": 1, "head_neck": 7, "adrenal gland": 0
})

class OrganPredictor:
    def __init__(self, device):
        self.device = device
        print("🧠 Loading CLIP for Organ Diagnosis...")
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        
        # 准备文本特征
        self.organs = [ID_TO_ORGAN[i] for i in range(len(ID_TO_ORGAN))]
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
            
        best_idx = indices.item()
        confidence = values.item()
        # 注意：这里 best_idx 对应 self.organs 的索引，恰好也是 ORGAN_ID
        return self.organs[best_idx], best_idx, confidence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-dynamic-final") 
    parser.add_argument("--text_prompt", type=str, default=None, help="Custom Class prompt override (e.g. 'Liver')")
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
    labeled = label(pred_mask)
    regions = regionprops(labeled)
    if not regions: return 0.0, 0.0, 0
    areas = [r.area for r in regions]
    roundnesses = [(4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
    return np.mean(areas), np.mean(roundnesses), len(regions)

# 🔥 [修正] 推理函数：明确区分 Class Prompt 和 Attribute Text
def sliding_window_inference(model, image, device, patch_size, image_size, stride, 
                             text_prompt, organ_id, attribute_text=None, filename=None):
    """
    text_prompt: 类别提示 (Class), 用于 DualPromptLearner (e.g., "Liver cell nuclei")
    attribute_text: 属性提示 (Attribute), 用于 PNuRL. 推理时应为空，触发自动预测。
    """
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
    
    # 🔥 [核心逻辑] 推理时属性提示默认为空，让模型自己看图
    if attribute_text is None:
        attribute_text = ""

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
                    'text_prompt': text_prompt,       # ✅ 这是一个类别提示 (Class)
                    'organ_id': organ_id,             # ✅ 这是一个类别ID (Class)
                    'attribute_text': attribute_text  # ✅ 这是一个属性提示 (Attribute) -> 为空!
                }]
                
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

def get_organ_from_json(img_path):
    """
    尝试读取同名 JSON 获取 organ_type (Ground Truth Class)
    """
    # 优先找同名json，其次找上级目录
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
         json_path = img_path.rsplit('.', 1)[0] + ".json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 兼容多种JSON结构
            organ = data.get('organ_type', None)
            if not organ: 
                meta = data.get('metadata', {})
                organ = meta.get('organ_type', None)
                
            if organ:
                key = organ.lower().replace("-", "_")
                if key in ORGAN_TO_ID:
                    return ID_TO_ORGAN[ORGAN_TO_ID[key]], ORGAN_TO_ID[key]
                else:
                    # 尝试模糊匹配
                    for k_id, v_name in ID_TO_ORGAN.items():
                        if key in v_name.lower():
                            return v_name, k_id
                    print(f"⚠️ Unknown organ in JSON: {organ}")
        except Exception as e:
            print(f"❌ Error reading JSON {json_path}: {e}")
            
    return None, None

def load_filtered_gt(img_path, attr_data, target_tag=None):
    # (此函数保持不变，用于加载 GT Mask 计算指标)
    base_name = os.path.basename(img_path)
    filename_key = None
    if base_name in attr_data:
        filename_key = base_name
    else:
        for k in attr_data.keys():
            if base_name in k or k in base_name:
                filename_key = k
                break
    
    valid_ids = None 
    if filename_key and filename_key in attr_data:
        instances = attr_data[filename_key]
        if target_tag and target_tag not in ["Generic", "Cell nuclei", "Auto_Organ", None]:
            valid_ids = set()
            for inst in instances:
                tags = [t.lower() for t in inst.get('tags', [])]
                search_key = target_tag.split()[0].lower()
                if search_key in tags:
                    valid_ids.add(inst['id'])
    
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        possible_json = img_path.rsplit('.', 1)[0] + ".json"
        if os.path.exists(possible_json):
            json_path = possible_json
        else:
            return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            anns = data.get('annotations', [])
        else:
            anns = data
            
        if not anns: return None
        
        temp_img = cv2.imread(img_path)
        if temp_img is None: return None
        h, w = temp_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for idx, ann in enumerate(anns):
            if valid_ids is not None and idx not in valid_ids:
                continue
                
            if 'segmentation' not in ann: continue
            seg = ann['segmentation']
            
            if isinstance(seg, dict) and 'counts' in seg: 
                if isinstance(seg['counts'], list):
                    rle = coco_mask.frPyObjects(seg, h, w)
                    rle_mask = coco_mask.decode(rle)
                else:
                    if isinstance(seg['counts'], str):
                        seg['counts'] = seg['counts'].encode('utf-8')
                    rle_mask = coco_mask.decode(seg)
                
                if len(rle_mask.shape) == 3: 
                     rle_mask = np.max(rle_mask, axis=2)
                mask[rle_mask > 0] = 1

            elif isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [pts], 1)
        
        return mask

    except Exception as e:
        print(f"⚠️ Error loading GT for {base_name}: {e}")
        return None

def main(args):
    # 初始化模型
    vanilla_sam = sam_model_registry[args.model_type](args)
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256,
        num_organs=21 
    ).to(args.device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    organ_predictor = OrganPredictor(args.device)
    attr_data = {} # 占位，GT Mask加载用

    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    all_metrics = defaultdict(list)
    save_dir = os.path.join(args.work_dir, args.run_name, "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    print('*'*60)
    print(f"🚀 Running Inference: {args.run_name}")
    print('*'*60)

    pbar = tqdm(image_files)
    for img_path in pbar:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # 🌟 Step 1: 确定器官类别 (Class Info) -> 解决 "Who am I?"
        # 优先级: JSON真值 > AI预测 > 用户覆盖
        # ==========================================
        
        # A. 查 JSON
        gt_organ, gt_id = get_organ_from_json(img_path)
        
        if gt_organ:
            pred_organ = gt_organ
            current_organ_id = gt_id
            log_msg = f"📂 JSON: {pred_organ} (ID={gt_id})"
        else:
            # B. 用 CLIP 预测
            pred_organ, pred_id, conf = organ_predictor.predict(image)
            current_organ_id = pred_id
            log_msg = f"🧠 AI: {pred_organ} ({conf:.1%})"

        # C. 用户强制覆盖
        if args.text_prompt:
             # 注意：这里的 text_prompt 也是指类别，不是属性
             class_prompt_text = args.text_prompt
             current_organ_id = 20 # Generic
             log_msg = f"👤 Override: '{class_prompt_text}'"
        else:
             # 构造类别提示词
             class_prompt_text = f"{pred_organ} cell nuclei"

        # ==========================================
        # 🌟 Step 2: 确定属性提示 (Attribute Info) -> 解决 "What do I look like?"
        # 推理时必须为空，触发 PNuRL 内部自动预测
        # ==========================================
        attribute_prompt_text = ""

        pbar.write(f"🖼️  {filename} | {log_msg} -> Class: '{class_prompt_text}' | Attr: [AUTO]")
        
        # 推理
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size, image_size=args.image_size, stride=args.stride,
            
            # ✅ 明确区分两个通道
            text_prompt=class_prompt_text,        # Class 通道 (给 DualPromptLearner)
            organ_id=current_organ_id,            # Class 通道 (给 DualPromptLearner)
            attribute_text=attribute_prompt_text, # Attribute 通道 (给 PNuRL, 必须为空)
            
            filename=filename
        )
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # 计算指标
        gt_mask = load_filtered_gt(img_path, attr_data, target_tag="Auto_Organ")
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items(): all_metrics[k].append(v)
        
        # 保存图片
        if args.save_pred:
            vis = image.copy()
            if gt_mask is not None:
                cnts_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts_gt, -1, (0, 0, 255), 2)
            cnts_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts_pred, -1, (0, 255, 0), 2)
            cv2.putText(vis, f"Org: {pred_organ}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            save_path = os.path.join(save_dir, filename.replace('.tif','.jpg').replace('.png', '.jpg'))
            cv2.imwrite(save_path, vis)

    print("\n" + "="*40)
    print(f"📊 Final Results:")
    for k, v in all_metrics.items():
        if len(v) > 0: print(f"  {k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)