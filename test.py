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
from PIL import Image # 新增：用于 CLIP 处理
import clip           # 新增：CLIP 库

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

# 🔥 全局定义：确保顺序与训练时一致
ID_TO_ORGAN = {
    0: "Kidney", 1: "Breast", 2: "Prostate", 3: "Lung", 
    4: "Colon", 5: "Stomach", 6: "Liver", 7: "Bladder", 
    8: "Brain", 9: "Generic"
}

# 🌟 新增：基于 CLIP 的器官诊断器
class OrganPredictor:
    def __init__(self, device):
        self.device = device
        print("🧠 Loading CLIP for Organ Diagnosis...")
        # 加载 CLIP 模型 (需确保显存足够，ViT-B/16 约需几百MB)
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        
        # 准备文本特征
        self.organs = [ID_TO_ORGAN[i] for i in range(len(ID_TO_ORGAN))]
        # 构造 prompt 模板，Generic 作为兜底
        templates = [f"A histology image of {org} tissue." for org in self.organs]
        
        with torch.no_grad():
            text_inputs = clip.tokenize(templates).to(device)
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_cv2):
        """
        输入: BGR 格式的 OpenCV 图片
        输出: (预测器官名, 器官ID, 置信度)
        """
        # 转为 PIL 并进行预处理
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        image_input = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            
        best_idx = indices.item()
        confidence = values.item()
        return self.organs[best_idx], best_idx, confidence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="text-guided-sam-dynamic-final") 
    
    parser.add_argument("--text_prompt", type=str, default=None, help="Custom prompt override")
    parser.add_argument("--test_attr_path", type=str, default="data/MoNuSeg_SA1B/test_attributes.json")
    
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

# 🔥 [修改] 增加了 organ_id 参数
def sliding_window_inference(model, image, device, patch_size, image_size, stride, text_prompt, organ_id, filename=None):
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
                    'text_prompt': text_prompt,
                    # 🔥 [关键] 注入 organ_id 以激活隐式知识库 (DualPromptLearner)
                    'organ_id': organ_id,
                    'attribute_text': text_prompt # KIM 模块同时也需要显式文本
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

# 🔥 [修复版] test.py 中的 load_filtered_gt
def load_filtered_gt(img_path, attr_data, target_tag=None):
    # 1. 尝试从属性文件中匹配文件名
    base_name = os.path.basename(img_path)
    filename_key = None
    
    # 模糊匹配逻辑
    if base_name in attr_data:
        filename_key = base_name
    else:
        # 尝试找包含关系的 key
        for k in attr_data.keys():
            if base_name in k or k in base_name:
                filename_key = k
                break
    
    # 2. 确定要保留的 ID
    valid_ids = None # None 表示保留所有
    if filename_key and filename_key in attr_data:
        instances = attr_data[filename_key]
        # 如果指定了特定 Tag (如 "Tumor")，则筛选
        if target_tag and target_tag not in ["Generic", "Cell nuclei", "Auto_Organ", None]:
            valid_ids = set()
            for inst in instances:
                tags = [t.lower() for t in inst.get('tags', [])]
                search_key = target_tag.split()[0].lower() # 取第一个词匹配
                if search_key in tags:
                    valid_ids.add(inst['id'])
    
    # 3. 读取原始 JSON (兜底逻辑)
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        # 尝试替换后缀查找
        possible_json = img_path.rsplit('.', 1)[0] + ".json"
        if os.path.exists(possible_json):
            json_path = possible_json
        else:
            return None # 真的没有 GT 文件

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 兼容不同格式
        if isinstance(data, dict):
            anns = data.get('annotations', [])
        else:
            anns = data
            
        if not anns: return None
        
        # 初始化 Mask
        temp_img = cv2.imread(img_path)
        h, w = temp_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 4. 绘制 Mask
        for idx, ann in enumerate(anns):
            # 🔥 [核心修复] 如果 valid_ids 是 None，说明不过滤，全部绘制
            if valid_ids is not None and idx not in valid_ids:
                continue
                
            if 'segmentation' not in ann: continue
            seg = ann['segmentation']
            
            if isinstance(seg, dict) and 'counts' in seg: 
                rle_mask = coco_mask.decode(seg)
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
        num_organs=10 # 确保和你训练时一致
    ).to(args.device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    # 🔥 初始化器官预测器
    organ_predictor = OrganPredictor(args.device)

    attr_data = {}
    if os.path.exists(args.test_attr_path):
        with open(args.test_attr_path, 'r') as f:
            content = json.load(f)
            attr_data = content.get("images", {})

    image_files = []
    for root, dirs, files in os.walk(args.data_path):
        for f in files:
            if f.lower().endswith(('.tif', '.png', '.jpg')) and 'mask' not in f.lower():
                image_files.append(os.path.join(root, f))
    
    all_metrics = defaultdict(list)
    prompt_stats = {"avg_area": [], "avg_roundness": [], "count": []}
    
    save_dir = os.path.join(args.work_dir, args.run_name, "inference_viz")
    if args.save_pred: os.makedirs(save_dir, exist_ok=True)

    print('*'*60)
    print(f"🚀 Running Inference: {args.run_name}")
    print(f"🧠 AI-Diagnosis Mode Active")
    print('*'*60)

    pbar = tqdm(image_files)
    for img_path in pbar:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # 🌟 核心步骤：AI 自动诊断
        # ==========================================
        pred_organ, pred_id, conf = organ_predictor.predict(image)
        
        # 逻辑分支
        if args.text_prompt:
            # 用户强制指定 (Expert Mode)
            prompt_text = args.text_prompt
            current_organ_id = 9 # Generic
            log_msg = f"👤 Override: '{prompt_text}' (Ignored AI: {pred_organ})"
        else:
            # AI 自适应 (Auto Mode)
            prompt_text = f"{pred_organ} cell nuclei"
            current_organ_id = pred_id
            log_msg = f"🧠 AI: {pred_organ} ({conf:.1%}) -> '{prompt_text}'"

        pbar.write(f"🖼️  {filename} | {log_msg}")
        
        # ==========================================
        # 🚀 带着器官信息去分割
        # ==========================================
        pred_prob = sliding_window_inference(
            model, image_rgb, args.device, 
            patch_size=args.patch_size,
            image_size=args.image_size,
            stride=args.stride,
            text_prompt=prompt_text, 
            organ_id=current_organ_id, # 🔥 传入 ID
            filename=filename
        )
        
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # 统计信息
        p_area, p_round, p_count = analyze_predictions(pred_mask)
        if p_count > 0:
            prompt_stats["avg_area"].append(p_area)
            prompt_stats["avg_roundness"].append(p_round)
            prompt_stats["count"].append(p_count)

        # 指标计算 (如果有关联的GT)
        # 注意：这里 load_filtered_gt 仍然使用 text_prompt 来过滤
        # 如果是 Auto Mode，我们暂且认为就是评估"所有细胞"
        target_tag_for_eval = args.text_prompt if args.text_prompt else "Auto_Organ"
        gt_mask = load_filtered_gt(img_path, attr_data, target_tag=target_tag_for_eval)
        
        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                all_metrics[k].append(v)
        
        # 可视化
        if args.save_pred:
            # 1. 复制原图
            vis = image.copy()
            
            # 2. 绘制 GT (红色) - BGR: (0, 0, 255)
            # 注意：gt_mask 可能为 None (如果没有对应的 JSON)
            if gt_mask is not None:
                # 确保尺寸一致
                if gt_mask.shape != pred_mask.shape:
                    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                cnts_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 使用线宽 2
                cv2.drawContours(vis, cnts_gt, -1, (0, 0, 255), 2)

            # 3. 绘制预测 (绿色) - BGR: (0, 255, 0)
            cnts_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts_pred, -1, (0, 255, 0), 2)
            
            # 4. 添加图例和信息
            # 顶部信息：Prompt 和 AI 诊断
            cv2.putText(vis, f"AI: {pred_organ} ({conf:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # 青色文字
            
            # 底部信息：图例 (红=GT, 绿=Pred)
            h_img, w_img = vis.shape[:2]
            legend_text = f"Red: GT | Green: Pred | Area: {int(p_area)}"
            cv2.putText(vis, legend_text, (10, h_img - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # 白色文字

            # 5. 保存
            save_path = os.path.join(save_dir, filename.replace('.tif','.jpg').replace('.png', '.jpg'))
            cv2.imwrite(save_path, vis)

    print("\n" + "="*40)
    print(f"📊 Final Results:")
    for k, v in all_metrics.items():
        if len(v) > 0:
            print(f"  {k:>10}: {np.mean(v):.4f}")
    print("="*40)

if __name__ == '__main__':
    args = parse_args()
    main(args)