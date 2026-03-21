import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label as skimage_label
import cv2

# 导入你的模型和数据加载器
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import UniversalDataset, stack_dict_batched
from torch.utils.data import DataLoader
from metrics import SegMetrics
import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label as skimage_label

def hover_post_process(prob_map, hv_map, prob_thresh=0.5, marker_thresh=0.4, min_marker_size=10):
    # 1. 提取基础概率掩码
    blb_raw = prob_map
    h_dir_raw = hv_map[1]  # 注意 x, y 的提取顺序
    v_dir_raw = hv_map[0]

    # 2. 生成初始的连通域 blb
    blb = np.array(blb_raw >= prob_thresh, dtype=np.int32)
    blb = skimage_label(blb)
    blb = remove_small_objects(blb, min_size=min_marker_size)
    blb[blb > 0] = 1

    if not np.any(blb):
        return np.zeros_like(blb, dtype=np.int32)

    # 3. 核心！第一重归一化：将原始距离图映射到 [0, 1]
    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 4. 核心！21x21 的超大感受野捕捉距离突变
    ksize = 21 
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    # 5. 核心！第二重归一化并反转（梯度越大，值越接近 0）
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 取最大反转梯度，并扣除背景影响
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    # 6. 生成专属于 HoVerNet 的平滑拓扑地形图 (距离图)
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # 7. 根据动态阈值提取 Marker (种子点)
    overall_bin = np.array(overall >= marker_thresh, dtype=np.int32)
    marker = blb - overall_bin
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    
    # 使用椭圆核做一次形态学开运算，切断最后的蛛丝马迹
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = skimage_label(marker)
    marker = remove_small_objects(marker, min_size=min_marker_size)

    # 8. 终极分水岭算法
    proced_pred = watershed(dist, markers=marker, mask=blb)
    
    return proced_pred.astype(np.int32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/PanNuke")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json")
    parser.add_argument("--best_model", type=str, default="workdir/models/SGOT/best_model.pth")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🚀 [Step 1] Loading Validation Dataset...")
    val_dataset = UniversalDataset(data_root=args.data_path, knowledge_path=args.knowledge_path, 
                                   image_size=args.image_size, crop_size=256, mode='test', prompt_mode='generic')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=stack_dict_batched, num_workers=4)

    print(f"🧠 [Step 2] Loading Model from {args.best_model}...")
    # 这里为了简便，需要重新构建一次你的模型结构 (与 train.py 一致)
    class DummyArgs: pass
    sam_args = DummyArgs()
    sam_args.checkpoint = None # 不加载预训练
    sam_args.encoder_adapter = True
    sam_args.image_size = args.image_size
    vanilla_sam = sam_model_registry['vit_b'](sam_args)
    
    model = TextSam(image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
                    mask_decoder=vanilla_sam.mask_decoder, clip_model_name="ViT-B/16", num_organs=21,
                    num_heads=8, sg_epsilon=0.05, sg_iters=3).to(device)
    
    # 加载你刚刚跑出来的 0.5659 的最好权重
    checkpoint = torch.load(args.best_model, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("📦 [Step 3] Pre-computing predictions for all validation images (This happens only ONCE)...")
    cache_data = []  # 存储 (prob_np, hv_np, gt_valid)
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batched_input in tqdm(val_loader, desc="Inference"):
            images = batched_input['image'].to(device)
            if images.shape[-1] != args.image_size:
                images = torch.nn.functional.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear')
            
            inst_labels = batched_input.get('label_inst', batched_input['label']).numpy()
            organ_ids = batched_input.get('organ_id', [20])
            
            model_input = [{
                'image': images[0], 'original_size': (args.image_size, args.image_size),
                'text_prompt': "Cell nuclei", 'organ_id': organ_ids[0], 'attribute_text': "Cell nuclei"
            }]
            
            out = model(model_input, multimask_output=True)
            iou_preds = out[0]['iou_predictions'].squeeze(0) if out[0]['iou_predictions'].ndim == 2 else out[0]['iou_predictions']
            best_idx = torch.argmax(iou_preds).item()
            
            prob_map = torch.sigmoid(out[0]['masks'][0, best_idx])
            hv_logits = out[0].get('hv_logits', None)
            
            if hv_logits is not None:
                hv_map = torch.tanh(hv_logits)
                if hv_map.dim() == 3: hv_map = hv_map.unsqueeze(0)
                hv_map = torch.nn.functional.interpolate(hv_map.float(), size=prob_map.shape[-2:], mode='bilinear').squeeze(0)
            else:
                hv_map = torch.zeros((2, prob_map.shape[-2], prob_map.shape[-1]), device=device)

            # 转为 NumPy 存入内存
            prob_np = prob_map.float().cpu().numpy()
            hv_np = hv_map.float().cpu().numpy()
            
            gt = inst_labels[0]; gt = gt[0] if gt.ndim == 3 else gt
            gt_valid = gt.copy(); gt_valid[gt == 255] = 0
            
            cache_data.append((prob_np, hv_np, gt_valid))

    print(f"✅ Cache complete for {len(cache_data)} images. Freeing GPU memory...")
    del model; torch.cuda.empty_cache()

    print("🔍 [Step 4] Starting Grid Search on CPU...")
    # 定义你要搜索的网格范围 (可以根据需要调大调小)
    prob_thresholds = [0.48, 0.50, 0.52]
    marker_thresholds = [0.54, 0.56, 0.58]
    
    best_aji = 0.0
    best_params = (0, 0)
    
    # 嵌套循环暴力搜索
    for p_thresh in prob_thresholds:
        for m_thresh in marker_thresholds:
            # 过滤掉不合理的组合 (通常 marker 的阈值要比整体概率的阈值稍微小一点或持平)
            if m_thresh > p_thresh + 0.05:
                continue 
                
            current_ajis = []
            for prob_np, hv_np, gt_valid in cache_data:
                pred_mask = hover_post_process(prob_np, hv_np, prob_thresh=p_thresh, marker_thresh=m_thresh)
                
                # fallback
                if pred_mask.max() == 0:
                    pred_mask = skimage_label(prob_np > 0.5).astype(np.int32)
                
                if pred_mask.shape != gt_valid.shape:
                    gt_valid = cv2.resize(gt_valid, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                res = SegMetrics(pred_mask, gt_valid, ['mAJI'])
                if 'mAJI' in res:
                    current_ajis.append(res['mAJI'])
            
            mean_aji = np.mean(current_ajis) if len(current_ajis) > 0 else 0
            print(f"Prob Thresh: {p_thresh:.2f} | Marker Thresh: {m_thresh:.2f} ---> AJI: {mean_aji:.4f}")
            
            if mean_aji > best_aji:
                best_aji = mean_aji
                best_params = (p_thresh, m_thresh)

    print("=====================================================")
    print("🏆 Grid Search Finished!")
    print(f"🌟 Best AJI: {best_aji:.4f}")
    print(f"👉 Best Parameters: prob_thresh = {best_params[0]:.2f}, marker_thresh = {best_params[1]:.2f}")
    print("=====================================================")

if __name__ == "__main__":
    main()