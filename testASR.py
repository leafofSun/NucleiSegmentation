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
from scipy.ndimage.morphology import binary_fill_holes

# ==================================================================================================
# 1. 核心后处理：完全对齐 train.py 的基础版 HoVer-Watershed (np.gradient)
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.45, marker_thresh=0.4, min_marker_size=10):
    mask = prob_map > prob_thresh
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    # 🚀 使用 train.py 中的基础梯度计算
    diff_v = np.gradient(v_map, axis=0) 
    diff_h = np.gradient(h_map, axis=1) 
    
    sobel_mag = np.sqrt(diff_v**2 + diff_h**2)
    
    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    
    marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    markers = skimage_label(marker_map).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/PanNuke")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json")
    parser.add_argument("--best_model", type=str, default="workdir/models/MP_SAM_HoverNet/best_model.pth")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🚀 [Step 1] Loading Validation Dataset...")
    val_dataset = UniversalDataset(data_root=args.data_path, knowledge_path=args.knowledge_path, 
                                   image_size=args.image_size, crop_size=256, mode='test', prompt_mode='generic')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=stack_dict_batched, num_workers=4)

    print(f"🧠 [Step 2] Loading Model from {args.best_model}...")
    class DummyArgs: pass
    sam_args = DummyArgs()
    sam_args.checkpoint = None 
    sam_args.encoder_adapter = True
    sam_args.image_size = args.image_size
    vanilla_sam = sam_model_registry['vit_b'](sam_args)
    
    model = TextSam(image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
                    mask_decoder=vanilla_sam.mask_decoder, clip_model_name="ViT-B/16", num_organs=21,
                    num_heads=8, sg_epsilon=0.05, sg_iters=3).to(device)
    
    checkpoint = torch.load(args.best_model, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("📦 [Step 3] Pre-computing predictions for all validation images (This happens only ONCE)...")
    cache_data = []  
    
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

            prob_np = prob_map.float().cpu().numpy()
            hv_np = hv_map.float().cpu().numpy()
            
            gt = inst_labels[0]; gt = gt[0] if gt.ndim == 3 else gt
            gt_valid = gt.copy(); gt_valid[gt == 255] = 0
            
            cache_data.append((prob_np, hv_np, gt_valid))

    print(f"✅ Cache complete for {len(cache_data)} images. Freeing GPU memory...")
    del model; torch.cuda.empty_cache()

    print("🔍 [Step 4] Starting Grid Search on CPU (Using train.py original logic)...")
    
    # 🚀 适应 np.gradient 的经典阈值网格
    prob_thresholds = [0.40, 0.45, 0.50]
    marker_thresholds = [0.35, 0.40, 0.45]
    
    best_aji = 0.0
    best_params = (0, 0)
    
    for p_thresh in prob_thresholds:
        for m_thresh in marker_thresholds:
            # 🔥 删除了那个坑人的 continue 限制，确保每个组合都被搜到！
            
            current_ajis = []
            for prob_np, hv_np, gt_valid in cache_data:
                pred_mask = hover_post_process(prob_np, hv_np, prob_thresh=p_thresh, marker_thresh=m_thresh)
                
                # fallback 保护
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
    print(f"🌟 Best AJI (Train.py Logic): {best_aji:.4f}")
    print(f"👉 Best Parameters: prob_thresh = {best_params[0]:.2f}, marker_thresh = {best_params[1]:.2f}")
    print("=====================================================")

if __name__ == "__main__":
    main()