import argparse
import os
import time
import datetime
import random
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label as skimage_label
import cv2
import itertools  
import concurrent.futures
import multiprocessing

# 🔥 核心防御机制：防止 Numpy 在多进程下发生底层线程死锁 (针对你的 28 核 CPU)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam 
from DataLoader import UniversalDataset, stack_dict_batched
from torch.utils.data import DataLoader
from metrics import SegMetrics
import torch

# ==================================================================================================
# 🚀 全局共享缓存：利用你 240GB 的超大内存，实现多进程零拷贝读取
# ==================================================================================================
GLOBAL_CACHE = []

from scipy.ndimage import binary_fill_holes, binary_erosion

def hover_post_process(prob_map, hv_map, prob_thresh=0.40, marker_thresh=0.45, min_marker_size=12):
    mask = prob_map > prob_thresh
    mask = binary_fill_holes(mask)
    
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    # 🔥 绝技 1：高斯平滑抵抗梯度噪声
    v_map = cv2.GaussianBlur(v_map, (3, 3), 0)
    h_map = cv2.GaussianBlur(h_map, (3, 3), 0)

    diff_v = np.gradient(v_map, axis=0) 
    diff_h = np.gradient(h_map, axis=1) 
    sobel_mag = np.sqrt(diff_v**2 + diff_h**2)
    
    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    
    # 🔥 绝技 2：腐蚀操作剥离粘连种子
    marker_map = binary_erosion(marker_map, iterations=1)
    
    try:
        marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    except TypeError:
        marker_map = remove_small_objects(marker_map, max_size=min_marker_size)
        
    markers = skimage_label(marker_map).astype(np.int32)
    
    # 🔥 绝技 3：重塑分水岭地形能量图
    energy_map = prob_map - sobel_mag
    inst_map = watershed(-energy_map, markers, mask=mask)
    
    try:
        inst_map = remove_small_objects(inst_map, min_size=15)
    except TypeError:
        inst_map = remove_small_objects(inst_map, max_size=15)
        
    return inst_map.astype(np.int32)

# ==================================================================================================
# 🚀 独立并行的工作函数 (Worker)
# ==================================================================================================
def evaluate_single_combination(params):
    idx, p_thresh, m_thresh, mapping_dict = params
    mapping_name = mapping_dict["name"]
    size_map = mapping_dict["map"]
    
    current_ajis = []
    current_pqs = []
    
    # 直接读取内存中海量的缓存数据
    for prob_np, hv_np, gt_valid, pred_size_idx in GLOBAL_CACHE:
        dynamic_min_size = size_map.get(pred_size_idx, 10)
        
        pred_mask = hover_post_process(prob_np, hv_np, prob_thresh=p_thresh, marker_thresh=m_thresh, min_marker_size=dynamic_min_size)
        
        if pred_mask.max() == 0:
            pred_mask = skimage_label(prob_np > p_thresh).astype(np.int32)
        
        if pred_mask.shape != gt_valid.shape:
            gt_valid = cv2.resize(gt_valid, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        res = SegMetrics(pred_mask, gt_valid, ['mAJI', 'mPQ'])
        if 'mAJI' in res: current_ajis.append(res['mAJI'])
        if 'mPQ' in res: current_pqs.append(res['mPQ'])
    
    mean_aji = np.mean(current_ajis) if len(current_ajis) > 0 else 0
    mean_pq = np.mean(current_pqs) if len(current_pqs) > 0 else 0
    
    return idx, p_thresh, m_thresh, mapping_name, size_map, mean_aji, mean_pq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/PanNuke")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json")
    parser.add_argument("--best_model", type=str, default="workdir/models/mp_sam_dgot_stage2/best_model.pth") # 注意核对你的最佳权重路径
    parser.add_argument("--image_size", type=int, default=512)
    
    parser.add_argument("--use_pnurl", action='store_true', default=False)
    parser.add_argument("--use_coop", action='store_true', default=False)
    parser.add_argument("--use_sgot", action='store_true', default=False)
    parser.add_argument("--use_asr", action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🚀 [Step 1] Loading Validation Dataset...")
    val_dataset = UniversalDataset(data_root=args.data_path, knowledge_path=args.knowledge_path, 
                                   image_size=args.image_size, crop_size=256, mode='test', prompt_mode='generic')
    # 因为你内存有 240G，我们把 num_workers 开到 8 加速数据加载
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=stack_dict_batched, num_workers=8)

    print(f"🧠 [Step 2] Loading Model from {args.best_model}...")
    class DummyArgs: pass
    sam_args = DummyArgs()
    sam_args.checkpoint = None 
    sam_args.encoder_adapter = True
    sam_args.image_size = args.image_size
    vanilla_sam = sam_model_registry['vit_b'](sam_args)
    
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder, prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder, clip_model_name="ViT-B/16", num_organs=21, num_heads=8, sg_epsilon=0.05, sg_iters=3,
        use_pnurl=args.use_pnurl, use_coop=args.use_coop, use_sgot=args.use_sgot, use_asr=args.use_asr        
    ).to(device)
    
    checkpoint = torch.load(args.best_model, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("📦 [Step 3] RTX 5090 Inferencing & Caching to 240GB RAM...")
    
    # 声明使用全局缓存变量
    global GLOBAL_CACHE 
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batched_input in tqdm(val_loader, desc="Inference"):
            images = batched_input['image'].to(device)
            if images.shape[-1] != args.image_size:
                images = torch.nn.functional.interpolate(images, size=(args.image_size, args.image_size), mode='bilinear')
            
            inst_labels = batched_input.get('label_inst', batched_input['label']).numpy()
            
            model_input = [{
                'image': images[0], 'original_size': (args.image_size, args.image_size),
                'text_prompt': "Cell nuclei", 'organ_id': 20, 'attribute_text': "Cell nuclei", 'attr_labels': None
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

            attr_logits = out[0].get('attr_logits', {})
            if 'size' in attr_logits:
                pred_size_idx = torch.argmax(attr_logits['size']).item()
            else:
                pred_size_idx = 1 
                
            prob_np = prob_map.float().cpu().numpy()
            hv_np = hv_map.float().cpu().numpy()
            
            gt = inst_labels[0]; gt = gt[0] if gt.ndim == 3 else gt
            gt_valid = gt.copy(); gt_valid[gt == 255] = 0
            
            GLOBAL_CACHE.append((prob_np, hv_np, gt_valid, pred_size_idx))

    del model
    torch.cuda.empty_cache()

    print("\n🔍 [Step 4] Launching 28-Core Multi-Process Grid Search (Expanded Scales)...")
    
    # 既然之前的 0.5 兜底效果很好，我们把搜索重心往高概率偏移一点
    prob_thresh_list = [0.40, 0.45, 0.50, 0.55]
    marker_thresh_list = [0.40, 0.45, 0.50, 0.55]
    
    # 🔥🔥🔥 全面扩大的动态面积搜索空间 🔥🔥🔥
    size_mappings = [
        {"name": "Med_Scale",  "map": {0: 10,  1: 15,  2: 30}},  # 你刚才跑的 Baseline
        {"name": "Large_1",    "map": {0: 30,  1: 60,  2: 90}},  # 中大尺度
        {"name": "Large_2",    "map": {0: 50,  1: 100, 2: 150}}, # 大尺度
        {"name": "Huge_250",   "map": {0: 150, 1: 200, 2: 250}}, # 你非常关心的 250 像素级别
        {"name": "Extreme",    "map": {0: 250, 1: 300, 2: 400}}, # 极限尺度
        {"name": "Pure_CC",    "map": {0: 9999,1: 9999,2: 9999}} # 终极对照组：强行触发兜底，只做二值化连通域
    ]
    
    search_space = list(itertools.product(prob_thresh_list, marker_thresh_list, size_mappings))
    total_combinations = len(search_space)
    
    tasks = [(idx, p, m, d) for idx, (p, m, d) in enumerate(search_space)]
    
    best_aji = -1.0; best_pq = -1.0; best_params = {}
    
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🔥 Engines ON! Using {max_workers} CPU Cores for parallel evaluation. Total combinations: {total_combinations}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_combination, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_combinations, desc="Grid Search"):
            idx, p_thresh, m_thresh, mapping_name, size_map, mean_aji, mean_pq = future.result()
            
            print(f"✔ [Combo {idx+1:2d}] prob={p_thresh:.2f}, marker={m_thresh:.2f}, map={mapping_name:12s} | AJI: {mean_aji:.4f} | PQ: {mean_pq:.4f}")
            
            if mean_aji > best_aji:
                best_aji = mean_aji; best_pq = mean_pq
                best_params = {'prob': p_thresh, 'marker': m_thresh, 'map': mapping_name, 'map_dict': size_map}

    print("="*60)
    print("🏆 28-Core Grid Search Finished!")
    print(f"🌟 Best Params -> prob_thresh: {best_params['prob']:.2f}, marker_thresh: {best_params['marker']:.2f}")
    print(f"🌟 Best Mapping-> Strategy: {best_params['map']} {best_params['map_dict']}")
    print(f"🚀 Best Score  -> AJI: {best_aji:.4f} | PQ: {best_pq:.4f}")
if __name__ == "__main__":
    main()