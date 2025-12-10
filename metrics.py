import torch
import numpy as np
from skimage import measure
from scipy.optimize import linear_sum_assignment

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if isinstance(x, list):
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        if x.min() < 0:
            x = m(x)
    return x, y

def get_bounding_box(img):
    """获取二值图像的边界框，用于加速计算"""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1

def compute_aji_single_image(pred, gt):
    """计算单张图像的 Aggregated Jaccard Index (AJI)"""
    # 确保输入是实例 ID 映射 (H, W)，如果是二值图则进行连通域标记
    if pred.max() <= 1:
        pred = measure.label(pred)
    if gt.max() <= 1:
        gt = measure.label(gt)

    pred_id_list = list(np.unique(pred))
    gt_id_list = list(np.unique(gt))
    
    # 移除背景 (0)
    if 0 in pred_id_list: pred_id_list.remove(0)
    if 0 in gt_id_list: gt_id_list.remove(0)

    pred_masks = {p: (pred == p).astype(np.uint8) for p in pred_id_list}
    gt_masks = {g: (gt == g).astype(np.uint8) for g in gt_id_list}

    pairwise_inter = np.zeros([len(gt_id_list), len(pred_id_list)], dtype=np.float64)
    pairwise_union = np.zeros([len(gt_id_list), len(pred_id_list)], dtype=np.float64)

    # 计算成对的交集和并集
    for i, gt_id in enumerate(gt_id_list):
        g_mask = gt_masks[gt_id]
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(g_mask)
        
        # 只检查有重叠的预测区域
        g_mask_crop = g_mask[rmin1:rmax1, cmin1:cmax1]
        pred_crop = pred[rmin1:rmax1, cmin1:cmax1]
        overlap_ids = np.unique(pred_crop[g_mask_crop > 0])
        
        for pred_id in overlap_ids:
            if pred_id == 0: continue
            j = pred_id_list.index(pred_id)
            p_mask = pred_masks[pred_id]
            
            total = (g_mask + p_mask).sum()
            intersect = (g_mask * p_mask).sum()
            pairwise_inter[i, j] = intersect
            pairwise_union[i, j] = (g_mask + p_mask > 0).sum()

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    
    overall_inter = 0.0
    overall_union = 0.0
    
    if len(gt_id_list) > 0 and len(pred_id_list) > 0:
        # 为每个GT找到最佳匹配的Pred
        paired_pred_indices = np.argmax(pairwise_iou, axis=1)
        max_ious = np.max(pairwise_iou, axis=1)
        
        # 记录已匹配的 Pred ID
        matched_pred_indices = []
        
        for i, iou in enumerate(max_ious):
            if iou > 0:
                j = paired_pred_indices[i]
                overall_inter += pairwise_inter[i, j]
                overall_union += pairwise_union[i, j]
                matched_pred_indices.append(j)
            else:
                # GT 未匹配到任何 Pred，计入 Union
                overall_union += gt_masks[gt_id_list[i]].sum()
        
        # 处理未匹配的 Prediction (假阳性)
        unmatched_pred_indices = set(range(len(pred_id_list))) - set(matched_pred_indices)
        for j in unmatched_pred_indices:
            overall_union += pred_masks[pred_id_list[j]].sum()
            
    else:
        # 处理完全无匹配的情况
        for g in gt_id_list: overall_union += gt_masks[g].sum()
        for p in pred_id_list: overall_union += pred_masks[p].sum()

    return overall_inter / (overall_union + 1.0e-6)

def compute_pq_stats_single_image(pred, gt, match_iou=0.5):
    """计算单张图像的 PQ, SQ, DQ 统计数据"""
    if pred.max() <= 1: pred = measure.label(pred)
    if gt.max() <= 1: gt = measure.label(gt)

    pred_id_list = list(np.unique(pred))
    gt_id_list = list(np.unique(gt))
    if 0 in pred_id_list: pred_id_list.remove(0)
    if 0 in gt_id_list: gt_id_list.remove(0)

    # 构建 IoU 矩阵
    iou_matrix = np.zeros((len(gt_id_list), len(pred_id_list)))
    
    if len(gt_id_list) > 0 and len(pred_id_list) > 0:
        for i, gt_id in enumerate(gt_id_list):
            g_mask = (gt == gt_id).astype(np.uint8)
            rmin, rmax, cmin, cmax = get_bounding_box(g_mask)
            
            g_mask_crop = g_mask[rmin:rmax, cmin:cmax]
            pred_crop = pred[rmin:rmax, cmin:cmax]
            overlap_ids = np.unique(pred_crop[g_mask_crop > 0])
            
            for pred_id in overlap_ids:
                if pred_id == 0: continue
                j = pred_id_list.index(pred_id)
                p_mask = (pred == pred_id).astype(np.uint8)
                
                intersect = (g_mask * p_mask).sum()
                union = (g_mask + p_mask > 0).sum()
                iou_matrix[i, j] = intersect / (union + 1e-6)

        # 使用匈牙利算法进行最大匹配 (IoU > 0.5 实际上可以直接贪心，但匈牙利更通用)
        # 这里按照 tiseg 逻辑，对于 threshold >= 0.5，唯一匹配是确定的
        paired_gt_indices, paired_pred_indices = np.nonzero(iou_matrix > match_iou)
    else:
        paired_gt_indices, paired_pred_indices = [], []

    tp = len(paired_gt_indices)
    fp = len(pred_id_list) - tp
    fn = len(gt_id_list) - tp
    
    # 计算匹配对象的总 IoU
    iou_sum = 0.0
    if tp > 0:
        iou_sum = iou_matrix[paired_gt_indices, paired_pred_indices].sum()

    return tp, fp, fn, iou_sum

def iou(pr, gt, eps=1e-7, threshold=0.5):
    # 保持原有的语义分割 IoU 逻辑，但改为按 Batch 平均
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    
    # 修改为按样本计算 (dim=[1,2,3] 假设是 BCHW)
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()

def dice(pr, gt, eps=1e-7, threshold=0.5):
    # 保持原有的语义分割 Dice 逻辑
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()

def SegMetrics(pred, label, metrics):
    """
    计算分割指标。
    pred: (B, C, H, W) 或 (B, H, W) 的预测结果（logits、binary 或 instance map）
    label: (B, C, H, W) 或 (B, H, W) 的真实标签（binary 或 instance map）
    metrics: 字符串列表，如 ['mDice', 'mAJI', 'mPQ', 'mDQ', 'mSQ']
    """
    results = {}

    if isinstance(metrics, str):
        metrics = [metrics, ]
    

    # 1. 统一转换为 Tensor/Numpy 并在 CPU 上操作
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        

    if isinstance(label, list):
        label = np.array(label)
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
        

    # 2. 判断是否为实例图 (Max > 1)
    is_instance_map = (pred.max() > 1) or (label.max() > 1)
    

    if is_instance_map:
        # 如果已经是实例图，直接取整
        pr_np = pred.astype(int)
        gt_np = label.astype(int)
    else:
        # 如果是二值 Logits (0~1 或 负数)，需要阈值化
        # 注意：这里假设输入已经是 sigmoid 后的概率，或者 logits
        # 为了保险，如果 max <= 1 且 min < 0，做 sigmoid
        if pred.min() < 0:
            pred = 1 / (1 + np.exp(-pred)) # Sigmoid
            

        pr_np = (pred > 0.5).astype(int)
        gt_np = (label > 0.5).astype(int)



    # 3. 维度对齐 (B, H, W)
    if pr_np.ndim == 4: pr_np = pr_np.squeeze(1)
    if gt_np.ndim == 4: gt_np = gt_np.squeeze(1)
    if pr_np.ndim == 2: pr_np = pr_np[np.newaxis, ...]
    if gt_np.ndim == 2: gt_np = gt_np[np.newaxis, ...]
    

    # 4. 如果 GT 是二值图但 Pred 是实例图，需要对 GT 做连通域标记 (用于 AJI/PQ)
    # 这种情况常见于测试集 GT 是二值 mask 的情况
    if is_instance_map and gt_np.max() <= 1:
        gt_np_instance = np.zeros_like(gt_np)
        for i in range(gt_np.shape[0]):
            gt_np_instance[i] = measure.label(gt_np[i])
        gt_np = gt_np_instance



    batch_size = pr_np.shape[0]



    for metric in metrics:
        if metric == 'iou':
            # IoU 通常基于二值图
            pr_bin = (pr_np > 0).astype(int)
            gt_bin = (gt_np > 0).astype(int)
            intersection = (pr_bin * gt_bin).sum()
            union = pr_bin.sum() + gt_bin.sum() - intersection
            results['iou'] = (intersection + 1e-7) / (union + 1e-7)
            

        elif metric == 'dice' or metric == 'mDice':
            # Dice 基于二值图
            pr_bin = (pr_np > 0).astype(int)
            gt_bin = (gt_np > 0).astype(int)
            

            dice_sum = 0.0
            for i in range(batch_size):
                inter = (pr_bin[i] * gt_bin[i]).sum()
                union = pr_bin[i].sum() + gt_bin[i].sum()
                dice_sum += (2. * inter + 1e-7) / (union + 1e-7)
            results['dice'] = dice_sum / batch_size
            

        elif metric == 'mAJI':
            # AJI 必须基于实例图
            aji_sum = 0.0
            for i in range(batch_size):
                aji_sum += compute_aji_single_image(pr_np[i], gt_np[i])
            results['mAJI'] = aji_sum / batch_size
            

        elif metric in ['mPQ', 'mDQ', 'mSQ']:
            # PQ 必须基于实例图
            pq_list, dq_list, sq_list = [], [], []
            for i in range(batch_size):
                tp, fp, fn, iou_sum = compute_pq_stats_single_image(pr_np[i], gt_np[i])
                dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
                sq = iou_sum / (tp + 1e-6)
                pq = dq * sq
                pq_list.append(pq)
                dq_list.append(dq)
                sq_list.append(sq)
            

            if metric == 'mPQ': results['mPQ'] = np.mean(pq_list)
            if metric == 'mDQ': results['mDQ'] = np.mean(dq_list)
            if metric == 'mSQ': results['mSQ'] = np.mean(sq_list)



    return results