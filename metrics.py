import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from skimage import measure

def get_fast_aji(true, pred):
    """
    使用矩阵运算加速 AJI 计算
    true: (H, W) 实例图
    pred: (H, W) 实例图
    """
    true = np.copy(true)
    pred = np.copy(pred)
    
    # 确保 ID 是连续的，方便矩阵索引
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # 移除背景
    if 0 in true_id_list: true_id_list.remove(0)
    if 0 in pred_id_list: pred_id_list.remove(0)
    
    if len(true_id_list) == 0 and len(pred_id_list) == 0: return 1.0
    if len(true_id_list) == 0 or len(pred_id_list) == 0: return 0.0

    # 重映射 ID 到 0..N (为了构建混淆矩阵)
    # 这是一个优化技巧：将不连续的 ID 映射为连续索引
    t_map = {uid: i for i, uid in enumerate(true_id_list)}
    p_map = {uid: i for i, uid in enumerate(pred_id_list)}
    
    # 构建 Intersection 矩阵
    # 这种方法比双重 for 循环快得多
    # 我们只关心有重叠的像素
    mask = (true > 0) & (pred > 0)
    true_masked = true[mask]
    pred_masked = pred[mask]
    
    # 使用 bincount 或 histogram 计算重叠矩阵 (Intersection)
    # 这是一个稀疏矩阵的思想
    if len(true_masked) > 0:
        # 将二维坐标 (t_idx, p_idx) 编码为一维索引
        # idx = t_idx * num_pred + p_idx
        t_indices = np.array([t_map[t] for t in true_masked])
        p_indices = np.array([p_map[p] for p in pred_masked])
        
        flat_indices = t_indices * len(pred_id_list) + p_indices
        inter_counts = np.bincount(flat_indices, minlength=len(true_id_list)*len(pred_id_list))
        pairwise_inter = inter_counts.reshape(len(true_id_list), len(pred_id_list))
    else:
        pairwise_inter = np.zeros((len(true_id_list), len(pred_id_list)))

    # 计算 Union 矩阵
    # Union[i, j] = Area[i] + Area[j] - Inter[i, j]
    true_areas = np.array([np.sum(true == t) for t in true_id_list])
    pred_areas = np.array([np.sum(pred == p) for p in pred_id_list])
    
    # 利用广播机制计算 Union
    pairwise_union = true_areas[:, None] + pred_areas[None, :] - pairwise_inter

    # 匈牙利匹配 (AJI 核心)
    # 我们要最大化 Intersection 的总和 (或者说 Jaccard，但在 AJI 定义里通常是匹配 Intersection)
    # AJI 分子是匹配对的 Intersection 之和
    # AJI 分母是匹配对的 Union 之和 + 未匹配 GT 的面积 + 未匹配 Pred 的面积
    
    # 使用 linear_sum_assignment 寻找最大重叠匹配
    # cost 设为负的 intersection，求解最小 cost 等于最大 intersection
    row_ind, col_ind = linear_sum_assignment(-pairwise_inter)

    aji_numerator = 0.0
    aji_denominator = 0.0
    
    used_true_indices = set(row_ind)
    used_pred_indices = set(col_ind)

    # 累加匹配对
    for r, c in zip(row_ind, col_ind):
        # 只有真正有交集才算匹配
        if pairwise_inter[r, c] > 0:
            aji_numerator += pairwise_inter[r, c]
            aji_denominator += pairwise_union[r, c]
        else:
            # 虽然算法配对了，但交集为0，视为未匹配
            aji_denominator += true_areas[r]
            used_pred_indices.remove(c) # 这个 Pred 实际上没被用掉

    # 累加未匹配的 GT
    for i in range(len(true_id_list)):
        if i not in used_true_indices:
            aji_denominator += true_areas[i]
            
    # 累加未匹配的 Pred
    for j in range(len(pred_id_list)):
        if j not in used_pred_indices:
            aji_denominator += pred_areas[j]

    if aji_denominator == 0: return 0.0
    return aji_numerator / aji_denominator

def get_fast_pq(true, pred, match_iou=0.5):
    """
    使用矩阵运算加速 PQ 计算
    """
    true = np.copy(true)
    pred = np.copy(pred)
    
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if 0 in true_id_list: true_id_list.remove(0)
    if 0 in pred_id_list: pred_id_list.remove(0)
    
    if len(true_id_list) == 0 and len(pred_id_list) == 0: return 1.0, 1.0, 1.0
    if len(true_id_list) == 0 or len(pred_id_list) == 0: return 0.0, 0.0, 0.0

    # 类似 AJI 的矩阵构建逻辑
    t_map = {uid: i for i, uid in enumerate(true_id_list)}
    p_map = {uid: i for i, uid in enumerate(pred_id_list)}
    
    mask = (true > 0) & (pred > 0)
    true_masked = true[mask]
    pred_masked = pred[mask]
    
    if len(true_masked) > 0:
        t_indices = np.array([t_map[t] for t in true_masked])
        p_indices = np.array([p_map[p] for p in pred_masked])
        flat_indices = t_indices * len(pred_id_list) + p_indices
        inter_counts = np.bincount(flat_indices, minlength=len(true_id_list)*len(pred_id_list))
        pairwise_inter = inter_counts.reshape(len(true_id_list), len(pred_id_list))
    else:
        pairwise_inter = np.zeros((len(true_id_list), len(pred_id_list)))

    true_areas = np.array([np.sum(true == t) for t in true_id_list])
    pred_areas = np.array([np.sum(pred == p) for p in pred_id_list])
    pairwise_union = true_areas[:, None] + pred_areas[None, :] - pairwise_inter
    
    # 计算 IoU 矩阵
    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)

    # 匹配：IoU > 0.5
    # 因为阈值是 0.5，一个 GT 最多只能匹配一个 Pred，反之亦然，所以不需要匈牙利算法，直接贪心即可
    tp = 0
    iou_sum = 0
    
    # 获取所有满足条件的匹配索引
    # rows, cols 是匹配上的索引对
    rows, cols = np.where(pairwise_iou > match_iou)
    
    # 因为 > 0.5，所以不用去重，直接统计
    tp = len(rows)
    iou_sum = np.sum(pairwise_iou[rows, cols])
    
    fp = len(pred_id_list) - tp
    fn = len(true_id_list) - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = iou_sum / (tp + 1e-6)
    pq = dq * sq
    
    return pq, dq, sq

def SegMetrics(pred, label, metrics):
    """
    计算分割指标的总入口
    """
    results = {}

    if isinstance(metrics, str):
        metrics = [metrics, ]

    # 1. 转换为 Numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
        
    # 2. 维度处理 (B, H, W)
    # 如果输入是 (B, C, H, W)，压缩 C
    if pred.ndim == 4: pred = pred.squeeze(1)
    if label.ndim == 4: label = label.squeeze(1)
    # 如果输入是 (H, W)，增加 B
    if pred.ndim == 2: pred = pred[np.newaxis, ...]
    if label.ndim == 2: label = label[np.newaxis, ...]

    batch_size = pred.shape[0]

    # 3. 预处理 (二值化 & 实例标记)
    # 注意：我们在这里统一处理，避免后面重复计算
    
    # 3.1 二值图 (用于 Dice/IoU)
    # 假设 pred 已经是 logits 或 prob，需要阈值化
    # 如果 pred max > 1，说明已经是实例图，转化为二值图
    if pred.max() > 1:
        pr_bin = (pred > 0).astype(np.uint8)
    else:
        # 如果是 logits/prob
        if pred.min() < 0: pred = 1 / (1 + np.exp(-pred)) # Sigmoid
        pr_bin = (pred > 0.5).astype(np.uint8)

    if label.max() > 1:
        gt_bin = (label > 0).astype(np.uint8)
    else:
        gt_bin = (label > 0.5).astype(np.uint8)

    # 3.2 实例图 (用于 AJI/PQ)
    # 如果需要计算 AJI/PQ，且输入不是实例图，则进行连通域分析
    need_instance = any(m in metrics for m in ['mAJI', 'mPQ', 'mDQ', 'mSQ'])
    
    if need_instance:
        pr_inst_batch = []
        gt_inst_batch = []
        
        for i in range(batch_size):
            # 处理 Prediction
            if pred[i].max() > 1: # 已经是实例图
                pr_inst_batch.append(pred[i].astype(np.int32))
            else: # 需要生成实例图
                pr_inst_batch.append(measure.label(pr_bin[i]))
            
            # 处理 GT
            if label[i].max() > 1:
                gt_inst_batch.append(label[i].astype(np.int32))
            else:
                gt_inst_batch.append(measure.label(gt_bin[i]))

    # 4. 指标计算
    for metric in metrics:
        if metric == 'dice':
            dice_sum = 0.0
            for i in range(batch_size):
                inter = (pr_bin[i] * gt_bin[i]).sum()
                union = pr_bin[i].sum() + gt_bin[i].sum()
                dice_sum += (2. * inter + 1e-6) / (union + 1e-6)
            results['dice'] = dice_sum / batch_size
            
        elif metric == 'iou':
            iou_sum = 0.0
            for i in range(batch_size):
                inter = (pr_bin[i] * gt_bin[i]).sum()
                union = (pr_bin[i] | gt_bin[i]).sum()
                iou_sum += (inter + 1e-6) / (union + 1e-6)
            results['iou'] = iou_sum / batch_size

        elif metric == 'mAJI':
            aji_sum = 0.0
            for i in range(batch_size):
                aji_sum += get_fast_aji(gt_inst_batch[i], pr_inst_batch[i])
            results['mAJI'] = aji_sum / batch_size

        elif metric in ['mPQ', 'mDQ', 'mSQ']:
            pq_sum, dq_sum, sq_sum = 0.0, 0.0, 0.0
            for i in range(batch_size):
                pq, dq, sq = get_fast_pq(gt_inst_batch[i], pr_inst_batch[i])
                pq_sum += pq; dq_sum += dq; sq_sum += sq
            
            if metric == 'mPQ': results['mPQ'] = pq_sum / batch_size
            if metric == 'mDQ': results['mDQ'] = dq_sum / batch_size
            if metric == 'mSQ': results['mSQ'] = sq_sum / batch_size

    return results