import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import KDTree

class TextGuidedPointGenerator(nn.Module):
    def __init__(self, embed_dim=256, text_dim=512, num_heads=8):
        super().__init__()
        # 1. 文本投影层
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # 2. 图像卷积层 (提取局部特征)
        self.img_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. 🔥 [新增] Cross-Attention 融合模块
        # batch_first=True 允许输入形状为 (Batch, Seq_Len, Dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # 加入 LayerNorm 稳定深层特征的训练
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 4. Logit Scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embeddings, text_embeddings):
        B, C, H, W = image_embeddings.shape
        _, N_Classes, _ = text_embeddings.shape 
        
        # 提取初始特征
        img_feat = self.img_conv(image_embeddings)  # [B, C, H, W]
        txt_feat = self.text_proj(text_embeddings)  # [B, N_Classes, C]
        
        # === 🔥 [新增] 跨模态深度交互 (Cross-Attention) ===
        # 1. 展平图像特征：[B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        img_flat = img_feat.flatten(2).transpose(1, 2)
        
        # 2. Attention 计算
        # Query: 图像的每个像素点; Key/Value: 文本特征
        attn_output, _ = self.cross_attn(query=img_flat, key=txt_feat, value=txt_feat)
        
        # 3. 残差连接与归一化 (防梯度消失)
        img_fused = self.layer_norm(img_flat + attn_output)
        # =================================================
        
        # 恢复通道维度在前，准备计算相似度: [B, H*W, C] -> [B, C, H*W]
        img_fused = img_fused.transpose(1, 2)
        
        # L2 归一化 (依然遵循 CLIP 的对比学习流形)
        img_norm = F.normalize(img_fused, dim=1)      
        txt_norm = F.normalize(txt_feat, dim=-1)     

        # 相似度矩阵乘法: [B, N_Classes, C] @ [B, C, H*W] -> [B, N_Classes, H*W]
        match_score = torch.bmm(txt_norm, img_norm)  
        
        logit_scale = self.logit_scale.exp().clamp(max=100)
        match_score = match_score * logit_scale
        
        # 还原为空间热力图
        heatmap_logits = match_score.view(B, N_Classes, H, W)
        return heatmap_logits

    @torch.no_grad()
    def count_peaks_from_heatmap(self, heatmap_logits, threshold=0.3):
        """
        从预测的热力图中统计峰值数量 $N_{pred}$
        
        Args:
            heatmap_logits: [B, C, H, W] 热力图 logits（未过 sigmoid）
            threshold: float 峰值检测阈值
        
        Returns:
            peak_counts: [B] 每个样本的峰值数量
        """
        B, C, H, W = heatmap_logits.shape
        scores = torch.sigmoid(heatmap_logits)  # [B, C, H, W]
        
        # 使用局部最大值检测峰值
        local_max = F.max_pool2d(scores, kernel_size=5, stride=1, padding=2)
        is_local_max = (scores == local_max) & (scores > threshold)
        
        # 统计每个样本的峰值数量（只统计前景类别，即 C=0）
        peak_counts = []
        for b in range(B):
            fg_map = is_local_max[b, 0]  # 前景热力图
            count = fg_map.sum().item()
            peak_counts.append(count)
        
        return torch.tensor(peak_counts, device=heatmap_logits.device, dtype=torch.float32)

    @torch.no_grad()
    def generate_adaptive_prompts(self, heatmap_logits, threshold=0.3, k_neighbors=3, dense_dist_thresh=15.0, max_points=None):
        """
        🔥 [核心修正] 全局邻域构建 + 随机采样训练 (Global Neighborhood + Random Sampling)
        
        逻辑流：
        1. 提取全图所有潜在细胞点 (All Points)。
        2. 基于 All Points 构建 KDTree，确保邻居关系的物理真实性 (Neighbor Integrity)。
        3. 如果训练需要限制数量 (max_points)，则从 All Points 中【随机采样】N 个作为目标。
           注意：这里使用 Random 而不是 Top-K，以保证模型见过"差生"(低置信度样本)。
        4. 为这 N 个目标构建 Prompt，其负提示来源于 KDTree (即来源于全集)。
        
        Args:
            dense_dist_thresh: float 或 torch.Tensor [B] - 每个样本的动态阈值
        """
        B, C, H, W = heatmap_logits.shape
        device = heatmap_logits.device
        
        # 🔥 [新增] 处理动态阈值：支持 float 或 Tensor
        if isinstance(dense_dist_thresh, torch.Tensor):
            # 确保是 1D tensor，长度为 B
            if dense_dist_thresh.dim() == 0:
                dense_dist_thresh = dense_dist_thresh.unsqueeze(0).expand(B)
            elif dense_dist_thresh.shape[0] != B:
                raise ValueError(f"dense_dist_thresh tensor length ({dense_dist_thresh.shape[0]}) must match batch size ({B})")
            # 转换为 numpy 数组以便后续使用
            dense_dist_thresh_np = dense_dist_thresh.cpu().numpy()
        else:
            # float 类型：为所有样本使用相同阈值
            dense_dist_thresh_np = np.full(B, float(dense_dist_thresh))
        
        # 1. NMS 提取所有点
        scores = torch.sigmoid(heatmap_logits)
        local_max = F.max_pool2d(scores, kernel_size=5, stride=1, padding=2)
        is_local_max = (scores == local_max) & (scores > threshold)
        
        batch_prompts = []

        for b in range(B):
            fg_map = is_local_max[b, 0] 
            y_inds, x_inds = torch.where(fg_map)
            
            # === 情况 A: 图中无细胞 ===
            if len(y_inds) == 0:
                batch_prompts.append({
                    "point_coords": torch.empty((0, k_neighbors + 1, 2), device=device),
                    "point_labels": torch.empty((0, k_neighbors + 1), device=device),
                    "has_points": False
                })
                continue
                
            # === 关键步骤 1: 获取全量点集 (Global Set) ===
            # 这些点既是潜在的 Target，也是潜在的 Negative Neighbor
            all_points_np = torch.stack([x_inds.float(), y_inds.float()], dim=1).cpu().numpy()
            total_num_points = len(all_points_np)
            
            # === 关键步骤 2: 基于全量点集构建 KDTree (保证邻居完整性) ===
            # 无论我们后面采样哪 50 个训练，找邻居必须在全集里找！
            tree = None
            dists_all, indices_all = None, None
            
            if total_num_points > 1:
                tree = KDTree(all_points_np)
                # 预先计算所有点的邻居信息 (查询 k+1 个，包含自己)
                k_query = min(total_num_points, k_neighbors + 1)
                dists_all, indices_all = tree.query(all_points_np, k=k_query)

            # === 关键步骤 3: 确定训练目标 (Target Selection) ===
            # 默认使用所有点
            target_indices = np.arange(total_num_points)
            
            # 如果点数超过限制，进行【随机采样】，而不是 Top-K
            # max_points 通常在训练时设为 50，验证时设为 None
            if max_points is not None and total_num_points > max_points:
                # 🔥 [策略修正] 使用随机采样，保证泛化性
                # replace=False 表示不重复采样
                target_indices = np.random.choice(total_num_points, max_points, replace=False)
                
            # === 关键步骤 4: 构建 Prompt (只针对选中的 Target) ===
            image_point_coords = []
            image_point_labels = []

            for i in target_indices:
                # 1. 正提示 (Self) - 来源于全集
                current_pt = all_points_np[i]
                pts = [current_pt]
                lbls = [1] # 1 = Positive
                
                # 2. 密度判断 & 负提示注入 (使用全集的 KDTree 结果)
                is_crowded = False
                if dists_all is not None:
                    # 获取第 i 个点的邻居距离信息
                    d_i = dists_all[i] # 距离数组
                    idx_i = indices_all[i] # 邻居索引数组
                    
                    if np.size(d_i) > 1:
                        # 兼容 shape
                        if d_i.ndim == 0: d_i = [d_i] 
                        
                        # dists_all[i][1] 是最近邻居的距离 (index 0 是自己)
                        if len(d_i) > 1:
                            nearest_dist = d_i[1]
                            # 🔥 [核心修改] 使用当前样本的动态阈值
                            current_thresh = dense_dist_thresh_np[b]
                            if nearest_dist < current_thresh:
                                is_crowded = True
                
                # 3. 负提示注入 (Neighboring Negatives)
                if is_crowded:
                    # 遍历邻居 (跳过下标0，因为是自己)
                    # 注意：idx_i 里面存的是在 all_points_np 中的下标
                    # 即使某个邻居没有被选进 target_indices，它依然会被加进来做负提示！✅
                    current_neighbors = idx_i if np.ndim(idx_i) > 0 else [idx_i]
                    
                    for j in range(1, len(current_neighbors)):
                        neighbor_idx = current_neighbors[j]
                        # 只有当 neighbor_idx 有效时
                        if neighbor_idx < total_num_points:
                            neighbor_pt = all_points_np[neighbor_idx]
                            pts.append(neighbor_pt)
                            lbls.append(0) # 0 = Negative
                
                # 4. Padding
                while len(pts) < k_neighbors + 1:
                    pts.append([0.0, 0.0]) 
                    lbls.append(-1)
                
                image_point_coords.append(pts)
                image_point_labels.append(lbls)

            # 转为 Tensor
            batch_prompts.append({
                "point_coords": torch.tensor(np.array(image_point_coords), device=device).float(),
                "point_labels": torch.tensor(np.array(image_point_labels), device=device).long(),
                "has_points": True
            })
            
        return batch_prompts

# === Loss 函数 ===
def point_guidance_loss(pred_heatmap_logits, target_heatmap):
    """
    pred_heatmap_logits: [B, C, H, W] (未过 sigmoid)
    target_heatmap:      [B, C, H, W] (DataLoader生成的椭圆热力图)
    """
    pred_prob = torch.sigmoid(pred_heatmap_logits)
    return focal_loss(pred_prob, target_heatmap)

def focal_loss(pred, target, alpha=2, beta=4):
    """
    CenterNet 风格 Focal Loss
    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)
    
    loss = 0
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss