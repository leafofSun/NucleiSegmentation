import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import KDTree

class TextGuidedPointGenerator(nn.Module):
    def __init__(self, embed_dim=256, text_dim=512):
        super().__init__()
        # 1. 文本投影层
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # 2. 图像卷积层 (提取局部特征)
        self.img_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. Logit Scale (用于放大相似度，防止梯度消失)
        # 初始化为 log(1/0.07) ≈ 2.65
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embeddings, text_embeddings):
        """
        输入:
            image_embeddings: [B, 256, 64, 64] (SAM Image Encoder 输出)
            text_embeddings:  [B, N_Classes, 512] (CLIP Text Encoder 输出)
        输出:
            heatmap_logits:   [B, N_Classes, 64, 64]
        """
        B, C, H, W = image_embeddings.shape
        _, N_Classes, _ = text_embeddings.shape 
        
        # 特征提取与归一化
        img_feat = self.img_conv(image_embeddings) 
        txt_feat = self.text_proj(text_embeddings)
        
        img_feat = F.normalize(img_feat, dim=1)      # [B, 256, 64, 64]
        txt_feat = F.normalize(txt_feat, dim=-1)     # [B, N, 256]

        # 计算相似度矩阵 (Attention Map)
        img_flat = img_feat.view(B, C, -1)           # [B, 256, 4096]
        match_score = torch.bmm(txt_feat, img_flat)  # [B, N, 4096]
        
        # 缩放 Logits
        logit_scale = self.logit_scale.exp().clamp(max=100)
        match_score = match_score * logit_scale
        
        heatmap_logits = match_score.view(B, N_Classes, H, W)
        return heatmap_logits

    @torch.no_grad()
    def generate_adaptive_prompts(self, heatmap_logits, threshold=0.3, k_neighbors=3, dense_dist_thresh=15.0):
        """
        🔥 [核心功能] 密度自适应 + 邻域负提示采样 (Density-Adaptive Sampling)
        
        策略：
        1. 稀疏区: [1 正提示] (相信 SAM 的泛化能力)
        2. 拥挤区: [1 正提示 + K 负提示] (利用邻居作为负样本，切断粘连)
        
        Args:
            heatmap_logits: [B, C, H, W]
            threshold: 热力图阈值
            k_neighbors: 最多取多少个邻居作为负提示
            dense_dist_thresh: 判定为“拥挤”的距离阈值 (像素)
            
        Returns:
            prompts_list: List[Dict], 长度为 B。
                          每个元素包含:
                          - 'point_coords': [N_cells, K+1, 2]
                          - 'point_labels': [N_cells, K+1]
        """
        B, C, H, W = heatmap_logits.shape
        device = heatmap_logits.device
        
        # 1. 归一化并进行 NMS (非极大值抑制)，提取峰值点
        scores = torch.sigmoid(heatmap_logits)
        # MaxPool 做 NMS (窗口大小 5x5)
        local_max = F.max_pool2d(scores, kernel_size=5, stride=1, padding=2)
        is_local_max = (scores == local_max) & (scores > threshold)
        
        batch_prompts = []

        for b in range(B):
            # 获取当前图的所有前景点 (假设 Channel 0 是 Nuclei 前景)
            fg_map = is_local_max[b, 0] 
            y_inds, x_inds = torch.where(fg_map)
            
            # === 情况 A: 图中无细胞 ===
            if len(y_inds) == 0:
                # 返回空 tensor 防止报错
                batch_prompts.append({
                    "point_coords": torch.empty((0, k_neighbors + 1, 2), device=device),
                    "point_labels": torch.empty((0, k_neighbors + 1), device=device),
                    "has_points": False
                })
                continue
                
            # 构建坐标数组 [N, 2] (x, y) - 注意 SAM 需要 (x, y) 格式
            points_np = torch.stack([x_inds.float(), y_inds.float()], dim=1).cpu().numpy()
            num_points = len(points_np)
            
            # === 构建 KDTree 查找邻居 ===
            dists, indices = None, None
            if num_points > 1:
                tree = KDTree(points_np)
                # 查询最近的 k+1 个点 (第1个是自己，后k个是邻居)
                k_query = min(num_points, k_neighbors + 1)
                dists, indices = tree.query(points_np, k=k_query)

            # === 构造 Prompt (N 个细胞，每个细胞有一组 Points) ===
            image_point_coords = []
            image_point_labels = []

            for i in range(num_points):
                # 1. 正提示 (Self)
                current_pt = points_np[i]
                pts = [current_pt]
                lbls = [1] # 1 = Positive
                
                # 2. 密度判断
                is_crowded = False
                if dists is not None:
                    # dists[i, 1] 是离自己最近的邻居距离 (下标0是自己)
                    # 如果最近的邻居距离小于阈值，说明是拥挤区域
                    if len(dists[i]) > 1:
                        nearest_dist = dists[i, 1] 
                        if nearest_dist < dense_dist_thresh:
                            is_crowded = True
                
                # 3. 负提示注入 (Neighboring Negatives)
                if is_crowded:
                    # 遍历邻居 (跳过下标0，因为是自己)
                    for j in range(1, len(indices[i])):
                        neighbor_idx = indices[i][j]
                        neighbor_pt = points_np[neighbor_idx]
                        
                        pts.append(neighbor_pt)
                        lbls.append(0) # 0 = Negative (告诉 SAM 这里不是我)
                
                # 4. Padding (补齐到固定长度 k+1)
                # 必须 Pad 到固定长度才能 stack 成 Tensor
                while len(pts) < k_neighbors + 1:
                    pts.append([0.0, 0.0]) # Pad 坐标 (0,0)
                    lbls.append(-1)        # -1 = Ignore Label (SAM 会忽略此点)
                
                image_point_coords.append(pts)
                image_point_labels.append(lbls)

            # 转为 Tensor
            batch_prompts.append({
                # coords: [N_cells, K+1, 2]
                "point_coords": torch.tensor(np.array(image_point_coords), device=device).float(),
                # labels: [N_cells, K+1]
                "point_labels": torch.tensor(np.array(image_point_labels), device=device).long(),
                "has_points": True
            })
            
        return batch_prompts

    def get_points_from_heatmap(self, heatmap_logits, topk=1):
        """
        [旧方法] 简单的 Top-K 采样 (保留作为 fallback 或 baseline)
        仅用于简单验证，不具备密度自适应能力。
        """
        B, C, H, W = heatmap_logits.shape
        device = heatmap_logits.device
        all_points = []
        all_labels = []

        for b in range(B):
            flat_fg = heatmap_logits[b, 0].view(-1)
            val, idx = torch.topk(flat_fg, k=topk)
            y = (idx // W).float()
            x = (idx % W).float()
            
            batch_points = []
            batch_labels = []
            for i in range(topk):
                batch_points.append([x[i], y[i]])
                batch_labels.append(1) 
            
            all_points.append(torch.tensor(batch_points, device=device))
            all_labels.append(torch.tensor(batch_labels, device=device))

        return torch.stack(all_points), torch.stack(all_labels)

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