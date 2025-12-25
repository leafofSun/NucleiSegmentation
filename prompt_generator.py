import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedPointGenerator(nn.Module):
    def __init__(self, embed_dim=256, text_dim=512): # 注意这里 text_dim=512 (CLIP standard)
        super().__init__()
        
        # ============================================================
        # 🔑 关键模块 1: 对齐层 (Alignment Layer)
        # ============================================================
        # 这一层负责“翻译”。
        # 它是可训练的！在训练过程中，它会学习如何把 CLIP 的文本特征
        # 映射到 SAM 的图像特征空间，使得它们可以进行数学交互。
        self.text_proj = nn.Linear(text_dim, embed_dim) # 512 -> 256
        
        # ============================================================
        # 🔑 关键模块 2: 融合层 (Fusion Layer)
        # ============================================================
        # 这里的输入是 concat 后的结果，我们将利用卷积层来进一步
        # 处理“对齐后”的特征。
        self.fusion_convs = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 预测热力图 (单通道)
        self.heatmap_head = nn.Conv2d(128, 1, kernel_size=1)
        # 初始化 bias 为负数，防止初期预测过多的前景点 (Focal Loss 常用技巧)
        self.heatmap_head.bias.data.fill_(-2.19)

    def forward(self, image_embeddings, text_embeddings):
        """
        image_embeddings: [B, 256, 64, 64] (来自 SAM，冻结的)
        text_embeddings:  [B, 512] (来自 CLIP，冻结的)
        """
        B, C, H, W = image_embeddings.shape
        
        # ------------------------------------------------------------
        # 1. 维度对齐 (Feature Alignment)
        # ------------------------------------------------------------
        # 将 CLIP 特征 (512) 投影到 SAM 空间 (256)
        # 这就是“翻译”过程
        text_feat = self.text_proj(text_embeddings) # [B, 512] -> [B, 256]
        
        # 扩展到图像尺寸，准备融合
        text_feat = text_feat.view(B, C, 1, 1) # [B, 256, 1, 1]
        
        # ------------------------------------------------------------
        # 2. 交互融合 (Interaction / Modulation)
        # ------------------------------------------------------------
        # 乘法融合 (显式对齐)
        # 类似于 Attention：用文本去“激活”图像中匹配的区域
        # 如果 text_proj 训练得好，这里的乘积就能高亮出目标细胞
        activated_features = image_embeddings * text_feat 
        
        # 拼接 (保留原始信息)
        # 把“激活后的特征”和“原始特征”拼在一起，防止信息丢失
        fusion_input = torch.cat([activated_features, image_embeddings], dim=1) # [B, 512, 64, 64]
        
        # ------------------------------------------------------------
        # 3. 生成热力图
        # ------------------------------------------------------------
        features = self.fusion_convs(fusion_input)
        heatmap_logits = self.heatmap_head(features)
        
        return heatmap_logits

    def get_coordinates_differentiable(self, heatmap_logits, temperature=1.0):
        """
        Spatial Soft-Argmax (保持不变，用于生成可导坐标)
        """
        B, _, H, W = heatmap_logits.shape
        device = heatmap_logits.device
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        flat_logits = heatmap_logits.view(B, -1)
        prob_map = F.softmax(flat_logits / temperature, dim=1).view(B, 1, H, W)
        
        pred_x = torch.sum(prob_map * grid_x, dim=[2, 3])
        pred_y = torch.sum(prob_map * grid_y, dim=[2, 3])
        
        return torch.cat([pred_x, pred_y], dim=1).unsqueeze(1)

# =====================================================================
# 3. 辅助 Loss 函数
# =====================================================================
def point_guidance_loss(pred_heatmap_logits, target_heatmap):
    """
    辅助损失：让生成的热力图去拟合高斯分布的 GT
    """
    pred_prob = torch.sigmoid(pred_heatmap_logits)
    return focal_loss(pred_prob, target_heatmap)

def focal_loss(pred, target, alpha=2, beta=4):
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