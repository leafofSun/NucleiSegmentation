import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextGuidedPointGenerator(nn.Module):
    def __init__(self, embed_dim=256, text_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        self.img_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # [核心修复] 初始化为较大值 (log(1/0.07) ≈ 2.65)，且使用乘法
        # 这样初始的缩放因子 exp(2.65) ≈ 14.3，可以让 Logits 范围扩大到 [-14, 14]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embeddings, text_embeddings):
        B, C, H, W = image_embeddings.shape
        _, N_Classes, _ = text_embeddings.shape 
        
        img_feat = self.img_conv(image_embeddings) 
        txt_feat = self.text_proj(text_embeddings)
        
        # 归一化
        img_feat = F.normalize(img_feat, dim=1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        img_flat = img_feat.view(B, C, -1) 
        match_score = torch.bmm(txt_feat, img_flat)
        
        # [核心修复] 使用乘法放大 Logits，使其能覆盖 0~1 的概率空间
        # 限制最大值防止溢出 (Clamp)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        match_score = match_score * logit_scale
        
        heatmap_logits = match_score.view(B, N_Classes, H, W)
        return heatmap_logits

    def get_points_from_heatmap(self, heatmap_logits, topk=1):
        B, C, H, W = heatmap_logits.shape
        device = heatmap_logits.device
        all_points = []
        all_labels = []

        for b in range(B):
            batch_points = []
            batch_labels = []
            
            # Channel 0: Foreground (Nuclei) -> Label 1
            flat_fg = heatmap_logits[b, 0].view(-1)
            val, idx = torch.topk(flat_fg, k=topk)
            y = (idx // W).float()
            x = (idx % W).float()
            for i in range(topk):
                batch_points.append([x[i], y[i]])
                batch_labels.append(1) 
            
            # Channel 1: Background -> Label 0
            flat_bg = heatmap_logits[b, 1].view(-1)
            val, idx = torch.topk(flat_bg, k=topk)
            y = (idx // W).float()
            x = (idx % W).float()
            for i in range(topk):
                batch_points.append([x[i], y[i]])
                batch_labels.append(0)

            all_points.append(torch.tensor(batch_points, device=device))
            all_labels.append(torch.tensor(batch_labels, device=device))

        return torch.stack(all_points), torch.stack(all_labels)

# Loss 函数保持不变
def point_guidance_loss(pred_heatmap_logits, target_heatmap):
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