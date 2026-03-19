import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGuidedOT(nn.Module):
    def __init__(self, img_dim=256, txt_dim=512, epsilon=0.05, sinkhorn_iters=3, num_heads=8):
        super().__init__()
        self.epsilon = epsilon
        self.iters = sinkhorn_iters

        self.text_proj = nn.Linear(txt_dim, img_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(img_dim)

        self.out_proj = nn.Conv2d(img_dim, img_dim, kernel_size=1)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)
        )
        
        # 🔥 新增：HV 物理距离预测头 (随模型一起从头生长)
        self.hv_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, img_feat, txt_feat, density_map):
        B, C, H, W = img_feat.shape
        N = H * W

        if density_map.shape[-2:] != (H, W):
            density_map = F.interpolate(density_map, size=(H, W), mode='bilinear', align_corners=False)

        txt_emb = self.text_proj(txt_feat).unsqueeze(1) 
        img_flat = img_feat.view(B, C, N).permute(0, 2, 1) 
        
        attn_out, _ = self.cross_attn(query=img_flat, key=txt_emb, value=txt_emb)
        
        mod_img_flat = self.attn_norm(img_flat + attn_out)
        source_feat = mod_img_flat 
        mod_img_feat = mod_img_flat.permute(0, 2, 1).view(B, C, H, W)

        # =====================================================================
        # 🔥 核心防线：进入数值敏感区，强制关闭 AMP，提升到 FP32，防止 NaN
        # =====================================================================
        with torch.autocast('cuda', enabled=False):
            # 将所有参与运算的张量强制转为单精度 (float32)
            mod_img_feat_fp32 = mod_img_feat.float()
            txt_emb_fp32 = txt_emb.float()
            source_feat_fp32 = source_feat.float()
            density_map_fp32 = density_map.float()

            target_dist = F.relu(density_map_fp32.view(B, N)) + 1e-6
            q = target_dist / target_dist.sum(dim=1, keepdim=True) 

            img_norm = F.normalize(mod_img_feat_fp32, p=2, dim=1)
            txt_norm = F.normalize(txt_emb_fp32.permute(0, 2, 1).unsqueeze(-1), p=2, dim=1)
            cos_sim = torch.sum(img_norm * txt_norm, dim=1)
            
            # 🔥 关键修复：temperature 下限拉高到 0.01，绝对避免除以极小值导致 softmax 爆炸
            spatial_logits = cos_sim.view(B, N) / self.temperature.float().clamp(min=0.01)
            p = F.softmax(spatial_logits, dim=1)

            source_norm_1d = F.normalize(source_feat_fp32, p=2, dim=2)
            sim_matrix = torch.bmm(source_norm_1d, source_norm_1d.transpose(1, 2))
            cost_matrix = 1.0 - sim_matrix

            K = torch.exp(-cost_matrix / self.epsilon)
            u = torch.ones_like(p)
            v = torch.ones_like(q)

            # Sinkhorn 迭代 (在 FP32 下极度稳定)
            for _ in range(self.iters):
                v = q / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
                u = p / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

            T = u.unsqueeze(-1) * K * v.unsqueeze(1)
            fused_source = torch.bmm(T.transpose(1, 2), source_feat_fp32)
        # =====================================================================
        
        # 安全退出 FP32 保护区，降级回原本的精度 (bfloat16) 以维持训练速度
        fused_source = fused_source.to(img_feat.dtype)
        
        fused_feat = fused_source.permute(0, 2, 1).view(B, C, H, W).contiguous()
        out_feat = img_feat + self.out_proj(fused_feat)

        heatmap_logits = self.heatmap_head(out_feat)
        hv_logits = self.hv_head(out_feat)

        return out_feat, heatmap_logits, hv_logits