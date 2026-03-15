import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGuidedOT(nn.Module):
    def __init__(self, img_dim=256, txt_dim=512, epsilon=0.05, sinkhorn_iters=3):
        super().__init__()
        # 将外部传入的超参数保存为实例属性，便于随时查阅和调试
        self.epsilon = epsilon
        self.iters = sinkhorn_iters

        # 1. 维度对齐
        self.text_proj = nn.Linear(txt_dim, img_dim)

        # 2. FiLM 调制层 (用文本语义生成仿射变换参数)
        self.gamma_fc = nn.Linear(img_dim, img_dim)
        self.beta_fc = nn.Linear(img_dim, img_dim)

        # 3. 融合残差投影
        self.out_proj = nn.Conv2d(img_dim, img_dim, kernel_size=1)

        # 4. 🔥 极速热力图预测头 (替代原先臃肿的注意力生成器)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)  # 输出2通道: 正/负例
        )

    def forward(self, img_feat, txt_feat, density_map):
        B, C, H, W = img_feat.shape
        N = H * W

        # === 1. 安全屏障: 保证特征与密度图空间一致 ===
        if density_map.shape[-2:] != (H, W):
            density_map = F.interpolate(density_map, size=(H, W), mode='bilinear', align_corners=False)

        # === 2. 语义调制 (FiLM) ===
        txt_emb = self.text_proj(txt_feat)  # [B, C]
        gamma = self.gamma_fc(txt_emb).view(B, C, 1, 1)
        beta = self.beta_fc(txt_emb).view(B, C, 1, 1)

        # 文本"照亮"图像特征中与细胞核相关的通道
        mod_img_feat = img_feat * (1 + gamma) + beta
        source_feat = mod_img_feat.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # === 3. 物理密度引导的目标坑位 (Marginals) ===
        target_dist = F.relu(density_map.view(B, N)) + 1e-6
        q = target_dist / target_dist.sum(dim=1, keepdim=True)  # Target: 坑位
        p = torch.ones(B, N, device=img_feat.device) / N  # Source: 均分

        # === 4. 计算语义代价矩阵 (Cost Matrix) ===
        source_norm = F.normalize(source_feat, p=2, dim=2)
        sim_matrix = torch.bmm(source_norm, source_norm.transpose(1, 2))
        cost_matrix = 1.0 - sim_matrix

        # === 5. Sinkhorn 最优传输极速迭代 ===
        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones_like(p)
        v = torch.ones_like(q)

        for _ in range(self.iters):
            v = q / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = p / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

        T = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, N, N] 的传输矩阵

        # === 6. 特征搬运与残差融合 ===
        fused_source = torch.bmm(T.transpose(1, 2), source_feat)
        fused_feat = fused_source.permute(0, 2, 1).view(B, C, H, W)

        out_feat = img_feat + self.out_proj(fused_feat)

        # === 7. 生成点监督的热力图 ===
        heatmap_logits = self.heatmap_head(out_feat)

        return out_feat, heatmap_logits
