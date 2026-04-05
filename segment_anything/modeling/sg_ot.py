import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityGuidedOT(nn.Module):
    """
    基于 DeNSe 论文思想的纯分布感知最优传输模块 (DG-OT)
    剔除了模糊的文本语义引导，完全依赖空间密度与特征几何结构
    """
    def __init__(self, img_dim=256, epsilon=0.05, sinkhorn_iters=3, num_heads=8):
        super().__init__()
        self.epsilon = epsilon
        self.iters = sinkhorn_iters

        # 核心改动 1：移除 text_proj，改为纯粹的 Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(img_dim)

        self.out_proj = nn.Conv2d(img_dim, img_dim, kernel_size=1)

        # 预测头保持不变
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)
        )
        
        self.hv_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)
        )
        
        # 空间代价矩阵权重
        self.spatial_weight = nn.Parameter(torch.ones(1) * 0.1)

    # 核心改动 2：前向传播移除 txt_feat 参数
    def forward(self, img_feat, density_map):
        B, C, H, W = img_feat.shape
        N = H * W

        # 密度图下采样对齐 (如果有必要)
        if density_map.shape[-2:] != (H, W):
            density_map = F.interpolate(density_map, size=(H, W), mode='bilinear', align_corners=False)

        img_flat = img_feat.view(B, C, N).permute(0, 2, 1) # [B, N, C]

        # =====================================================================
        # 数值敏感区：保持 FP32 进行 Sinkhorn 迭代
        # =====================================================================
        with torch.autocast('cuda', enabled=False):
            img_flat_fp32 = img_flat.float()
            density_map_fp32 = density_map.float()

            # --- 步骤 A: 构建目标分布 q 与源分布 p ---
            # 目标分布 q: 来源于密度图
            target_dist = F.relu(density_map_fp32.view(B, N)) + 1e-6
            q = target_dist / target_dist.sum(dim=1, keepdim=True) 

            # 源分布 p: 摒弃语义硬截断，直接使用空间均匀分布 (严格守恒)
            p = torch.ones(B, N, device=img_feat.device, dtype=torch.float32) / N

            # --- 步骤 B: 构建代价矩阵 (Cost Matrix) ---
            # 1. 图像特征自相似性距离 (Feature Cost)
            img_norm_1d = F.normalize(img_flat_fp32, p=2, dim=2)
            sim_matrix = torch.bmm(img_norm_1d, img_norm_1d.transpose(1, 2))
            feature_cost = 1.0 - sim_matrix # [B, N, N]

            # 2. 物理空间欧氏距离 (Spatial Cost)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H, device=img_feat.device),
                                            torch.linspace(0, 1, W, device=img_feat.device), indexing='ij')
            coords = torch.stack([grid_y, grid_x], dim=-1).view(N, 2) # [N, 2]
            spatial_cost = torch.cdist(coords, coords, p=2) 
            spatial_cost = spatial_cost / 1.41421356 # 归一化到 [0, 1]
            spatial_cost = spatial_cost.unsqueeze(0).expand(B, -1, -1) 

            # 联合代价
            cost_matrix = feature_cost + self.spatial_weight.float() * spatial_cost

            # --- 步骤 C: Log-Domain Sinkhorn 计算传输计划 T ---
            log_p = torch.log(p + 1e-12)
            log_q = torch.log(q + 1e-12)
            
            f = torch.zeros_like(p) 
            g = torch.zeros_like(q) 

            for _ in range(self.iters):
                arg_f = (g.unsqueeze(1) - cost_matrix) / self.epsilon
                f = self.epsilon * (log_p - torch.logsumexp(arg_f, dim=2))
                
                arg_g = (f.unsqueeze(2) - cost_matrix) / self.epsilon
                g = self.epsilon * (log_q - torch.logsumexp(arg_g, dim=1))

            T_args = (f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) / self.epsilon
            T = torch.exp(T_args) # [B, N, N]
            
            # --- 步骤 D: OT 调制特征聚合 (对应 DeNSe 公式5: G = T * V) ---
            # 利用传输计划将源图像特征重新分配到高密度区域
            guided_v_fp32 = torch.bmm(T.transpose(1, 2), img_flat_fp32)
        # =====================================================================
        
        # 安全退出 FP32 保护区
        guided_v = guided_v_fp32.to(img_feat.dtype)
        
        # 核心改动 3：使用 Self-Attention，Value 由 OT 调制的引导特征充当
        # Query = Image, Key = Image, Value = Guided Image
        attn_out, _ = self.self_attn(query=img_flat, key=img_flat, value=guided_v)
        
        mod_img_flat = self.attn_norm(img_flat + attn_out)
        mod_img_feat = mod_img_flat.permute(0, 2, 1).view(B, C, H, W).contiguous()

        # 残差连接
        out_feat = img_feat + self.out_proj(mod_img_feat)

        # 生成预测输出
        heatmap_logits = self.heatmap_head(out_feat)
        hv_logits = self.hv_head(out_feat)

        return out_feat, heatmap_logits, hv_logits

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
        
        # HV 物理距离预测头
        self.hv_head = nn.Sequential(
            nn.Conv2d(img_dim, img_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(img_dim // 2),
            nn.ReLU(),
            nn.Conv2d(img_dim // 2, 2, kernel_size=1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # 🔥 [新增参数] 空间代价矩阵的权重系数。设为可学习或常数均可，初始值为 0.5
        # 如果消融实验想退化为纯语义 OT，可在传参时将其置为 0.0
        self.spatial_weight = nn.Parameter(torch.ones(1) * 0.1)

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
        # 数值敏感区：保持 FP32 强制转换
        # =====================================================================
        with torch.autocast('cuda', enabled=False):
            mod_img_feat_fp32 = mod_img_feat.float()
            txt_emb_fp32 = txt_emb.float()
            source_feat_fp32 = source_feat.float()
            density_map_fp32 = density_map.float()

            target_dist = F.relu(density_map_fp32.view(B, N)) + 1e-6
            q = target_dist / target_dist.sum(dim=1, keepdim=True) 

            img_norm = F.normalize(mod_img_feat_fp32, p=2, dim=1)
            txt_norm = F.normalize(txt_emb_fp32.permute(0, 2, 1).unsqueeze(-1), p=2, dim=1)
            cos_sim = torch.sum(img_norm * txt_norm, dim=1)
            
            # 🔥 [修复 1: 源分布硬截断] 过滤负相关背景像素，避免背景特征被错误搬运
            spatial_logits = cos_sim.view(B, N) / self.temperature.float().clamp(min=0.01)
            mask = (cos_sim.view(B, N) > 0.0).float() # 仅保留语义正相关区域
            
            # 为保证数值稳定，减去最大值后计算 exp
            spatial_logits_shifted = spatial_logits - spatial_logits.max(dim=1, keepdim=True)[0]
            p_unnorm = torch.exp(spatial_logits_shifted) * mask + 1e-8 # 加上 1e-8 防止全图无正相关导致的 NaN
            p = p_unnorm / p_unnorm.sum(dim=1, keepdim=True)

            source_norm_1d = F.normalize(source_feat_fp32, p=2, dim=2)
            sim_matrix = torch.bmm(source_norm_1d, source_norm_1d.transpose(1, 2))
            feature_cost = 1.0 - sim_matrix

            # 🔥 [修复 2: 构建空间代价矩阵] 计算归一化物理欧氏距离
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H, device=img_feat.device),
                                            torch.linspace(0, 1, W, device=img_feat.device), indexing='ij')
            coords = torch.stack([grid_y, grid_x], dim=-1).view(N, 2) # [N, 2]
            spatial_cost = torch.cdist(coords, coords, p=2) # [N, N]
            spatial_cost = spatial_cost / 1.41421356 # 最大距离为 sqrt(2)，归一化到 [0, 1]
            spatial_cost = spatial_cost.unsqueeze(0).expand(B, -1, -1) # [B, N, N]

            # 联合特征代价与空间代价
            cost_matrix = feature_cost + self.spatial_weight.float() * spatial_cost

            # 🔥 [修复 3: Log-Domain Sinkhorn] 彻底解决 exp(-C/eps) 带来的数值溢出崩溃
            log_p = torch.log(p + 1e-12)
            log_q = torch.log(q + 1e-12)
            
            # 初始化对数域的潜变量 f 和 g (分别对应原算法中的 eps*log(u) 和 eps*log(v))
            f = torch.zeros_like(p) # [B, N]
            g = torch.zeros_like(q) # [B, N]

            for _ in range(self.iters):
                # Update f: f = eps * (log_p - logsumexp((g - C)/eps))
                arg_f = (g.unsqueeze(1) - cost_matrix) / self.epsilon # [B, N, N]
                f = self.epsilon * (log_p - torch.logsumexp(arg_f, dim=2)) # [B, N]
                
                # Update g: g = eps * (log_q - logsumexp((f - C)/eps))
                arg_g = (f.unsqueeze(2) - cost_matrix) / self.epsilon # [B, N, N]
                g = self.epsilon * (log_q - torch.logsumexp(arg_g, dim=1)) # [B, N]

            # 计算最终的传输计划 T: T_{ij} = exp((f_i + g_j - C_{ij}) / eps)
            T_args = (f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) / self.epsilon
            T = torch.exp(T_args)
            
            T_sum_j = T.sum(dim=1) # [B, N_target]
            fused_source_unnorm = torch.bmm(T.transpose(1, 2), source_feat_fp32) # [B, N, C]
            fused_source = fused_source_unnorm / (T_sum_j.unsqueeze(-1) + 1e-8)
        # =====================================================================
        
        # 安全退出 FP32 保护区，降级回原本的精度
        fused_source = fused_source.to(img_feat.dtype)
        
        fused_feat = fused_source.permute(0, 2, 1).view(B, C, H, W).contiguous()
        out_feat = img_feat + self.out_proj(fused_feat)

        heatmap_logits = self.heatmap_head(out_feat)
        hv_logits = self.hv_head(out_feat)

        return out_feat, heatmap_logits, hv_logits