"""
PNuRL-v3: MGST Attribute Head + Class-Balanced/Focal Attribute Loss

目标：
1. 修复旧版属性头只依赖全局池化、难以学习 density / arrangement 的问题。
2. 引入 Multi-Grid Statistical Token (MGST)：全局统计 + 2x2 网格统计 + 网格极差。
3. 使用 class-balanced focal loss 抑制 majority-class cheating。
4. 保持原有对外接口兼容：
   - PNuRL.forward(...) 返回 semantic_delta / attr_logits / density_map / low_freq_prompt / high_freq_prompt / pnurl_loss。
   - 模块命名保持 attribute_classifiers / attribute_prompt_bank / density_decoder / semantic_delta_adapter，便于 train.py 参数组拆分。

属性顺序：
    [color, shape, arrange, size, density]
类别数默认：
    color=2, shape=3, arrange=2, size=3, density=3
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. MGST: Multi-Grid Statistical Token
# ==============================================================================
class MultiGridStatToken(nn.Module):
    """
    Multi-Grid Statistical Token (MGST).

    对输入特征 F in [B, C, H, W] 提取：
        global_avg:      C
        global_max:      C
        global_std:      C
        grid_mean_2x2:   4C
        grid_std_2x2:    4C
        grid_mean_range: C
        grid_std_range:  C

    总维度：13C。

    物理意义：
        density / arrangement 不是单纯全局属性，而是空间分布属性。
        2x2 网格统计和网格间极差能显式编码“有的区域密、有的区域空”的分布不均。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        grid_size: int = 2,
        dropout: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim or in_dim)
        self.grid_size = int(grid_size)
        self.eps = float(eps)

        stat_dim = self.in_dim * (3 + 2 * self.grid_size * self.grid_size + 2)
        hidden_dim = max(self.out_dim * 2, 128)

        self.projector = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"MGST expects [B, C, H, W], got {tuple(x.shape)}")

        x_float = x.float()
        B, C, _, _ = x_float.shape

        flat = x_float.flatten(2)
        global_avg = flat.mean(dim=-1)
        global_max = flat.max(dim=-1).values
        global_std = flat.std(dim=-1, unbiased=False)

        g = self.grid_size
        grid_mean = F.adaptive_avg_pool2d(x_float, output_size=(g, g))
        grid_sqmean = F.adaptive_avg_pool2d(x_float * x_float, output_size=(g, g))
        grid_var = torch.clamp(grid_sqmean - grid_mean * grid_mean, min=0.0)
        grid_std = torch.sqrt(grid_var + self.eps)

        grid_mean_flat = grid_mean.flatten(1)
        grid_std_flat = grid_std.flatten(1)

        grid_mean_per_c = grid_mean.flatten(2)
        grid_std_per_c = grid_std.flatten(2)

        grid_mean_range = grid_mean_per_c.max(dim=-1).values - grid_mean_per_c.min(dim=-1).values
        grid_std_range = grid_std_per_c.max(dim=-1).values - grid_std_per_c.min(dim=-1).values

        stat_token = torch.cat(
            [
                global_avg,
                global_max,
                global_std,
                grid_mean_flat,
                grid_std_flat,
                grid_mean_range,
                grid_std_range,
            ],
            dim=-1,
        )

        return self.projector(stat_token)


class MGSTAttributeHead(nn.Module):
    """单个属性分类头。"""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or max(in_dim // 2, 64))

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.classifier(token.float())


class AttributeClassifiers(nn.Module):
    """
    MGST 属性分类器。

    结构：
        image_features
            -> shared local conv encoder
            -> MGST token
            -> 5 个 attribute heads

    相比旧版 GAP-only / deep_branch-only：
        1. 保留局部卷积响应。
        2. 显式加入 2x2 网格均值/方差/极差。
        3. 对 density / arrangement 更友好。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes_per_attr: List[int],
        grid_size: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        if len(num_classes_per_attr) != 5:
            raise ValueError("AttributeClassifiers expects 5 attributes.")

        self.in_dim = int(in_dim)
        self.num_classes_per_attr = list(num_classes_per_attr)

        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
        )

        self.mgst = MultiGridStatToken(
            in_dim=in_dim,
            out_dim=in_dim,
            grid_size=grid_size,
            dropout=dropout,
        )

        self.heads = nn.ModuleList(
            [
                MGSTAttributeHead(in_dim=in_dim, num_classes=num_classes, dropout=dropout)
                for num_classes in num_classes_per_attr
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if x.dim() != 4:
            raise ValueError(f"image_features must be [B, C, H, W], got {tuple(x.shape)}")

        local_feat = self.local_encoder(x)
        token = self.mgst(local_feat)
        logits_list = [head(token) for head in self.heads]

        if return_feats:
            diagnostics = {
                "mgst_token_norm": token.detach().float().norm(dim=-1).mean(),
                "local_feat_norm": local_feat.detach().float().norm(dim=1).mean(),
            }
            return logits_list, diagnostics

        return logits_list, None


# ==============================================================================
# 2. Attribute Prompt Bank
# ==============================================================================
class AttributePromptBank(nn.Module):
    """
    PromptNu-style attribute semantic bank.

    使用属性概率加权属性类别 embedding，而不是旧版 text_embed * gate。

    low_freq_prompt:
        size + density + arrangement

    high_freq_prompt:
        shape + morphology text context

    color:
        默认不进入 high prompt，只作为极弱 stain/style residual。
    """

    def __init__(
        self,
        text_dim: int,
        num_classes_per_attr: List[int],
        color_high_scale: float = 0.0,
        text_context_scale: float = 0.10,
        dropout: float = 0.05,
    ):
        super().__init__()
        if len(num_classes_per_attr) != 5:
            raise ValueError("AttributePromptBank expects 5 attributes.")

        self.text_dim = int(text_dim)
        self.num_classes_per_attr = list(num_classes_per_attr)
        self.attribute_names = ["color", "shape", "arrange", "size", "density"]

        self.attr_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, text_dim) for num_classes in num_classes_per_attr]
        )

        self.low_fuse = nn.Sequential(
            nn.Linear(text_dim * 3, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim),
        )

        self.high_fuse = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim),
        )

        self.low_text_context = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
        )

        self.high_text_context = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
        )

        self.register_buffer(
            "color_high_scale",
            torch.tensor(float(color_high_scale), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "text_context_scale",
            torch.tensor(float(text_context_scale), dtype=torch.float32),
            persistent=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.attr_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _weighted_attribute_semantic(probs: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        if probs.dim() != 2:
            raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")

        weight = embedding.weight.to(device=probs.device, dtype=probs.dtype)
        if probs.shape[1] != weight.shape[0]:
            raise ValueError(
                f"Attribute prob/class mismatch: probs has {probs.shape[1]} classes, "
                f"embedding has {weight.shape[0]} classes."
            )

        return probs @ weight

    def forward(
        self,
        probs_list: List[torch.Tensor],
        text_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if len(probs_list) != 5:
            raise ValueError(f"Expected 5 probability tensors, got {len(probs_list)}.")

        semantics = [
            self._weighted_attribute_semantic(probs, emb)
            for probs, emb in zip(probs_list, self.attr_embeddings)
        ]

        color_sem = semantics[0]
        shape_sem = semantics[1]
        arrange_sem = semantics[2]
        size_sem = semantics[3]
        density_sem = semantics[4]

        dtype = size_sem.dtype
        device = size_sem.device

        low_input = torch.cat([size_sem, density_sem, arrange_sem], dim=-1)

        color_scale = self.color_high_scale.to(device=device, dtype=dtype).clamp(0.0, 0.20)
        high_input = torch.cat([shape_sem, color_sem * color_scale], dim=-1)

        low_prompt = self.low_fuse(low_input.float()).to(dtype=dtype)
        high_prompt = self.high_fuse(high_input.float()).to(dtype=dtype)

        if text_embed is not None:
            text_scale = self.text_context_scale.to(device=device, dtype=dtype).clamp(0.0, 0.50)
            text_embed = text_embed.to(device=device, dtype=dtype)
            low_prompt = low_prompt + text_scale * self.low_text_context(text_embed.float()).to(dtype=dtype)
            high_prompt = high_prompt + text_scale * self.high_text_context(text_embed.float()).to(dtype=dtype)

        diagnostics = {
            "color_sem_norm": color_sem.detach().float().norm(dim=-1).mean(),
            "shape_sem_norm": shape_sem.detach().float().norm(dim=-1).mean(),
            "arrange_sem_norm": arrange_sem.detach().float().norm(dim=-1).mean(),
            "size_sem_norm": size_sem.detach().float().norm(dim=-1).mean(),
            "density_sem_norm": density_sem.detach().float().norm(dim=-1).mean(),
            "low_prompt_raw_norm": low_prompt.detach().float().norm(dim=-1).mean(),
            "high_prompt_raw_norm": high_prompt.detach().float().norm(dim=-1).mean(),
        }

        return low_prompt, high_prompt, diagnostics


# ==============================================================================
# 3. Controlled semantic delta
# ==============================================================================
class ControlledSemanticDeltaAdapter(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        text_dim: int,
        reduction: int = 4,
        max_delta_ratio: float = 0.10,
        init_delta_ratio: float = 0.02,
        max_residual_scale: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        hidden_dim = max(feat_dim // reduction, 32)
        ratio_hidden_dim = max(text_dim // 4, 32)

        self.feat_dim = int(feat_dim)
        self.text_dim = int(text_dim)
        self.max_delta_ratio = float(max_delta_ratio)
        self.init_delta_ratio = float(init_delta_ratio)
        self.max_residual_scale = float(max_residual_scale)
        self.eps = float(eps)

        if not (0.0 < self.init_delta_ratio < self.max_delta_ratio):
            raise ValueError(
                f"init_delta_ratio must be in (0, max_delta_ratio), "
                f"got init_delta_ratio={init_delta_ratio}, max_delta_ratio={max_delta_ratio}"
            )

        self.text_affine = nn.Linear(text_dim, feat_dim * 2)

        self.delta_projector = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feat_dim, kernel_size=1, bias=True),
        )

        self.ratio_head = nn.Sequential(
            nn.Linear(text_dim, ratio_hidden_dim),
            nn.GELU(),
            nn.Linear(ratio_hidden_dim, 1),
        )

        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self):
        last_conv = self.delta_projector[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

        ratio = self.init_delta_ratio / self.max_delta_ratio
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(ratio / (1.0 - ratio))
        nn.init.zeros_(self.ratio_head[-1].weight)
        nn.init.constant_(self.ratio_head[-1].bias, init_bias)

    @staticmethod
    def _rms(x: torch.Tensor, dims: Tuple[int, ...], keepdim: bool = True, eps: float = 1e-6) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float().pow(2), dim=dims, keepdim=keepdim) + eps)

    def forward(self, image_features: torch.Tensor, fused_prompt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if image_features.dim() != 4:
            raise ValueError(f"image_features must be [B, C, H, W], got {tuple(image_features.shape)}")
        if fused_prompt.dim() != 2:
            raise ValueError(f"fused_prompt must be [B, C], got {tuple(fused_prompt.shape)}")

        B, C, _, _ = image_features.shape
        if C != self.feat_dim:
            raise ValueError(f"image_features channel mismatch: expected {self.feat_dim}, got {C}")
        if fused_prompt.shape[0] != B:
            raise ValueError(f"fused_prompt batch mismatch: prompt batch={fused_prompt.shape[0]}, image batch={B}")
        if fused_prompt.shape[-1] != self.text_dim:
            raise ValueError(f"fused_prompt channel mismatch: expected {self.text_dim}, got {fused_prompt.shape[-1]}")

        dtype = image_features.dtype
        device = image_features.device
        fused_prompt = fused_prompt.to(device=device, dtype=dtype)

        gamma, beta = self.text_affine(fused_prompt).chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        conditioned_features = image_features * (1.0 + gamma) + beta
        raw_delta = self.delta_projector(conditioned_features)
        delta_direction = torch.tanh(raw_delta)

        base_rms = self._rms(image_features.detach(), dims=(1, 2, 3), keepdim=True, eps=self.eps).to(
            device=device,
            dtype=dtype,
        )

        ratio_logits = self.ratio_head(fused_prompt.float()).to(device=device, dtype=dtype)
        semantic_delta_ratio = self.max_delta_ratio * torch.sigmoid(ratio_logits)
        semantic_delta_ratio = semantic_delta_ratio.view(B, 1, 1, 1)

        residual_scale = torch.clamp(
            self.residual_scale.to(device=device, dtype=dtype),
            min=0.0,
            max=self.max_residual_scale,
        )

        semantic_delta = delta_direction * base_rms * semantic_delta_ratio * residual_scale
        semantic_delta_reg_loss = (semantic_delta.float() / (base_rms.float() + self.eps)).pow(2).mean()

        diagnostics = {
            "semantic_delta_ratio": semantic_delta_ratio.detach().view(B),
            "semantic_delta_raw_norm": raw_delta.detach().float().norm(dim=1).mean(),
            "semantic_delta_direction_norm": delta_direction.detach().float().norm(dim=1).mean(),
            "semantic_delta_reg_loss": semantic_delta_reg_loss,
        }

        return semantic_delta, diagnostics


SemanticDeltaAdapter = ControlledSemanticDeltaAdapter


# ==============================================================================
# 4. Main PNuRL
# ==============================================================================
class PNuRL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        text_dim: int = 256,
        num_classes_per_attr: List[int] = [2, 3, 2, 3, 3],
        attr_loss_weight: float = 1.0,
        normalize_text_features: bool = True,
        max_delta_ratio: float = 0.10,
        init_delta_ratio: float = 0.02,
        # v3 loss controls
        use_class_balanced_loss: bool = True,
        use_focal_loss: bool = True,
        focal_gamma: float = 1.5,
        class_balanced_beta: float = 0.999,
        color_loss_weight: float = 0.05,
    ):
        super().__init__()

        self.feat_dim = int(embed_dim)
        self.embed_dim = int(text_dim)
        self.attr_loss_weight = float(attr_loss_weight)
        self.normalize_text_features = bool(normalize_text_features)
        self.max_delta_ratio = float(max_delta_ratio)
        self.init_delta_ratio = float(init_delta_ratio)

        self.use_class_balanced_loss = bool(use_class_balanced_loss)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_gamma = float(focal_gamma)
        self.class_balanced_beta = float(class_balanced_beta)

        self.attribute_names = ["color", "shape", "arrange", "size", "density"]
        self.num_classes_per_attr = list(num_classes_per_attr)

        self.attribute_classifiers = AttributeClassifiers(
            in_dim=embed_dim,
            num_classes_per_attr=num_classes_per_attr,
            grid_size=2,
            dropout=0.10,
        )

        self.attribute_prompt_bank = AttributePromptBank(
            text_dim=text_dim,
            num_classes_per_attr=num_classes_per_attr,
            color_high_scale=0.0,
            text_context_scale=0.10,
            dropout=0.05,
        )

        self.semantic_delta_adapter = ControlledSemanticDeltaAdapter(
            feat_dim=embed_dim,
            text_dim=text_dim,
            max_delta_ratio=max_delta_ratio,
            init_delta_ratio=init_delta_ratio,
        )

        self.density_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, 1, kernel_size=1),
            nn.Softplus(beta=1.0),
        )

        # 属性维度权重。color/stain 更像风格域因素，默认极低权重。
        dim_weights = torch.tensor(
            [float(color_loss_weight), 2.0, 2.0, 2.0, 2.0],
            dtype=torch.float32,
        )
        self.register_buffer("attribute_dim_weights", dim_weights, persistent=True)

        # 默认类别计数近似来自 v2 train split 的期望分布。
        # 后续如果 train.py 从 medical_knowledge_v2 统计出精确 counts，可调用 set_attr_class_counts 覆盖。
        default_counts = [
            torch.tensor([500.0, 500.0]),       # color
            torch.tensor([330.0, 330.0, 340.0]), # shape
            torch.tensor([660.0, 340.0]),       # arrange
            torch.tensor([330.0, 330.0, 340.0]), # size
            torch.tensor([360.0, 300.0, 340.0]), # density
        ]
        for i, counts in enumerate(default_counts):
            if i < len(num_classes_per_attr):
                k = num_classes_per_attr[i]
                c = counts[:k]
                if c.numel() < k:
                    c = F.pad(c, (0, k - c.numel()), value=float(c.mean().item()))
                self.register_buffer(f"attr_class_counts_{i}", c.float(), persistent=True)

    # ------------------------------------------------------------------
    # Optional external class-count override
    # ------------------------------------------------------------------
    def set_attr_class_counts(self, counts: Union[List[List[float]], Dict[str, List[float]]]):
        """允许 train.py 使用 medical_knowledge_v2 的真实训练集类别计数覆盖默认值。"""
        if isinstance(counts, dict):
            ordered = [counts.get(name, None) for name in self.attribute_names]
        else:
            ordered = counts

        if len(ordered) != len(self.attribute_names):
            raise ValueError(f"Expected counts for {len(self.attribute_names)} attributes, got {len(ordered)}")

        for i, item in enumerate(ordered):
            if item is None:
                continue
            tensor = torch.as_tensor(item, dtype=torch.float32)
            k = self.num_classes_per_attr[i]
            if tensor.numel() != k:
                raise ValueError(f"Count length mismatch for {self.attribute_names[i]}: expected {k}, got {tensor.numel()}")
            getattr(self, f"attr_class_counts_{i}").copy_(tensor.clamp_min(1.0))

    def _get_class_weights(self, attr_idx: int, labels: Optional[torch.Tensor], num_classes: int, device: torch.device) -> torch.Tensor:
        if self.use_class_balanced_loss and hasattr(self, f"attr_class_counts_{attr_idx}"):
            counts = getattr(self, f"attr_class_counts_{attr_idx}").to(device=device).float().clamp_min(1.0)
            counts = counts[:num_classes]
        else:
            # fallback：当前 batch 逆频率，做最小平滑，避免除零。
            counts = torch.ones(num_classes, device=device, dtype=torch.float32)
            if labels is not None and labels.numel() > 0:
                valid = labels[(labels >= 0) & (labels < num_classes) & (labels != 255)]
                if valid.numel() > 0:
                    counts = torch.bincount(valid, minlength=num_classes).float().to(device=device).clamp_min(1.0)

        beta = min(max(self.class_balanced_beta, 0.0), 0.999999)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, device=device), counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-8)
        weights = weights / weights.sum().clamp_min(1e-8) * float(num_classes)
        return weights.detach()

    def forward(
        self,
        image_features: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        attribute_labels: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        if image_features.dim() != 4:
            raise ValueError(f"image_features must be [B, C, H, W], got {tuple(image_features.shape)}")

        B, _, _, _ = image_features.shape
        device = image_features.device
        dtype = image_features.dtype

        has_external_text = text_embed is not None
        text_embed = self._prepare_text_embed(text_embed, B, device, dtype)
        text_context = text_embed if has_external_text else None

        attribute_logits, attr_diag = self.attribute_classifiers(image_features, return_feats=True)
        probs_list = [F.softmax(logits.float(), dim=1).to(dtype=dtype) for logits in attribute_logits]

        low_freq_prompt, high_freq_prompt, prompt_diagnostics = self.attribute_prompt_bank(
            probs_list=probs_list,
            text_embed=text_context,
        )

        if self.normalize_text_features:
            low_freq_prompt = F.normalize(low_freq_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)
            high_freq_prompt = F.normalize(high_freq_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        fused_prompt = low_freq_prompt + high_freq_prompt
        if self.normalize_text_features:
            fused_prompt = F.normalize(fused_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        semantic_delta, delta_diagnostics = self.semantic_delta_adapter(image_features, fused_prompt)
        density_map = self.density_decoder(image_features)

        pnurl_loss = image_features.new_tensor(0.0)
        attr_loss_dict: Dict[str, torch.Tensor] = {}

        if return_loss and attribute_labels is not None:
            pnurl_loss, attr_loss_dict = self.compute_attribute_loss(
                attribute_logits,
                attribute_labels,
                return_dict=True,
            )
            pnurl_loss = pnurl_loss * self.attr_loss_weight

        attr_logits = {
            "color": attribute_logits[0],
            "shape": attribute_logits[1],
            "arrange": attribute_logits[2],
            "size": attribute_logits[3],
            "density": attribute_logits[4],
        }

        out = {
            "semantic_delta": semantic_delta,
            "attr_logits": attr_logits,
            "density_map": density_map,
            "low_freq_prompt": low_freq_prompt,
            "high_freq_prompt": high_freq_prompt,
            "pnurl_loss": pnurl_loss,

            "semantic_delta_ratio": delta_diagnostics["semantic_delta_ratio"],
            "semantic_delta_raw_norm": delta_diagnostics["semantic_delta_raw_norm"],
            "semantic_delta_direction_norm": delta_diagnostics["semantic_delta_direction_norm"],
            "semantic_delta_reg_loss": delta_diagnostics["semantic_delta_reg_loss"],

            "low_prompt_raw_norm": prompt_diagnostics["low_prompt_raw_norm"],
            "high_prompt_raw_norm": prompt_diagnostics["high_prompt_raw_norm"],
            "size_sem_norm": prompt_diagnostics["size_sem_norm"],
            "density_sem_norm": prompt_diagnostics["density_sem_norm"],
            "arrange_sem_norm": prompt_diagnostics["arrange_sem_norm"],
            "shape_sem_norm": prompt_diagnostics["shape_sem_norm"],
            "color_sem_norm": prompt_diagnostics["color_sem_norm"],
        }

        if attr_diag is not None:
            out.update(attr_diag)

        for name, loss_value in attr_loss_dict.items():
            out[f"pnurl_loss_{name}"] = loss_value

        return out

    def _prepare_text_embed(
        self,
        text_embed: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if text_embed is None:
            return torch.zeros(batch_size, self.embed_dim, device=device, dtype=dtype)

        text_embed = text_embed.to(device=device, dtype=dtype)

        if text_embed.dim() == 3:
            text_embed = text_embed.mean(dim=1)
        elif text_embed.dim() == 1:
            text_embed = text_embed.unsqueeze(0)

        if text_embed.size(0) == 1 and batch_size > 1:
            text_embed = text_embed.expand(batch_size, -1)

        if text_embed.size(0) != batch_size:
            raise ValueError(f"text_embed batch size mismatch: got {text_embed.size(0)}, expected {batch_size}.")
        if text_embed.size(-1) != self.embed_dim:
            raise ValueError(f"text_embed dim mismatch: got {text_embed.size(-1)}, expected {self.embed_dim}.")

        if self.normalize_text_features:
            text_embed = F.normalize(text_embed.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        return text_embed

    def _normalize_attribute_labels(
        self,
        labels: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)

            if labels.dim() == 0:
                raise ValueError("attribute_labels should contain 5 attributes, but got a scalar tensor.")

            if labels.dim() == 1:
                if labels.numel() == len(self.attribute_names) and batch_size == 1:
                    return [labels[i].view(1) for i in range(len(self.attribute_names))]
                raise ValueError(f"Unsupported attribute_labels shape {tuple(labels.shape)} for batch_size={batch_size}.")

            if labels.dim() >= 2:
                if labels.shape[0] == batch_size and labels.shape[1] >= len(self.attribute_names):
                    return [labels[:, i].view(batch_size) for i in range(len(self.attribute_names))]
                if labels.shape[0] >= len(self.attribute_names) and labels.shape[1] == batch_size:
                    return [labels[i, :].view(batch_size) for i in range(len(self.attribute_names))]

            raise ValueError(f"Unsupported attribute_labels tensor shape: {tuple(labels.shape)}.")

        if isinstance(labels, (list, tuple)):
            labels = list(labels)

            if len(labels) == len(self.attribute_names):
                out = []
                for item in labels:
                    if not isinstance(item, torch.Tensor):
                        item = torch.as_tensor(item, device=device)
                    item = item.to(device)
                    if item.dim() == 0:
                        item = item.view(1)
                    if item.dim() > 1:
                        item = item.view(item.shape[0], -1)[:, 0]
                    out.append(item.view(-1))
                return out

            if len(labels) == batch_size:
                stacked = []
                for item in labels:
                    if not isinstance(item, torch.Tensor):
                        item = torch.as_tensor(item, device=device)
                    item = item.to(device)
                    stacked.append(item.view(-1))
                labels_tensor = torch.stack(stacked, dim=0)
                if labels_tensor.shape[1] < len(self.attribute_names):
                    raise ValueError(
                        f"Each attribute label should contain at least {len(self.attribute_names)} values, "
                        f"but got shape {tuple(labels_tensor.shape)}."
                    )
                return [labels_tensor[:, i].view(batch_size) for i in range(len(self.attribute_names))]

        raise TypeError(f"Unsupported attribute_labels type: {type(labels)}.")

    def _balanced_focal_ce(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        attr_idx: int,
    ) -> torch.Tensor:
        device = logits.device
        num_classes = logits.shape[1]
        label = label.to(device=device).long().view(-1)

        valid_mask = (label >= 0) & (label < num_classes) & (label != 255)
        if not valid_mask.any():
            return logits.new_tensor(0.0)

        logits_valid = logits[valid_mask].float()
        label_valid = label[valid_mask]

        class_weights = self._get_class_weights(attr_idx, label_valid, num_classes, device=device)

        ce = F.cross_entropy(
            logits_valid,
            label_valid,
            weight=class_weights,
            reduction="none",
        )

        if self.use_focal_loss and self.focal_gamma > 0:
            log_probs = F.log_softmax(logits_valid, dim=1)
            pt = log_probs.gather(dim=1, index=label_valid.view(-1, 1)).exp().view(-1)
            focal_factor = torch.pow(1.0 - pt.clamp(0.0, 1.0), self.focal_gamma)
            ce = focal_factor * ce

        return ce.mean()

    def compute_attribute_loss(
        self,
        logits_list: List[torch.Tensor],
        labels: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor],
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        device = logits_list[0].device
        batch_size = logits_list[0].shape[0]
        labels_list = self._normalize_attribute_labels(labels, batch_size, device)

        total_loss = logits_list[0].new_tensor(0.0)
        total_weight = 0.0
        loss_dict: Dict[str, torch.Tensor] = {}

        for i, (name, logits, label) in enumerate(zip(self.attribute_names, logits_list, labels_list)):
            label = label.to(device=device).long().view(-1)
            if label.numel() == 1 and logits.shape[0] > 1:
                label = label.expand(logits.shape[0])
            if label.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"Attribute label batch mismatch at {name}: "
                    f"label batch={label.shape[0]}, logits batch={logits.shape[0]}."
                )

            loss_i = self._balanced_focal_ce(logits, label, attr_idx=i)
            weight_i = float(self.attribute_dim_weights[i].detach().cpu().item())

            total_loss = total_loss + weight_i * loss_i
            total_weight += weight_i
            loss_dict[name] = loss_i.detach()

        if total_weight <= 0:
            total = logits_list[0].new_tensor(0.0)
        else:
            total = total_loss / float(total_weight)

        if return_dict:
            return total, loss_dict
        return total
