"""
PNuRL (Prompting Nuclei Representation Learning)

当前版本的职责：
1. 预测 nuclei 相关物理属性，形成 PNuRL warmup 阶段的属性监督。
2. 将文本特征按频率语义解耦为：
   - low_freq_prompt：低频属性语义，偏 color / arrangement / density。
   - high_freq_prompt：高频形态语义，偏 shape / size。
3. 显式输出受控 semantic_delta，而不是 refined image embedding。
   PNuRL 不替换视觉特征，只提供病理语义残差增量：
       image_embeddings + SemanticChannelGate(semantic_delta)
4. semantic_delta 是相对 image_features 尺度的 bounded residual，避免 Stage C 中自由残差爆炸。
5. 输出 density_map，作为 density 辅助任务的监督对象。

返回协议：
{
    "semantic_delta": semantic_delta,
    "attr_logits": attr_logits,
    "density_map": density_map,
    "low_freq_prompt": low_freq_prompt,
    "high_freq_prompt": high_freq_prompt,
    "pnurl_loss": pnurl_loss,

    # diagnostics / later regularization
    "semantic_delta_ratio": semantic_delta_ratio,
    "semantic_delta_raw_norm": semantic_delta_raw_norm,
    "semantic_delta_direction_norm": semantic_delta_direction_norm,
    "semantic_delta_reg_loss": semantic_delta_reg_loss,
}
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeClassifier(nn.Module):
    """通用属性分类器。"""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hidden_dim = max(in_dim // 2, 16)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiScaleAttributeHead(nn.Module):
    """多尺度属性分类头，用于 Shape / Size / Density。"""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hidden_low = max(in_dim // 2, 16)
        hidden_low_out = max(in_dim // 4, 16)
        hidden_high = max(in_dim // 2, 16)

        self.shallow_branch = nn.Sequential(
            nn.Conv2d(in_dim, hidden_low, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_low, hidden_low_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_low_out),
            nn.ReLU(inplace=True),
        )

        self.deep_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_high),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Linear(hidden_high, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feat_low = self.shallow_branch(x)
        feat_high = self.deep_branch(x)
        logits = self.classifier(feat_high)

        # 保留浅层分支参与计算图，避免 DDP 在某些配置下误判 unused parameter。
        logits = logits + feat_low.sum() * 0.0
        return logits, [feat_low, feat_high]


class AttributeClassifiers(nn.Module):
    """
    属性顺序：
    0: color
    1: shape
    2: arrange
    3: size
    4: density
    """

    def __init__(self, in_dim: int, num_classes_per_attr: List[int]):
        super().__init__()
        assert len(num_classes_per_attr) == 5, "Must provide class counts for 5 attributes."

        self.heads = nn.ModuleList()
        self.multiscale_indices = {1, 3, 4}

        for i, num_classes in enumerate(num_classes_per_attr):
            if i in self.multiscale_indices:
                self.heads.append(MultiScaleAttributeHead(in_dim, num_classes))
            else:
                self.heads.append(AttributeClassifier(in_dim, num_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        logits_list = []
        visual_feats_low = []
        visual_feats_high = []

        for i, head in enumerate(self.heads):
            if i in self.multiscale_indices:
                logits, feats = head(x)
                logits_list.append(logits)

                if return_feats:
                    visual_feats_low.append(feats[0])
                    visual_feats_high.append(feats[1])
            else:
                logits = head(x)
                logits_list.append(logits)

        if return_feats:
            fused_low = torch.cat(visual_feats_low, dim=1) if visual_feats_low else None
            fused_high = torch.cat(visual_feats_high, dim=1) if visual_feats_high else None
            return logits_list, [fused_low, fused_high]

        return logits_list, None


class ControlledSemanticDeltaAdapter(nn.Module):
    """
    文本条件化的受控语义残差适配器。

    设计目标：
        Stage C 中 semantic_delta 不能是自由 4D 残差，否则会出现
        DeltaNorm 远大于 BaseNorm 并破坏视觉底盘的问题。

    当前实现：
        1. delta_projector 生成 raw_delta。
        2. tanh(raw_delta) 只保留有界残差方向，范围 [-1, 1]。
        3. 使用 image_features 的 RMS 作为尺度参考。
        4. 使用 semantic_delta_ratio 将残差限制为视觉特征尺度的一个小比例。
        5. 最后一层卷积 zero-init，初始 semantic_delta 为 0。
    """

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

        self.feat_dim = feat_dim
        self.text_dim = text_dim
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

        # 保留该名字，便于与现有 optimizer/checkpoint 命名耦合。
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        # zero-init residual branch：初始不破坏视觉路径。
        last_conv = self.delta_projector[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

        # ratio 初始为 init_delta_ratio / max_delta_ratio。
        ratio = self.init_delta_ratio / self.max_delta_ratio
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(ratio / (1.0 - ratio))

        nn.init.zeros_(self.ratio_head[-1].weight)
        nn.init.constant_(self.ratio_head[-1].bias, init_bias)

    @staticmethod
    def _rms(
        x: torch.Tensor,
        dims: Tuple[int, ...],
        keepdim: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float().pow(2), dim=dims, keepdim=keepdim) + eps)

    def forward(
        self,
        image_features: torch.Tensor,
        fused_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if image_features.dim() != 4:
            raise ValueError(f"image_features must be [B, C, H, W], got {tuple(image_features.shape)}")

        if fused_prompt.dim() != 2:
            raise ValueError(f"fused_prompt must be [B, C], got {tuple(fused_prompt.shape)}")

        B, C, _, _ = image_features.shape
        if C != self.feat_dim:
            raise ValueError(f"image_features channel mismatch: expected {self.feat_dim}, got {C}")

        if fused_prompt.shape[0] != B:
            raise ValueError(
                f"fused_prompt batch mismatch: prompt batch={fused_prompt.shape[0]}, image batch={B}"
            )

        if fused_prompt.shape[-1] != self.text_dim:
            raise ValueError(
                f"fused_prompt channel mismatch: expected {self.text_dim}, got {fused_prompt.shape[-1]}"
            )

        dtype = image_features.dtype
        device = image_features.device

        fused_prompt = fused_prompt.to(device=device, dtype=dtype)

        gamma, beta = self.text_affine(fused_prompt).chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        conditioned_features = image_features * (1.0 + gamma) + beta

        raw_delta = self.delta_projector(conditioned_features)

        # 有界方向，防止 raw_delta 通过幅度本身绕过 channel gate。
        delta_direction = torch.tanh(raw_delta)

        # 相对视觉特征尺度的比例约束。base_rms detach，避免 PNuRL 通过改变视觉底盘尺度逃避约束。
        base_rms = self._rms(
            image_features.detach(),
            dims=(1, 2, 3),
            keepdim=True,
            eps=self.eps,
        ).to(device=device, dtype=dtype)

        ratio_logits = self.ratio_head(fused_prompt.float()).to(device=device, dtype=dtype)
        semantic_delta_ratio = self.max_delta_ratio * torch.sigmoid(ratio_logits)
        semantic_delta_ratio = semantic_delta_ratio.view(B, 1, 1, 1)

        residual_scale = torch.clamp(
            self.residual_scale.to(device=device, dtype=dtype),
            min=0.0,
            max=self.max_residual_scale,
        )

        semantic_delta = delta_direction * base_rms * semantic_delta_ratio * residual_scale

        # 供 train.py 后续加入正则，也供 sam.py 记录诊断。
        semantic_delta_reg_loss = (
            semantic_delta.float() / (base_rms.float() + self.eps)
        ).pow(2).mean()

        diagnostics = {
            "semantic_delta_ratio": semantic_delta_ratio.detach().view(B),
            "semantic_delta_raw_norm": raw_delta.detach().float().norm(dim=1).mean(),
            "semantic_delta_direction_norm": delta_direction.detach().float().norm(dim=1).mean(),
            "semantic_delta_reg_loss": semantic_delta_reg_loss,
        }

        return semantic_delta, diagnostics


# 兼容旧 import / 旧命名。正式代码中建议使用 ControlledSemanticDeltaAdapter。
SemanticDeltaAdapter = ControlledSemanticDeltaAdapter


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
    ):
        super().__init__()

        self.feat_dim = embed_dim
        self.embed_dim = text_dim
        self.attr_loss_weight = attr_loss_weight
        self.normalize_text_features = normalize_text_features
        self.max_delta_ratio = float(max_delta_ratio)
        self.init_delta_ratio = float(init_delta_ratio)

        self.attribute_names = ["color", "shape", "arrange", "size", "density"]
        self.num_classes_per_attr = num_classes_per_attr

        self.attribute_classifiers = AttributeClassifiers(embed_dim, num_classes_per_attr)

        # low-frequency semantic group: Color + Arrange + Density
        num_low_freq_classes = (
            num_classes_per_attr[0]
            + num_classes_per_attr[2]
            + num_classes_per_attr[4]
        )
        self.low_freq_prob_proj = nn.Sequential(
            nn.Linear(num_low_freq_classes, text_dim),
            nn.Sigmoid(),
        )

        # high-frequency morphology group: Shape + Size
        num_high_freq_classes = num_classes_per_attr[1] + num_classes_per_attr[3]
        self.high_freq_prob_proj = nn.Sequential(
            nn.Linear(num_high_freq_classes, text_dim),
            nn.Sigmoid(),
        )

        # PNuRL 只生成受控 semantic_delta，不生成 refined_features。
        self.semantic_delta_adapter = ControlledSemanticDeltaAdapter(
            feat_dim=embed_dim,
            text_dim=text_dim,
            max_delta_ratio=max_delta_ratio,
            init_delta_ratio=init_delta_ratio,
        )

        # 密度回归头：输出非负 density map。
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

    def forward(
        self,
        image_features: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        attribute_labels: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        B, _, _, _ = image_features.shape
        device = image_features.device
        dtype = image_features.dtype

        text_embed = self._prepare_text_embed(
            text_embed=text_embed,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        # 1. 属性分类。
        attribute_logits, _ = self.attribute_classifiers(image_features, return_feats=False)
        probs_list = [F.softmax(logits.float(), dim=1).to(dtype=dtype) for logits in attribute_logits]

        # 2. 文本特征解耦：低频属性语义 / 高频形态语义。
        low_freq_probs = torch.cat([probs_list[0], probs_list[2], probs_list[4]], dim=1)
        low_freq_prompt = text_embed * self.low_freq_prob_proj(low_freq_probs.float()).to(dtype=dtype)

        high_freq_probs = torch.cat([probs_list[1], probs_list[3]], dim=1)
        high_freq_prompt = text_embed * self.high_freq_prob_proj(high_freq_probs.float()).to(dtype=dtype)

        if self.normalize_text_features:
            low_freq_prompt = F.normalize(low_freq_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)
            high_freq_prompt = F.normalize(high_freq_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        fused_prompt = low_freq_prompt + high_freq_prompt
        if self.normalize_text_features:
            fused_prompt = F.normalize(fused_prompt.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        # 3. 只生成受控病理语义残差，不替换 image_features。
        semantic_delta, delta_diagnostics = self.semantic_delta_adapter(image_features, fused_prompt)

        # 4. density auxiliary output。
        density_map = self.density_decoder(image_features)

        # 5. PNuRL attribute loss。
        pnurl_loss = image_features.new_tensor(0.0)
        if return_loss and attribute_labels is not None:
            pnurl_loss = self.compute_attribute_loss(attribute_logits, attribute_labels)
            pnurl_loss = pnurl_loss * self.attr_loss_weight

        attr_logits = {
            "color": attribute_logits[0],
            "shape": attribute_logits[1],
            "arrange": attribute_logits[2],
            "size": attribute_logits[3],
            "density": attribute_logits[4],
        }

        semantic_delta_reg_loss = delta_diagnostics["semantic_delta_reg_loss"]

        return {
            "semantic_delta": semantic_delta,
            "attr_logits": attr_logits,
            "density_map": density_map,
            "low_freq_prompt": low_freq_prompt,
            "high_freq_prompt": high_freq_prompt,
            "pnurl_loss": pnurl_loss,

            # Diagnostics / regularization hooks.
            "semantic_delta_ratio": delta_diagnostics["semantic_delta_ratio"],
            "semantic_delta_raw_norm": delta_diagnostics["semantic_delta_raw_norm"],
            "semantic_delta_direction_norm": delta_diagnostics["semantic_delta_direction_norm"],
            "semantic_delta_reg_loss": semantic_delta_reg_loss,
        }

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
            raise ValueError(
                f"text_embed batch size mismatch: got {text_embed.size(0)}, expected {batch_size}."
            )

        if text_embed.size(-1) != self.embed_dim:
            raise ValueError(
                f"text_embed dim mismatch: got {text_embed.size(-1)}, expected {self.embed_dim}."
            )

        if self.normalize_text_features:
            text_embed = F.normalize(text_embed.float(), dim=-1, eps=1e-6).to(dtype=dtype)

        return text_embed

    def _normalize_attribute_labels(
        self,
        labels: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        兼容以下几种输入：
        1. Tensor[B, 5]
        2. Tensor[5]，通常来自单样本
        3. list/tuple，长度为 5，每个元素是 Tensor[B]
        4. list/tuple，长度为 B，每个元素是 Tensor[5]
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)

            if labels.dim() == 0:
                raise ValueError("attribute_labels should contain 5 attributes, but got a scalar tensor.")

            if labels.dim() == 1:
                if labels.numel() == len(self.attribute_names) and batch_size == 1:
                    return [labels[i].view(1) for i in range(len(self.attribute_names))]
                raise ValueError(
                    f"Unsupported attribute_labels shape {tuple(labels.shape)} for batch_size={batch_size}."
                )

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

    def compute_attribute_loss(
        self,
        logits_list: List[torch.Tensor],
        labels: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor],
    ) -> torch.Tensor:
        device = logits_list[0].device
        batch_size = logits_list[0].shape[0]
        labels_list = self._normalize_attribute_labels(labels, batch_size, device)

        # Shape / Size / Density 对 nuclei 结构更关键，因此略高权重。
        weights = [1.0, 1.0, 1.0, 2.0, 2.0]

        total_loss = logits_list[0].new_tensor(0.0)
        valid_terms = 0

        for i, (logits, label) in enumerate(zip(logits_list, labels_list)):
            label = label.to(device=device).long().view(-1)

            if label.numel() == 1 and logits.shape[0] > 1:
                label = label.expand(logits.shape[0])

            if label.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"Attribute label batch mismatch at {self.attribute_names[i]}: "
                    f"label batch={label.shape[0]}, logits batch={logits.shape[0]}."
                )

            num_classes = logits.shape[1]
            valid_mask = (label >= 0) & (label < num_classes) & (label != 255)

            if valid_mask.any():
                loss_i = F.cross_entropy(logits[valid_mask].float(), label[valid_mask])
                weight_i = weights[i] if i < len(weights) else 1.0
                total_loss = total_loss + weight_i * loss_i
                valid_terms += 1

        if valid_terms == 0:
            return logits_list[0].new_tensor(0.0)

        return total_loss / valid_terms