# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List, Tuple, Type, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MorphologyEncoder(nn.Module):
    """
    形态学提示编码器。

    仅用于 freqpath ASR。
    legacy ASR 不使用该模块，避免语义/形态提示干扰纯视觉回归实验。
    """

    def __init__(self, text_dim: int = 512, cnn_dims: List[int] = [512, 256]):
        super().__init__()
        self.text_dim = text_dim
        self.cnn_dims = cnn_dims

        self.joint_fusion = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )

        self.scale_projections = nn.ModuleList(
            [nn.Linear(text_dim, dim) for dim in cnn_dims]
        )

    def forward(self, morph_feat: torch.Tensor) -> List[torch.Tensor]:
        if morph_feat.dim() == 3:
            morph_feat = morph_feat.mean(dim=1)

        if morph_feat.dim() != 2:
            raise ValueError(
                f"morph_feat must be [B, C] or [B, N, C], got {morph_feat.shape}"
            )

        dtype = self.joint_fusion[0].weight.dtype
        device = self.joint_fusion[0].weight.device
        morph_feat = morph_feat.to(device=device, dtype=dtype)

        joint_morph = self.joint_fusion(morph_feat)
        layer_prompts = [proj(joint_morph) for proj in self.scale_projections]
        return layer_prompts


class LegacyASRBlock(nn.Module):
    """
    旧版纯视觉 ASR。

    目标：
    1. 只验证 ResNet 高频细节分支 + SAM 上采样是否能复现旧结果。
    2. 不接收 attr_prompt。
    3. 不接收 morph_feat。
    4. 不做语义高频/低频调制。
    5. residual_scale 初始化为 0.1，降低 CNN 高频分支破坏 SAM 表征的风险。

    论文主线中的低频/高频语义约束不在这里做；
    这里是 baseline recovery / regression check。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cnn_dim: Optional[int] = None,
        text_dim: int = 512,  # 保留参数，只为和 FreqPathASRBlock 构造接口兼容
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cnn_dim = cnn_dim
        self.has_cnn = cnn_dim is not None

        self.structure_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm2d(out_dim),
            activation(),
        )

        if self.has_cnn:
            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
            )

            self.cnn_fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            )

            # 初始 detail = 0，保证刚接入 CNN 分支时不破坏原始 SAM 上采样。
            nn.init.zeros_(self.cnn_fusion[-1].weight)

            # 旧版 ASR 更保守，先让 CNN 高频细节小幅进入。
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        cnn_feat: Optional[torch.Tensor] = None,
        attr_prompt: Optional[torch.Tensor] = None,
        layer_morph_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # attr_prompt / layer_morph_prompt 故意不使用。
        x_up = self.structure_upsample(x)

        if self.has_cnn and cnn_feat is not None:
            cnn_feat = cnn_feat.to(device=x_up.device, dtype=x_up.dtype)
            c = self.cnn_proj(cnn_feat)

            if c.shape[-2:] != x_up.shape[-2:]:
                c = F.interpolate(
                    c,
                    size=x_up.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            detail = self.cnn_fusion(torch.cat([x_up, c], dim=1))
            scale = self.residual_scale.to(device=x_up.device, dtype=x_up.dtype)
            x_up = x_up + detail * scale

        return x_up


class FreqPathASRBlock(nn.Module):
    """
    论文主线 ASR：低频语义结构约束 + 高频形态/边界约束。

    这个模块保留给后续突破实验：
    1. 低频语义流：attr_prompt 调制 SAM structure upsample。
    2. 高频形态流：morph_feat 调制 CNN 高频边界特征。
    3. 再融合低频结构与高频边界。

    注意：
    只有当 legacy ASR 能复现旧 mAJI 后，才应该启用该模式。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cnn_dim: Optional[int] = None,
        text_dim: int = 512,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cnn_dim = cnn_dim
        self.text_dim = text_dim
        self.has_cnn = cnn_dim is not None

        self.structure_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm2d(out_dim),
            activation(),
        )

        self.attr_modulator = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            activation(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.zeros_(self.attr_modulator[-1].weight)
        nn.init.zeros_(self.attr_modulator[-1].bias)

        if self.has_cnn:
            self.morphology_modulator = nn.Sequential(
                nn.Linear(cnn_dim, cnn_dim),
                nn.LayerNorm(cnn_dim),
                activation(),
                nn.Linear(cnn_dim, cnn_dim),
            )
            nn.init.zeros_(self.morphology_modulator[-1].weight)
            nn.init.zeros_(self.morphology_modulator[-1].bias)

            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
            )

            self.cnn_fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            )
            nn.init.zeros_(self.cnn_fusion[-1].weight)

            # 这里也先改成 0.1，避免 freqpath 一启用就过强。
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _to_prompt_vector(prompt: torch.Tensor, target_batch: int) -> torch.Tensor:
        if prompt.dim() == 3:
            prompt = prompt.mean(dim=1)

        if prompt.dim() != 2:
            raise ValueError(
                f"Prompt must be [B, C] or [B, N, C], got {prompt.shape}"
            )

        if prompt.shape[0] == 1 and target_batch > 1:
            prompt = prompt.expand(target_batch, -1).contiguous()

        if prompt.shape[0] != target_batch:
            raise ValueError(
                f"Prompt batch size mismatch: "
                f"prompt batch={prompt.shape[0]}, target batch={target_batch}"
            )

        return prompt

    def forward(
        self,
        x: torch.Tensor,
        cnn_feat: Optional[torch.Tensor] = None,
        attr_prompt: Optional[torch.Tensor] = None,
        layer_morph_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_up = self.structure_upsample(x)
        batch_size = x_up.shape[0]

        # 低频语义结构调制。
        if attr_prompt is not None:
            attr_prompt = self._to_prompt_vector(attr_prompt, batch_size)
            attr_prompt = attr_prompt.to(
                device=x_up.device,
                dtype=self.attr_modulator[0].weight.dtype,
            )
            gamma_low = self.attr_modulator(attr_prompt).to(dtype=x_up.dtype)
            gamma_low = torch.tanh(gamma_low).unsqueeze(-1).unsqueeze(-1)
            x_up = x_up * (1.0 + gamma_low)

        # 高频 CNN 边界/形态调制。
        if self.has_cnn and cnn_feat is not None:
            cnn_feat = cnn_feat.to(device=x_up.device, dtype=x_up.dtype)

            if layer_morph_prompt is not None:
                layer_morph_prompt = self._to_prompt_vector(
                    layer_morph_prompt,
                    cnn_feat.shape[0],
                )
                layer_morph_prompt = layer_morph_prompt.to(
                    device=cnn_feat.device,
                    dtype=self.morphology_modulator[0].weight.dtype,
                )
                gamma_high = self.morphology_modulator(layer_morph_prompt).to(
                    dtype=cnn_feat.dtype
                )
                gamma_high = torch.tanh(gamma_high).unsqueeze(-1).unsqueeze(-1)
                cnn_feat = cnn_feat * (1.0 + gamma_high)

            c = self.cnn_proj(cnn_feat)

            if c.shape[-2:] != x_up.shape[-2:]:
                c = F.interpolate(
                    c,
                    size=x_up.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            detail = self.cnn_fusion(torch.cat([x_up, c], dim=1))
            scale = self.residual_scale.to(device=x_up.device, dtype=x_up.dtype)
            x_up = x_up + detail * scale

        return x_up


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_asr: bool = True,
        asr_variant: str = "legacy",
    ) -> None:
        super().__init__()

        if asr_variant not in ("legacy", "freqpath"):
            raise ValueError(
                f"asr_variant must be 'legacy' or 'freqpath', got {asr_variant}"
            )

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.use_asr = use_asr
        self.asr_variant = asr_variant

        self.iou_token = nn.Embedding(1, transformer_dim)

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        if self.use_asr:
            block_cls = LegacyASRBlock if asr_variant == "legacy" else FreqPathASRBlock

            self.asr_upscale_1 = block_cls(
                in_dim=transformer_dim,
                out_dim=transformer_dim // 4,
                cnn_dim=512,
                text_dim=512,
                activation=activation,
            )

            self.asr_upscale_2 = block_cls(
                in_dim=transformer_dim // 4,
                out_dim=transformer_dim // 8,
                cnn_dim=256,
                text_dim=512,
                activation=activation,
            )

            if asr_variant == "freqpath":
                self.morph_encoder = MorphologyEncoder(
                    text_dim=512,
                    cnn_dims=[512, 256],
                )
            else:
                self.morph_encoder = None

        else:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim,
                    transformer_dim // 4,
                    kernel_size=2,
                    stride=2,
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(
                    transformer_dim // 4,
                    transformer_dim // 8,
                    kernel_size=2,
                    stride=2,
                ),
                activation(),
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        cnn_feat_s1: Optional[torch.Tensor] = None,
        cnn_feat_s2: Optional[torch.Tensor] = None,
        attr_prompt: Optional[torch.Tensor] = None,
        morph_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            cnn_feat_s1=cnn_feat_s1,
            cnn_feat_s2=cnn_feat_s2,
            attr_prompt=attr_prompt,
            morph_feat=morph_feat,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        cnn_feat_s1: Optional[torch.Tensor] = None,
        cnn_feat_s2: Optional[torch.Tensor] = None,
        attr_prompt: Optional[torch.Tensor] = None,
        morph_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight],
            dim=0,
        )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0),
            -1,
            -1,
        )

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)

        if self.use_asr:
            if self.asr_variant == "legacy":
                # 纯视觉 ASR：只接 SAM 特征 + CNN 高频细节。
                upscaled_embedding = self.asr_upscale_1(
                    src,
                    cnn_feat=cnn_feat_s2,
                    attr_prompt=None,
                    layer_morph_prompt=None,
                )
                upscaled_embedding = self.asr_upscale_2(
                    upscaled_embedding,
                    cnn_feat=cnn_feat_s1,
                    attr_prompt=None,
                    layer_morph_prompt=None,
                )

            else:
                # 论文主线：低频语义 + 高频形态。
                layer_morph_prompts = [None, None]

                if morph_feat is not None and self.morph_encoder is not None:
                    layer_morph_prompts = self.morph_encoder(morph_feat)

                upscaled_embedding = self.asr_upscale_1(
                    src,
                    cnn_feat=cnn_feat_s2,
                    attr_prompt=attr_prompt,
                    layer_morph_prompt=layer_morph_prompts[0],
                )
                upscaled_embedding = self.asr_upscale_2(
                    upscaled_embedding,
                    cnn_feat=cnn_feat_s1,
                    attr_prompt=attr_prompt,
                    layer_morph_prompt=layer_morph_prompts[1],
                )

        else:
            upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []

        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )

        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape

        masks = (
            hyper_in @ upscaled_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

        h = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x), inplace=False)
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = torch.sigmoid(x)

        return x