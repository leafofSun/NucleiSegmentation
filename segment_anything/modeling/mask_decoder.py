# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List, Tuple, Type, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class HighFrequencyPromptEncoder(nn.Module):
    """
    高频形态提示编码器。

    输入:
        high_freq_prompt:
            来自 PNuRL 的高频形态特征，通常为 [B, C] 或 [B, N, C]。

    输出:
        layer_high_freq_prompts:
            layer_high_freq_prompts[0] -> 512 dim，对应 ResNet Stage2 / cnn_feat_s2。
            layer_high_freq_prompts[1] -> 256 dim，对应 ResNet Stage1 / cnn_feat_s1。

    约束:
        该模块只服务 high-frequency boundary branch。
        不参与 low-frequency semantic branch。
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

    def forward(self, high_freq_prompt: torch.Tensor) -> List[torch.Tensor]:
        if high_freq_prompt.dim() == 3:
            high_freq_prompt = high_freq_prompt.mean(dim=1)

        if high_freq_prompt.dim() != 2:
            raise ValueError(
                f"high_freq_prompt must be [B, C] or [B, N, C], "
                f"got {tuple(high_freq_prompt.shape)}"
            )

        if high_freq_prompt.shape[-1] != self.text_dim:
            raise ValueError(
                f"high_freq_prompt channel mismatch: expected {self.text_dim}, "
                f"got {high_freq_prompt.shape[-1]} from shape={tuple(high_freq_prompt.shape)}"
            )

        dtype = self.joint_fusion[0].weight.dtype
        device = self.joint_fusion[0].weight.device
        high_freq_prompt = high_freq_prompt.to(device=device, dtype=dtype)

        fused_high_freq_prompt = self.joint_fusion(high_freq_prompt)
        layer_high_freq_prompts = [
            proj(fused_high_freq_prompt) for proj in self.scale_projections
        ]

        return layer_high_freq_prompts


class ASRBlock(nn.Module):
    """
    频域解耦式 SAM 上采样模块。

    分支定义:
        1. low-frequency semantic branch
           x -> structure_upsample -> x_up
           low_freq_prompt -> low_freq_modulator -> gamma_low
           x_up = x_up * (1 + gamma_low)

        2. high-frequency boundary branch
           high_freq_prompt -> HighFrequencyPromptEncoder -> layer_high_freq_prompt
           layer_high_freq_prompt -> high_freq_modulator -> gamma_high
           cnn_feat = cnn_feat * (1 + gamma_high)

        3. CNN residual fusion
           cnn_feat -> cnn_proj -> c
           concat(x_up, c) -> cnn_residual_fusion -> detail
           x_up = x_up + residual_scale * detail

    稳定性设计:
        1. low_freq_modulator 最后一层 zero-init，初始 gamma_low = 0。
        2. high_freq_modulator 最后一层 zero-init，初始 gamma_high = 0。
        3. cnn_residual_fusion 最后一层 zero-init，初始 detail = 0。
        4. 初始状态接近原 SAM decoder。
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

        self.low_freq_modulator = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            activation(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.zeros_(self.low_freq_modulator[-1].weight)
        nn.init.zeros_(self.low_freq_modulator[-1].bias)
        self.low_freq_residual_scale = nn.Parameter(torch.tensor(1.0))

        if self.has_cnn:
            self.high_freq_modulator = nn.Sequential(
                nn.Linear(cnn_dim, cnn_dim),
                nn.LayerNorm(cnn_dim),
                activation(),
                nn.Linear(cnn_dim, cnn_dim),
            )
            nn.init.zeros_(self.high_freq_modulator[-1].weight)
            nn.init.zeros_(self.high_freq_modulator[-1].bias)
            self.high_freq_residual_scale = nn.Parameter(torch.tensor(1.0))

            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
            )

            self.cnn_residual_fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            )
            nn.init.zeros_(self.cnn_residual_fusion[-1].weight)

            self.residual_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _to_prompt_vector(
        prompt: torch.Tensor,
        target_batch: int,
        expected_dim: int,
        prompt_name: str,
    ) -> torch.Tensor:
        if prompt.dim() == 3:
            prompt = prompt.mean(dim=1)

        if prompt.dim() != 2:
            raise ValueError(
                f"{prompt_name} must be [B, C] or [B, N, C], got {tuple(prompt.shape)}"
            )

        if prompt.shape[0] == 1 and target_batch > 1:
            prompt = prompt.expand(target_batch, -1).contiguous()

        if prompt.shape[0] != target_batch:
            raise ValueError(
                f"{prompt_name} batch size mismatch: "
                f"prompt batch={prompt.shape[0]}, target batch={target_batch}"
            )

        if prompt.shape[-1] != expected_dim:
            raise ValueError(
                f"{prompt_name} channel mismatch: expected {expected_dim}, "
                f"got {prompt.shape[-1]} from shape={tuple(prompt.shape)}"
            )

        return prompt

    @staticmethod
    def _match_cnn_batch(cnn_feat: torch.Tensor, target_batch: int) -> torch.Tensor:
        if cnn_feat.shape[0] == 1 and target_batch > 1:
            return cnn_feat.expand(target_batch, -1, -1, -1).contiguous()

        if cnn_feat.shape[0] != target_batch:
            raise ValueError(
                f"cnn_feat batch mismatch: cnn batch={cnn_feat.shape[0]}, "
                f"target batch={target_batch}"
            )

        return cnn_feat

    def forward(
        self,
        x: torch.Tensor,
        cnn_feat: Optional[torch.Tensor] = None,
        low_freq_prompt: Optional[torch.Tensor] = None,
        layer_high_freq_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_up = self.structure_upsample(x)
        batch_size = x_up.shape[0]

        # 1. low-frequency semantic branch：只接收 low_freq_prompt。
        if low_freq_prompt is not None:
            low_freq_prompt = self._to_prompt_vector(
                low_freq_prompt,
                target_batch=batch_size,
                expected_dim=self.text_dim,
                prompt_name="low_freq_prompt",
            )
            low_freq_prompt = low_freq_prompt.to(
                device=x_up.device,
                dtype=self.low_freq_modulator[0].weight.dtype,
            )

            gamma_low = self.low_freq_modulator(low_freq_prompt).to(dtype=x_up.dtype)
            low_scale = self.low_freq_residual_scale.to(device=x_up.device, dtype=x_up.dtype)

            gamma_low = torch.tanh(gamma_low) * low_scale
            gamma_low = gamma_low.unsqueeze(-1).unsqueeze(-1)

            x_up = x_up * (1.0 + gamma_low)

        # 2. high-frequency boundary branch：只接收 layer_high_freq_prompt 调制 CNN feature。
        if self.has_cnn and cnn_feat is not None:
            cnn_feat = self._match_cnn_batch(cnn_feat, batch_size)
            cnn_feat = cnn_feat.to(device=x_up.device, dtype=x_up.dtype)

            if layer_high_freq_prompt is not None:
                layer_high_freq_prompt = self._to_prompt_vector(
                    layer_high_freq_prompt,
                    target_batch=batch_size,
                    expected_dim=self.cnn_dim,
                    prompt_name="layer_high_freq_prompt",
                )
                layer_high_freq_prompt = layer_high_freq_prompt.to(
                    device=cnn_feat.device,
                    dtype=self.high_freq_modulator[0].weight.dtype,
                )

                gamma_high = self.high_freq_modulator(layer_high_freq_prompt).to(dtype=cnn_feat.dtype)
                high_scale = self.high_freq_residual_scale.to(
                    device=cnn_feat.device,
                    dtype=cnn_feat.dtype,
                )

                gamma_high = torch.tanh(gamma_high) * high_scale
                gamma_high = gamma_high.unsqueeze(-1).unsqueeze(-1)

                cnn_feat = cnn_feat * (1.0 + gamma_high)

            c = self.cnn_proj(cnn_feat)

            if c.shape[-2:] != x_up.shape[-2:]:
                c = F.interpolate(
                    c,
                    size=x_up.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            detail = self.cnn_residual_fusion(torch.cat([x_up, c], dim=1))
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
    ) -> None:
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.use_asr = use_asr

        self.iou_token = nn.Embedding(1, transformer_dim)

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        if self.use_asr:
            self.asr_upscale_1 = ASRBlock(
                in_dim=transformer_dim,
                out_dim=transformer_dim // 4,
                cnn_dim=512,
                text_dim=512,
                activation=activation,
            )

            self.asr_upscale_2 = ASRBlock(
                in_dim=transformer_dim // 4,
                out_dim=transformer_dim // 8,
                cnn_dim=256,
                text_dim=512,
                activation=activation,
            )

            self.high_freq_prompt_encoder = HighFrequencyPromptEncoder(
                text_dim=512,
                cnn_dims=[512, 256],
            )
        else:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
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
        low_freq_prompt: Optional[torch.Tensor] = None,
        high_freq_prompt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            cnn_feat_s1=cnn_feat_s1,
            cnn_feat_s2=cnn_feat_s2,
            low_freq_prompt=low_freq_prompt,
            high_freq_prompt=high_freq_prompt,
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
        low_freq_prompt: Optional[torch.Tensor] = None,
        high_freq_prompt: Optional[torch.Tensor] = None,
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

        # image_pe 通常 batch=1，这里按 tokens batch 扩展。
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)

        if self.use_asr:
            layer_high_freq_prompts = [None, None]

            # high_freq_prompt 只进入 high-frequency boundary branch。
            if high_freq_prompt is not None:
                layer_high_freq_prompts = self.high_freq_prompt_encoder(high_freq_prompt)

            upscaled_embedding = self.asr_upscale_1(
                src,
                cnn_feat=cnn_feat_s2,
                low_freq_prompt=low_freq_prompt,
                layer_high_freq_prompt=layer_high_freq_prompts[0],
            )

            upscaled_embedding = self.asr_upscale_2(
                upscaled_embedding,
                cnn_feat=cnn_feat_s1,
                low_freq_prompt=low_freq_prompt,
                layer_high_freq_prompt=layer_high_freq_prompts[1],
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