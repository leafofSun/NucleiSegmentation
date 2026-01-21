# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class ASRBlock(nn.Module):
    """
    🔥 [核心改进] 基于 WeaveSeg ASR 思想的自适应谱细化上采样模块
    替代普通的 ConvTranspose2d，用于恢复高频边界信息
    
    设计思路：
    - 低频流 (Structure): 来自 SAM Transformer 的特征（语义强，边界糊）
    - 高频流 (Detail): 来自 PNuRL 的 fused_low 特征（纹理强，边界清）
    - 融合: 使用 PNuRL 的特征作为 Guide，去"指导" SAM 完成上采样
    """
    def __init__(self, in_dim, out_dim, guide_dim=None, activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        # 1. 基础结构流 (Low-Frequency / Structure from SAM)
        self.structure_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm2d(out_dim),
            activation(),
        )
        
        # 2. 高频细节流 (High-Frequency / Boundary from PNuRL)
        self.has_guide = guide_dim is not None
        if self.has_guide:
            # 对齐通道
            self.guide_proj = nn.Sequential(
                nn.Conv2d(guide_dim, in_dim, kernel_size=1),
                activation(),
            )
            # 细节细化：使用 PixelShuffle 模拟高频锐化
            # WeaveSeg 使用了自适应滤波器，这里我们用更高效的 PixelShuffle 实现类似效果
            self.detail_refine = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # 上采样 2x: [B, out_dim*4, H, W] -> [B, out_dim, H*2, W*2]
                nn.Conv2d(out_dim, out_dim, kernel_size=1),  # 融合
            )
            
            # 门控机制：决定注入多少高频信息（关注 x 和 g 的差异区，即边界）
            self.gate = nn.Sequential(
                nn.Conv2d(in_dim * 2, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x, guide=None, density_map=None):
        """
        Args:
            x: [B, C, H, W] (SAM Transformer 特征 - 模糊)
            guide: [B, C_g, H, W] (PNuRL 融合特征 - Size+Shape+Density -> 形状稳定剂 🦴)
            density_map: [B, 1, H, W] (AutoPoint 热力图 -> 空间门控 🔦)
        
        Returns:
            [B, out_dim, H*2, W*2] 上采样后的特征
        """
        # 1. 基础语义上采样
        x_struct = self.structure_upsample(x)
        
        if self.has_guide and guide is not None:
            # 2. 处理向导特征（对齐空间尺寸）
            # guide 可能来自 64x64，需要确保与 x 的空间尺寸一致
            if guide.shape[-2:] != x.shape[-2:]:
                guide = F.interpolate(guide, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
            g = self.guide_proj(guide)  # [B, C, H, W]
            
            # 🔥 [核心创新: 空间-语义双重控制]
            if density_map is not None:
                # 对齐尺寸
                if density_map.shape[-2:] != g.shape[-2:]:
                    d_map = F.interpolate(density_map, size=g.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    d_map = density_map
                
                # ⚠️ 关键操作：Spatial Gating
                # 只有在密度高的地方，才注入"形状特征"。
                # 这保证了背景处不会因为注入了 Shape 特征而产生幻觉。
                g = g * d_map  # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
            
            # 3. 计算门控权重 (关注 x 和 g 的差异区，即边界)
            combined = torch.cat([x, g], dim=1)
            alpha = self.gate(combined)
            
            # 4. 注入高频细节
            # 使用 guide 加权后的特征进行锐化上采样
            x_detail_input = x + alpha * g
            x_high_freq = self.detail_refine(x_detail_input)
            
            # 5. 最终融合：结构 + 细节
            return x_struct + x_high_freq
        else:
            return x_struct


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
        use_asr: bool = True,  # 🔥 新增开关：是否使用 ASR 高频细化
        guide_dim: int = None,  # 🔥 PNuRL fused_low 的维度，默认 192 (3 * 256 // 4)
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 🔥 [核心修改] 替换 output_upscaling 为 ASR 模块
        self.use_asr = use_asr
        
        if self.use_asr:
            # 默认 guide_dim = 192 (3 * embed_dim // 4)，如果 embed_dim=256
            if guide_dim is None:
                guide_dim = 192  # 3 * (256 // 4)
            
            # 第一层：256 -> 64 (注入高频细节)
            self.asr_upscale_1 = ASRBlock(
                transformer_dim, 
                transformer_dim // 4, 
                guide_dim=guide_dim,
                activation=activation
            )
            # 第二层：64 -> 32 (常规上采样，不再需要 guide)
            self.asr_upscale_2 = ASRBlock(
                transformer_dim // 4, 
                transformer_dim // 8, 
                guide_dim=None,
                activation=activation
            )
        else:
            # 原版 SAM 上采样
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
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )  #256 256 4 3

    def forward(
        self,
        image_embeddings: torch.Tensor,   #[B, 256, 64, 64]
        image_pe: torch.Tensor,           #[1, 256, 64, 64]
        sparse_prompt_embeddings: torch.Tensor, #[B, 3, 256]
        dense_prompt_embeddings: torch.Tensor,  #[B, 256, 64, 64]
        multimask_output: bool,
        high_freq_features: torch.Tensor = None,  # 🔥 新增：PNuRL 的 fused_low 特征 [B, 192, 64, 64]
        density_map: torch.Tensor = None,  # 🔥 [New] AutoPoint 生成的密度图 [B, 1, 64, 64]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_freq_features=high_freq_features,  # 🔥 传递高频特征
            density_map=density_map,  # 🔥 [New] 传递密度图
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_freq_features: torch.Tensor = None,  # 🔥 新增参数
        density_map: torch.Tensor = None,  # 🔥 [New] 密度图参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  #iou_token:[1,256]  mask_tokens:[4,256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # 🔥 [核心修改] 使用 ASR 或原版上采样
        if self.use_asr:
            # 第一层：注入高频细节（如果有 guide）和密度图
            upscaled_embedding = self.asr_upscale_1(src, guide=high_freq_features, density_map=density_map)
            # 第二层：常规上采样（不需要密度图）
            upscaled_embedding = self.asr_upscale_2(upscaled_embedding)
        else:
            upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  #[1,4,32]

        b, c, h, w = upscaled_embedding.shape  #[1, 32, 256, 256]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
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
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x) #源码
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
