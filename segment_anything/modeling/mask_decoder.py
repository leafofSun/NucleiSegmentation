# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d

class ASRBlock(nn.Module):
    """
    🔥 [纯视觉消融版] ASR上采样模块 (CNN + ViT Hybrid)
    仅融合两种信息：
    1. x_struct: 来自 SAM 的低频全局语义特征
    2. cnn_feat: 来自 ResNet 的高频物理边缘特征 (Skip Connection)
    """
    def __init__(self, in_dim, out_dim, cnn_dim=None, activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        # 1. 基础结构上采样 (SAM 流)
        self.structure_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm2d(out_dim),
            activation(),
        )
        
        # 2. 真实物理边缘流 (ResNet Skip Connection)
        self.has_cnn = cnn_dim is not None
        if self.has_cnn:
            # 压缩 CNN 通道
            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
            )
            # 特征融合
            self.cnn_fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                activation()
            )
            # 🔧 [零初始化]: 确保初期 CNN 增量为 0，防止破坏预训练流形，平滑过渡
            nn.init.zeros_(self.cnn_fusion[0].weight)
            
            # 🔥 [防梯度爆炸优化]: 引入可学习的 Residual Scale 标量，初始化为 0.1
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, cnn_feat=None):
        # 1. 基础语义上采样
        x_up = self.structure_upsample(x)
        
        # 2. 注入真实物理边缘 (ResNet)
        if self.has_cnn and cnn_feat is not None:
            c = self.cnn_proj(cnn_feat)
            # 空间尺寸对齐
            if c.shape[-2:] != x_up.shape[-2:]:
                c = F.interpolate(c, size=x_up.shape[-2:], mode='bilinear', align_corners=False)
            
            # 拼接并计算真实的边缘增量
            detail = self.cnn_fusion(torch.cat([x_up, c], dim=1))
            
            # 🔥 获取动态缩放因子，并确保其设备和数据类型一致
            scale = self.residual_scale.to(x_up.device, dtype=x_up.dtype)
            
            # 残差融合 (引入动态缩放保护)
            x_up = x_up + (detail * scale)
            
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

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.use_asr = use_asr
        
        if self.use_asr:
            # 第一层上采样：1/16 -> 1/8 尺度，接收 ResNet Stage 2 (512通道)
            self.asr_upscale_1 = ASRBlock(
                transformer_dim, transformer_dim // 4, cnn_dim=512, activation=activation
            )
            # 第二层上采样：1/8 -> 1/4 尺度，接收 ResNet Stage 1 (256通道)
            self.asr_upscale_2 = ASRBlock(
                transformer_dim // 4, transformer_dim // 8, cnn_dim=256, activation=activation
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
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        cnn_feat_s1: torch.Tensor = None,  # 🔥 ResNet 1/4 特征
        cnn_feat_s2: torch.Tensor = None,  # 🔥 ResNet 1/8 特征
        # (屏蔽了所有与文本/密度相关的入参)
        **kwargs 
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            cnn_feat_s1=cnn_feat_s1,
            cnn_feat_s2=cnn_feat_s2,
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
        cnn_feat_s1: torch.Tensor = None,
        cnn_feat_s2: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        
        # 🔥 [核心修改] 仅注入物理边缘，剥离文本先验
        if self.use_asr:
            upscaled_embedding = self.asr_upscale_1(src, cnn_feat=cnn_feat_s2)
            upscaled_embedding = self.asr_upscale_2(upscaled_embedding, cnn_feat=cnn_feat_s1)
        else:
            upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x