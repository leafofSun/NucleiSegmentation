#Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d

class MorphologyEncoder(nn.Module):
    """
    🔴 形态学提示编码器 (微观高频流)
    接收已融合的形态特征 (Shape+Size)，逐层派发给 CNN 高频特征流
    """
    def __init__(self, text_dim=256, cnn_dims=[512, 256]):
        super().__init__()
        # 1. 进一步潜空间映射提纯 (输入已经是融合后的 text_dim 维 txt_mor_feat)
        self.joint_fusion = nn.Sequential(
            nn.Linear(text_dim, text_dim), 
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # 2. 为不同层级的 CNN 生成专属的引导向量
        self.scale_projections = nn.ModuleList([
            nn.Linear(text_dim, dim) for dim in cnn_dims
        ])

    def forward(self, morph_feat):
        # 鲁棒性保护：如果输入是 Sequence [B, N, C]，则取均值降维到 [B, C]
        if morph_feat.dim() == 3:
            morph_feat = morph_feat.mean(dim=1)
            
        # 联合约束：对输入的形态学特征进行自适应提纯
        joint_morph = self.joint_fusion(morph_feat)
        
        # 逐层派发：生成对应深浅层级的引导向量
        layer_prompts = [proj(joint_morph) for proj in self.scale_projections]
        
        # 返回 [morph_prompt_s1 (512维), morph_prompt_s2 (256维)]
        return layer_prompts


class ASRBlock(nn.Module):
    """
    🔥 [频域解耦版] ASR上采样模块 (CNN + ViT Hybrid)
    1. x_up (Low-freq): 接收低频属性 (Color, Arrange, Density) 调制的全局语义
    2. cnn_feat (High-freq): 接收高频形态学 (Shape, Size) 提纯的物理边缘
    """
    def __init__(self, in_dim, out_dim, cnn_dim=None, text_dim=256, activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        # 1. 基础结构上采样 (SAM 低频语义流)
        self.structure_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm2d(out_dim),
            activation(),
        )
        
        # 🟢 宏观低频属性提示 (Attribute Prompt Modulator)
        self.attr_attn = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.Sigmoid() # 生成 0~1 的权重，调制全局感受野
        )
        
        # 2. 真实物理边缘流 (ResNet Skip Connection 高频流)
        self.has_cnn = cnn_dim is not None
        if self.has_cnn:
            # 🔴 微观高频形态提示 (Morphology Prompt Modulator)
            self.morphology_attn = nn.Sequential(
                nn.Linear(cnn_dim, cnn_dim),
                nn.Sigmoid() # 生成 0~1 的权重，裁剪/提纯CNN边缘
            )
            
            self.cnn_proj = nn.Sequential(
                nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
                LayerNorm2d(out_dim),
                activation(),
            )
            self.cnn_fusion = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                activation()
            )
            # 🔧 零初始化，确保残差结构的初始等效性（极其重要的稳定技巧）
            nn.init.zeros_(self.cnn_fusion[0].weight)
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, cnn_feat=None, attr_prompt=None, layer_morph_prompt=None):
        # 1. 基础语义上采样 (低频)
        x_up = self.structure_upsample(x)
        
        # 🟢 Attribute Prompt 调制低频特征 (感知环境/材质)
        if attr_prompt is not None:
            # 鲁棒性保护：适配 sequence 输入
            if attr_prompt.dim() == 3:
                attr_prompt = attr_prompt.mean(dim=1)
            attn_weight_low = self.attr_attn(attr_prompt).unsqueeze(-1).unsqueeze(-1)
            x_up = x_up * attn_weight_low  
        
        # 2. 注入真实物理边缘 (高频)
        if self.has_cnn and cnn_feat is not None:
            # 🔴 Morphology Prompt 调制高频特征 (裁剪指定形状和大小的边缘)
            if layer_morph_prompt is not None:
                # 鲁棒性保护
                if layer_morph_prompt.dim() == 3:
                    layer_morph_prompt = layer_morph_prompt.mean(dim=1)
                attn_weight_high = self.morphology_attn(layer_morph_prompt).unsqueeze(-1).unsqueeze(-1)
                cnn_feat = cnn_feat * attn_weight_high
                
            c = self.cnn_proj(cnn_feat)
            if c.shape[-2:] != x_up.shape[-2:]:
                c = F.interpolate(c, size=x_up.shape[-2:], mode='bilinear', align_corners=False)
            
            # 拼接并计算真实的边缘增量
            detail = self.cnn_fusion(torch.cat([x_up, c], dim=1))
            scale = self.residual_scale.to(x_up.device, dtype=x_up.dtype)
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
            # 统一使用 transformer_dim 作为 text_dim 传入，避免硬编码 256
            self.asr_upscale_1 = ASRBlock(
                transformer_dim, transformer_dim // 4, cnn_dim=512, text_dim=512, activation=activation
            )
            self.asr_upscale_2 = ASRBlock(
                transformer_dim // 4, transformer_dim // 8, cnn_dim=256, text_dim=512, activation=activation
            )
            # 🔴 初始化形态学编码器 (cnn_feat_s2 对应 512 维, cnn_feat_s1 对应 256 维)
            self.morph_encoder = MorphologyEncoder(text_dim=512, cnn_dims=[512, 256])
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
        cnn_feat_s1: torch.Tensor = None,  
        cnn_feat_s2: torch.Tensor = None,  
        attr_prompt: torch.Tensor = None,   # 🟢 低频宏观特征 (Color, Arrange, Density)
        morph_feat: torch.Tensor = None,    # 🔴 高频形态特征 (Shape + Size，即 PNuRL 的 txt_mor_feat)
        **kwargs 
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
        cnn_feat_s1: torch.Tensor = None,
        cnn_feat_s2: torch.Tensor = None,
        attr_prompt: torch.Tensor = None,
        morph_feat: torch.Tensor = None,
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
        
        if self.use_asr:
            # 🔴 构建逐层的形态学高频引导
            layer_morph_prompts = [None, None]
            if morph_feat is not None:
                layer_morph_prompts = self.morph_encoder(morph_feat) # 传入融合的形态特征
                
            # layer_morph_prompts[0] 对应 512维, 用于 cnn_feat_s2
            upscaled_embedding = self.asr_upscale_1(
                src, 
                cnn_feat=cnn_feat_s2, 
                attr_prompt=attr_prompt, 
                layer_morph_prompt=layer_morph_prompts[0]
            )
            # layer_morph_prompts[1] 对应 256维, 用于 cnn_feat_s1
            upscaled_embedding = self.asr_upscale_2(
                upscaled_embedding, 
                cnn_feat=cnn_feat_s1, 
                attr_prompt=attr_prompt, 
                layer_morph_prompt=layer_morph_prompts[1]
            )
        else:
            upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] =[]
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