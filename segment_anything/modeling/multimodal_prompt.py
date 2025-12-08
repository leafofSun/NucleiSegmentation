"""
多模态提示模块 - 集成CLIP用于SAM-Med2D的多模态提示
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/conic/models/hovernet/net_desc.py
"""

from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip package not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class AttentionPool2d(nn.Module):
    """CLIP风格的注意力池化层"""
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


# 移除Bottleneck类，因为不再需要ResNet结构


class CLIPViT(nn.Module):
    """
    基于SAM ImageEncoderViT的CLIP ViT特征提取器
    直接使用SAM的ViT编码器，更强大且与SAM架构一致
    """
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        output_dim: int = 256,
        use_sam_encoder: bool = True,
    ):
        """
        Args:
            image_encoder: SAM的ImageEncoderViT实例，如果为None则创建新的
            output_dim: 输出特征维度
            use_sam_encoder: 是否直接使用SAM的编码器
        """
        super().__init__()
        self.use_sam_encoder = use_sam_encoder
        self.output_dim = output_dim
        
        if image_encoder is not None:
            # 直接使用SAM的image_encoder
            self.image_encoder = image_encoder
            # 冻结SAM编码器参数（可选）
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            # 如果没有提供，将在forward中从外部获取
            self.image_encoder = None
        
        # 特征投影层（如果需要调整维度）
        self.feature_proj = None

    def set_image_encoder(self, image_encoder: nn.Module):
        """设置SAM的image_encoder"""
        self.image_encoder = image_encoder
        # 冻结参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, image_encoder: Optional[nn.Module] = None) -> torch.Tensor:
        """
        使用SAM的ViT编码器提取特征
        
        Args:
            x: 输入图像 [B, 3, H, W]
            image_encoder: 可选的SAM image_encoder（如果self.image_encoder为None）
        
        Returns:
            features: ViT特征 [B, C, H', W']
        """
        # 优先使用传入的encoder，然后是self.image_encoder
        encoder = image_encoder if image_encoder is not None else self.image_encoder
        
        if encoder is None:
            raise ValueError("image_encoder must be provided either in __init__ or forward()")
        
        # 使用SAM的ViT编码器提取特征
        with torch.set_grad_enabled(False):
            features = encoder(x)  # [B, out_chans, H', W']
        
        # 如果需要调整维度
        if self.feature_proj is not None:
            features = self.feature_proj(features)
        elif features.shape[1] != self.output_dim:
            # 动态创建投影层
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(
                    features.shape[1], 
                    self.output_dim, 
                    kernel_size=1
                ).to(features.device)
            features = self.feature_proj(features)
        
        return features


class GlobalClassifier(nn.Module):
    """全局分类器，用于多任务分类"""
    def __init__(self, in_c, out_c):
        super(GlobalClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.ModuleList([
            self._make_fc_layer(in_c, out) for out in out_c
        ])
    
    def _make_fc_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False),
            nn.ReLU(),
            nn.Linear(in_c // 8, out_c, bias=False)
        )
    
    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.size(0), -1)
        outputs = [fc_layer(pool) for fc_layer in self.fc_layers]
        return outputs


class GlobalFeatureFusion(nn.Module):
    """全局特征融合模块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        total_in_c = in_c[0] + in_c[1] + in_c[2] + in_c[3] + in_c[4]
        self.fc = nn.Sequential(
            nn.Conv2d(total_in_c * in_c[5], out_c, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, global_feature, label):
        prob_list = []
        for i in range(len(global_feature)):
            prob_list.append(torch.softmax(global_feature[i], axis=1))
        prob = torch.cat(prob_list, axis=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class LabelAttention(nn.Module):
    """标签注意力模块"""
    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c[1], in_c[0], kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c[0], in_c[0], kernel_size=1, padding=0, bias=False)
        )

    def forward(self, feats, label):
        b, c = label.shape
        label = label.reshape(b, c, 1, 1)
        ch_attn = self.c1(label)
        ch_map = torch.sigmoid(ch_attn)
        feats = feats * ch_map
        ch_attn = ch_attn.reshape(ch_attn.shape[0], ch_attn.shape[1])
        return ch_attn, feats


class MultimodalPromptEncoder(nn.Module):
    """
    多模态提示编码器
    集成CLIP文本提示和SAM ViT图像特征，生成用于SAM的提示嵌入
    使用SAM的ImageEncoderViT而不是ResNet，更强大且架构一致
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_model_path: Optional[str] = None,
        use_global_features: bool = True,
        num_classes: int = 8,
        image_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_global_features = use_global_features
        
        # CLIP模型初始化（仅用于文本编码）
        if CLIP_AVAILABLE:
            try:
                # 加载CLIP模型用于文本编码
                if clip_model_path:
                    self.clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
                    try:
                        # 尝试加载预训练权重
                        checkpoint = torch.jit.load(clip_model_path, map_location='cpu')
                        state_dict = checkpoint.state_dict()
                        # 只加载文本编码器部分
                        text_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('transformer.') or k.startswith('token_embedding') or k.startswith('text_projection'):
                                text_state_dict[k] = v
                        if text_state_dict:
                            self.clip_model.load_state_dict(text_state_dict, strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load CLIP text encoder weights: {e}")
                else:
                    self.clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.clip_model = None
        else:
            self.clip_model = None
        
        # SAM ViT特征提取器（使用SAM的ImageEncoderViT）
        self.clip_vit = CLIPViT(
            image_encoder=image_encoder,
            output_dim=embed_dim,
            use_sam_encoder=True,
        )
        
        # 文本特征投影层
        if self.clip_model is not None:
            text_dim = self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else 512
            self.text_proj = nn.Linear(text_dim, embed_dim)
        else:
            # 如果没有CLIP，使用简单的文本嵌入
            self.text_proj = nn.Linear(512, embed_dim)
        
        # 全局特征相关模块（使用SAM ViT的embed_dim，通常是256）
        if use_global_features:
            # SAM ViT的输出维度是embed_dim（通常是256），而不是1024
            self.global_classifier = GlobalClassifier(embed_dim, [1, 3, 3, 6, 5])
            self.global_fc = GlobalFeatureFusion([1, 3, 3, 6, 5, embed_dim], embed_dim)
            self.label_attention = LabelAttention([embed_dim, embed_dim])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 图像特征投影层（用于维度对齐，当不使用全局特征时）
        # 注意：实际维度会在forward中动态确定，这里先设为None
        self.image_proj = None
        self._image_feat_dim = None

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """编码文本提示"""
        if self.clip_model is not None and CLIP_AVAILABLE:
            with torch.no_grad():
                text_tokens = clip.tokenize(text_prompts).to(next(self.parameters()).device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            # 如果没有CLIP，使用随机特征（实际应用中应该使用其他文本编码器）
            batch_size = len(text_prompts)
            text_features = torch.randn(batch_size, 512, device=next(self.parameters()).device)
        
        text_embed = self.text_proj(text_features)
        return text_embed

    def forward(
        self,
        image_features: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        global_labels: Optional[torch.Tensor] = None,
        raw_image: Optional[torch.Tensor] = None,
        image_encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        生成多模态提示嵌入
        
        Args:
            image_features: 图像特征 [B, C, H, W]（来自SAM的image_encoder，已经是ViT特征）
            text_prompts: 文本提示列表
            global_labels: 全局标签 [B, num_classes]
            raw_image: 原始图像 [B, 3, H, W]（可选，如果需要重新提取特征）
            image_encoder: SAM的image_encoder（可选，如果需要在forward中提取特征）
        
        Returns:
            prompt_embedding: 提示嵌入 [B, embed_dim]
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # 如果提供了原始图像和编码器，可以使用CLIPViT重新提取特征
        # 否则直接使用传入的image_features（已经是SAM ViT特征）
        if raw_image is not None and image_encoder is not None:
            # 使用SAM ViT重新提取特征
            vit_features = self.clip_vit(raw_image, image_encoder=image_encoder)
        else:
            # 直接使用传入的ViT特征
            vit_features = image_features
        
        # 文本特征
        if text_prompts is not None and self.clip_model is not None:
            text_embed = self.encode_text(text_prompts)  # [B, embed_dim]
        else:
            text_embed = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # 全局特征处理（使用SAM ViT特征）
        if self.use_global_features and global_labels is not None:
            # 使用全局分类器处理ViT特征
            global_logit = self.global_classifier(vit_features)
            # 使用全局特征融合
            global_features = self.global_fc(global_logit, global_labels)
            # 应用标签注意力
            _, enhanced_features = self.label_attention(vit_features, global_features)
            # 池化图像特征
            image_pooled = F.adaptive_avg_pool2d(enhanced_features, 1).view(batch_size, -1)
        else:
            # 简单池化ViT特征
            image_pooled = F.adaptive_avg_pool2d(vit_features, 1).view(batch_size, -1)
            feat_dim = image_pooled.shape[1]
            
            if feat_dim != self.embed_dim:
                # 动态创建投影层（如果需要）
                if self.image_proj is None or self._image_feat_dim != feat_dim:
                    self.image_proj = nn.Linear(feat_dim, self.embed_dim).to(device)
                    self._image_feat_dim = feat_dim
                image_pooled = self.image_proj(image_pooled)
        
        # 融合文本和图像特征
        combined = torch.cat([text_embed, image_pooled], dim=1)
        prompt_embedding = self.feature_fusion(combined)
        
        return prompt_embedding

