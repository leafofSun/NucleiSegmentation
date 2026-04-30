"""
多模态提示模块 - 集成 CONCH/CLIP 用于 SAM-Med2D 的多模态提示
Modified from vqdang code at https://github.com/vqdang/hover_net/blob/conic/models/hovernet/net_desc.py
"""

from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 优先尝试导入 CONCH
try:
    from conch.open_clip_custom import tokenize
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False
    print("Warning: CONCH package not available. Falling back to CLIP if available.")

# 其次尝试导入原生 CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    if not CONCH_AVAILABLE:
        print("Warning: Neither CONCH nor CLIP package is available.")


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


class CLIPViT(nn.Module):
    """
    基于SAM ImageEncoderViT的特征提取器。
    在解耦架构中，这个提取器专门为宏观语义分类和全局 prompt 提供特征。
    不要直接将其输出用于 ASR 的高频指导。
    """
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        output_dim: int = 256,
        use_sam_encoder: bool = True,
    ):
        super().__init__()
        self.use_sam_encoder = use_sam_encoder
        self.output_dim = output_dim
        
        if image_encoder is not None:
            self.image_encoder = image_encoder
            # 冻结SAM编码器参数（可选）
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            self.image_encoder = None
        
        self.feature_proj = None

    def set_image_encoder(self, image_encoder: nn.Module):
        """设置SAM的image_encoder"""
        self.image_encoder = image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, image_encoder: Optional[nn.Module] = None) -> torch.Tensor:
        encoder = image_encoder if image_encoder is not None else self.image_encoder
        
        if encoder is None:
            raise ValueError("image_encoder must be provided either in __init__ or forward()")
        
        with torch.set_grad_enabled(False):
            features = encoder(x)  # [B, out_chans, H', W']
        
        if self.feature_proj is not None:
            features = self.feature_proj(features)
        elif features.shape[1] != self.output_dim:
            if self.feature_proj is None:
                self.feature_proj = nn.Conv2d(
                    features.shape[1], 
                    self.output_dim, 
                    kernel_size=1
                ).to(features.device)
            features = self.feature_proj(features)
        
        return features


class GlobalClassifier(nn.Module):
    """全局分类器，用于多任务分类 (例如器官分类)"""
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
        total_in_c = sum(in_c[:-1])  # 修正: 动态计算前面所有通道的总和
        self.fc = nn.Sequential(
            nn.Conv2d(total_in_c * in_c[-1], out_c, 1, bias=False), # 修正: in_c[-1] 通常是 embed_dim
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
    集成 CONCH/CLIP 文本提示和 SAM ViT 图像特征，生成用于 SAM 的提示嵌入。
    注意：在解耦架构中，此模块专司“宏观语义引导”，不再为 ASR 提供高频细节。
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_model_path: Optional[str] = None,
        use_global_features: bool = True,
        num_classes: int = 8,
        image_encoder: Optional[nn.Module] = None,
        is_conch: bool = False # 新增标识，用于区分是否使用 CONCH
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_global_features = use_global_features
        self.is_conch = is_conch
        
        self.clip_model = None
        self.tokenizer = None
        
        # 文本模型初始化 (支持 CONCH 和 CLIP)
        if self.is_conch and CONCH_AVAILABLE:
             # 注意：在实际应用中，由于 CONCH 加载需要 auth token，
             # 建议从外部传入已经加载好的 conch_model 和 tokenizer，或者保留为 None 并通过外部注入
             print("MultimodalPromptEncoder initialized in CONCH mode.")
             pass # CONCH model 应该在顶层加载并通过 forward 或 setter 传入
        elif CLIP_AVAILABLE and not self.is_conch:
            try:
                if clip_model_path:
                    self.clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
                    try:
                        checkpoint = torch.jit.load(clip_model_path, map_location='cpu')
                        state_dict = checkpoint.state_dict()
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
        
        # SAM ViT特征提取器
        self.clip_vit = CLIPViT(
            image_encoder=image_encoder,
            output_dim=embed_dim,
            use_sam_encoder=True,
        )
        
        # 文本特征投影层 (CONCH 和 CLIP ViT-B/16 的文本特征维度默认都是 512)
        if self.clip_model is not None and not self.is_conch:
            text_dim = self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else 512
            self.text_proj = nn.Linear(text_dim, embed_dim)
        else:
            # 默认给 512 维的投影
            self.text_proj = nn.Linear(512, embed_dim)
        
        # 全局特征相关模块
        if use_global_features:
            self.global_classifier = GlobalClassifier(embed_dim, [1, 3, 3, 6, 5])
            self.global_fc = GlobalFeatureFusion([1, 3, 3, 6, 5, embed_dim], embed_dim)
            self.label_attention = LabelAttention([embed_dim, embed_dim])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.image_proj = None
        self._image_feat_dim = None

    def encode_text(self, text_prompts: List[str], vlm_model=None, tokenizer=None) -> torch.Tensor:
        """编码文本提示 (适配 CONCH 或 原生 CLIP)"""
        
        # 优先使用传入的模型
        active_model = vlm_model if vlm_model is not None else self.clip_model
        
        if active_model is not None:
            if self.is_conch and CONCH_AVAILABLE:
                if tokenizer is None:
                     raise ValueError("A tokenizer must be provided for CONCH text encoding.")
                with torch.no_grad():
                    # 1. 使用 CONCH 的 tokenize
                    text_tokens = tokenize(texts=text_prompts, tokenizer=tokenizer).to(next(self.parameters()).device)
                    # 2. 提取文本特征
                    text_features = active_model.encode_text(text_tokens)
                    # 3. CONCH 必需的归一化 (用于空间对齐)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            elif CLIP_AVAILABLE:
                with torch.no_grad():
                    # 使用原生 CLIP
                    text_tokens = clip.tokenize(text_prompts).to(next(self.parameters()).device)
                    text_features = active_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                 raise RuntimeError("VLM Model provided but neither CONCH nor CLIP is available.")
        else:
            # Fallback：如果没有 VLM 模型，生成随机特征
            batch_size = len(text_prompts)
            text_features = torch.randn(batch_size, 512, device=next(self.parameters()).device)
        
        # 投影到 256 维 (SAM 的嵌入维度)
        text_embed = self.text_proj(text_features)
        return text_embed

    def forward(
        self,
        image_features: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        global_labels: Optional[torch.Tensor] = None,
        raw_image: Optional[torch.Tensor] = None,
        image_encoder: Optional[nn.Module] = None,
        vlm_model: Optional[nn.Module] = None,     # 新增：允许从外部传入加载好的 CONCH 模型
        vlm_tokenizer: Optional[Any] = None,       # 新增：允许从外部传入 CONCH 分词器
    ) -> torch.Tensor:
        """
        生成多模态提示嵌入
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        if raw_image is not None and image_encoder is not None:
            vit_features = self.clip_vit(raw_image, image_encoder=image_encoder)
        else:
            vit_features = image_features
        
        # 文本特征编码
        if text_prompts is not None:
            text_embed = self.encode_text(text_prompts, vlm_model=vlm_model, tokenizer=vlm_tokenizer)
        else:
            text_embed = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # 全局特征处理 (宏观语义分类)
        if self.use_global_features and global_labels is not None:
            global_logit = self.global_classifier(vit_features)
            global_features = self.global_fc(global_logit, global_labels)
            _, enhanced_features = self.label_attention(vit_features, global_features)
            image_pooled = F.adaptive_avg_pool2d(enhanced_features, 1).view(batch_size, -1)
        else:
            image_pooled = F.adaptive_avg_pool2d(vit_features, 1).view(batch_size, -1)
            feat_dim = image_pooled.shape[1]
            
            if feat_dim != self.embed_dim:
                if self.image_proj is None or self._image_feat_dim != feat_dim:
                    self.image_proj = nn.Linear(feat_dim, self.embed_dim).to(device)
                    self._image_feat_dim = feat_dim
                image_pooled = self.image_proj(image_pooled)
        
        # 融合文本提示和全局图像特征
        combined = torch.cat([text_embed, image_pooled], dim=1)
        prompt_embedding = self.feature_fusion(combined)
        
        return prompt_embedding