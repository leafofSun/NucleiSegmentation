"""
PNuRL (Prompting Nuclei Representation Learning) 模块
功能：
1. 宏观监督：通过 5 个分类头强制 Image Encoder 学习物理属性。
2. 特征矫正：利用属性文本 (CLIP Embedding) 对图像特征进行 Attention 加权。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import os

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip package not available. PNuRL will use random embeddings.")


class AttributeClassifier(nn.Module):
    """单个属性分类器"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 2), # 稍微增加中间层维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AttributeClassifiers(nn.Module):
    """5个属性分类器：颜色、形状、排列、大小、分布"""
    def __init__(
        self,
        in_dim: int,
        num_classes_per_attr: List[int],
    ):
        super().__init__()
        self.classifiers = nn.ModuleList([
            AttributeClassifier(in_dim, num_classes) 
            for num_classes in num_classes_per_attr
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [classifier(x) for classifier in self.classifiers]


class AttributeAttention(nn.Module):
    """属性注意力机制: Use Attribute Embedding to refine Image Features"""
    def __init__(self, feat_dim: int, embed_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        
        # 将属性嵌入投影到特征维度
        self.attr_proj = nn.Linear(embed_dim, feat_dim)
        
        # Channel Attention
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        attribute_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        image_features: [B, C, H, W]
        attribute_embedding: [B, embed_dim]
        """
        B, C, H, W = image_features.shape
        
        # 投影属性嵌入 -> [B, C]
        attr_proj = self.attr_proj(attribute_embedding)
        
        # 全局池化图像特征 -> [B, C]
        image_pooled = F.adaptive_avg_pool2d(image_features, 1).view(B, C)
        
        # 拼接 -> [B, 2C]
        combined = torch.cat([image_pooled, attr_proj], dim=1)
        
        # 计算权重 -> [B, C]
        attention_weights = self.attention(combined)
        
        # 应用权重 (Channel-wise scaling)
        # Residual connection: F_new = F_old * (1 + Attention)
        attention_weights = attention_weights.view(B, C, 1, 1)
        weighted_features = image_features * (1 + attention_weights)
        
        return weighted_features


class PNuRL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,  # 🔥 [修正] 统一参数名为 embed_dim (对应 SAM feature dim)
        text_dim: int = 256,   # 投影后的文本维度
        clip_model_path: Optional[str] = "ViT-B/16",
        # 🔥 [修正] 默认类别数匹配 DataLoader: [Color(2), Shape(3), Arrange(2), Size(3), Density(3)]
        num_classes_per_attr: List[int] = [2, 3, 2, 3, 3], 
        attr_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = embed_dim
        self.embed_dim = text_dim # 这里复用变量名，实际是 projected text dim
        self.attr_loss_weight = attr_loss_weight
        
        # 1. 属性分类头
        self.attribute_classifiers = AttributeClassifiers(
            in_dim=embed_dim,
            num_classes_per_attr=num_classes_per_attr
        )
        
        # 2. CLIP 加载
        self.clip_model = None
        if CLIP_AVAILABLE:
            try:
                print(f"Loading CLIP for PNuRL: {clip_model_path}...")
                model, _ = clip.load(clip_model_path, device="cpu", jit=False)
                self.clip_model = model
                # 冻结 CLIP
                for param in self.clip_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: PNuRL CLIP load failed: {e}")
        
        # 3. 文本投影 (CLIP 512 -> SAM 256)
        clip_out_dim = 512 # ViT-B/16 default
        if self.clip_model is not None and hasattr(self.clip_model, 'text_projection'):
            clip_out_dim = self.clip_model.text_projection.shape[1]
            
        self.text_proj = nn.Linear(clip_out_dim, text_dim)
        
        # 4. 属性注意力
        self.attribute_attention = AttributeAttention(embed_dim, text_dim)
        
        # 5. 上下文融合 (生成额外的 context token 喂给 decoder)
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim + text_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def encode_attribute_text(self, attribute_prompts: List[str], device) -> torch.Tensor:
        """编码文本提示"""
        if self.clip_model is None:
            # Fallback: 随机向量
            return self.text_proj(torch.randn(len(attribute_prompts), 512, device=device))
            
        with torch.no_grad():
            # 确保 CLIP 在正确设备
            if next(self.clip_model.parameters()).device != device:
                self.clip_model.to(device)
            
            # Tokenize
            # 处理可能的空字符串或 list nesting
            clean_prompts = []
            for p in attribute_prompts:
                if isinstance(p, (list, tuple)): p = " ".join([str(x) for x in p])
                clean_prompts.append(str(p)[:77]) # 截断防止过长
            
            tokens = clip.tokenize(clean_prompts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return self.text_proj(text_features.float()) # [B, embed_dim]

    def forward(
        self,
        image_features: torch.Tensor,
        attribute_labels: Optional[List[torch.Tensor]] = None,
        attribute_prompts: Optional[List[str]] = None,
        return_loss: bool = True,
    ):
        """
        Returns:
            refined_features: [B, C, H, W]
            context_embedding: [B, C]
            loss: scalar
            logits_dict: dict
        """
        B, C, H, W = image_features.shape
        device = image_features.device
        
        # === 1. 属性分类 (Auxiliary Task) ===
        attribute_logits = self.attribute_classifiers(image_features) # List[[B, N_cls]]
        
        # 计算概率用于后续加权 (Soft Attribute Representation)
        # 拼接所有属性的概率分布
        probs_list = [F.softmax(l, dim=1) for l in attribute_logits]
        p_i = torch.cat(probs_list, dim=1) # [B, Total_Classes]
        
        # === 2. 文本编码 ===
        if attribute_prompts is not None:
            text_embed = self.encode_attribute_text(attribute_prompts, device) # [B, embed_dim]
        else:
            text_embed = torch.zeros(B, self.embed_dim, device=device)
            
        # === 3. 嵌入融合 (Text * Predicted_Probabilities) ===
        # 我们需要将 p_i 映射到与 text_embed 相同的维度才能相乘
        if not hasattr(self, 'prob_proj'):
            self.prob_proj = nn.Linear(p_i.shape[1], self.embed_dim).to(device)
        
        p_i_proj = self.prob_proj(p_i)
        
        # E = Text_Embedding * Predicted_Attributes
        # 只有当模型预测的属性与文本描述一致时，E 才会激活
        E = text_embed * p_i_proj # [B, embed_dim]
        
        # === 4. 特征矫正 (Refinement) ===
        refined_features = self.attribute_attention(image_features, E)
        
        # === 5. 生成上下文 Context ===
        image_pooled = F.adaptive_avg_pool2d(refined_features, 1).view(B, C)
        context_in = torch.cat([image_pooled, E], dim=1)
        learnable_context = self.context_fusion(context_in)
        
        # === 6. 计算 Loss ===
        loss = torch.tensor(0.0, device=device)
        if return_loss and attribute_labels is not None:
            loss = self.compute_attribute_loss(attribute_logits, attribute_labels)
            
        logits_dict = {
            'color': attribute_logits[0],
            'shape': attribute_logits[1],
            'arrange': attribute_logits[2],
            'size': attribute_logits[3],
            'density': attribute_logits[4],
        }
        
        return refined_features, learnable_context, loss, logits_dict

    def compute_attribute_loss(self, logits_list, labels_list):
        total_loss = 0.0
        # 权重: Color, Shape, Arrange, Size, Density
        # 给 Size 和 Density 更高的权重，因为它们对分割影响最大
        weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        for i, (logits, label) in enumerate(zip(logits_list, labels_list)):
            # label shape: [B] (indices)
            if label.dim() > 1: label = label.squeeze()
            
            # 安全检查
            if i < len(weights):
                w = weights[i]
            else:
                w = 1.0
                
            loss_i = F.cross_entropy(logits, label.long())
            total_loss += w * loss_i
            
        return total_loss / len(logits_list)