"""
PNuRL (Prompting Nuclei Representation Learning) 模块
功能：
1. 宏观监督：通过 5 个分类头强制 Image Encoder 学习物理属性。
2. 文本特征解耦 (🔥核心)：利用分类概率将 CONCH 文本特征调制并解耦为 T_attr (低频属性) 和 T_mor (高频形态)。
3. 特征矫正：利用融合后的文本特征对图像特征进行 Attention 加权。
4. 密度回归：生成宏观密度图，提供全局密度正则化指导 (Density Regularization)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class AttributeClassifier(nn.Module):
    """通用属性分类器"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class MultiScaleAttributeHead(nn.Module):
    """多尺度属性分类头 (用于 Shape, Size, Density)"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        # 浅层局部特征分支
        self.shallow_branch = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, in_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU()
        )
        # 深层全局特征分支
        self.deep_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(in_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feat_low = self.shallow_branch(x) 
        feat_high = self.deep_branch(x)
        logits = self.classifier(feat_high)
        logits = logits + feat_low.sum() * 0.0
        return logits, [feat_low, feat_high]

class AttributeClassifiers(nn.Module):
    """
    属性顺序: 0:Color, 1:Shape, 2:Arrange, 3:Size, 4:Density
    """
    def __init__(self, in_dim: int, num_classes_per_attr: List[int]):
        super().__init__()
        assert len(num_classes_per_attr) == 5, "Must provide class counts for 5 attributes"
        self.heads = nn.ModuleList()
        self.multiscale_indices = {1, 3, 4}  # Shape, Size, Density
        
        for i, num_classes in enumerate(num_classes_per_attr):
            if i in self.multiscale_indices:
                self.heads.append(MultiScaleAttributeHead(in_dim, num_classes))
            else:
                self.heads.append(AttributeClassifier(in_dim, num_classes))
    
    def forward(self, x: torch.Tensor, return_feats: bool = False) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        logits_list =[]
        visual_feats_low = []
        visual_feats_high =[]
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
        
        # 🔧 优化：如果不使用额外特征，则不再强行 Cat 占用显存
        if return_feats:
            fused_low = torch.cat(visual_feats_low, dim=1)
            fused_high = torch.cat(visual_feats_high, dim=1)
            return logits_list, [fused_low, fused_high]
        return logits_list, None

class AttributeAttention(nn.Module):
    """属性注意力机制 (编码器特征矫正)"""
    def __init__(self, feat_dim: int, embed_dim: int):
        super().__init__()
        self.attr_proj = nn.Linear(embed_dim, feat_dim)
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
    
    def forward(self, image_features: torch.Tensor, fused_text_embed: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_features.shape
        attr_proj = self.attr_proj(fused_text_embed) # [B, C]
        image_pooled = F.adaptive_avg_pool2d(image_features, 1).view(B, C) # [B, C]
        
        combined = torch.cat([image_pooled, attr_proj], dim=1) # [B, 2C]
        attention_weights = self.attention(combined).view(B, C, 1, 1) # [B, C, 1, 1]
        
        return image_features * (1 + attention_weights)

class PNuRL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        text_dim: int = 256, 
        num_classes_per_attr: List[int] = [2, 3, 2, 3, 3], 
        attr_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = embed_dim
        self.embed_dim = text_dim
        self.attr_loss_weight = attr_loss_weight
        
        self.attribute_classifiers = AttributeClassifiers(embed_dim, num_classes_per_attr)
        self.attribute_attention = AttributeAttention(embed_dim, text_dim)
        
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim + text_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 🔥 [核心改造]：将文本投影层拆分为两路 (解耦属性与形态)
        # 属性组 (Color=2, Arrange=2, Density=3) -> 共 7 类
        num_attr_classes = num_classes_per_attr[0] + num_classes_per_attr[2] + num_classes_per_attr[4]
        self.attr_prob_proj = nn.Sequential(
            nn.Linear(num_attr_classes, text_dim),
            nn.Sigmoid()  # 🔧 修复：加入门控，防止混合精度(AMP)下文本特征乘法溢出
        )
        
        # 形态组 (Shape=3, Size=3) -> 共 6 类
        num_mor_classes = num_classes_per_attr[1] + num_classes_per_attr[3]
        self.mor_prob_proj = nn.Sequential(
            nn.Linear(num_mor_classes, text_dim),
            nn.Sigmoid()  # 🔧 修复：同上
        )
        
        # 密度回归头 (用于宏观密度正则化，剔除了 OT 概念)
        self.density_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, 1, kernel_size=1),
            nn.ReLU()  # 密度图必定为正，ReLU非常合理
        ) 

    def forward(
        self,
        image_features: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None, 
        attribute_labels: Optional[List[torch.Tensor]] = None,
        return_loss: bool = True,
    ):
        B, C, H, W = image_features.shape
        device = image_features.device
        
        # 🔧 维度安全：确保 text_embed 为 [B, D]
        if text_embed is None:
            text_embed = torch.zeros(B, self.embed_dim, device=device)
        elif text_embed.dim() == 3:
            text_embed = text_embed.mean(dim=1)
            
        # === 1. 属性分类 ===
        attribute_logits, _ = self.attribute_classifiers(image_features, return_feats=False)
        probs_list = [F.softmax(l, dim=1) for l in attribute_logits]
        
        # === 2. 🔥 核心逻辑：文本特征的频域解耦调制 ===
        # 组 A：低频属性 (Color, Arrange, Density) -> 索引 0, 2, 4
        p_attr = torch.cat([probs_list[0], probs_list[2], probs_list[4]], dim=1)
        p_attr_proj = self.attr_prob_proj(p_attr)
        txt_attr_feat = text_embed * p_attr_proj  # [B, text_dim] -> 送给 ASR 的低频语义分支
        
        # 组 B：高频形态 (Shape, Size) -> 索引 1, 3
        p_mor = torch.cat([probs_list[1], probs_list[3]], dim=1)
        p_mor_proj = self.mor_prob_proj(p_mor)
        txt_mor_feat = text_embed * p_mor_proj    # [B, text_dim] -> 送给 ASR 的高频边缘分支
        
        # === 3. 特征矫正 (使用联合特征校准 Image Encoder) ===
        fused_text_embed = txt_attr_feat + txt_mor_feat
        refined_features = self.attribute_attention(image_features, fused_text_embed)
        
        # === 4. 生成上下文 ===
        image_pooled = F.adaptive_avg_pool2d(refined_features, 1).view(B, C)
        context_in = torch.cat([image_pooled, fused_text_embed], dim=1)
        learnable_context = self.context_fusion(context_in)
        
        # === 5. 密度图回归 ===
        density_map = self.density_decoder(refined_features)  
        
        # === 6. Loss ===
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
        
        # 返回值说明：
        # - txt_attr_feat: [B, D] 送给 MaskDecoder 的低频对齐头
        # - txt_mor_feat: [B, D] 送给 MaskDecoder 的高频对齐头
        return refined_features, learnable_context, loss, logits_dict, density_map, txt_attr_feat, txt_mor_feat

    def compute_attribute_loss(self, logits_list, labels_list):
        total_loss = 0.0
        # 给 Shape(1), Size(3), Density(4) 更高的权重
        weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        for i, (logits, label) in enumerate(zip(logits_list, labels_list)):
            if label.dim() > 1 and label.shape[1] == 1:
                label = label.squeeze(1)
            w = weights[i] if i < len(weights) else 1.0
            loss_i = F.cross_entropy(logits, label.long())
            total_loss += w * loss_i
        return total_loss / len(logits_list)