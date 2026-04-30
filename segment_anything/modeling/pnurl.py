"""
PNuRL (Prompting Nuclei Representation Learning) 模块
功能：
1. 宏观监督：通过 5 个分类头强制 Image Encoder 学习物理属性。
2. 特征矫正：利用属性文本 (CONCH Embedding) 对图像特征进行 Attention 加权。
3. 密度特征提取：专门为 Density 属性设计多尺度分支，生成密度图供后续 OT 模块使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class AttributeClassifier(nn.Module):
    """通用属性分类器 (用于 Color, Shape, Arrange, Size)"""
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
    """
    多尺度属性分类头
    适用于需要同时关注局部纹理 (Texture) 和全局语义 (Semantics) 的属性
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        # 浅层分支：提取纹理/边缘特征 -> [B, C/4, H, W]
        self.shallow_branch = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, in_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU()
        )
        
        # 深层分支：提取全局语义 -> [B, C/2]
        self.deep_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 分类头
        self.classifier = nn.Linear(in_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feat_low = self.shallow_branch(x) 
        feat_high = self.deep_branch(x)
        logits = self.classifier(feat_high)
        return logits, [feat_low, feat_high]


class AttributeClassifiers(nn.Module):
    """
    组合分类器管理器
    属性顺序: 0:Color, 1:Shape, 2:Arrange, 3:Size, 4:Density
    """
    def __init__(
        self,
        in_dim: int,
        num_classes_per_attr: List[int],
    ):
        super().__init__()
        assert len(num_classes_per_attr) == 5, "Must provide class counts for 5 attributes"
        
        self.heads = nn.ModuleList()
        self.multiscale_indices = {1, 3, 4}  # Shape, Size, Density
        
        for i, num_classes in enumerate(num_classes_per_attr):
            if i in self.multiscale_indices:
                self.heads.append(MultiScaleAttributeHead(in_dim, num_classes))
            else:
                self.heads.append(AttributeClassifier(in_dim, num_classes))
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits_list = []
        visual_feats_low = []
        visual_feats_high = []
        
        for i, head in enumerate(self.heads):
            if i in self.multiscale_indices:
                logits, feats = head(x)
                logits_list.append(logits)
                visual_feats_low.append(feats[0])
                visual_feats_high.append(feats[1])
            else:
                logits = head(x)
                logits_list.append(logits)
        
        if len(visual_feats_low) != 3 or len(visual_feats_high) != 3:
            raise ValueError(f"Expected 3 multiscale features")
        fused_low = torch.cat(visual_feats_low, dim=1)
        fused_high = torch.cat(visual_feats_high, dim=1)
        
        return logits_list, [fused_low, fused_high]


class AttributeAttention(nn.Module):
    """属性注意力机制"""
    def __init__(self, feat_dim: int, embed_dim: int):
        super().__init__()
        self.attr_proj = nn.Linear(embed_dim, feat_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
    
    def forward(self, image_features: torch.Tensor, attribute_embedding: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_features.shape
        
        attr_proj = self.attr_proj(attribute_embedding) # [B, C]
        image_pooled = F.adaptive_avg_pool2d(image_features, 1).view(B, C) # [B, C]
        
        combined = torch.cat([image_pooled, attr_proj], dim=1) # [B, 2C]
        attention_weights = self.attention(combined).view(B, C, 1, 1) # [B, C, 1, 1]
        
        return image_features * (1 + attention_weights)


class PNuRL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        text_dim: int = 256, # 外部传入的文本嵌入维度 (投影后的维度)
        num_classes_per_attr: List[int] = [2, 3, 2, 3, 3], 
        attr_loss_weight: float = 1.0,
    ):
        """
        初始化 PNuRL。
        注意：此处已移除了原生 CLIP 的加载逻辑。
        文本特征的提取（使用 CONCH）应当由外部模块完成并通过 forward 传入 `text_embed`。
        """
        super().__init__()
        self.feat_dim = embed_dim
        self.embed_dim = text_dim
        self.attr_loss_weight = attr_loss_weight
        
        # 1. 属性分类头
        self.attribute_classifiers = AttributeClassifiers(
            in_dim=embed_dim,
            num_classes_per_attr=num_classes_per_attr
        )
        
        # 2. 属性注意力
        self.attribute_attention = AttributeAttention(embed_dim, text_dim)
        
        # 3. 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim + text_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 4. 预定义概率投影层 (Prob Projection)
        total_classes = sum(num_classes_per_attr) 
        self.prob_proj = nn.Linear(total_classes, text_dim)
        
        # 5. 密度回归头：生成像素级密度图，供后续 OT 模块进行纯密度指导
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
            nn.ReLU()  # 密度必须 >= 0
        ) 

    def forward(
        self,
        image_features: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None, # [NEW] 从外部 (例如 multimodal_prompt) 传入已编码的 CONCH 文本特征
        attribute_labels: Optional[List[torch.Tensor]] = None,
        return_loss: bool = True,
    ):
        """
        前向传播。
        重要变更：不再接收 attribute_prompts 字符串列表，而是接收编码好的 text_embed 张量。
        这样彻底解耦了 PNuRL 与底层 VLM 模型。
        """
        B, C, H, W = image_features.shape
        device = image_features.device
        
        # === 1. 属性分类 & 特征提取 ===
        attribute_logits, fused_features = self.attribute_classifiers(image_features)
        
        probs_list = [F.softmax(l, dim=1) for l in attribute_logits]
        p_i = torch.cat(probs_list, dim=1) # [B, Total_Classes]
        
        # === 2. 处理文本嵌入 ===
        if text_embed is None:
            # 如果外部没有提供文本特征，使用全零占位
            text_embed = torch.zeros(B, self.embed_dim, device=device)
            
        # === 3. 嵌入融合 ===
        p_i_proj = self.prob_proj(p_i)
        E = text_embed * p_i_proj
        
        # === 4. 特征矫正 ===
        refined_features = self.attribute_attention(image_features, E)
        
        # === 5. 生成上下文 ===
        image_pooled = F.adaptive_avg_pool2d(refined_features, 1).view(B, C)
        context_in = torch.cat([image_pooled, E], dim=1)
        learnable_context = self.context_fusion(context_in)
        
        # === 6. 密度回归头 ===
        # 生成的密度图用于指导后续的最优传输 (OT)
        density_map = self.density_decoder(refined_features)  # [B, 1, H', W']
        
        # === 7. Loss ===
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
        # - refined_features: 提供给 SAM 进行后续解码
        # - learnable_context: 上下文特征
        # - loss: 属性分类损失
        # - logits_dict: 各项属性的分类结果
        # - fused_features: (已解耦) 建议不要将此特征直接传给 ASR，可丢弃或仅用于分析
        # - density_map: (关键输出) 提供给后续的纯密度 OT 模块
        return refined_features, learnable_context, loss, logits_dict, fused_features, density_map

    def compute_attribute_loss(self, logits_list, labels_list):
        total_loss = 0.0
        weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        for i, (logits, label) in enumerate(zip(logits_list, labels_list)):
            if label.dim() > 1 and label.shape[1] == 1:
                label = label.squeeze(1)
            
            w = weights[i] if i < len(weights) else 1.0
            loss_i = F.cross_entropy(logits, label.long())
            total_loss += w * loss_i
            
        return total_loss / len(logits_list)