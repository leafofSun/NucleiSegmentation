"""
PNuRL (Prompting Nuclei Representation Learning) 模块
使用 SAM 的 ViT 编码器替代 ResNet-50，实现属性感知的表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
import os

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip package not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class AttributeClassifier(nn.Module):
    """单个属性分类器"""
    def __init__(self, in_dim: int, num_classes: int):
        """
        Args:
            in_dim: 输入特征维度
            num_classes: 该属性的类别数量
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 4, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像特征 [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        return self.classifier(x)


class AttributeClassifiers(nn.Module):
    """5个属性分类器：颜色、形状、排列、大小、分布"""
    def __init__(
        self,
        in_dim: int,
        num_classes_per_attr: List[int] = [3, 5, 4, 3, 3],  # 默认类别数
    ):
        """
        Args:
            in_dim: 输入特征维度（ViT 的 out_chans，通常是 256）
            num_classes_per_attr: 每个属性的类别数量
                [颜色, 形状, 排列, 大小, 分布]
        """
        super().__init__()
        self.num_attributes = 5
        self.classifiers = nn.ModuleList([
            AttributeClassifier(in_dim, num_classes) 
            for num_classes in num_classes_per_attr
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 图像特征 [B, C, H, W]
        Returns:
            logits_list: 5个属性的 logits 列表，每个 [B, num_classes_j]
        """
        return [classifier(x) for classifier in self.classifiers]


class AttributeAttention(nn.Module):
    """属性注意力机制，用于加权图像特征"""
    def __init__(self, feat_dim: int, embed_dim: int):
        """
        Args:
            feat_dim: 图像特征维度
            embed_dim: 属性嵌入维度
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        
        # 将属性嵌入投影到特征维度
        self.attr_proj = nn.Linear(embed_dim, feat_dim)
        
        # 注意力计算
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
        Args:
            image_features: [B, C, H, W] 图像特征
            attribute_embedding: [B, embed_dim] 属性嵌入
        Returns:
            weighted_features: [B, C, H, W] 加权后的图像特征
        """
        B, C, H, W = image_features.shape
        
        # 投影属性嵌入
        attr_proj = self.attr_proj(attribute_embedding)  # [B, C]
        
        # 池化图像特征
        image_pooled = F.adaptive_avg_pool2d(image_features, 1).view(B, C)  # [B, C]
        
        # 拼接并计算注意力权重
        combined = torch.cat([image_pooled, attr_proj], dim=1)  # [B, 2C]
        attention_weights = self.attention(combined)  # [B, C]
        
        # 应用注意力权重
        attention_weights = attention_weights.view(B, C, 1, 1)
        weighted_features = image_features * attention_weights
        
        return weighted_features


class PNuRL(nn.Module):
    """
    Prompting Nuclei Representation Learning 模块
    架构融合：SAM ViT Image Encoder + CLIP ViT Text Encoder
    """
    def __init__(
        self,
        feat_dim: int = 256,  # SAM ViT 的 out_chans
        embed_dim: int = 256,  # 嵌入维度
        clip_model_path: Optional[str] = "ViT-B/16",  # 默认改为 ViT-B/16
        num_classes_per_attr: List[int] = [3, 5, 4, 3, 3],
        attr_loss_weight: float = 1.0,
    ):
        """
        Args:
            feat_dim: 图像特征维度（SAM ViT 输出维度）
            embed_dim: 文本嵌入维度
            clip_model_path: CLIP 模型名称 (如 "ViT-B/16") 或 本地权重路径
            num_classes_per_attr: 每个属性的类别数量 [颜色, 形状, 排列, 大小, 分布]
            attr_loss_weight: 属性损失权重
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.attr_loss_weight = attr_loss_weight
        
        # 5个属性分类器
        self.attribute_classifiers = AttributeClassifiers(
            in_dim=feat_dim,
            num_classes_per_attr=num_classes_per_attr
        )
        
        # === 修改部分开始：CLIP 文本编码器加载逻辑 ===
        self.clip_model = None
        if CLIP_AVAILABLE:
            try:
                model_name = clip_model_path if clip_model_path else "ViT-B/16"
                print(f"Loading CLIP Text Encoder: {model_name}...")
                
                model, _ = clip.load(model_name, device="cpu", jit=False)
                self.clip_model = model
                
                # 保留 visual 以维持 dtype 依赖，但放在 CPU 并冻结
                if hasattr(self.clip_model, 'visual'):
                    self.clip_model.visual = self.clip_model.visual.cpu()
                    for param in self.clip_model.visual.parameters():
                        param.requires_grad = False
                    print("  - CLIP Visual Encoder 已冻结并保留在 CPU (维持 dtype 依赖)")
                
                # 冻结全部参数
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                
                print(f"✓ 成功加载 CLIP: {model_name}")

            except Exception as e:
                print(f"Warning: 加载 CLIP 模型失败: {e}")
                self.clip_model = None
        else:
            print("Warning: CLIP 包未安装。")
        # === 修改部分结束 ===
        
        # 文本特征投影层 (CLIP输出通常是512或768，需要投影到 embed_dim=256)
        if self.clip_model is not None:
            text_dim = self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else 512
            self.text_proj = nn.Linear(text_dim, embed_dim)
        else:
            # 如果没有 CLIP，使用随机层兜底
            self.text_proj = nn.Linear(512, embed_dim)
        
        # 属性注意力机制
        self.attribute_attention = AttributeAttention(feat_dim, embed_dim)
        
        # 特征融合层（用于生成可学习上下文）
        self.context_fusion = nn.Sequential(
            nn.Linear(feat_dim + embed_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
    
    def encode_attribute_text(self, attribute_prompts) -> torch.Tensor:
        """
        编码属性感知文本提示，支持 DataLoader collate 后的批量格式。
        """
        if self.clip_model is not None and CLIP_AVAILABLE:
            with torch.no_grad():
                device = next(self.parameters()).device
                
                combined_prompts = []
                if isinstance(attribute_prompts, list) and len(attribute_prompts) > 0 and isinstance(attribute_prompts[0], tuple):
                    batch_size = len(attribute_prompts[0])
                    for i in range(batch_size):
                        sample_attrs = [str(row[i]) for row in attribute_prompts]
                        combined_prompts.append(", ".join(sample_attrs))
                else:
                    combined_prompts = [", ".join(attribute_prompts)]
                
                text_tokens = clip.tokenize(combined_prompts, truncate=True).to(device)
                
                # 确保编码时权重在正确设备；encode_text 依赖 dtype (由 visual 决定)
                if self.clip_model.token_embedding.weight.device != device:
                    self.clip_model.to(device)
                    if hasattr(self.clip_model, 'visual'):
                        self.clip_model.visual.cpu()
                
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.float()
        else:
            if isinstance(attribute_prompts, list) and len(attribute_prompts) > 0 and isinstance(attribute_prompts[0], tuple):
                batch_size = len(attribute_prompts[0])
            else:
                batch_size = 1
            text_features = torch.randn(batch_size, 512, device=next(self.parameters()).device)
        
        text_embed = self.text_proj(text_features)
        return text_embed
    
    def forward(
        self,
        image_features: torch.Tensor,
        attribute_prompts: Optional[List[str]] = None,
        attribute_labels: Optional[List[torch.Tensor]] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        前向传播
        
        Args:
            image_features: [B, C, H, W] SAM ViT 编码器的输出特征
            attribute_prompts: 属性感知文本提示列表
            attribute_labels: 5个属性的真实标签列表，每个 [B, num_classes_j]
            return_loss: 是否返回损失
        
        Returns:
            weighted_features: [B, C, H, W] 加权后的ViT特征（用于分割）
            learnable_context: [B, feat_dim] 可学习上下文
            loss: 属性损失（如果 return_loss=True）
            logits_dict: 属性 logits 字典（用于调试）
        """
        B, C, H, W = image_features.shape
        
        # 1. 通过5个属性分类器获取 logits
        attribute_logits = self.attribute_classifiers(image_features)  # List of [B, num_classes_j]
        
        # 2. 计算属性概率
        attribute_probs = []
        for logits in attribute_logits:
            probs = F.softmax(logits, dim=1)  # [B, num_classes_j]
            attribute_probs.append(probs)
        
        # 3. 拼接所有属性的概率
        p_i = torch.cat(attribute_probs, dim=1)  # [B, total_classes]
        
        # 4. 编码属性感知文本提示
        if attribute_prompts is not None:
            T_i_k_a = self.encode_attribute_text(attribute_prompts)  # [B, embed_dim]
        else:
            # 如果没有提供提示，使用零向量
            T_i_k_a = torch.zeros(B, self.embed_dim, device=image_features.device)
        
        # 5. 嵌入融合：E = p_i ◦ g(T_i,k^a)
        # 注意：这里需要调整维度以进行逐元素相乘
        # p_i: [B, total_classes], T_i_k_a: [B, embed_dim]
        # 我们需要将 p_i 投影到 embed_dim 维度
        if p_i.shape[1] != self.embed_dim:
            # 动态创建投影层
            if not hasattr(self, 'prob_proj') or self.prob_proj is None:
                self.prob_proj = nn.Linear(p_i.shape[1], self.embed_dim).to(image_features.device)
            p_i_proj = self.prob_proj(p_i)  # [B, embed_dim]
        else:
            p_i_proj = p_i
        
        # 逐元素相乘
        E = p_i_proj * T_i_k_a  # [B, embed_dim]
        
        # 6. 注意力机制加权图像特征（关键步骤：使用属性信息对ViT特征进行加权）
        weighted_features = self.attribute_attention(image_features, E)  # [B, C, H, W]
        
        # 7. 池化加权特征
        weighted_pooled = F.adaptive_avg_pool2d(weighted_features, 1).view(B, C)  # [B, C]
        
        # 8. 生成可学习上下文：Concat([V1, V2, ..., Vm, Attribute])
        # 这里 V1, V2, ..., Vm 对应池化后的图像特征，Attribute 对应 E
        learnable_context = torch.cat([weighted_pooled, E], dim=1)  # [B, C + embed_dim]
        learnable_context = self.context_fusion(learnable_context)  # [B, feat_dim]
        
        # 计算损失（如果需要）
        loss = None
        if return_loss and attribute_labels is not None:
            loss = self.compute_attribute_loss(attribute_logits, attribute_labels)
        
        # 构建 logits 字典（用于调试）
        logits_dict = {
            'color': attribute_logits[0],
            'shape': attribute_logits[1],
            'arrangement': attribute_logits[2],
            'size': attribute_logits[3],
            'distribution': attribute_logits[4],
        }
        
        # 返回加权后的特征（用于分割）和可学习上下文
        return weighted_features, learnable_context, loss, logits_dict
    
    def compute_attribute_loss(
        self,
        attribute_logits: List[torch.Tensor],
        attribute_labels: List[torch.Tensor],
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        计算属性损失（二元交叉熵）
        
        Args:
            attribute_logits: 5个属性的 logits 列表
            attribute_labels: 5个属性的真实标签列表
            alpha: 损失权重
        
        Returns:
            loss: 总属性损失
        """
        total_loss = 0.0
        num_attributes = len(attribute_logits)
        
        for j in range(num_attributes):
            logits = attribute_logits[j]  # [B, num_classes_j]
            labels = attribute_labels[j]  # [B] (类别索引) 或 [B, num_classes_j] (one-hot)
            
            # 确保标签在正确的设备上
            labels = labels.to(logits.device)
            
            # 处理标签格式
            if labels.dim() == 1:
                # 类别索引格式 [B]，直接使用 cross_entropy
                loss_j = F.cross_entropy(logits, labels.long())
            elif labels.dim() == 2 and labels.sum(dim=1).max() > 1:
                # one-hot 编码，取最大值的索引
                labels = labels.argmax(dim=1)  # [B]
                loss_j = F.cross_entropy(logits, labels.long())
            else:
                # binary 编码，使用二元交叉熵
                labels = labels.float()
                probs = torch.sigmoid(logits)
                loss_j = F.binary_cross_entropy(probs, labels, reduction='mean')
            
            total_loss += alpha * loss_j
        
        return total_loss / num_attributes

