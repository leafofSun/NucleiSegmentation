"""
PNuRL (Prompting Nuclei Representation Learning) 模块
功能：
1. 宏观监督：通过 5 个分类头强制 Image Encoder 学习物理属性。
2. 特征矫正：利用属性文本 (CLIP Embedding) 对图像特征进行 Attention 加权。
3. 密度特征提取：专门为 Density 属性设计多尺度分支，为后续模块提供纹理特征。
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
    例如: Shape (局部边缘), Size (全局面积), Density (局部密集度)
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
        """
        Returns:
            logits: [B, num_classes]
            features: [feat_low, feat_high]
                - feat_low: [B, C//4, H, W] (保留空间信息)
                - feat_high: [B, C//2] (全局向量)
        """
        feat_low = self.shallow_branch(x) 
        feat_high = self.deep_branch(x)
        logits = self.classifier(feat_high)
        return logits, [feat_low, feat_high]


class AttributeClassifiers(nn.Module):
    """
    组合分类器管理器 (升级版)
    属性顺序: 0:Color, 1:Shape, 2:Arrange, 3:Size, 4:Density
    策略: 对 Shape(1), Size(3), Density(4) 使用多尺度头
    """
    def __init__(
        self,
        in_dim: int,
        num_classes_per_attr: List[int],
    ):
        super().__init__()
        # 确保属性数量正确 (默认为5个)
        assert len(num_classes_per_attr) == 5, "Must provide class counts for 5 attributes"
        
        self.heads = nn.ModuleList()
        # 定义哪些属性需要多尺度特征 (Indices)
        self.multiscale_indices = {1, 3, 4}  # Shape, Size, Density
        
        for i, num_classes in enumerate(num_classes_per_attr):
            if i in self.multiscale_indices:
                self.heads.append(MultiScaleAttributeHead(in_dim, num_classes))
            else:
                self.heads.append(AttributeClassifier(in_dim, num_classes))
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            logits_list: List of 5 logits tensors
            fused_features: [fused_low, fused_high] - 拼接后的多尺度特征
        """
        logits_list = []
        visual_feats_low = []
        visual_feats_high = []
        
        for i, head in enumerate(self.heads):
            if i in self.multiscale_indices:
                # 多尺度头返回: logits, [low, high]
                logits, feats = head(x)
                logits_list.append(logits)
                visual_feats_low.append(feats[0])
                visual_feats_high.append(feats[1])
            else:
                # 普通头返回: logits
                logits = head(x)
                logits_list.append(logits)
        
        # 🔥 [核心融合] 将 Shape, Size, Density 的特征拼接
        # low: 3 * [B, C/4, H, W] -> [B, 3*C/4, H, W]
        # high: 3 * [B, C/2]      -> [B, 3*C/2]
        if len(visual_feats_low) != 3 or len(visual_feats_high) != 3:
            raise ValueError(f"Expected 3 multiscale features, got {len(visual_feats_low)} low and {len(visual_feats_high)} high")
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
        
        # Channel-wise scaling (Residual)
        return image_features * (1 + attention_weights)


class PNuRL(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        text_dim: int = 256,
        feat_dim: int = 256,
        clip_model_path: Optional[str] = "ViT-B/16",
        num_classes_per_attr: List[int] = [2, 3, 2, 3, 3], 
        attr_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = embed_dim
        self.embed_dim = text_dim
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
                for param in self.clip_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: PNuRL CLIP load failed: {e}")
        
        # 3. 文本投影
        clip_out_dim = 512
        if self.clip_model is not None and hasattr(self.clip_model, 'text_projection'):
            clip_out_dim = self.clip_model.text_projection.shape[1]
        self.text_proj = nn.Linear(clip_out_dim, text_dim)
        
        # 4. 属性注意力
        self.attribute_attention = AttributeAttention(embed_dim, text_dim)
        
        # 5. 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim + text_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 🔥 [修正] 6. 预定义概率投影层 (Prob Projection)
        # 必须在 __init__ 中定义，否则优化器无法更新参数
        total_classes = sum(num_classes_per_attr) # e.g. 2+3+2+3+3 = 13
        self.prob_proj = nn.Linear(total_classes, text_dim)
        
        # 🔥 [新增] 7. 密度回归头 (Multi-Task: 保留分类 + 新增回归)
        # 用于生成像素级密度图，参考 DeNSe 论文的强对齐策略
        # 输入: image_features [B, embed_dim, H, W]
        # 输出: density_map [B, 1, target_H, target_W] (与 mask 大小匹配)
        self.density_decoder = nn.Sequential(
            # 第一层上采样: H/4 -> H/2 (假设输入是 H/4 大小)
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            # 第二层上采样: H/2 -> H
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            # 第三层上采样: H -> 2H (如果需要更大尺寸)
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(),
            # 最终输出层: 单通道密度图
            nn.Conv2d(embed_dim // 8, 1, kernel_size=1),
            nn.ReLU()  # 密度必须 >= 0
        ) 

    def encode_attribute_text(self, attribute_prompts: List[str], device) -> torch.Tensor:
        if self.clip_model is None:
            return self.text_proj(torch.randn(len(attribute_prompts), 512, device=device))
            
        with torch.no_grad():
            if next(self.clip_model.parameters()).device != device:
                self.clip_model.to(device)
            
            clean_prompts = []
            for p in attribute_prompts:
                if isinstance(p, (list, tuple)): p = " ".join([str(x) for x in p])
                clean_prompts.append(str(p)[:77])
            
            tokens = clip.tokenize(clean_prompts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return self.text_proj(text_features.float())

    def forward(
        self,
        image_features: torch.Tensor,
        attribute_labels: Optional[List[torch.Tensor]] = None,
        attribute_prompts: Optional[List[str]] = None,
        return_loss: bool = True,
    ):
        B, C, H, W = image_features.shape
        device = image_features.device
        
        # === 1. 属性分类 & 特征提取 ===
        # logits_list 包含 5 个属性的 logit
        # fused_features 包含 [Concat_Low, Concat_High]
        attribute_logits, fused_features = self.attribute_classifiers(image_features)
        
        # Soft Attribute Representation
        probs_list = [F.softmax(l, dim=1) for l in attribute_logits]
        p_i = torch.cat(probs_list, dim=1) # [B, Total_Classes]
        
        # === 2. 文本编码 ===
        if attribute_prompts is not None:
            text_embed = self.encode_attribute_text(attribute_prompts, device)
        else:
            text_embed = torch.zeros(B, self.embed_dim, device=device)
            
        # === 3. 嵌入融合 ===
        # 🔥 [修正] 使用在 init 中定义的层
        p_i_proj = self.prob_proj(p_i)
        E = text_embed * p_i_proj
        
        # === 4. 特征矫正 ===
        refined_features = self.attribute_attention(image_features, E)
        
        # === 5. 生成上下文 ===
        image_pooled = F.adaptive_avg_pool2d(refined_features, 1).view(B, C)
        context_in = torch.cat([image_pooled, E], dim=1)
        learnable_context = self.context_fusion(context_in)
        
        # === 6. 🔥 [新增] 密度回归头：生成像素级密度图 ===
        # 使用 refined_features 作为输入（已经经过属性注意力调制）
        density_map = self.density_decoder(refined_features)  # [B, 1, H', W']
        # 注意：密度图的大小取决于 decoder 的上采样层数
        # 如果需要特定大小（如与 mask 匹配），可以在外部进行插值
        
        # === 7. Loss ===
        loss = torch.tensor(0.0, device=device)
        if return_loss and attribute_labels is not None:
            loss = self.compute_attribute_loss(attribute_logits, attribute_labels)
            
        logits_dict = {
            'color': attribute_logits[0],
            'shape': attribute_logits[1],
            'arrange': attribute_logits[2],
            'size': attribute_logits[3],
            'density': attribute_logits[4],  # 保留分类 logits（用于 Adapter 指导）
        }
        
        # 🔥 [升级] 返回：保留原有输出 + 新增密度图
        # - refined_features: 用于 SAM 的特征
        # - learnable_context: 用于 Prompt 的上下文
        # - loss: 分类损失
        # - logits_dict: 包含 density 分类 logits（用于 PhysicalAdapter）
        # - fused_features: 多尺度特征（用于 PhysicalAdapter）
        # - density_map: 新增的像素级密度图（用于 DeNSe 式强对齐）
        return refined_features, learnable_context, loss, logits_dict, fused_features, density_map

    def compute_attribute_loss(self, logits_list, labels_list):
        total_loss = 0.0
        # 权重: Color, Shape, Arrange, Size, Density
        weights = [1.0, 1.0, 1.0, 2.0, 2.0]
        
        for i, (logits, label) in enumerate(zip(logits_list, labels_list)):
            # 🔥 [修正] 安全的 squeeze
            # 只有当 label 是 [B, 1] 时才 squeeze，如果是 [B] 则不动
            if label.dim() > 1 and label.shape[1] == 1:
                label = label.squeeze(1)
            
            w = weights[i] if i < len(weights) else 1.0
            loss_i = F.cross_entropy(logits, label.long())
            total_loss += w * loss_i
            
        return total_loss / len(logits_list)