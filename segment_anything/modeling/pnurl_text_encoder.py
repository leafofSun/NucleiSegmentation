"""
PNuRL Text Encoder - 封装 CoOp 的可学习提示生成器
将 CoOp/PNuRL 的可学习语义信息转换为文本嵌入，作为 SAM 的 sparse prompts
"""

import torch
import torch.nn as nn
from typing import List, Optional
import os

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip package not available. Install with: pip install git+https://github.com/openai/CLIP.git")

from .coop import PromptLearner, TextEncoder


class PNuRLTextEncoder(nn.Module):
    """
    PNuRL 文本编码器
    使用 CoOp 的 PromptLearner 和 TextEncoder 生成可学习的文本嵌入
    这些嵌入可以直接作为 SAM 的 sparse prompts 使用
    """
    def __init__(
        self,
        classnames: List[str],
        clip_model_name: str = "ViT-B/16",
        clip_model_path: Optional[str] = None,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        text_embed_dim: Optional[int] = None,
    ):
        """
        Args:
            classnames: 类别名称列表，例如 ["Nuclei", "Cell", "Tissue"]
            clip_model_name: CLIP 模型名称，例如 "RN50", "ViT-B/16"
            clip_model_path: CLIP 预训练模型路径（可选）
            n_ctx: 可学习上下文的 token 数量
            ctx_init: 初始上下文字符串（可选），例如 "a photo of a"
            text_embed_dim: 文本嵌入维度（如果为 None，将从 CLIP 模型自动获取）
        """
        super().__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP package not available. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        # 加载 CLIP 模型
        self.clip_model, _ = clip.load(clip_model_name, device="cpu", jit=False)
        
        # 如果提供了预训练路径，尝试加载权重
        if clip_model_path and os.path.isfile(clip_model_path):
            try:
                checkpoint = torch.jit.load(clip_model_path, map_location='cpu')
                state_dict = checkpoint.state_dict()
                # 只加载文本编码器部分
                text_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('transformer.') or k.startswith('token_embedding') or k.startswith('text_projection'):
                        text_state_dict[k] = v
                if text_state_dict:
                    self.clip_model.load_state_dict(text_state_dict, strict=False)
                    print(f"✓ 成功加载 CLIP 文本编码器权重: {clip_model_path}")
            except Exception as e:
                print(f"Warning: 无法从 {clip_model_path} 加载 CLIP 文本编码器权重: {e}")
                print("  将使用默认的 CLIP 预训练模型")
        
        # 冻结 CLIP 参数（只训练可学习的上下文）
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_dtype = self.clip_model.dtype
        
        # 初始化 PromptLearner (CoOp 的核心)
        self.prompt_learner = PromptLearner(
            classnames=classnames,
            clip_model=self.clip_model,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            dtype=self.clip_dtype,
        )
        
        # 使用 TextEncoder
        self.text_encoder = TextEncoder(self.clip_model)
        
        # 获取文本嵌入维度
        if text_embed_dim is None:
            # 从 CLIP 模型的 text_projection 获取维度
            if hasattr(self.clip_model, 'text_projection'):
                self.text_embed_dim = self.clip_model.text_projection.shape[1]
            else:
                self.text_embed_dim = 512  # 默认值
        else:
            self.text_embed_dim = text_embed_dim
        
        self.num_classes = len(classnames)
        self.classnames = classnames
    
    def forward(self, target_class_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成可学习的文本嵌入
        
        Args:
            target_class_idx: 目标类别索引 [B]，如果为 None，则返回所有类别的嵌入 [num_classes, text_embed_dim]
        
        Returns:
            text_features: 
                - 如果 target_class_idx 为 None: [num_classes, text_embed_dim]
                - 如果 target_class_idx 不为 None: [B, text_embed_dim]
        """
        # 1. 生成带有 Learnable Context 的 Prompts
        prompts = self.prompt_learner()  # shape: [n_cls, n_ctx+2, dim] (包含 prefix 和 suffix)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 2. 通过 CLIP Text Encoder 提取特征
        text_features = self.text_encoder(prompts, tokenized_prompts)  # [n_cls, text_embed_dim]
        
        # 3. 归一化（可选，根据 SAM 训练需求）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 4. 如果指定了目标类别，选择对应的嵌入
        if target_class_idx is not None:
            # target_class_idx: [B]
            # text_features: [num_classes, text_embed_dim]
            # 返回: [B, text_embed_dim]
            batch_size = target_class_idx.shape[0]
            device = target_class_idx.device
            text_features = text_features.to(device)
            
            # 选择对应的类别嵌入
            selected_features = text_features[target_class_idx]  # [B, text_embed_dim]
            return selected_features
        else:
            # 返回所有类别的嵌入
            return text_features  # [num_classes, text_embed_dim]
    
    def get_text_embed_dim(self) -> int:
        """返回文本嵌入维度"""
        return self.text_embed_dim

