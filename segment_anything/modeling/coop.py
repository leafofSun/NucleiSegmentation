"""
CoOp (Context Optimization) 模块 - 用于CLIP的提示学习
简化版本，用于SAM-Med2D的多模态提示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: clip package not available")


class PromptLearner(nn.Module):
    """可学习的提示学习器"""
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        n_cls = len(classnames)
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # 初始化上下文
        if ctx_init:
            # 使用预定义的上下文
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化上下文
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 构建提示模板
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # 这些token不会被优化
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat([
            prefix,  # (n_cls, 1, dim)
            ctx,     # (n_cls, n_ctx, dim)
            suffix,  # (n_cls, *, dim)
        ], dim=1)
        
        return prompts


class CustomCLIP(nn.Module):
    """自定义CLIP模型，支持可学习的提示"""
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx, ctx_init)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, image_features=None):
        if image_features is None:
            image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 取EOS token的特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x


def load_clip_to_cpu(pretrained_path: Optional[str] = None):
    """加载CLIP模型到CPU"""
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP package not available")
    
    model, _ = clip.load("RN50", device="cpu", jit=False)
    if pretrained_path:
        try:
            checkpoint = torch.jit.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.state_dict()
            # 处理权重加载
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    return model


# 为了兼容用户代码中的不同CoOp变体
class CustomCLIP_global(CustomCLIP):
    """用于全局特征的CLIP"""
    pass


class CustomCLIP_np(CustomCLIP):
    """用于nuclei pixel的CLIP"""
    pass


class CustomCLIP_ns(CustomCLIP):
    """用于nuclei semantic的CLIP"""
    pass


class CustomCLIP_nc(CustomCLIP):
    """用于nuclei classification的CLIP"""
    pass

