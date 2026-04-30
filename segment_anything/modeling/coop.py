"""
CoOp (Context Optimization) 模块 - 用于 CONCH 的提示学习
适配 MahmoodLab/CONCH 大模型的多模态提示
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False
    print("Warning: CONCH package not available. Please install it via git+https://github.com/Mahmoodlab/CONCH.git")


class PromptLearner(nn.Module):
    """可学习的提示学习器 (适配 CONCH)"""
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        tokenizer,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        n_cls = len(classnames)
        self.dtype = dtype
        
        # 兼容 OpenCLIP 架构，获取 token_embedding
        token_embedding_module = getattr(clip_model, 'token_embedding', None)
        if token_embedding_module is None and hasattr(clip_model, 'text'):
            token_embedding_module = clip_model.text.token_embedding
            
        ctx_dim = token_embedding_module.weight.shape[1]
        
        # 初始化上下文
        if ctx_init:
            # 使用预定义的临床上下文 (建议针对 CONCH 使用病理学术语)
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(texts=[ctx_init], tokenizer=tokenizer)
            with torch.no_grad():
                embedding = token_embedding_module(prompt).type(dtype)
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
        
        tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer)
        with torch.no_grad():
            embedding = token_embedding_module(tokenized_prompts).type(dtype)
        
        # 获取 SOS, CLS, EOS 等 token
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # 后续的 tokens
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

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


class TextEncoder(nn.Module):
    """文本编码器 (适配 OpenCLIP/CONCH 架构)"""
    def __init__(self, clip_model):
        super().__init__()
        # 兼容处理：获取 OpenCLIP 的内部组件
        if hasattr(clip_model, 'transformer'):
            self.transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
        elif hasattr(clip_model, 'text'):
            self.transformer = clip_model.text.transformer
            self.positional_embedding = clip_model.text.positional_embedding
            self.ln_final = clip_model.text.ln_final
            self.text_projection = clip_model.text.text_projection
            
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 取 EOS token 的特征并进行投影
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        if self.text_projection is not None:
            x = x @ self.text_projection
            
        return x


class CustomCLIP(nn.Module):
    """自定义 CONCH 模型，支持可学习的提示"""
    def __init__(
        self,
        classnames: List[str],
        clip_model,
        tokenizer,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(classnames, clip_model, tokenizer, n_ctx, ctx_init)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, image_features=None):
        if image_features is None:
            # 【核心修改】CONCH 必须开启 proj_contrast 和 normalize 来对齐特征空间
            image_features = self.clip_model.encode_image(
                image.type(self.dtype), 
                proj_contrast=True, 
                normalize=True
            )
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # 再次确保特征归一化 (对比学习的必要步骤)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


def load_conch_model(hf_auth_token: str, device="cuda"):
    """
    通过 HuggingFace Token 加载 CONCH 模型
    """
    if not CONCH_AVAILABLE:
        raise ImportError("CONCH package is not installed.")
    
    print("Loading CONCH model from Hugging Face...")
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16', 
        "hf_hub:MahmoodLab/conch", 
        hf_auth_token=hf_auth_token
    )
    model = model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer()
    
    return model, preprocess, tokenizer


class CustomCLIP_global(CustomCLIP): pass
class CustomCLIP_np(CustomCLIP): pass
class CustomCLIP_ns(CustomCLIP): pass
class CustomCLIP_nc(CustomCLIP): pass