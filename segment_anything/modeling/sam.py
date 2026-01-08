# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL
import sys
import os

try:
    import clip
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../CLIP")) 
    try:
        import clip
    except ImportError:
        print("⚠️ Warning: CLIP not found.")

try:
    from prompt_generator import TextGuidedPointGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    try:
        from prompt_generator import TextGuidedPointGenerator
    except ImportError:
        print("⚠️ Warning: prompt_generator.py not found.")

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        use_pnurl: bool = False,
        pnurl_config: Optional[Dict[str, Any]] = None,
        use_coop_prompt: bool = False,
        coop_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.use_pnurl = use_pnurl
        self.use_coop_prompt = use_coop_prompt
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        if self.use_pnurl and pnurl_config is not None:
            self.pnurl = PNuRL(
                feat_dim=prompt_encoder.embed_dim,
                embed_dim=prompt_encoder.embed_dim,
                clip_model_path=pnurl_config.get('clip_model_path'),
                num_classes_per_attr=pnurl_config.get('num_classes_per_attr', [3, 5, 4, 3, 3]),
                attr_loss_weight=pnurl_config.get('attr_loss_weight', 1.0)
            )
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(prompt_encoder.embed_dim, prompt_encoder.embed_dim)
        else:
            self.pnurl = None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        if masks.dim() == 3:
          masks = masks.unsqueeze(0)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [Fix] 这里的归一化是必要的，但必须确保输入是 0-255 的 Tensor
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


import torch
import torch.nn as nn
from torch.nn import functional as F
from .sam import Sam # 假设您原来的 Sam 类在这里
import clip

# === 1. 定义 CoOp 提示学习器 ===
class PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx=16, ctx_init=None):
        super().__init__()
        n_cls = 1
        n_ctx = n_ctx  # 上下文向量的数量 (例如 16 个单词长度)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        # 初始化可学习的上下文向量 (Context Vectors)
        if ctx_init:
            # 如果有初始化词 (比如 "microscopy pathology image")
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(ctx_vectors.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"🧠 CoOp Initialized: {n_ctx} learnable context tokens.")

        self.ctx = nn.Parameter(ctx_vectors) # [n_ctx, dim]
        
        # 保存 CLIP 的组件以供前向传播使用
        self.clip_token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.clip_ln_final = clip_model.ln_final
        self.clip_text_projection = clip_model.text_projection
        self.dtype = dtype
        self.n_ctx = n_ctx

    def forward(self, tokenized_prompts):
        # tokenized_prompts: [batch, 77]
        
        # 1. 获取输入文本的 Embedding (Specific Descriptions)
        # [batch, 77, dim]
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)

        # 2. 获取可学习的上下文 Embedding (General Context)
        # [n_ctx, dim] -> [batch, n_ctx, dim]
        ctx = self.ctx.unsqueeze(0).expand(len(tokenized_prompts), -1, -1)

        # 3. 拼接: [SOS] + [CTX] + [Specific Text] + [EOS]
        # CLIP 的 SOS 在 index 0
        prefix = embedding[:, :1, :] 
        # 截断原始文本的前半部分，给 CTX 腾位置
        # 注意：这里假设输入的 specific text 不会超级长，否则会被截断
        suffix = embedding[:, 1 : 77 - self.n_ctx, :] 

        x = torch.cat([prefix, ctx, suffix], dim=1) # [batch, 77, dim]

        # 4. 通过 CLIP Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 5. 提取特征
        x = self.clip_ln_final(x).type(self.dtype)

        # 6. 找到 EOS 位置并提取特征
        # 由于我们插入了 n_ctx 个 token，EOS 的位置向后移动了 n_ctx
        # 原始 tokenized_prompts.argmax(dim=-1) 是原始 EOS 位置
        original_eos_idx = tokenized_prompts.argmax(dim=-1)
        eos_idx = original_eos_idx + self.n_ctx
        # 限制最大索引防止越界
        eos_idx = torch.clamp(eos_idx, max=76)
        
        # 提取 [EOS] 处的特征作为句子特征
        text_features = x[torch.arange(x.shape[0]), eos_idx] @ self.clip_text_projection

        return text_features


# === 2. 修改后的 TextSam 类 ===
class TextSam(Sam):
    def __init__(
        self, 
        image_encoder, 
        prompt_encoder, 
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
        
        print(f"Loading CLIP model: {clip_model_name}...")
        # 加载 CLIP (CPU 加载，稍后转 GPU)
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        
        # 🔥 [关键] 冻结原始 CLIP 的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 🔥 [关键] 初始化 CoOp Prompt Learner
        # n_ctx=16 表示学习 16 个上下文单词，足以捕捉 "pathology microscopy" 等语义
        self.prompt_learner = PromptLearner(self.clip_model, n_ctx=16)
        
        # 🔥 [关键] 只解冻 Prompt Learner 的参数 (ctx)
        for param in self.prompt_learner.parameters():
            param.requires_grad = True

        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim
        )
        
        # 冻结其他部分
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        for param in self.mask_decoder.parameters(): param.requires_grad = True
        
        # 解冻 Adapter
        adapter_count = 0
        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True
                adapter_count += 1
                
        print(f"✅ TextSam Initialized: {adapter_count} Adapter Layers & CoOp Context Unfrozen.")

    def forward(self, batched_input, multimask_output=False):
        # 1. 图像编码
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) 

        device = image_embeddings.device
        
        # 确保 CLIP 组件在正确的设备上 (CoOp 的参数会自动随模型移动，但 CLIP 的 buffer 可能需要手动)
        if self.clip_model.visual.conv1.weight.device != device:
            self.clip_model = self.clip_model.to(device)
            # prompt_learner 是 nn.Module，通常不需要手动 to(device) 如果整个 TextSam 已经 to(device)

        # === 🔥 动态生成文本特征 (结合 CoOp) ===
        batch_text_features = []
        
        # 收集所有文本以进行批处理 (Batch Processing 效率更高)
        all_prompts = []
        for x in batched_input:
            # 这里的 text_prompt 现在是 "Microscopic image of large..." 这样的长句
            positive_prompt = x.get("text_prompt", "Cell nuclei")
            # 负样本: Background
            # 我们也让 CoOp 学习 Background 的上下文，保持域一致性
            all_prompts.extend([positive_prompt, "Background"])
            
        # 统一 Tokenize
        text_tokens = clip.tokenize(all_prompts, truncate=True).to(device)
        
        # 通过 Prompt Learner 编码 (而不是直接用 clip.encode_text)
        # 这里会注入可学习的 [CTX] 向量
        text_features_all = self.prompt_learner(text_tokens)
        
        # 归一化
        text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
        text_features_all = text_features_all.float()
        
        # 重新变回 [B, 2, 512]
        # all_prompts 是 [P1, Neg1, P2, Neg2, ...]
        batch_size = len(batched_input)
        text_features = text_features_all.view(batch_size, 2, -1)

        # 3. 热力图
        heatmap_logits = self.prompt_generator(image_embeddings, text_features)
        
        # 4. 提取点
        points_in_feat, point_labels = self.prompt_generator.get_points_from_heatmap(heatmap_logits, topk=1)
        
        # 5. 坐标映射
        feat_size = image_embeddings.shape[-1] 
        input_size = self.image_encoder.img_size 
        scale_factor = input_size / feat_size
        point_coords = (points_in_feat * scale_factor) + (scale_factor * 0.5)

        # 6. SAM 解码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        # 7. 结果封装
        outputs = []
        for i in range(len(batched_input)):
            mask_post = self.postprocess_masks(
                low_res_masks[i],
                input_size=batched_input[i]["image"].shape[-2:], 
                original_size=batched_input[i]["original_size"],
            )
            
            outputs.append({
                "masks": mask_post,
                "iou_predictions": iou_predictions[i],
                "low_res_masks": low_res_masks[i],
                "heatmap_logits": heatmap_logits[i].unsqueeze(0)
            })
            
        return outputs