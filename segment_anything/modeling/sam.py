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

# === 依赖检查 ===
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

# === 基础 SAM 类 ===
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
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(self, batched_input: List[Dict[str, Any]], multimask_output: bool):
        # 基础 SAM forward 逻辑保持不变，主要逻辑在 TextSam 中重写
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
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })
        return outputs

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> torch.Tensor:
        if masks.dim() == 3: masks = masks.unsqueeze(0)
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
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

# === 🔥 [关键修改] 1. 定义 Dual-Prompt Learner (双层提示库) ===
# 灵感来源: CA-SAM2 (Context-Aware)
class DualPromptLearner(nn.Module):
    def __init__(self, clip_model, num_organs=14, n_ctx_gen=8, n_ctx_spec=8):
        super().__init__()
        # 获取 CLIP 属性
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        dtype = clip_model.dtype
        self.dtype = dtype

        # --- A. 通用特征库 (General Bank) ---
        # 所有细胞核共享的知识 (Implicit Knowledge)
        print(f"🧠 Init DualLearner: General Ctx ({n_ctx_gen}) + Specific Ctx ({n_ctx_spec} x {num_organs} organs)")
        self.ctx_general = nn.Parameter(torch.empty(n_ctx_gen, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_general, std=0.02)
        
        # --- B. 特定特征库 (Specific Bank) ---
        # 针对不同器官/组织的特定知识库
        self.ctx_specific = nn.Parameter(torch.empty(num_organs, n_ctx_spec, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_specific, std=0.02)
        
        # 保存 CLIP 组件
        self.clip_token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.clip_ln_final = clip_model.ln_final
        self.clip_text_projection = clip_model.text_projection
        
        self.n_ctx_gen = n_ctx_gen
        self.n_ctx_spec = n_ctx_spec
        self.total_ctx = n_ctx_gen + n_ctx_spec

    def forward(self, organ_indices, tokenized_prompts):
        """
        Args:
            organ_indices: [Batch] 当前 batch 对应的器官 ID
            tokenized_prompts: [Batch, 77] 输入的基础文本 (e.g. "Cell nuclei")
        """
        batch_size = len(organ_indices)
        
        # 1. 准备文本 Embedding (e.g., "Cell nuclei")
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)
        
        # 2. 准备通用 Context (扩展到 Batch)
        # [Batch, n_gen, dim]
        ctx_gen = self.ctx_general.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. 准备特定 Context (查表)
        # [Batch, n_spec, dim]
        ctx_spec = self.ctx_specific[organ_indices]
        
        # 4. 融合 Context: [通用] + [特定]
        ctx = torch.cat([ctx_gen, ctx_spec], dim=1) # [Batch, total_ctx, dim]

        # 5. 拼接最终序列: [SOS] + [Dual_CTX] + [Text] + [EOS]
        prefix = embedding[:, :1, :] 
        suffix = embedding[:, 1 : 77 - self.total_ctx, :] 

        x = torch.cat([prefix, ctx, suffix], dim=1)

        # 6. CLIP 编码流程
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_ln_final(x).type(self.dtype)

        # 7. 提取特征 (EOS位置)
        original_eos_idx = tokenized_prompts.argmax(dim=-1)
        eos_idx = torch.clamp(original_eos_idx + self.total_ctx, max=76)
        text_features = x[torch.arange(x.shape[0]), eos_idx] @ self.clip_text_projection

        return text_features


# === 🔥 [关键修改] 2. MP-SAM (TextSam) 核心类 ===
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
        embed_dim=256,
        num_organs=14 # MoNuSeg 默认 14 类，可视情况调整
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
        
        print(f"🚀 Initializing MP-SAM (Multi-granularity Prompt SAM)...")
        
        # 1. 加载 CLIP
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False # 冻结原始 CLIP
            
        # 2. 初始化 Dual-Prompt Learner (CA-SAM2)
        # 学习通用的和器官特定的 Context
        self.prompt_learner = DualPromptLearner(
            self.clip_model, 
            num_organs=num_organs, 
            n_ctx_gen=8,  # 通用长度
            n_ctx_spec=8  # 特定长度
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = True # 解冻 Learner
            
        # 3. 初始化 PNuRL (PromptNu)
        # 用于 Explicit Attribute Injection (显式属性注入)
        self.pnurl = PNuRL(
            feat_dim=embed_dim, # SAM ViT 的输出通常是 256
            embed_dim=embed_dim,
            clip_model_path=None # 已经有 CLIP 了，PNuRL 内部如果不传 path 可以复用逻辑或跳过加载
        )
        # 这里我们需要手动共享一下 CLIP 给 PNuRL (如果 PNuRL 代码支持) 或者让 PNuRL 独立加载
        # 为简化，假设 PNuRL 作为一个 Attention 模块使用
        
        # 4. 初始化 Auto-Prompt Generator (SAC)
        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim
        )
        
        # 5. 冻结策略
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        for param in self.mask_decoder.parameters(): param.requires_grad = True
        
        # 解冻 Adapter
        adapter_count = 0
        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True
                adapter_count += 1
                
        print(f"✅ Model Ready: Adapters({adapter_count}), DualLearner, PNuRL Attention Unfrozen.")

    def forward(self, batched_input, multimask_output=False):
        # === Step 1: 基础图像编码 ===
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) # [B, 256, 64, 64]
        device = image_embeddings.device

        # 确保 CLIP 在正确设备
        if self.clip_model.visual.conv1.weight.device != device:
            self.clip_model = self.clip_model.to(device)

        # === Step 2: 数据提取 (Organ ID & Attribute Text) ===
        # 需要 DataLoader 配合传入 'organ_id' 和 'attribute_text'
        # 如果没有，使用默认值兜底
        organ_indices = []
        attribute_texts = []
        base_texts = [] # "Cell nuclei"

        for x in batched_input:
            # Organ ID: 用于特定库 (DualLearner)
            organ_indices.append(x.get("organ_id", 0)) 
            # Attribute Text: 用于显式规则 (PNuRL) - e.g. "Large, dark nuclei"
            attribute_texts.append(x.get("attribute_text", "")) 
            # Base Text: 用于 DualLearner 的基础 - e.g. "Cell nuclei"
            base_texts.append(x.get("text_prompt", "Cell nuclei"))

        organ_indices = torch.tensor(organ_indices).to(device)

        # === Step 3: Dual-Prompt Learner (Implicit Context) ===
        # 生成隐式的、包含通用和特定知识的文本特征
        # 同时构造负样本 (Background) 用于 Heatmap
        
        # Positive Prompts
        pos_tokens = clip.tokenize(base_texts, truncate=True).to(device)
        pos_feats = self.prompt_learner(organ_indices, pos_tokens) # [B, 512]
        
        # Negative Prompts (Background)
        # 我们可以认为 Background 也是一种“器官”，或者使用通用的 Background
        neg_tokens = clip.tokenize(["Background"] * len(base_texts), truncate=True).to(device)
        # 对于 Background，我们可能只用通用库，或者设定一个特殊的 organ_id
        # 这里简化处理：复用 organ_indices，假设每个器官的背景也不同
        neg_feats = self.prompt_learner(organ_indices, neg_tokens) # [B, 512]

        # 归一化并拼接
        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        text_features = torch.stack([pos_feats, neg_feats], dim=1).float() # [B, 2, 512]

        # === Step 4: PNuRL (Explicit Attribute Injection) ===
        # 利用显式的属性描述，对图像特征进行 Attention 加权
        # 这是 MP-SAM 的关键：Explicit Knowledge guiding Vision
        
        # 注意：我们需要确保 PNuRL 在正确设备
        if next(self.pnurl.parameters()).device != device:
            self.pnurl = self.pnurl.to(device)
            
        # PNuRL Forward
        # 返回: refined_embeddings (加权后的图像特征), context (属性上下文向量)
        # 如果 attribute_texts 为空，PNuRL 内部应处理为 Identity 或 Zero
        refined_image_embeddings, pnurl_context, _, _ = self.pnurl(
            image_features=image_embeddings,
            attribute_prompts=attribute_texts
        )
        
        # === Step 5: Auto-Prompt Generation (SAC) ===
        # 使用 "Refined" 的图像特征 + "Dual-Learned" 的文本特征
        # 生成 Heatmap 和 Points
        heatmap_logits = self.prompt_generator(refined_image_embeddings, text_features)
        
        # 提取点
        points_in_feat, point_labels = self.prompt_generator.get_points_from_heatmap(heatmap_logits, topk=1)
        
        # 坐标映射 (Feature Grid -> Original Image)
        feat_size = image_embeddings.shape[-1] 
        input_size = self.image_encoder.img_size 
        scale_factor = input_size / feat_size
        point_coords = (points_in_feat * scale_factor) + (scale_factor * 0.5)

        # === Step 6: SAM Mask Decoder ===
        # 融合 PNuRL 的属性上下文到 Prompt 中
        # 最终 Prompt Embedding = [Sparse(Points)] + [Dense(Refined Image)]
        # 也可以将 pnurl_context 作为额外的 Token 输入 Decoder (如果 Decoder 支持)
        # 这里我们主要依靠 Refined Image Embeddings 来传递属性信息
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # 使用 Refined Image Embeddings 进行解码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=refined_image_embeddings, # 🔥 使用 PNuRL 增强后的特征
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        # === Step 7: 结果封装 ===
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