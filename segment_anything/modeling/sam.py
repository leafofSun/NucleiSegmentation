# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL  # 假设 pnurl.py 已创建
import sys
import os

# === 依赖检查 ===
try:
    import clip
except ImportError:
    # 尝试添加路径 (根据你的项目结构调整)
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../CLIP")) 
    try:
        import clip
    except ImportError:
        print("⚠️ Warning: CLIP not found. DualPromptLearner will fail.")

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

# === 🔥 [模块 1] Dual-Prompt Learner (双层提示库) ===
class DualPromptLearner(nn.Module):
    def __init__(self, clip_model, num_organs=14, n_ctx_gen=8, n_ctx_spec=8):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        dtype = clip_model.dtype
        self.dtype = dtype

        # 通用特征库
        print(f"🧠 Init DualLearner: General({n_ctx_gen}) + Specific({n_ctx_spec}x{num_organs})")
        self.ctx_general = nn.Parameter(torch.empty(n_ctx_gen, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_general, std=0.02)
        
        # 特定特征库
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
        batch_size = len(organ_indices)
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)
        
        ctx_gen = self.ctx_general.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_spec = self.ctx_specific[organ_indices]
        ctx = torch.cat([ctx_gen, ctx_spec], dim=1) # [B, total_ctx, dim]

        prefix = embedding[:, :1, :] 
        suffix = embedding[:, 1 : 77 - self.total_ctx, :] 
        x = torch.cat([prefix, ctx, suffix], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_ln_final(x).type(self.dtype)

        original_eos_idx = tokenized_prompts.argmax(dim=-1)
        eos_idx = torch.clamp(original_eos_idx + self.total_ctx, max=76)
        text_features = x[torch.arange(x.shape[0]), eos_idx] @ self.clip_text_projection

        return text_features


# === 🔥 [模块 2] MP-SAM (TextSam) 核心类 ===
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
        num_organs=14 
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
        
        print(f"🚀 Initializing MP-SAM (Multi-granularity Prompt SAM)...")
        
        # 1. 加载 CLIP (Freeze)
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False 
            
        # 2. Dual-Prompt Learner (Trainable)
        self.prompt_learner = DualPromptLearner(
            self.clip_model, 
            num_organs=num_organs, 
            n_ctx_gen=8, 
            n_ctx_spec=8 
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = True 
            
        # 3. PNuRL (Trainable)
        self.pnurl = PNuRL(
            feature_dim=embed_dim, # 注意参数名可能要对应 pnurl.py
            # clip_model_path=None, 
            # num_classes_per_attr=[2, 3, 2, 3, 3] 
        )
        
        # 4. Auto-Prompt Generator (Trainable)
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
                
        print(f"✅ Model Ready: Adapters({adapter_count}), DualLearner, PNuRL, Generator Unfrozen.")

    def forward(self, batched_input, multimask_output=False):
        # === Step 1: 基础图像编码 ===
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) # [B, 256, 64, 64]
        device = image_embeddings.device

        if self.clip_model.visual.conv1.weight.device != device:
            self.clip_model = self.clip_model.to(device)

        # === Step 2: 数据提取 ===
        organ_indices = []
        attribute_texts = []
        base_texts = [] 

        for x in batched_input:
            organ_indices.append(x.get("organ_id", 0)) 
            attribute_texts.append(x.get("attribute_text", "")) 
            base_texts.append(x.get("text_prompt", "Cell nuclei"))

        organ_indices = torch.tensor(organ_indices).to(device)

        # === Step 3: Dual-Prompt Learner (Implicit Context) ===
        # Positive
        pos_tokens = clip.tokenize(base_texts, truncate=True).to(device)
        pos_feats = self.prompt_learner(organ_indices, pos_tokens) # [B, 512]
        
        # Negative (Background)
        neg_tokens = clip.tokenize(["Background"] * len(base_texts), truncate=True).to(device)
        neg_feats = self.prompt_learner(organ_indices, neg_tokens) # [B, 512]

        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        text_features = torch.stack([pos_feats, neg_feats], dim=1).float() # [B, 2, 512]

        # === Step 4: PNuRL (Explicit Attribute Injection) ===
        if next(self.pnurl.parameters()).device != device:
            self.pnurl = self.pnurl.to(device)
        
        # 准备属性标签 (Attribute Labels)
        attribute_labels_list = []
        for x in batched_input:
            attr_labels = x.get("attr_labels", None)
            if attr_labels is not None:
                attribute_labels_list.append(attr_labels)
            else:
                attribute_labels_list.append(torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))
        
        if len(attribute_labels_list) > 0:
            attr_labels_batch = torch.stack(attribute_labels_list).to(device)  # [B, 5]
        else:
            attr_labels_batch = None
            
        # PNuRL Forward
        # 返回: [logits_list], loss (refined_embeddings 暂未实现，如果 PNuRL 只是分类头)
        # 如果你希望 PNuRL 修正 Image Embedding，需要在 PNuRL forward 中实现 Attention
        # 假设 PNuRL 只负责计算 Loss，不改变 Feature (宏观监督)
        # 或者 PNuRL 返回 refined_features (如果实现了)
        
        # 这里假设 PNuRL 只是简单的分类头集合，不修改 image_features
        # 如果需要修改 image_features，请确保 PNuRL forward 返回修改后的特征
        # 这里我们直接使用原始 image_embeddings 继续，PNuRL 仅作为辅助 Loss
        attr_logits, pnurl_loss = self.pnurl(image_embeddings, attr_labels_batch)
        
        # 如果 PNuRL 返回 refined_features，则更新
        # refined_image_embeddings = ...
        refined_image_embeddings = image_embeddings # 目前保持不变

        # === Step 5: Auto-Prompt Generation (SAC - Adaptive) ===
        heatmap_logits = self.prompt_generator(refined_image_embeddings, text_features)
        
        # 🔥 [核心修改] 使用自适应采样 (Adaptive Sampling)
        # 获取含有 正点+负邻居 的 Prompt 列表
        prompts_list = self.prompt_generator.generate_adaptive_prompts(
            heatmap_logits, 
            threshold=0.3,       # 热力图阈值
            k_neighbors=3,       # 邻居数量
            dense_dist_thresh=15.0 # 拥挤距离阈值
        )
        
        # 坐标映射 (Feature Grid -> Original Image)
        feat_size = image_embeddings.shape[-1] 
        input_size = self.image_encoder.img_size 
        scale_factor = input_size / feat_size

        # === Step 6: SAM Mask Decoder (Loop Batch) ===
        outputs = []
        
        for i in range(len(batched_input)):
            # 获取当前样本的 Prompt 数据
            prompt_data = prompts_list[i]
            
            # 如果没有找到点 (全背景)，创建一个 Dummy Prompt 防止报错
            # 或者直接预测空 Mask (更合理)
            if not prompt_data["has_points"]:
                # 无点 -> 输出全黑 Mask
                # 构造一个空的 output 结构
                outputs.append({
                    "masks": torch.zeros((1, 1, 1024, 1024), device=device, dtype=torch.bool),
                    "iou_predictions": torch.zeros((1, 1), device=device),
                    "low_res_logits": torch.zeros((1, 1, 256, 256), device=device),
                    "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                    "pnurl_loss": pnurl_loss
                })
                continue

            # 提取坐标和标签
            # coords: [N_cells, K+1, 2]
            # labels: [N_cells, K+1]
            point_coords = prompt_data["point_coords"]
            point_labels = prompt_data["point_labels"]
            
            # 缩放坐标到 1024
            point_coords = (point_coords * scale_factor) + (scale_factor * 0.5)
            
            # 喂给 Prompt Encoder
            # sparse_embeddings: [N_cells, tokens, channel]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            # 扩展 Image Embedding 以匹配 N_cells
            # curr_embedding: [256, 64, 64] -> [1, 256, 64, 64] -> [N_cells, 256, 64, 64]
            num_cells = point_coords.shape[0]
            curr_img_embed = refined_image_embeddings[i].unsqueeze(0).expand(num_cells, -1, -1, -1)
            
            # 解码
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_img_embed,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # === Step 7: 后处理 & 聚合 ===
            # low_res_masks: [N_cells, 1, 256, 256]
            # 我们需要把它合并成一张图 (Instance Segmentation -> Semantic Mask for Loss)
            # 或者保留 Instance 形式计算 Loss (如果 Loss 支持)
            # 这里为了适配原来的 pipeline，我们将 N 个 Mask 取 Max 合并
            # (注意：这是简化处理，严格来说应该匹配 GT 的 Instance ID)
            
            # 合并策略: Max Pool (只要有一个细胞预测是前景，就是前景)
            merged_logits, _ = torch.max(low_res_masks, dim=0, keepdim=True) # [1, 1, 256, 256]
            
            # IoU 也可以取平均或最大
            merged_iou, _ = torch.max(iou_predictions, dim=0, keepdim=True)

            mask_post = self.postprocess_masks(
                merged_logits,
                input_size=batched_input[i]["image"].shape[-2:], 
                original_size=batched_input[i]["original_size"],
            )
            
            outputs.append({
                "masks": mask_post > self.mask_threshold, # Boolean Mask
                "iou_predictions": merged_iou,
                "low_res_logits": merged_logits,
                "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                "attr_logits": None, # 暂不返回详细 Logits 以省显存
                "pnurl_loss": pnurl_loss
            })
            
        return outputs