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
        # 基础 SAM forward 逻辑保持不变
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

# === Physical Adapter (适配器) ===
class PhysicalAdapter(nn.Module):
    """
    物理特征适配器：将融合后的 Shape+Size+Density 特征转换为 FiLM 参数
    """
    def __init__(self, feat_dim_low: int, feat_dim_high: int, ctx_dim: int):
        """
        Args:
            feat_dim_low: 浅层特征维度 (3 * in_dim // 4) - 拼接后的维度
            feat_dim_high: 深层特征维度 (3 * in_dim // 2) - 拼接后的维度
            ctx_dim: Prompt 的维度 (CLIP ctx_dim)
        """
        super().__init__()
        
        # 注意：这里的 feat_dim_low 和 high 已经是拼接后的维度 (x3)
        
        # 浅层特征适配器（用于调制浅层 Prompt）
        self.adapter_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim_low, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2)  # 输出 [γ, β]，每个维度为 ctx_dim
        )
        
        # 深层特征适配器（用于调制深层 Prompt）
        self.adapter_high = nn.Sequential(
            nn.Linear(feat_dim_high, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2)  # 输出 [γ, β]，每个维度为 ctx_dim
        )
        
        # Zero-initialization: 将最后一层权重设为0
        nn.init.zeros_(self.adapter_low[-1].weight)
        nn.init.zeros_(self.adapter_low[-1].bias)
        nn.init.zeros_(self.adapter_high[-1].weight)
        nn.init.zeros_(self.adapter_high[-1].bias)
    
    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_low: [B, feat_dim_low, H, W] 浅层特征（拼接后的）
            feat_high: [B, feat_dim_high] 深层特征（拼接后的）
        
        Returns:
            gamma_low: [B, ctx_dim] 浅层缩放参数
            beta_low: [B, ctx_dim] 浅层偏置参数
            gamma_high: [B, ctx_dim] 深层缩放参数
            beta_high: [B, ctx_dim] 深层偏置参数
        """
        # 浅层适配器
        low_params = self.adapter_low(feat_low)  # [B, ctx_dim * 2]
        gamma_low, beta_low = torch.chunk(low_params, 2, dim=1)  # 各 [B, ctx_dim]
        
        # 深层适配器
        high_params = self.adapter_high(feat_high)  # [B, ctx_dim * 2]
        gamma_high, beta_high = torch.chunk(high_params, 2, dim=1)  # 各 [B, ctx_dim]
        
        return gamma_low, beta_low, gamma_high, beta_high


# === 🔥 [模块 1] Dual-Prompt Learner (双层提示库) ===
class DualPromptLearner(nn.Module):
    def __init__(self, clip_model, num_organs=21, n_ctx_gen=8, n_ctx_spec=8, embed_dim=256):
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
        self.ctx_dim = ctx_dim
        
        # 🔥 [关键修改] 计算输入维度
        # PNuRL 拼接了 3 个头 (Shape, Size, Density)
        # 每个头的 shallow_branch 输出 C/4 -> 拼接后 3 * C/4
        # 每个头的 deep_branch 输出 C/2    -> 拼接后 3 * C/2
        num_fused_heads = 3 
        
        feat_dim_low = (embed_dim // 4) * num_fused_heads
        feat_dim_high = (embed_dim // 2) * num_fused_heads
        
        self.physical_adapter = PhysicalAdapter(feat_dim_low, feat_dim_high, ctx_dim)
        print(f"✅ PhysicalAdapter initialized: in_low={feat_dim_low}, in_high={feat_dim_high} (Fused {num_fused_heads} heads)")

    def forward(self, organ_indices, tokenized_prompts, density_features: Optional[List[torch.Tensor]] = None):
        """
        Args:
            organ_indices: [B] 器官索引
            tokenized_prompts: [B, 77] tokenized prompts
            density_features: Optional[List[torch.Tensor]] = [fused_low, fused_high]
                - fused_low: [B, 3*C/4, H, W] 拼接后的浅层特征 (Shape+Size+Density)
                - fused_high: [B, 3*C/2] 拼接后的深层特征 (Shape+Size+Density)
        """
        batch_size = len(organ_indices)
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)
        
        ctx_gen = self.ctx_general.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_spec = self.ctx_specific[organ_indices]
        ctx = torch.cat([ctx_gen, ctx_spec], dim=1) # [B, total_ctx, dim]

        # === FiLM 调制：使用物理特征调制 Prompt ===
        if density_features is not None:
            # 这里 density_features 实际上是 [fused_low, fused_high]
            feat_low, feat_high = density_features
            gamma_low, beta_low, gamma_high, beta_high = self.physical_adapter(feat_low, feat_high)
            
            # 多尺度策略：
            # - 浅层特征调制浅层 Prompt (ctx_gen 的前半部分)
            # - 深层特征调制深层 Prompt (ctx_gen 的后半部分 + ctx_spec)
            n_gen_low = self.n_ctx_gen // 2
            n_gen_high = self.n_ctx_gen - n_gen_low
            
            # 调制浅层 Prompt (ctx_gen 的前半部分)
            ctx_gen_low = ctx_gen[:, :n_gen_low, :]  # [B, n_gen_low, ctx_dim]
            gamma_low_expanded = gamma_low.unsqueeze(1).expand(-1, n_gen_low, -1)  # [B, n_gen_low, ctx_dim]
            beta_low_expanded = beta_low.unsqueeze(1).expand(-1, n_gen_low, -1)  # [B, n_gen_low, ctx_dim]
            ctx_gen_low_modulated = (1 + gamma_low_expanded) * ctx_gen_low + beta_low_expanded
            
            # 调制深层 Prompt (ctx_gen 的后半部分 + ctx_spec)
            ctx_gen_high = ctx_gen[:, n_gen_low:, :]  # [B, n_gen_high, ctx_dim]
            ctx_spec_mod = ctx_spec  # [B, n_ctx_spec, ctx_dim]
            
            gamma_high_expanded_gen = gamma_high.unsqueeze(1).expand(-1, n_gen_high, -1)  # [B, n_gen_high, ctx_dim]
            beta_high_expanded_gen = beta_high.unsqueeze(1).expand(-1, n_gen_high, -1)  # [B, n_gen_high, ctx_dim]
            ctx_gen_high_modulated = (1 + gamma_high_expanded_gen) * ctx_gen_high + beta_high_expanded_gen
            
            gamma_high_expanded_spec = gamma_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  # [B, n_ctx_spec, ctx_dim]
            beta_high_expanded_spec = beta_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  # [B, n_ctx_spec, ctx_dim]
            ctx_spec_modulated = (1 + gamma_high_expanded_spec) * ctx_spec_mod + beta_high_expanded_spec
            
            # 重新组合
            ctx_gen = torch.cat([ctx_gen_low_modulated, ctx_gen_high_modulated], dim=1)
            ctx_spec = ctx_spec_modulated
            ctx = torch.cat([ctx_gen, ctx_spec], dim=1)

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
        num_organs=21 
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
            n_ctx_spec=8,
            embed_dim=embed_dim  # 传递 embed_dim 用于初始化 DensityAdapter
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = True 
            
        # 3. PNuRL (Trainable)
        self.pnurl = PNuRL(
            embed_dim=embed_dim,
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

        # === Step 3: PNuRL (先获取密度特征) ===
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
            # 拆分为 5 个 tensor list
            attribute_labels = [attr_labels_batch[:, i] for i in range(5)]
        else:
            attribute_labels = None
        
        # PNuRL Forward - 获取密度特征（多任务：分类 + 回归）
        refined_image_embeddings, pnurl_context, pnurl_loss, attr_logits, density_features, density_map = self.pnurl(
            image_features=image_embeddings,
            attribute_labels=attribute_labels,
            attribute_prompts=attribute_texts,
            return_loss=True
        )
        # density_map: [B, 1, H', W'] - 像素级密度图（用于 DeNSe 式强对齐）
        # density_features: [fused_low, fused_high] - 多尺度特征
        # 🔥 [核心改进] 提取高频特征用于 ASR-Guided Decoder
        high_freq_guide = density_features[0] if isinstance(density_features, (list, tuple)) and len(density_features) > 0 else None
        # high_freq_guide: [B, 192, 64, 64] - PNuRL 浅层特征（用于边界细化）
        
        # === Step 4: Dual-Prompt Learner (Implicit Context with Density Modulation) ===
        # Positive
        pos_tokens = clip.tokenize(base_texts, truncate=True).to(device)
        pos_feats = self.prompt_learner(organ_indices, pos_tokens, density_features=density_features) # [B, 512]
        
        # Negative (Background)
        neg_tokens = clip.tokenize(["Background"] * len(base_texts), truncate=True).to(device)
        neg_feats = self.prompt_learner(organ_indices, neg_tokens, density_features=density_features) # [B, 512]

        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        text_features = torch.stack([pos_feats, neg_feats], dim=1).float() # [B, 2, 512]

        # === Step 5: Auto-Prompt Generation (SAC - Adaptive) ===
        heatmap_logits = self.prompt_generator(refined_image_embeddings, text_features)
        
        # 🔥 [New] 准备传给 ASR 的密度图
        # Sigmoid 归一化到 0~1，作为空间门控
        density_map_proxy = torch.sigmoid(heatmap_logits[:, 0:1, :, :])  # [B, 1, 64, 64]
        
        # 🔥 [核心改进] 动态阈值计算：基于 PNuRL 的 Size 预测
        # 1. 获取 Size 预测类别
        size_logits = attr_logits.get('size', None)  # [B, num_classes] 或 None
        if size_logits is not None and size_logits.numel() > 0:
            # 获取预测的 Size 类别 (0=Small, 1=Medium, 2=Large)
            pred_size_class = torch.argmax(size_logits, dim=1)  # [B] -> 0, 1, 2
            
            # 2. 定义映射规则：小细胞允许靠得更近，大细胞需要更严格
            # Small(0) -> 10.0, Medium(1) -> 15.0, Large(2) -> 20.0
            size_threshold_map = torch.tensor([10.0, 15.0, 20.0], device=device)
            adaptive_thresh = size_threshold_map[pred_size_class]  # [B]
        else:
            # 如果 PNuRL 未输出 Size 或处于训练初期，使用固定阈值
            batch_size = image_embeddings.shape[0]
            adaptive_thresh = torch.tensor(15.0, device=device).expand(batch_size)
        
        # 智能决定是否限制点数: 训练时限制(50)，验证时不限制
        limit_points = 50 if self.training else None

        prompts_list = self.prompt_generator.generate_adaptive_prompts(
            heatmap_logits, 
            threshold=0.3,       
            k_neighbors=3,       
            dense_dist_thresh=adaptive_thresh,  # 🔥 传入动态阈值 Tensor
            max_points=limit_points
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
            
            # 🔥 [修复] 处理无点情况，防止 DDP 死锁
            if not prompt_data["has_points"]:
                # ✅ 正确写法：利用 image_embeddings 构造 dummy，保持计算图连接
                # 0 * sum() 既保留了连接，又不会产生实际梯度影响
                dummy_connection = refined_image_embeddings[i].sum() * 0.0
                
                # 获取正确的目标尺寸 (Original Size, e.g., 512x512)
                target_h, target_w = batched_input[i]["original_size"]
                
                # 🔥 处理 density_map
                density_map_i = None
                if density_map is not None:
                    density_map_raw = density_map[i]  # [1, H', W']
                    # 如果尺寸不对，插值到 original_size
                    if density_map_raw.shape[-2:] != (target_h, target_w):
                        density_map_i = F.interpolate(
                            density_map_raw.unsqueeze(0), 
                            size=(target_h, target_w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)  # [1, target_h, target_w]
                    else:
                        density_map_i = density_map_raw
                    
                    # 加上 dummy_connection (虽然 density_map 本身就在图里，加这个双保险)
                    density_map_i = density_map_i + dummy_connection
                
                outputs.append({
                    # 🔥 [修正] 使用 (target_h, target_w) 而不是 input_size (1024)
                    "masks": (torch.zeros((1, 1, target_h, target_w), device=device, dtype=torch.float32) - 100.0) + dummy_connection,
                    "iou_predictions": torch.zeros((1, 1), device=device) + dummy_connection,
                    # low_res_logits 保持 256x256 是对的，这是 Decoder 的原始输出尺寸
                    "low_res_logits": (torch.zeros((1, 1, 256, 256), device=device) - 100.0) + dummy_connection,
                    "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                    "attr_logits": attr_logits,
                    "density_map": density_map_i,
                    "pnurl_loss": pnurl_loss
                })
                continue

            # 提取坐标和标签
            point_coords = prompt_data["point_coords"]
            point_labels = prompt_data["point_labels"]
            
            # 缩放坐标到 1024
            point_coords = (point_coords * scale_factor) + (scale_factor * 0.5)
            
            # 喂给 Prompt Encoder (points=(N_cells, K+1, 2))
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            # 扩展 Image Embedding 以匹配 N_cells
            # refined_image_embeddings[i]: [256, 64, 64] -> [1, 256, 64, 64] -> [N_cells, 256, 64, 64]
            num_cells = point_coords.shape[0]
            curr_img_embed = refined_image_embeddings[i].unsqueeze(0).expand(num_cells, -1, -1, -1)
            
            # 🔥 [核心改进] 扩展高频特征以匹配 N_cells（用于 ASR-Guided Decoder）
            curr_high_freq = None
            if high_freq_guide is not None:
                # high_freq_guide[i]: [192, 64, 64] -> [1, 192, 64, 64] -> [N_cells, 192, 64, 64]
                curr_high_freq = high_freq_guide[i].unsqueeze(0).expand(num_cells, -1, -1, -1)
            
            # 🔥 [New] 准备当前样本的密度图
            # density_map_proxy: [B, 1, 64, 64] -> 取第 i 个样本
            curr_density_map = density_map_proxy[i].unsqueeze(0)  # [1, 1, 64, 64]
            # 需要扩展到 N_cells 维度
            curr_density_map = curr_density_map.expand(num_cells, -1, -1, -1)  # [N_cells, 1, 64, 64]
            
            # 解码（注入高频特征用于边界细化，同时传入密度图）
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_img_embed,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                high_freq_features=curr_high_freq,  # 🔥 注入高频特征
                density_map=curr_density_map,  # 🔥 [New] 传入密度图
            )
            
            # === Step 7: 后处理 & 聚合 ===
            # low_res_masks: [N_cells, 1, 256, 256] -> 合并成单图 [1, 1, 256, 256]
            merged_logits, _ = torch.max(low_res_masks, dim=0, keepdim=True) 
            
            # IoU 聚合: 取平均
            merged_iou = torch.mean(iou_predictions, dim=0, keepdim=True)

            mask_post = self.postprocess_masks(
                merged_logits,
                input_size=batched_input[i]["image"].shape[-2:], 
                original_size=batched_input[i]["original_size"],
            )
            
            # 🔥 [新增] 将 density_map 调整到与 mask 相同的大小
            # mask_post 的大小是 original_size，density_map 需要匹配
            target_h, target_w = mask_post.shape[-2:]
            density_map_i = density_map[i]  # [1, H', W']
            if density_map_i.shape[-2:] != (target_h, target_w):
                density_map_i = F.interpolate(
                    density_map_i.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)  # [1, target_h, target_w]
            
            outputs.append({
                "masks": mask_post, # ✅ 确认：返回 Float Logits
                "iou_predictions": merged_iou,
                "low_res_logits": merged_logits,
                "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                "attr_logits": attr_logits,  # 传递属性 logits（包含 density 分类）
                "density_map": density_map_i,  # 🔥 [新增] 像素级密度图（用于 DeNSe 式强对齐）
                "pnurl_loss": pnurl_loss
            })
            
        return outputs