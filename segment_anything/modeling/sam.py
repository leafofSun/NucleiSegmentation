# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
import torchvision.models as models  # 🔥 [新增] 导入 torchvision 模型

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL
from .sg_ot import SemanticGuidedOT
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        super().__init__()
        
        # 浅层特征适配器
        self.adapter_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim_low, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2) 
        )
        
        # 深层特征适配器
        self.adapter_high = nn.Sequential(
            nn.Linear(feat_dim_high, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2) 
        )
        
        # Zero-initialization
        nn.init.zeros_(self.adapter_low[-1].weight)
        nn.init.zeros_(self.adapter_low[-1].bias)
        nn.init.zeros_(self.adapter_high[-1].weight)
        nn.init.zeros_(self.adapter_high[-1].bias)
    
    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_params = self.adapter_low(feat_low)  
        gamma_low, beta_low = torch.chunk(low_params, 2, dim=1) 
        
        high_params = self.adapter_high(feat_high)  
        gamma_high, beta_high = torch.chunk(high_params, 2, dim=1) 
        
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
        
        num_fused_heads = 3 
        
        feat_dim_low = (embed_dim // 4) * num_fused_heads
        feat_dim_high = (embed_dim // 2) * num_fused_heads
        
        self.physical_adapter = PhysicalAdapter(feat_dim_low, feat_dim_high, ctx_dim)
        print(f"✅ PhysicalAdapter initialized: in_low={feat_dim_low}, in_high={feat_dim_high} (Fused {num_fused_heads} heads)")

    def forward(self, organ_indices, tokenized_prompts, density_features: Optional[List[torch.Tensor]] = None):
        batch_size = len(organ_indices)
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)
        
        ctx_gen = self.ctx_general.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_spec = self.ctx_specific[organ_indices]
        ctx = torch.cat([ctx_gen, ctx_spec], dim=1) # [B, total_ctx, dim]

        # === FiLM 调制：使用物理特征调制 Prompt ===
        if density_features is not None:
            feat_low, feat_high = density_features
            gamma_low, beta_low, gamma_high, beta_high = self.physical_adapter(feat_low, feat_high)
            
            n_gen_low = self.n_ctx_gen // 2
            n_gen_high = self.n_ctx_gen - n_gen_low
            
            ctx_gen_low = ctx_gen[:, :n_gen_low, :]  
            gamma_low_expanded = gamma_low.unsqueeze(1).expand(-1, n_gen_low, -1)  
            beta_low_expanded = beta_low.unsqueeze(1).expand(-1, n_gen_low, -1)  
            ctx_gen_low_modulated = (1 + gamma_low_expanded) * ctx_gen_low + beta_low_expanded
            
            ctx_gen_high = ctx_gen[:, n_gen_low:, :]  
            ctx_spec_mod = ctx_spec  
            
            gamma_high_expanded_gen = gamma_high.unsqueeze(1).expand(-1, n_gen_high, -1)  
            beta_high_expanded_gen = beta_high.unsqueeze(1).expand(-1, n_gen_high, -1)  
            ctx_gen_high_modulated = (1 + gamma_high_expanded_gen) * ctx_gen_high + beta_high_expanded_gen
            
            gamma_high_expanded_spec = gamma_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  
            beta_high_expanded_spec = beta_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  
            ctx_spec_modulated = (1 + gamma_high_expanded_spec) * ctx_spec_mod + beta_high_expanded_spec
            
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
        num_organs=21,
        num_heads=8,
        sg_epsilon=0.05,
        sg_iters=3,
        use_pnurl: bool = True,
        use_coop: bool = True,
        use_sgot: bool = True,
        use_asr: bool = True,
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)

        self.use_pnurl = use_pnurl
        self.use_coop = use_coop
        self.use_sgot = use_sgot
        self.use_asr = use_asr

        print(f"🚀 Initializing MP-SAM (Multi-granularity Prompt SAM)...")
        print(
            f"   Ablation: PNuRL={use_pnurl}, CoOp={use_coop}, SG-OT={use_sgot}, ASR={use_asr}"
        )

        # 1. 加载 CLIP (Freeze)
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 2. Dual-Prompt Learner (Trainable; 关闭 CoOp 时不更新)
        self.prompt_learner = DualPromptLearner(
            self.clip_model,
            num_organs=num_organs,
            n_ctx_gen=8,
            n_ctx_spec=8,
            embed_dim=embed_dim,
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = use_coop

        # 3. PNuRL (Trainable; 关闭时不更新)
        self.pnurl = PNuRL(embed_dim=embed_dim)
        for param in self.pnurl.parameters():
            param.requires_grad = use_pnurl

        # 4. Auto-Prompt Generator (Trainable)
        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        # SG-OT（关闭时不更新）或纯视觉独立 HV 头
        if self.use_sgot:
            self.sg_ot = SemanticGuidedOT(
                img_dim=embed_dim,
                txt_dim=text_dim,
                epsilon=sg_epsilon,
                sinkhorn_iters=sg_iters,
            )
            for param in self.sg_ot.parameters():
                param.requires_grad = True
        else:
            # 🔥 [新增] 纯视觉基线的独立 HV 预测头
            self.basic_hv_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, 2, kernel_size=1)
            )

        # 🔥 [新增] 引入纯视觉基线的高频特征提取器 (ResNet-50)
        if self.use_asr:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.cnn_stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.cnn_stage1 = resnet.layer1 # 输出: [B, 256, H/4, W/4]
            self.cnn_stage2 = resnet.layer2 # 输出: [B, 512, H/8, W/8]

        # 5. 冻结策略
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        adapter_count = 0
        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True
                adapter_count += 1

        print(
            f"✅ Model Ready: Adapters({adapter_count}), "
            f"DualLearner({'on' if use_coop else 'off'}), PNuRL({'on' if use_pnurl else 'off'}), "
            f"SG-OT({'on' if use_sgot else 'off'}), Generator."
        )

    def forward(self, batched_input, multimask_output=False):
        # === Step 1: 基础图像编码 ===
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) # [B, 256, 64, 64]
        device = image_embeddings.device

        # 🔥 [新增] 同步提取 CNN 高频物理特征
        feat_s1, feat_s2 = None, None
        if self.use_asr:
            with torch.autocast('cuda', enabled=True):
                feat_s0 = self.cnn_stage0(input_images)
                feat_s1 = self.cnn_stage1(feat_s0) # 1/4 尺度
                feat_s2 = self.cnn_stage2(feat_s1) # 1/8 尺度

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
        attribute_labels_list = []
        for x in batched_input:
            attr_labels = x.get("attr_labels", None)
            if attr_labels is not None:
                attribute_labels_list.append(attr_labels)
            else:
                attribute_labels_list.append(torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))

        if len(attribute_labels_list) > 0:
            attr_labels_batch = torch.stack(attribute_labels_list).to(device)
            attribute_labels = [attr_labels_batch[:, i] for i in range(5)]
        else:
            attribute_labels = None

        if self.use_pnurl:
            if next(self.pnurl.parameters()).device != device:
                self.pnurl = self.pnurl.to(device)
            refined_image_embeddings, pnurl_context, pnurl_loss, attr_logits, density_features, density_map = self.pnurl(
                image_features=image_embeddings,
                attribute_labels=attribute_labels,
                attribute_prompts=attribute_texts,
                return_loss=True,
            )
            high_freq_guide = (
                density_features[0]
                if isinstance(density_features, (list, tuple)) and len(density_features) > 0
                else None
            )
        else:
            refined_image_embeddings = image_embeddings
            pnurl_loss = torch.tensor(0.0, device=device)
            attr_logits = {}
            density_features = None
            density_map = None
            high_freq_guide = None

        # === Step 4: CoOp 可学习提示 或 冻结 CLIP 文本编码 ===
        pos_tokens = clip.tokenize(base_texts, truncate=True).to(device)
        neg_tokens = clip.tokenize(["Background"] * len(base_texts), truncate=True).to(device)
        if self.use_coop:
            if next(self.prompt_learner.parameters()).device != device:
                self.prompt_learner = self.prompt_learner.to(device)
            pos_feats = self.prompt_learner(organ_indices, pos_tokens, density_features=density_features)
            neg_feats = self.prompt_learner(organ_indices, neg_tokens, density_features=density_features)
        else:
            with torch.no_grad():
                pos_feats = self.clip_model.encode_text(pos_tokens).float()
                neg_feats = self.clip_model.encode_text(neg_tokens).float()

        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        text_features = torch.stack([pos_feats, neg_feats], dim=1).float()

        # === Step 5: SG-OT 或 纯视觉独立头 ===
        pos_text_feats = text_features[:, 0, :]
        B, C, H, W = refined_image_embeddings.shape
        sg_ot_density = density_map
        if sg_ot_density is None:
            sg_ot_density = torch.ones(B, 1, H, W, device=device) / (H * W)

        if self.use_sgot:
            if next(self.sg_ot.parameters()).device != device:
                self.sg_ot = self.sg_ot.to(device)
            fused_image_embeddings, heatmap_logits, hv_logits = self.sg_ot(
                img_feat=refined_image_embeddings,
                txt_feat=pos_text_feats,
                density_map=sg_ot_density,
            )
        else:
            # 🔥 [修复] 纯视觉通路：直接使用独立卷积头生成 HV 距离图
            fused_image_embeddings = refined_image_embeddings
            heatmap_logits = self.prompt_generator(refined_image_embeddings, text_features)
            hv_logits = self.basic_hv_head(refined_image_embeddings)

        density_map_proxy = torch.sigmoid(heatmap_logits[:, 0:1, :, :])

        # 动态阈值计算
        size_logits = attr_logits.get('size', None)
        if size_logits is not None and size_logits.numel() > 0:
            pred_size_class = torch.argmax(size_logits, dim=1)
            size_threshold_map = torch.tensor([10.0, 15.0, 20.0], device=device)
            adaptive_thresh = size_threshold_map[pred_size_class]
        else:
            batch_size = image_embeddings.shape[0]
            adaptive_thresh = torch.tensor(15.0, device=device).expand(batch_size)

        prompts_list = self.prompt_generator.generate_adaptive_prompts(
            heatmap_logits,
            threshold=0.3,
            k_neighbors=3,
            dense_dist_thresh=adaptive_thresh,
            pred_density=density_map if self.use_pnurl else None,
        )
        
        # 坐标映射 (Feature Grid -> Original Image)
        feat_size = image_embeddings.shape[-1] 
        input_size = self.image_encoder.img_size 
        scale_factor = input_size / feat_size

        # === Step 6: SAM Mask Decoder (Loop Batch) ===
        outputs = []
        
        for i in range(len(batched_input)):
            prompt_data = prompts_list[i]
            target_h, target_w = batched_input[i]["original_size"]
            
            density_map_i = None
            if density_map is not None:
                density_map_raw = density_map[i]
                if density_map_raw.shape[-2:] != (target_h, target_w):
                    density_map_i = F.interpolate(
                        density_map_raw.unsqueeze(0), 
                        size=(target_h, target_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                else:
                    density_map_i = density_map_raw

            if not prompt_data["has_points"]:
                dummy_connection = fused_image_embeddings[i].sum() * 0.0
                if density_map_i is not None:
                    density_map_i = density_map_i + dummy_connection
                
                hv_out = hv_logits[i].unsqueeze(0) if hv_logits is not None else None
                outputs.append({
                    "masks": (torch.zeros((1, 1, target_h, target_w), device=device, dtype=torch.float32) - 100.0) + dummy_connection,
                    "iou_predictions": torch.zeros((1, 1), device=device) + dummy_connection,
                    "low_res_logits": (torch.zeros((1, 1, 256, 256), device=device) - 100.0) + dummy_connection,
                    "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                    "hv_logits": hv_out,
                    "attr_logits": attr_logits,
                    "density_map": density_map_i,
                    "pnurl_loss": pnurl_loss
                })
                continue

            point_coords = prompt_data["point_coords"]
            point_labels = prompt_data["point_labels"]
            
            point_coords = (point_coords * scale_factor) + (scale_factor * 0.5)
            if self.training:
                # 训练模式：随机采样最多 512 个点，既防爆显存，又当做数据增强 (Point Dropout)
                MAX_POINTS = 512 
                if point_coords.shape[0] > MAX_POINTS:
                    indices = torch.randperm(point_coords.shape[0], device=device)[:MAX_POINTS]
                    point_coords = point_coords[indices]
                    point_labels = point_labels[indices]
            else:
                # 测试模式：绝对不截断！保留所有点，确保不会漏检任何一个细胞核
                pass

            # 🔥🔥🔥 [核心显存优化: Chunked Decoding] 🔥🔥🔥
            num_cells = point_coords.shape[0]
            chunk_size = 16 
            chunk_masks = []
            chunk_ious = []
            
            for start_idx in range(0, num_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, num_cells)
                
                sub_coords = point_coords[start_idx:end_idx] 
                sub_labels = point_labels[start_idx:end_idx] 
                current_batch = sub_coords.shape[0]

                sub_img_embed = fused_image_embeddings[i].unsqueeze(0).expand(current_batch, -1, -1, -1)
                
                sub_high_freq = None
                if self.use_asr and high_freq_guide is not None:
                    sub_high_freq = high_freq_guide[i].unsqueeze(0).expand(current_batch, -1, -1, -1)

                sub_density_map = None
                if self.use_asr and density_map_proxy is not None:
                    sub_density_map = density_map_proxy[i].unsqueeze(0).expand(current_batch, -1, -1, -1)
                sub_text_feat = pos_text_feats[i].unsqueeze(0).expand(current_batch, -1)

                # 🔥 扩展 CNN 侧向特征匹配 Chunk Size
                sub_cnn_s1 = feat_s1[i].unsqueeze(0).expand(current_batch, -1, -1, -1) if self.use_asr else None
                sub_cnn_s2 = feat_s2[i].unsqueeze(0).expand(current_batch, -1, -1, -1) if self.use_asr else None

                sparse, dense = self.prompt_encoder(
                    points=(sub_coords, sub_labels),
                    boxes=None,
                    masks=None,
                )

                # 🔥 [核心注入] 仅传入物理边缘，纯视觉 ASR
                sub_mask, sub_iou = self.mask_decoder(
                    image_embeddings=sub_img_embed,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=multimask_output,
                    cnn_feat_s1=sub_cnn_s1,  
                    cnn_feat_s2=sub_cnn_s2,  
                )
                chunk_masks.append(sub_mask)
                chunk_ious.append(sub_iou)
            
            low_res_masks = torch.cat(chunk_masks, dim=0) 
            iou_predictions = torch.cat(chunk_ious, dim=0) 

            merged_logits, _ = torch.max(low_res_masks, dim=0, keepdim=True) 
            merged_iou = torch.mean(iou_predictions, dim=0, keepdim=True)

            mask_post = self.postprocess_masks(
                merged_logits,
                input_size=batched_input[i]["image"].shape[-2:], 
                original_size=batched_input[i]["original_size"],
            )
            
            hv_out = hv_logits[i].unsqueeze(0) if hv_logits is not None else None
            outputs.append({
                "masks": mask_post,
                "iou_predictions": merged_iou,
                "low_res_logits": merged_logits,
                "heatmap_logits": heatmap_logits[i].unsqueeze(0),
                "hv_logits": hv_out,
                "attr_logits": attr_logits,
                "density_map": density_map_i,
                "pnurl_loss": pnurl_loss
            })
            
        return outputs