import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
import torchvision.models as models 

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL
from .ot import DensityGuidedOT 
import sys
import os
from dotenv import load_dotenv


try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
except ImportError:
    print("⚠️ Warning: CONCH not found. Please install it.")

try:
    from prompt_generator import TextGuidedPointGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    from prompt_generator import TextGuidedPointGenerator

load_dotenv()
hf_auth_token = os.environ.get("HF_TOKEN")

class GlobalASRUpsampler(nn.Module):
    """
    全局高分辨率特征上采样器 (ASR-HV)
    现在的高频输入不再来源于 PNuRL 的融合特征，而是直接来源于底层的纯 CNN。
    """
    def __init__(self, embed_dim=256, hm_channels=2, use_asr=True):
        super().__init__()
        self.use_asr = use_asr
        
        self.init_conv = nn.Conv2d(embed_dim + 2 + hm_channels, 256, kernel_size=3, padding=1)
        
        # 上采样 1: 64x64 -> 128x128 (拼接 feat_s2: 512通道)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + (512 if use_asr else 0), 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 2: 128x128 -> 256x256 (拼接 feat_s1: 256通道)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + (256 if use_asr else 0), 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 3: 256x256 -> 512x512 (拼接 feat_half: 64通道)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 + (64 if use_asr else 0), 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 上采样 4: 512x512 -> 1024x1024
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.hv_out = nn.Conv2d(16, 2, kernel_size=1)
        self.hm_out = nn.Conv2d(16, hm_channels, kernel_size=1)

    def forward(self, sam_feat, hv_logits, hm_logits, feat_s2=None, feat_s1=None, feat_half=None):
        x = torch.cat([sam_feat, hv_logits, hm_logits], dim=1)
        x = self.init_conv(x)
        
        x = self.up1(x) 
        if self.use_asr and feat_s2 is not None:
            x = torch.cat([x, feat_s2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x) 
        if self.use_asr and feat_s1 is not None:
            x = torch.cat([x, feat_s1], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x) 
        if self.use_asr and feat_half is not None:
            x = torch.cat([x, feat_half], dim=1)
        x = self.conv3(x)

        x = self.up4(x) 
        x = self.conv4(x)
        
        return self.hv_out(x), self.hm_out(x)


class PhysicalAdapter(nn.Module):
    def __init__(self, feat_dim_low: int, feat_dim_high: int, ctx_dim: int):
        super().__init__()
        self.adapter_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim_low, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2) 
        )
        self.adapter_high = nn.Sequential(
            nn.Linear(feat_dim_high, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2) 
        )
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


class DualPromptLearner(nn.Module):
    def __init__(self, clip_model, num_organs=21, n_ctx_gen=8, n_ctx_spec=8, embed_dim=256):
        super().__init__()
        
        # 兼容 CONCH (OpenCLIP) 架构
        if hasattr(clip_model, 'text'):
            text_encoder = clip_model.text
        else:
            text_encoder = clip_model

        ctx_dim = text_encoder.ln_final.weight.shape[0] # 512
        dtype = next(clip_model.parameters()).dtype 
        self.dtype = dtype

        print(f"🧠 Init DualLearner: General({n_ctx_gen}) + Specific({n_ctx_spec}x{num_organs})")
        self.ctx_general = nn.Parameter(torch.empty(n_ctx_gen, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_general, std=0.02)
        
        self.ctx_specific = nn.Parameter(torch.empty(num_organs, n_ctx_spec, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_specific, std=0.02)
        
        self.clip_token_embedding = text_encoder.token_embedding
        self.clip_transformer = text_encoder.transformer
        self.clip_ln_final = text_encoder.ln_final
        self.clip_text_projection = text_encoder.text_projection
        
        self.n_ctx_gen = n_ctx_gen
        self.n_ctx_spec = n_ctx_spec
        self.total_ctx = n_ctx_gen + n_ctx_spec
        self.ctx_dim = ctx_dim
        
        num_fused_heads = 3 
        feat_dim_low = (embed_dim // 4) * num_fused_heads
        feat_dim_high = (embed_dim // 2) * num_fused_heads
        self.physical_adapter = PhysicalAdapter(feat_dim_low, feat_dim_high, ctx_dim)

    def forward(self, organ_indices, tokenized_prompts, density_features: Optional[List[torch.Tensor]] = None):
        batch_size = len(organ_indices)
        
        # tokenized_prompts 直接是 input_ids (Tensor)，传给 Embedding 是安全的
        embedding = self.clip_token_embedding(tokenized_prompts).type(self.dtype)
        
        ctx_gen = self.ctx_general.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_spec = self.ctx_specific[organ_indices]
        ctx = torch.cat([ctx_gen, ctx_spec], dim=1) 

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
            gamma_high_expanded_gen = gamma_high.unsqueeze(1).expand(-1, n_gen_high, -1)  
            beta_high_expanded_gen = beta_high.unsqueeze(1).expand(-1, n_gen_high, -1)  
            ctx_gen_high_modulated = (1 + gamma_high_expanded_gen) * ctx_gen_high + beta_high_expanded_gen
            
            ctx_spec_mod = ctx_spec  
            gamma_high_expanded_spec = gamma_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  
            beta_high_expanded_spec = beta_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)  
            ctx_spec_modulated = (1 + gamma_high_expanded_spec) * ctx_spec_mod + beta_high_expanded_spec
            
            ctx_gen = torch.cat([ctx_gen_low_modulated, ctx_gen_high_modulated], dim=1)
            ctx_spec = ctx_spec_modulated
            ctx = torch.cat([ctx_gen, ctx_spec], dim=1)
        else:
            dummy_adapter = sum(p.sum() * 0.0 for p in self.physical_adapter.parameters())
            ctx = ctx + dummy_adapter

        prefix = embedding[:, :1, :] 
        suffix = embedding[:, 1 : 77 - self.total_ctx, :] 
        x = torch.cat([prefix, ctx, suffix], dim=1)

        x = x.permute(1, 0, 2)  
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.clip_ln_final(x).type(self.dtype)

        original_eos_idx = tokenized_prompts.argmax(dim=-1)
        eos_idx = torch.clamp(original_eos_idx + self.total_ctx, max=76)
        text_features = x[torch.arange(x.shape[0]), eos_idx] 
        
        if self.clip_text_projection is not None:
             text_features = text_features @ self.clip_text_projection

        return text_features


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


import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models


class TextSam(Sam):
    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        clip_model_name="ViT-B/16",
        text_dim=512,  # CONCH 默认输出 512
        embed_dim=256,
        num_organs=21,
        num_heads=8,
        sg_epsilon=0.05,
        sg_iters=3,
        use_pnurl: bool = True,
        use_coop: bool = True,
        use_ot: bool = True,
        use_asr: bool = True,
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)

        self.use_pnurl = use_pnurl
        self.use_coop = use_coop
        self.use_ot = use_ot
        self.use_asr = use_asr

        print(f"🚀 Initializing MP-SAM (Multi-granularity Prompt SAM) with CONCH...")
        
        # 1. 加载 CONCH (Freeze)
        hf_auth_token = os.environ.get("HF_TOKEN")
        if not hf_auth_token:
            print("⚠️ Warning: HF_TOKEN environment variable is not set. Model load may fail if not cached.")
            
        self.clip_model, _ = create_model_from_pretrained(
            'conch_ViT-B-16', 
            "hf_hub:MahmoodLab/conch", 
            hf_auth_token=hf_auth_token 
        )
        self.tokenizer = get_tokenizer()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 2. Dual-Prompt Learner (适配 CONCH)
        self.prompt_learner = DualPromptLearner(
            self.clip_model,
            num_organs=num_organs,
            n_ctx_gen=8,
            n_ctx_spec=8,
            embed_dim=embed_dim,
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = use_coop

        # 3. PNuRL (频域-语义解耦版)
        self.pnurl = PNuRL(
            embed_dim=embed_dim, 
            text_dim=512,
            num_classes_per_attr=[2, 3, 2, 3, 3], # [Color, Shape, Arrange, Size, Density]
            attr_loss_weight=1.0
        )
        for param in self.pnurl.parameters():
            param.requires_grad = use_pnurl

        # 4. Auto-Prompt Generator
        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        # 5. OT (保留结构开关)
        if self.use_ot:
            print("🚀 Switched to Density-Guided Optimal Transport (DG-OT) for pure spatial alignment!")
            self.ot = DensityGuidedOT(
                img_dim=embed_dim,
                epsilon=sg_epsilon,
                sinkhorn_iters=sg_iters,
            )
            for param in self.ot.parameters():
                param.requires_grad = True
        else:
            self.basic_hv_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, 2, kernel_size=1)
            )

        # 6. 纯视觉基线的高频特征提取器 (解耦 ASR)
        if self.use_asr:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.cnn_stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.cnn_stage1 = resnet.layer1 
            self.cnn_stage2 = resnet.layer2 
            self.global_asr_upsampler = GlobalASRUpsampler(embed_dim, use_asr=True, hm_channels=2)

        # 7. SAM 冻结策略
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True

    def forward(self, batched_input, multimask_output=False):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) 
        device = image_embeddings.device

        # 🔥 提取纯视觉高频特征 (给 ASR 使用，此时与语义完全无关，提供纯物理边缘)
        feat_half, feat_s1, feat_s2 = None, None, None
        if self.use_asr:
            with torch.autocast('cuda', enabled=True):
                x_cnn = input_images
                for i in range(3): 
                    x_cnn = self.cnn_stage0[i](x_cnn)
                feat_half = x_cnn 
                feat_s0 = self.cnn_stage0[3](feat_half) 
                feat_s1 = self.cnn_stage1(feat_s0)      
                feat_s2 = self.cnn_stage2(feat_s1)      

        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)

        # === 提取元数据 ===
        organ_indices =[]
        attribute_texts = []
        base_texts =[] 
        for x in batched_input:
            organ_indices.append(x.get("organ_id", 0)) 
            attribute_texts.append(x.get("attribute_text", "Cell nuclei")) 
            base_texts.append(x.get("text_prompt", "Cell nuclei"))
        organ_indices = torch.tensor(organ_indices).to(device)

        # === 提取并组装属性 Labels ===
        attribute_labels_list =[]
        for x in batched_input:
            attr_labels = x.get("attr_labels", None)
            if attr_labels is not None:
                attribute_labels_list.append(attr_labels)
            else:
                # 默认: Color, Shape, Arrange, Size, Density 的缺省索引
                attribute_labels_list.append(torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))

        attribute_labels = None
        if len(attribute_labels_list) > 0:
            attr_labels_batch = torch.stack(attribute_labels_list).to(device)
            # 分解成 5 个一维 tensor
            attribute_labels = [attr_labels_batch[:, i] for i in range(5)]

        # === 🟢🔴 PNuRL 特征解耦 (Spectral-Semantic Decoupling) ===
        if self.use_pnurl:
            if next(self.pnurl.parameters()).device != device:
                self.pnurl = self.pnurl.to(device)
            
            with torch.no_grad():
                attr_tokenized = self.tokenizer(
                    attribute_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
                )
                attr_tokens = attr_tokenized["input_ids"].to(device)
                attr_text_embed = self.clip_model.encode_text(attr_tokens)
                attr_text_embed = attr_text_embed / attr_text_embed.norm(dim=-1, keepdim=True)
                
            # 接收 PNuRL 吐出的解耦变量：txt_attr_feat (宏观/低频) 和 txt_mor_feat (微观/高频)
            refined_image_embeddings, pnurl_context, pnurl_loss, attr_logits, density_map, txt_attr_feat, txt_mor_feat = self.pnurl(
                image_features=image_embeddings,
                text_embed=attr_text_embed, 
                attribute_labels=attribute_labels,
                return_loss=True,
            )
        else:
            refined_image_embeddings = image_embeddings
            pnurl_loss = torch.tensor(0.0, device=device)
            attr_logits = {}
            density_map = None
            txt_attr_feat = None
            txt_mor_feat = None

        # === CoOp 可学习提示 ===
        pos_tokenized = self.tokenizer(
            base_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        pos_tokens = pos_tokenized["input_ids"].to(device)
        
        neg_tokenized = self.tokenizer(
            ["Background"] * len(base_texts), padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        neg_tokens = neg_tokenized["input_ids"].to(device)
        
        if self.use_coop:
            if next(self.prompt_learner.parameters()).device != device:
                self.prompt_learner = self.prompt_learner.to(device)
            pos_feats = self.prompt_learner(organ_indices, pos_tokens, density_features=None)
            neg_feats = self.prompt_learner(organ_indices, neg_tokens, density_features=None)
        else:
            with torch.no_grad():
                pos_feats = self.clip_model.encode_text(pos_tokens).float()
                neg_feats = self.clip_model.encode_text(neg_tokens).float()

        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        text_features = torch.stack([pos_feats, neg_feats], dim=1).float()

        # === OT/Point Generator 模块 ===
        B, C, H, W = refined_image_embeddings.shape
        ot_density = density_map if density_map is not None else torch.ones(B, 1, H, W, device=device) / (H * W)

        if self.use_ot:
            if next(self.ot.parameters()).device != device:
                self.ot = self.ot.to(device)
            fused_image_embeddings, heatmap_logits_coarse, hv_logits_coarse = self.ot(
                img_feat=refined_image_embeddings,
                density_map=ot_density,
            )
        else:
            fused_image_embeddings = refined_image_embeddings
            heatmap_logits_coarse = self.prompt_generator(refined_image_embeddings, text_features)
            hv_logits_coarse = self.basic_hv_head(refined_image_embeddings)

        # 🔥 全局 ASR 上采样 (输出 Heatmap 和 HV Map)
        if self.use_asr:
            hv_logits_out, heatmap_logits_out = self.global_asr_upsampler(
                fused_image_embeddings, 
                hv_logits_coarse, 
                heatmap_logits_coarse, 
                feat_s2, feat_s1, feat_half 
            )
        else:
            hv_logits_out = hv_logits_coarse
            heatmap_logits_out = heatmap_logits_coarse

        # 🚀 亮点：利用 Size 属性自适应调节点生成间距！
        size_logits = attr_logits.get('size', None)
        if size_logits is not None and size_logits.numel() > 0:
            pred_size_class = torch.argmax(size_logits, dim=1)
            size_threshold_map = torch.tensor([10.0, 15.0, 20.0], device=device)
            adaptive_thresh = size_threshold_map[pred_size_class]
        else:
            adaptive_thresh = torch.tensor(15.0, device=device).expand(image_embeddings.shape[0])

        prompts_list = self.prompt_generator.generate_adaptive_prompts(
            heatmap_logits_coarse,
            threshold=0.3,
            k_neighbors=3,
            dense_dist_thresh=adaptive_thresh,
            pred_density=density_map if self.use_pnurl else None,
        )
        
        feat_size = image_embeddings.shape[-1] 
        input_size = self.image_encoder.img_size 
        scale_factor = input_size / feat_size

        # === SAM Mask Decoder Loop ===
        outputs =[]
        for i in range(len(batched_input)):
            prompt_data = prompts_list[i]
            target_h, target_w = batched_input[i]["original_size"]
            input_h, input_w = batched_input[i]["image"].shape[-2:] 
            
            if self.use_asr:
                hv_out_i = hv_logits_out[i:i+1, :, :input_h, :input_w]
                hm_out_i = heatmap_logits_out[i:i+1, :, :input_h, :input_w]
                if (input_h, input_w) != (target_h, target_w):
                    hv_out_i = F.interpolate(hv_out_i, size=(target_h, target_w), mode='nearest')
                    hm_out_i = F.interpolate(hm_out_i, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                hv_out_i = hv_logits_out[i].unsqueeze(0) if hv_logits_out is not None else None
                hm_out_i = heatmap_logits_out[i].unsqueeze(0)

            density_map_i = None
            if density_map is not None:
                density_map_raw = density_map[i]
                if density_map_raw.shape[-2:] != (target_h, target_w):
                    density_map_i = F.interpolate(
                        density_map_raw.unsqueeze(0), 
                        size=(target_h, target_w), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                else:
                    density_map_i = density_map_raw

            if not prompt_data["has_points"]:
                dummy = fused_image_embeddings[i].sum() * 0.0
                if density_map_i is not None:
                    density_map_i = density_map_i + dummy
                outputs.append({
                    "masks": (torch.zeros((1, 1, target_h, target_w), device=device, dtype=torch.float32) - 100.0) + dummy,
                    "iou_predictions": torch.zeros((1, 1), device=device) + dummy,
                    "low_res_logits": (torch.zeros((1, 1, 256, 256), device=device) - 100.0) + dummy,
                    "heatmap_logits": hm_out_i + dummy, 
                    "hv_logits": hv_out_i + dummy if hv_out_i is not None else None, 
                    "attr_logits": attr_logits,
                    "density_map": density_map_i,
                    "pnurl_loss": pnurl_loss,
                    "organ_cls_loss": getattr(self, 'organ_cls_loss_cache', torch.tensor(0.0, device=device)) 
                })
                continue

            point_coords = prompt_data["point_coords"]
            point_labels = prompt_data["point_labels"]
            point_coords = (point_coords * scale_factor) + (scale_factor * 0.5)
            
            if self.training:
                MAX_POINTS = 512 
                if point_coords.shape[0] > MAX_POINTS:
                    indices = torch.randperm(point_coords.shape[0], device=device)[:MAX_POINTS]
                    point_coords = point_coords[indices]
                    point_labels = point_labels[indices]

            num_cells = point_coords.shape[0]
            chunk_size = 16 
            chunk_masks =[]
            chunk_ious =[]
            
            # 🔥 提取当前图片的宏观与微观 Prompt 特征
            curr_attr_prompt = txt_attr_feat[i:i+1] if txt_attr_feat is not None else None
            curr_morph_feat = txt_mor_feat[i:i+1] if txt_mor_feat is not None else None
            
            for start_idx in range(0, num_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, num_cells)
                sub_coords = point_coords[start_idx:end_idx] 
                sub_labels = point_labels[start_idx:end_idx] 
                current_batch = sub_coords.shape[0]

                sub_img_embed = fused_image_embeddings[i].unsqueeze(0).expand(current_batch, -1, -1, -1)
                
                # ASR 需要的高频特征随之切分扩维
                sub_cnn_s1 = feat_s1[i].unsqueeze(0).expand(current_batch, -1, -1, -1).contiguous() if self.use_asr else None
                sub_cnn_s2 = feat_s2[i].unsqueeze(0).expand(current_batch, -1, -1, -1).contiguous() if self.use_asr else None

                sparse, dense = self.prompt_encoder(points=(sub_coords, sub_labels), boxes=None, masks=None)

                # 🚨 安全扩维防御：判断文本Prompt是 2D 还是 3D (防止 Runtime Error)
                sub_attr_prompt, sub_morph_feat = None, None
                if curr_attr_prompt is not None:
                    if curr_attr_prompt.dim() == 2:   # [1, C]
                        sub_attr_prompt = curr_attr_prompt.expand(current_batch, -1).contiguous()
                    elif curr_attr_prompt.dim() == 3: # [1, Seq, C]
                        sub_attr_prompt = curr_attr_prompt.expand(current_batch, -1, -1).contiguous()
                
                if curr_morph_feat is not None:
                    if curr_morph_feat.dim() == 2:
                        sub_morph_feat = curr_morph_feat.expand(current_batch, -1).contiguous()
                    elif curr_morph_feat.dim() == 3:
                        sub_morph_feat = curr_morph_feat.expand(current_batch, -1, -1).contiguous()

                # 🔥 关键投送：将解耦的宏观低频与微观高频的Prompt送入 MaskDecoder
                sub_mask, sub_iou = self.mask_decoder(
                    image_embeddings=sub_img_embed,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=multimask_output,
                    cnn_feat_s1=sub_cnn_s1,  
                    cnn_feat_s2=sub_cnn_s2, 
                    attr_prompt=sub_attr_prompt, # 🟢 低频特征流 (Color, Arrange, Density)
                    morph_feat=sub_morph_feat,   # 🔴 高频特征流 (Shape, Size)
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
            
            outputs.append({
                "masks": mask_post,
                "iou_predictions": merged_iou,
                "low_res_logits": merged_logits,
                "heatmap_logits": hm_out_i,   
                "hv_logits": hv_out_i,        
                "attr_logits": attr_logits,
                "density_map": density_map_i,
                "pnurl_loss": pnurl_loss,
                "organ_cls_loss": getattr(self, 'organ_cls_loss_cache', torch.tensor(0.0, device=device)) 
            })
        if self.training and len(outputs) > 0:
            dummy = torch.tensor(0.0, device=device)
            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
            # 强行注入到 heatmap_logits 中 (它必定会参与 loss_h 的计算)
            outputs[0]["heatmap_logits"] = outputs[0]["heatmap_logits"] + dummy

        return outputs