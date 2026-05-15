import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from dotenv import load_dotenv

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL

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
    全局高分辨率特征上采样器。

    输入:
        sam_feat: SAM image embedding, usually [B, 256, 32/64, 32/64]
        hv_logits: coarse HV logits
        hm_logits: coarse heatmap logits
        feat_s2 / feat_s1 / feat_half: CNN shallow features

    作用:
        利用纯 CNN 高频特征补充边界细节。
        这里不再依赖 OT，也不再将 OT 作为 heatmap/HV 的上游。
    """

    def __init__(self, embed_dim=256, hm_channels=2, use_asr=True):
        super().__init__()
        self.use_asr = use_asr

        self.init_conv = nn.Conv2d(embed_dim + 2 + hm_channels, 256, kernel_size=3, padding=1)

        # 上采样 1: 64x64 -> 128x128, 拼接 ResNet Stage2: 512 channels
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + (512 if use_asr else 0), 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 上采样 2: 128x128 -> 256x256, 拼接 ResNet Stage1: 256 channels
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + (256 if use_asr else 0), 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 上采样 3: 256x256 -> 512x512, 拼接 ResNet conv stem: 64 channels
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 + (64 if use_asr else 0), 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 上采样 4: 512x512 -> 1024x1024
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
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
            nn.Linear(ctx_dim, ctx_dim * 2),
        )

        self.adapter_high = nn.Sequential(
            nn.Linear(feat_dim_high, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim * 2),
        )

        nn.init.zeros_(self.adapter_low[-1].weight)
        nn.init.zeros_(self.adapter_low[-1].bias)
        nn.init.zeros_(self.adapter_high[-1].weight)
        nn.init.zeros_(self.adapter_high[-1].bias)

    def forward(
        self,
        feat_low: torch.Tensor,
        feat_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_params = self.adapter_low(feat_low)
        gamma_low, beta_low = torch.chunk(low_params, 2, dim=1)

        high_params = self.adapter_high(feat_high)
        gamma_high, beta_high = torch.chunk(high_params, 2, dim=1)

        return gamma_low, beta_low, gamma_high, beta_high


class DualPromptLearner(nn.Module):
    """
    CoOp-style prompt learner for CONCH text encoder.

    关键修复:
        1. 只让 ctx_general / ctx_specific / physical_adapter 参与训练。
        2. CONCH 的 token_embedding / transformer / ln_final / text_projection
           不注册为本模块参数，避免 train.py 中 prompt_learner.parameters()
           把整个 CONCH 文本编码器误解冻。
    """

    def __init__(self, clip_model, num_organs=21, n_ctx_gen=8, n_ctx_spec=8, embed_dim=256):
        super().__init__()

        if hasattr(clip_model, "text"):
            text_encoder = clip_model.text
        else:
            text_encoder = clip_model

        ctx_dim = text_encoder.ln_final.weight.shape[0]
        dtype = next(clip_model.parameters()).dtype
        self.dtype = dtype

        print(f"🧠 Init DualLearner: General({n_ctx_gen}) + Specific({n_ctx_spec}x{num_organs})")

        self.ctx_general = nn.Parameter(torch.empty(n_ctx_gen, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_general, std=0.02)

        self.ctx_specific = nn.Parameter(torch.empty(num_organs, n_ctx_spec, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_specific, std=0.02)

        # 重点：使用 object.__setattr__，避免这些 CONCH 子模块被注册进 prompt_learner.parameters()
        object.__setattr__(self, "clip_token_embedding", text_encoder.token_embedding)
        object.__setattr__(self, "clip_transformer", text_encoder.transformer)
        object.__setattr__(self, "clip_ln_final", text_encoder.ln_final)
        object.__setattr__(self, "clip_text_projection", text_encoder.text_projection)

        for module in [text_encoder.token_embedding, text_encoder.transformer, text_encoder.ln_final]:
            for p in module.parameters():
                p.requires_grad = False

        if isinstance(text_encoder.text_projection, nn.Parameter):
            text_encoder.text_projection.requires_grad = False

        self.n_ctx_gen = n_ctx_gen
        self.n_ctx_spec = n_ctx_spec
        self.total_ctx = n_ctx_gen + n_ctx_spec
        self.ctx_dim = ctx_dim

        num_fused_heads = 3
        feat_dim_low = (embed_dim // 4) * num_fused_heads
        feat_dim_high = (embed_dim // 2) * num_fused_heads

        self.physical_adapter = PhysicalAdapter(feat_dim_low, feat_dim_high, ctx_dim)

    def forward(self, organ_indices, tokenized_prompts, density_features: Optional[List[torch.Tensor]] = None):
        if not torch.is_tensor(organ_indices):
            organ_indices = torch.as_tensor(organ_indices, dtype=torch.long, device=tokenized_prompts.device)
        else:
            organ_indices = organ_indices.to(device=tokenized_prompts.device, dtype=torch.long)

        batch_size = int(organ_indices.numel())

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

            gamma_high_expanded_spec = gamma_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)
            beta_high_expanded_spec = beta_high.unsqueeze(1).expand(-1, self.n_ctx_spec, -1)
            ctx_spec_modulated = (1 + gamma_high_expanded_spec) * ctx_spec + beta_high_expanded_spec

            ctx_gen = torch.cat([ctx_gen_low_modulated, ctx_gen_high_modulated], dim=1)
            ctx_spec = ctx_spec_modulated
            ctx = torch.cat([ctx_gen, ctx_spec], dim=1)
        else:
            # 保证 physical_adapter 在 DDP 下不会完全 unused
            dummy_adapter = sum(p.sum() * 0.0 for p in self.physical_adapter.parameters())
            ctx = ctx + dummy_adapter

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1: 77 - self.total_ctx, :]
        x = torch.cat([prefix, ctx, suffix], dim=1)

        x = x.permute(1, 0, 2)
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_ln_final(x).type(self.dtype)

        original_eos_idx = tokenized_prompts.argmax(dim=-1)
        eos_idx = torch.clamp(original_eos_idx + self.total_ctx, max=76)
        text_features = x[torch.arange(x.shape[0], device=x.device), eos_idx]

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
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w

        x = F.pad(x, (0, padw, 0, padh))

        return x


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
        use_ot: bool = False,
        use_asr: bool = True,
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)

        self.use_pnurl = use_pnurl
        self.use_coop = use_coop

        # OT 已从当前主线移除。保留 use_ot 入参只是为了兼容旧 train.py / 旧命令。
        if use_ot:
            print("⚠️ use_ot=True was passed, but OT is disabled in this version of TextSam.")
        self.use_ot = False

        self.use_asr = use_asr

        print("🚀 Initializing MP-SAM / FreqPath-SAM with CONCH...")

        # 1. Load CONCH and freeze it
        hf_auth_token = os.environ.get("HF_TOKEN")
        if not hf_auth_token:
            print("⚠️ Warning: HF_TOKEN environment variable is not set. Model load may fail if not cached.")

        self.clip_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=hf_auth_token,
        )
        self.tokenizer = get_tokenizer()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 2. CoOp / Dual Prompt Learner
        self.prompt_learner = DualPromptLearner(
            self.clip_model,
            num_organs=num_organs,
            n_ctx_gen=8,
            n_ctx_spec=8,
            embed_dim=embed_dim,
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = use_coop

        # 3. PNuRL
        self.pnurl = PNuRL(
            embed_dim=embed_dim,
            text_dim=512,
            num_classes_per_attr=[2, 3, 2, 3, 3],
            attr_loss_weight=1.0,
        )

        # 关键修复：把 PNuRL residual gate 注册到 pnurl 内部。
        # 这样 train.py 中 add_to_params(raw_model.pnurl, args.lr) 会自动包含这个 gate。
        # sigmoid(-6) ≈ 0.002，第二阶段起步时几乎不扰动 vision best。
        self.pnurl.residual_gate = nn.Parameter(torch.tensor(-6.0))

        for param in self.pnurl.parameters():
            param.requires_grad = use_pnurl

        # 4. Auto Prompt Generator
        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        # 5. HV head: OT 移除后，始终使用普通 HV head
        self.basic_hv_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 2, kernel_size=1),
        )

        # 6. CNN high-frequency branch for ASR
        if self.use_asr:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            self.cnn_stage0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
            )
            self.cnn_stage1 = resnet.layer1
            self.cnn_stage2 = resnet.layer2

            self.global_asr_upsampler = GlobalASRUpsampler(
                embed_dim=embed_dim,
                use_asr=True,
                hm_channels=2,
            )

        # 7. SAM freeze strategy
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True

    def _tokenize_to_input_ids(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        if isinstance(tokenized, dict):
            tokens = tokenized["input_ids"]
        else:
            tokens = tokenized

        return tokens.to(device)

    @staticmethod
    def _get_int_value(value, default: int = 20) -> int:
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return int(value.detach().cpu().view(-1)[0].item())
        return int(value)

    def forward(self, batched_input, multimask_output=False):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        device = image_embeddings.device

        # 1. CNN high-frequency features for ASR
        feat_half, feat_s1, feat_s2 = None, None, None

        if self.use_asr:
            with torch.autocast("cuda", enabled=input_images.is_cuda):
                x_cnn = input_images

                # conv1 + bn1 + relu
                for i in range(3):
                    x_cnn = self.cnn_stage0[i](x_cnn)

                feat_half = x_cnn
                feat_s0 = self.cnn_stage0[3](feat_half)
                feat_s1 = self.cnn_stage1(feat_s0)
                feat_s2 = self.cnn_stage2(feat_s1)

        # 2. Ensure CONCH is on the same device
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)

        # 3. Metadata
        organ_indices_list = []
        attribute_texts = []
        base_texts = []

        for x in batched_input:
            organ_indices_list.append(self._get_int_value(x.get("organ_id", 20), default=20))
            attribute_texts.append(x.get("attribute_text", "Cell nuclei"))
            base_texts.append(x.get("text_prompt", "Cell nuclei"))

        organ_indices = torch.tensor(organ_indices_list, dtype=torch.long, device=device)

        # 4. Attribute labels
        attribute_labels_list = []
        for x in batched_input:
            attr_labels = x.get("attr_labels", None)

            if attr_labels is not None:
                if not torch.is_tensor(attr_labels):
                    attr_labels = torch.tensor(attr_labels, dtype=torch.long)
                attribute_labels_list.append(attr_labels.to(device=device, dtype=torch.long))
            else:
                attribute_labels_list.append(
                    torch.tensor([0, 0, 0, 1, 1], dtype=torch.long, device=device)
                )

        attribute_labels = None
        if len(attribute_labels_list) > 0:
            attr_labels_batch = torch.stack(attribute_labels_list, dim=0).to(device=device, dtype=torch.long)
            attribute_labels = [attr_labels_batch[:, i] for i in range(5)]

        # 5. PNuRL: pathology semantic disentanglement with residual gate
        if self.use_pnurl:
            if next(self.pnurl.parameters()).device != device:
                self.pnurl = self.pnurl.to(device)

            with torch.no_grad():
                attr_tokens = self._tokenize_to_input_ids(attribute_texts, device)
                attr_text_embed = self.clip_model.encode_text(attr_tokens).float()
                attr_text_embed = F.normalize(attr_text_embed, dim=-1, eps=1e-6)

            (
                pnurl_refined_embeddings,
                pnurl_context,
                pnurl_loss,
                attr_logits,
                density_map,
                txt_attr_feat,
                txt_mor_feat,
            ) = self.pnurl(
                image_features=image_embeddings,
                text_embed=attr_text_embed,
                attribute_labels=attribute_labels,
                return_loss=True,
            )

            # 关键修复：PNuRL 只通过 residual gate 渐进影响视觉底盘
            gate = torch.sigmoid(self.pnurl.residual_gate).to(device=device, dtype=image_embeddings.dtype)
            refined_image_embeddings = image_embeddings + gate * (pnurl_refined_embeddings - image_embeddings)

            if txt_attr_feat is not None:
                txt_attr_feat = gate * txt_attr_feat
            if txt_mor_feat is not None:
                txt_mor_feat = gate * txt_mor_feat

        else:
            refined_image_embeddings = image_embeddings
            pnurl_loss = torch.tensor(0.0, device=device)
            attr_logits = {}
            density_map = None
            txt_attr_feat = None
            txt_mor_feat = None

        # 6. Text features for positive / negative prompt generation
        pos_tokens = self._tokenize_to_input_ids(base_texts, device)
        neg_tokens = self._tokenize_to_input_ids(["Background"] * len(base_texts), device)

        if self.use_coop:
            if next(self.prompt_learner.parameters()).device != device:
                self.prompt_learner = self.prompt_learner.to(device)

            pos_feats = self.prompt_learner(organ_indices, pos_tokens, density_features=None)
            neg_feats = self.prompt_learner(organ_indices, neg_tokens, density_features=None)
        else:
            with torch.no_grad():
                pos_feats = self.clip_model.encode_text(pos_tokens).float()
                neg_feats = self.clip_model.encode_text(neg_tokens).float()

        # 关键修复：无论 CoOp 是否启用，都统一归一化
        pos_feats = F.normalize(pos_feats.float(), dim=-1, eps=1e-6)
        neg_feats = F.normalize(neg_feats.float(), dim=-1, eps=1e-6)

        text_features = torch.stack([pos_feats, neg_feats], dim=1).float()

        # 7. Point Generator and HV head
        fused_image_embeddings = refined_image_embeddings
        heatmap_logits_coarse = self.prompt_generator(refined_image_embeddings, text_features)
        hv_logits_coarse = self.basic_hv_head(refined_image_embeddings)

        # 8. Global ASR upsampling
        if self.use_asr:
            hv_logits_out, heatmap_logits_out = self.global_asr_upsampler(
                fused_image_embeddings,
                hv_logits_coarse,
                heatmap_logits_coarse,
                feat_s2,
                feat_s1,
                feat_half,
            )
        else:
            hv_logits_out = hv_logits_coarse
            heatmap_logits_out = heatmap_logits_coarse

        # 9. Adaptive point distance according to size attribute
        size_logits = attr_logits.get("size", None)
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

        # 10. SAM Mask Decoder loop
        outputs = []

        for i in range(len(batched_input)):
            prompt_data = prompts_list[i]

            target_h, target_w = batched_input[i]["original_size"]
            input_h, input_w = batched_input[i]["image"].shape[-2:]

            if self.use_asr:
                hv_out_i = hv_logits_out[i:i + 1, :, :input_h, :input_w]
                hm_out_i = heatmap_logits_out[i:i + 1, :, :input_h, :input_w]

                if (input_h, input_w) != (target_h, target_w):
                    hv_out_i = F.interpolate(hv_out_i, size=(target_h, target_w), mode="nearest")
                    hm_out_i = F.interpolate(hm_out_i, size=(target_h, target_w), mode="bilinear", align_corners=False)
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
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    density_map_i = density_map_raw

            if not prompt_data["has_points"]:
                dummy = fused_image_embeddings[i].sum() * 0.0

                if density_map_i is not None:
                    density_map_i = density_map_i + dummy

                outputs.append(
                    {
                        "masks": (
                            torch.zeros((1, 1, target_h, target_w), device=device, dtype=torch.float32) - 100.0
                        ) + dummy,
                        "iou_predictions": torch.zeros((1, 1), device=device) + dummy,
                        "low_res_logits": (
                            torch.zeros((1, 1, 256, 256), device=device, dtype=torch.float32) - 100.0
                        ) + dummy,
                        "heatmap_logits": hm_out_i + dummy,
                        "hv_logits": hv_out_i + dummy if hv_out_i is not None else None,
                        "attr_logits": attr_logits,
                        "density_map": density_map_i,
                        "pnurl_loss": pnurl_loss,
                        "organ_cls_loss": getattr(self, "organ_cls_loss_cache", torch.tensor(0.0, device=device)),
                    }
                )
                continue

            point_coords = prompt_data["point_coords"]
            point_labels = prompt_data["point_labels"]

            point_coords = (point_coords * scale_factor) + (scale_factor * 0.5)

            if self.training:
                max_points = 512
                if point_coords.shape[0] > max_points:
                    indices = torch.randperm(point_coords.shape[0], device=device)[:max_points]
                    point_coords = point_coords[indices]
                    point_labels = point_labels[indices]

            num_cells = point_coords.shape[0]
            chunk_size = 16

            chunk_masks = []
            chunk_ious = []

            curr_attr_prompt = txt_attr_feat[i:i + 1] if txt_attr_feat is not None else None
            curr_morph_feat = txt_mor_feat[i:i + 1] if txt_mor_feat is not None else None

            for start_idx in range(0, num_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, num_cells)

                sub_coords = point_coords[start_idx:end_idx]
                sub_labels = point_labels[start_idx:end_idx]
                current_batch = sub_coords.shape[0]

                sub_img_embed = fused_image_embeddings[i].unsqueeze(0).expand(current_batch, -1, -1, -1)

                sub_cnn_s1 = (
                    feat_s1[i].unsqueeze(0).expand(current_batch, -1, -1, -1).contiguous()
                    if self.use_asr and feat_s1 is not None
                    else None
                )

                sub_cnn_s2 = (
                    feat_s2[i].unsqueeze(0).expand(current_batch, -1, -1, -1).contiguous()
                    if self.use_asr and feat_s2 is not None
                    else None
                )

                sparse, dense = self.prompt_encoder(
                    points=(sub_coords, sub_labels),
                    boxes=None,
                    masks=None,
                )

                sub_attr_prompt = None
                sub_morph_feat = None

                if curr_attr_prompt is not None:
                    if curr_attr_prompt.dim() == 2:
                        sub_attr_prompt = curr_attr_prompt.expand(current_batch, -1).contiguous()
                    elif curr_attr_prompt.dim() == 3:
                        sub_attr_prompt = curr_attr_prompt.expand(current_batch, -1, -1).contiguous()

                if curr_morph_feat is not None:
                    if curr_morph_feat.dim() == 2:
                        sub_morph_feat = curr_morph_feat.expand(current_batch, -1).contiguous()
                    elif curr_morph_feat.dim() == 3:
                        sub_morph_feat = curr_morph_feat.expand(current_batch, -1, -1).contiguous()

                sub_mask, sub_iou = self.mask_decoder(
                    image_embeddings=sub_img_embed,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=multimask_output,
                    cnn_feat_s1=sub_cnn_s1,
                    cnn_feat_s2=sub_cnn_s2,
                    attr_prompt=sub_attr_prompt,
                    morph_feat=sub_morph_feat,
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

            outputs.append(
                {
                    "masks": mask_post,
                    "iou_predictions": merged_iou,
                    "low_res_logits": merged_logits,
                    "heatmap_logits": hm_out_i,
                    "hv_logits": hv_out_i,
                    "attr_logits": attr_logits,
                    "density_map": density_map_i,
                    "pnurl_loss": pnurl_loss,
                    "organ_cls_loss": getattr(self, "organ_cls_loss_cache", torch.tensor(0.0, device=device)),
                }
            )

        # DDP safety: ensure all trainable parameters have a zero-gradient path.
        if self.training and len(outputs) > 0:
            dummy = torch.tensor(0.0, device=device)

            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0

            outputs[0]["heatmap_logits"] = outputs[0]["heatmap_logits"] + dummy

        return outputs