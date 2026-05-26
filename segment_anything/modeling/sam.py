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
    全局高分辨率 HV / heatmap 上采样器。

    这里显式拆成两种模式：

    1. legacy:
       用于回归旧版纯视觉 ASR。
       - heatmap 输出 1 通道
       - 只做 SAM coarse HV/heatmap + ResNet 细节融合
       - 不引入语义高低频调制
       - 默认输出到 512 级别，不额外 up4 到 1024

    2. freqpath:
       用于后续论文主线。
       - 保留当前 FreqPath/Frequency-aware 版本
       - heatmap 可为 2 通道
       - 保留 up4 到更高分辨率
       - 与后续低频语义 / 高频形态约束配合
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hm_channels: int = 2,
        use_asr: bool = True,
        asr_variant: str = "legacy",
    ):
        super().__init__()

        asr_variant = str(asr_variant).lower().strip()
        if asr_variant not in ("legacy", "freqpath"):
            raise ValueError(
                f"GlobalASRUpsampler asr_variant must be 'legacy' or 'freqpath', got {asr_variant}"
            )

        self.use_asr = use_asr
        self.asr_variant = asr_variant
        self.hm_channels = 1 if asr_variant == "legacy" else int(hm_channels)

        if self.asr_variant == "legacy":
            self.init_conv = nn.Conv2d(embed_dim + 2 + self.hm_channels, 256, kernel_size=3, padding=1)

            self.up1 = nn.ConvTranspose2d(
                256 + (512 if use_asr else 0),
                128,
                kernel_size=2,
                stride=2,
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(128 + (256 if use_asr else 0), 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv2 = nn.Sequential(
                nn.Conv2d(64 + (64 if use_asr else 0), 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.hv_out = nn.Conv2d(32, 2, kernel_size=1)
            self.hm_out = nn.Conv2d(32, self.hm_channels, kernel_size=1)

        else:
            self.init_conv = nn.Conv2d(embed_dim + 2 + self.hm_channels, 256, kernel_size=3, padding=1)

            self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv1 = nn.Sequential(
                nn.Conv2d(128 + (512 if use_asr else 0), 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv2 = nn.Sequential(
                nn.Conv2d(64 + (256 if use_asr else 0), 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv3 = nn.Sequential(
                nn.Conv2d(32 + (64 if use_asr else 0), 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.conv4 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )

            self.hv_out = nn.Conv2d(16, 2, kernel_size=1)
            self.hm_out = nn.Conv2d(16, self.hm_channels, kernel_size=1)

    @staticmethod
    def _resize_like(feat: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        if feat is None:
            return None
        if feat.shape[-2:] != ref.shape[-2:]:
            feat = F.interpolate(
                feat,
                size=ref.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return feat

    @staticmethod
    def _match_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
        current = x.shape[1]
        if current == channels:
            return x
        if current > channels:
            return x[:, :channels, :, :]

        pad = torch.zeros(
            x.shape[0],
            channels - current,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=1)

    def forward(
        self,
        sam_feat: torch.Tensor,
        hv_logits: torch.Tensor,
        hm_logits: torch.Tensor,
        feat_s2: Optional[torch.Tensor] = None,
        feat_s1: Optional[torch.Tensor] = None,
        feat_half: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hm_logits = self._match_channels(hm_logits, self.hm_channels)

        if hv_logits.shape[-2:] != sam_feat.shape[-2:]:
            hv_logits = F.interpolate(
                hv_logits,
                size=sam_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if hm_logits.shape[-2:] != sam_feat.shape[-2:]:
            hm_logits = F.interpolate(
                hm_logits,
                size=sam_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([sam_feat, hv_logits, hm_logits], dim=1)
        x = self.init_conv(x)

        if self.asr_variant == "legacy":
            if self.use_asr and feat_s2 is not None:
                feat_s2 = self._resize_like(feat_s2, x)
                x = torch.cat([x, feat_s2], dim=1)

            x = self.up1(x)

            if self.use_asr and feat_s1 is not None:
                feat_s1 = self._resize_like(feat_s1, x)
                x = torch.cat([x, feat_s1], dim=1)

            x = self.conv1(x)
            x = self.up2(x)

            if self.use_asr and feat_half is not None:
                feat_half = self._resize_like(feat_half, x)
                x = torch.cat([x, feat_half], dim=1)

            x = self.conv2(x)
            x = self.up3(x)
            x = self.conv3(x)

            return self.hv_out(x), self.hm_out(x)

        x = self.up1(x)

        if self.use_asr and feat_s2 is not None:
            feat_s2 = self._resize_like(feat_s2, x)
            x = torch.cat([x, feat_s2], dim=1)

        x = self.conv1(x)
        x = self.up2(x)

        if self.use_asr and feat_s1 is not None:
            feat_s1 = self._resize_like(feat_s1, x)
            x = torch.cat([x, feat_s1], dim=1)

        x = self.conv2(x)
        x = self.up3(x)

        if self.use_asr and feat_half is not None:
            feat_half = self._resize_like(feat_half, x)
            x = torch.cat([x, feat_half], dim=1)

        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)

        return self.hv_out(x), self.hm_out(x)


class SemanticChannelGate(nn.Module):
    """
    Pathology-aware channel recalibration gate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        reduction: int = 16,
        init_bias: float = -4.0,
        max_gate: float = 0.10,
    ):
        super().__init__()
        hidden_dim = max(embed_dim // reduction, 16)
        self.max_gate = float(max_gate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_bias)

    def forward(self, semantic_delta: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(semantic_delta))
        return gate * self.max_gate


class PhysicalAdapter(nn.Module):
    """
    将低频 / 高频视觉特征转换成 CoOp context modulation 参数。
    """

    def __init__(self, feat_dim_low: int, feat_dim_high: int, ctx_dim: int):
        super().__init__()

        self.feat_dim_low = feat_dim_low
        self.feat_dim_high = feat_dim_high

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.adapter_low = nn.Sequential(
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

    def _to_vector(self, feat: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
        if feat.dim() == 4:
            feat = self.pool(feat).flatten(1)
        elif feat.dim() == 2:
            feat = feat
        else:
            raise ValueError(
                f"PhysicalAdapter expects {name} to be 2D [B, C] or 4D [B, C, H, W], "
                f"but got shape={tuple(feat.shape)}"
            )

        if feat.shape[-1] != expected_dim:
            raise ValueError(
                f"PhysicalAdapter {name} channel mismatch: expected {expected_dim}, "
                f"got {feat.shape[-1]} from shape={tuple(feat.shape)}"
            )

        return feat

    def forward(
        self,
        feat_low: torch.Tensor,
        feat_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_low_vec = self._to_vector(feat_low, self.feat_dim_low, "feat_low")
        feat_high_vec = self._to_vector(feat_high, self.feat_dim_high, "feat_high")

        low_params = self.adapter_low(feat_low_vec)
        gamma_low, beta_low = torch.chunk(low_params, 2, dim=1)

        high_params = self.adapter_high(feat_high_vec)
        gamma_high, beta_high = torch.chunk(high_params, 2, dim=1)

        return gamma_low, beta_low, gamma_high, beta_high


class DualPromptLearner(nn.Module):
    """
    CoOp-style prompt learner for CONCH text encoder。
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
        asr_variant: str = "legacy",
        asr_regression: Optional[bool] = None,
        max_semantic_gate: float = 0.10,
        max_delta_ratio: float = 0.10,
        init_delta_ratio: float = 0.02,
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)

        asr_variant = str(asr_variant).lower().strip()
        if asr_variant not in ("legacy", "freqpath"):
            raise ValueError(f"asr_variant must be 'legacy' or 'freqpath', got {asr_variant}")

        if asr_regression is None:
            asr_regression = False

        self.asr_variant = asr_variant
        self.asr_regression = bool(asr_regression)

        if self.asr_regression:
            if use_pnurl or use_coop or use_ot:
                print(
                    "🔒 ASR regression mode enabled: forcing PNuRL=False, CoOp=False, OT=False, "
                    "and using base prompt 'Cell nuclei'."
                )
            use_pnurl = False
            use_coop = False
            use_ot = False

        self.use_pnurl = use_pnurl
        self.use_coop = use_coop

        if use_ot:
            print("⚠️ use_ot=True was passed, but OT is disabled in this version of TextSam.")
        self.use_ot = False

        self.use_asr = use_asr
        self.max_semantic_gate = float(max_semantic_gate)
        self.max_delta_ratio = float(max_delta_ratio)
        self.init_delta_ratio = float(init_delta_ratio)

        print(
            f"🚀 Initializing MP-SAM / FreqPath-SAM with CONCH | "
            f"ASR={use_asr}, asr_variant={self.asr_variant}, asr_regression={self.asr_regression}"
        )

        if hasattr(self.mask_decoder, "asr_variant"):
            if getattr(self.mask_decoder, "asr_variant") != self.asr_variant:
                print(
                    f"⚠️ mask_decoder.asr_variant={getattr(self.mask_decoder, 'asr_variant')} "
                    f"but TextSam.asr_variant={self.asr_variant}. "
                    "Please update build_sam.py to pass the same asr_variant into MaskDecoder."
                )

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

        self.prompt_learner = DualPromptLearner(
            self.clip_model,
            num_organs=num_organs,
            n_ctx_gen=8,
            n_ctx_spec=8,
            embed_dim=embed_dim,
        )
        for param in self.prompt_learner.parameters():
            param.requires_grad = use_coop

        self.pnurl = PNuRL(
            embed_dim=embed_dim,
            text_dim=512,
            num_classes_per_attr=[2, 3, 2, 3, 3],
            attr_loss_weight=1.0,
            max_delta_ratio=max_delta_ratio,
            init_delta_ratio=init_delta_ratio,
        )

        self.pnurl.semantic_channel_gate = SemanticChannelGate(
            embed_dim=embed_dim,
            reduction=16,
            init_bias=-4.0,
            max_gate=max_semantic_gate,
        )

        for param in self.pnurl.parameters():
            param.requires_grad = use_pnurl

        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_heads=num_heads,
        )

        self.basic_hv_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 2, kernel_size=1),
        )

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
                asr_variant=self.asr_variant,
            )

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        for param in self.mask_decoder.parameters():
            param.requires_grad = True

        for param in self.pnurl.parameters():
            param.requires_grad = use_pnurl

        for param in self.prompt_learner.parameters():
            param.requires_grad = use_coop

    def _tokenize_to_input_ids(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(tokenized, "input_ids"):
            tokens = tokenized.input_ids
        elif isinstance(tokenized, dict) and "input_ids" in tokenized:
            tokens = tokenized["input_ids"]
        else:
            tokens = tokenized

        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens)

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

    @staticmethod
    def _safe_text(value, default: str = "Cell nuclei") -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value if value.strip() else default
        return str(value)

    @staticmethod
    def _safe_scalar(
        value: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        default: float = 0.0,
    ) -> torch.Tensor:
        if value is None:
            return torch.tensor(default, device=device, dtype=dtype)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return torch.tensor(default, device=device, dtype=dtype)
            return value.detach().float().mean().to(device=device, dtype=dtype)
        return torch.tensor(float(value), device=device, dtype=dtype)

    @staticmethod
    def _feature_norm(x: Optional[torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
        if x is None:
            if device is None:
                return torch.tensor(0.0)
            return torch.tensor(0.0, device=device)
        if x.dim() >= 2:
            return x.detach().float().norm(dim=1).mean()
        return x.detach().float().norm()

    def _get_semantic_channel_gate(
        self,
        semantic_delta: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not hasattr(self.pnurl, "semantic_channel_gate"):
            self.pnurl.semantic_channel_gate = SemanticChannelGate(
                embed_dim=semantic_delta.shape[1],
                reduction=16,
                init_bias=-4.0,
                max_gate=self.max_semantic_gate,
            ).to(device)

        channel_gate = self.pnurl.semantic_channel_gate(semantic_delta.to(dtype=torch.float32))
        return channel_gate.to(device=device, dtype=dtype)

    @staticmethod
    def _controlled_residual_injection(
        image_embeddings: torch.Tensor,
        semantic_delta: torch.Tensor,
        channel_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if semantic_delta.shape != image_embeddings.shape:
            raise ValueError(
                f"semantic_delta shape mismatch: expected {tuple(image_embeddings.shape)}, "
                f"got {tuple(semantic_delta.shape)}"
            )

        injected_delta = channel_gate * semantic_delta
        refined_image_embeddings = image_embeddings + injected_delta

        base_feat_norm = image_embeddings.detach().float().norm(dim=1).mean()
        injected_delta_norm = injected_delta.detach().float().norm(dim=1).mean()
        injection_ratio = injected_delta_norm / (base_feat_norm + 1e-6)

        return refined_image_embeddings, injected_delta, injected_delta_norm, injection_ratio

    def forward(self, batched_input, multimask_output=False):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        device = image_embeddings.device

        feat_half, feat_s1, feat_s2 = None, None, None

        if self.use_asr:
            with torch.autocast("cuda", enabled=input_images.is_cuda):
                x_cnn = input_images

                for i in range(3):
                    x_cnn = self.cnn_stage0[i](x_cnn)

                feat_half = x_cnn
                feat_s0 = self.cnn_stage0[3](feat_half)
                feat_s1 = self.cnn_stage1(feat_s0)
                feat_s2 = self.cnn_stage2(feat_s1)

        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)

        organ_indices_list = []
        attribute_texts = []
        base_texts = []

        for x in batched_input:
            organ_indices_list.append(self._get_int_value(x.get("organ_id", 20), default=20))

            if self.asr_regression:
                attribute_texts.append("Cell nuclei")
                base_texts.append("Cell nuclei")
            else:
                attribute_texts.append(self._safe_text(x.get("attribute_text", "Cell nuclei"), "Cell nuclei"))
                base_texts.append(self._safe_text(x.get("text_prompt", "Cell nuclei"), "Cell nuclei"))

        organ_indices = torch.tensor(organ_indices_list, dtype=torch.long, device=device)

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

        if self.use_pnurl:
            if next(self.pnurl.parameters()).device != device:
                self.pnurl = self.pnurl.to(device)

            with torch.no_grad():
                attr_tokens = self._tokenize_to_input_ids(attribute_texts, device)
                attr_text_embed = self.clip_model.encode_text(attr_tokens).float()
                attr_text_embed = F.normalize(attr_text_embed, dim=-1, eps=1e-6)

            pnurl_out = self.pnurl(
                image_features=image_embeddings,
                text_embed=attr_text_embed,
                attribute_labels=attribute_labels,
                return_loss=True,
            )

            if not isinstance(pnurl_out, dict):
                raise TypeError(
                    "PNuRL.forward must return a dict with keys: "
                    "semantic_delta, attr_logits, density_map, "
                    "low_freq_prompt, high_freq_prompt, pnurl_loss. "
                    "The old tuple return protocol is no longer supported."
                )

            required_keys = ("semantic_delta", "low_freq_prompt", "high_freq_prompt")
            missing_keys = [key for key in required_keys if key not in pnurl_out]
            if missing_keys:
                raise KeyError(f"PNuRL output missing required keys: {missing_keys}")

            semantic_delta = pnurl_out["semantic_delta"].to(dtype=image_embeddings.dtype)
            attr_logits = pnurl_out.get("attr_logits", {})
            density_map = pnurl_out.get("density_map", None)

            low_freq_prompt = pnurl_out["low_freq_prompt"]
            high_freq_prompt = pnurl_out["high_freq_prompt"]

            pnurl_loss = pnurl_out.get("pnurl_loss", torch.tensor(0.0, device=device))
            semantic_delta_reg_loss = pnurl_out.get(
                "semantic_delta_reg_loss",
                torch.tensor(0.0, device=device),
            )
            semantic_delta_ratio = pnurl_out.get("semantic_delta_ratio", None)
            semantic_delta_raw_norm = pnurl_out.get("semantic_delta_raw_norm", None)
            semantic_delta_direction_norm = pnurl_out.get("semantic_delta_direction_norm", None)

            channel_gate = self._get_semantic_channel_gate(
                semantic_delta=semantic_delta,
                device=device,
                dtype=image_embeddings.dtype,
            )

            (
                refined_image_embeddings,
                injected_delta,
                injected_delta_norm,
                injection_ratio,
            ) = self._controlled_residual_injection(
                image_embeddings=image_embeddings,
                semantic_delta=semantic_delta,
                channel_gate=channel_gate,
            )

            semantic_delta_norm = semantic_delta.detach().float().norm(dim=1).mean()
            base_feat_norm = image_embeddings.detach().float().norm(dim=1).mean()
            channel_gate_mean = channel_gate.detach().float().mean()
            channel_gate_min = channel_gate.detach().float().min()
            channel_gate_max = channel_gate.detach().float().max()
        else:
            semantic_delta = torch.zeros_like(image_embeddings)
            channel_gate = torch.zeros(
                image_embeddings.shape[0],
                image_embeddings.shape[1],
                1,
                1,
                device=device,
                dtype=image_embeddings.dtype,
            )
            injected_delta = torch.zeros_like(image_embeddings)
            refined_image_embeddings = image_embeddings

            channel_gate_mean = torch.tensor(0.0, device=device)
            channel_gate_min = torch.tensor(0.0, device=device)
            channel_gate_max = torch.tensor(0.0, device=device)

            semantic_delta_norm = torch.tensor(0.0, device=device)
            base_feat_norm = image_embeddings.detach().float().norm(dim=1).mean()
            injected_delta_norm = torch.tensor(0.0, device=device)
            injection_ratio = torch.tensor(0.0, device=device)

            pnurl_loss = torch.tensor(0.0, device=device)
            semantic_delta_reg_loss = torch.tensor(0.0, device=device)
            semantic_delta_ratio = torch.tensor(0.0, device=device)
            semantic_delta_raw_norm = torch.tensor(0.0, device=device)
            semantic_delta_direction_norm = torch.tensor(0.0, device=device)

            attr_logits = {}
            density_map = None
            low_freq_prompt = None
            high_freq_prompt = None

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

        pos_feats = F.normalize(pos_feats.float(), dim=-1, eps=1e-6)
        neg_feats = F.normalize(neg_feats.float(), dim=-1, eps=1e-6)

        text_features = torch.stack([pos_feats, neg_feats], dim=1).float()

        diagnostics = {
            "semantic_channel_gate_mean": channel_gate_mean.detach().float(),
            "semantic_channel_gate_min": channel_gate_min.detach().float(),
            "semantic_channel_gate_max": channel_gate_max.detach().float(),
            "semantic_delta_norm": semantic_delta_norm.detach().float(),
            "base_feat_norm": base_feat_norm.detach().float(),
            "injected_delta_norm": injected_delta_norm.detach().float(),
            "injection_ratio": injection_ratio.detach().float(),
            "semantic_delta_reg_loss": self._safe_scalar(semantic_delta_reg_loss, device),
            "semantic_delta_ratio": self._safe_scalar(semantic_delta_ratio, device),
            "semantic_delta_raw_norm": self._safe_scalar(semantic_delta_raw_norm, device),
            "semantic_delta_direction_norm": self._safe_scalar(semantic_delta_direction_norm, device),
            "pos_text_norm": pos_feats.detach().float().norm(dim=-1).mean(),
            "neg_text_norm": neg_feats.detach().float().norm(dim=-1).mean(),
            "use_pnurl": torch.tensor(float(self.use_pnurl), device=device),
            "use_coop": torch.tensor(float(self.use_coop), device=device),
            "use_ot": torch.tensor(0.0, device=device),
            "asr_variant_legacy": torch.tensor(float(self.asr_variant == "legacy"), device=device),
            "asr_regression": torch.tensor(float(self.asr_regression), device=device),
        }

        fused_image_embeddings = refined_image_embeddings
        heatmap_logits_coarse = self.prompt_generator(refined_image_embeddings, text_features)
        hv_logits_coarse = self.basic_hv_head(refined_image_embeddings)

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

        outputs = []

        for i in range(len(batched_input)):
            prompt_data = prompts_list[i]

            target_h, target_w = batched_input[i]["original_size"]
            input_h, input_w = batched_input[i]["image"].shape[-2:]

            if self.use_asr:
                hv_out_i = hv_logits_out[i:i + 1, :, :input_h, :input_w]
                hm_out_i = heatmap_logits_out[i:i + 1, :, :input_h, :input_w]

                if (input_h, input_w) != (target_h, target_w):
                    hv_out_i = F.interpolate(
                        hv_out_i,
                        size=(target_h, target_w),
                        mode="nearest",
                    )
                    hm_out_i = F.interpolate(
                        hm_out_i,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
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

            common_output = {
                "heatmap_logits": hm_out_i,
                "hv_logits": hv_out_i,
                "attr_logits": attr_logits,
                "density_map": density_map_i,
                "pnurl_loss": pnurl_loss,
                "semantic_delta": semantic_delta[i:i + 1],
                "semantic_channel_gate": channel_gate[i:i + 1],
                "injected_delta": injected_delta[i:i + 1],
                "base_feat": image_embeddings[i:i + 1],
                "semantic_delta_norm": semantic_delta_norm,
                "base_feat_norm": base_feat_norm,
                "injected_delta_norm": injected_delta_norm,
                "injection_ratio": injection_ratio,
                "semantic_delta_reg_loss": semantic_delta_reg_loss,
                "semantic_delta_ratio": diagnostics["semantic_delta_ratio"],
                "semantic_delta_raw_norm": diagnostics["semantic_delta_raw_norm"],
                "semantic_delta_direction_norm": diagnostics["semantic_delta_direction_norm"],
                "diagnostics": diagnostics,
                "organ_cls_loss": getattr(self, "organ_cls_loss_cache", torch.tensor(0.0, device=device)),
            }

            if not prompt_data["has_points"]:
                dummy = fused_image_embeddings[i].sum() * 0.0

                if density_map_i is not None:
                    common_output["density_map"] = density_map_i + dummy

                common_output.update(
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
                    }
                )
                outputs.append(common_output)
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

            curr_low_freq_prompt = low_freq_prompt[i:i + 1] if low_freq_prompt is not None else None
            curr_high_freq_prompt = high_freq_prompt[i:i + 1] if high_freq_prompt is not None else None

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

                sub_low_freq_prompt = None
                sub_high_freq_prompt = None

                if self.asr_variant == "freqpath":
                    if curr_low_freq_prompt is not None:
                        if curr_low_freq_prompt.dim() == 2:
                            sub_low_freq_prompt = curr_low_freq_prompt.expand(current_batch, -1).contiguous()
                        elif curr_low_freq_prompt.dim() == 3:
                            sub_low_freq_prompt = curr_low_freq_prompt.expand(current_batch, -1, -1).contiguous()

                    if curr_high_freq_prompt is not None:
                        if curr_high_freq_prompt.dim() == 2:
                            sub_high_freq_prompt = curr_high_freq_prompt.expand(current_batch, -1).contiguous()
                        elif curr_high_freq_prompt.dim() == 3:
                            sub_high_freq_prompt = curr_high_freq_prompt.expand(current_batch, -1, -1).contiguous()

                sub_mask, sub_iou = self.mask_decoder(
                    image_embeddings=sub_img_embed,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=multimask_output,
                    cnn_feat_s1=sub_cnn_s1,
                    cnn_feat_s2=sub_cnn_s2,
                    attr_prompt=sub_low_freq_prompt,
                    morph_feat=sub_high_freq_prompt,
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

            common_output.update(
                {
                    "masks": mask_post,
                    "iou_predictions": merged_iou,
                    "low_res_logits": merged_logits,
                }
            )
            outputs.append(common_output)

        if self.training and len(outputs) > 0:
            dummy = torch.tensor(0.0, device=device)

            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0

            outputs[0]["heatmap_logits"] = outputs[0]["heatmap_logits"] + dummy

        return outputs