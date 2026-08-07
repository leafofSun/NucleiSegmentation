"""
PNuDP Dense Diagnostic Module (PromptNu-style Dense Text-Image Matching).

A standalone, test-only diagnostic module that:
  1. Projects spatial features [B, C, H, W] → text embedding space [B, D, H, W]
  2. Computes dense similarity maps [B, K, H, W] with a text bank [K, D]
  3. Supports fusion ablations: none, logit_add, feature_concat, film
  4. Prints structured [PNUDP_DENSE_DIAG] diagnostics

Reference: PromptNu dense prediction branch.

NOTE: No training. No full test. Compare-script-only diagnostic.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Stage D audit-log gating ──
from training.logging_utils import audit_print


# ====================================================================
# Constants
# ====================================================================
# Structure (5 attrs) + Boundary (4 attrs) = 9 attrs, each with 3 levels (low/mid/high)
NUM_SB_PROMPTS = (5 + 4) * 3  # 27


# ====================================================================
# Deterministic Projection Modes for [B, K, H, W] → [B, 1, H, W]
# ====================================================================
PNUDP_DENSE_PROJECT_MODES = [
    "zero_conv",
    "mean",
    "max",
    "top1_margin",
    "entropy_conf",
    "mean_centered",
    "zscore_mean",
    "zscore_top1_margin",
]


def project_dense_logits_deterministic(
    dense_text_logits: torch.Tensor,
    project_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Project [B, K, H, W] dense text logits to [B, 1, H, W] using a
    deterministic (non-learnable) reduction.

    Args:
        dense_text_logits: [B, K, H, W] dense similarity maps.
        project_mode: One of PNUDP_DENSE_PROJECT_MODES (excluding zero_conv).
        eps: Small constant for numerical stability.

    Returns:
        projected: [B, 1, H, W] projected logits.

    Raises:
        ValueError: If project_mode is unknown.
    """
    B, K, H, W = dense_text_logits.shape
    mode = str(project_mode).strip().lower()

    if mode == "mean":
        # Simple mean over prompt dimension
        return dense_text_logits.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    elif mode == "max":
        # Maximum over prompt dimension
        return dense_text_logits.max(dim=1, keepdim=True).values  # [B, 1, H, W]

    elif mode == "top1_margin":
        # Margin between top-1 and top-2 prompt logits
        top2 = dense_text_logits.topk(2, dim=1).values  # [B, 2, H, W]
        return top2[:, 0:1] - top2[:, 1:2]  # [B, 1, H, W]

    elif mode == "entropy_conf":
        # Confidence measured as log(K) - entropy
        # Higher certainty → larger logit bias
        p = torch.softmax(dense_text_logits, dim=1)  # [B, K, H, W]
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        logK = math.log(K)
        return logK - entropy  # [B, 1, H, W]

    elif mode == "mean_centered":
        # Mean with spatial centering
        mean = dense_text_logits.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_mean = mean.mean(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        return mean - spatial_mean

    elif mode == "zscore_mean":
        # Spatial z-score of the mean projection
        mean = dense_text_logits.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        mu = mean.mean(dim=(2, 3), keepdim=True)            # [B, 1, 1, 1]
        sigma = mean.std(dim=(2, 3), keepdim=True) + eps    # [B, 1, 1, 1]
        return (mean - mu) / sigma

    elif mode == "zscore_top1_margin":
        # Spatial z-score of the top-1 margin projection
        top2 = dense_text_logits.topk(2, dim=1).values  # [B, 2, H, W]
        margin = top2[:, 0:1] - top2[:, 1:2]            # [B, 1, H, W]
        mu = margin.mean(dim=(2, 3), keepdim=True)       # [B, 1, 1, 1]
        sigma = margin.std(dim=(2, 3), keepdim=True) + eps  # [B, 1, 1, 1]
        return (margin - mu) / sigma

    else:
        raise ValueError(
            f"Unknown project_mode='{project_mode}'. "
            f"Must be one of {PNUDP_DENSE_PROJECT_MODES}"
        )


class PromptNuDenseDiag(nn.Module):
    """
    PNuDP dense text prediction diagnostic module.

    Features:
        - Spatial feature → text space projection (Conv1x1)
        - Dense similarity computation via einsum("bdhw,kd->bkhw")
        - Fusion modes for test-time ablation
        - Diagnostic statistics (norm, entropy, top-1 distribution)

    Fusion modes:
        - none       : diagnostic only, no feature/logit change
        - logit_add  : mask_logits += alpha * projected_dense_text_logits
        - feature_concat : concat(F, dense_feat) → 1x1 conv → original channels
        - film       : FiLM modulate F from dense_feat
    """

    def __init__(
        self,
        feat_dim: int = 256,
        text_dim: int = 512,
        logit_scale: float = 20.0,
        num_prompts: int = NUM_SB_PROMPTS,
        alpha: float = 0.1,
    ):
        """
        Args:
            feat_dim: Input spatial feature channels (e.g., 256 for image_embedding,
                      or 32 for decoder_upscaled_feature).
            text_dim: CONCH text embedding dimension (default: 512).
            logit_scale: Scale factor for dense similarity (default: 20.0).
            num_prompts: Number of text prompts in the text bank (default: 27).
            alpha: Weight for logit_add fusion (default: 0.1).
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.logit_scale = logit_scale
        self.num_prompts = num_prompts
        self.alpha = alpha

        # ---- 1. Projection: spatial feat → text space ----
        # Conv1x1(C, 512), no bias, Kaiming init so initial similarity is meaningful
        self.proj = nn.Conv2d(feat_dim, text_dim, kernel_size=1, bias=False)

        # ---- 2. Logit projection: [B, K, H, W] → [B, 1, H, W] for logit_add fusion ----
        self.logit_proj = nn.Conv2d(num_prompts, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.logit_proj.weight)

        # ---- 3. Feature concat fusion: [B, C+D, H, W] → [B, C, H, W] ----
        self.fusion_proj = nn.Conv2d(feat_dim + text_dim, feat_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion_proj.weight)

        # ---- 4. FiLM fusion: [B, D, H, W] → [B, 2*C, H, W] ----
        self.film_proj = nn.Conv2d(text_dim, feat_dim * 2, kernel_size=1, bias=False)
        nn.init.zeros_(self.film_proj.weight)

        audit_print(
            "PNUDP_DENSE_DIAG_INIT",
            f"[PNUDP_DENSE_DIAG_INIT] feat_dim={feat_dim} text_dim={text_dim} "
            f"logit_scale={logit_scale:.2f} num_prompts={num_prompts} alpha={alpha:.4f}",
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        text_bank: torch.Tensor,
        fusion_mode: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            spatial_feat: [B, C, H, W] spatial features (image_embedding or decoder feature).
            text_bank: [K, D] text embeddings (normalized or not; will be normalized internally).
            fusion_mode: One of "none", "logit_add", "feature_concat", "film".

        Returns:
            dense_text_logits: [B, K, H, W] dense similarity maps.
            fused_feat: [B, C, H, W] (spatial_feat or fused version).
            diagnostics: Dict with diagnostic scalars.
        """
        B, C, H, W = spatial_feat.shape
        K, D = text_bank.shape
        device = spatial_feat.device

        # ---- 1. Project to text space ----
        dense_feat = self.proj(spatial_feat)  # [B, D, H, W]

        # ---- 2. L2 normalize both ----
        dense_feat = F.normalize(dense_feat, dim=1, eps=1e-6)
        text_bank_norm = F.normalize(text_bank, dim=1, eps=1e-6)

        # ---- 3. Dense similarity: [B, D, H, W] x [K, D] → [B, K, H, W] ----
        dense_text_logits = self.logit_scale * torch.einsum(
            "bdhw,kd->bkhw", dense_feat, text_bank_norm
        )

        # ---- 4. Diagnostics ----
        diagnostics = self._compute_diagnostics(dense_text_logits, dense_feat, text_bank_norm)

        # ---- 5. Fusion ----
        fused_feat = self._apply_fusion(spatial_feat, dense_feat, dense_text_logits, fusion_mode)

        return dense_text_logits, fused_feat, diagnostics

    # ================================================================
    # Diagnostics
    # ================================================================
    def _compute_diagnostics(
        self,
        dense_text_logits: torch.Tensor,
        dense_feat: torch.Tensor,
        text_bank_norm: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute diagnostic statistics from dense similarity maps."""
        diagnostics: Dict[str, Any] = {}
        with torch.no_grad():
            _logits = dense_text_logits.detach().float()

            # Feature/bank norms
            diagnostics["dense_feat_norm"] = float(dense_feat.norm(dim=1).mean().cpu().item())
            diagnostics["text_bank_norm"] = float(text_bank_norm.norm(dim=1).mean().cpu().item())

            # Logit statistics
            diagnostics["dense_text_logits_mean"] = float(_logits.mean().item())
            diagnostics["dense_text_logits_std"] = float(_logits.std().item())
            diagnostics["dense_text_logits_max"] = float(_logits.max().item())
            diagnostics["dense_text_logits_min"] = float(_logits.min().item())

            # Entropy over prompt dimension
            _probs = torch.softmax(_logits, dim=1)  # [B, K, H, W]
            _entropy = (-_probs * torch.log(_probs + 1e-8)).sum(dim=1).mean().item()
            diagnostics["dense_text_logits_entropy"] = float(_entropy)

            # Top-1 prompt index distribution
            _top1 = _probs.argmax(dim=1)  # [B, H, W]
            K_total = _logits.shape[1]
            for kk in range(min(K_total, 27)):
                _key = f"top1_prompt_{kk}_ratio"
                diagnostics[_key] = float((_top1 == kk).float().mean().cpu().item())

        return diagnostics

    # ================================================================
    # Fusion
    # ================================================================
    def _apply_fusion(
        self,
        spatial_feat: torch.Tensor,
        dense_feat: torch.Tensor,
        dense_text_logits: torch.Tensor,
        fusion_mode: str,
    ) -> torch.Tensor:
        """
        Apply fusion and return (possibly modified) spatial_feat.

        For logit_add fusion, the actual addition happens outside this method
        (on mask_logits). Here, we just return spatial_feat unchanged for logit_add.
        """
        mode = str(fusion_mode).strip().lower()

        if mode in (None, "none", ""):
            return spatial_feat

        elif mode == "logit_add":
            # Fusion is applied to mask_logits externally; spatial_feat unchanged.
            return spatial_feat

        elif mode == "feature_concat":
            # Concatenate spatial_feat + dense_feat, project back to original channels
            cat_feat = torch.cat([spatial_feat, dense_feat], dim=1)  # [B, C+D, H, W]
            return spatial_feat + self.fusion_proj(cat_feat)

        elif mode == "film":
            # FiLM modulate spatial_feat from dense_feat
            _film = self.film_proj(dense_feat)  # [B, 2*C, H, W]
            _scale, _bias = _film.chunk(2, dim=1)
            return spatial_feat * (1.0 + _scale) + _bias

        else:
            print(
                f"[PNUDP_DENSE_DIAG] WARNING: unknown fusion_mode={fusion_mode}, "
                f"falling back to none",
                flush=True,
            )
            return spatial_feat

    def project_dense_logits_to_mask(
        self,
        dense_text_logits: torch.Tensor,
        project_mode: str = "zero_conv",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Project [B, K, H, W] dense similarity to [B, C, H, W] mask logits,
        where C = self.logit_proj.out_channels (num_mask_channels).

        Used for logit_add fusion:
            mask_logits_on = mask_logits_base + alpha * projected

        Args:
            dense_text_logits: [B, K, H, W] dense similarity maps.
            project_mode: Projection mode. If "zero_conv", uses the learnable
                          self.logit_proj Conv2d (may be zero-initialized).
                          Otherwise, uses the corresponding deterministic reduction.
            eps: Numerical stability constant (for entropy / z-score modes).

        Returns:
            projected: [B, C, H, W] projected logits (C = num_mask_channels).
        """
        mode = str(project_mode).strip().lower()
        if mode == "zero_conv":
            return self.logit_proj(dense_text_logits)  # [B, 1, H, W]
        else:
            return project_dense_logits_deterministic(
                dense_text_logits, project_mode=mode, eps=eps,
            )


# ====================================================================
# Helper: Build text bank from model's SB cache
# ====================================================================
def build_sb_text_bank(
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a combined structure+boundary text bank [27, D] from model's
    cached CONCH embeddings.

    Returns:
        text_bank: [27, D] float32 tensor on device.
    """
    _s_bank = model._get_sb_text_bank("structure", device)  # [5, 3, D]
    _b_bank = model._get_sb_text_bank("boundary", device)    # [4, 3, D]
    _s_flat = _s_bank.reshape(-1, _s_bank.shape[-1])         # [15, D]
    _b_flat = _b_bank.reshape(-1, _b_bank.shape[-1])         # [12, D]
    text_bank = torch.cat([_s_flat, _b_flat], dim=0)          # [27, D]
    return text_bank


def build_uniform_text_bank(
    num_prompts: int,
    text_dim: int,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Build a random text bank [K, D] with normalized uniform random vectors.
    """
    bank = torch.randn(num_prompts, text_dim, device=device, dtype=torch.float32)
    bank = F.normalize(bank, dim=1, eps=1e-6)
    bank = bank * scale
    return bank


def build_fixed_global_text_bank(
    text_dim: int,
    device: torch.device,
    num_prompts: int = 27,
) -> torch.Tensor:
    """
    Build a text bank where all K rows are the same fixed vector.
    Useful to check if spatial differentiation is driven by text variation.
    """
    fixed = torch.ones(text_dim, device=device, dtype=torch.float32)
    fixed = F.normalize(fixed, dim=0, eps=1e-6)
    bank = fixed.unsqueeze(0).expand(num_prompts, -1).contiguous()
    return bank


def build_oracle_gt_text_bank(
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Same as build_sb_text_bank in test-only mode (no GT available).
    In a training scenario, this would use GT attribute labels to select
    specific text embeddings.
    """
    return build_sb_text_bank(model, device)


# ====================================================================
# Helper: Print [PNUDP_LOGIT_ADD_AUDIT] enhanced audit
# ====================================================================
def print_pnudp_logit_add_audit(
    project_mode: str,
    alpha: float,
    dense_text_logits: torch.Tensor,
    projected: torch.Tensor,
    logits_base: torch.Tensor,
    logits_fused: torch.Tensor,
    fusion_delta_norm: float,
    prob_mask_l1: float,
    binary_diff_ratio: float,
):
    """
    Print structured [PNUDP_LOGIT_ADD_AUDIT] with all projection stats.

    Args:
        project_mode: The projection mode used.
        alpha: Logit add fusion weight.
        dense_text_logits: [B, K, H, W] original dense similarity.
        projected: [B, 1, H, W] projected logits.
        logits_base: [B, 1, H', W'] base mask logits.
        logits_fused: [B, 1, H', W'] fused mask logits.
        fusion_delta_norm: Per-element norm of alpha * projected.
        prob_mask_l1: Mean abs diff between sigmoid(base) and sigmoid(fused).
        binary_diff_ratio: Fraction of pixels where binary mask changed.
    """
    with torch.no_grad():
        _p = projected.detach().float()
        _lb = logits_base.detach().float()
        _lf = logits_fused.detach().float()
        _dtl = dense_text_logits.detach().float()

        _proj_mean = float(_p.mean().item())
        _proj_std = float(_p.std().item())
        _proj_min = float(_p.min().item())
        _proj_max = float(_p.max().item())
        _base_norm = float(_lb.norm().item() / max(_lb.numel(), 1))
        _fused_norm = float(_lf.norm().item() / max(_lf.numel(), 1))

    print("[PNUDP_LOGIT_ADD_AUDIT]", flush=True)
    print(f"  project_mode={project_mode}", flush=True)
    print(f"  alpha={alpha:.6e}", flush=True)
    print(f"  dense_text_logits_shape={list(_dtl.shape)}", flush=True)
    print(f"  projected_dense_logits_shape={list(_p.shape)}", flush=True)
    print(f"  projected_dense_logits_mean={_proj_mean:.8e}", flush=True)
    print(f"  projected_dense_logits_std={_proj_std:.8e}", flush=True)
    print(f"  projected_dense_logits_min={_proj_min:.8e}", flush=True)
    print(f"  projected_dense_logits_max={_proj_max:.8e}", flush=True)
    print(f"  fusion_delta_norm={fusion_delta_norm:.8e}", flush=True)
    print(f"  base_logits_norm={_base_norm:.8e}", flush=True)
    print(f"  fused_logits_norm={_fused_norm:.8e}", flush=True)
    print(f"  logit_add_prob_mask_l1={prob_mask_l1:.8e}", flush=True)
    print(f"  logit_add_binary_diff_ratio={binary_diff_ratio:.8e}", flush=True)


# ====================================================================
# Helper: Print [PNUDP_DENSE_DIAG] structured diagnostic
# ====================================================================
def print_pnudp_dense_diag(
    diag: Dict[str, Any],
    feature_source: str,
    fusion_mode: str,
    text_source: str,
    alpha: float,
):
    """
    Print structured [PNUDP_DENSE_DIAG] diagnostic block.
    """
    _dfn = diag.get("dense_feat_norm", float("nan"))
    _tbn = diag.get("text_bank_norm", float("nan"))
    _lm = diag.get("dense_text_logits_mean", float("nan"))
    _ls = diag.get("dense_text_logits_std", float("nan"))
    _lx = diag.get("dense_text_logits_max", float("nan"))
    _ent = diag.get("dense_text_logits_entropy", float("nan"))

    # Top-1 prompt distribution (first 6)
    _top1_parts = []
    for kk in range(6):
        _val = diag.get(f"top1_prompt_{kk}_ratio", 0.0)
        if isinstance(_val, float):
            _top1_parts.append(f"p{kk}={_val:.4f}")
    _top1_str = ", ".join(_top1_parts)

    print("[PNUDP_DENSE_DIAG]", flush=True)
    print(f"  feature_source={feature_source}", flush=True)
    print(f"  fusion_mode={fusion_mode}", flush=True)
    print(f"  text_source={text_source}", flush=True)
    print(f"  alpha={alpha:.4f}", flush=True)
    print(f"  dense_feat_norm={_dfn:.6e}", flush=True)
    print(f"  text_bank_norm={_tbn:.6e}", flush=True)
    print(f"  dense_text_logits_mean={_lm:.6e}  std={_ls:.6e}  max={_lx:.6e}", flush=True)
    print(f"  dense_text_logits_entropy={_ent:.6e}", flush=True)
    print(f"  top1_prompt_hist={_top1_str}", flush=True)


# ====================================================================
# PNuDPDenseTrain: Trainable PNuDP dense projection module (Stage D)
#
# Architecture (identical to PromptNuDenseDiag but always trainable):
#   1. self.proj: Conv1x1(C, 512), Kaiming init — project spatial feat → text space
#   2. self.logit_proj: Conv2d(K, C_mask, 1), zero init — project [B,K,H,W] → [B,C_mask,H,W]
#                       where C_mask = pnudp_dense_num_mask_channels (default=1).
#                       When C_mask=3, bias directly matches 3-channel merged_logits
#                       without broadcasting, enabling channel-specific PNuDP.
#   3. self.dense_alpha: learnable scalar (initialized to 0.0) — controls fusion strength
#
# Forward:
#   dense_feat = normalize(proj(spatial_feat))           # [B,512,H,W]
#   dense_text_logits = logit_scale * einsum(bdhw,kd->bkhw)  # [B,K,H,W]
#   pnudp_bias = logit_proj(dense_text_logits)           # [B,C_mask,H,W]
#   fused_logits = base_low_res_logits + alpha * pnudp_bias
#
# Training: only proj, logit_proj, and alpha are trainable.
# Everything else (image encoder, CONCH text encoder, mask decoder backbone) is frozen.
# ====================================================================
class PNuDPDenseTrain(nn.Module):
    """
    Trainable PNuDP dense prediction module for Stage D training.
    Produces a spatial bias term that is added to base mask logits.

    Only 3 trainable components:
      - proj: Conv1x1(C→512, bias=False), Kaiming normal init
      - logit_proj: Conv2d(K→1, kernel_size=1, bias=False), zero init
      - dense_alpha: learnable scalar (float32, init=0.0)
    """

    def __init__(
        self,
        feat_dim: int = 256,
        text_dim: int = 512,
        logit_scale: float = 20.0,
        num_prompts: int = NUM_SB_PROMPTS,
        alpha_init: float = 0.0,
        logit_proj_init: str = "zero",
        logit_proj_init_std: float = 1.0,
        pnudp_dense_num_mask_channels: int = 1,
    ):
        """
        Args:
            feat_dim: Input spatial feature channels (default: 256 for image_embedding).
            text_dim: CONCH text embedding dimension (default: 512).
            logit_scale: Scale factor for dense similarity (default: 20.0).
            num_prompts: Number of text prompts in the text bank (default: 27).
            alpha_init: Initial value for learnable alpha scalar (default: 0.0).
            logit_proj_init: Initialization mode for logit_proj.weight.
                'zero' = nn.init.zeros_;
                'normal' = N(0, logit_proj_init_std);
                'mean' = constant(1/num_prompts * logit_proj_init_std).
                Default: 'zero'.
            logit_proj_init_std: Standard deviation for 'normal' mode, or
                multiplier for 'mean' mode. Default: 1.0.
            pnudp_dense_num_mask_channels: Number of output channels for logit_proj.
                When >1, produces [B, C, H, W] bias that directly matches
                merged_logits channels (e.g., C=3 for 3 multimask outputs).
                Default: 1 (original broadcast behavior).
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.logit_scale = logit_scale
        self.num_prompts = num_prompts
        self.num_mask_channels = pnudp_dense_num_mask_channels

        # ---- 1. Spatial → text space projection ----
        # Conv1x1(C, 512), no bias, Kaiming normal init
        self.proj = nn.Conv2d(feat_dim, text_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

        # ---- 2. Logit projection: [B, K, H, W] → [B, C, H, W] ----
        # C = pnudp_dense_num_mask_channels (default=1, channel-specific if >1)
        self.logit_proj = nn.Conv2d(num_prompts, pnudp_dense_num_mask_channels, kernel_size=1, bias=False)
        # Initialize based on logit_proj_init mode
        _logit_init = logit_proj_init.lower()
        if _logit_init == "zero":
            nn.init.zeros_(self.logit_proj.weight)
        elif _logit_init == "normal":
            nn.init.normal_(self.logit_proj.weight, mean=0.0, std=logit_proj_init_std)
        elif _logit_init == "mean":
            # All K channels initialized to 1/K * init_std
            _mean_val = (1.0 / num_prompts) * logit_proj_init_std
            nn.init.constant_(self.logit_proj.weight, _mean_val)
        else:
            raise ValueError(
                f"Unknown logit_proj_init='{logit_proj_init}'. "
                f"Choose from: zero, normal, mean."
            )
        # Compute init-time weight statistics
        with torch.no_grad():
            _w = self.logit_proj.weight.detach().float()
            _weight_norm = float(_w.norm().item())
            _weight_std = float(_w.std().item())
        audit_print(
            "PNUDP_DENSE_LOGIT_PROJ_INIT",
            f"[PNUDP_DENSE_LOGIT_PROJ_INIT] mode={logit_proj_init} "
            f"init_std={logit_proj_init_std:.4f} "
            f"num_prompts={num_prompts} "
            f"num_mask_channels={pnudp_dense_num_mask_channels} "
            f"weight_shape={list(self.logit_proj.weight.shape)} "
            f"weight_norm={_weight_norm:.6e} "
            f"weight_std={_weight_std:.6e}"
            + (
                f" mean_val={_mean_val:.6e}"
                if _logit_init == "mean"
                else ""
            ),
        )

        # ---- 3. Learnable fusion alpha ----
        self.dense_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        audit_print(
            "PNUDP_DENSE_TRAIN_INIT",
            f"[PNUDP_DENSE_TRAIN_INIT] feat_dim={feat_dim} text_dim={text_dim} "
            f"logit_scale={logit_scale:.2f} num_prompts={num_prompts} "
            f"num_mask_channels={pnudp_dense_num_mask_channels} "
            f"alpha_init={alpha_init:.4f} "
            f"logit_proj_init={logit_proj_init} logit_proj_init_std={logit_proj_init_std:.4f}",
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        text_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat: [B, C, H, W] spatial features (image_embedding).
            text_bank: [K, D] text embeddings. Will be L2-normalized internally.

        Returns:
            dense_text_logits: [B, K, H, W] dense similarity maps (detached for audit).
            pnudp_bias: [B, C_mask, H, W] projected bias term = logit_proj(dense_text_logits),
                        where C_mask = self.num_mask_channels (default=1).
                        When C_mask=3, bias directly matches 3-channel merged_logits
                        without broadcasting.
        """
        B, C, H, W = spatial_feat.shape
        K, D = text_bank.shape

        # ---- 1. Project to text space ----
        dense_feat = self.proj(spatial_feat)  # [B, D, H, W]

        # ---- 2. L2 normalize both ----
        dense_feat = F.normalize(dense_feat, dim=1, eps=1e-6)
        text_bank_norm = F.normalize(text_bank, dim=1, eps=1e-6)

        # ---- 3. Dense similarity: [B, D, H, W] x [K, D] → [B, K, H, W] ----
        dense_text_logits = self.logit_scale * torch.einsum(
            "bdhw,kd->bkhw", dense_feat, text_bank_norm
        )

        # ---- 4. Project to bias: [B, K, H, W] → [B, 1, H, W] ----
        pnudp_bias = self.logit_proj(dense_text_logits)  # [B, 1, H, W]

        return dense_text_logits, pnudp_bias


# ====================================================================
# Helper: Build PNuDP dense diag module based on feature_source
# ====================================================================
def build_pnudp_dense_diag(
    args,
    device: torch.device,
) -> PromptNuDenseDiag:
    """
    Factory function: create PromptNuDenseDiag with correct feat_dim.

    Args:
        args: Namespace with pnudp arguments.
        device: Target device.

    Returns:
        PromptNuDenseDiag module on device.
    """
    feature_source = str(getattr(args, "pnudp_feature_source", "image_embedding")).strip().lower()
    alpha = float(getattr(args, "pnudp_alpha", 0.1))

    if feature_source == "image_embedding":
        feat_dim = 256  # SAM ViT-B image encoder output dim
    elif feature_source == "decoder_upscaled_feature":
        feat_dim = 32   # MaskDecoder upscaled feature dim (transformer_dim // 8 = 256 // 8)
    else:
        raise ValueError(f"Unknown pnudp_feature_source={feature_source}")

    diag_module = PromptNuDenseDiag(
        feat_dim=feat_dim,
        text_dim=512,
        logit_scale=20.0,
        num_prompts=NUM_SB_PROMPTS,
        alpha=alpha,
    )
    diag_module.to(device)
    diag_module.eval()
    return diag_module


# ====================================================================
# Helper: Build text bank based on text_source
# ====================================================================
def build_text_bank(
    text_source: str,
    model: nn.Module,
    device: torch.device,
    num_prompts: int = NUM_SB_PROMPTS,
    text_dim: int = 512,
) -> torch.Tensor:
    """
    Build text bank [K, D] according to text_source.

    Args:
        text_source: One of "pred_attr", "uniform_bank", "fixed_global", "oracle_gt_attr".
        model: The TextSam model (needed for pred_attr / oracle_gt_attr).
        device: Target device.

    Returns:
        text_bank: [K, D] float32 tensor on device.
    """
    source = str(text_source).strip().lower()

    if source == "pred_attr":
        return build_sb_text_bank(model, device)
    elif source == "uniform_bank":
        return build_uniform_text_bank(num_prompts, text_dim, device)
    elif source == "fixed_global":
        return build_fixed_global_text_bank(text_dim, device, num_prompts)
    elif source == "oracle_gt_attr":
        return build_oracle_gt_text_bank(model, device)
    else:
        print(
            f"[PNUDP_DENSE_DIAG] WARNING: unknown text_source={text_source}, "
            f"falling back to pred_attr",
            flush=True,
        )
        return build_sb_text_bank(model, device)


