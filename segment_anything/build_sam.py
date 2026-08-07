# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM builder for NucleiSegmentation.

This version is modified for the ASR regression / FreqPath-SAM workflow:

1. The builder instantiates TextSam from modeling/sam.py instead of the base Sam
   exported by modeling/__init__.py.
2. asr_variant is passed into both MaskDecoder and TextSam, so the command line
   can switch between:
      - legacy  : pure-visual ASR regression baseline
      - freqpath: low-frequency semantic + high-frequency morphology branch
3. asr_regression is passed into TextSam. In TextSam, this mode should force
   PNuRL=False, CoOp=False, OT=False, and base prompt "Cell nuclei".
4. The builder keeps checkpoint loading tolerant because architecture changes
   introduce expected missing/unexpected keys when switching ASR variants.
"""

import os
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from .modeling.image_encoder import ImageEncoderViT
from .modeling.mask_decoder import MaskDecoder
from .modeling.prompt_encoder import PromptEncoder
from .modeling.sam import TextSam
from .modeling.transformer import TwoWayTransformer


def _build_sam_print(*args, **kwargs):
    """Print only on DDP rank 0 (reads RANK from environment)."""
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _get_checkpoint(args: Any, prefer: str = "checkpoint") -> Optional[str]:
    """
    Keep compatibility with different scripts:
    - old scripts may use args.sam_checkpoint
    - current vit_b path often uses args.checkpoint
    """
    if prefer == "sam_checkpoint":
        return _get_arg(args, "sam_checkpoint", _get_arg(args, "checkpoint", None))
    return _get_arg(args, "checkpoint", _get_arg(args, "sam_checkpoint", None))


def _get_asr_variant(args: Any) -> str:
    variant = str(_get_arg(args, "asr_variant", "legacy")).lower().strip()
    if variant not in ("legacy", "freqpath"):
        raise ValueError(f"--asr_variant must be 'legacy' or 'freqpath', got {variant}")
    return variant


def _get_asr_regression(args: Any) -> Optional[bool]:
    """
    Return None when the argument is absent, so TextSam can apply its own default:
    legacy -> True, freqpath -> False.
    """
    return _get_arg(args, "asr_regression", None)


def _get_use_coop(args: Any) -> bool:
    """
    Support both names because different train.py versions use different flags.
    """
    return bool(_get_arg(args, "use_coop", _get_arg(args, "use_coop_prompt", False)))


def build_sam_vit_h(args):
    return _build_sam(
        model_type="vit_h",
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="sam_checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", True),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
        semantic_gate_bias_init=_get_arg(args, "semantic_gate_bias_init", None),
        semantic_injection_scale=float(_get_arg(args, "semantic_injection_scale", 1.0)),
        enable_structure_boundary_attr_heads=bool(_get_arg(args, "enable_structure_boundary_attr_heads", False)),
        # --- MultiLevel Attribute Heads (Phase B) ---
        enable_multilevel_attr_heads=bool(_get_arg(args, "enable_multilevel_attr_heads", False)),
        # --- Phase C: Attribute-Text Alignment ---
        enable_attr_text_alignment=bool(_get_arg(args, "enable_attr_text_alignment", False)),
        # --- Phase D: Instance align audit gating ---
        debug_instance_align_audit=bool(_get_arg(args, "debug_instance_align_audit", False)),
        # --- SB GT-CONCH guidance ---
        sb_guidance_mode=str(_get_arg(args, "sb_guidance_mode", "none")),
        sb_guidance_weight=float(_get_arg(args, "sb_guidance_weight", 1.0)),
        sb_conch_freeze=bool(_get_arg(args, "sb_conch_freeze", True)),
        sb_prompt_template_path=str(_get_arg(args, "sb_prompt_template_path", "workdir/attr_stats/structure_boundary_prompt_templates.json")),
        sb_guidance_routing=str(_get_arg(args, "sb_guidance_routing", "structure_low_boundary_high")),
        # --- CONCH text encoder gating ---
        enable_conch_text_encoder=bool(_get_arg(args, "enable_conch_text_encoder", True)),
        # --- SGA-SB: Spatial Granularity-Aligned Structure/Boundary Guidance ---
        spatial_sb_mode=str(_get_arg(args, "spatial_sb_mode", "none")),
        spatial_sb_branch=str(_get_arg(args, "spatial_sb_branch", "both")),
        spatial_structure_guidance_init=float(_get_arg(args, "spatial_structure_guidance_init", 0.05)),
        spatial_boundary_guidance_init=float(_get_arg(args, "spatial_boundary_guidance_init", 0.05)),
        spatial_instance_attr_mode=str(_get_arg(args, "spatial_instance_attr_mode", "none")),
        # --- Numeric Attribute → FreqPath guidance (Exp5) ---
        enable_numeric_attr_freqpath_guidance=bool(_get_arg(args, "enable_numeric_attr_freqpath_guidance", False)),
        numeric_attr_freqpath_hidden_dim=int(_get_arg(args, "numeric_attr_freqpath_hidden_dim", 128)),
        numeric_attr_freqpath_init=str(_get_arg(args, "numeric_attr_freqpath_init", "zero")),
        # --- L1-A local-region text alignment ---
        enable_local_region_text_alignment=bool(_get_arg(args, "enable_local_region_text_alignment", False)),
        local_region_text_prototype_path=_get_arg(args, "local_region_text_prototype_path", None),
        local_region_window_size=int(_get_arg(args, "local_region_window_size", 192)),
        local_region_text_temperature=float(_get_arg(args, "local_region_text_temperature", 0.07)),
        local_region_text_attributes=str(_get_arg(args, "local_region_text_attributes", "density,size_heterogeneity,crowding,boundary_irregularity,elongation")),
        local_region_policy=str(_get_arg(args, "local_region_policy", "complete_only")),
        local_region_text_supervision_only=bool(_get_arg(args, "local_region_text_supervision_only", False)),
        # --- RSGR-1 ---
        enable_rsgr=bool(_get_arg(args, "enable_rsgr", False)),
        rsgr_mode=str(_get_arg(args, "rsgr_mode", "no_local")),
        rsgr_num_regions=int(_get_arg(args, "rsgr_num_regions", 4)),
        rsgr_region_size=int(_get_arg(args, "rsgr_region_size", 192)),
        rsgr_injection_scale=float(_get_arg(args, "rsgr_injection_scale", 0.05)),
        rsgr_max_injection_ratio=float(_get_arg(args, "rsgr_max_injection_ratio", 0.02)),
        rsgr_prototype_source=str(_get_arg(args, "rsgr_prototype_source", "conch")),
        rsgr_prototype_path=_get_arg(args, "rsgr_prototype_path", None),
        rsgr_prototype_detach=bool(_get_arg(args, "rsgr_prototype_detach", True)),
        rsgr_attr_detach=bool(_get_arg(args, "rsgr_attr_detach", False)),
        rsgr_shuffle_scope=str(_get_arg(args, "rsgr_shuffle_scope", "within_sample")),
        rsgr_random_seed=int(_get_arg(args, "rsgr_random_seed", 42)),
        rsgr_overlap_blend=str(_get_arg(args, "rsgr_overlap_blend", "normalized")),
        # --- PromptNu-lite v2 ---
        enable_promptnu_lite_align=bool(_get_arg(args, "enable_promptnu_lite_align", False)),
        promptnu_lite_target=str(_get_arg(args, "promptnu_lite_target", "semantic_delta")),
        promptnu_lite_struct_weight=float(_get_arg(args, "promptnu_lite_struct_weight", 0.0)),
        promptnu_lite_boundary_weight=float(_get_arg(args, "promptnu_lite_boundary_weight", 0.0)),
        promptnu_lite_instance_weight=float(_get_arg(args, "promptnu_lite_instance_weight", 0.0)),
        promptnu_lite_detach_text=bool(_get_arg(args, "promptnu_lite_detach_text", True)),
        promptnu_lite_detach_visual=bool(_get_arg(args, "promptnu_lite_detach_visual", False)),
        promptnu_lite_proj_lr_mult=float(_get_arg(args, "promptnu_lite_proj_lr_mult", 0.5)),
        promptnu_lite_pool_mode=str(_get_arg(args, "promptnu_lite_pool_mode", "gap")),
        # --- PromptNu-guided v3 ---
        enable_promptnu_guided_v3=bool(_get_arg(args, "enable_promptnu_guided_v3", False)),
        promptnu_guided_v3_struct_weight=float(_get_arg(args, "promptnu_guided_v3_struct_weight", 1.0)),
        promptnu_guided_v3_boundary_weight=float(_get_arg(args, "promptnu_guided_v3_boundary_weight", 1.0)),
        promptnu_guided_v3_text_weight=float(_get_arg(args, "promptnu_guided_v3_text_weight", 0.01)),
        promptnu_guided_v3_embed_dim=int(_get_arg(args, "promptnu_guided_v3_embed_dim", 256)),
        promptnu_guided_v3_hidden_dim=int(_get_arg(args, "promptnu_guided_v3_hidden_dim", 128)),
        promptnu_guided_v3_vis_proj_dim=int(_get_arg(args, "promptnu_guided_v3_vis_proj_dim", 512)),
        promptnu_guided_v3_align_loss_weight=float(_get_arg(args, "promptnu_guided_v3_align_loss_weight", 0.1)),
        # --- PromptNu-guided v3.1: CONCH text bank & GT align target ---
        promptnu_guided_v3_use_text_bank=bool(_get_arg(args, "promptnu_guided_v3_use_text_bank", False)),
        # --- PromptNu-guided v3.3: Scale + Additive guidance ---
        promptnu_guided_v3_guidance_mode=str(_get_arg(args, "promptnu_guided_v3_guidance_mode", "scale_add")),
        promptnu_guided_v3_scale_weight=_get_arg(args, "promptnu_guided_v3_scale_weight", None),
        promptnu_guided_v3_delta_weight=float(_get_arg(args, "promptnu_guided_v3_delta_weight", 0.001)),
        promptnu_guided_v3_delta_init_std=float(_get_arg(args, "promptnu_guided_v3_delta_init_std", 1e-5)),
        promptnu_guided_v3_max_guided_delta_ratio=float(_get_arg(args, "promptnu_guided_v3_max_guided_delta_ratio", 0.0)),
        # --- PromptNu-guided v3 diagnostic: prompt source ---
        promptnu_guided_v3_prompt_source=str(_get_arg(args, "promptnu_guided_v3_prompt_source", "pred_attr")),
        # --- PromptNu-guided v3.3 alignment stability ---
        promptnu_guided_v3_align_eps=float(_get_arg(args, "promptnu_guided_v3_align_eps", 1e-8)),
        promptnu_guided_v3_cosine_eps=float(_get_arg(args, "promptnu_guided_v3_cosine_eps", 1e-8)),
        promptnu_guided_v3_min_align_delta_norm=float(_get_arg(args, "promptnu_guided_v3_min_align_delta_norm", 0.0)),
        promptnu_guided_v3_align_low_norm_mode=str(_get_arg(args, "promptnu_guided_v3_align_low_norm_mode", "detach_guided")),
        promptnu_guided_v3_use_gt_align_target=bool(_get_arg(args, "promptnu_guided_v3_use_gt_align_target", False)),
        promptnu_guided_v3_semantic_dim=int(_get_arg(args, "promptnu_guided_v3_semantic_dim", 256)),
        promptnu_guided_v3_text_dim=int(_get_arg(args, "promptnu_guided_v3_text_dim", 512)),
        ablate_semantic_injection=bool(_get_arg(args, "ablate_semantic_injection", False)),
        ablate_pred_attr_guidance=bool(_get_arg(args, "ablate_pred_attr_guidance", False)),
        # --- PromptNu-guided v3 injection ablation ---
        promptnu_guided_v3_injection_ablation=str(_get_arg(args, "promptnu_guided_v3_injection_ablation", "default")),
        promptnu_guided_v3_post_gate_alpha=float(_get_arg(args, "promptnu_guided_v3_post_gate_alpha", 1.0)),
        # --- PNuDP: PromptNu Dense Prediction diagnostic ---
        enable_pnudp_diag=bool(_get_arg(args, "enable_pnudp_diag", False)),
        pnudp_fusion_mode=str(_get_arg(args, "pnudp_fusion_mode", "none")),
        pnudp_scale=float(_get_arg(args, "pnudp_scale", 20.0)),
        # --- PNuDP Dense Training (Stage D) ---
        pnudp_dense_apply_in_eval=bool(_get_arg(args, "pnudp_dense_apply_in_eval", False)),
        pnudp_dense_num_mask_channels=int(_get_arg(args, "pnudp_dense_num_mask_channels", 1)),
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        model_type="vit_l",
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="sam_checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", False),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
        semantic_gate_bias_init=_get_arg(args, "semantic_gate_bias_init", None),
        semantic_injection_scale=float(_get_arg(args, "semantic_injection_scale", 1.0)),
        enable_structure_boundary_attr_heads=bool(_get_arg(args, "enable_structure_boundary_attr_heads", False)),
        # --- MultiLevel Attribute Heads (Phase B) ---
        enable_multilevel_attr_heads=bool(_get_arg(args, "enable_multilevel_attr_heads", False)),
        # --- Phase C: Attribute-Text Alignment ---
        enable_attr_text_alignment=bool(_get_arg(args, "enable_attr_text_alignment", False)),
        # --- Phase D: Instance align audit gating ---
        debug_instance_align_audit=bool(_get_arg(args, "debug_instance_align_audit", False)),
        # --- SB GT-CONCH guidance ---
        sb_guidance_mode=str(_get_arg(args, "sb_guidance_mode", "none")),
        sb_guidance_weight=float(_get_arg(args, "sb_guidance_weight", 1.0)),
        sb_conch_freeze=bool(_get_arg(args, "sb_conch_freeze", True)),
        sb_prompt_template_path=str(_get_arg(args, "sb_prompt_template_path", "workdir/attr_stats/structure_boundary_prompt_templates.json")),
        sb_guidance_routing=str(_get_arg(args, "sb_guidance_routing", "structure_low_boundary_high")),
        # --- CONCH text encoder gating ---
        enable_conch_text_encoder=bool(_get_arg(args, "enable_conch_text_encoder", True)),
        # --- SGA-SB: Spatial Granularity-Aligned Structure/Boundary Guidance ---
        spatial_sb_mode=str(_get_arg(args, "spatial_sb_mode", "none")),
        spatial_sb_branch=str(_get_arg(args, "spatial_sb_branch", "both")),
        spatial_structure_guidance_init=float(_get_arg(args, "spatial_structure_guidance_init", 0.05)),
        spatial_boundary_guidance_init=float(_get_arg(args, "spatial_boundary_guidance_init", 0.05)),
        spatial_instance_attr_mode=str(_get_arg(args, "spatial_instance_attr_mode", "none")),
        # --- Numeric Attribute → FreqPath guidance (Exp5) ---
        enable_numeric_attr_freqpath_guidance=bool(_get_arg(args, "enable_numeric_attr_freqpath_guidance", False)),
        numeric_attr_freqpath_hidden_dim=int(_get_arg(args, "numeric_attr_freqpath_hidden_dim", 128)),
        numeric_attr_freqpath_init=str(_get_arg(args, "numeric_attr_freqpath_init", "zero")),
        # --- L1-A local-region text alignment ---
        enable_local_region_text_alignment=bool(_get_arg(args, "enable_local_region_text_alignment", False)),
        local_region_text_prototype_path=_get_arg(args, "local_region_text_prototype_path", None),
        local_region_window_size=int(_get_arg(args, "local_region_window_size", 192)),
        local_region_text_temperature=float(_get_arg(args, "local_region_text_temperature", 0.07)),
        local_region_text_attributes=str(_get_arg(args, "local_region_text_attributes", "density,size_heterogeneity,crowding,boundary_irregularity,elongation")),
        local_region_policy=str(_get_arg(args, "local_region_policy", "complete_only")),
        local_region_text_supervision_only=bool(_get_arg(args, "local_region_text_supervision_only", False)),
        # --- RSGR-1 ---
        enable_rsgr=bool(_get_arg(args, "enable_rsgr", False)),
        rsgr_mode=str(_get_arg(args, "rsgr_mode", "no_local")),
        rsgr_num_regions=int(_get_arg(args, "rsgr_num_regions", 4)),
        rsgr_region_size=int(_get_arg(args, "rsgr_region_size", 192)),
        rsgr_injection_scale=float(_get_arg(args, "rsgr_injection_scale", 0.05)),
        rsgr_max_injection_ratio=float(_get_arg(args, "rsgr_max_injection_ratio", 0.02)),
        rsgr_prototype_source=str(_get_arg(args, "rsgr_prototype_source", "conch")),
        rsgr_prototype_path=_get_arg(args, "rsgr_prototype_path", None),
        rsgr_prototype_detach=bool(_get_arg(args, "rsgr_prototype_detach", True)),
        rsgr_attr_detach=bool(_get_arg(args, "rsgr_attr_detach", False)),
        rsgr_shuffle_scope=str(_get_arg(args, "rsgr_shuffle_scope", "within_sample")),
        rsgr_random_seed=int(_get_arg(args, "rsgr_random_seed", 42)),
        rsgr_overlap_blend=str(_get_arg(args, "rsgr_overlap_blend", "normalized")),
        # --- PromptNu-lite v2 ---
        enable_promptnu_lite_align=bool(_get_arg(args, "enable_promptnu_lite_align", False)),
        promptnu_lite_target=str(_get_arg(args, "promptnu_lite_target", "semantic_delta")),
        promptnu_lite_struct_weight=float(_get_arg(args, "promptnu_lite_struct_weight", 0.0)),
        promptnu_lite_boundary_weight=float(_get_arg(args, "promptnu_lite_boundary_weight", 0.0)),
        promptnu_lite_instance_weight=float(_get_arg(args, "promptnu_lite_instance_weight", 0.0)),
        promptnu_lite_detach_text=bool(_get_arg(args, "promptnu_lite_detach_text", True)),
        promptnu_lite_detach_visual=bool(_get_arg(args, "promptnu_lite_detach_visual", False)),
        promptnu_lite_proj_lr_mult=float(_get_arg(args, "promptnu_lite_proj_lr_mult", 0.5)),
        promptnu_lite_pool_mode=str(_get_arg(args, "promptnu_lite_pool_mode", "gap")),
        # --- PromptNu-guided v3 ---
        enable_promptnu_guided_v3=bool(_get_arg(args, "enable_promptnu_guided_v3", False)),
        promptnu_guided_v3_struct_weight=float(_get_arg(args, "promptnu_guided_v3_struct_weight", 1.0)),
        promptnu_guided_v3_boundary_weight=float(_get_arg(args, "promptnu_guided_v3_boundary_weight", 1.0)),
        promptnu_guided_v3_text_weight=float(_get_arg(args, "promptnu_guided_v3_text_weight", 0.01)),
        promptnu_guided_v3_embed_dim=int(_get_arg(args, "promptnu_guided_v3_embed_dim", 256)),
        promptnu_guided_v3_hidden_dim=int(_get_arg(args, "promptnu_guided_v3_hidden_dim", 128)),
        promptnu_guided_v3_vis_proj_dim=int(_get_arg(args, "promptnu_guided_v3_vis_proj_dim", 512)),
        promptnu_guided_v3_align_loss_weight=float(_get_arg(args, "promptnu_guided_v3_align_loss_weight", 0.1)),
        # --- PromptNu-guided v3.1: CONCH text bank & GT align target ---
        promptnu_guided_v3_use_text_bank=bool(_get_arg(args, "promptnu_guided_v3_use_text_bank", False)),
        # --- PromptNu-guided v3.3: Scale + Additive guidance ---
        promptnu_guided_v3_guidance_mode=str(_get_arg(args, "promptnu_guided_v3_guidance_mode", "scale_add")),
        promptnu_guided_v3_scale_weight=_get_arg(args, "promptnu_guided_v3_scale_weight", None),
        promptnu_guided_v3_delta_weight=float(_get_arg(args, "promptnu_guided_v3_delta_weight", 0.001)),
        promptnu_guided_v3_delta_init_std=float(_get_arg(args, "promptnu_guided_v3_delta_init_std", 1e-5)),
        promptnu_guided_v3_max_guided_delta_ratio=float(_get_arg(args, "promptnu_guided_v3_max_guided_delta_ratio", 0.0)),
        # --- PromptNu-guided v3 diagnostic: prompt source ---
        promptnu_guided_v3_prompt_source=str(_get_arg(args, "promptnu_guided_v3_prompt_source", "pred_attr")),
        # --- PromptNu-guided v3.3 alignment stability ---
        promptnu_guided_v3_align_eps=float(_get_arg(args, "promptnu_guided_v3_align_eps", 1e-8)),
        promptnu_guided_v3_cosine_eps=float(_get_arg(args, "promptnu_guided_v3_cosine_eps", 1e-8)),
        promptnu_guided_v3_min_align_delta_norm=float(_get_arg(args, "promptnu_guided_v3_min_align_delta_norm", 0.0)),
        promptnu_guided_v3_align_low_norm_mode=str(_get_arg(args, "promptnu_guided_v3_align_low_norm_mode", "detach_guided")),
        promptnu_guided_v3_use_gt_align_target=bool(_get_arg(args, "promptnu_guided_v3_use_gt_align_target", False)),
        promptnu_guided_v3_semantic_dim=int(_get_arg(args, "promptnu_guided_v3_semantic_dim", 256)),
        promptnu_guided_v3_text_dim=int(_get_arg(args, "promptnu_guided_v3_text_dim", 512)),
        ablate_semantic_injection=bool(_get_arg(args, "ablate_semantic_injection", False)),
        ablate_pred_attr_guidance=bool(_get_arg(args, "ablate_pred_attr_guidance", False)),
        # --- PromptNu-guided v3 injection ablation ---
        promptnu_guided_v3_injection_ablation=str(_get_arg(args, "promptnu_guided_v3_injection_ablation", "default")),
        promptnu_guided_v3_post_gate_alpha=float(_get_arg(args, "promptnu_guided_v3_post_gate_alpha", 1.0)),
        # --- PNuDP: PromptNu Dense Prediction diagnostic ---
        enable_pnudp_diag=bool(_get_arg(args, "enable_pnudp_diag", False)),
        pnudp_fusion_mode=str(_get_arg(args, "pnudp_fusion_mode", "none")),
        pnudp_scale=float(_get_arg(args, "pnudp_scale", 20.0)),
        # --- PNuDP Dense Training (Stage D) ---
        pnudp_dense_apply_in_eval=bool(_get_arg(args, "pnudp_dense_apply_in_eval", False)),
        pnudp_dense_num_mask_channels=int(_get_arg(args, "pnudp_dense_num_mask_channels", 1)),
    )


def build_sam_vit_b(args):
    return _build_sam(
        model_type="vit_b",
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", False),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
        semantic_gate_bias_init=_get_arg(args, "semantic_gate_bias_init", None),
        semantic_injection_scale=float(_get_arg(args, "semantic_injection_scale", 1.0)),
        enable_structure_boundary_attr_heads=bool(_get_arg(args, "enable_structure_boundary_attr_heads", False)),
        # --- MultiLevel Attribute Heads (Phase B) ---
        enable_multilevel_attr_heads=bool(_get_arg(args, "enable_multilevel_attr_heads", False)),
        # --- Phase C: Attribute-Text Alignment ---
        enable_attr_text_alignment=bool(_get_arg(args, "enable_attr_text_alignment", False)),
        # --- Phase D: Instance align audit gating ---
        debug_instance_align_audit=bool(_get_arg(args, "debug_instance_align_audit", False)),
        # --- SB GT-CONCH guidance ---
        sb_guidance_mode=str(_get_arg(args, "sb_guidance_mode", "none")),
        sb_guidance_weight=float(_get_arg(args, "sb_guidance_weight", 1.0)),
        sb_conch_freeze=bool(_get_arg(args, "sb_conch_freeze", True)),
        sb_prompt_template_path=str(_get_arg(args, "sb_prompt_template_path", "workdir/attr_stats/structure_boundary_prompt_templates.json")),
        sb_guidance_routing=str(_get_arg(args, "sb_guidance_routing", "structure_low_boundary_high")),
        # --- Text encoder backend ---
        enable_conch_text_encoder=bool(_get_arg(args, "enable_conch_text_encoder", True)),
        # ── CONCHLESS test mode: use text_bank from checkpoint, skip CONCH loading ──
        use_checkpoint_text_bank_without_conch=bool(_get_arg(args, "use_checkpoint_text_bank_without_conch", False)),
        # ── CLIP text encoder backend (Exp7: CLIP_BACKEND_ABLATION) ──
        clip_text_encoder=bool(_get_arg(args, "clip_text_encoder", False)),
        clip_text_encoder_model=str(_get_arg(args, "clip_text_encoder_model", "ViT-B/32")),
        clip_text_encoder_cache_path=str(_get_arg(args, "clip_text_encoder_cache_path", "hf_cache/clip")),
        # --- SGA-SB: Spatial Granularity-Aligned Structure/Boundary Guidance ---
        spatial_sb_mode=str(_get_arg(args, "spatial_sb_mode", "none")),
        spatial_sb_branch=str(_get_arg(args, "spatial_sb_branch", "both")),
        spatial_structure_guidance_init=float(_get_arg(args, "spatial_structure_guidance_init", 0.05)),
        spatial_boundary_guidance_init=float(_get_arg(args, "spatial_boundary_guidance_init", 0.05)),
        spatial_instance_attr_mode=str(_get_arg(args, "spatial_instance_attr_mode", "none")),
        # --- Numeric Attribute → FreqPath guidance (Exp5) ---
        enable_numeric_attr_freqpath_guidance=bool(_get_arg(args, "enable_numeric_attr_freqpath_guidance", False)),
        numeric_attr_freqpath_hidden_dim=int(_get_arg(args, "numeric_attr_freqpath_hidden_dim", 128)),
        numeric_attr_freqpath_init=str(_get_arg(args, "numeric_attr_freqpath_init", "zero")),
        # --- L1-A local-region text alignment ---
        enable_local_region_text_alignment=bool(_get_arg(args, "enable_local_region_text_alignment", False)),
        local_region_text_prototype_path=_get_arg(args, "local_region_text_prototype_path", None),
        local_region_window_size=int(_get_arg(args, "local_region_window_size", 192)),
        local_region_text_temperature=float(_get_arg(args, "local_region_text_temperature", 0.07)),
        local_region_text_attributes=str(_get_arg(args, "local_region_text_attributes", "density,size_heterogeneity,crowding,boundary_irregularity,elongation")),
        local_region_policy=str(_get_arg(args, "local_region_policy", "complete_only")),
        local_region_text_supervision_only=bool(_get_arg(args, "local_region_text_supervision_only", False)),
        # --- RSGR-1 ---
        enable_rsgr=bool(_get_arg(args, "enable_rsgr", False)),
        rsgr_mode=str(_get_arg(args, "rsgr_mode", "no_local")),
        rsgr_num_regions=int(_get_arg(args, "rsgr_num_regions", 4)),
        rsgr_region_size=int(_get_arg(args, "rsgr_region_size", 192)),
        rsgr_injection_scale=float(_get_arg(args, "rsgr_injection_scale", 0.05)),
        rsgr_max_injection_ratio=float(_get_arg(args, "rsgr_max_injection_ratio", 0.02)),
        rsgr_prototype_source=str(_get_arg(args, "rsgr_prototype_source", "conch")),
        rsgr_prototype_path=_get_arg(args, "rsgr_prototype_path", None),
        rsgr_prototype_detach=bool(_get_arg(args, "rsgr_prototype_detach", True)),
        rsgr_attr_detach=bool(_get_arg(args, "rsgr_attr_detach", False)),
        rsgr_shuffle_scope=str(_get_arg(args, "rsgr_shuffle_scope", "within_sample")),
        rsgr_random_seed=int(_get_arg(args, "rsgr_random_seed", 42)),
        rsgr_overlap_blend=str(_get_arg(args, "rsgr_overlap_blend", "normalized")),
        # --- PromptNu-lite v2 ---
        enable_promptnu_lite_align=bool(_get_arg(args, "enable_promptnu_lite_align", False)),
        promptnu_lite_target=str(_get_arg(args, "promptnu_lite_target", "semantic_delta")),
        promptnu_lite_struct_weight=float(_get_arg(args, "promptnu_lite_struct_weight", 0.0)),
        promptnu_lite_boundary_weight=float(_get_arg(args, "promptnu_lite_boundary_weight", 0.0)),
        promptnu_lite_instance_weight=float(_get_arg(args, "promptnu_lite_instance_weight", 0.0)),
        promptnu_lite_detach_text=bool(_get_arg(args, "promptnu_lite_detach_text", True)),
        promptnu_lite_detach_visual=bool(_get_arg(args, "promptnu_lite_detach_visual", False)),
        promptnu_lite_proj_lr_mult=float(_get_arg(args, "promptnu_lite_proj_lr_mult", 0.5)),
        promptnu_lite_pool_mode=str(_get_arg(args, "promptnu_lite_pool_mode", "gap")),
        # --- PromptNu-guided v3 ---
        enable_promptnu_guided_v3=bool(_get_arg(args, "enable_promptnu_guided_v3", False)),
        promptnu_guided_v3_struct_weight=float(_get_arg(args, "promptnu_guided_v3_struct_weight", 1.0)),
        promptnu_guided_v3_boundary_weight=float(_get_arg(args, "promptnu_guided_v3_boundary_weight", 1.0)),
        promptnu_guided_v3_text_weight=float(_get_arg(args, "promptnu_guided_v3_text_weight", 0.01)),
        promptnu_guided_v3_embed_dim=int(_get_arg(args, "promptnu_guided_v3_embed_dim", 256)),
        promptnu_guided_v3_hidden_dim=int(_get_arg(args, "promptnu_guided_v3_hidden_dim", 128)),
        promptnu_guided_v3_vis_proj_dim=int(_get_arg(args, "promptnu_guided_v3_vis_proj_dim", 512)),
        promptnu_guided_v3_align_loss_weight=float(_get_arg(args, "promptnu_guided_v3_align_loss_weight", 0.1)),
        # --- PromptNu-guided v3.1: CONCH text bank & GT align target ---
        promptnu_guided_v3_use_text_bank=bool(_get_arg(args, "promptnu_guided_v3_use_text_bank", False)),
        # --- PromptNu-guided v3.3: Scale + Additive guidance ---
        promptnu_guided_v3_guidance_mode=str(_get_arg(args, "promptnu_guided_v3_guidance_mode", "scale_add")),
        promptnu_guided_v3_scale_weight=_get_arg(args, "promptnu_guided_v3_scale_weight", None),
        promptnu_guided_v3_delta_weight=float(_get_arg(args, "promptnu_guided_v3_delta_weight", 0.001)),
        promptnu_guided_v3_delta_init_std=float(_get_arg(args, "promptnu_guided_v3_delta_init_std", 1e-5)),
        promptnu_guided_v3_max_guided_delta_ratio=float(_get_arg(args, "promptnu_guided_v3_max_guided_delta_ratio", 0.0)),
        # --- PromptNu-guided v3 diagnostic: prompt source ---
        promptnu_guided_v3_prompt_source=str(_get_arg(args, "promptnu_guided_v3_prompt_source", "pred_attr")),
        # --- PromptNu-guided v3.3 alignment stability ---
        promptnu_guided_v3_align_eps=float(_get_arg(args, "promptnu_guided_v3_align_eps", 1e-8)),
        promptnu_guided_v3_cosine_eps=float(_get_arg(args, "promptnu_guided_v3_cosine_eps", 1e-8)),
        promptnu_guided_v3_min_align_delta_norm=float(_get_arg(args, "promptnu_guided_v3_min_align_delta_norm", 0.0)),
        promptnu_guided_v3_align_low_norm_mode=str(_get_arg(args, "promptnu_guided_v3_align_low_norm_mode", "detach_guided")),
        promptnu_guided_v3_use_gt_align_target=bool(_get_arg(args, "promptnu_guided_v3_use_gt_align_target", False)),
        promptnu_guided_v3_semantic_dim=int(_get_arg(args, "promptnu_guided_v3_semantic_dim", 256)),
        promptnu_guided_v3_text_dim=int(_get_arg(args, "promptnu_guided_v3_text_dim", 512)),
        ablate_semantic_injection=bool(_get_arg(args, "ablate_semantic_injection", False)),
        ablate_pred_attr_guidance=bool(_get_arg(args, "ablate_pred_attr_guidance", False)),
        # --- PromptNu-guided v3 injection ablation ---
        promptnu_guided_v3_injection_ablation=str(_get_arg(args, "promptnu_guided_v3_injection_ablation", "default")),
        promptnu_guided_v3_post_gate_alpha=float(_get_arg(args, "promptnu_guided_v3_post_gate_alpha", 1.0)),
        # --- PNuDP: PromptNu Dense Prediction diagnostic ---
        enable_pnudp_diag=bool(_get_arg(args, "enable_pnudp_diag", False)),
        pnudp_fusion_mode=str(_get_arg(args, "pnudp_fusion_mode", "none")),
        pnudp_scale=float(_get_arg(args, "pnudp_scale", 20.0)),
        # --- PNuDP Dense Training (Stage D) ---
        enable_pnudp_dense_train=bool(_get_arg(args, "enable_pnudp_dense_train", False)),
        pnudp_dense_alpha_init=float(_get_arg(args, "pnudp_dense_alpha_init", 0.0)),
        pnudp_dense_logit_proj_init=str(_get_arg(args, "pnudp_dense_logit_proj_init", "zero")),
        pnudp_dense_logit_proj_init_std=float(_get_arg(args, "pnudp_dense_logit_proj_init_std", 1.0)),
        pnudp_dense_apply_in_eval=bool(_get_arg(args, "pnudp_dense_apply_in_eval", False)),
        pnudp_dense_num_mask_channels=int(_get_arg(args, "pnudp_dense_num_mask_channels", 1)),
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    model_type: str,
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes,
    image_size: int,
    checkpoint: Optional[str],
    encoder_adapter: bool,
    use_multimodal_prompt: bool = True,
    clip_model_path: Optional[str] = None,
    num_classes: int = 8,
    use_pnurl: bool = False,
    use_coop: bool = False,
    use_asr: bool = True,
    asr_variant: str = "legacy",
    asr_regression: Optional[bool] = None,
    max_semantic_gate: float = 0.10,
    max_delta_ratio: float = 0.10,
    init_delta_ratio: float = 0.02,
    semantic_gate_bias_init: Optional[float] = None,
    semantic_injection_scale: float = 1.0,
    enable_structure_boundary_attr_heads: bool = False,
    # --- MultiLevel Attribute Heads (Phase B) ---
    enable_multilevel_attr_heads: bool = False,
    # --- Phase C: Attribute-Text Alignment ---
    enable_attr_text_alignment: bool = False,
    # --- Phase D: Instance align audit gating ---
    debug_instance_align_audit: bool = False,
    # --- Numeric Attribute → FreqPath guidance (Exp5) ---
    enable_numeric_attr_freqpath_guidance: bool = False,
    numeric_attr_freqpath_hidden_dim: int = 128,
    numeric_attr_freqpath_init: str = "zero",
    # --- L1-A local-region text alignment ---
    enable_local_region_text_alignment: bool = False,
    local_region_text_prototype_path: Optional[str] = None,
    local_region_window_size: int = 192,
    local_region_text_temperature: float = 0.07,
    local_region_text_attributes: str = "density,size_heterogeneity,crowding,boundary_irregularity,elongation",
    local_region_policy: str = "complete_only",
    local_region_text_supervision_only: bool = False,
    # --- RSGR-1 ---
    enable_rsgr: bool = False,
    rsgr_mode: str = "no_local",
    rsgr_num_regions: int = 4,
    rsgr_region_size: int = 192,
    rsgr_injection_scale: float = 0.05,
    rsgr_max_injection_ratio: float = 0.02,
    rsgr_prototype_source: str = "conch",
    rsgr_prototype_path: Optional[str] = None,
    rsgr_prototype_detach: bool = True,
    rsgr_attr_detach: bool = False,
    rsgr_shuffle_scope: str = "within_sample",
    rsgr_random_seed: int = 42,
    rsgr_overlap_blend: str = "normalized",
    # --- SB GT-CONCH / GT-DIRECT guidance ---
    sb_guidance_mode: str = "none",
    sb_guidance_weight: float = 1.0,
    sb_conch_freeze: bool = True,
    sb_prompt_template_path: str = "workdir/attr_stats/structure_boundary_prompt_templates.json",
    sb_guidance_routing: str = "structure_low_boundary_high",
    sb_direct_adapter_hidden_dim: int = 64,
    # --- Text encoder backend ---
    enable_conch_text_encoder: bool = True,
    # ── CONCHLESS test mode: use text_bank from checkpoint, skip CONCH loading ──
    use_checkpoint_text_bank_without_conch: bool = False,
    # ── CLIP text encoder backend (Exp7: CLIP_BACKEND_ABLATION) ──
    clip_text_encoder: bool = False,
    clip_text_encoder_model: str = "ViT-B/32",
    clip_text_encoder_cache_path: str = "hf_cache/clip",
    # --- SGA-SB v1 CORRECTION: Spatial Granularity-Aligned Structure/Boundary Guidance ---
    spatial_sb_mode: str = "none",
    spatial_sb_branch: str = "both",
    spatial_structure_guidance_init: float = 0.05,
    spatial_boundary_guidance_init: float = 0.05,
    spatial_instance_attr_mode: str = "none",
    # --- PromptNu-lite v2: Residual-Coupled Semantic Alignment ---
    enable_promptnu_lite_align: bool = False,
    promptnu_lite_target: str = "semantic_delta",
    promptnu_lite_struct_weight: float = 0.0,
    promptnu_lite_boundary_weight: float = 0.0,
    promptnu_lite_instance_weight: float = 0.0,
    promptnu_lite_detach_text: bool = True,
    promptnu_lite_detach_visual: bool = False,
    promptnu_lite_proj_lr_mult: float = 0.5,
    promptnu_lite_pool_mode: str = "gap",
    # --- PromptNu-guided v3: Predicted-Attribute Semantic Guidance ---
    enable_promptnu_guided_v3: bool = False,
    promptnu_guided_v3_struct_weight: float = 1.0,
    promptnu_guided_v3_boundary_weight: float = 1.0,
    promptnu_guided_v3_text_weight: float = 0.01,
    promptnu_guided_v3_embed_dim: int = 256,
    promptnu_guided_v3_hidden_dim: int = 128,
    promptnu_guided_v3_vis_proj_dim: int = 512,
    promptnu_guided_v3_align_loss_weight: float = 0.1,
    # --- PromptNu-guided v3.1: CONCH text bank & GT align target ---
    promptnu_guided_v3_use_text_bank: bool = False,
    promptnu_guided_v3_use_gt_align_target: bool = False,
    promptnu_guided_v3_semantic_dim: int = 256,
    promptnu_guided_v3_text_dim: int = 512,
    promptnu_guided_v3_strict_audit: bool = False,
    # --- PromptNu-guided v3.3: Scale + Additive guidance ---
    promptnu_guided_v3_guidance_mode: str = "scale_add",
    promptnu_guided_v3_scale_weight: Optional[float] = None,
    promptnu_guided_v3_delta_weight: float = 0.001,
    promptnu_guided_v3_delta_init_std: float = 1e-5,
    promptnu_guided_v3_max_guided_delta_ratio: float = 0.0,
    # --- PromptNu-guided v3 diagnostic: prompt source ---
    promptnu_guided_v3_prompt_source: str = "pred_attr",
    # --- PromptNu-guided v3.3 alignment stability ---
    promptnu_guided_v3_align_eps: float = 1e-8,
    promptnu_guided_v3_cosine_eps: float = 1e-8,
    promptnu_guided_v3_min_align_delta_norm: float = 0.0,
    promptnu_guided_v3_align_low_norm_mode: str = "detach_guided",
    ablate_semantic_injection: bool = False,
    ablate_pred_attr_guidance: bool = False,
    # --- PromptNu-guided v3 injection ablation ---
    promptnu_guided_v3_injection_ablation: str = "default",
    promptnu_guided_v3_post_gate_alpha: float = 1.0,
    # --- PNuDP: PromptNu Dense Prediction diagnostic ---
    enable_pnudp_diag: bool = False,
    pnudp_fusion_mode: str = "none",
    pnudp_scale: float = 20.0,
    # --- PNuDP Dense Training (Stage D) ---
    enable_pnudp_dense_train: bool = False,
    pnudp_dense_alpha_init: float = 0.0,
    pnudp_dense_logit_proj_init: str = "zero",
    pnudp_dense_logit_proj_init_std: float = 1.0,
    pnudp_dense_apply_in_eval: bool = False,
    pnudp_dense_num_mask_channels: int = 1,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # PNuRL output is aligned with SAM prompt embedding dimension.
    text_embed_dim = prompt_embed_dim if use_pnurl else None

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_asr=use_asr,
        asr_variant=asr_variant,
    )

    sam = TextSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train=encoder_adapter,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            use_multimodal_prompt=use_multimodal_prompt,
            clip_model_path=clip_model_path,
            num_classes=num_classes,
            text_embed_dim=text_embed_dim,
        ),
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        embed_dim=prompt_embed_dim,
        num_organs=_get_num_organs(num_classes),
        use_pnurl=use_pnurl,
        use_coop=use_coop,
        use_ot=False,
        use_asr=use_asr,
        asr_variant=asr_variant,
        asr_regression=asr_regression,
        max_semantic_gate=max_semantic_gate,
        max_delta_ratio=max_delta_ratio,
        init_delta_ratio=init_delta_ratio,
        semantic_gate_bias_init=semantic_gate_bias_init,
        semantic_injection_scale=semantic_injection_scale,
        enable_structure_boundary_attr_heads=enable_structure_boundary_attr_heads,
        enable_multilevel_attr_heads=enable_multilevel_attr_heads,
        enable_attr_text_alignment=enable_attr_text_alignment,
        # --- SB GT-CONCH / GT-DIRECT guidance ---
        sb_guidance_mode=sb_guidance_mode,
        sb_guidance_weight=sb_guidance_weight,
        sb_conch_freeze=sb_conch_freeze,
        sb_prompt_template_path=sb_prompt_template_path,
        sb_guidance_routing=sb_guidance_routing,
        sb_direct_adapter_hidden_dim=sb_direct_adapter_hidden_dim,
        # --- Phase D: Instance align audit gating ---
        debug_instance_align_audit=debug_instance_align_audit,
        # --- Numeric Attribute → FreqPath guidance (Exp5) ---
        enable_numeric_attr_freqpath_guidance=enable_numeric_attr_freqpath_guidance,
        numeric_attr_freqpath_hidden_dim=numeric_attr_freqpath_hidden_dim,
        numeric_attr_freqpath_init=numeric_attr_freqpath_init,
        # --- L1-A local-region text alignment ---
        enable_local_region_text_alignment=enable_local_region_text_alignment,
        local_region_text_prototype_path=local_region_text_prototype_path,
        local_region_window_size=local_region_window_size,
        local_region_text_temperature=local_region_text_temperature,
        local_region_text_attributes=local_region_text_attributes,
        local_region_policy=local_region_policy,
        local_region_text_supervision_only=local_region_text_supervision_only,
        # --- RSGR-1 ---
        enable_rsgr=enable_rsgr,
        rsgr_mode=rsgr_mode,
        rsgr_num_regions=rsgr_num_regions,
        rsgr_region_size=rsgr_region_size,
        rsgr_injection_scale=rsgr_injection_scale,
        rsgr_max_injection_ratio=rsgr_max_injection_ratio,
        rsgr_prototype_source=rsgr_prototype_source,
        rsgr_prototype_path=rsgr_prototype_path,
        rsgr_prototype_detach=rsgr_prototype_detach,
        rsgr_attr_detach=rsgr_attr_detach,
        rsgr_shuffle_scope=rsgr_shuffle_scope,
        rsgr_random_seed=rsgr_random_seed,
        rsgr_overlap_blend=rsgr_overlap_blend,
        # --- Text encoder backend ---
        enable_conch_text_encoder=enable_conch_text_encoder,
        # ── CONCHLESS test mode ──
        use_checkpoint_text_bank_without_conch=use_checkpoint_text_bank_without_conch,
        # ── CLIP text encoder backend (Exp7: CLIP_BACKEND_ABLATION) ──
        clip_text_encoder=clip_text_encoder,
        clip_text_encoder_model=clip_text_encoder_model,
        clip_text_encoder_cache_path=clip_text_encoder_cache_path,
        # --- SGA-SB v1 CORRECTION: Spatial Granularity-Aligned Structure/Boundary Guidance ---
        spatial_sb_mode=spatial_sb_mode,
        spatial_sb_branch=spatial_sb_branch,
        spatial_structure_guidance_init=spatial_structure_guidance_init,
        spatial_boundary_guidance_init=spatial_boundary_guidance_init,
        spatial_instance_attr_mode=spatial_instance_attr_mode,
        # --- PromptNu-lite v2: Residual-Coupled Semantic Alignment ---
        enable_promptnu_lite_align=enable_promptnu_lite_align,
        promptnu_lite_target=promptnu_lite_target,
        promptnu_lite_struct_weight=promptnu_lite_struct_weight,
        promptnu_lite_boundary_weight=promptnu_lite_boundary_weight,
        promptnu_lite_instance_weight=promptnu_lite_instance_weight,
        promptnu_lite_detach_text=promptnu_lite_detach_text,
        promptnu_lite_detach_visual=promptnu_lite_detach_visual,
        promptnu_lite_proj_lr_mult=promptnu_lite_proj_lr_mult,
        promptnu_lite_pool_mode=promptnu_lite_pool_mode,
        # --- PromptNu-guided v3 ---
        enable_promptnu_guided_v3=enable_promptnu_guided_v3,
        promptnu_guided_v3_struct_weight=promptnu_guided_v3_struct_weight,
        promptnu_guided_v3_boundary_weight=promptnu_guided_v3_boundary_weight,
        promptnu_guided_v3_text_weight=promptnu_guided_v3_text_weight,
        promptnu_guided_v3_embed_dim=promptnu_guided_v3_embed_dim,
        promptnu_guided_v3_hidden_dim=promptnu_guided_v3_hidden_dim,
        promptnu_guided_v3_vis_proj_dim=promptnu_guided_v3_vis_proj_dim,
        promptnu_guided_v3_align_loss_weight=promptnu_guided_v3_align_loss_weight,
        # --- PromptNu-guided v3.1: CONCH text bank & GT align target ---
        promptnu_guided_v3_use_text_bank=promptnu_guided_v3_use_text_bank,
        promptnu_guided_v3_use_gt_align_target=promptnu_guided_v3_use_gt_align_target,
        promptnu_guided_v3_semantic_dim=promptnu_guided_v3_semantic_dim,
        promptnu_guided_v3_text_dim=promptnu_guided_v3_text_dim,
        promptnu_guided_v3_strict_audit=promptnu_guided_v3_strict_audit,
        # --- PromptNu-guided v3 diagnostic: prompt source ---
        promptnu_guided_v3_prompt_source=promptnu_guided_v3_prompt_source,
        # --- PromptNu-guided v3.3: Scale + Additive guidance (MUST explicitly pass) ---
        promptnu_guided_v3_guidance_mode=promptnu_guided_v3_guidance_mode,
        promptnu_guided_v3_scale_weight=promptnu_guided_v3_scale_weight,
        promptnu_guided_v3_delta_weight=promptnu_guided_v3_delta_weight,
        promptnu_guided_v3_delta_init_std=promptnu_guided_v3_delta_init_std,
        promptnu_guided_v3_max_guided_delta_ratio=promptnu_guided_v3_max_guided_delta_ratio,
        # --- PromptNu-guided v3.3 alignment stability ---
        promptnu_guided_v3_align_eps=promptnu_guided_v3_align_eps,
        promptnu_guided_v3_cosine_eps=promptnu_guided_v3_cosine_eps,
        promptnu_guided_v3_min_align_delta_norm=promptnu_guided_v3_min_align_delta_norm,
        promptnu_guided_v3_align_low_norm_mode=promptnu_guided_v3_align_low_norm_mode,
        ablate_semantic_injection=ablate_semantic_injection,
        ablate_pred_attr_guidance=ablate_pred_attr_guidance,
        # --- PromptNu-guided v3 injection ablation ---
        promptnu_guided_v3_injection_ablation=promptnu_guided_v3_injection_ablation,
        promptnu_guided_v3_post_gate_alpha=promptnu_guided_v3_post_gate_alpha,
        # --- PNuDP: PromptNu Dense Prediction diagnostic ---
        enable_pnudp_diag=enable_pnudp_diag,
        pnudp_fusion_mode=pnudp_fusion_mode,
        pnudp_scale=pnudp_scale,
        # --- PNuDP Dense Training (Stage D) ---
        enable_pnudp_dense_train=enable_pnudp_dense_train,
        pnudp_dense_alpha_init=pnudp_dense_alpha_init,
        pnudp_dense_logit_proj_init=pnudp_dense_logit_proj_init,
        pnudp_dense_logit_proj_init_std=pnudp_dense_logit_proj_init_std,
        pnudp_dense_apply_in_eval=pnudp_dense_apply_in_eval,
        pnudp_dense_num_mask_channels=pnudp_dense_num_mask_channels,
    )

    _build_sam_print(
        f"[CONCHLESS_ARG_AUDIT:build_sam.py] use_checkpoint_text_bank_without_conch={use_checkpoint_text_bank_without_conch}"
    )
    _build_sam_print(
        f"[build_sam] image_size={image_size} | use_asr={use_asr} | "
        f"asr_variant={asr_variant} | asr_regression={asr_regression} | "
        f"use_pnurl={use_pnurl} | use_coop={use_coop} | "
        f"enable_structure_boundary_attr_heads={enable_structure_boundary_attr_heads} | "
        f"sb_guidance_mode={sb_guidance_mode} | "
        f"spatial_sb_mode={spatial_sb_mode}"
    )
    _build_sam_print(
        f"[CONCH_CONFIG][build_sam.py] "
        f"enable_conch_text_encoder={enable_conch_text_encoder}"
    )
    _build_sam_print(
        f"[PROMPTNU_GUIDED_V3_CONFIG] "
        f"model_type={model_type} | "
        f"enable_promptnu_guided_v3={enable_promptnu_guided_v3} | "
        f"use_text_bank={promptnu_guided_v3_use_text_bank} | "
        f"use_gt_align_target={promptnu_guided_v3_use_gt_align_target} | "
        f"semantic_dim={promptnu_guided_v3_semantic_dim} | "
        f"text_dim={promptnu_guided_v3_text_dim} | "
        f"embed_dim={promptnu_guided_v3_embed_dim} | "
        f"hidden_dim={promptnu_guided_v3_hidden_dim} | "
        f"vis_proj_dim={promptnu_guided_v3_vis_proj_dim} | "
        f"align_loss_weight={promptnu_guided_v3_align_loss_weight} | "
        f"struct_weight={promptnu_guided_v3_struct_weight} | "
        f"boundary_weight={promptnu_guided_v3_boundary_weight} | "
        f"text_weight={promptnu_guided_v3_text_weight} | "
        # --- v3.3 scale+additive ---
        f"guidance_mode={promptnu_guided_v3_guidance_mode} | "
        f"scale_weight={promptnu_guided_v3_scale_weight} | "
        f"delta_weight={promptnu_guided_v3_delta_weight} | "
        f"delta_init_std={promptnu_guided_v3_delta_init_std} | "
        f"max_guided_delta_ratio={promptnu_guided_v3_max_guided_delta_ratio} | "
        f"ablate_semantic_injection={ablate_semantic_injection} | "
        f"ablate_pred_attr_guidance={ablate_pred_attr_guidance}"
    )

    if checkpoint is not None:
        _load_checkpoint_into_model(
            sam=sam,
            checkpoint=checkpoint,
            image_size=image_size,
            vit_patch_size=vit_patch_size,
            encoder_adapter=encoder_adapter,
        )

    return sam


def _get_num_organs(num_classes: int) -> int:
    """
    TextSam's DualPromptLearner defaults to 21 organs.
    num_classes in PromptEncoder is not always organ count, so keep the previous
    TextSam default unless a larger value is explicitly requested.
    """
    return max(int(num_classes), 21)


def _load_checkpoint_into_model(
    sam: torch.nn.Module,
    checkpoint: str,
    image_size: int,
    vit_patch_size: int,
    encoder_adapter: bool,
) -> None:
    with open(checkpoint, "rb") as f:
        # Training checkpoints can include optimizer / scheduler objects.
        state_dict = torch.load(f, map_location="cpu", weights_only=False)

    actual_state_dict = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict

    # ── Handle text_bank buffer shape mismatches (CONCHLESS/CLIP mode) ──
    # The model registers _structure/boundary_text_bank_buffer as torch.zeros(0),
    # but the checkpoint may have them as [5,3,512] / [4,3,512]. Re-register with
    # correct checkpoint shape before loading to avoid load_state_dict error.
    _text_bank_buffers = ("_structure_text_bank_buffer", "_boundary_text_bank_buffer")
    _model_sd = sam.state_dict()
    for _buf_name in _text_bank_buffers:
        if _buf_name in actual_state_dict and _buf_name in _model_sd:
            _ckpt_shape = actual_state_dict[_buf_name].shape
            _mdl_shape = _model_sd[_buf_name].shape
            if _ckpt_shape != _mdl_shape:
                _build_sam_print(f"[TEXT_BANK_RESIZE] {_buf_name}: model={_mdl_shape} → checkpoint={_ckpt_shape}")
                sam.register_buffer(_buf_name, torch.zeros(_ckpt_shape), persistent=True)

    try:
        missing, unexpected = sam.load_state_dict(actual_state_dict, strict=False)
        # Empty text-bank placeholders are compatibility buffers, not learned
        # model weights. Old visual/Phase-B/Exp5 checkpoints legitimately omit them.
        _expected_empty_buffer_missing = [
            key for key in missing
            if key in _text_bank_buffers
            and key in sam.state_dict()
            and sam.state_dict()[key].numel() == 0
        ]
        _unexpected_missing = [key for key in missing if key not in _expected_empty_buffer_missing]
        _build_sam_print(f"*******load {checkpoint}")
        _build_sam_print(
            f"[checkpoint] missing_keys={len(_unexpected_missing)} | "
            f"expected_empty_buffer_missing={len(_expected_empty_buffer_missing)} | "
            f"unexpected_keys={len(unexpected)}"
        )
        if len(_unexpected_missing) > 0:
            _build_sam_print(f"[checkpoint] first missing keys: {_unexpected_missing[:20]}")
        if len(unexpected) > 0:
            _build_sam_print(f"[checkpoint] first unexpected keys: {unexpected[:20]}")
        if bool(getattr(sam, "enable_rsgr", False)):
            _rsgr_missing = sorted(key for key in _unexpected_missing if key.startswith("rsgr."))
            _non_rsgr_missing = sorted(key for key in _unexpected_missing if not key.startswith("rsgr."))
            _unexpected_non_rsgr = sorted(key for key in unexpected if not key.startswith("rsgr."))
            _build_sam_print(
                f"[RSGR_RESUME] parent_checkpoint={checkpoint} | "
                f"rsgr_missing_keys={_rsgr_missing} | "
                f"non_rsgr_missing_keys={_non_rsgr_missing} | "
                f"unexpected_non_rsgr_keys={_unexpected_non_rsgr} | initialized_rsgr=True"
            )

        # ── Audit v3 adapter keys: detect silent shape mismatches ──
        _v3_ckpt_keys = [k for k in actual_state_dict.keys() if "promptnu_guided_adapter" in k]
        if len(_v3_ckpt_keys) > 0:
            _v3_loaded = [k for k in _v3_ckpt_keys if k not in missing]
            _v3_skipped = [k for k in _v3_ckpt_keys if k in missing]
            if len(_v3_skipped) > 0:
                _build_sam_print(
                    f"[PROMPTNU_GUIDED_V3_CKPT_ERROR] "
                    f"v3 adapter key mismatch; comparison invalid. "
                    f"Skipped {len(_v3_skipped)}/{len(_v3_ckpt_keys)} keys: {_v3_skipped[:10]}"
                )
                _build_sam_print(
                    f"[PROMPTNU_GUIDED_V3_CKPT_ERROR] "
                    f"Check that promptnu_guided_v3_use_text_bank=True and "
                    f"text_encoder dim matches checkpoint (2*text_dim={2*int(getattr(getattr(sam, 'promptnu_guided_v3_text_dim', 512), 'promptnu_guided_v3_text_dim', 512) if hasattr(sam, 'promptnu_guided_v3_text_dim') else 512)})."
                )
            else:
                _build_sam_print(
                    f"[PROMPTNU_GUIDED_V3_CKPT_OK] "
                    f"All {len(_v3_loaded)} v3 adapter keys loaded successfully."
                )
        return
    except RuntimeError as exc:
        _build_sam_print(f"[checkpoint] direct non-strict load failed: {exc}")
        _build_sam_print("*******interpolate")

    new_state_dict = load_from(sam, actual_state_dict, image_size, vit_patch_size)
    missing, unexpected = sam.load_state_dict(new_state_dict, strict=False)
    _build_sam_print(f"*******load {checkpoint}")
    _build_sam_print(f"[checkpoint/interpolate] missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")
    _build_sam_print(f"[checkpoint] encoder_adapter={encoder_adapter}")


def load_from(sam, state_dicts: Dict[str, torch.Tensor], image_size: int, vit_patch_size: int):
    sam_dict = sam.state_dict()
    except_keys = ["mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head"]

    new_state_dict = {
        k: v
        for k, v in state_dicts.items()
        if k in sam_dict.keys()
        and except_keys[0] not in k
        and except_keys[1] not in k
        and except_keys[2] not in k
    }

    if "image_encoder.pos_embed" not in new_state_dict:
        _build_sam_print(
            "Warning: 'image_encoder.pos_embed' not found in checkpoint. "
            f"Available keys: {list(new_state_dict.keys())[:10]}..."
        )
        sam_dict.update(new_state_dict)
        return sam_dict

    pos_embed = new_state_dict["image_encoder.pos_embed"]
    token_size = int(image_size // vit_patch_size)

    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            (token_size, token_size),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        new_state_dict["image_encoder.pos_embed"] = pos_embed

        rel_pos_keys = [k for k in sam_dict.keys() if "rel_pos" in k]
        global_rel_pos_keys = [
            k
            for k in rel_pos_keys
            if "2" in k
            or "5" in k
            or "7" in k
            or "8" in k
            or "11" in k
            or "13" in k
            or "15" in k
            or "23" in k
            or "31" in k
        ]

        for k in global_rel_pos_keys:
            if k not in new_state_dict:
                continue

            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)

            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(
                    rel_pos_params,
                    (h_check, w_check),
                    mode="bilinear",
                    align_corners=False,
                )

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict
