#!/usr/bin/env python3
"""
compare_v3_on_off_inference.py

Purpose:
    Same checkpoint, same images → compare v3-off vs v3-on model outputs.
    Diagnostic tool for verifying whether PromptNu-guided v3 guidance actually
    changes mask predictions during test-time inference.

Usage:
    python scripts/compare_v3_on_off_inference.py \
        --checkpoint /path/to/best_model.pth \
        --data_path /path/to/test/images \
        [other args matching test.py defaults]

Output:
    [COMPARE_V3_ON_OFF] per-sample comparison metrics
    [COMPARE_V3_ON_OFF_SUMMARY] aggregated comparison metrics
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Project imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
from segment_anything.build_sam import build_sam_vit_b

# ── PNuDP Dense Diagnostic ──
from training.pnudp_dense_diag import (
    PromptNuDenseDiag,
    build_pnudp_dense_diag,
    build_text_bank,
    print_pnudp_dense_diag,
    print_pnudp_logit_add_audit,
    project_dense_logits_deterministic,
    PNUDP_DENSE_PROJECT_MODES,
    NUM_SB_PROMPTS,
)


# ==============================================================================
# 1. Argument parser
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v3-off vs v3-on inference on the same images."
    )

    # ── Core paths ──
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the shared checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test image directory")

    # ── Model architecture ──
    parser.add_argument("--model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type (default: vit_b)")

    # ── Test args matching test.py default ──
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of test images to compare (default: 8)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference (default: cuda:0)")

    # ── DataLoader / transform ──
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize image size (default: 512)")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Crop size (default: 256)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")

    # ── PNuRL / attr ──
    parser.add_argument("--use_pnurl", action="store_true", default=True,
                        help="Enable PNuRL semantic residual path")
    parser.add_argument("--enable_attr_text_alignment", action="store_true", default=False,
                        help="Enable attribute-text alignment heads")
    parser.add_argument("--enable_multilevel_attr_heads", action="store_true", default=True,
                        help="Enable multi-level attribute heads")

    # ── ASR ──
    parser.add_argument("--use_asr", action="store_true", default=True,
                        help="Enable ASR upsampler")
    parser.add_argument("--asr_variant", type=str, default="freqpath",
                        choices=["legacy", "freqpath"],
                        help="ASR variant (default: freqpath)")

    # ── HF / CONCH ──
    parser.add_argument("--hf_hub_offline", action="store_true", default=False,
                        help="Set HF_HUB_OFFLINE=1")
    parser.add_argument("--conch_cache_path", type=str, default=None,
                        help="HuggingFace cache path for CONCH")

    # ── v3-off args (model_off uses enable_promptnu_guided_v3=False) ──
    # v3-on args: all promptnu_guided_v3_* parameters
    parser.add_argument("--promptnu_guided_v3_use_text_bank", action="store_true", default=True,
                        help="Enable CONCH text bank for v3 guidance (v3.3 checkpoint requires True)")
    parser.add_argument("--promptnu_guided_v3_prompt_source", type=str, default="pred_attr",
                        choices=["pred_attr", "fixed_global", "uniform_bank", "oracle_gt_attr"],
                        help="Prompt source for v3 guidance (default: pred_attr)")
    parser.add_argument("--promptnu_guided_v3_guidance_mode", type=str, default="scale_add",
                        choices=["scale", "additive", "scale_add"],
                        help="v3 guidance mode (default: scale_add)")
    parser.add_argument("--promptnu_guided_v3_scale_weight", type=float, default=0.05,
                        help="Scale weight for v3 guidance (default: 0.05)")
    parser.add_argument("--promptnu_guided_v3_delta_weight", type=float, default=0.001,
                        help="Delta weight for v3 additive branch (default: 0.001)")
    parser.add_argument("--promptnu_guided_v3_max_guided_delta_ratio", type=float, default=0.02,
                        help="Max additive delta / base norm ratio (default: 0.02)")

    # ── v3 injection ablation ──
    parser.add_argument("--promptnu_guided_v3_injection_ablation", type=str, default="default",
                        choices=["default", "bypass_gate", "post_gate_add", "replace_semantic_delta"],
                        help="V3 injection ablation mode (default: default)")
    parser.add_argument("--promptnu_guided_v3_post_gate_alpha", type=float, default=1.0,
                        help="Alpha for post_gate_add mode (default: 1.0)")

    # ── PNuDP: PromptNu Dense Prediction diagnostic ──
    parser.add_argument("--enable_pnudp_diag", action="store_true", default=False,
                        help="Enable PNuDP dense text prediction diagnostic (legacy)")
    parser.add_argument("--pnudp_fusion_mode", type=str, default="none",
                        choices=["none", "pnudp_aux_only", "pnudp_concat_fusion", "pnudp_film_fusion"],
                        help="PNuDP fusion mode (default: none)")
    parser.add_argument("--pnudp_scale", type=float, default=20.0,
                        help="Scale factor for PNuDP dense similarity (default: 20.0)")

    # ── PNuDP Dense Diagnostic (new, from user spec) ──
    parser.add_argument("--enable_pnudp_dense_diag", action="store_true", default=False,
                        help="Enable PNuDP dense diagnostic (PromptNu-style dense text-image matching)")
    parser.add_argument("--pnudp_dense_fusion_mode", type=str, default="none",
                        choices=["none", "logit_add", "feature_concat", "film"],
                        help="PNuDP dense fusion ablation mode (default: none)")
    parser.add_argument("--pnudp_dense_alpha", type=float, default=0.1,
                        help="Alpha weight for logit_add fusion (default: 0.1)")
    parser.add_argument("--pnudp_text_source", type=str, default="pred_attr",
                        choices=["pred_attr", "uniform_bank", "fixed_global", "oracle_gt_attr"],
                        help="Text bank source for PNuDP dense diag (default: pred_attr)")
    parser.add_argument("--pnudp_feature_source", type=str, default="image_embedding",
                        choices=["image_embedding", "decoder_upscaled_feature"],
                        help="Spatial feature source for PNuDP dense diag (default: image_embedding)")
    parser.add_argument("--pnudp_dense_debug_constant_logit", type=float, default=None,
                        help="If set, replace projected_dense_logits with constant tensor of this value "
                             "(e.g. 1.0). Used to verify logit_add wiring: with alpha=0.1, prob_mask_l1 "
                             "must be > 0 if fusion is correctly wired into compare_outputs.",
                        )

    # ── PNuDP Dense Project Mode (deterministic projection) ──
    parser.add_argument("--pnudp_dense_project_mode", type=str, default="zero_conv",
                        choices=PNUDP_DENSE_PROJECT_MODES,
                        help="Projection mode for [B,K,H,W] → [B,1,H,W] reduction "
                             f"(default: zero_conv). Choices: {PNUDP_DENSE_PROJECT_MODES}")
    parser.add_argument("--pnudp_dense_project_eps", type=float, default=1e-6,
                        help="Numerical stability epsilon for deterministic projection "
                             "(default: 1e-6)")

    # ── Semantic injection ──
    parser.add_argument("--max_semantic_gate", type=float, default=0.10,
                        help="Max semantic channel gate (default: 0.10)")
    parser.add_argument("--init_delta_ratio", type=float, default=0.005,
                        help="Initial delta ratio (default: 0.005)")
    parser.add_argument("--max_delta_ratio", type=float, default=0.02,
                        help="Max delta ratio (default: 0.02)")
    parser.add_argument("--semantic_injection_scale", type=float, default=1.0,
                        help="Semantic injection scale (default: 1.0)")

    # ── Encoder ──
    parser.add_argument("--encoder_adapter", action="store_true", default=True,
                        help="Use encoder adapter")

    # ── SB (structure/boundary) ──
    parser.add_argument("--enable_structure_boundary_attr_heads", action="store_true", default=False)
    parser.add_argument("--sb_guidance_mode", type=str, default="none")
    parser.add_argument("--sb_prompt_template_path", type=str,
                        default="workdir/attr_stats/structure_boundary_prompt_templates.json")

    # ── CoOp ──
    parser.add_argument("--use_coop", action="store_true", default=False)

    # ── Misc ──
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--use_multimodal_prompt", action="store_true", default=False)
    parser.add_argument("--semantic_gate_bias_init", type=float, default=None)
    parser.add_argument("--enable_conch_text_encoder", action="store_true", default=True)

    # ── Test-only attr-text alignment disable ──
    parser.add_argument("--disable_attr_text_alignment_forward_in_test", action="store_true", default=True,
                        help="Disable training-only attr-text alignment forward in eval/test comparison. "
                             "Modules remain built and loaded; only the forward branch that calls "
                             "_get_attr_text_embeddings() is suppressed (default: True).")

    return parser.parse_args()


# ==============================================================================
# 2. Build base args namespace (shared params for both models)
# ==============================================================================
def _build_base_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build a SimpleNamespace with all params shared by v3-off and v3-on."""
    base = SimpleNamespace(**vars(args))
    # Remove flags that differ between v3-on and v3-off
    # (these will be set explicitly when building each model)
    return base


def _build_model(args_ns: SimpleNamespace, enable_v3: bool) -> TextSam:
    """Build TextSam with the given config, enable/disable v3 guidance."""
    ns = SimpleNamespace(**vars(args_ns))
    ns.enable_promptnu_guided_v3 = enable_v3
    # When v3 is off, ensure v3-related build flags are still present (they'll be False)
    return build_sam_vit_b(ns)


# ==============================================================================
# 3. Device alignment helpers
# ==============================================================================
def _ensure_sam_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Move the entire model (params + buffers) to device, then forcibly migrate
    pixel_mean / pixel_std at every submodule level.

    This is a belt-and-suspenders approach: even if pixel_mean / pixel_std were
    reassigned as plain CPU tensors after model construction or checkpoint loading,
    we catch and move them here.
    """
    model.to(device)
    for m in model.modules():
        if hasattr(m, "pixel_mean") and torch.is_tensor(m.pixel_mean):
            m.pixel_mean = m.pixel_mean.to(device)
        if hasattr(m, "pixel_std") and torch.is_tensor(m.pixel_std):
            m.pixel_std = m.pixel_std.to(device)
    return model


def _move_batch_to_device(
    batch_input: List[Dict[str, Any]], device: torch.device
) -> List[Dict[str, Any]]:
    """
    Move every tensor inside each sample dict to *device*.
    Non-tensor fields (strings, ints, None, etc.) are left unchanged.
    """
    moved = []
    for sample in batch_input:
        item = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                item[k] = v.to(device, non_blocking=True)
            else:
                item[k] = v
        moved.append(item)
    return moved


# ==============================================================================
# 4. Load checkpoint
# ==============================================================================
def load_checkpoint(model: TextSam, ckpt_path: str, device: torch.device):
    """Load checkpoint weights into model, handling key mismatches."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Try common checkpoint key formats
    state_dict = ckpt
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break
    # Strip 'module.' prefix if present
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # Load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # ── Strict audit: v3 adapter keys must NOT be skipped ──
    v3_ckpt_keys = [k for k in state_dict.keys() if "promptnu_guided_adapter" in k]
    v3_required = [
        "promptnu_guided_adapter.text_encoder.0.weight",
        "promptnu_guided_adapter.text_encoder.0.bias",
        "promptnu_guided_adapter.scale_head.weight",
        "promptnu_guided_adapter.scale_head.bias",
        "promptnu_guided_adapter.delta_head.weight",
        "promptnu_guided_adapter.delta_head.bias",
    ]
    if len(v3_ckpt_keys) > 0:
        v3_skipped = [k for k in v3_required if k in missing]
        if len(v3_skipped) > 0:
            print(
                f"[PROMPTNU_GUIDED_V3_CKPT_ERROR] "
                f"v3 adapter key mismatch; comparison invalid. "
                f"Skipped {len(v3_skipped)}/{len(v3_required)} required keys: {v3_skipped}"
            )
            print(
                f"[PROMPTNU_GUIDED_V3_CKPT_ERROR] "
                f"This means v3 adapter is randomly initialized — comparison results are INVALID. "
                f"Check --promptnu_guided_v3_use_text_bank (must be True for v3.3 checkpoint)."
            )
        else:
            print(
                f"[PROMPTNU_GUIDED_V3_CKPT_OK] "
                f"All {len(v3_required)} required v3 adapter keys loaded successfully."
            )
    else:
        print(f"[WARN] No promptnu_guided_adapter keys found in checkpoint.")

    # NOTE: .to(device) / .eval() are now done in main() via _ensure_sam_device()
    return model


# ==============================================================================
# 5. Load test images
# ==============================================================================
def load_test_images(data_path: str, num_samples: int, image_size: int) -> List[Dict[str, Any]]:
    """Load first N images from data_path, create batched input dicts.

    NOTE: returned tensors are on CPU. Caller must move them to device
    via _move_batch_to_device() before forwarding.
    """
    image_files = sorted([
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.lower().endswith((".png", ".tif", ".tiff"))
    ])
    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in {data_path}")
    image_files = image_files[:num_samples]

    samples = []
    for img_path in image_files:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Resize to image_size
        h, w = image_rgb.shape[:2]
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
        # Normalize to [0,1] float tensor [C,H,W] (CPU)
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Extract organ info from sidecar json
        json_path = os.path.splitext(img_path)[0] + ".json"
        organ_name = "Generic"
        organ_id = 20
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                if isinstance(data, dict):
                    organ_name = data.get("organ_id", "Generic")
                    organ_id = {
                        "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2,
                        "Breast": 3, "Cervix": 4, "Colon": 5, "Esophagus": 6,
                        "HeadNeck": 7, "Kidney": 8, "Liver": 9, "Lung": 10,
                        "Ovarian": 11, "Pancreatic": 12, "Prostate": 13,
                        "Skin": 14, "Stomach": 15, "Testis": 16, "Thyroid": 17,
                        "Uterus": 18, "Brain": 19, "Generic": 20,
                    }.get(organ_name, 20)
            except Exception:
                pass

        samples.append({
            "image": img_tensor,
            "original_size": (h, w),
            "organ_id": organ_id,
            "text_prompt": f"Cell nuclei in {organ_name.lower()} tissue.",
            "attribute_text": f"H&E-stained {organ_name.lower()} histopathology patch.",
            "morphology_text": f"H&E-stained {organ_name.lower()} histopathology patch.",
            "attr_labels": None,
        })
    return samples


# ==============================================================================
# 6. Comparison utilities
# ==============================================================================

def _unwrap_sam_output(out, name="output"):
    """Normalize SAM forward output from list[dict] to dict.

    TextSam.forward() returns List[Dict] (one dict per sample in the batch).
    When called with batch_input=[sample], the list has length 1.
    This helper unwraps the list safely and validates the result is a dict.
    """
    if isinstance(out, list):
        if len(out) == 0:
            raise RuntimeError(f"[COMPARE_OUTPUT_ERROR] {name} is empty list")
        if len(out) > 1:
            print(f"[COMPARE_OUTPUT_WARN] {name} is list with len={len(out)}; using first item")
        out = out[0]
    if not isinstance(out, dict):
        raise TypeError(f"[COMPARE_OUTPUT_ERROR] {name} expected dict or list[dict], got {type(out)}")
    return out


def compare_outputs(
    out_off: Any,
    out_on: Any,
    sample_id: int,
) -> Dict[str, float]:
    """Compare outputs from v3-off and v3-on models for a single sample."""
    metrics: Dict[str, float] = {"sample_id": float(sample_id)}

    # ── Normalize: SAM forward returns List[Dict]; unwrap first element ──
    out_off = _unwrap_sam_output(out_off, "out_off")
    out_on = _unwrap_sam_output(out_on, "out_on")

    # ── Print available keys for first few samples ──
    if sample_id < 5:
        off_keys = list(out_off.keys())
        on_keys = list(out_on.keys())
        print(f"[COMPARE_OUTPUT_KEYS] sample_id={sample_id} off_keys={off_keys}", flush=True)
        print(f"[COMPARE_OUTPUT_KEYS] sample_id={sample_id} on_keys={on_keys}", flush=True)

    # ── Resolve mask tensors: prefer low_res_logits, fallback to masks ──
    logits_off = out_off.get("low_res_logits", None)
    logits_on = out_on.get("low_res_logits", None)
    masks_off = out_off.get("masks", None)
    masks_on = out_on.get("masks", None)

    has_logits = logits_off is not None and logits_on is not None
    has_masks = masks_off is not None and masks_on is not None

    if has_logits:
        # ── low_res_logits comparison ──
        m_off = logits_off.detach().float()
        m_on = logits_on.detach().float()
        if m_off.dim() > 2:
            m_off = m_off.reshape(m_off.shape[0], -1)
        if m_on.dim() > 2:
            m_on = m_on.reshape(m_on.shape[0], -1)
        diff = (m_on - m_off)
        metrics["low_res_logits_l1"] = diff.abs().mean().item()
        metrics["low_res_logits_l2"] = (diff ** 2).mean().sqrt().item()
        metrics["low_res_logits_max_abs"] = diff.abs().max().item()

        # Prob mask from sigmoid(logits)
        prob_off = torch.sigmoid(logits_off.float())
        prob_on = torch.sigmoid(logits_on.float())
    elif has_masks:
        print(f"[COMPARE_OUTPUT_FALLBACK] sample_id={sample_id} low_res_logits missing; using masks", flush=True)
        metrics["low_res_logits_l1"] = float("nan")
        metrics["low_res_logits_l2"] = float("nan")
        metrics["low_res_logits_max_abs"] = float("nan")

        prob_off = masks_off.float()
        prob_on = masks_on.float()
    else:
        raise RuntimeError(
            f"[COMPARE_OUTPUT_ERROR] sample_id={sample_id}: "
            f"neither low_res_logits nor masks found in output dict. "
            f"off_keys={list(out_off.keys())} on_keys={list(out_on.keys())}"
        )

    # ── Prob mask comparison ──
    p_off = prob_off.reshape(prob_off.shape[0], -1)
    p_on = prob_on.reshape(prob_on.shape[0], -1)
    metrics["prob_mask_l1"] = (p_on - p_off).abs().mean().item()

    # ── Binary mask comparison ──
    bin_off = (prob_off > 0.5).float()
    bin_on = (prob_on > 0.5).float()
    diff_pixels = (bin_on != bin_off).float().sum().item()
    total_pixels = bin_off.numel()
    metrics["binary_mask_diff_pixels"] = diff_pixels
    metrics["binary_mask_diff_ratio"] = diff_pixels / max(total_pixels, 1)

    # ── v3-on diagnostics ──
    # Diagnostics are attached as flat keys in the output dict by TextSam.forward
    # (see sam.py lines 5634-5638). Use out_on directly.
    diag = out_on  # diagnostics are flat in the dict
    _get = lambda k, d=float("nan"): float(diag.get(k, d)) if not isinstance(diag.get(k, d), str) else float("nan")
    metrics["v3_on_active"] = _get("v3_active", 0.0)
    metrics["v3_on_skipped"] = _get("v3_skipped", 0.0)
    metrics["v3_on_text_delta_std"] = _get("promptnu_guided_v3_text_delta_std", 0.0)
    metrics["v3_on_additive_delta_norm"] = _get("promptnu_guided_v3_additive_delta_norm", 0.0)
    metrics["v3_on_semantic_delta_before_norm"] = _get("semantic_delta_before_v3_norm", 0.0)
    metrics["v3_on_semantic_delta_after_norm"] = _get("semantic_delta_after_v3_norm", 0.0)
    metrics["v3_on_injected_delta_norm"] = _get("injected_delta_norm", 0.0)
    metrics["v3_on_uses_guided_delta_for_injection"] = _get("uses_guided_delta_for_injection", 0.0)
    metrics["v3_on_skip_reason"] = float("nan")  # string, not included in summary

    # ── v3 injection ablation diagnostics ──
    metrics["v3_ablation_actual_inj_norm"] = _get("v3_ablation_actual_inj_norm", float("nan"))
    metrics["v3_ablation_v3_additive_delta_norm"] = _get("v3_ablation_v3_additive_delta_norm", float("nan"))
    metrics["v3_ablation_actual_to_default_ratio"] = _get("v3_ablation_actual_to_default_ratio", float("nan"))
    metrics["v3_ablation_original_sd_norm"] = _get("v3_ablation_original_sd_norm", float("nan"))
    metrics["v3_ablation_guided_sd_norm"] = _get("v3_ablation_guided_sd_norm", float("nan"))
    metrics["v3_ablation_gate_mean"] = _get("v3_ablation_gate_mean", float("nan"))
    metrics["v3_ablation_default_inj_norm"] = _get("v3_ablation_default_inj_norm", float("nan"))
    metrics["v3_ablation_uses_gate"] = _get("v3_ablation_uses_gate", float("nan"))
    metrics["v3_ablation_uses_post_gate_add"] = _get("v3_ablation_uses_post_gate_add", float("nan"))
    # Read injection ablation mode as string (stored directly)
    _mode_str = diag.get("v3_ablation_mode", "default")
    if isinstance(_mode_str, str):
        metrics["v3_ablation_mode_str"] = _mode_str
    else:
        metrics["v3_ablation_mode_str"] = "default"

    # ── PNuDP diagnostics (if enabled) ──
    metrics["pnudp_dense_feat_norm"] = _get("dense_feat_norm", float("nan"))
    metrics["pnudp_dense_text_logits_mean"] = _get("dense_text_logits_mean", float("nan"))
    metrics["pnudp_dense_text_logits_std"] = _get("dense_text_logits_std", float("nan"))
    metrics["pnudp_dense_text_logits_max"] = _get("dense_text_logits_max", float("nan"))
    metrics["pnudp_dense_text_entropy"] = _get("dense_text_entropy", float("nan"))

    # ── PNuDP dense diag metrics (new diagnostic module) ──
    # These are populated by _run_pnudp_dense_diag() and stored as flat keys.
    # We capture them from the v3-on output dict (they're attached as diagnostics).
    # NOTE: If logit_add fusion is active, the PNuDP metrics are also merged
    # from pnudp_metrics (set in main). Here we only capture diagnostics that
    # might exist in the model output dict (e.g. from legacy pnudp_diag).
    pnudp_dense_keys = [
        "pnudp_dense_dense_feat_norm",
        "pnudp_dense_text_bank_norm",
        "pnudp_dense_dense_text_logits_mean",
        "pnudp_dense_dense_text_logits_std",
        "pnudp_dense_dense_text_logits_max",
        "pnudp_dense_dense_text_logits_entropy",
        "pnudp_fusion_delta_norm",
        "pnudp_logit_add_prob_mask_l1",
        "pnudp_logit_add_binary_diff_ratio",
        "pnudp_logit_add_projected_std",
        "pnudp_fusion_feat_l1",
    ]
    for _pk in pnudp_dense_keys:
        if _pk not in metrics:
            _val = _get(_pk, float("nan"))
            if _val == 0.0 or (isinstance(_val, float) and (np.isnan(_val) or np.isinf(_val))):
                print(f"[PNUDP_METRIC_MISSING] key='{_pk}' not in out_on dict; got value={_val}", flush=True)
            metrics[_pk] = _val

    return metrics


def print_comparison(metrics: Dict[str, float]):
    """Print per-sample comparison metrics."""
    print("[COMPARE_V3_ON_OFF]", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}={v:.8e}", flush=True)
        else:
            print(f"  {k}={v}", flush=True)


def print_summary(all_metrics: List[Dict[str, float]]):
    """Print aggregated summary over all samples."""
    if len(all_metrics) == 0:
        return
    summary = {"num_samples": len(all_metrics)}
    # Numeric keys to average
    numeric_keys = [
        "low_res_logits_l1", "low_res_logits_l2", "low_res_logits_max_abs",
        "prob_mask_l1", "binary_mask_diff_pixels", "binary_mask_diff_ratio",
        "v3_on_additive_delta_norm", "v3_on_injected_delta_norm",
        "v3_on_semantic_delta_before_norm", "v3_on_semantic_delta_after_norm",
        "v3_on_text_delta_std",
        # v3 injection ablation metrics
        "v3_ablation_actual_inj_norm", "v3_ablation_v3_additive_delta_norm",
        "v3_ablation_actual_to_default_ratio",
        "v3_ablation_original_sd_norm", "v3_ablation_guided_sd_norm",
        "v3_ablation_gate_mean", "v3_ablation_default_inj_norm",
        "v3_ablation_uses_gate", "v3_ablation_uses_post_gate_add",
        # PNuDP metrics (legacy)
        "pnudp_dense_feat_norm", "pnudp_dense_text_logits_mean",
        "pnudp_dense_text_logits_std", "pnudp_dense_text_logits_max",
        "pnudp_dense_text_entropy",
        # PNuDP dense diag metrics (new diagnostic module)
        "pnudp_dense_dense_feat_norm", "pnudp_dense_text_bank_norm",
        "pnudp_dense_dense_text_logits_mean", "pnudp_dense_dense_text_logits_std",
        "pnudp_dense_dense_text_logits_max", "pnudp_dense_dense_text_logits_entropy",
        "pnudp_fusion_delta_norm", "pnudp_logit_add_prob_mask_l1",
        "pnudp_logit_add_binary_diff_ratio", "pnudp_logit_add_projected_std",
        "pnudp_fusion_feat_l1",
    ]
    for key in numeric_keys:
        vals = [m.get(key, float("nan")) for m in all_metrics]
        valid = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if len(valid) > 0:
            summary[f"mean_{key}"] = float(np.mean(valid))
        else:
            summary[f"mean_{key}"] = float("nan")

    # Count active/skipped
    active_vals = [m.get("v3_on_active", 0.0) for m in all_metrics]
    skipped_vals = [m.get("v3_on_skipped", 0.0) for m in all_metrics]
    summary["v3_on_active_count"] = sum(1 for v in active_vals if abs(v - 1.0) < 0.5)
    summary["v3_on_skipped_count"] = sum(1 for v in skipped_vals if abs(v - 1.0) < 0.5)

    print("[COMPARE_V3_ON_OFF_SUMMARY]", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}={v:.8e}", flush=True)
        else:
            print(f"  {k}={v}", flush=True)


def _print_device_audit(
    device: torch.device,
    model_off: torch.nn.Module,
    model_on: torch.nn.Module,
    batch_input: List[Dict[str, Any]],
):
    """Print device locations of key tensors for debugging."""
    # First param of each model
    off_param = next(model_off.parameters())
    on_param = next(model_on.parameters())

    # pixel_mean / pixel_std
    off_pm = model_off.pixel_mean if hasattr(model_off, "pixel_mean") else None
    off_ps = model_off.pixel_std if hasattr(model_off, "pixel_std") else None
    on_pm = model_on.pixel_mean if hasattr(model_on, "pixel_mean") else None
    on_ps = model_on.pixel_std if hasattr(model_on, "pixel_std") else None

    # input image
    input_img = batch_input[0].get("image", None) if batch_input else None

    print("[COMPARE_DEVICE_AUDIT]", flush=True)
    print(f"  device={device}", flush=True)
    print(f"  model_off_first_param_device={off_param.device}", flush=True)
    print(f"  model_on_first_param_device={on_param.device}", flush=True)
    print(f"  model_off_pixel_mean_device={off_pm.device if torch.is_tensor(off_pm) else 'N/A'}", flush=True)
    print(f"  model_off_pixel_std_device={off_ps.device if torch.is_tensor(off_ps) else 'N/A'}", flush=True)
    print(f"  model_on_pixel_mean_device={on_pm.device if torch.is_tensor(on_pm) else 'N/A'}", flush=True)
    print(f"  model_on_pixel_std_device={on_ps.device if torch.is_tensor(on_ps) else 'N/A'}", flush=True)
    print(f"  input_image_device={input_img.device if torch.is_tensor(input_img) else 'N/A'}", flush=True)

    # Sanity checks
    errors = []
    if off_param.device != device:
        errors.append(f"model_off params on {off_param.device}, expected {device}")
    if on_param.device != device:
        errors.append(f"model_on params on {on_param.device}, expected {device}")
    if torch.is_tensor(off_pm) and off_pm.device != device:
        errors.append(f"model_off.pixel_mean on {off_pm.device}, expected {device}")
    if torch.is_tensor(on_pm) and on_pm.device != device:
        errors.append(f"model_on.pixel_mean on {on_pm.device}, expected {device}")
    if torch.is_tensor(input_img) and input_img.device != device:
        errors.append(f"input_image on {input_img.device}, expected {device}")
    if errors:
        for e in errors:
            print(f"  [DEVICE_MISMATCH] {e}", flush=True)
    else:
        print(f"  [DEVICE_OK] All tensors on {device}", flush=True)


# ==============================================================================
# 7. PNuDP Dense Diagnostic Integration
# ==============================================================================

def _safe_get_metric(metrics: Dict[str, float], key: str, default: float = float("nan")) -> float:
    """Get a metric with [PNUDP_METRIC_MISSING] warning if key is absent."""
    if key not in metrics:
        print(f"[PNUDP_METRIC_MISSING] key='{key}' not found in metrics; returning {default}", flush=True)
        return default
    val = metrics[key]
    if val is None:
        print(f"[PNUDP_METRIC_MISSING] key='{key}' is None; returning {default}", flush=True)
        return default
    return float(val)


def _run_pnudp_dense_diag(
    model_off: TextSam,
    batch_input: List[Dict[str, Any]],
    args,
    device: torch.device,
    out_on: Any = None,
) -> tuple:
    """
    Run PNuDP dense diagnostic on the v3-off model output.

    This function:
      1. Captures spatial features from the model
      2. Builds text bank
      3. Computes dense similarity and diagnostics
      4. Applies fusion ablation (logit_add / feature_concat / film)
      5. Returns comparison metrics and (for logit_add) fused output dict

    Returns:
        (metrics_dict, fused_out_or_None)
            metrics: dict with pnudp_dense_* keys and fusion metrics
            fused_out: for logit_add, a dict like out_on with low_res_logits
                       replaced by fused logits; None for other fusion modes.
    """
    metrics: Dict[str, float] = {}
    fused_out: Any = None

    # ── 1. Build PNuDP dense diag module ──
    diag_module = build_pnudp_dense_diag(args, device)
    fusion_mode = str(getattr(args, "pnudp_dense_fusion_mode", "none")).strip().lower()
    text_source = str(getattr(args, "pnudp_text_source", "pred_attr")).strip().lower()
    feature_source = str(getattr(args, "pnudp_feature_source", "image_embedding")).strip().lower()
    alpha = float(getattr(args, "pnudp_dense_alpha", 0.1))
    constant_logit = getattr(args, "pnudp_dense_debug_constant_logit", None)
    if constant_logit is not None:
        constant_logit = float(constant_logit)
    project_mode = str(getattr(args, "pnudp_dense_project_mode", "zero_conv")).strip().lower()
    project_eps = float(getattr(args, "pnudp_dense_project_eps", 1e-6))

    # ── 2. Build text bank ──
    text_bank = build_text_bank(
        text_source=text_source,
        model=model_off,
        device=device,
        num_prompts=NUM_SB_PROMPTS,
        text_dim=512,
    )

    # ── 3. Capture spatial features via forward hooks ──
    image_emb_list: List[torch.Tensor] = []
    img_enc_handle = model_off.image_encoder.register_forward_hook(
        lambda m, i, o: image_emb_list.append(o.detach().float())
    )

    decoder_feat_list: List[torch.Tensor] = []
    md = model_off.mask_decoder
    _decoder_hook_handle = None
    if md.use_asr and hasattr(md, "asr_upscale_2"):
        _decoder_hook_handle = md.asr_upscale_2.register_forward_hook(
            lambda m, i, o: decoder_feat_list.append(
                o.detach().float() if torch.is_tensor(o) else o[0].detach().float()
            )
        )
    elif not md.use_asr and hasattr(md, "output_upscaling"):
        _last_upscale = md.output_upscaling[-1]
        _decoder_hook_handle = _last_upscale.register_forward_hook(
            lambda m, i, o: decoder_feat_list.append(o.detach().float())
        )

    with torch.no_grad():
        _ = model_off(batch_input, multimask_output=True)

    img_enc_handle.remove()
    if _decoder_hook_handle is not None:
        _decoder_hook_handle.remove()

    # ── 4. Select spatial feature ──
    if feature_source == "image_embedding":
        if len(image_emb_list) == 0:
            print("[PNUDP_DENSE_DIAG] ERROR: image_embeddings not captured", flush=True)
            return metrics, None
        spatial_feat = image_emb_list[0]
    elif feature_source == "decoder_upscaled_feature":
        if len(decoder_feat_list) == 0:
            print("[PNUDP_DENSE_DIAG] ERROR: decoder upscaled feature not captured", flush=True)
            return metrics, None
        spatial_feat = decoder_feat_list[0]
    else:
        print(f"[PNUDP_DENSE_DIAG] ERROR: unknown feature_source={feature_source}", flush=True)
        return metrics, None

    if spatial_feat.device != device:
        spatial_feat = spatial_feat.to(device)

    # ── 5. Run PNuDP dense diag ──
    with torch.no_grad():
        dense_text_logits, fused_feat, diagnostics = diag_module(
            spatial_feat=spatial_feat,
            text_bank=text_bank,
            fusion_mode=fusion_mode,
        )

    for k, v in diagnostics.items():
        if isinstance(v, (float, int)):
            metrics[f"pnudp_dense_{k}"] = float(v)

    # ── 6. Apply fusion and measure effect on mask logits ──
    if fusion_mode in ("feature_concat", "film"):
        _apply_feature_fusion_and_rerun_decoder(
            model_off, batch_input, fused_feat, spatial_feat, metrics,
            device, feature_source, fusion_mode, alpha,
        )
    elif fusion_mode == "logit_add":
        pnudp_metrics, fused_out = _apply_logit_add_fusion(
            model_off, batch_input, dense_text_logits, diag_module, metrics,
            device, alpha, out_on=out_on, constant_logit=constant_logit,
            project_mode=project_mode, project_eps=project_eps,
        )
        metrics.update(pnudp_metrics)
    else:
        metrics["pnudp_fusion_delta_norm"] = 0.0

    # ── 7. Print structured diagnostic ──
    print_pnudp_dense_diag(
        diag=diagnostics,
        feature_source=feature_source,
        fusion_mode=fusion_mode,
        text_source=text_source,
        alpha=alpha,
    )

    return metrics, fused_out


def _apply_logit_add_fusion(
    model_off: TextSam,
    batch_input: List[Dict[str, Any]],
    dense_text_logits: torch.Tensor,
    diag_module: PromptNuDenseDiag,
    metrics: Dict[str, float],
    device: torch.device,
    alpha: float,
    out_on: Any = None,
    constant_logit: Optional[float] = None,
    project_mode: str = "zero_conv",
    project_eps: float = 1e-6,
) -> tuple:
    """
    Apply logit_add fusion on top of out_on's low_res_logits.

    fused_logits = out_on["low_res_logits"] + alpha * projected_dense_text_logits

    Supports deterministic projection modes via project_mode argument.

    Returns:
        (fusion_metrics_dict, fused_out_dict)
            fusion_metrics: dict with pnudp_fusion_delta_norm, prob_mask_l1, etc.
            fused_out: copy of out_on with low_res_logits replaced by fused_logits.
    """
    fusion_metrics: Dict[str, float] = {}

    # ── Get base logits from out_on (v3-on output) ──
    if out_on is None:
        print("[PNUDP_LOGIT_ADD_ERROR] out_on is None; cannot apply logit_add fusion", flush=True)
        return fusion_metrics, None

    _out_on = _unwrap_sam_output(out_on, "out_on")
    logits_base = _out_on.get("low_res_logits", None)
    if logits_base is None:
        print("[PNUDP_LOGIT_ADD_ERROR] low_res_logits not found in out_on", flush=True)
        return fusion_metrics, None

    # ── Project dense_text_logits to mask logit shape ──
    projected = diag_module.project_dense_logits_to_mask(
        dense_text_logits,
        project_mode=project_mode,
        eps=project_eps,
    )  # [B, 1, H, W]

    # Resize projected to match logits_base spatial dims if needed
    if projected.shape[-2:] != logits_base.shape[-2:]:
        projected = F.interpolate(
            projected,
            size=logits_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # ── Debug: constant logit override ──
    if constant_logit is not None:
        print(f"[PNUDP_LOGIT_ADD_DEBUG] Overriding projected with constant {constant_logit}", flush=True)
        projected = torch.ones_like(logits_base.float()) * constant_logit
        project_mode = "constant_override"

    # ── Compute fusion delta ──
    fusion_delta = alpha * projected.float()  # [B, 1, H, W]
    fusion_delta_norm = float(fusion_delta.norm().item() / max(fusion_delta.numel(), 1))
    projected_std = float(projected.float().std().item())

    # ── Compute fusion metrics (before error check, so they're available) ──
    logits_fused = logits_base.float() + fusion_delta

    prob_base = torch.sigmoid(logits_base.float())
    prob_fused = torch.sigmoid(logits_fused)
    p_base = prob_base.reshape(prob_base.shape[0], -1)
    p_fused = prob_fused.reshape(prob_fused.shape[0], -1)
    prob_mask_l1 = float((p_fused - p_base).abs().mean().item())

    bin_base = (prob_base > 0.5).float()
    bin_fused = (prob_fused > 0.5).float()
    diff_pixels = (bin_fused != bin_base).float().sum().item()
    total_pixels = bin_base.numel()
    binary_diff_ratio = diff_pixels / max(total_pixels, 1)

    fusion_metrics["pnudp_fusion_delta_norm"] = fusion_delta_norm
    fusion_metrics["pnudp_logit_add_prob_mask_l1"] = prob_mask_l1
    fusion_metrics["pnudp_logit_add_binary_diff_ratio"] = binary_diff_ratio
    fusion_metrics["pnudp_logit_add_projected_std"] = projected_std

    # ── [PNUDP_LOGIT_ADD_AUDIT] structured print via shared helper ──
    print_pnudp_logit_add_audit(
        project_mode=project_mode,
        alpha=alpha,
        dense_text_logits=dense_text_logits,
        projected=projected,
        logits_base=logits_base,
        logits_fused=logits_fused,
        fusion_delta_norm=fusion_delta_norm,
        prob_mask_l1=prob_mask_l1,
        binary_diff_ratio=binary_diff_ratio,
    )

    # ── [PNUDP_PROJECT_ERROR] check ──
    # If project_mode != zero_conv (deterministic mode), projected should have
    # non-zero std and non-zero fusion_delta_norm. Zero values indicate a bug.
    _is_deterministic = (project_mode != "zero_conv") and (constant_logit is None)
    if _is_deterministic and (projected_std == 0.0 or fusion_delta_norm == 0.0):
        print("[PNUDP_PROJECT_ERROR]", flush=True)
        print(
            f"  project_mode={project_mode} projected_std={projected_std:.8e} "
            f"fusion_delta_norm={fusion_delta_norm:.8e}",
            flush=True,
        )
        print(
            f"  ERROR: Deterministic projection produced zero-variance output. "
            f"Check dense_text_logits: shape={list(dense_text_logits.shape)} "
            f"std={float(dense_text_logits.float().std().item()):.8e} "
            f"mean={float(dense_text_logits.float().mean().item()):.8e}",
            flush=True,
        )

    # ── Create fused output dict ──
    fused_out = dict(_out_on)
    fused_out["low_res_logits"] = logits_fused.to(dtype=_out_on.get("low_res_logits", logits_base).dtype)

    if "masks" in _out_on:
        fused_out["masks"] = (prob_fused > 0.5).float().to(dtype=_out_on["masks"].dtype)

    return fusion_metrics, fused_out


def _apply_feature_fusion_and_rerun_decoder(
    model_off: TextSam,
    batch_input: List[Dict[str, Any]],
    fused_feat: torch.Tensor,
    original_feat: torch.Tensor,
    metrics: Dict[str, float],
    device: torch.device,
    feature_source: str,
    fusion_mode: str,
    alpha: float,
):
    """
    For feature_concat or film fusion: measure the effect of feature-space
    PNuDP fusion on mask logits by comparing original vs fused decoder outputs.

    Simplified approach: Instead of re-running the full model, we measure
    the delta between fused and original features as a proxy, and attempt
    to compute the effect on decoder output.

    NOTE: Full decoder re-run with fused features requires careful handling
    of the image_encoder output dimensions, which may differ between
    image_embedding (256d) and decoder_upscaled_feature (32d).
    """
    # Measure feature-level delta as a diagnostic
    feat_diff = (fused_feat.float() - original_feat.float())
    metrics["pnudp_fusion_delta_norm"] = float(
        feat_diff.norm().item() / max(feat_diff.numel(), 1)
    )
    metrics["pnudp_fusion_feat_l1"] = float(feat_diff.abs().mean().item())

    # If the feature source is decoder_upscaled_feature, we can re-run the
    # decoder forward to get the actual mask logits change.
    if feature_source == "decoder_upscaled_feature":
        _rerun_decoder_with_fused_feat(
            model_off, batch_input, fused_feat, metrics, device,
        )


def _rerun_decoder_with_fused_feat(
    model_off: TextSam,
    batch_input: List[Dict[str, Any]],
    fused_upscaled_feat: torch.Tensor,
    metrics: Dict[str, float],
    device: torch.device,
):
    """
    Re-run the mask decoder forward with a modified upscaled feature.

    The decoder's predict_masks method:
      1. Runs transformer (we skip this, keeping original tokens)
      2. Upscales src via ASR blocks -> upscaled_embedding
      3. Computes masks = hyper_in @ upscaled_embedding

    We replace step 2's upscaled_embedding with our fused version.
    This requires patching the decoder temporarily.
    """
    # This is complex. For now, we log that a full re-run is needed.
    print(
        f"[PNUDP_DENSE_DIAG] Feature fusion mode={model_off.pnudp_dense_fusion_mode} "
        f"applied at decoder_upscaled_feature level. "
        f"Feature delta norm={metrics.get('pnudp_fusion_delta_norm', 'N/A')}. "
        f"Full mask re-run requires decoder patch.",
        flush=True,
    )
    metrics["pnudp_decoder_rerun_status"] = 0.0  # 0 = not re-run


# ==============================================================================
# 8. Main
# ==============================================================================
def main():
    args = parse_args()

    # ── HF offline ──
    if args.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.conch_cache_path is not None:
        os.environ["HF_HOME"] = args.conch_cache_path
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.conch_cache_path

    # ── Unified device ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[COMPARE_V3_ON_OFF] device={device}", flush=True)

    # ── Build base args (shared config) ──
    base_ns = _build_base_args(args)

    # ── Build model_off (v3 guided disabled) ──
    print("[COMPARE_V3_ON_OFF] Building model_off (enable_promptnu_guided_v3=False)...", flush=True)
    model_off = _build_model(base_ns, enable_v3=False)
    model_off = load_checkpoint(model_off, args.checkpoint, device)
    model_off = _ensure_sam_device(model_off, device)
    model_off.eval()
    print(f"  model_off built, param_count={sum(p.numel() for p in model_off.parameters()):,}", flush=True)

    # ── Build model_on (v3 guided enabled) ──
    print("[COMPARE_V3_ON_OFF] Building model_on (enable_promptnu_guided_v3=True)...", flush=True)
    model_on = _build_model(base_ns, enable_v3=True)
    model_on = load_checkpoint(model_on, args.checkpoint, device)
    model_on = _ensure_sam_device(model_on, device)
    model_on.eval()
    print(f"  model_on built, param_count={sum(p.numel() for p in model_on.parameters()):,}", flush=True)

    # ==============================================================
    # [TEST_INFERENCE_MODE] Disable training-only forward paths
    # ==============================================================
    # After building attr_align heads and loading checkpoint weights,
    # disable enable_attr_text_alignment and enable_promptnu_lite_align
    # so that test forward does not need GT structure/boundary labels.
    # The attr_align modules remain built and loaded; only the forward
    # branch that calls _get_attr_text_embeddings() is suppressed.
    # This follows the same pattern as test.py line 1127-1133.
    # ==============================================================
    _disable_forward = getattr(args, "disable_attr_text_alignment_forward_in_test", True)
    if _disable_forward:
        _attr_before_off = getattr(model_off, "enable_attr_text_alignment", False)
        _attr_before_on = getattr(model_on, "enable_attr_text_alignment", False)
        _pnurl_before_off = getattr(model_off, "enable_promptnu_lite_align", False)
        _pnurl_before_on = getattr(model_on, "enable_promptnu_lite_align", False)

        if _attr_before_off:
            model_off.enable_attr_text_alignment = False
        if _attr_before_on:
            model_on.enable_attr_text_alignment = False
        if _pnurl_before_off:
            model_off.enable_promptnu_lite_align = False
        if _pnurl_before_on:
            model_on.enable_promptnu_lite_align = False

        print("[TEST_INFERENCE_MODE]", flush=True)
        print(f"  model_off: enable_attr_text_alignment was {_attr_before_off}, set to False", flush=True)
        print(f"  model_off: enable_promptnu_lite_align was {_pnurl_before_off}, set to False", flush=True)
        print(f"  model_on:  enable_attr_text_alignment was {_attr_before_on}, set to False", flush=True)
        print(f"  model_on:  enable_promptnu_lite_align was {_pnurl_before_on}, set to False", flush=True)

    # ==============================================================
    # [COMPARE_FORWARD_MODE] Forward path audit
    # ==============================================================
    print("[COMPARE_FORWARD_MODE]", flush=True)
    print(f"  enable_attr_text_alignment_model_off={getattr(model_off, 'enable_attr_text_alignment', False)}", flush=True)
    print(f"  enable_attr_text_alignment_model_on={getattr(model_on, 'enable_attr_text_alignment', False)}", flush=True)
    print(f"  disable_attr_text_alignment_forward_in_test={_disable_forward}", flush=True)
    print(f"  enable_promptnu_guided_v3_off_model={getattr(model_off, 'enable_promptnu_guided_v3', False)}", flush=True)
    print(f"  enable_promptnu_guided_v3_on_model={getattr(model_on, 'enable_promptnu_guided_v3', False)}", flush=True)
    print(f"  v3_guidance_should_run_for_off=False (v3 off)", flush=True)
    print(f"  v3_guidance_should_run_for_on={getattr(model_on, 'enable_promptnu_guided_v3', False)}", flush=True)
    # NOTE: model_off (v3 disabled) output will NOT contain v3 diagnostic keys
    # (v3_active, v3_skipped, promptnu_guided_v3_text_scale_*, etc.)
    # because those keys are only added when enable_promptnu_guided_v3=True.
    # The compare_outputs() function reads diagnostics ONLY from out_on (line 428).
    # Any v3 keys visible in model_off output would indicate a wiring bug.
    # v3 guidance (line 3917-3921) is gated by enable_promptnu_guided_v3, NOT enable_attr_text_alignment
    print(f"  phase_c_text_alignment_should_run=False (disabled by _disable_forward)", flush=True)
    # PromptNu-lite v2 (line 4929-4934) is gated by enable_promptnu_lite_align AND enable_attr_text_alignment
    # Both are now False, so PromptNu-lite is also disabled.
    print(f"  promptnu_lite_align_model_off={getattr(model_off, 'enable_promptnu_lite_align', False)}", flush=True)
    print(f"  promptnu_lite_align_model_on={getattr(model_on, 'enable_promptnu_lite_align', False)}", flush=True)
    print(f"  promptnu_lite_should_run=False (disabled)", flush=True)
    # v3.2 cosine alignment loss (line 4195-4201) is gated by self.training — model is in eval mode
    print(f"  model_off_training={model_off.training}", flush=True)
    print(f"  model_on_training={model_on.training}", flush=True)
    print(f"  v3_cosine_align_loss_should_run=False (eval mode)", flush=True)

    # ── PNuDP Dense Diag args ──
    _pnudp_dense_enabled = getattr(args, "enable_pnudp_dense_diag", False)
    if _pnudp_dense_enabled:
        print("[PNUDP_DENSE_DIAG] PNuDP dense diagnostic ENABLED", flush=True)
        print(f"  fusion_mode={getattr(args, 'pnudp_dense_fusion_mode', 'none')}", flush=True)
        print(f"  alpha={getattr(args, 'pnudp_dense_alpha', 0.1)}", flush=True)
        print(f"  text_source={getattr(args, 'pnudp_text_source', 'pred_attr')}", flush=True)
        print(f"  feature_source={getattr(args, 'pnudp_feature_source', 'image_embedding')}", flush=True)
        print(f"  project_mode={getattr(args, 'pnudp_dense_project_mode', 'zero_conv')}", flush=True)
        print(f"  project_eps={getattr(args, 'pnudp_dense_project_eps', 1e-6)}", flush=True)
        print(f"  NOTE: Diagnostic only. No training. No full test.", flush=True)

    # ── Load test images ──
    print(f"[COMPARE_V3_ON_OFF] Loading up to {args.num_samples} images from {args.data_path}...", flush=True)
    samples = load_test_images(args.data_path, args.num_samples, args.image_size)
    print(f"  Loaded {len(samples)} samples", flush=True)

    if len(samples) == 0:
        print("[COMPARE_V3_ON_OFF] No samples loaded, exiting.", flush=True)
        return

    # ── Compare ──
    all_metrics: List[Dict[str, float]] = []

    # ── Device audit on first sample ──
    first_batch = _move_batch_to_device([samples[0]], device)
    _print_device_audit(device, model_off, model_on, first_batch)

    # ── PNuDP dense diag state ──
    _pnudp_dense_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            # Single-sample batch, moved to device
            batch_input = _move_batch_to_device([sample], device)

            # Forward through model_off (v3 disabled)
            out_off = model_off(batch_input, multimask_output=True)
            # Forward through model_on (v3 enabled)
            out_on_raw = model_on(batch_input, multimask_output=True)

            # ── Default: compare off vs on (no PNuDP fusion) ──
            out_on_for_compare = out_on_raw
            pnudp_metrics: Dict[str, float] = {}

            # ── PNuDP Dense Diag: run on model_off (base features) ──
            if _pnudp_dense_enabled:
                pnudp_metrics, out_on_fused = _run_pnudp_dense_diag(
                    model_off=model_off,
                    batch_input=batch_input,
                    args=args,
                    device=device,
                    out_on=out_on_raw,
                )
                # If logit_add fusion produced a fused output, use it for comparison
                _fusion_mode = str(getattr(args, "pnudp_dense_fusion_mode", "none")).strip().lower()
                if _fusion_mode == "logit_add" and out_on_fused is not None:
                    out_on_for_compare = out_on_fused
                    print(
                        f"[COMPARE_V3_ON_OFF] sample_id={i}: using PNuDP logit_add fused output "
                        f"for comparison (alpha={getattr(args, 'pnudp_dense_alpha', 0.1):.4f})",
                        flush=True,
                    )
                _pnudp_dense_metrics.append(pnudp_metrics)

            # Compare
            metrics = compare_outputs(out_off, out_on_for_compare, sample_id=i)
            # Merge PNuDP metrics into main metrics
            metrics.update(pnudp_metrics)

            all_metrics.append(metrics)
            print_comparison(metrics)

    # ── Summary ──
    print_summary(all_metrics)
    if _pnudp_dense_enabled and len(_pnudp_dense_metrics) > 0:
        _pnudp_summary = {"num_samples": len(_pnudp_dense_metrics)}
        _pnudp_keys = [
            "pnudp_dense_dense_feat_norm", "pnudp_dense_text_bank_norm",
            "pnudp_dense_dense_text_logits_mean", "pnudp_dense_dense_text_logits_std",
            "pnudp_dense_dense_text_logits_max", "pnudp_dense_dense_text_logits_entropy",
            "pnudp_fusion_delta_norm", "pnudp_logit_add_prob_mask_l1",
            "pnudp_logit_add_binary_diff_ratio", "pnudp_logit_add_projected_std",
            "pnudp_fusion_feat_l1",
        ]
        for _pk in _pnudp_keys:
            _vals = [m.get(_pk, float("nan")) for m in _pnudp_dense_metrics]
            _valid = [v for v in _vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
            if len(_valid) > 0:
                _pnudp_summary[f"mean_{_pk}"] = float(np.mean(_valid))
            else:
                _pnudp_summary[f"mean_{_pk}"] = float("nan")
        print("[PNUDP_DENSE_DIAG_SUMMARY]", flush=True)
        for _k, _v in _pnudp_summary.items():
            if isinstance(_v, float):
                print(f"  {_k}={_v:.8e}", flush=True)
            else:
                print(f"  {_k}={_v}", flush=True)

    print("[COMPARE_V3_ON_OFF] Done.", flush=True)


if __name__ == "__main__":
    main()
