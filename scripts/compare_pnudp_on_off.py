#!/usr/bin/env python3
"""
compare_pnudp_on_off.py

Same-checkpoint PNuDP Dense Train on/off comparison (single-model toggle).

Uses a SINGLE model built with enable_pnudp_dense_train=True. The PNuDP fusion
is toggled via model.pnudp_dense_apply_in_eval:

  - pnudp_dense_apply_in_eval=False → base logits (no PNuDP fusion in eval mode)
  - pnudp_dense_apply_in_eval=True  → fused logits (PNuDP fusion applied in eval mode)

Since the model now supports pnudp_dense_apply_in_eval=True, there is NO need
to manually compute bias via forward hooks. The model's own forward method
handles fusion when pnudp_dense_apply_in_eval=True is set.

Output sections:
  [PNUDP_COMPARE_LOAD_AUDIT]   – checkpoint loading audit
  [PNUDP_COMPARE_OUTPUT_AUDIT]  – output key verification (fused vs base)
  [PNUDP_DENSE_TRAIN_COMPARE]   – per-sample comparison metrics
  [PNUDP_DENSE_TRAIN_COMPARE_SUMMARY] – aggregated summary

Usage:
    python scripts/compare_pnudp_on_off.py \
        --checkpoint workdir/models/pnudp_dense_train_1ep_v1/best_aji_model.pth \
        --data_path data/MoNuSeg/test \
        --num_samples 8

Dependencies:
    - Same project imports as compare_v3_on_off_inference.py
    - Requires trained Stage D checkpoint with pnudp_dense_train.* keys
"""

import argparse
import os
import sys
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Project imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
from segment_anything.build_sam import build_sam_vit_b
from training.pnudp_dense_diag import NUM_SB_PROMPTS


def str2bool(v):
    """Convert a string or bool to a boolean value for argparse.

    Supports --flag and --flag True / --flag False styles.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ==============================================================================
# 1. Argument parser
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PNuDP Dense Train on/off inference (single-model toggle)."
    )

    # ── Core paths ──
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the Stage D checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test image directory")

    # ── Model architecture ──
    parser.add_argument("--model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type (default: vit_b)")

    # ── Test args ──
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
    parser.add_argument("--enable_attr_text_alignment", action="store_true", default=False)
    parser.add_argument("--enable_multilevel_attr_heads", action="store_true", default=True)
    parser.add_argument("--enable_structure_boundary_attr_heads", action="store_true", default=False)

    # ── ASR ──
    parser.add_argument("--use_asr", action="store_true", default=True)
    parser.add_argument("--asr_variant", type=str, default="freqpath",
                        choices=["legacy", "freqpath"])

    # ── HF / CONCH ──
    parser.add_argument("--hf_hub_offline", action="store_true", default=False)
    parser.add_argument("--conch_cache_path", type=str, default=None)

    # ── PromptNu-guided v3 (must match checkpoint architecture) ──
    parser.add_argument("--promptnu_guided_v3_use_text_bank", action="store_true", default=True)
    parser.add_argument("--promptnu_guided_v3_prompt_source", type=str, default="pred_attr",
                        choices=["pred_attr", "fixed_global", "uniform_bank", "oracle_gt_attr"])
    parser.add_argument("--promptnu_guided_v3_guidance_mode", type=str, default="scale_add",
                        choices=["scale", "additive", "scale_add"])
    parser.add_argument("--promptnu_guided_v3_scale_weight", type=float, default=0.05)
    parser.add_argument("--promptnu_guided_v3_delta_weight", type=float, default=0.001)
    parser.add_argument("--promptnu_guided_v3_max_guided_delta_ratio", type=float, default=0.02)
    parser.add_argument("--promptnu_guided_v3_injection_ablation", type=str, default="default",
                        choices=["default", "bypass_gate", "post_gate_add", "replace_semantic_delta"])
    parser.add_argument("--promptnu_guided_v3_post_gate_alpha", type=float, default=1.0)

    # ── Enable v3 ──
    parser.add_argument("--enable_promptnu_guided_v3", action="store_true", default=True)

    # ── PNuDP Dense Training (Stage D) args ──
    parser.add_argument("--enable_pnudp_dense_train", nargs="?", const=True, default=False, type=str2bool,
                        help="Enable PNuDP Dense Training (Stage D). Supports --flag and --flag True/False.")
    parser.add_argument("--pnudp_dense_fusion_mode", type=str, default="logit_add",
                        help="PNuDP dense fusion mode (default: logit_add)")
    parser.add_argument("--pnudp_dense_project_mode", type=str, default="zero_conv",
                        help="PNuDP dense project mode (default: zero_conv)")
    parser.add_argument("--pnudp_dense_alpha_init", type=float, default=0.05,
                        help="PNuDP dense alpha init (default: 0.05)")
    parser.add_argument("--pnudp_dense_logit_proj_init", type=str, default="zero",
                        choices=["zero", "normal", "mean"],
                        help="PNuDP dense logit proj init (default: zero)")
    parser.add_argument("--pnudp_dense_logit_proj_init_std", type=float, default=1.0,
                        help="PNuDP dense logit proj init std (default: 1.0)")
    parser.add_argument("--pnudp_dense_num_mask_channels", type=int, default=1,
                        help="PNuDP dense num mask channels (default: 1). "
                             "When >1 (e.g., 3), produces channel-specific bias.")

    # ── PNuDP dense apply in eval (this script toggles this at runtime) ──
    # Default False; the script manually sets it per-forward-pass.
    parser.add_argument("--pnudp_dense_apply_in_eval", type=str2bool, default=False,
                        help="Ignored; this script toggles pnudp_dense_apply_in_eval at runtime.")

    # ── PNuDP eval scale: amplify alpha for test-time eval scale sweep ──
    parser.add_argument("--pnudp_eval_scale", type=float, default=1.0,
                        help="Eval-time alpha multiplier for PNuDP bias fusion (default: 1.0)")

    # ── Misc ──
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--use_multimodal_prompt", action="store_true", default=False)
    parser.add_argument("--semantic_gate_bias_init", type=float, default=None)
    parser.add_argument("--enable_conch_text_encoder", action="store_true", default=True)
    parser.add_argument("--encoder_adapter", action="store_true", default=True)
    parser.add_argument("--max_semantic_gate", type=float, default=0.10)
    parser.add_argument("--init_delta_ratio", type=float, default=0.005)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--semantic_injection_scale", type=float, default=1.0)
    parser.add_argument("--sb_guidance_mode", type=str, default="none")
    parser.add_argument("--sb_prompt_template_path", type=str,
                        default="workdir/attr_stats/structure_boundary_prompt_templates.json")
    parser.add_argument("--use_coop", action="store_true", default=False)

    # ── Test-only attr-text alignment disable ──
    parser.add_argument("--disable_attr_text_alignment_forward_in_test", action="store_true", default=True)

    return parser.parse_args()


# ==============================================================================
# 2. Model building helpers
# ==============================================================================
def _build_base_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build a SimpleNamespace with all params for model building.

    The model is built with enable_pnudp_dense_train=True so the PNuDP dense
    module exists. The apply_in_eval flag is toggled at runtime.
    """
    ns = SimpleNamespace(**vars(args))
    # Force enable_pnudp_dense_train=True so the module is built
    ns.enable_pnudp_dense_train = True
    # apply_in_eval defaults to False; we toggle it at runtime
    ns.pnudp_dense_apply_in_eval = False
    return ns


def _build_model(args_ns: SimpleNamespace) -> TextSam:
    """Build a single TextSam with enable_pnudp_dense_train=True."""
    return build_sam_vit_b(args_ns)


# ==============================================================================
# 3. Device / data helpers (adapted from compare_v3_on_off_inference.py)
# ==============================================================================
def _ensure_sam_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
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
# 4. Checkpoint loading (single-model, improved audit)
# ==============================================================================
def load_checkpoint(model: TextSam, ckpt_path: str, device: torch.device):
    """Load checkpoint and print comprehensive [PNUDP_COMPARE_LOAD_AUDIT].

    Since a single model is used (with pnudp_dense_train=True module always
    built), the audit checks that the checkpoint actually contains the required
    PNuDP dense keys and that they are properly loaded.

    Args:
        model: TextSam instance (enable_pnudp_dense_train=True) to load into.
        ckpt_path: Path to .pth checkpoint.
        device: Torch device.

    Returns:
        model with loaded state_dict.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # ── PNuDP Dense Train key audit ──
    pnudp_ckpt_keys = sorted([k for k in state_dict.keys() if "pnudp_dense_train" in k])
    pnudp_loaded_keys = sorted([k for k in pnudp_ckpt_keys if k not in missing])
    has_pnudp_module = getattr(model, "pnudp_dense_train", None) is not None

    # Required keys that MUST be loaded from checkpoint
    _required_pnudp_keys = [
        "pnudp_dense_train.dense_alpha",
        "pnudp_dense_train.proj.weight",
        "pnudp_dense_train.logit_proj.weight",
    ]
    _missing_required = [k for k in _required_pnudp_keys if k not in pnudp_loaded_keys]

    print(f"[PNUDP_COMPARE_LOAD_AUDIT]", flush=True)
    print(f"  model_enable_pnudp_dense_train={getattr(model, 'enable_pnudp_dense_train', False)}", flush=True)
    print(f"  checkpoint_pnudp_keys_found={len(pnudp_ckpt_keys)}", flush=True)
    print(f"  checkpoint_pnudp_keys={pnudp_ckpt_keys}", flush=True)
    print(f"  loaded_pnudp_keys={pnudp_loaded_keys}", flush=True)
    print(f"  has_pnudp_module={has_pnudp_module}", flush=True)
    print(f"  missing_keys_count={len(missing)}", flush=True)
    print(f"  unexpected_keys_count={len(unexpected)}", flush=True)

    if _missing_required:
        print(f"  [WARN] Missing required PNuDP dense keys: {_missing_required}", flush=True)
        print(f"  [WARN] PNuDP fusion will have NO effect (bias=0).", flush=True)
    else:
        print(f"  [OK] All required PNuDP dense keys loaded (dense_alpha, proj.weight, logit_proj.weight)", flush=True)

    # Print non-pnudp missing/unexpected for debugging
    non_pnudp_missing = [k for k in missing if "pnudp_dense_train" not in k]
    non_pnudp_unexpected = [k for k in unexpected if "pnudp_dense_train" not in k]
    if len(non_pnudp_missing) > 0:
        print(f"  [WARN] Non-PNuDP missing keys ({len(non_pnudp_missing)}): {non_pnudp_missing[:10]}...", flush=True)
    if len(non_pnudp_unexpected) > 0:
        print(f"  [WARN] Non-PNuDP unexpected keys ({len(non_pnudp_unexpected)}): {non_pnudp_unexpected[:10]}...", flush=True)

    return model


# ==============================================================================
# 5. Load test images (from compare_v3_on_off_inference.py)
# ==============================================================================
def load_test_images(data_path: str, num_samples: int, image_size: int) -> List[Dict[str, Any]]:
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
        h, w = image_rgb.shape[:2]
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

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
# 6. Compare outputs
# ==============================================================================
def compare_outputs(
    logits_off: torch.Tensor,
    logits_on: torch.Tensor,
    sample_id: int,
    pnudp_debug: Dict[str, float],
) -> Dict[str, float]:
    """Compare outputs from off (no fusion) and on (fusion) runs."""
    metrics: Dict[str, float] = {"sample_id": float(sample_id)}

    # ── low_res_logits comparison ──
    m_off = logits_off.detach().float()
    m_on = logits_on.detach().float()
    if m_off.dim() > 2:
        m_off = m_off.reshape(m_off.shape[0], -1)
    if m_on.dim() > 2:
        m_on = m_on.reshape(m_on.shape[0], -1)
    diff = m_on - m_off
    metrics["low_res_logits_l1"] = float(diff.abs().mean().item())
    metrics["low_res_logits_l2"] = float((diff ** 2).mean().sqrt().item())
    metrics["low_res_logits_max_abs"] = float(diff.abs().max().item())

    # ── Prob mask comparison ──
    prob_off = torch.sigmoid(logits_off.float())
    prob_on = torch.sigmoid(logits_on.float())
    p_off = prob_off.reshape(prob_off.shape[0], -1)
    p_on = prob_on.reshape(prob_on.shape[0], -1)
    metrics["prob_mask_l1"] = float((p_on - p_off).abs().mean().item())

    # ── Binary mask comparison ──
    bin_off = (prob_off > 0.5).float()
    bin_on = (prob_on > 0.5).float()
    diff_pixels = float((bin_on != bin_off).float().sum().item())
    total_pixels = float(bin_off.numel())
    metrics["binary_mask_diff_pixels"] = diff_pixels
    metrics["binary_mask_diff_ratio"] = diff_pixels / max(total_pixels, 1.0)

    # ── PNuDP debug metrics (from model's own forward) ──
    for k, v in pnudp_debug.items():
        metrics[k] = v

    # pnudp_active flag (check if bias had non-zero effect)
    _fused_l1 = pnudp_debug.get("fused_minus_base_l1", 0.0)
    metrics["pnudp_active"] = 1.0 if _fused_l1 > 1e-12 else 0.0

    return metrics


def print_comparison(metrics: Dict[str, float]):
    """Print per-sample comparison metrics with [PNUDP_DENSE_TRAIN_COMPARE] tag.

    Format matches the required fields:
      num_samples, low_res_logits_l1, prob_mask_l1, binary_mask_diff_ratio,
      fused_minus_base_l1, pnudp_bias_abs_mean, pnudp_bias_abs_max, pnudp_bias_std,
      alpha, dense_text_logits_mean, dense_text_logits_std
    """
    _sid = metrics.get("sample_id", -1)
    print(f"[PNUDP_DENSE_TRAIN_COMPARE] sample_id={int(_sid)}", flush=True)
    _keys = [
        "num_samples", "low_res_logits_l1", "prob_mask_l1", "binary_mask_diff_ratio",
        "fused_minus_base_l1", "pnudp_bias_abs_mean", "pnudp_bias_abs_max",
        "pnudp_bias_std", "alpha", "dense_text_logits_mean", "dense_text_logits_std",
        "dense_alpha_value",
        "pnudp_dense_eval_scale", "alpha_effective", "effective_alpha_x_bias_abs_mean",
    ]
    for k in _keys:
        v = metrics.get(k, None)
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}={v:.12e}", flush=True)
            else:
                print(f"  {k}={v}", flush=True)
    # Also print dice/aji/pq if available
    for k in ("dice_off", "dice_on", "aji_off", "aji_on", "pq_off", "pq_on"):
        v = metrics.get(k, None)
        if v is not None:
            print(f"  {k}={v:.12e}", flush=True)


def print_summary(all_metrics: List[Dict[str, float]]):
    """Print aggregated summary."""
    if len(all_metrics) == 0:
        return
    summary = {"num_samples": len(all_metrics)}
    numeric_keys = [
        "low_res_logits_l1", "low_res_logits_l2", "low_res_logits_max_abs",
        "prob_mask_l1", "binary_mask_diff_pixels", "binary_mask_diff_ratio",
        "pnudp_bias_norm", "pnudp_bias_std", "pnudp_bias_abs_mean", "pnudp_bias_abs_max",
        "alpha", "dense_text_logits_std", "dense_text_logits_mean",
        "fused_minus_base_l1", "pnudp_active", "dense_alpha_value",
        "alpha_x_bias_abs_mean",
        "pnudp_dense_eval_scale", "alpha_effective", "effective_alpha_x_bias_abs_mean",
    ]
    for key in numeric_keys:
        vals = [m.get(key, float("nan")) for m in all_metrics]
        valid = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if len(valid) > 0:
            summary[f"mean_{key}"] = float(np.mean(valid))
        else:
            summary[f"mean_{key}"] = float("nan")

    print("[PNUDP_DENSE_TRAIN_COMPARE_SUMMARY]", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}={v:.12e}", flush=True)
        else:
            print(f"  {k}={v}", flush=True)


# ==============================================================================
# 7. Main
# ==============================================================================
def main():
    args = parse_args()

    # ── HF offline ──
    if args.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.conch_cache_path is not None:
        os.environ["HF_HOME"] = args.conch_cache_path
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.conch_cache_path

    # ── Device ──
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[PNUDP_DENSE_TRAIN_COMPARE] device={device}", flush=True)

    # ── Build base args (force enable_pnudp_dense_train=True) ──
    base_ns = _build_base_args(args)

    # ── Build single model (with pnudp_dense_train module) ──
    print("[PNUDP_DENSE_TRAIN_COMPARE] Building model (enable_pnudp_dense_train=True)...", flush=True)
    model = _build_model(base_ns)
    model = load_checkpoint(model, args.checkpoint, device)
    model = _ensure_sam_device(model, device)
    model.eval()
    print(f"  model built, param_count={sum(p.numel() for p in model.parameters()):,}", flush=True)
    _has_pnudp = getattr(model, "pnudp_dense_train", None) is not None
    print(f"  model has pnudp_dense_train module: {_has_pnudp}", flush=True)

    # ── Verify pnudp_dense_apply_in_eval attribute exists ──
    _apply_in_eval_default = getattr(model, "pnudp_dense_apply_in_eval", None)
    print(f"  model.pnudp_dense_apply_in_eval (initial) = {_apply_in_eval_default}", flush=True)
    if _apply_in_eval_default is None:
        print("  [WARN] pnudp_dense_apply_in_eval not found on model. Is the code up-to-date?", flush=True)

    # ── Disable training-only forward paths ──
    if getattr(args, "disable_attr_text_alignment_forward_in_test", True):
        _attr_before = getattr(model, "enable_attr_text_alignment", False)
        _pnurl_before = getattr(model, "enable_promptnu_lite_align", False)
        if _attr_before:
            model.enable_attr_text_alignment = False
        if _pnurl_before:
            model.enable_promptnu_lite_align = False
        print(f"[TEST_INFERENCE_MODE] enable_attr_text_alignment was {_attr_before}, set to False", flush=True)
        print(f"[TEST_INFERENCE_MODE] enable_promptnu_lite_align was {_pnurl_before}, set to False", flush=True)

    # ── Load test images ──
    print(f"[PNUDP_DENSE_TRAIN_COMPARE] Loading up to {args.num_samples} images from {args.data_path}...", flush=True)
    samples = load_test_images(args.data_path, args.num_samples, args.image_size)
    print(f"  Loaded {len(samples)} samples", flush=True)
    if len(samples) == 0:
        print("[PNUDP_DENSE_TRAIN_COMPARE] No samples loaded, exiting.", flush=True)
        return

    # ── Compare ──
    all_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            batch_input = _move_batch_to_device([sample], device)

            # ── Pass 1: PNuDP fusion OFF (pnudp_dense_apply_in_eval=False) ──
            model.pnudp_dense_apply_in_eval = False
            out_off = model(batch_input, multimask_output=True)
            if isinstance(out_off, list):
                out_off = out_off[0]
            logits_off = out_off.get("low_res_logits", None)
            if logits_off is None:
                print(f"[ERROR] sample_id={i}: low_res_logits missing in pass 1", flush=True)
                continue

            # ── Pass 2: PNuDP fusion ON (pnudp_dense_apply_in_eval=True, with eval scale) ──
            model.pnudp_dense_apply_in_eval = True
            model.pnudp_dense_eval_scale = args.pnudp_eval_scale
            out_on = model(batch_input, multimask_output=True)
            if isinstance(out_on, list):
                out_on = out_on[0]
            logits_on = out_on.get("low_res_logits", None)
            if logits_on is None:
                print(f"[ERROR] sample_id={i}: low_res_logits missing in pass 2", flush=True)
                continue

            # ── [PNUDP_COMPARE_OUTPUT_AUDIT] ──
            _lr_fused = out_on.get("low_res_logits", None)
            _lr_base = out_on.get("low_res_logits_base", None)
            if _lr_fused is not None and _lr_base is not None:
                _lr_fused_dtype = str(_lr_fused.dtype)
                _lr_base_dtype = str(_lr_base.dtype)
                _l1 = float((_lr_fused.detach() - _lr_base.detach()).abs().mean().item())
                print(f"[PNUDP_COMPARE_OUTPUT_AUDIT]", flush=True)
                print(f"  low_res_logits_dtype={_lr_fused_dtype}", flush=True)
                print(f"  low_res_logits_base_dtype={_lr_base_dtype}", flush=True)
                print(f"  low_res_logits_vs_base_l1={_l1:.12e}", flush=True)
                print(f"  output_uses_fused_logits=True (pnudp_dense_apply_in_eval=True)", flush=True)
            else:
                print(f"[PNUDP_COMPARE_OUTPUT_AUDIT]", flush=True)
                print(f"  low_res_logits_dtype={str(_lr_fused.dtype) if _lr_fused is not None else 'N/A'}", flush=True)
                print(f"  low_res_logits_base_dtype={'N/A' if _lr_base is None else str(_lr_base.dtype)}", flush=True)
                print(f"  low_res_logits_vs_base_l1=0.0 (no low_res_logits_base key)", flush=True)
                print(f"  output_uses_fused_logits=False", flush=True)

            # ── Extract PNuDP dense debug from model's own forward ──
            pnudp_debug = out_on.get("pnudp_dense_debug", {})
            if pnudp_debug:
                print(f"[PNUDP_DENSE_TRAIN_DEBUG] sample_id={i}: "
                      f"fused_minus_base_l1={pnudp_debug.get('fused_minus_base_l1', 'N/A'):.12e} | "
                      f"alpha={pnudp_debug.get('dense_alpha_value', 'N/A')} | "
                      f"alpha_effective={pnudp_debug.get('alpha_effective', 'N/A')} | "
                      f"eval_scale={pnudp_debug.get('pnudp_dense_eval_scale', 'N/A')} | "
                      f"effective_alpha_x_bias_abs_mean={pnudp_debug.get('effective_alpha_x_bias_abs_mean', 'N/A'):.12e} | "
                      f"bias_abs_mean={pnudp_debug.get('pnudp_bias_abs_mean', 'N/A'):.12e}",
                      flush=True)
            else:
                print(f"[PNUDP_DENSE_TRAIN_DEBUG] sample_id={i}: no pnudp_dense_debug in output", flush=True)

            # ── Compare ──
            metrics = compare_outputs(
                logits_off=logits_off,
                logits_on=logits_on,
                sample_id=i,
                pnudp_debug=pnudp_debug,
            )

            metrics["num_samples"] = float(len(all_metrics) + 1)
            all_metrics.append(metrics)
            print_comparison(metrics)

    # ── Summary ──
    if len(all_metrics) > 0:
        print_summary(all_metrics)

    print("[PNUDP_DENSE_TRAIN_COMPARE] Done.", flush=True)


if __name__ == "__main__":
    main()
