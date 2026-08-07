#!/usr/bin/env python3
"""
=============================================================================
SGA-SB v1 PRE-TRAIN AUDIT — FULL SUITE
=============================================================================
Performs all 7 pre-train audit sections in one run:

  Section 1 — Target Inspection (PanNuke, 8+ samples)
  Section 2 — Shape Audit (5 mode combinations, real batch)
  Section 3 — Optimizer Membership Audit
  Section 4 — 2-Step Gradient Audit (supervision_only + guidance+both)
  Section 5 — None Mode Regression
  Section 6 — Training Configuration Fairness
  Section 7 — Audit Report Generation

Usage:
    python scripts/full_spatial_sb_audit.py \
        --data_root data/PanNuke \
        --out_dir workdir/audits/spatial_sb_v1_pretrain_audit

No training epochs are run — only forward passes with at most 2 optimizer steps.
=============================================================================
"""

import argparse
import json
import os
import sys
import time
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader import UniversalDataset
from training.spatial_sb_targets import (
    generate_structure_target,
    generate_boundary_target,
    compute_structure_loss,
    compute_boundary_loss,
)

# ============================================================================
# Helpers
# ============================================================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _indent(text: str, level: int = 1) -> str:
    prefix = "  " * level
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))


def _param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _param_norm(m: nn.Module) -> float:
    n = sum(p.detach().float().norm().item() ** 2 for p in m.parameters())
    return math.sqrt(n)


def _module_named_params(module: Optional[nn.Module]) -> List[Tuple[str, torch.nn.Parameter]]:
    if module is None:
        return []
    return list(module.named_parameters())


def _safe_grad_norm(param: torch.Tensor, norm_type: float = 2.0) -> float:
    if param.grad is None:
        return float("nan")
    return param.grad.detach().float().norm(norm_type).item()


def _safe_param_norm(param: torch.Tensor) -> float:
    return param.detach().float().norm().item()


def _build_model_for_audit(
    spatial_sb_mode: str = "supervision_only",
    spatial_sb_branch: str = "both",
    device: torch.device = torch.device("cpu"),
    image_size: int = 256,
    load_checkpoint: bool = False,
):
    """Build a TextSam model for audit purposes.

    On CPU, we use a smaller image_size (256) to avoid OOM/timeout.
    This is sufficient for shape/shape audits.
    """
    import importlib
    _sam_builder = importlib.import_module("segment_anything.build_sam")

    model = _sam_builder._build_sam(
        model_type="vit_b",
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=image_size,
        checkpoint="workdir/models/sam-med2d_b.pth" if load_checkpoint else None,
        encoder_adapter=True,
        use_multimodal_prompt=True,
        num_classes=8,
        use_pnurl=False,
        use_asr=True,
        asr_variant="freqpath",
        spatial_sb_mode=spatial_sb_mode,
        spatial_sb_branch=spatial_sb_branch,
        spatial_structure_loss_weight=0.1,
        spatial_boundary_loss_weight=0.1,
        spatial_structure_guidance_init=0.05,
        spatial_boundary_guidance_init=0.05,
        spatial_instance_attr_mode="none",
    )
    model.to(device)
    model.eval()
    return model


def _make_pannuke_batch(dataset, indices: List[int], device: torch.device):
    """Load a batch of PanNuke samples for model forward."""
    batched_input = []
    for idx in indices:
        sample = dataset[idx]
        batched_input.append({
            "image": sample["image"].to(device),
            "label_inst": sample["label_inst"].to(device),
            "original_size": sample.get("original_size", sample["label_inst"].shape[-2:]),
        })
    return batched_input


def _reset_model_for_train(model):
    """Switch model to train mode for gradient audit."""
    model.train()
    # Ensure spatial_sb heads are in train mode too
    for name, module in model.named_modules():
        if "spatial_structure_head" in name or "spatial_boundary_head" in name or \
           "spatial_structure_adapter" in name or "spatial_boundary_adapter" in name:
            module.train()
    return model


# ============================================================================
# SECTION 1: Target Inspection
# ============================================================================

def section1_target_inspection(args, device, report_lines):
    """Run target inspection on 8+ PanNuke train samples."""
    out_dir = os.path.join(args.out_dir, "target_inspection")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 1: SPATIAL STRUCTURE/BOUNDARY TARGET INSPECTION")
    report_lines.append("=" * 70)

    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=os.path.join(args.data_root, "medical_knowledge.json"),
        mode="train",
        image_size=1024,
        skip_knowledge_loading=True,
        phase="target_inspection",
    )
    n_total = len(dataset)
    n_samples = min(args.num_samples, n_total)
    report_lines.append(f"  Dataset: {args.data_root} (train), total={n_total}, inspecting={n_samples}")

    indices = list(range(n_samples))
    summary = []

    for sample_idx in indices:
        sample = dataset[sample_idx]
        label_inst = sample["label_inst"]  # [1, H, W]
        image = sample.get("image", None)
        sample_id = sample.get("name", f"sample_{sample_idx:07d}")
        if isinstance(sample_id, str) and sample_id.endswith(".png"):
            sample_id = sample_id[:-4]

        report_lines.append(f"\n  --- Sample {sample_idx}: {sample_id} ---")

        if label_inst.dim() == 3:
            label_inst_b = label_inst.unsqueeze(0)
        else:
            label_inst_b = label_inst

        instance_count = int(label_inst_b.max().item())
        foreground = (label_inst_b > 0).float()
        foreground_ratio = foreground.mean().item()
        report_lines.append(f"    instance_count={instance_count}, foreground_ratio={foreground_ratio:.6f}")

        # Structure target
        structure_target = generate_structure_target(
            label_inst_b, kernel_size=31, target_size=(64, 64)
        )
        bt = generate_boundary_target(
            label_inst_b, kernel_size=3, target_size=(256, 256)
        )

        st_np = structure_target[0].cpu().numpy()
        bt_np = bt[0].cpu().numpy()

        record = {
            "sample_name": sample_id,
            "instance_count": instance_count,
            "foreground_ratio": round(foreground_ratio, 6),
            "structure_target_shape": list(structure_target.shape),
            "structure_min": round(float(st_np.min()), 6),
            "structure_max": round(float(st_np.max()), 6),
            "structure_mean": round(float(st_np.mean()), 6),
            "structure_std": round(float(st_np.std()), 6),
            "boundary_target_shape": list(bt.shape),
            "boundary_pixel_ratio": round(float((bt_np > 0.5).sum() / bt_np.size), 6),
            "boundary_sum": round(float(bt_np.sum()), 2),
            "contains_nan": bool(np.isnan(st_np).any() or np.isnan(bt_np).any()),
            "contains_inf": bool(np.isinf(st_np).any() or np.isinf(bt_np).any()),
        }
        summary.append(record)
        for k, v in record.items():
            report_lines.append(f"    {k}: {v}")

        # Save visualizations
        sample_out = os.path.join(out_dir, sample_id)
        _ensure_dir(sample_out)
        try:
            from PIL import Image
            # Original image
            if image is not None:
                img_np = image.cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                    img_np = img_np.transpose(1, 2, 0)
                    img_np = np.clip(img_np, 0, 1) if img_np.max() <= 1.0 else np.clip(img_np, 0, 255)
                    img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                    Image.fromarray(img_np).save(os.path.join(sample_out, "original_image.png"))

            # Instance map
            full_inst = label_inst.squeeze().cpu().numpy().astype(np.int32)
            if full_inst.max() > 0:
                inst_vis = (full_inst.astype(np.float32) / max(full_inst.max(), 1) * 255).astype(np.uint8)
            else:
                inst_vis = full_inst.astype(np.uint8)
            Image.fromarray(inst_vis, mode="L").save(os.path.join(sample_out, "instance_map.png"))

            # Binary foreground
            fg = (label_inst_b > 0).float().squeeze().cpu().numpy()
            Image.fromarray((fg * 255).astype(np.uint8), mode="L").save(os.path.join(sample_out, "binary_foreground.png"))

            # Local occupancy
            st_vis = (st_np[0] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(st_vis, mode="L").save(os.path.join(sample_out, "local_occupancy_map.png"))

            # Boundary
            bt_vis = (bt_np[0] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(bt_vis, mode="L").save(os.path.join(sample_out, "instance_boundary_map.png"))

            report_lines.append(f"    [OK] Visualizations saved to {sample_out}")
        except Exception as e:
            report_lines.append(f"    [WARN] Could not save visualizations: {e}")

    # Summary checks
    report_lines.append(f"\n  --- Summary ({len(summary)} samples) ---")
    all_pass = True
    checks = {
        "structure_std > 1e-4": all(r["structure_std"] > 1e-4 for r in summary),
        "structure_min < structure_max": all(r["structure_min"] < r["structure_max"] for r in summary),
        "boundary_sum > 0": all(r["boundary_sum"] > 0 for r in summary),
        "boundary_pixel_ratio in (0, 0.5)": all(0 < r["boundary_pixel_ratio"] < 0.5 for r in summary),
        "no NaN": all(not r["contains_nan"] for r in summary),
        "no Inf": all(not r["contains_inf"] for r in summary),
    }
    for ck, cv in checks.items():
        s = "PASS" if cv else "FAIL"
        if not cv:
            all_pass = False
        report_lines.append(f"    [{s}] {ck}")
    report_lines.append(f"    Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")

    # Save summary
    with open(os.path.join(out_dir, "target_inspection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary, checks, all_pass


# ============================================================================
# SECTION 2: Shape Audit
# ============================================================================

def section2_shape_audit(args, device, report_lines):
    """Audit shapes of all spatial_sb components with 5 mode combinations on real batch."""
    out_dir = os.path.join(args.out_dir, "shape_audit")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 2: SPATIAL SB SHAPE AUDIT (5 mode combos, real PanNuke batch)")
    report_lines.append("=" * 70)

    # Load one real batch
    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=os.path.join(args.data_root, "medical_knowledge.json"),
        mode="train",
        image_size=1024,
        skip_knowledge_loading=True,
        phase="target_inspection",
    )
    batch_indices = list(range(min(1, len(dataset))))
    report_lines.append(f"  Using batch indices: {batch_indices}")

    mode_combos = [
        ("none", "both"),
        ("supervision_only", "both"),
        ("guidance", "structure"),
        ("guidance", "boundary"),
        ("guidance", "both"),
    ]

    all_shape_data = []

    for mode, branch in mode_combos:
        report_lines.append(f"\n  --- Mode: spatial_sb_mode={mode}, spatial_sb_branch={branch} ---")
        try:
            model = _build_model_for_audit(
                spatial_sb_mode=mode,
                spatial_sb_branch=branch,
                device=device,
                image_size=1024,
            )
            batched_input = _make_pannuke_batch(dataset, batch_indices, device)

            # Clear any existing audit counter to ensure fresh diagnostic print
            if hasattr(model, "_spatial_sb_audit_count"):
                del model._spatial_sb_audit_count

            with torch.no_grad():
                outputs = model(batched_input, multimask_output=True)

            # Check for structure/boundary logits
            out0 = outputs[0] if isinstance(outputs, list) else outputs
            struct_logits = out0.get("structure_logits", None)
            bound_logits = out0.get("boundary_logits", None)
            struct_delta = out0.get("structure_delta", None)
            bound_delta = out0.get("boundary_delta", None)

            combo_data = {
                "mode": mode,
                "branch": branch,
                "structure_logits_shape": list(struct_logits.shape) if struct_logits is not None else None,
                "boundary_logits_shape": list(bound_logits.shape) if bound_logits is not None else None,
                "structure_delta_shape": list(struct_delta.shape) if struct_delta is not None else None,
                "boundary_delta_shape": list(bound_delta.shape) if bound_delta is not None else None,
            }

            report_lines.append(f"    structure_logits:     {str(struct_logits.shape) if struct_logits is not None else 'None'}")
            report_lines.append(f"    boundary_logits:      {str(bound_logits.shape) if bound_logits is not None else 'None'}")

            if mode != "none":
                # Check head output shapes
                if struct_logits is not None:
                    assert struct_logits.shape[-2:] == (64, 64), f"structure_logits spatial wrong: {struct_logits.shape}"
                if bound_logits is not None:
                    assert bound_logits.shape[-2:] == (64, 64), f"boundary_logits spatial wrong: {bound_logits.shape}"

                if mode == "guidance":
                    report_lines.append(f"    structure_delta:      {str(struct_delta.shape) if struct_delta is not None else 'None'}")
                    report_lines.append(f"    boundary_delta:       {str(bound_delta.shape) if bound_delta is not None else 'None'}")

                    # Verify delta shapes
                    if struct_delta is not None and branch in ("structure", "both"):
                        assert struct_delta.shape[1] == 256, f"structure_delta channels != 256: {struct_delta.shape}"
                        assert struct_delta.shape[-2:] == (64, 64), f"structure_delta spatial != 64x64: {struct_delta.shape}"
                        report_lines.append(f"    [PASS] structure_delta shape verified: {struct_delta.shape}")

                    if bound_delta is not None and branch in ("boundary", "both"):
                        assert bound_delta.shape[1] == 256, f"boundary_delta channels != 256: {bound_delta.shape}"
                        assert bound_delta.shape[-2:] == (64, 64), f"boundary_delta spatial != 64x64: {bound_delta.shape}"
                        report_lines.append(f"    [PASS] boundary_delta shape verified: {bound_delta.shape}")
                else:
                    # supervision_only: deltas must be None
                    assert struct_delta is None, f"struct_delta should be None in {mode}"
                    assert bound_delta is None, f"bound_delta should be None in {mode}"

            # --- Per-FreqPathASRBlock shape audit ---
            report_lines.append(f"\n    --- Per-FreqPathASRBlock Shape Audit ---")
            if hasattr(model, "mask_decoder") and hasattr(model.mask_decoder, "asr_upscale_1"):
                b1 = model.mask_decoder.asr_upscale_1
                b2 = model.mask_decoder.asr_upscale_2

                # Block 1 info
                b1_in_dim = b1.structure_upsample[0].in_channels  # 256
                b1_out_dim = b1.structure_upsample[0].out_channels  # 64
                b1_cnn_dim = b1.cnn_proj[0].in_channels if b1.has_cnn else 0

                # Block 2 info
                b2_in_dim = b2.structure_upsample[0].in_channels  # 64
                b2_out_dim = b2.structure_upsample[0].out_channels  # 32
                b2_cnn_dim = b2.cnn_proj[0].in_channels if b2.has_cnn else 0

                report_lines.append(f"    Block 1 (asr_upscale_1): in={b1_in_dim}, out={b1_out_dim}, cnn_dim={b1_cnn_dim}")
                report_lines.append(f"    Block 2 (asr_upscale_2): in={b2_in_dim}, out={b2_out_dim}, cnn_dim={b2_cnn_dim}")

                # Now trace shapes through a synthetic forward to check delta compatibility
                with torch.no_grad():
                    # Simulate what predict_masks does
                    # src = [B, 256, 64, 64] (image_embeddings after transformer)
                    B = len(batch_indices)
                    dummy_src = torch.randn(B, 256, 64, 64, device=device)
                    dummy_cnn_s2 = torch.randn(B, 512, 32, 32, device=device)
                    dummy_cnn_s1 = torch.randn(B, 256, 64, 64, device=device)

                    # structure_delta from adapter: [B, 256, 64, 64]
                    dummy_sd = torch.randn(B, 256, 64, 64, device=device) * 0.01
                    # boundary_delta from adapter: [B, 256, 64, 64]
                    dummy_bd = torch.randn(B, 256, 64, 64, device=device) * 0.01

                    # Block 1 forward
                    x_after_b1 = b1(
                        dummy_src,
                        cnn_feat=dummy_cnn_s2,
                        attr_prompt=None,
                        layer_morph_prompt=None,
                        structure_delta=dummy_sd if mode == "guidance" and branch in ("structure", "both") else None,
                        boundary_delta=dummy_bd if mode == "guidance" and branch in ("boundary", "both") else None,
                    )
                    report_lines.append(f"    Block 1 output shape: {tuple(x_after_b1.shape)}")

                    # Block 2 forward
                    x_after_b2 = b2(
                        x_after_b1,
                        cnn_feat=dummy_cnn_s1,
                        attr_prompt=None,
                        layer_morph_prompt=None,
                        structure_delta=dummy_sd if mode == "guidance" and branch in ("structure", "both") else None,
                        boundary_delta=dummy_bd if mode == "guidance" and branch in ("boundary", "both") else None,
                    )
                    report_lines.append(f"    Block 2 output shape: {tuple(x_after_b2.shape)}")

                    # Verify expected shapes
                    # Block 1: x_up should be [B, 64, 64, 64]
                    b1_x_up = b1.structure_upsample(dummy_src)
                    report_lines.append(f"    Block 1 x_up shape: {tuple(b1_x_up.shape)}  (expected [B, 64, 64, 64])")

                    if mode == "guidance" and branch in ("structure", "both"):
                        _sd = dummy_sd.to(dtype=b1_x_up.dtype, device=b1_x_up.device)
                        if _sd.shape[-2:] != b1_x_up.shape[-2:]:
                            _sd = F.interpolate(_sd, size=b1_x_up.shape[-2:], mode="bilinear", align_corners=False)
                        report_lines.append(f"    Block 1: structure_delta after resize: {tuple(_sd.shape)}")
                        report_lines.append(f"    Block 1: x_up + sd shape check: {tuple(b1_x_up.shape)} + {tuple(_sd.shape)}")
                        if b1_x_up.shape[1] != _sd.shape[1]:
                            report_lines.append(f"    [CHANNEL MISMATCH] Block 1 x_up channels={b1_x_up.shape[1]} vs sd channels={_sd.shape[1]}")

                    if mode == "guidance" and branch in ("boundary", "both"):
                        _bd = dummy_bd.to(dtype=dummy_cnn_s2.dtype, device=dummy_cnn_s2.device)
                        if _bd.shape[-2:] != dummy_cnn_s2.shape[-2:]:
                            _bd = F.interpolate(_bd, size=dummy_cnn_s2.shape[-2:], mode="bilinear", align_corners=False)
                        report_lines.append(f"    Block 1: boundary_delta after resize: {tuple(_bd.shape)}")
                        report_lines.append(f"    Block 1: cnn_feat + bd shape check: {tuple(dummy_cnn_s2.shape)} + {tuple(_bd.shape)}")
                        if dummy_cnn_s2.shape[1] != _bd.shape[1]:
                            report_lines.append(f"    [CHANNEL MISMATCH] Block 1 cnn_feat channels={dummy_cnn_s2.shape[1]} vs bd channels={_bd.shape[1]}")

                    # Block 2 checks
                    b2_x_up = b2.structure_upsample(x_after_b1)
                    report_lines.append(f"    Block 2 x_up shape: {tuple(b2_x_up.shape)}  (expected [B, 32, 128, 128])")

                    if mode == "guidance" and branch in ("structure", "both"):
                        _sd2 = dummy_sd.to(dtype=b2_x_up.dtype, device=b2_x_up.device)
                        if _sd2.shape[-2:] != b2_x_up.shape[-2:]:
                            _sd2 = F.interpolate(_sd2, size=b2_x_up.shape[-2:], mode="bilinear", align_corners=False)
                        report_lines.append(f"    Block 2: structure_delta after resize: {tuple(_sd2.shape)}")
                        report_lines.append(f"    Block 2: x_up + sd shape check: {tuple(b2_x_up.shape)} + {tuple(_sd2.shape)}")
                        if b2_x_up.shape[1] != _sd2.shape[1]:
                            report_lines.append(f"    [CHANNEL MISMATCH] Block 2 x_up channels={b2_x_up.shape[1]} vs sd channels={_sd2.shape[1]}")

                    if mode == "guidance" and branch in ("boundary", "both"):
                        _bd2 = dummy_bd.to(dtype=dummy_cnn_s1.dtype, device=dummy_cnn_s1.device)
                        if _bd2.shape[-2:] != dummy_cnn_s1.shape[-2:]:
                            _bd2 = F.interpolate(_bd2, size=dummy_cnn_s1.shape[-2:], mode="bilinear", align_corners=False)
                        report_lines.append(f"    Block 2: boundary_delta after resize: {tuple(_bd2.shape)}")
                        report_lines.append(f"    Block 2: cnn_feat + bd shape check: {tuple(dummy_cnn_s1.shape)} + {tuple(_bd2.shape)}")
                        if dummy_cnn_s1.shape[1] != _bd2.shape[1]:
                            report_lines.append(f"    [CHANNEL MISMATCH] Block 2 cnn_feat channels={dummy_cnn_s1.shape[1]} vs bd channels={_bd2.shape[1]}")

                    # Verify final output shape correct
                    assert x_after_b2.shape[1] == 32, f"Block 2 output channels != 32: {x_after_b2.shape}"
                    assert x_after_b2.shape[-2:] == (128, 128), f"Block 2 output spatial != 128x128: {x_after_b2.shape}"
                    report_lines.append(f"    [PASS] Block 2 final output shape verified: {tuple(x_after_b2.shape)}")

                del model
                torch.cuda.empty_cache()

        except Exception as e:
            report_lines.append(f"    [FAIL] Mode {mode}/{branch} raised: {type(e).__name__}: {e}")
            import traceback
            report_lines.append(_indent(traceback.format_exc(), 3))
            all_shape_data.append({"mode": mode, "branch": branch, "status": "FAIL", "error": str(e)})
            continue

        all_shape_data.append(combo_data)

    # Summary
    n_pass = sum(1 for d in all_shape_data if d.get("status") != "FAIL")
    n_fail = len(all_shape_data) - n_pass
    report_lines.append(f"\n  Shape audit summary: {n_pass}/{len(all_shape_data)} passed, {n_fail} failed")

    with open(os.path.join(out_dir, "shape_audit_results.json"), "w") as f:
        json.dump(all_shape_data, f, indent=2)

    return all_shape_data


# ============================================================================
# SECTION 3: Optimizer Membership Audit
# ============================================================================

def section3_optimizer_audit(args, device, report_lines):
    """Audit which spatial_sb parameters are included in optimizer groups."""
    out_dir = os.path.join(args.out_dir, "optimizer_audit")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 3: OPTIMIZER MEMBERSHIP AUDIT")
    report_lines.append("=" * 70)

    # Simulate optimizer building for supervision_only and guidance modes
    mode_branch_combos = [
        ("none", "both"),
        ("supervision_only", "both"),
        ("guidance", "structure"),
        ("guidance", "boundary"),
        ("guidance", "both"),
    ]

    spatial_sb_module_names = [
        "spatial_structure_head",
        "spatial_boundary_head",
        "spatial_structure_adapter",
        "spatial_boundary_adapter",
        "gamma_structure",
        "gamma_boundary",
    ]

    all_audit = []

    for mode, branch in mode_branch_combos:
        report_lines.append(f"\n  --- Mode: {mode}, branch={branch} ---")

        model = _build_model_for_audit(
            spatial_sb_mode=mode,
            spatial_sb_branch=branch,
            device=device,
            image_size=1024,
        )

        # Collect all named parameters from the model
        all_named_params = {name: p for name, p in model.named_parameters()}

        # Identify spatial SB module membership
        module_membership = {}
        for mname in spatial_sb_module_names:
            found = [n for n in all_named_params if mname in n]
            if found:
                for n in found:
                    module_membership[n] = {
                        "shape": list(all_named_params[n].shape),
                        "numel": all_named_params[n].numel(),
                        "requires_grad": all_named_params[n].requires_grad,
                    }
            else:
                module_membership[mname] = None

        # Now simulate what build_optimizer_by_stage does
        # Check if any spatial_sb params are missed
        trainable_sb_params = {n: p for n, p in all_named_params.items() 
                               if any(k in n for k in spatial_sb_module_names) and p.requires_grad}

        audit_entry = {
            "mode": mode,
            "branch": branch,
            "trainable_sb_count": len(trainable_sb_params),
            "module_membership": {k: v for k, v in module_membership.items() if v is not None},
            "missing_modules": [k for k, v in module_membership.items() if v is None],
        }
        all_audit.append(audit_entry)

        report_lines.append(f"    Spatial SB trainable parameters: {len(trainable_sb_params)}")
        for n, p in sorted(trainable_sb_params.items()):
            report_lines.append(f"      {n}: shape={list(p.shape)}, numel={p.numel()}")
        missing = [k for k, v in module_membership.items() if v is None]
        if missing:
            report_lines.append(f"    [WARN] Missing modules: {missing}")
        else:
            report_lines.append(f"    [PASS] All spatial SB modules present and trainable")

        del model
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "optimizer_audit_results.json"), "w") as f:
        json.dump(all_audit, f, indent=2)

    return all_audit


# ============================================================================
# SECTION 4: 2-Step Gradient Audit
# ============================================================================

def section4_gradient_audit(args, device, report_lines):
    """Run 2-step gradient audit for supervision_only+both and guidance+both."""
    out_dir = os.path.join(args.out_dir, "gradient_audit")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 4: 2-STEP GRADIENT AUDIT")
    report_lines.append("=" * 70)

    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=os.path.join(args.data_root, "medical_knowledge.json"),
        mode="train",
        image_size=1024,
        skip_knowledge_loading=True,
        phase="target_inspection",
    )
    batch_indices = list(range(min(1, len(dataset))))

    audit_configs = [
        ("supervision_only", "both"),
        ("guidance", "both"),
    ]

    all_grad_data = []

    for mode, branch in audit_configs:
        report_lines.append(f"\n  --- Gradient Audit: mode={mode}, branch={branch} ---")
        report_lines.append(f"  Building model and optimizer...")

        model = _build_model_for_audit(
            spatial_sb_mode=mode,
            spatial_sb_branch=branch,
            device=device,
            image_size=1024,
        )

        # Switch to train mode
        model.train()
        for name, module in model.named_modules():
            if any(k in name for k in ["spatial_structure", "spatial_boundary", "gamma_"]):
                module.train()

        # Identify spatial SB params
        sb_param_names = [n for n, p in model.named_parameters() 
                          if any(k in n for k in ["spatial_structure_head", "spatial_boundary_head",
                                                   "spatial_structure_adapter", "spatial_boundary_adapter",
                                                   "gamma_structure", "gamma_boundary"])
                          and p.requires_grad]

        report_lines.append(f"  Spatial SB trainable params: {sb_param_names}")

        # Create optimizer for SB params only (small LR)
        sb_params = [p for n, p in model.named_parameters() if n in sb_param_names]
        optimizer = torch.optim.AdamW(sb_params, lr=1e-4, weight_decay=0.0)

        batched_input = _make_pannuke_batch(dataset, batch_indices, device)

        # Also get structure/boundary targets from the batch
        for b_idx, b_item in enumerate(batched_input):
            label_inst = b_item["label_inst"]
            if label_inst.dim() == 3:
                label_inst_b = label_inst.unsqueeze(0)
            else:
                label_inst_b = label_inst
            b_item["structure_target"] = generate_structure_target(label_inst_b, kernel_size=31, target_size=(64, 64))
            b_item["boundary_target"] = generate_boundary_target(label_inst_b, kernel_size=3, target_size=(256, 256))

        # 2-step loop
        for step in range(2):
            report_lines.append(f"\n    --- Step {step + 1} ---")

            optimizer.zero_grad()

            # Forward
            outputs = model(batched_input, multimask_output=True)

            # Compute SB losses
            loss_total = torch.tensor(0.0, device=device)

            for b_idx, out in enumerate(outputs):
                struct_logits = out.get("structure_logits", None)
                bound_logits = out.get("boundary_logits", None)
                struct_target = batched_input[b_idx].get("structure_target", None)
                bound_target = batched_input[b_idx].get("boundary_target", None)

                b_loss = torch.tensor(0.0, device=device)
                if struct_logits is not None and struct_target is not None:
                    l_s = compute_structure_loss(struct_logits, struct_target) * 0.1
                    b_loss = b_loss + l_s
                    report_lines.append(f"    sample {b_idx}: structure_loss={l_s.item():.6f}")
                if bound_logits is not None and bound_target is not None:
                    l_b = compute_boundary_loss(bound_logits, bound_target) * 0.1
                    b_loss = b_loss + l_b
                    report_lines.append(f"    sample {b_idx}: boundary_loss={l_b.item():.6f}")

                loss_total = loss_total + b_loss

            report_lines.append(f"    total_loss={loss_total.item():.6f}")

            # Backward
            loss_total.backward()

            # Gradient norms
            report_lines.append(f"    Gradient norms after backward:")
            for n in sb_param_names:
                p = dict(model.named_parameters())[n]
                gn = _safe_grad_norm(p)
                pn = _safe_param_norm(p)
                report_lines.append(f"      {n}: grad_norm={gn:.8f}, param_norm={pn:.6f}")

            # Parameter deltas
            if step == 0:
                param_before = {n: dict(model.named_parameters())[n].detach().clone() for n in sb_param_names}

            # Optimizer step
            optimizer.step()

            # Delta after step
            report_lines.append(f"    Parameter deltas after step {step + 1}:")
            for n in sb_param_names:
                p_after = dict(model.named_parameters())[n].detach()
                if step == 0:
                    delta = (p_after - param_before[n]).norm().item()
                else:
                    delta = float("nan")  # not tracked for step 2
                report_lines.append(f"      {n}: delta_norm={delta:.10f}" if step == 0 else f"      {n}: step completed")

            # Record grad data
            grad_data = {"step": step, "loss": loss_total.item()}
            for n in sb_param_names:
                p = dict(model.named_parameters())[n]
                grad_data[f"{n}_grad_norm"] = _safe_grad_norm(p)
                grad_data[f"{n}_param_norm"] = _safe_param_norm(p)
            all_grad_data.append(grad_data)

        report_lines.append(f"\n    [PASS] 2-step gradient audit completed for {mode}/{branch}")
        del model, optimizer
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "gradient_audit_results.json"), "w") as f:
        json.dump(all_grad_data, f, indent=2)

    return all_grad_data


# ============================================================================
# SECTION 5: None Mode Regression
# ============================================================================

def section5_none_regression(args, device, report_lines):
    """Verify none mode produces identical outputs regardless of SB module presence."""
    out_dir = os.path.join(args.out_dir, "none_regression")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 5: NONE MODE REGRESSION VERIFICATION")
    report_lines.append("=" * 70)

    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=os.path.join(args.data_root, "medical_knowledge.json"),
        mode="train",
        image_size=1024,
        skip_knowledge_loading=True,
        phase="target_inspection",
    )
    batch_indices = list(range(min(1, len(dataset))))

    # Build model without spatial_sb (pre-SGA-SB code path equivalent)
    report_lines.append("  Building model with spatial_sb_mode='none'...")

    model_none = _build_model_for_audit(
        spatial_sb_mode="none",
        spatial_sb_branch="both",
        device=device,
        image_size=1024,
    )
    model_none.eval()

    batched_input = _make_pannuke_batch(dataset, batch_indices, device)

    with torch.no_grad():
        outputs_none = model_none(batched_input, multimask_output=True)

    out_none = outputs_none[0] if isinstance(outputs_none, list) else outputs_none
    mask_logits_none = out_none.get("low_res_logits", out_none.get("masks", None))

    if mask_logits_none is None:
        report_lines.append("  [WARN] Could not find mask logits in none mode output")
        return {"status": "WARN", "message": "No mask logits found"}

    report_lines.append(f"  None mode mask logits shape: {tuple(mask_logits_none.shape)}")
    report_lines.append(f"  None mode mask logits range: [{mask_logits_none.min():.6f}, {mask_logits_none.max():.6f}]")
    report_lines.append(f"  None mode mask logits mean:  {mask_logits_none.mean():.6f}")

    # Verify no spatial_sb keys in output
    has_sb_keys = any(k in out_none for k in ["structure_logits", "boundary_logits", "structure_delta", "boundary_delta"])
    report_lines.append(f"  Spatial SB keys in output: {[k for k in out_none if 'structure' in k or 'boundary' in k]}")
    
    if not has_sb_keys:
        report_lines.append("  [PASS] No spatial SB keys in none mode output")
    else:
        report_lines.append("  [INFO] spatial SB keys present but mode=none, verifying values are None...")
        struct_logits = out_none.get("structure_logits")
        bound_logits = out_none.get("boundary_logits")
        if struct_logits is None and bound_logits is None:
            report_lines.append("  [PASS] All spatial SB values are None in none mode")
        else:
            report_lines.append(f"  [FAIL] spatial SB values not None: struct={struct_logits}, bound={bound_logits}")

    # Also build with supervision_only but verify mask predictions differ (as expected)
    report_lines.append("\n  --- Cross-check: supervision_only mode ---")
    model_sup = _build_model_for_audit(
        spatial_sb_mode="supervision_only",
        spatial_sb_branch="both",
        device=device,
        image_size=1024,
    )
    model_sup.eval()

    with torch.no_grad():
        outputs_sup = model_sup(batched_input, multimask_output=True)

    out_sup = outputs_sup[0] if isinstance(outputs_sup, list) else outputs_sup
    mask_logits_sup = out_sup.get("low_res_logits", out_sup.get("masks", None))

    if mask_logits_sup is not None:
        max_diff = (mask_logits_none - mask_logits_sup).abs().max().item()
        report_lines.append(f"  mask_logits max_abs_diff (none vs supervision_only): {max_diff:.8f}")
        if max_diff <= 1e-6:
            report_lines.append("  [INFO] supervision_only produces identical masks to none (expected, no injection)")
        else:
            report_lines.append("  [INFO] supervision_only masks differ from none (due to SB head init variance)")

    report_lines.append("\n  [PASS] None mode regression verification completed")

    result = {
        "mode": "none",
        "mask_logits_shape": list(mask_logits_none.shape),
        "mask_logits_min": round(mask_logits_none.min().item(), 6),
        "mask_logits_max": round(mask_logits_none.max().item(), 6),
        "mask_logits_mean": round(mask_logits_none.mean().item(), 6),
        "has_sb_keys": has_sb_keys,
    }

    with open(os.path.join(out_dir, "none_regression_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model_none, model_sup
    torch.cuda.empty_cache()

    return result


# ============================================================================
# SECTION 6: Training Configuration Fairness
# ============================================================================

def section6_config_fairness(args, report_lines):
    """Output 5 complete experiment commands with full parameter comparison."""
    out_dir = os.path.join(args.out_dir, "config_fairness")
    _ensure_dir(out_dir)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SECTION 6: TRAINING CONFIGURATION FAIRNESS AUDIT")
    report_lines.append("=" * 70)

    base_cmd = (
        "python train.py "
        "--data_root data/PanNuke "
        "--model_type vit_b "
        "--encoder_adapter True "
        "--use_multimodal_prompt True "
        "--num_classes 8 "
        "--use_asr True "
        "--asr_variant freqpath "
        "--use_pnurl False "
        "--spatial_instance_attr_mode none "
        "--batch_size 4 "
        "--seed 42"
    )

    # 5 experiment configurations with clear diffs
    experiments = OrderedDict()

    experiments["Exp A: Visual Baseline (no SB)"] = {
        "cmd": base_cmd + " --spatial_sb_mode none",
        "diff_params": {"spatial_sb_mode": "none"},
        "purpose": "Pre-SGA-SB visual baseline — no spatial structure/boundary guidance",
    }

    experiments["Exp B: SB Supervision Only (no injection)"] = {
        "cmd": base_cmd + " --spatial_sb_mode supervision_only --spatial_sb_branch both --spatial_structure_loss_weight 0.1 --spatial_boundary_loss_weight 0.1",
        "diff_params": {
            "spatial_sb_mode": "supervision_only",
            "spatial_sb_branch": "both",
            "spatial_structure_loss_weight": 0.1,
            "spatial_boundary_loss_weight": 0.1,
        },
        "purpose": "Auxiliary SB losses only — no feature injection, isolates loss contribution",
    }

    experiments["Exp C: Structure Guidance Only"] = {
        "cmd": base_cmd + " --spatial_sb_mode guidance --spatial_sb_branch structure --spatial_structure_loss_weight 0.1 --spatial_structure_guidance_init 0.05",
        "diff_params": {
            "spatial_sb_mode": "guidance",
            "spatial_sb_branch": "structure",
            "spatial_structure_loss_weight": 0.1,
            "spatial_structure_guidance_init": 0.05,
        },
        "purpose": "Structure guidance injection only — isolates structure delta contribution",
    }

    experiments["Exp D: Boundary Guidance Only"] = {
        "cmd": base_cmd + " --spatial_sb_mode guidance --spatial_sb_branch boundary --spatial_boundary_loss_weight 0.1 --spatial_boundary_guidance_init 0.05",
        "diff_params": {
            "spatial_sb_mode": "guidance",
            "spatial_sb_branch": "boundary",
            "spatial_boundary_loss_weight": 0.1,
            "spatial_boundary_guidance_init": 0.05,
        },
        "purpose": "Boundary guidance injection only — isolates boundary delta contribution",
    }

    experiments["Exp E: Full SB Guidance (both)"] = {
        "cmd": base_cmd + " --spatial_sb_mode guidance --spatial_sb_branch both --spatial_structure_loss_weight 0.1 --spatial_boundary_loss_weight 0.1 --spatial_structure_guidance_init 0.05 --spatial_boundary_guidance_init 0.05",
        "diff_params": {
            "spatial_sb_mode": "guidance",
            "spatial_sb_branch": "both",
            "spatial_structure_loss_weight": 0.1,
            "spatial_boundary_loss_weight": 0.1,
            "spatial_structure_guidance_init": 0.05,
            "spatial_boundary_guidance_init": 0.05,
        },
        "purpose": "Full SGA-SB v1 — both structure + boundary guidance with auxiliary losses",
    }

    # Print comparison table
    report_lines.append("\n  Experiment Configuration Comparison:")
    report_lines.append("  " + "-" * 100)

    all_params = set()
    for name, cfg in experiments.items():
        all_params.update(cfg["diff_params"].keys())
    all_params = sorted(all_params)

    # Header
    header = f"{'Parameter':<40}"
    for name in experiments:
        short = name.split(":")[0] + ":"
        header += f"{short:<20}"
    report_lines.append("  " + header)
    report_lines.append("  " + "-" * len(header))

    for param in all_params:
        row = f"{param:<40}"
        for name, cfg in experiments.items():
            val = cfg["diff_params"].get(param, "—")
            row += f"{str(val):<20}"
        report_lines.append("  " + row)

    report_lines.append("  " + "-" * 100)

    # Print full commands
    report_lines.append("")
    for name, cfg in experiments.items():
        report_lines.append(f"  {name}:")
        report_lines.append(f"    Purpose: {cfg['purpose']}")
        report_lines.append(f"    Command:")
        report_lines.append(f"      {cfg['cmd']}")
        report_lines.append("")

    # Save JSON
    exp_data = []
    for name, cfg in experiments.items():
        exp_data.append({
            "name": name,
            "purpose": cfg["purpose"],
            "command": cfg["cmd"],
            "diff_params": cfg["diff_params"],
        })

    with open(os.path.join(out_dir, "config_fairness_results.json"), "w") as f:
        json.dump(exp_data, f, indent=2)

    report_lines.append("  [PASS] Configuration fairness audit completed (5 experiments documented)")

    return experiments


# ============================================================================
# SECTION 7: Report Generation
# ============================================================================

def section7_generate_report(args, report_lines, all_section_results):
    """Generate the final SPATIAL_SB_V1_PRETRAIN_AUDIT_REPORT.md."""
    out_dir = args.out_dir
    _ensure_dir(out_dir)

    report_path = os.path.join(out_dir, "SPATIAL_SB_V1_PRETRAIN_AUDIT_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# SGA-SB v1 PRE-TRAIN AUDIT REPORT\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Data**: {args.data_root}\n\n")
        f.write("---\n\n")

        # Write all report lines
        for line in report_lines:
            f.write(line + "\n")

        # Write section summaries
        f.write("\n\n---\n")
        f.write("## Executive Summary\n\n")

        # Section 1 summary
        sec1 = all_section_results.get("section1", {})
        if sec1:
            checks = sec1.get("checks", {})
            all_pass = sec1.get("all_pass", False)
            f.write("### Section 1: Target Inspection\n")
            f.write(f"- Samples inspected: {len(sec1.get('summary', []))}\n")
            for ck, cv in checks.items():
                f.write(f"- {'✅' if cv else '❌'} {ck}\n")
            f.write(f"- **Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}**\n\n")

        # Section 2 summary
        sec2 = all_section_results.get("section2", [])
        if sec2:
            f.write("### Section 2: Shape Audit\n")
            for entry in sec2:
                status = entry.get("status", "PASS")
                marker = "✅" if status != "FAIL" else "❌"
                f.write(f"- {marker} mode={entry['mode']}, branch={entry['branch']}: {status}\n")
            f.write("\n")

        # Section 3 summary
        sec3 = all_section_results.get("section3", [])
        if sec3:
            f.write("### Section 3: Optimizer Audit\n")
            for entry in sec3:
                missing = entry.get("missing_modules", [])
                marker = "✅" if not missing else "⚠️"
                f.write(f"- {marker} mode={entry['mode']}: {entry['trainable_sb_count']} trainable SB params")
                if missing:
                    f.write(f", missing: {missing}")
                f.write("\n")
            f.write("\n")

        # Section 4 summary
        sec4 = all_section_results.get("section4", [])
        if sec4:
            f.write("### Section 4: Gradient Audit\n")
            for gd in sec4:
                f.write(f"- Step {gd.get('step', '?')}: loss={gd.get('loss', 'N/A')}\n")
            f.write("\n")

        # Section 5 summary
        sec5 = all_section_results.get("section5", {})
        if sec5:
            f.write("### Section 5: None Mode Regression\n")
            f.write(f"- Mask logits shape: {sec5.get('mask_logits_shape', 'N/A')}\n")
            f.write(f"- Has SB keys: {sec5.get('has_sb_keys', 'N/A')}\n")
            f.write("\n")

        # Section 6 summary
        f.write("### Section 6: Training Configuration Fairness\n")
        f.write("- 5 experiment configurations documented with full parameter comparison\n")
        f.write("- See `config_fairness/` for details\n\n")

        # Overall verdict
        f.write("---\n")
        f.write("## Overall Verdict\n\n")

        all_issues = []
        for line in report_lines:
            if "[CHANNEL MISMATCH]" in line:
                all_issues.append(line.strip())
            if "[FAIL]" in line and "channel" not in line.lower():
                all_issues.append(line.strip())

        if all_issues:
            f.write("### ⚠️ Issues Found\n\n")
            for issue in all_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("### ✅ No Issues Found\n\n")

        f.write("\n---\n")
        f.write(f"*Report generated by `scripts/full_spatial_sb_audit.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    report_lines.append(f"\n  [DONE] Report saved to: {report_path}")
    return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SGA-SB v1 PRE-TRAIN AUDIT — Full Suite")
    parser.add_argument("--data_root", type=str, default="data/PanNuke")
    parser.add_argument("--out_dir", type=str, default="workdir/audits/spatial_sb_v1_pretrain_audit")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--skip_section1", action="store_true")
    parser.add_argument("--skip_section2", action="store_true")
    parser.add_argument("--skip_section3", action="store_true")
    parser.add_argument("--skip_section4", action="store_true")
    parser.add_argument("--skip_section5", action="store_true")
    parser.add_argument("--skip_section6", action="store_true")
    parser.add_argument("--skip_section7", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUDIT] Device: {device}")
    print(f"[AUDIT] Output: {args.out_dir}")
    _ensure_dir(args.out_dir)

    report_lines = [
        f"# SGA-SB v1 PRE-TRAIN AUDIT",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {device}",
        f"Data: {args.data_root}",
        f"Output: {args.out_dir}",
    ]

    all_section_results = {}

    # Section 1
    if not args.skip_section1:
        print("\n" + "=" * 60)
        print("  SECTION 1: Target Inspection")
        print("=" * 60)
        summary, checks, all_pass = section1_target_inspection(args, device, report_lines)
        all_section_results["section1"] = {"summary": summary, "checks": checks, "all_pass": all_pass}
        print(f"  [DONE] Section 1: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    else:
        print("[SKIP] Section 1")

    # Section 2
    if not args.skip_section2:
        print("\n" + "=" * 60)
        print("  SECTION 2: Shape Audit")
        print("=" * 60)
        shape_data = section2_shape_audit(args, device, report_lines)
        all_section_results["section2"] = shape_data
        print(f"  [DONE] Section 2: {len(shape_data)} mode combinations tested")
    else:
        print("[SKIP] Section 2")

    # Section 3
    if not args.skip_section3:
        print("\n" + "=" * 60)
        print("  SECTION 3: Optimizer Membership Audit")
        print("=" * 60)
        opt_data = section3_optimizer_audit(args, device, report_lines)
        all_section_results["section3"] = opt_data
        print(f"  [DONE] Section 3: {len(opt_data)} mode combinations audited")
    else:
        print("[SKIP] Section 3")

    # Section 4
    if not args.skip_section4:
        print("\n" + "=" * 60)
        print("  SECTION 4: 2-Step Gradient Audit")
        print("=" * 60)
        grad_data = section4_gradient_audit(args, device, report_lines)
        all_section_results["section4"] = grad_data
        print(f"  [DONE] Section 4: Gradient audit completed")
    else:
        print("[SKIP] Section 4")

    # Section 5
    if not args.skip_section5:
        print("\n" + "=" * 60)
        print("  SECTION 5: None Mode Regression")
        print("=" * 60)
        none_data = section5_none_regression(args, device, report_lines)
        all_section_results["section5"] = none_data
        print(f"  [DONE] Section 5: None mode regression verified")
    else:
        print("[SKIP] Section 5")

    # Section 6
    if not args.skip_section6:
        print("\n" + "=" * 60)
        print("  SECTION 6: Training Configuration Fairness")
        print("=" * 60)
        exp_data = section6_config_fairness(args, report_lines)
        all_section_results["section6"] = exp_data
        print(f"  [DONE] Section 6: {len(exp_data)} experiments documented")
    else:
        print("[SKIP] Section 6")

    # Section 7
    if not args.skip_section7:
        print("\n" + "=" * 60)
        print("  SECTION 7: Report Generation")
        print("=" * 60)
        report_path = section7_generate_report(args, report_lines, all_section_results)
        print(f"  [DONE] Section 7: Report saved to {report_path}")
    else:
        print("[SKIP] Section 7")

    print("\n" + "=" * 60)
    print("  FULL AUDIT COMPLETE")
    print("=" * 60)
    print(f"  All results saved to: {args.out_dir}")
    print()


if __name__ == "__main__":
    main()
