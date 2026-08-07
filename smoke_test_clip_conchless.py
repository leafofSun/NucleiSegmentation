#!/usr/bin/env python3
"""
Smoke test: verify TextSam can load CLIP text bank checkpoint with CONCHLESS mode.

Tests:
1. Model build with use_checkpoint_text_bank_without_conch=True
2. Checkpoint loading with text bank buffer shape mismatch handling
3. Text bank buffer values are correctly populated
4. Forward pass on 2 real images (no NaN, no crash)

Usage:
    python smoke_test_clip_conchless.py
"""

import os
import sys
import torch
import argparse
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["PNURL_AUDIT_ENABLED"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

from segment_anything.build_sam import build_sam_vit_b
from segment_anything.modeling.sam import TextSam
from segment_anything import sam_model_registry

CKPT_PATH = "/hy-tmp/NuSeg/workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth"
TEST_IMAGES = [
    "/hy-tmp/NuSeg/data/MoNuSeg/test/TCGA-2Z-A9J9-01A-01-TS1.png",
    "/hy-tmp/NuSeg/data/MoNuSeg/test/TCGA-44-2665-01B-06-BS6.png",
]


def make_args(**overrides):
    defaults = {
        "image_size": 512,
        "checkpoint": None,
        "sam_checkpoint": None,
        "encoder_adapter": True,
        "use_multimodal_prompt": False,
        "clip_model_path": None,
        "num_classes": 8,
        "use_pnurl": False,
        "use_coop_prompt": False,
        "use_coop": False,
        "use_asr": True,
        "asr_variant": "freqpath",
        "asr_regression": False,
        "max_semantic_gate": 0.10,
        "max_delta_ratio": 0.10,
        "init_delta_ratio": 0.02,
        "semantic_gate_bias_init": None,
        "semantic_injection_scale": 1.0,
        "enable_structure_boundary_attr_heads": False,
        "enable_multilevel_attr_heads": False,
        "enable_attr_text_alignment": False,
        "sb_guidance_mode": "none",
        "sb_guidance_weight": 1.0,
        "sb_conch_freeze": True,
        "sb_prompt_template_path": "workdir/attr_stats/structure_boundary_prompt_templates.json",
        "sb_guidance_routing": "structure_low_boundary_high",
        "enable_conch_text_encoder": False,
        "enable_promptnu_lite_align": False,
        "promptnu_lite_target": "semantic_delta",
        "promptnu_lite_struct_weight": 0.0,
        "promptnu_lite_boundary_weight": 0.0,
        "promptnu_lite_instance_weight": 0.0,
        "promptnu_lite_detach_text": True,
        "promptnu_lite_detach_visual": False,
        "promptnu_lite_proj_lr_mult": 0.5,
        # ── CONCHLESS mode ──
        "use_checkpoint_text_bank_without_conch": True,
        # ── CLIP backend ──
        "clip_text_encoder": False,
        "clip_text_encoder_model": "ViT-B/32",
        "clip_text_encoder_cache_path": "hf_cache/clip",
        # ── PromptNu-guided v3 ──
        "enable_promptnu_guided_v3": False,
        "promptnu_guided_v3_struct_weight": 1.0,
        "promptnu_guided_v3_boundary_weight": 1.0,
        "promptnu_guided_v3_text_weight": 0.01,
        "promptnu_guided_v3_embed_dim": 256,
        "promptnu_guided_v3_hidden_dim": 128,
        "promptnu_guided_v3_vis_proj_dim": 512,
        "promptnu_guided_v3_align_loss_weight": 0.1,
        "promptnu_guided_v3_use_text_bank": False,
        "promptnu_guided_v3_use_gt_align_target": False,
        "promptnu_guided_v3_semantic_dim": 256,
        "promptnu_guided_v3_text_dim": 512,
        "promptnu_guided_v3_strict_audit": False,
        "promptnu_guided_v3_guidance_mode": "scale_add",
        "promptnu_guided_v3_scale_weight": None,
        "promptnu_guided_v3_delta_weight": 0.001,
        "promptnu_guided_v3_delta_init_std": 1e-5,
        "promptnu_guided_v3_max_guided_delta_ratio": 0.0,
        "promptnu_guided_v3_align_eps": 1e-8,
        "promptnu_guided_v3_cosine_eps": 1e-8,
        "promptnu_guided_v3_min_align_delta_norm": 0.0,
        "promptnu_guided_v3_align_low_norm_mode": "detach_guided",
        "ablate_semantic_injection": False,
        "ablate_pred_attr_guidance": False,
        "promptnu_guided_v3_prompt_source": "pred_attr",
        "enable_numeric_attr_freqpath_guidance": False,
        "numeric_attr_freqpath_hidden_dim": 128,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def load_model_checkpoint(model, ckpt_path, device, filter_mismatch=True, verbose=True):
    """Replica of test.py load_model_checkpoint with text bank buffer fix."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract state dict
    state_dict = None
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            sd_source = key
            break
    if state_dict is None:
        state_dict = ckpt
        sd_source = "flat"
    # Strip 'module.' prefix
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    model_sd = model.state_dict()

    # ── Handle text_bank buffer shape mismatches ──
    _text_bank_buffers = ("_structure_text_bank_buffer", "_boundary_text_bank_buffer")
    for _buf_name in _text_bank_buffers:
        if _buf_name in state_dict and _buf_name in model_sd:
            _ckpt_shape = state_dict[_buf_name].shape
            _mdl_shape = model_sd[_buf_name].shape
            if _ckpt_shape != _mdl_shape:
                if verbose:
                    print(f"[TEXT_BANK_RESIZE] {_buf_name}: model={_mdl_shape} → checkpoint={_ckpt_shape}")
                model.register_buffer(_buf_name, torch.zeros(_ckpt_shape, device=device), persistent=True)

    # Re-capture model state dict after buffer resizing
    model_sd = model.state_dict()

    # Filter: keep only keys that exist AND have matching shape
    filtered_dict = {}
    loaded_count = 0
    skipped_missing = []
    skipped_mismatch = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue
        if filter_mismatch and v.shape != model_sd[k].shape:
            skipped_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered_dict[k] = v
        loaded_count += 1

    load_ret = model.load_state_dict(filtered_dict, strict=False)
    missing_keys = getattr(load_ret, "missing_keys", [])
    unexpected_keys = getattr(load_ret, "unexpected_keys", [])

    if verbose:
        print(f"[CKPT_LOAD] source={sd_source}")
        print(f"[CKPT_LOAD] loaded={loaded_count} | missing={len(missing_keys)} | unexpected={len(unexpected_keys)}")
        print(f"[CKPT_LOAD] skipped_missing={len(skipped_missing)} | skipped_mismatch={len(skipped_mismatch)}")
        if skipped_mismatch:
            for k, cs, ms in skipped_mismatch[:5]:
                print(f"  mismatch: {k} ckpt={cs} model={ms}")

    return model


def main():
    print("=" * 70)
    print("  CLIP CONCHLESS SMOKE TEST")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}\n")

    # ── 1. Build base model ──
    print("[1] Building base SAM model...")
    args = make_args()
    args.checkpoint = None
    vanilla_sam = sam_model_registry["vit_b"](args)
    print(f"  ✅ Base SAM built: {type(vanilla_sam).__name__}")

    # ── 2. Build TextSam with CONCHLESS mode ──
    print("\n[2] Building TextSam (CONCHLESS mode)...")
    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name="ViT-B/16",
        num_organs=21,
        num_heads=8,
        sg_epsilon=0.05,
        sg_iters=3,
        use_pnurl=False,
        use_coop=False,
        use_ot=False,
        use_asr=True,
        asr_variant="freqpath",
        asr_regression=False,
        max_semantic_gate=0.10,
        max_delta_ratio=0.10,
        init_delta_ratio=0.02,
        semantic_gate_bias_init=None,
        semantic_injection_scale=1.0,
        enable_structure_boundary_attr_heads=False,
        sb_guidance_mode="none",
        sb_guidance_weight=0.05,
        sb_guidance_routing="structure_low_boundary_high",
        enable_multilevel_attr_heads=False,
        enable_attr_text_alignment=False,
        enable_promptnu_lite_align=False,
        promptnu_lite_target="semantic_delta",
        promptnu_lite_pool_mode="gap",
        promptnu_lite_struct_weight=0.0,
        promptnu_lite_boundary_weight=0.0,
        promptnu_lite_instance_weight=0.0,
        promptnu_lite_detach_text=True,
        promptnu_lite_detach_visual=False,
        promptnu_lite_proj_lr_mult=0.5,
        enable_promptnu_guided_v3=False,
        promptnu_guided_v3_struct_weight=1.0,
        promptnu_guided_v3_boundary_weight=1.0,
        promptnu_guided_v3_text_weight=0.01,
        promptnu_guided_v3_embed_dim=256,
        promptnu_guided_v3_hidden_dim=128,
        promptnu_guided_v3_vis_proj_dim=512,
        promptnu_guided_v3_align_loss_weight=0.1,
        promptnu_guided_v3_use_text_bank=False,
        promptnu_guided_v3_use_gt_align_target=False,
        promptnu_guided_v3_semantic_dim=256,
        promptnu_guided_v3_text_dim=512,
        promptnu_guided_v3_strict_audit=False,
        promptnu_guided_v3_guidance_mode="scale_add",
        promptnu_guided_v3_scale_weight=None,
        promptnu_guided_v3_delta_weight=0.001,
        promptnu_guided_v3_delta_init_std=1e-5,
        promptnu_guided_v3_max_guided_delta_ratio=0.0,
        promptnu_guided_v3_align_eps=1e-8,
        promptnu_guided_v3_cosine_eps=1e-8,
        promptnu_guided_v3_min_align_delta_norm=0.0,
        promptnu_guided_v3_align_low_norm_mode="detach_guided",
        ablate_semantic_injection=False,
        ablate_pred_attr_guidance=False,
        promptnu_guided_v3_prompt_source="pred_attr",
        enable_numeric_attr_freqpath_guidance=False,
        numeric_attr_freqpath_hidden_dim=128,
        enable_conch_text_encoder=False,
        use_checkpoint_text_bank_without_conch=True,
        clip_text_encoder=False,
        clip_text_encoder_model="ViT-B/32",
        clip_text_encoder_cache_path="hf_cache/clip",
    ).to(device)
    del vanilla_sam
    print(f"  ✅ TextSam built: {type(model).__name__}")
    print(f"  use_checkpoint_text_bank_without_conch={model.use_checkpoint_text_bank_without_conch}")

    # ── 3. Check initial buffer shapes ──
    print("\n[3] Checking initial text bank buffer shapes...")
    struct_buf = getattr(model, "_structure_text_bank_buffer", None)
    bound_buf = getattr(model, "_boundary_text_bank_buffer", None)
    print(f"  _structure_text_bank_buffer: {tuple(struct_buf.shape) if struct_buf is not None else 'MISSING'}")
    print(f"  _boundary_text_bank_buffer:  {tuple(bound_buf.shape) if bound_buf is not None else 'MISSING'}")

    if struct_buf is not None and struct_buf.shape == (0,):
        print("  ✅ Initial shape is (0,) as expected")
    else:
        print(f"  ⚠️  Initial shape is {tuple(struct_buf.shape)}")

    # ── 4. Load checkpoint with text bank buffer resize ──
    print(f"\n[4] Loading checkpoint: {CKPT_PATH}")
    model = load_model_checkpoint(model, CKPT_PATH, device)

    # ── 5. Verify text bank buffers after loading ──
    print("\n[5] Verifying text bank buffers after loading...")
    struct_buf = getattr(model, "_structure_text_bank_buffer", None)
    bound_buf = getattr(model, "_boundary_text_bank_buffer", None)

    struct_ok = struct_buf is not None and struct_buf.numel() > 0
    bound_ok = bound_buf is not None and bound_buf.numel() > 0

    print(f"  _structure_text_bank_buffer: {'✅' if struct_ok else '❌'} "
          f"{tuple(struct_buf.shape) if struct_ok else 'EMPTY'}")
    print(f"  _boundary_text_bank_buffer:  {'✅' if bound_ok else '❌'} "
          f"{tuple(bound_buf.shape) if bound_ok else 'EMPTY'}")

    if struct_ok:
        s_norm = struct_buf.float().norm(dim=-1).mean().item()
        print(f"  structure norm (mean): {s_norm:.4f}")
    if bound_ok:
        b_norm = bound_buf.float().norm(dim=-1).mean().item()
        print(f"  boundary norm (mean):  {b_norm:.4f}")

    if not struct_ok or not bound_ok:
        print("  ❌ FAIL: Text bank buffers missing or empty!")
        sys.exit(1)
    print("  ✅ Text bank buffers correctly populated!")

    # ── 6. Verify _get_sb_text_bank ──
    print("\n[6] Testing _get_sb_text_bank()...")
    with torch.no_grad():
        struct_out = model._get_sb_text_bank("structure", device)
        bound_out = model._get_sb_text_bank("boundary", device)
    print(f"  structure_text_bank via _get_sb_text_bank: {tuple(struct_out.shape)}")
    print(f"  boundary_text_bank via _get_sb_text_bank:  {tuple(bound_out.shape)}")
    assert struct_out.shape == (5, 3, 512), f"Expected (5,3,512), got {tuple(struct_out.shape)}"
    assert bound_out.shape == (4, 3, 512), f"Expected (4,3,512), got {tuple(bound_out.shape)}"
    print("  ✅ _get_sb_text_bank() returns correct shapes!")

    # ── 7. Forward pass on 2 test images ──
    print("\n[7] Forward pass on 2 test images...")
    import cv2
    from PIL import Image

    model.eval()
    for img_path in TEST_IMAGES:
        if not os.path.isfile(img_path):
            print(f"  ⚠️  Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Preprocess
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = model.preprocess(img_tensor)

        with torch.no_grad():
            batched_input = [{
                "image": img_tensor.squeeze(0),
                "original_size": (h, w),
            }]
            outputs = model(batched_input, multimask_output=False)

        # model returns a list of dicts, one per image
        out = outputs[0]
        masks = out["low_res_logits"]
        has_nan = torch.isnan(masks).any().item()
        print(f"  {os.path.basename(img_path)}: masks={tuple(masks.shape)}, contains_nan={has_nan}")

        if has_nan:
            print("  ❌ FAIL: NaN detected in output!")
            sys.exit(1)

    print("  ✅ Forward pass OK (no NaN)!")

    print("\n" + "=" * 70)
    print("  🎉 ALL CHECKS PASSED — CLIP CONCHLESS MODE WORKS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
