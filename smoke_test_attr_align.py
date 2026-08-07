#!/usr/bin/env python3
"""
Smoke test: verify that TextSam correctly creates attr_align modules
when enable_attr_text_alignment=True.

Usage:
    python smoke_test_attr_align.py

No training, no data loading, no torchrun.
"""

import os
import sys
import argparse
from types import SimpleNamespace

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress non-essential logs
os.environ["PNURL_AUDIT_ENABLED"] = "0"

from segment_anything.build_sam import build_sam_vit_b


def make_args(**overrides) -> SimpleNamespace:
    """Create a minimal argparse.Namespace with defaults compatible with build_sam_vit_b."""
    defaults = {
        # Required by build_sam_vit_b
        "image_size": 512,
        "checkpoint": None,
        "sam_checkpoint": None,
        # Encoder / Architecture
        "encoder_adapter": True,
        "use_multimodal_prompt": False,
        "clip_model_path": None,
        "num_classes": 8,
        # PNuRL / CoOp / ASR
        "use_pnurl": False,
        "use_coop_prompt": False,
        "use_coop": False,
        "use_asr": True,
        "asr_variant": "legacy",
        "asr_regression": False,
        # Semantic gating
        "max_semantic_gate": 0.10,
        "max_delta_ratio": 0.10,
        "init_delta_ratio": 0.02,
        "semantic_gate_bias_init": None,
        "semantic_injection_scale": 1.0,
        # Attribute heads
        "enable_structure_boundary_attr_heads": False,
        "enable_multilevel_attr_heads": False,
        # === THIS IS THE KEY FLAG ===
        "enable_attr_text_alignment": True,
        # Phase D audit
        "debug_instance_align_audit": False,
        # SB guidance
        "sb_guidance_mode": "none",
        "sb_guidance_weight": 1.0,
        "sb_conch_freeze": True,
        "sb_prompt_template_path": "workdir/attr_stats/structure_boundary_prompt_templates.json",
        "sb_guidance_routing": "structure_low_boundary_high",
        # CONCH
        "enable_conch_text_encoder": False,
        # PromptNu-lite v2
        "enable_promptnu_lite_align": False,
        "promptnu_lite_target": "semantic_delta",
        "promptnu_lite_struct_weight": 0.0,
        "promptnu_lite_boundary_weight": 0.0,
        "promptnu_lite_instance_weight": 0.0,
        "promptnu_lite_detach_text": True,
        "promptnu_lite_detach_visual": False,
        "promptnu_lite_proj_lr_mult": 0.5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main():
    print("=" * 70)
    print("  ATTR_ALIGN_BUILD_SMOKE")
    print("=" * 70)

    # ── 1. Build model with enable_attr_text_alignment=True ──────────────
    args = make_args()
    print(f"\n  enable_attr_text_alignment = {args.enable_attr_text_alignment}")
    print(f"  enable_promptnu_lite_align = {args.enable_promptnu_lite_align}")
    print(f"  use_pnurl                  = {args.use_pnurl}")
    print(f"  asr_variant                = {args.asr_variant}")
    print()

    # Build the model (checkpoint=None → no checkpoint loading)
    model = build_sam_vit_b(args)
    print(f"  Model type: {type(model).__name__}")
    print()

    # ── 2. Inspect model.enable_attr_text_alignment ─────────────────────
    actual_flag = getattr(model, "enable_attr_text_alignment", "MISSING")
    print(f"  model.enable_attr_text_alignment = {actual_flag}")

    # ── 3. Count attr_align parameters ──────────────────────────────────
    attr_align_params = []
    for name, param in model.named_parameters():
        if "attr_align" in name:
            attr_align_params.append((name, param.shape))

    attr_align_named_param_count = len(attr_align_params)
    print(f"  attr_align_named_param_count     = {attr_align_named_param_count}")

    # ── 4. Count attr_align in state_dict ──────────────────────────────
    attr_align_sd = {k: v.shape for k, v in model.state_dict().items() if "attr_align" in k}
    attr_align_state_dict_count = len(attr_align_sd)
    print(f"  attr_align_state_dict_count       = {attr_align_state_dict_count}")

    # ── 5. Print keys (first 20) ───────────────────────────────────────
    attr_align_keys = [k for k in model.state_dict().keys() if "attr_align" in k]
    print(f"  attr_align_keys                   = {attr_align_keys[:20]}")
    print()

    # ── 6. Verification ─────────────────────────────────────────────────
    EXPECTED_NAMED_PARAM_COUNT = 8  # 4 Linear layers × (weight + bias)
    EXPECTED_STATE_DICT_COUNT = 8

    all_ok = True
    if attr_align_named_param_count != EXPECTED_NAMED_PARAM_COUNT:
        print(f"  ❌ FAIL: attr_align_named_param_count={attr_align_named_param_count}, expected {EXPECTED_NAMED_PARAM_COUNT}")
        all_ok = False
    else:
        print(f"  ✅ PASS: attr_align_named_param_count == {EXPECTED_NAMED_PARAM_COUNT}")

    if attr_align_state_dict_count != EXPECTED_STATE_DICT_COUNT:
        print(f"  ❌ FAIL: attr_align_state_dict_count={attr_align_state_dict_count}, expected {EXPECTED_STATE_DICT_COUNT}")
        all_ok = False
    else:
        print(f"  ✅ PASS: attr_align_state_dict_count == {EXPECTED_STATE_DICT_COUNT}")

    # Check that all 4 expected modules exist
    expected_modules = [
        "attr_align_vis_proj_structure",
        "attr_align_vis_proj_boundary",
        "attr_align_vis_proj_instance",
        "attr_align_text_proj",
    ]
    for mod_name in expected_modules:
        mod = getattr(model, mod_name, None)
        if mod is None:
            print(f"  ❌ FAIL: model.{mod_name} is None (not created)")
            all_ok = False
        else:
            print(f"  ✅ PASS: model.{mod_name} = {type(mod).__name__}")

    # Check actual parameter names match expected
    expected_param_names = set()
    for mod_name in expected_modules:
        expected_param_names.add(f"{mod_name}.weight")
        expected_param_names.add(f"{mod_name}.bias")
    actual_param_names = {name for name, _ in attr_align_params}
    missing = expected_param_names - actual_param_names
    extra = actual_param_names - expected_param_names
    if missing:
        print(f"  ❌ FAIL: missing parameters: {missing}")
        all_ok = False
    if extra:
        print(f"  ❌ FAIL: extra parameters: {extra}")
        all_ok = False
    if not missing and not extra:
        print(f"  ✅ PASS: all 8 expected parameters present, no extras")

    print()
    if all_ok:
        print("  🎉 ALL CHECKS PASSED")
    else:
        print("  ❌ SOME CHECKS FAILED — see above for details")
        sys.exit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()
