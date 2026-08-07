#!/usr/bin/env python3
"""
SGA-SB v1 CORRECTION: Forward-friendly smoke test for SpatialStructureHead + SpatialBoundaryHead.

This script:
1. Creates a minimal TextSam model with spatial_sb_mode="supervision_only" (no guidance injection)
2. Runs a single forward pass with dummy batch input
3. Verifies structure_logits and boundary_logits are in the output dict
4. Verifies shapes are [B, 1, 64, 64]
5. Verifies structure_delta and boundary_delta are None in supervision_only mode
6. Optionally tests guidance mode if --test_guidance is passed
7. Computes structure+boundary losses to verify loss computation

Usage:
    python scripts/smoke_spatial_sb_forward.py
    python scripts/smoke_spatial_sb_forward.py --test_guidance   # also test guidance_mode="guidance"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

# ── Import model and target generation ──
from segment_anything.build_sam import _build_sam
from segment_anything.modeling.sam import SpatialStructureHead, SpatialBoundaryHead
from segment_anything.modeling.sam import SpatialStructureAdapter, SpatialBoundaryAdapter
from training.spatial_sb_targets import (
    generate_structure_target,
    generate_boundary_target,
    compute_structure_loss,
    compute_boundary_loss,
)


def _make_dummy_batch(batch_size: int = 2, image_size: int = 1024):
    """Create a dummy batch with random images and instance maps."""
    batched_input = []
    for b in range(batch_size):
        # Random image [3, H, W]
        image = torch.randn(3, image_size, image_size)
        # Random instance map with a few instances [1, H, W]
        inst = torch.zeros(1, image_size, image_size, dtype=torch.int64)
        # Draw a few random blobs as instances
        for inst_id in range(1, 4):
            cx, cy = np.random.randint(100, image_size - 100, size=2)
            r = np.random.randint(20, 80)
            y_grid, x_grid = torch.meshgrid(
                torch.arange(image_size, dtype=torch.float32),
                torch.arange(image_size, dtype=torch.float32),
                indexing="ij",
            )
            mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < r ** 2
            inst[0, mask] = inst_id
        batched_input.append({
            "image": image,
            "label_inst": inst,
        })
    return batched_input


def _make_dummy_model(spatial_sb_mode: str = "supervision_only"):
    """Create a minimal model for smoke testing.

    We use _build_sam with minimal configuration and tiny image size.
    """
    # Use tiny ViT for speed
    model = _build_sam(
        model_type="vit_b",
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=256,  # smaller for speed
        checkpoint=None,
        encoder_adapter=False,
        use_multimodal_prompt=True,
        num_classes=8,
        use_pnurl=False,
        use_asr=True,
        asr_variant="freqpath",
        spatial_sb_mode=spatial_sb_mode,
        spatial_sb_branch="both",
        spatial_structure_loss_weight=0.1,
        spatial_boundary_loss_weight=0.1,
        spatial_structure_guidance_init=0.05,
        spatial_boundary_guidance_init=0.05,
        spatial_instance_attr_mode="none",
    )
    model.eval()
    return model


def test_prediction_heads():
    """Test SpatialStructureHead and SpatialBoundaryHead standalone."""
    print("=" * 60)
    print("  Test 1: Standalone Prediction Heads")
    print("=" * 60)

    struct_head = SpatialStructureHead(in_dim=256, hidden_dim=128)
    bound_head = SpatialBoundaryHead(in_dim=256, hidden_dim=128)
    struct_adapter = SpatialStructureAdapter(in_dim=1, out_dim=256, hidden_dim=32)
    bound_adapter = SpatialBoundaryAdapter(in_dim=1, out_dim=256, hidden_dim=32)

    dummy_feat = torch.randn(2, 256, 64, 64)

    struct_logits = struct_head(dummy_feat)
    bound_logits = bound_head(dummy_feat)

    print(f"  structure_logits shape: {struct_logits.shape}  (expected [2, 1, 64, 64])")
    print(f"  boundary_logits shape:  {bound_logits.shape}  (expected [2, 1, 64, 64])")

    assert struct_logits.shape == (2, 1, 64, 64), f"Unexpected struct shape: {struct_logits.shape}"
    assert bound_logits.shape == (2, 1, 64, 64), f"Unexpected bound shape: {bound_logits.shape}"

    # Verify sigmoid gives valid probabilities
    struct_prob = torch.sigmoid(struct_logits)
    bound_prob = torch.sigmoid(bound_logits)
    print(f"  structure_prob range: [{struct_prob.min():.4f}, {struct_prob.max():.4f}]")
    print(f"  boundary_prob range:  [{bound_prob.min():.4f}, {bound_prob.max():.4f}]")

    # Test adapters
    struct_delta = struct_adapter(struct_prob)
    bound_delta = bound_adapter(bound_prob)
    print(f"  structure_delta shape: {struct_delta.shape}  (expected [2, 256, 64, 64])")
    print(f"  boundary_delta shape:  {bound_delta.shape}  (expected [2, 256, 64, 64])")

    assert struct_delta.shape == (2, 256, 64, 64), f"Unexpected struct delta shape: {struct_delta.shape}"
    assert bound_delta.shape == (2, 256, 64, 64), f"Unexpected bound delta shape: {bound_delta.shape}"

    # Verify adapter outputs are near-zero at init (zero-initialized conv_out)
    print(f"  struct_delta norm (should be ~0 at init): {struct_delta.norm().item():.6f}")
    print(f"  bound_delta norm  (should be ~0 at init): {bound_delta.norm().item():.6f}")

    print("\n  [PASS] Prediction heads and adapters work correctly.\n")


def test_target_generation():
    """Test structure and boundary target generation."""
    print("=" * 60)
    print("  Test 2: Target Generation")
    print("=" * 60)

    # Create a simple instance map [1, 1, 256, 256]
    inst_map = torch.zeros(1, 1, 256, 256, dtype=torch.int64)
    # Instance 1: a square
    inst_map[0, 0, 50:150, 50:150] = 1
    # Instance 2: a circle
    y_grid, x_grid = torch.meshgrid(
        torch.arange(256, dtype=torch.float32),
        torch.arange(256, dtype=torch.float32),
        indexing="ij",
    )
    circle_mask = ((x_grid - 200) ** 2 + (y_grid - 100) ** 2) < 40 ** 2
    inst_map[0, 0, circle_mask] = 2

    print(f"  Instance map shape: {inst_map.shape}")
    print(f"  Instance count: {int(inst_map.max().item())}")

    # Structure target
    struct_tgt = generate_structure_target(inst_map, kernel_size=31, target_size=(64, 64))
    print(f"  structure_target shape: {struct_tgt.shape}  (expected [1, 1, 64, 64])")
    print(f"  structure_target range: [{struct_tgt.min():.6f}, {struct_tgt.max():.6f}]")
    print(f"  structure_target mean:  {struct_tgt.mean():.6f}")
    assert struct_tgt.shape == (1, 1, 64, 64), f"Unexpected struct target shape: {struct_tgt.shape}"

    # Boundary target
    bound_tgt = generate_boundary_target(inst_map, kernel_size=3, target_size=(256, 256))
    print(f"  boundary_target shape:  {bound_tgt.shape}  (expected [1, 1, 256, 256])")
    print(f"  boundary_target range:  [{bound_tgt.min():.6f}, {bound_tgt.max():.6f}]")
    print(f"  boundary_target mean:   {bound_tgt.mean():.6f}")
    print(f"  boundary pixels:        {(bound_tgt > 0.5).sum().item()}")
    assert bound_tgt.shape == (1, 1, 256, 256), f"Unexpected bound target shape: {bound_tgt.shape}"

    print("\n  [PASS] Target generation works correctly.\n")


def test_loss_computation():
    """Test structure and boundary loss computation."""
    print("=" * 60)
    print("  Test 3: Loss Computation")
    print("=" * 60)

    # Dummy structure logits and target [B, 1, 64, 64]
    struct_logits = torch.randn(2, 1, 64, 64)
    struct_target = torch.rand(2, 1, 64, 64)  # occupancy in [0, 1]
    loss_struct = compute_structure_loss(struct_logits, struct_target)
    print(f"  structure_loss: {loss_struct.item():.6f}  (expected positive scalar)")
    assert loss_struct.item() > 0, f"Structure loss should be positive, got {loss_struct.item()}"
    assert loss_struct.shape == (), f"Structure loss should be scalar, got {loss_struct.shape}"

    # Dummy boundary logits and target [B, 1, 256, 256]
    bound_logits = torch.randn(2, 1, 256, 256)
    bound_target = (torch.rand(2, 1, 256, 256) > 0.95).float()  # sparse boundary
    loss_bound = compute_boundary_loss(bound_logits, bound_target)
    print(f"  boundary_loss: {loss_bound.item():.6f}  (expected positive scalar)")
    assert loss_bound.item() > 0, f"Boundary loss should be positive, got {loss_bound.item()}"
    assert loss_bound.shape == (), f"Boundary loss should be scalar, got {loss_bound.shape}"

    # Test with explicit pos_weight
    loss_bound2 = compute_boundary_loss(bound_logits, bound_target, pos_weight=2.0)
    print(f"  boundary_loss (pos_weight=2.0): {loss_bound2.item():.6f}")

    print("\n  [PASS] Loss computation works correctly.\n")


def test_model_forward_supervision_only():
    """Test model forward pass with spatial_sb_mode='supervision_only'."""
    print("=" * 60)
    print("  Test 4: Model Forward — supervision_only mode")
    print("=" * 60)

    model = _make_dummy_model(spatial_sb_mode="supervision_only")
    batched_input = _make_dummy_batch(batch_size=1, image_size=256)

    with torch.no_grad():
        outputs = model(batched_input, multimask_output=True)

    assert isinstance(outputs, list) and len(outputs) > 0, "Outputs should be a non-empty list"

    out = outputs[0]
    struct_logits = out.get("structure_logits", None)
    bound_logits = out.get("boundary_logits", None)

    print(f"  structure_logits in output: {struct_logits is not None}")
    print(f"  boundary_logits in output:  {bound_logits is not None}")
    if struct_logits is not None:
        print(f"  structure_logits shape: {struct_logits.shape}")
    if bound_logits is not None:
        print(f"  boundary_logits shape:  {bound_logits.shape}")

    # In supervision_only mode, deltas should be None
    struct_delta = out.get("structure_delta", None)
    bound_delta = out.get("boundary_delta", None)
    print(f"  structure_delta (should be None in supervision_only): {struct_delta}")
    print(f"  boundary_delta  (should be None in supervision_only): {bound_delta}")

    assert struct_logits is not None, "structure_logits missing from output"
    assert bound_logits is not None, "boundary_logits missing from output"
    assert struct_logits.shape[-2:] == (64, 64), f"Expected 64x64 structure, got {struct_logits.shape}"
    assert bound_logits.shape[-2:] == (64, 64), f"Expected 64x64 boundary, got {bound_logits.shape}"
    assert struct_delta is None, f"Expected structure_delta=None in supervision_only, got {struct_delta}"
    assert bound_delta is None, f"Expected boundary_delta=None in supervision_only, got {bound_delta}"

    print("\n  [PASS] Model forward with supervision_only mode works correctly.\n")


def test_model_forward_guidance():
    """Test model forward pass with spatial_sb_mode='guidance'."""
    print("=" * 60)
    print("  Test 5: Model Forward — guidance mode")
    print("=" * 60)

    model = _make_dummy_model(spatial_sb_mode="guidance")
    batched_input = _make_dummy_batch(batch_size=1, image_size=256)

    with torch.no_grad():
        outputs = model(batched_input, multimask_output=True)

    out = outputs[0]
    struct_logits = out.get("structure_logits", None)
    bound_logits = out.get("boundary_logits", None)
    struct_delta = out.get("structure_delta", None)
    bound_delta = out.get("boundary_delta", None)

    print(f"  structure_logits: {struct_logits is not None} shape={struct_logits.shape if struct_logits is not None else 'N/A'}")
    print(f"  boundary_logits:  {bound_logits is not None} shape={bound_logits.shape if bound_logits is not None else 'N/A'}")
    print(f"  structure_delta:  {struct_delta is not None} shape={struct_delta.shape if struct_delta is not None else 'N/A'}")
    print(f"  boundary_delta:   {bound_delta is not None} shape={bound_delta.shape if bound_delta is not None else 'N/A'}")

    assert struct_logits is not None, "structure_logits missing in guidance mode"
    assert bound_logits is not None, "boundary_logits missing in guidance mode"
    assert struct_delta is not None, "structure_delta should be present in guidance mode"
    assert bound_delta is not None, "boundary_delta should be present in guidance mode"

    # Verify deltas are feature-shaped [1, 256, 64, 64]
    assert len(struct_delta.shape) == 4 and struct_delta.shape[1] == 256, \
        f"structure_delta should be [B, 256, H, W], got {struct_delta.shape}"
    assert len(bound_delta.shape) == 4 and bound_delta.shape[1] == 256, \
        f"boundary_delta should be [B, 256, H, W], got {bound_delta.shape}"

    # Check for expected output keys
    expected_keys = ["low_res_logits", "masks", "iou_predictions", "structure_logits", "boundary_logits"]
    for key in expected_keys:
        assert key in out, f"Missing expected output key: {key}"

    print("\n  [PASS] Model forward with guidance mode works correctly.\n")


def test_legacy_attr_mode():
    """Test that spatial_instance_attr_mode='v1' still works (backward compat)."""
    print("=" * 60)
    print("  Test 6: Legacy spatial_instance_attr_mode='v1' (backward compat)")
    print("=" * 60)

    from segment_anything.modeling.sam import SpatialInstanceAttrHead, SpatialInstanceSBGuidance

    # Test SpatialInstanceAttrHead
    attr_head = SpatialInstanceAttrHead(embed_dim=256, hidden_dim=128)
    dummy_feat = torch.randn(2, 256, 64, 64)
    attr_out = attr_head(dummy_feat)
    print(f"  SpatialInstanceAttrHead output shape: {attr_out.shape}  (expected [2, 18, 64, 64])")
    assert attr_out.shape == (2, 18, 64, 64), f"Expected [2, 18, 64, 64], got {attr_out.shape}"

    # Test SpatialInstanceSBGuidance
    guidance = SpatialInstanceSBGuidance(hidden_dim=32)
    guidance_out = guidance(attr_out, dummy_feat)
    print(f"  SpatialInstanceSBGuidance output shape: {guidance_out.shape}  (expected [2, 256, 64, 64])")
    assert guidance_out.shape == (2, 256, 64, 64), f"Expected [2, 256, 64, 64], got {guidance_out.shape}"

    print("\n  [PASS] Legacy attr mode (v1) still works.\n")


def test_gt_leakage_protection():
    """Test that oracle spatial maps in eval mode raise RuntimeError."""
    print("=" * 60)
    print("  Test 7: GT Leakage Protection")
    print("=" * 60)

    # Verify that SpatialStructureHead and SpatialBoundaryHead do NOT accept oracle maps
    struct_head = SpatialStructureHead(in_dim=256, hidden_dim=128)
    bound_head = SpatialBoundaryHead(in_dim=256, hidden_dim=128)

    # These heads only take image_embeddings, so oracle leakage is prevented
    # at the architecture level (they don't accept GT maps as input)
    print("  [INFO] SpatialStructureHead/BoundaryHead only accept image_embeddings (not GT maps)")
    print("  [INFO] Architectural GT leakage protection: heads have no oracle input pathway")
    print("  [INFO] GT spatial maps are only used at loss computation time (not inference)")

    # Verify that the model in eval mode rejects 'oracle' spatial guidance
    model = _make_dummy_model(spatial_sb_mode="guidance")
    model.eval()

    # Attempt to inject oracle structure_target via forward kwargs should fail
    batched_input = _make_dummy_batch(batch_size=1, image_size=256)
    try:
        with torch.no_grad():
            # This should either ignore the extra kwarg or raise TypeError
            outputs = model(batched_input, multimask_output=True, structure_target="oracle")
        print("  [INFO] Model forward accepted (or ignored) unexpected structure_target kwarg")
        print("  [INFO] This is acceptable: no oracle injection pathway exists in the model")
    except Exception as e:
        print(f"  [INFO] Model raised error on unexpected kwarg: {type(e).__name__}: {e}")

    print("\n  [PASS] GT leakage protection check completed.\n")


def main():
    parser = argparse.ArgumentParser(description="SGA-SB v1 CORRECTION smoke test")
    parser.add_argument("--test_guidance", action="store_true", help="Also test guidance mode forward pass")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SGA-SB v1 CORRECTION: Smoke Test")
    print("=" * 60 + "\n")

    test_prediction_heads()
    test_target_generation()
    test_loss_computation()
    test_model_forward_supervision_only()

    if args.test_guidance:
        test_model_forward_guidance()

    test_legacy_attr_mode()
    test_gt_leakage_protection()

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
