#!/usr/bin/env python3
"""
SGA-SB v1 CORRECTION: Step 1 — Runtime Channel Audit.

Checks:
1. SpatialAttrHead actual output channels
2. SpatialSBGuidance expected input channels
3. Forward slice/reshape logic
4. Runs one synthetic batch through the model
5. Reports 18 vs 27 channel mismatch (if any)

Usage:
    python scripts/audit_spatial_sb_channels.py

This script does NOT run training — only synthetic forward.
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry
from segment_anything.modeling.sam import SpatialAttrHead, SpatialSBGuidance
from training.spatial_sb_targets import NUM_ATTRS, NUM_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[AUDIT] Device: {device}")

    # ──────────────────────────────────────────────────────────────────
    # 1. SpatialAttrHead channel audit
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [SPATIAL_SB_CHANNEL_AUDIT]")
    print("=" * 60)

    head = SpatialAttrHead(embed_dim=256, hidden_dim=128).to(device)
    dummy_input = torch.randn(2, 256, 64, 64, device=device)

    with torch.no_grad():
        spatial_logits = head(dummy_input)

    assert spatial_logits.ndim == 4, f"Expected 4D, got {spatial_logits.ndim}D"
    expected_channels = NUM_ATTRS * NUM_CLASSES  # 6 * 3 = 18
    assert spatial_logits.shape[1] == expected_channels, \
        f"SpatialAttrHead output channels={spatial_logits.shape[1]}, expected={expected_channels}"

    print(f"  spatial_logits_shape={tuple(spatial_logits.shape)}  # [B, {expected_channels}, 64, 64]")

    # ──────────────────────────────────────────────────────────────────
    # 2. SpatialSBGuidance channel audit
    # ──────────────────────────────────────────────────────────────────
    guidance = SpatialSBGuidance(hidden_dim=32).to(device)
    with torch.no_grad():
        guidance_out = guidance(spatial_logits)

    guidance_map = guidance_out["guidance_64"]
    print(f"  guidance_expected_channels=18 (via Conv2d(18, 32, 1))")
    print(f"  guidance_map_shape={tuple(guidance_map.shape)}  # [B, 1, 64, 64]")

    # ──────────────────────────────────────────────────────────────────
    # 3. Examine the slice/reshape logic in compute_spatial_sb_loss
    # ──────────────────────────────────────────────────────────────────
    B = spatial_logits.shape[0]
    logits_reshaped = spatial_logits.view(B, NUM_ATTRS, NUM_CLASSES, 64, 64)
    print(f"  loss_reshape: {tuple(spatial_logits.shape)} → {tuple(logits_reshaped.shape)}")
    print(f"    → 6 attribute groups × 3 class channels")

    structure_slice = slice(0, 5)     # Not used in current code — purely informational
    boundary_slice = slice(5, 9)      # Not used in current code — purely informational
    print(f"  structure_slice (informational)={structure_slice}")
    print(f"  boundary_slice (informational)={boundary_slice}")

    # ──────────────────────────────────────────────────────────────────
    # 4. Report: 18 vs 27 channel issue
    # ──────────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  CHANNEL ALIGNMENT REPORT")
    print(f"  {'─' * 50}")
    print(f"  Actual output:        {spatial_logits.shape[1]} channels (6 attrs × 3 classes)")
    print(f"  Paper claim:          27 channels (5×3 structure + 4×3 boundary)")
    print(f"  Channel GAP:          {27 - spatial_logits.shape[1]} channels")
    print(f"  {'─' * 50}")
    print(f"  [CHANNEL_MISMATCH] 18 vs 27: YES — current implementation uses")
    print(f"  6 per-instance morphology attributes × 3 classes = 18 channels,")
    print(f"  but the report claims 5×3 structure + 4×3 boundary = 27 channels.")
    print(f"  {'─' * 50}")

    # ──────────────────────────────────────────────────────────────────
    # 5. Verify that fused_image_embeddings modulation is a single map
    # ──────────────────────────────────────────────────────────────────
    fused = torch.randn(2, 256, 64, 64, device=device)
    guided = fused * (1.0 + guidance_map)
    print(f"\n  Current guidance injection: fused *= (1 + guidance_map)")
    print(f"    Before shape: {tuple(fused.shape)}")
    print(f"    After shape:  {tuple(guided.shape)}")
    print(f"    Issue: Single unified map modulates ALL channels equally.")
    print(f"    Expected: Separate low (structure) and high (boundary) maps")
    print(f"    injected into different branches of FreqPathASRBlock.")

    print(f"\n  [AUDIT COMPLETE] 18 vs 27 channel mismatch confirmed.")
    print(f"  Proceeding with SGA-SB v1 CORRECTION.\n")


if __name__ == "__main__":
    main()
