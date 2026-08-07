#!/usr/bin/env python3
"""
Inspect SGA-SB v1 CORRECTION spatial structure/boundary targets from GT instance maps.

This script:
1. Loads a few samples from the dataset
2. Generates structure target (local occupancy via avg_pool2d) and boundary target (per-instance erosion)
3. Visualises both targets as colour maps
4. Reports statistics (coverage, mean, std, boundary density, etc.)

Usage:
    python scripts/inspect_spatial_sb_targets.py \
        --data_root data/MoNuSeg \
        --split train \
        --num_samples 4 \
        --out_dir workdir/spatial_sb_target_vis
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from DataLoader import UniversalDataset
from training.spatial_sb_targets import (
    generate_structure_target,
    generate_boundary_target,
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def visualize_structure_target(target: torch.Tensor, out_path: str, prefix: str = ""):
    """Save structure occupancy map as a float heatmap [0,1]."""
    target_np = target.cpu().numpy()  # [1, 64, 64] or [64, 64]
    if target_np.ndim == 3:
        target_np = target_np[0]  # [64, 64]

    # Scale to 0-255 and save
    vis = (target_np * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(vis, mode="L")
    fname = f"{prefix}structure_occupancy.png"
    img.save(os.path.join(out_path, fname))

    # Also save a colourised version (jet colormap)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(target_np, cmap="jet", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Structure Occupancy (local density)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"{prefix}structure_occupancy_cmap.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not available, skip colour version


def visualize_boundary_target(target: torch.Tensor, out_path: str, prefix: str = ""):
    """Save boundary map as a binary image."""
    target_np = target.cpu().numpy()  # [1, H, W] or [H, W]
    if target_np.ndim == 3:
        target_np = target_np[0]

    vis = (target_np * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(vis, mode="L")
    fname = f"{prefix}boundary_map.png"
    img.save(os.path.join(out_path, fname))

    # Also save a colourised overlay version
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(target_np, cmap="Reds", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Instance Boundary (per-instance erosion)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"{prefix}boundary_map_cmap.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def report_structure_stats(target: torch.Tensor, prefix: str = ""):
    """Print structure occupancy statistics."""
    target_np = target.cpu().numpy()
    if target_np.ndim == 3:
        target_np = target_np[0]

    print(f"\n  ── Structure Target Stats ──")
    print(f"    Shape: {target.shape}")
    print(f"    Min: {target_np.min():.6f}")
    print(f"    Max: {target_np.max():.6f}")
    print(f"    Mean: {target_np.mean():.6f}")
    print(f"    Std: {target_np.std():.6f}")
    print(f"    Non-zero pixels: {(target_np > 0).sum()} / {target_np.size} ({(target_np > 0).sum() / target_np.size * 100:.1f}%)")
    print(f"    Zero pixels: {(target_np == 0).sum()} / {target_np.size} ({(target_np == 0).sum() / target_np.size * 100:.1f}%)")


def report_boundary_stats(target: torch.Tensor, prefix: str = ""):
    """Print boundary statistics."""
    target_np = target.cpu().numpy()
    if target_np.ndim == 3:
        target_np = target_np[0]

    boundary_pixels = (target_np > 0.5).sum()
    total_pixels = target_np.size

    print(f"\n  ── Boundary Target Stats ──")
    print(f"    Shape: {target.shape}")
    print(f"    Boundary pixels: {boundary_pixels} / {total_pixels} ({boundary_pixels / total_pixels * 100:.2f}%)")
    print(f"    Min: {target_np.min():.6f}")
    print(f"    Max: {target_np.max():.6f}")
    print(f"    Mean: {target_np.mean():.6f}")
    print(f"    Std: {target_np.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SGA-SB v1 CORRECTION spatial structure/boundary targets")
    parser.add_argument("--data_root", type=str, default="data/MoNuSeg")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="workdir/spatial_sb_target_vis")
    parser.add_argument("--struct_kernel_size", type=int, default=31, help="Kernel size for structure occupancy avg_pool. Default: 31.")
    parser.add_argument("--bound_kernel_size", type=int, default=3, help="Kernel size for boundary erosion. Default: 3.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(args.out_dir)

    # ── Build a minimal dataset to get GT data ──
    attr_path = os.path.join(
        "workdir/attr_stats",
        f"gt_structure_boundary_attr_{args.split}.jsonl",
    )
    if not os.path.isfile(attr_path):
        print(f"[WARN] Attribute file not found: {attr_path}")
        attr_path = None

    dataset = UniversalDataset(
        data_root=args.data_root,
        split=args.split,
        image_size=1024,
        attr_path=attr_path,
        structure_boundary_attr_path=attr_path,
        enable_per_instance_attrs=True,
        enable_dense_boundary_maps=False,
    )

    print(f"\n[INFO] Dataset size: {len(dataset)}")
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # ── Process samples ──
    indices = list(range(min(args.num_samples, len(dataset))))

    for sample_idx in indices:
        sample = dataset[sample_idx]
        label_inst = sample["label_inst"]  # [1, H, W]
        sample_id = sample.get("sample_id", f"sample_{sample_idx:07d}")

        print(f"\n{'#'*60}")
        print(f"  Sample: {sample_id}")
        print(f"  label_inst shape: {label_inst.shape}")
        print(f"  Instance count: {int(label_inst.max().item())}")

        # Ensure label_inst is [1, 1, H, W] for target generation
        if label_inst.dim() == 3:
            label_inst_b = label_inst.unsqueeze(0)  # [1, 1, H, W]
        else:
            label_inst_b = label_inst

        # ── Generate structure target (local occupancy) ──
        print(f"\n  ── Generating Structure Target (kernel_size={args.struct_kernel_size}) ──")
        structure_target = generate_structure_target(
            label_inst_b,
            kernel_size=args.struct_kernel_size,
            target_size=(64, 64),
        )  # [1, 1, 64, 64]

        # ── Generate boundary target (per-instance erosion) ──
        print(f"\n  ── Generating Boundary Target (kernel_size={args.bound_kernel_size}) ──")
        boundary_target = generate_boundary_target(
            label_inst_b,
            kernel_size=args.bound_kernel_size,
            target_size=(256, 256),  # higher resolution for boundaries
        )  # [1, 1, 256, 256]

        # ── Visualise and report ──
        sample_out_dir = os.path.join(args.out_dir, sample_id)
        _ensure_dir(sample_out_dir)

        visualize_structure_target(structure_target[0], sample_out_dir, prefix="")
        report_structure_stats(structure_target[0], prefix=sample_id)

        visualize_boundary_target(boundary_target[0], sample_out_dir, prefix="")
        report_boundary_stats(boundary_target[0], prefix=sample_id)

        # ── Save instance map at low resolution for reference ──
        inst_64 = F.interpolate(
            label_inst_b.float(),
            size=(64, 64),
            mode="nearest",
        ).squeeze().cpu().numpy().astype(np.uint8)

        if inst_64.max() > 0:
            inst_vis = (inst_64.astype(np.float32) / inst_64.max() * 255).astype(np.uint8)
        else:
            inst_vis = inst_64
        Image.fromarray(inst_vis, mode="L").save(
            os.path.join(sample_out_dir, "inst_map_64x64.png")
        )

        # Save full-res instance map as reference
        full_inst = label_inst.squeeze().cpu().numpy().astype(np.uint16)
        if full_inst.max() > 0:
            full_vis = (full_inst.astype(np.float32) / full_inst.max() * 255).astype(np.uint8)
        else:
            full_vis = full_inst.astype(np.uint8)
        Image.fromarray(full_vis, mode="L").save(
            os.path.join(sample_out_dir, "inst_map_fullres.png")
        )

        print(f"\n  [OK] Saved visualisations to {sample_out_dir}")

    # ── Summary statistics across all processed samples ──
    print(f"\n{'='*60}")
    print(f"  Summary: Processed {len(indices)} samples")
    print(f"  Output directory: {args.out_dir}")
    print(f"{'='*60}")
    print("\n[DONE] Inspection complete.")


if __name__ == "__main__":
    main()
