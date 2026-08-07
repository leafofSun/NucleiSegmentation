#!/usr/bin/env python3
"""
SGA-SB v1 PRE-TRAIN AUDIT: Enhanced target inspection for spatial structure/boundary targets.

This script:
1. Loads 8+ samples from PanNuke train set
2. Generates structure target (local occupancy) and boundary target (per-instance erosion)
3. Reports comprehensive statistics per sample
4. Saves all required visualizations
5. Outputs a summary table

Usage:
    python scripts/inspect_spatial_sb_targets_audit.py \
        --data_root data/MoNuSeg \
        --split train \
        --num_samples 8 \
        --out_dir workdir/audits/spatial_sb_targets_corrected
"""

import argparse
import os
import sys
import json

import numpy as np
from PIL import Image

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


def main():
    parser = argparse.ArgumentParser(description="SGA-SB v1 PRE-TRAIN AUDIT: target inspection")
    parser.add_argument("--data_root", type=str, default="data/MoNuSeg")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="workdir/audits/spatial_sb_targets_corrected")
    parser.add_argument("--struct_kernel_size", type=int, default=31)
    parser.add_argument("--bound_kernel_size", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(args.out_dir)

    # ── Build dataset (skip knowledge loading, just scan for .png/.json pairs) ──
    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=os.path.join(args.data_root, "medical_knowledge.json"),
        mode=args.split,
        image_size=1024,
        skip_knowledge_loading=True,
        phase="target_inspection",
    )

    print(f"\n[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] Device: {device}")

    indices = list(range(min(args.num_samples, len(dataset))))

    # ── Summary collector ──
    summary = []

    for sample_idx in indices:
        sample = dataset[sample_idx]
        label_inst = sample["label_inst"]  # [1, H, W]
        image = sample.get("image", None)  # [3, H, W]
        sample_id = sample.get("name", sample.get("sample_id", f"sample_{sample_idx:07d}"))
        # Strip file extension from sample_id if present
        if sample_id.endswith(".png"):
            sample_id = sample_id[:-4]

        print(f"\n{'#'*65}")
        print(f"  Sample: {sample_id}  (index={sample_idx})")
        print(f"  label_inst shape: {label_inst.shape}")
        instance_count = int(label_inst.max().item())
        print(f"  Instance count: {instance_count}")

        # Ensure label_inst is [1, 1, H, W]
        if label_inst.dim() == 3:
            label_inst_b = label_inst.unsqueeze(0)  # [1, 1, H, W]
        else:
            label_inst_b = label_inst

        # ── Foreground ratio (at full res) ──
        foreground = (label_inst_b > 0).float()
        foreground_ratio = foreground.mean().item()
        print(f"  Foreground ratio: {foreground_ratio:.6f}")

        # ── Generate structure target ──
        print(f"\n  ── Structure Target (kernel_size={args.struct_kernel_size}, target=64x64) ──")
        structure_target = generate_structure_target(
            label_inst_b,
            kernel_size=args.struct_kernel_size,
            target_size=(64, 64),
        )  # [1, 1, 64, 64]

        # ── Generate boundary target ──
        print(f"\n  ── Boundary Target (kernel_size={args.bound_kernel_size}, target=256x256) ──")
        boundary_target = generate_boundary_target(
            label_inst_b,
            kernel_size=args.bound_kernel_size,
            target_size=(256, 256),
        )  # [1, 1, 256, 256]

        # ── Compute stats ──
        st = structure_target[0].cpu()  # [1, 64, 64]
        bt = boundary_target[0].cpu()   # [1, 256, 256]

        st_np = st.numpy()
        bt_np = bt.numpy()

        structure_min = float(st_np.min())
        structure_max = float(st_np.max())
        structure_mean = float(st_np.mean())
        structure_std = float(st_np.std())

        boundary_pixel_ratio = float((bt_np > 0.5).sum() / bt_np.size)
        boundary_sum = float(bt_np.sum())

        contains_nan = bool(np.isnan(st_np).any() or np.isnan(bt_np).any())
        contains_inf = bool(np.isinf(st_np).any() or np.isinf(bt_np).any())

        sample_record = {
            "sample_name": sample_id,
            "instance_count": instance_count,
            "foreground_ratio": round(foreground_ratio, 6),
            "structure_target_shape": list(structure_target.shape),
            "structure_min": round(structure_min, 6),
            "structure_max": round(structure_max, 6),
            "structure_mean": round(structure_mean, 6),
            "structure_std": round(structure_std, 6),
            "boundary_target_shape": list(boundary_target.shape),
            "boundary_pixel_ratio": round(boundary_pixel_ratio, 6),
            "boundary_sum": round(boundary_sum, 2),
            "contains_nan": contains_nan,
            "contains_inf": contains_inf,
        }
        summary.append(sample_record)

        # ── Print detailed stats ──
        print(f"\n  ── Sample Summary ──")
        for k, v in sample_record.items():
            print(f"    {k}: {v}")

        # ── Save per-sample output ──
        sample_out_dir = os.path.join(args.out_dir, sample_id)
        _ensure_dir(sample_out_dir)

        # Save original image if available
        if image is not None:
            img_np = image.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = img_np.transpose(1, 2, 0)  # [H, W, C]
                img_np = np.clip(img_np, 0, 1) if img_np.max() <= 1.0 else np.clip(img_np, 0, 255)
                img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                Image.fromarray(img_np).save(os.path.join(sample_out_dir, "original_image.png"))

        # Save integer instance map (full res)
        full_inst = label_inst.squeeze().cpu().numpy().astype(np.int32)
        # Save as normalized visualization
        if full_inst.max() > 0:
            inst_vis = (full_inst.astype(np.float32) / full_inst.max() * 255).astype(np.uint8)
        else:
            inst_vis = full_inst.astype(np.uint8)
        Image.fromarray(inst_vis, mode="L").save(os.path.join(sample_out_dir, "instance_map.png"))

        # Save binary foreground
        fg = (label_inst_b > 0).float().squeeze().cpu().numpy()
        fg_vis = (fg * 255).astype(np.uint8)
        Image.fromarray(fg_vis, mode="L").save(os.path.join(sample_out_dir, "binary_foreground.png"))

        # Save structure target (local occupancy map)
        st_vis = (st_np[0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(st_vis, mode="L").save(os.path.join(sample_out_dir, "local_occupancy_map.png"))

        # Save instance boundary map
        bt_vis = (bt_np[0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(bt_vis, mode="L").save(os.path.join(sample_out_dir, "instance_boundary_map.png"))

        # Save color versions using matplotlib if available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Structure target colormap
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(st_np[0], cmap="jet", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Structure Occupancy (local density)")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_out_dir, "local_occupancy_map_cmap.png"), dpi=150)
            plt.close(fig)

            # Boundary target colormap
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(bt_np[0], cmap="Reds", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Instance Boundary (per-instance erosion)")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_out_dir, "instance_boundary_map_cmap.png"), dpi=150)
            plt.close(fig)

            # Composite figure: original image + instance map + occupancy + boundary
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            if image is not None:
                img_display = img_np
                axes[0, 0].imshow(img_display)
                axes[0, 0].set_title("Original Image")
            else:
                axes[0, 0].text(0.5, 0.5, "No image", ha="center", va="center")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(inst_vis, cmap="nipy_spectral")
            axes[0, 1].set_title(f"Instance Map ({instance_count} instances)")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(st_np[0], cmap="jet", vmin=0.0, vmax=1.0)
            axes[1, 0].set_title(f"Structure Occupancy\nmean={structure_mean:.4f} std={structure_std:.4f}")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(bt_np[0], cmap="Reds", vmin=0.0, vmax=1.0)
            axes[1, 1].set_title(f"Instance Boundary\npixel_ratio={boundary_pixel_ratio:.4f}")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(sample_out_dir, "composite.png"), dpi=150)
            plt.close(fig)

        except ImportError:
            print("  [INFO] matplotlib not available, skipping color visualizations")

        print(f"\n  [OK] Saved visualizations to {sample_out_dir}")

    # ── Summary table ──
    print(f"\n{'='*65}")
    print(f"  TARGET INSPECTION SUMMARY — {len(summary)} samples")
    print(f"{'='*65}")

    # Header
    header_fields = [
        "sample_name", "instance_count", "foreground_ratio",
        "structure_min", "structure_max", "structure_mean", "structure_std",
        "boundary_pixel_ratio", "boundary_sum", "contains_nan", "contains_inf"
    ]
    header_str = " | ".join(f"{f:>20}" for f in header_fields)
    sep_str = "-" * len(header_str)
    print(f"  {header_str}")
    print(f"  {sep_str}")

    for rec in summary:
        row = []
        for f in header_fields:
            v = rec.get(f, "N/A")
            if isinstance(v, float):
                row.append(f"{v:>20.6f}")
            elif isinstance(v, bool):
                row.append(f"{str(v):>20}")
            elif isinstance(v, int):
                row.append(f"{v:>20}")
            else:
                row.append(f"{str(v):>20}")
        print(f"  {' | '.join(row)}")

    # ── Success criteria check ──
    print(f"\n{'='*65}")
    print(f"  SUCCESS CRITERIA CHECK")
    print(f"{'='*65}")

    all_pass = True
    checks = {
        "structure_std > 1e-4": all(r["structure_std"] > 1e-4 for r in summary),
        "structure_min < structure_max": all(r["structure_min"] < r["structure_max"] for r in summary),
        "boundary_sum > 0": all(r["boundary_sum"] > 0 for r in summary),
        "boundary_pixel_ratio not 0 and not ~1": all(0 < r["boundary_pixel_ratio"] < 0.5 for r in summary),
        "no NaN": all(not r["contains_nan"] for r in summary),
        "no Inf": all(not r["contains_inf"] for r in summary),
    }

    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {check_name}")

    # Check for dense vs sparse occupancy variation
    dense_occ = [r["structure_mean"] for r in summary[:4]]
    sparse_occ = [r["structure_mean"] for r in summary[4:]]
    occ_varies = max(dense_occ) > min(sparse_occ) if len(dense_occ) > 0 and len(sparse_occ) > 0 else True
    print(f"  [{'PASS' if occ_varies else 'WARN'}] Dense area occupancy differs from sparse areas")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")

    # ── Save summary JSON ──
    summary_path = os.path.join(args.out_dir, "target_inspection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # ── Save summary report ──
    report_path = os.path.join(args.out_dir, "target_inspection_report.txt")
    with open(report_path, "w") as f:
        f.write("SGA-SB v1 PRE-TRAIN AUDIT — Target Inspection Report\n")
        f.write(f"{'='*65}\n\n")
        for rec in summary:
            f.write(f"Sample: {rec['sample_name']}\n")
            for k, v in rec.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write(f"\nChecks:\n")
        for check_name, result in checks.items():
            f.write(f"  [{ 'PASS' if result else 'FAIL' }] {check_name}\n")
        f.write(f"\nOverall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}\n")
    print(f"  Report saved to: {report_path}")

    print(f"\n[DONE] Target inspection complete.\n")


if __name__ == "__main__":
    main()
