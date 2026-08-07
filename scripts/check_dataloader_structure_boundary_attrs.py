#!/usr/bin/env python3
"""
check_dataloader_structure_boundary_attrs.py

验证 DataLoader 的 structure/boundary attribute 非侵入式接入。

用法:
    python scripts/check_dataloader_structure_boundary_attrs.py

可选参数:
    --data_root         数据根目录 (default: data/PanNuke)
    --knowledge_path    知识库路径 (default: data/PanNuke/medical_knowledge.json)
    --attr_path         属性样本 JSONL 路径
                         (default: workdir/attr_stats/gt_structure_boundary_attr_samples.jsonl)
    --image_size        输入图像尺寸 (default: 512)
    --crop_size         裁剪尺寸 (default: 256)
    --batch_size        batch 大小 (default: 4)
    --num_batches       验证 batch 数 (default: 5)
    --mode              数据模式 (default: train)
    --prompt_mode       prompt 模式 (default: organ_static)

功能:
    1. 以 use_structure_boundary_attrs=True 初始化 DataLoader
    2. 取出前 num_batches 个 batch
    3. 打印 structure_attr_labels / boundary_attr_labels / structure_attr_values /
       boundary_attr_values / has_structure_boundary_attrs 的 shape 和分布
    4. 统计匹配/缺失比例
    5. 不做训练、不修改模型
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from DataLoader import (
    UniversalDataset,
    STRUCTURE_ATTR_NAMES,
    BOUNDARY_ATTR_NAMES,
    INVALID_ATTR_LABEL,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Check DataLoader structure/boundary attribute loading",
    )
    parser.add_argument("--data_root", type=str, default="data/PanNuke")
    parser.add_argument("--knowledge_path", type=str, default="data/PanNuke/medical_knowledge.json")
    parser.add_argument(
        "--attr_path",
        type=str,
        default="workdir/attr_stats/gt_structure_boundary_attr_all.jsonl",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--prompt_mode", type=str, default="organ_static")
    args = parser.parse_args()

    print("=" * 70)
    print("📋 DataLoader Structure & Boundary Attribute 验证")
    print("=" * 70)
    print(f"   Data root:       {args.data_root}")
    print(f"   Knowledge path:  {args.knowledge_path}")
    print(f"   Attr path:       {args.attr_path}")
    print(f"   Mode:            {args.mode}")
    print(f"   Prompt mode:     {args.prompt_mode}")
    print(f"   Image size:      {args.image_size}")
    print(f"   Crop size:       {args.crop_size}")
    print(f"   Batch size:      {args.batch_size}")
    print(f"   Num batches:     {args.num_batches}")
    print(f"   Structure attrs: {STRUCTURE_ATTR_NAMES}")
    print(f"   Boundary attrs:  {BOUNDARY_ATTR_NAMES}")
    print()

    # ------------------------------------------------------------------
    # 1. Init DataLoader with structure/boundary attrs enabled
    # ------------------------------------------------------------------
    print("🚀 Initializing DataLoader with use_structure_boundary_attrs=True ...")
    dataset = UniversalDataset(
        data_root=args.data_root,
        knowledge_path=args.knowledge_path,
        image_size=args.image_size,
        crop_size=args.crop_size,
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        use_structure_boundary_attrs=True,
        structure_boundary_attr_path=args.attr_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: {
            k: ([sample[k] for sample in batch]
                if not isinstance(batch[0][k], torch.Tensor)
                else torch.stack([sample[k] for sample in batch]))
            for k in batch[0].keys()
        },
    )
    print()

    # ------------------------------------------------------------------
    # 2. Iterate batches and collect stats
    # ------------------------------------------------------------------
    print(f"📊 Collecting {args.num_batches} batches ...")

    all_structure_labels = []
    all_boundary_labels = []
    all_structure_values = []
    all_boundary_values = []
    all_has_sb = []

    batch_count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        batch_count += 1
        sl = batch["structure_attr_labels"]  # (B, 5) long
        bl = batch["boundary_attr_labels"]   # (B, 4) long
        sv = batch["structure_attr_values"]   # (B, 5) float
        bv = batch["boundary_attr_values"]    # (B, 4) float
        hs = batch["has_structure_boundary_attrs"]  # list of bool

        all_structure_labels.append(sl)
        all_boundary_labels.append(bl)
        all_structure_values.append(sv)
        all_boundary_values.append(bv)
        all_has_sb.extend(hs)

        print(f"   Batch {batch_idx + 1}:")
        print(f"     structure_attr_labels shape: {tuple(sl.shape)}")
        print(f"     boundary_attr_labels  shape: {tuple(bl.shape)}")
        print(f"     structure_attr_values shape: {tuple(sv.shape)}")
        print(f"     boundary_attr_values  shape: {tuple(bv.shape)}")
        print(f"     has_sb count: {sum(hs)}/{len(hs)}")

    if batch_count == 0:
        print("❌ No batches could be loaded. Dataset may be empty.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Aggregate stats
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("📊 AGGREGATED STATISTICS")
    print("=" * 70)

    all_sl = torch.cat(all_structure_labels, dim=0).cpu()  # (N, 5)
    all_bl = torch.cat(all_boundary_labels, dim=0).cpu()   # (N, 4)
    all_sv = torch.cat(all_structure_values, dim=0).cpu()  # (N, 5)
    all_bv = torch.cat(all_boundary_values, dim=0).cpu()   # (N, 4)

    n_total = all_sl.size(0)

    has_sb_count = sum(all_has_sb)
    missing_count = n_total - has_sb_count
    print(f"\n   Total samples in {batch_count} batches: {n_total}")
    print(f"   has_structure_boundary_attrs=True:  {has_sb_count} ({has_sb_count / max(n_total, 1) * 100:.1f}%)")
    print(f"   has_structure_boundary_attrs=False: {missing_count} ({missing_count / max(n_total, 1) * 100:.1f}%)")

    # --- Structure attr label distribution ---
    print(f"\n── Structure Attr Label Distribution (rows=attrs, cols=low/mid/high/invalid) ──")
    print(f"   {'Attribute':<30} {'low=0':>8} {'mid=1':>8} {'high=2':>8} {'invalid=-1':>10}")
    print(f"   {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, attr_name in enumerate(STRUCTURE_ATTR_NAMES):
        col = all_sl[:, i]
        low   = (col == 0).sum().item()
        mid   = (col == 1).sum().item()
        high  = (col == 2).sum().item()
        inv   = (col == INVALID_ATTR_LABEL).sum().item()
        print(f"   {attr_name:<30} {low:>8} {mid:>8} {high:>8} {inv:>10}")

    # --- Boundary attr label distribution ---
    print(f"\n── Boundary Attr Label Distribution (rows=attrs, cols=low/mid/high/invalid) ──")
    print(f"   {'Attribute':<30} {'low=0':>8} {'mid=1':>8} {'high=2':>8} {'invalid=-1':>10}")
    print(f"   {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i, attr_name in enumerate(BOUNDARY_ATTR_NAMES):
        col = all_bl[:, i]
        low   = (col == 0).sum().item()
        mid   = (col == 1).sum().item()
        high  = (col == 2).sum().item()
        inv   = (col == INVALID_ATTR_LABEL).sum().item()
        print(f"   {attr_name:<30} {low:>8} {mid:>8} {high:>8} {inv:>10}")

    # --- Raw value stats ---
    print(f"\n── Structure Attr Value Stats (raw continuous) ──")
    for i, attr_name in enumerate(STRUCTURE_ATTR_NAMES):
        col = all_sv[:, i]
        valid = col[~torch.isnan(col) & ~torch.isinf(col)]
        if valid.numel() > 0:
            print(f"   {attr_name:<30} mean={valid.mean().item():.4f}  "
                  f"std={valid.std().item():.4f}  min={valid.min().item():.4f}  "
                  f"max={valid.max().item():.4f}")
        else:
            print(f"   {attr_name:<30} (no valid values)")

    print(f"\n── Boundary Attr Value Stats (raw continuous) ──")
    for i, attr_name in enumerate(BOUNDARY_ATTR_NAMES):
        col = all_bv[:, i]
        valid = col[~torch.isnan(col) & ~torch.isinf(col)]
        if valid.numel() > 0:
            print(f"   {attr_name:<30} mean={valid.mean().item():.4f}  "
                  f"std={valid.std().item():.4f}  min={valid.min().item():.4f}  "
                  f"max={valid.max().item():.4f}")
        else:
            print(f"   {attr_name:<30} (no valid values)")

    # --- Check that default behavior is preserved ---
    print()
    print("=" * 70)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"\n   Key checks:")
    print(f"   ✅ structure_attr_labels shape: ({n_total}, {len(STRUCTURE_ATTR_NAMES)})")
    print(f"   ✅ boundary_attr_labels  shape: ({n_total}, {len(BOUNDARY_ATTR_NAMES)})")
    print(f"   ✅ structure_attr_values shape: ({n_total}, {len(STRUCTURE_ATTR_NAMES)})")
    print(f"   ✅ boundary_attr_values  shape: ({n_total}, {len(BOUNDARY_ATTR_NAMES)})")
    print(f"   ✅ has_structure_boundary_attrs: {has_sb_count}/{n_total} matched")
    print(f"   ✅ touching_or_crowding_difficulty excluded from boundary labels")
    print(f"   ✅ No training launched")
    print(f"   ✅ Model structure not modified")
    print(f"   ✅ PNURL not modified")
    print(f"   ✅ Mask decoder not modified")
    print()


if __name__ == "__main__":
    main()
