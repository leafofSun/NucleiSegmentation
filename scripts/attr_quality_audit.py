#!/usr/bin/env python3
"""
Attr Quality Audit for PromptNu-guided v3.3

Compares predicted structure/boundary attribute logits from a trained model's
multilevel_attr_heads against ground-truth attribute labels from the
PanNuke JSONL label files.

Usage:
    # Load a trained Phase D checkpoint and audit test-split images
    python scripts/attr_quality_audit.py \
        --checkpoint workdir/models/promptnu_guided_v3_3_stablepg3_scaleadd_vitb_5ep_v1/best_aji_model.pth \
        --data_path data/PanNuke/test \
        --attr_jsonl workdir/attr_stats/gt_structure_boundary_attr_all.jsonl \
        --out_dir workdir/attr_audit

    # Dry-run mode: validate that all model components are correctly wired
    python scripts/attr_quality_audit.py --dry_run

No training, no torchrun.
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["PNURL_AUDIT_ENABLED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from segment_anything.build_sam import build_sam_vit_b

# ===================================================================
# 1. Attribute name mapping
# ===================================================================
STRUCTURE_ATTR_NAMES: Tuple[str, ...] = (
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
)

BOUNDARY_ATTR_NAMES: Tuple[str, ...] = (
    "boundary_density",
    "nuclear_irregularity",
    "nuclear_elongation",
    "touching_or_crowding_difficulty",
    "small_nuclei_ratio",
)

# Model predicts only 4 boundary attrs (dense boundary maps)
# The "small_nuclei_ratio" is instance-level only, not in patch-level logits.
MODEL_BOUNDARY_ATTR_NAMES: Tuple[str, ...] = BOUNDARY_ATTR_NAMES[:4]  # first 4

LEVEL_NAMES: Tuple[str, ...] = ("low", "mid", "high")


def make_minimal_args(**overrides) -> SimpleNamespace:
    """Create minimal argparse.Namespace for model construction."""
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
        "asr_variant": "legacy",
        "asr_regression": False,
        "max_semantic_gate": 0.10,
        "max_delta_ratio": 0.10,
        "init_delta_ratio": 0.02,
        "semantic_gate_bias_init": None,
        "semantic_injection_scale": 1.0,
        "enable_structure_boundary_attr_heads": False,
        "enable_multilevel_attr_heads": True,
        "enable_attr_text_alignment": False,
        "debug_instance_align_audit": False,
        "sb_guidance_mode": "none",
        "sb_guidance_weight": 1.0,
        "sb_conch_freeze": True,
        "sb_prompt_template_path": "workdir/attr_stats/structure_boundary_prompt_templates.json",
        "sb_guidance_routing": "structure_low_boundary_high",
        "sb_direct_adapter_hidden_dim": 64,
        "enable_conch_text_encoder": False,
        "enable_promptnu_lite_align": False,
        "promptnu_lite_target": "semantic_delta",
        "promptnu_lite_struct_weight": 0.0,
        "promptnu_lite_boundary_weight": 0.0,
        "promptnu_lite_instance_weight": 0.0,
        "promptnu_lite_detach_text": True,
        "promptnu_lite_detach_visual": False,
        "promptnu_lite_proj_lr_mult": 0.5,
        "promptnu_lite_pool_mode": "gap",
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
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def load_attr_jsonl(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load GT attr JSONL, indexed by sample_id (image stem)."""
    records: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sample_id = rec.get("sample_id", "")
            records[sample_id] = rec
    return records


def load_image(image_path: str, image_size: int = 512) -> torch.Tensor:
    """Load and preprocess an image for the model."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (image_size, image_size))
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()  # [3, H, W]
    return img_tensor


def audit_attr_quality(
    model: torch.nn.Module,
    image_files: List[str],
    gt_records: Dict[str, Dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run audit on each image. Extract predicted attr logits from model
    and compare against GT labels from attr_jsonl.
    """
    model.eval()
    torch.set_grad_enabled(False)

    # Per-attribute accumulators
    struct_correct: Dict[str, int] = {name: 0 for name in STRUCTURE_ATTR_NAMES}
    struct_total: Dict[str, int] = {name: 0 for name in STRUCTURE_ATTR_NAMES}
    bound_correct: Dict[str, int] = {name: 0 for name in MODEL_BOUNDARY_ATTR_NAMES}
    bound_total: Dict[str, int] = {name: 0 for name in MODEL_BOUNDARY_ATTR_NAMES}

    # Entropy accumulators
    struct_entropies: Dict[str, List[float]] = defaultdict(list)
    bound_entropies: Dict[str, List[float]] = defaultdict(list)

    # Confidence (max prob) accumulators
    struct_confidences: Dict[str, List[float]] = defaultdict(list)
    bound_confidences: Dict[str, List[float]] = defaultdict(list)

    processed: int = 0
    skipped: int = 0

    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Look up GT record
        gt_rec = gt_records.get(stem)
        if gt_rec is None:
            print(f"  [SKIP] No GT record for {stem}")
            skipped += 1
            continue

        # Load and prepare image
        img_tensor = load_image(img_path, args.image_size).to(device)
        batched_input = [
            {
                "image": img_tensor,
                "original_size": (args.image_size, args.image_size),
                "organ_id": 0,
                "text_prompt": "cell nuclei",
                "attribute_text": "cell nuclei",
                "morphology_text": "cell nuclei",
                "attr_labels": None,
            }
        ]

        # Forward
        outputs = model(batched_input, multimask_output=True)
        if not isinstance(outputs, list) or len(outputs) == 0:
            print(f"  [SKIP] No outputs for {stem}")
            skipped += 1
            continue

        out0 = outputs[0]

        # Extract predicted structure/boundary logits
        struct_logits: Optional[torch.Tensor] = out0.get("structure_attr_logits", None)
        bound_logits: Optional[torch.Tensor] = out0.get("boundary_attr_logits", None)

        if struct_logits is None or bound_logits is None:
            print(f"  [SKIP] No attr logits for {stem}")
            skipped += 1
            continue

        # GT discretized labels
        gt_labels = gt_rec.get("discretized_labels", {})

        # ── Structure attrs ──
        # struct_logits: [1, 5, 3]
        s_probs = torch.softmax(struct_logits[0], dim=-1)  # [5, 3]
        s_pred = s_probs.argmax(dim=-1)  # [5]

        for i, attr_name in enumerate(STRUCTURE_ATTR_NAMES):
            gt_val = gt_labels.get(attr_name)
            if gt_val is None:
                continue
            gt_val = int(gt_val)
            if gt_val < 0 or gt_val > 2:
                continue
            if s_pred[i].item() == gt_val:
                struct_correct[attr_name] += 1
            struct_total[attr_name] += 1

            # Entropy
            ent = -(s_probs[i] * torch.log(s_probs[i].clamp_min(1e-8))).sum().item()
            struct_entropies[attr_name].append(ent)
            # Confidence
            conf = s_probs[i].max().item()
            struct_confidences[attr_name].append(conf)

        # ── Boundary attrs ──
        # bound_logits: [1, 4, 3] (model predicts 4 boundary attrs)
        b_probs = torch.softmax(bound_logits[0], dim=-1)  # [4, 3]
        b_pred = b_probs.argmax(dim=-1)  # [4]

        for i, attr_name in enumerate(MODEL_BOUNDARY_ATTR_NAMES):
            gt_val = gt_labels.get(attr_name)
            if gt_val is None:
                continue
            gt_val = int(gt_val)
            if gt_val < 0 or gt_val > 2:
                continue
            if b_pred[i].item() == gt_val:
                bound_correct[attr_name] += 1
            bound_total[attr_name] += 1

            # Entropy
            ent = -(b_probs[i] * torch.log(b_probs[i].clamp_min(1e-8))).sum().item()
            bound_entropies[attr_name].append(ent)
            # Confidence
            conf = b_probs[i].max().item()
            bound_confidences[attr_name].append(conf)

        # ── V3 diagnostics if enabled ──
        _v3_debug = out0.get("promptnu_guided_v3_debug", {})
        if _v3_debug and args.verbose:
            _ps = _v3_debug.get("prompt_source", "unknown")
            _active = _v3_debug.get("v3_active", -1)
            print(f"  [V3] {stem}: prompt_source={_ps} active={_active}")

        processed += 1
        if processed % 5 == 0:
            print(f"  ... processed {processed}/{len(image_files)} images", flush=True)

    # ── Aggregate results ──
    results: Dict[str, Any] = {
        "total_images": len(image_files),
        "processed": processed,
        "skipped": skipped,
        "structure_attrs": {},
        "boundary_attrs": {},
        "summary": {},
    }

    # Structure
    struct_all_correct = 0
    struct_all_total = 0
    for name in STRUCTURE_ATTR_NAMES:
        c = struct_correct.get(name, 0)
        t = struct_total.get(name, 0)
        acc = c / max(t, 1)
        mean_ent = np.mean(struct_entropies.get(name, [float("nan")]))
        mean_conf = np.mean(struct_confidences.get(name, [float("nan")]))
        results["structure_attrs"][name] = {
            "correct": c,
            "total": t,
            "accuracy": round(acc, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_confidence": round(mean_conf, 4),
        }
        struct_all_correct += c
        struct_all_total += t

    # Boundary
    bound_all_correct = 0
    bound_all_total = 0
    for name in MODEL_BOUNDARY_ATTR_NAMES:
        c = bound_correct.get(name, 0)
        t = bound_total.get(name, 0)
        acc = c / max(t, 1)
        mean_ent = np.mean(bound_entropies.get(name, [float("nan")]))
        mean_conf = np.mean(bound_confidences.get(name, [float("nan")]))
        results["boundary_attrs"][name] = {
            "correct": c,
            "total": t,
            "accuracy": round(acc, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_confidence": round(mean_conf, 4),
        }
        bound_all_correct += c
        bound_all_total += t

    results["summary"] = {
        "structure_mean_accuracy": round(struct_all_correct / max(struct_all_total, 1), 4),
        "boundary_mean_accuracy": round(bound_all_correct / max(bound_all_total, 1), 4),
        "overall_mean_accuracy": round(
            (struct_all_correct + bound_all_correct) / max(struct_all_total + bound_all_total, 1), 4
        ),
        "structure_total_samples": struct_all_total,
        "boundary_total_samples": bound_all_total,
    }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print the audit results."""
    print("\n" + "=" * 72)
    print("  ATTR QUALITY AUDIT RESULTS")
    print("=" * 72)
    print(f"  Total images:     {results['total_images']}")
    print(f"  Processed:        {results['processed']}")
    print(f"  Skipped (no GT):  {results['skipped']}")
    print()

    print(f"  ── Structure Attrs ({len(STRUCTURE_ATTR_NAMES)}×3 levels) ──")
    for name, stats in results["structure_attrs"].items():
        print(
            f"    {name:<32s}  acc={stats['accuracy']:.4f}  "
            f"({stats['correct']}/{stats['total']})  "
            f"entropy={stats['mean_entropy']:.4f}  "
            f"conf={stats['mean_confidence']:.4f}"
        )

    print()
    print(f"  ── Boundary Attrs ({len(MODEL_BOUNDARY_ATTR_NAMES)}×3 levels) ──")
    for name, stats in results["boundary_attrs"].items():
        print(
            f"    {name:<32s}  acc={stats['accuracy']:.4f}  "
            f"({stats['correct']}/{stats['total']})  "
            f"entropy={stats['mean_entropy']:.4f}  "
            f"conf={stats['mean_confidence']:.4f}"
        )

    print()
    print(f"  ── Summary ──")
    s = results["summary"]
    print(f"    Structure mean accuracy:  {s['structure_mean_accuracy']:.4f}")
    print(f"    Boundary mean accuracy:   {s['boundary_mean_accuracy']:.4f}")
    print(f"    Overall mean accuracy:    {s['overall_mean_accuracy']:.4f}")
    print(f"    Structure samples:        {s['structure_total_samples']}")
    print(f"    Boundary samples:         {s['boundary_total_samples']}")
    print("=" * 72)
    print()


def find_test_images(data_path: str) -> List[str]:
    """Find all PNG images in the test directory."""
    png_files: List[str] = []
    for fname in sorted(os.listdir(data_path)):
        if fname.lower().endswith(".png"):
            png_files.append(os.path.join(data_path, fname))
    return png_files


def dry_run_check(args: argparse.Namespace) -> None:
    """
    Dry-run mode: construct model and validate that multilevel_attr_heads
    are present and wired correctly, without loading data.
    """
    print("[DRY_RUN] Constructing model with multilevel_attr_heads=True ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_sam_vit_b(args)
    model = model.to(device)
    model.eval()

    # Validate components
    has_ml_heads = hasattr(model, "multilevel_attr_heads") and model.multilevel_attr_heads is not None
    print(f"  multilevel_attr_heads present: {has_ml_heads}")

    if has_ml_heads:
        ml = model.multilevel_attr_heads
        print(f"    patch_structure_head:  {type(ml.patch_structure_head).__name__}")
        print(f"    dense_boundary_head:   {type(ml.dense_boundary_head).__name__}")
        print(f"    instance_morph_head:   {type(ml.instance_morph_head).__name__}")

        # Forward dummy input
        dummy_img = torch.randn(1, 3, args.image_size, args.image_size, device=device)
        dummy_batch = [
            {
                "image": dummy_img,
                "original_size": (args.image_size, args.image_size),
                "organ_id": 0,
                "text_prompt": "cell nuclei",
                "attribute_text": "cell nuclei",
                "morphology_text": "cell nuclei",
                "attr_labels": None,
            }
        ]
        with torch.no_grad():
            out = model(dummy_batch, multimask_output=True)

        if isinstance(out, list) and len(out) > 0:
            o0 = out[0]
            s_logits = o0.get("structure_attr_logits", None)
            b_logits = o0.get("boundary_attr_logits", None)
            print(f"  structure_attr_logits shape: {list(s_logits.shape) if s_logits is not None else 'None'}")
            print(f"  boundary_attr_logits shape:  {list(b_logits.shape) if b_logits is not None else 'None'}")
            print(f"  Output keys: {sorted(o0.keys())}")
        else:
            print(f"  [WARN] Forward returned unexpected type: {type(out)}")

    print("[DRY_RUN] Validation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Attr Quality Audit: compare predicted vs GT attribute labels"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, default="data/PanNuke/test",
                        help="Path to test images directory")
    parser.add_argument("--attr_jsonl", type=str,
                        default="workdir/attr_stats/gt_structure_boundary_attr_all.jsonl",
                        help="Path to GT attribute JSONL file")
    parser.add_argument("--out_dir", type=str, default="workdir/attr_audit",
                        help="Output directory for audit results")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate model components without loading data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-image diagnostics")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of test images to process")
    args = parser.parse_args()

    if args.dry_run:
        model_args = make_minimal_args()
        dry_run_check(model_args)
        return

    # ── Validate required paths ──
    if args.checkpoint is None:
        print("[ERROR] --checkpoint is required (use --dry_run to validate without checkpoint)")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.isfile(args.attr_jsonl):
        print(f"[ERROR] GT attr JSONL not found: {args.attr_jsonl}")
        sys.exit(1)
    if not os.path.isdir(args.data_path):
        print(f"[ERROR] Data path not found: {args.data_path}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Determine device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUDIT] Using device: {device}")

    # ── Build model ──
    print("[AUDIT] Building model ...")
    model_args = make_minimal_args()
    model = build_sam_vit_b(model_args)
    model = model.to(device)

    # ── Load checkpoint ──
    print(f"[AUDIT] Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing_keys={len(missing)}  unexpected_keys={len(unexpected)}")

    # ── Load GT attr labels ──
    print(f"[AUDIT] Loading GT attr labels: {args.attr_jsonl}")
    gt_records = load_attr_jsonl(args.attr_jsonl)
    print(f"  Loaded {len(gt_records)} GT records")

    # ── Find test images ──
    image_files = find_test_images(args.data_path)
    print(f"[AUDIT] Found {len(image_files)} test images in {args.data_path}")
    if args.max_images is not None:
        image_files = image_files[:args.max_images]
        print(f"  Limiting to {args.max_images} images")

    # ── Run audit ──
    print("[AUDIT] Running attribute quality audit ...")
    results = audit_attr_quality(
        model=model,
        image_files=image_files,
        gt_records=gt_records,
        device=device,
        args=args,
    )

    # ── Print results ──
    print_results(results)

    # ── Save results ──
    out_path = os.path.join(args.out_dir, "attr_quality_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[AUDIT] Results saved to: {out_path}")

    print("[AUDIT] Done.")


if __name__ == "__main__":
    main()
