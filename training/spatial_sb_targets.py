"""
SGA-SB v1 CORRECTION: Spatial Structure/Boundary Target Generation.

Two independent targets:

1. Structure target:
   - Input: label_inst [B,1,H,W]
   - Foreground = (label_inst > 0).float()
   - Local average pooling → local occupancy map
   - Resized to structure branch resolution (H_low, W_low)
   - Output: [B, 1, H_low, W_low], float, range [0, 1]

2. Boundary target:
   - Input: label_inst [B,1,H,W]
   - Per-instance boundary via mask - erode(mask)
   - Merged across all instances
   - Preserves internal boundaries between adjacent instances
   - Output: [B, 1, H_high, W_high], binary float

Only integer instance maps are required — no per-instance morphology attributes.

References:
    - structure kernel_size is set based on original image resolution
    - erosion uses 3x3 kernel for per-instance boundary extraction
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def spatial_target_alignment_method(
    target: torch.Tensor,
    prediction: torch.Tensor,
    target_kind: str,
) -> str:
    """Return the semantic-preserving alignment method after strict shape checks."""
    if target_kind not in {"structure", "boundary", "boundary_soft"}:
        raise RuntimeError(
            "[SGA_SB_TARGET_SHAPE_ERROR] "
            f"kind={target_kind} prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    if (target.ndim != 4 or prediction.ndim != 4
            or target.shape[0] != prediction.shape[0]
            or target.shape[1] != 1 or prediction.shape[1] != 1):
        raise RuntimeError(
            "[SGA_SB_TARGET_SHAPE_ERROR] "
            f"kind={target_kind} prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    target_hw = tuple(target.shape[-2:])
    prediction_hw = tuple(prediction.shape[-2:])
    if target_hw == prediction_hw:
        return "none"
    if target_hw[0] >= prediction_hw[0] and target_hw[1] >= prediction_hw[1]:
        return "adaptive_avg_pool2d" if target_kind in {"structure", "boundary_soft"} else "adaptive_max_pool2d"
    if target_hw[0] <= prediction_hw[0] and target_hw[1] <= prediction_hw[1]:
        return "bilinear" if target_kind in {"structure", "boundary_soft"} else "nearest"
    raise RuntimeError(
        "[SGA_SB_TARGET_SHAPE_ERROR] "
        f"kind={target_kind} prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
    )


def align_spatial_target_to_prediction(
    target: torch.Tensor,
    prediction: torch.Tensor,
    target_kind: str,
) -> torch.Tensor:
    """Align a 1-channel spatial target without modifying the prediction tensor."""
    method = spatial_target_alignment_method(target, prediction, target_kind)
    target = target.float()
    output_size = tuple(prediction.shape[-2:])
    if method == "adaptive_avg_pool2d":
        aligned = F.adaptive_avg_pool2d(target, output_size=output_size)
    elif method == "adaptive_max_pool2d":
        aligned = F.adaptive_max_pool2d(target, output_size=output_size)
    elif method == "bilinear":
        aligned = F.interpolate(target, size=output_size, mode="bilinear", align_corners=False)
    elif method == "nearest":
        aligned = F.interpolate(target, size=output_size, mode="nearest")
    else:
        aligned = target
    if (aligned.shape != prediction.shape or not torch.isfinite(aligned).all()
            or aligned.min() < 0 or aligned.max() > 1):
        raise RuntimeError(
            "[SGA_SB_TARGET_SHAPE_ERROR] "
            f"kind={target_kind} prediction={tuple(prediction.shape)} target={tuple(target.shape)} "
            f"aligned={tuple(aligned.shape)}"
        )
    assert aligned.shape == prediction.shape
    return aligned


def generate_structure_target(
    label_inst: torch.Tensor,
    kernel_size: int = 31,
    stride: int = 1,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Generate local occupancy map from instance map.

    Args:
        label_inst: [1, H, W] or [B, 1, H, W] integer instance map (0 = background).
        kernel_size: Size of the averaging kernel (default 31 for ~1024px images).
        stride: Stride for average pooling (default 1).
        target_size: Optional (H_low, W_low) to resize output to.

    Returns:
        structure_target: [1, 1, H_out, W_out] or [B, 1, H_out, W_out]
            float tensor in range [0, 1].
    """
    if label_inst.dim() == 3:
        label_inst = label_inst.unsqueeze(0)  # [1, 1, H, W]

    B = label_inst.shape[0]

    # Foreground mask
    foreground = (label_inst > 0).float()  # [B, 1, H, W]

    # Local average pooling → local occupancy
    # Use average pooling with the specified kernel
    padding = kernel_size // 2
    occupancy = F.avg_pool2d(
        foreground,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=False,
    )  # [B, 1, H, W] (with padding, same spatial size)

    # Resize to target resolution if specified
    if target_size is not None:
        occupancy = F.interpolate(
            occupancy,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    # Target diagnostics are intentionally opt-in; target math is unchanged.
    with torch.no_grad():
        _min = occupancy.min().item()
        _max = occupancy.max().item()
        _mean = occupancy.mean().item()
        _std = occupancy.std().item()
        if os.environ.get("SGA_SB_VERBOSE_TARGETS", "0") == "1":
            print(
                f"[STRUCTURE_TARGET] "
                f"min={_min:.6f} max={_max:.6f} mean={_mean:.6f} std={_std:.6f} "
                f"shape={tuple(occupancy.shape)} kernel_size={kernel_size}"
            )
        assert _std > 0, (
            f"[STRUCTURE_TARGET] std={_std:.6f} <= 0 — occupancy map is degenerate. "
            f"Check label_inst for empty foreground."
        )

    return occupancy  # [B, 1, H_out, W_out]


def generate_boundary_target(
    label_inst: torch.Tensor,
    kernel_size: int = 3,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Generate instance boundary map preserving internal boundaries.

    Extracts boundaries for each instance independently using
    morphological erosion, then merges all instance boundaries.

    Args:
        label_inst: [1, H, W] or [B, 1, H, W] integer instance map.
        kernel_size: Erosion kernel size (default 3).
        target_size: Optional (H_high, W_high) to resize output to.

    Returns:
        boundary_target: [1, 1, H_out, W_out] or [B, 1, H_out, W_out]
            binary float tensor.
    """
    if label_inst.dim() == 3:
        label_inst = label_inst.unsqueeze(0)  # [B, 1, H, W]

    B = label_inst.shape[0]
    device = label_inst.device
    dtype = label_inst.dtype
    H, W = label_inst.shape[-2:]

    all_boundaries = []
    for b in range(B):
        inst_map = label_inst[b, 0]  # [H, W]
        unique_ids = inst_map.unique()
        instance_ids = unique_ids[unique_ids > 0]  # exclude background

        boundary = torch.zeros((H, W), device=device, dtype=dtype)

        if instance_ids.numel() > 0:
            # Erosion kernel
            from torch.nn.functional import max_pool2d
            # We'll use max_pool2d with ones kernel for erosion:
            # erode(mask) = 1 - max_pool2d(1 - mask)
            # Simpler: use conv with all-ones kernel
            #   erode(mask)[i,j] = 1 if all kernel_size² pixels == 1

            for inst_id in instance_ids:
                mask = (inst_map == inst_id).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Erosion: min_pool = -max_pool2d(-mask)
                # Using max_pool2d with large negative for non-1 pixels
                neg_mask = -mask
                # max_pool2d with padding
                p = kernel_size // 2
                neg_eroded = F.max_pool2d(
                    neg_mask,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=p,
                )
                eroded = -neg_eroded  # [1, 1, H, W]

                # Boundary = mask - eroded
                instance_boundary = mask - eroded  # [1, 1, H, W]
                instance_boundary = (instance_boundary > 0.5).float()
                all_boundaries.append(instance_boundary)

        if len(all_boundaries) > 0:
            boundary_map = torch.stack(all_boundaries, dim=0).max(dim=0).values  # [1, 1, H, W]
        else:
            boundary_map = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

        # Alternative simpler approach: process per-image
        # Let's do it properly
        pass

    # ── Proper per-image implementation ──
    boundary_targets = []
    for b in range(B):
        inst_map = label_inst[b, 0]  # [H, W]
        unique_ids = inst_map.unique()
        instance_ids = unique_ids[unique_ids > 0]

        # Start with zeros
        full_boundary = torch.zeros((H, W), device=device, dtype=dtype)

        for inst_id in instance_ids:
            mask = (inst_map == inst_id).float()  # [H, W]
            # Erosion using conv with all-ones kernel
            mask_4d = mask.view(1, 1, H, W)
            p = kernel_size // 2
            # Use max_pool2d for erosion: a pixel is 1 only if ALL kernel_size² neighbors are 1
            # erode = (max_pool2d(-mask, k) == -1).float()
            neg_mask_4d = -mask_4d
            neg_eroded = F.max_pool2d(
                neg_mask_4d,
                kernel_size=kernel_size,
                stride=1,
                padding=p,
            )
            eroded = (-neg_eroded).clamp(0, 1)  # [1, 1, H, W]
            inst_boundary = (mask_4d - eroded).clamp(0, 1).squeeze(0).squeeze(0)  # [H, W]
            full_boundary = torch.maximum(full_boundary, inst_boundary)

        boundary_targets.append(full_boundary.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]

    boundary = torch.cat(boundary_targets, dim=0)  # [B, 1, H, W]

    if target_size is not None:
        boundary = F.interpolate(
            boundary,
            size=target_size,
            mode="nearest",  # nearest to preserve binary nature
        )

    # Diagnostics
    with torch.no_grad():
        _sum = boundary.sum().item()
        _pixel_ratio = _sum / boundary.numel()
        if os.environ.get("SGA_SB_VERBOSE_TARGETS", "0") == "1":
            print(
                f"[BOUNDARY_TARGET] "
                f"boundary_pixel_ratio={_pixel_ratio:.6f} "
                f"boundary_sum={_sum:.0f} "
                f"shape={tuple(boundary.shape)}"
            )

    return boundary  # [B, 1, H_out, W_out]


def batch_generate_spatial_sb_targets(
    batched_input: list,
    structure_kernel_size: int = 31,
    boundary_kernel_size: int = 3,
    structure_target_size: Optional[Tuple[int, int]] = None,
    boundary_target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Generate structure and boundary targets for a full batch.

    Args:
        batched_input: List of dicts as returned by DataLoader.
        structure_kernel_size: Kernel size for local occupancy pooling.
        boundary_kernel_size: Kernel size for instance boundary erosion.
        structure_target_size: (H_low, W_low) for structure target.
        boundary_target_size: (H_high, W_high) for boundary target.

    Returns:
        Dict with keys:
            "structure_target": [B, 1, H_low, W_low] or None
            "boundary_target": [B, 1, H_high, W_high] or None
    """
    label_insts = []
    for item in batched_input:
        li = item.get("label_inst", None)
        if li is None:
            return {"structure_target": None, "boundary_target": None}
        label_insts.append(li)

    # Stack to [B, 1, H, W]
    label_inst = torch.stack(label_insts, dim=0)  # [B, 1, H, W]

    # Determine target sizes from input image size if not specified
    if structure_target_size is None:
        H, W = label_inst.shape[-2:]
        # Structure target at low resolution (e.g., 1/4 of input or 64x64)
        structure_target_size = (max(H // 16, 64), max(W // 16, 64))

    if boundary_target_size is None:
        H, W = label_inst.shape[-2:]
        # Boundary target at high resolution (e.g., 1/4 of input)
        boundary_target_size = (max(H // 4, 256), max(W // 4, 256))

    structure_target = generate_structure_target(
        label_inst,
        kernel_size=structure_kernel_size,
        target_size=structure_target_size,
    )

    boundary_target = generate_boundary_target(
        label_inst,
        kernel_size=boundary_kernel_size,
        target_size=boundary_target_size,
    )

    return {
        "structure_target": structure_target,
        "boundary_target": boundary_target,
    }


# ── Smooth L1 Loss for Structure ──────────────────────────────────


def compute_structure_loss(
    structure_logits: torch.Tensor,
    structure_target: torch.Tensor,
) -> torch.Tensor:
    """Structure loss: SmoothL1(sigmoid(structure_logits), structure_target).

    Args:
        structure_logits: [B, 1, H_low, W_low] raw logits.
        structure_target: [B, 1, H_low, W_low] float occupancy [0, 1].

    Returns:
        Scalar loss tensor.
    """
    aligned_structure_target = align_spatial_target_to_prediction(
        structure_target, structure_logits, "structure"
    )
    structure_prob = torch.sigmoid(structure_logits)
    loss = F.smooth_l1_loss(structure_prob, aligned_structure_target, reduction="mean")
    return loss


def compute_boundary_loss(
    boundary_logits: torch.Tensor,
    boundary_target: torch.Tensor,
    pos_weight: Optional[float] = None,
    target_mode: str = "legacy_max",
) -> torch.Tensor:
    """Boundary loss: BCEWithLogitsLoss + Dice loss.

    Args:
        boundary_logits: [B, 1, H_high, W_high] raw logits.
        boundary_target: [B, 1, H_high, W_high] binary or soft float in [0, 1].
        pos_weight: Positive class weight for BCE. If None, computed dynamically.
        target_mode: ``legacy_max`` or ``direct_area_soft``. The latter uses
            direct adaptive-average alignment and never thresholds the target.

    Returns:
        Scalar loss tensor.
    """
    if target_mode not in {"legacy_max", "direct_area_soft"}:
        raise ValueError(f"Unknown spatial boundary target mode: {target_mode}")
    target_kind = "boundary_soft" if target_mode == "direct_area_soft" else "boundary"
    boundary_target = align_spatial_target_to_prediction(
        boundary_target, boundary_logits, target_kind
    )

    # BCE loss with dynamic pos_weight
    if pos_weight is None:
        n_pos = boundary_target.sum()
        n_neg = boundary_target.numel() - n_pos
        if n_pos > 0 and n_neg > 0:
            denominator = n_pos + 1e-6 if target_mode == "direct_area_soft" else n_pos
            pos_weight = (n_neg / denominator).clamp(0.1, 10.0).item()
        else:
            pos_weight = 1.0

    weight = torch.tensor([pos_weight], device=boundary_logits.device, dtype=boundary_logits.dtype)
    loss_bce = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_target,
        pos_weight=weight,
        reduction="mean",
    )

    # Dice loss
    boundary_prob = torch.sigmoid(boundary_logits)
    smooth = 1e-6
    intersection = (boundary_prob * boundary_target).sum()
    union = boundary_prob.sum() + boundary_target.sum()
    loss_dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    return loss_bce + loss_dice
