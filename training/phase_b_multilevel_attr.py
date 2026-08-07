"""
Phase B — Multilevel Attribute Warmup Training Module
======================================================

Trains the 3 sub-modules of ``MultiLevelAttributeHeads``:

  1. **Patch-level Structure & Boundary Head** — ``StructureBoundaryAttrHeads``
     - Input: GAP of image_embeddings → Linear → [B, 5, 3] + [B, 4, 3]
     - Loss: CrossEntropy (5 struct attrs × 3 classes + 4 bound attrs × 3 classes)
     - Metric: per-attribute accuracy

  2. **Dense Boundary Head** — ``DenseBoundaryHead``
     - Input: image_embeddings → 4×Conv2d → 4 dense maps [B, 1, H, W]
     - Loss: BCE + Dice
     - Metric: per-pixel F1

  3. **Instance Morphology Head** — ``InstanceMorphologyHead``
     - Input: masked_avg_pool per instance → MLP → per-instance [N_kept, 6, 3]
     - Loss: CrossEntropy (6 morph attrs × 3 classes)
     - Metric: per-attribute accuracy

Output keys consumed from ``TextSam.forward``:
  - ``out["multilevel_attr_logits"]["structure_attr_logits"]`` — [B, 5, 3]
  - ``out["multilevel_attr_logits"]["boundary_attr_logits"]`` — [B, 4, 3]
  - ``out["multilevel_attr_logits"]["dense_boundary_maps"]`` — Dict[str, [B, 1, H, W]]
  - ``out["multilevel_attr_logits"]["instance_attr_logits"]`` — Dict with per-instance logits

Training flow mirrors Phase C's ``train_one_epoch_semantic_alignment``.

Checkpoint files:
  - ``latest_multilevel_attr_model.pth`` (last epoch)
  - ``best_multilevel_attr_model.pth`` (best aggregate score)
"""

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from training.common import (
    _autocast_context,
    _is_debug_run,
    _save_checkpoint_file,
    _to_float_or_nan,
    _write_scalar_if_finite,
    unwrap_model,
)
from training.logging_utils import rank0_tqdm

from segment_anything.modeling.sam import DenseBoundaryHead, InstanceMorphologyHead

# ---------------------------------------------------------------------------
# Helper: normalize per-image instance labels to a list of Tensors
# ---------------------------------------------------------------------------
def _normalize_per_image_instance_labels(
    per_instance_attr_labels,
    *,
    device=None,
) -> Optional[List[torch.Tensor]]:
    """Normalize variable-length per-image instance labels to a list of Tensors.

    Supported input types:
        1. ``list`` / ``tuple`` — each element is a Tensor[N_i, A] (per-image).
        2. ``Tensor[B, N, A]`` — stacked batch, split into per-image list.
        3. ``Tensor[N, A]`` — single image, wrapped as single-element list.
        4. ``None`` — returns ``None``.

    Returns:
        ``List[Tensor[N_i, A]]`` with ``dtype=torch.long``, or ``None`` if input is ``None``.
        Each element is on the requested *device* (or CPU if not specified).

    Raises:
        ``RuntimeError`` if any element has unexpected shape or dtype.
    """
    if per_instance_attr_labels is None:
        return None

    # --- Case 1: list / tuple ---
    if isinstance(per_instance_attr_labels, (list, tuple)):
        result: List[torch.Tensor] = []
        for i, elem in enumerate(per_instance_attr_labels):
            if elem is None:
                # Empty-instance image
                result.append(torch.zeros(0, 0, dtype=torch.long))
                continue
            if not isinstance(elem, torch.Tensor):
                raise RuntimeError(
                    f"[PHASE_B_INST_NORM] Element {i} in per_instance_attr_labels list is "
                    f"not a Tensor: type={type(elem).__name__}."
                )
            t = elem.long()
            if device is not None:
                t = t.to(device)
            if t.dim() != 2:
                raise RuntimeError(
                    f"[PHASE_B_INST_NORM] Element {i} in per_instance_attr_labels list has "
                    f"{t.dim()}D shape (expected 2D [N_i, A]). shape={tuple(t.shape)}."
                )
            result.append(t)
        return result

    # --- Case 2: Tensor ---
    if isinstance(per_instance_attr_labels, torch.Tensor):
        t = per_instance_attr_labels
        if t.dim() == 3:
            # [B, N, A] → list of [N_i, A] per image
            B = t.shape[0]
            result = []
            for b in range(B):
                elem = t[b].long()
                if device is not None:
                    elem = elem.to(device)
                result.append(elem)
            return result
        elif t.dim() == 2:
            # [N, A] — single image, wrap in list
            elem = t.long()
            if device is not None:
                elem = elem.to(device)
            return [elem]
        else:
            raise RuntimeError(
                f"[PHASE_B_INST_NORM] per_instance_attr_labels Tensor has {t.dim()}D "
                f"shape (expected 2D [N, A] or 3D [B, N, A]). shape={tuple(t.shape)}."
            )

    raise RuntimeError(
        f"[PHASE_B_INST_NORM] Unsupported type for per_instance_attr_labels: "
        f"{type(per_instance_attr_labels).__name__}. Expected Tensor, list, or tuple."
    )


def _ensure_target_like_pred(
    target: Optional[torch.Tensor],
    pred: torch.Tensor,
    *,
    mode: str = "area",
) -> Optional[torch.Tensor]:
    """Resize *target* to match *pred* spatial dimensions.

    - Returns ``None`` if *target* is ``None``.
    - Handles 3D → 4D unsqueeze for channel-less targets.
    - Moves to ``pred.device`` and casts to ``pred.dtype``.
    - Uses ``F.interpolate(…, mode="area")`` for soft boundary targets
      (preserves 0–1 values) or ``"nearest"`` for hard labels.

    Args:
        target: Ground-truth tensor, shape ``[B, 1, H, W]`` or ``[B, H, W]``.
        pred:   Prediction tensor, shape ``[B, 1, H_pred, W_pred]``.
        mode:   Interpolation mode — ``"area"`` (default, for soft targets)
                or ``"nearest"`` (for hard labels).

    Returns:
        Resized target with shape ``[B, 1, H_pred, W_pred]``, or ``None``.
    """
    if target is None:
        return None
    # Ensure 4D [B, 1, H, W]
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.to(device=pred.device, dtype=pred.dtype)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target.float(), size=pred.shape[-2:], mode=mode)
        target = target.to(dtype=pred.dtype)
    return target


# ====================================================================
# 1. Loss Functions
# ====================================================================

def _ce_loss(
    logits: torch.Tensor,          # [N, num_attrs, num_classes]
    labels: torch.Tensor,           # [N, num_attrs]  long, values 0/1/2
    ignore_index: int = -1,
) -> Optional[torch.Tensor]:
    """Per-attribute cross-entropy loss, averaged over valid (non-ignore) entries.

    Handles empty tensors gracefully.

    Raises ``RuntimeError`` with diagnostic message if batch dimensions
    mismatch (logits N != labels N).
    """
    if logits is None or labels is None:
        return None
    if logits.numel() == 0 or labels.numel() == 0:
        return None

    N, num_attrs, num_classes = logits.shape
    if logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"[PHASE_B_CE_SHAPE] Batch dim mismatch: "
            f"logits.shape={tuple(logits.shape)} vs labels.shape={tuple(labels.shape)}. "
            f"Expected logits[0] == labels[0], got {logits.shape[0]} != {labels.shape[0]}."
        )
    logits_flat = logits.reshape(-1, num_classes)          # [N*num_attrs, num_classes]
    labels_flat = labels.reshape(-1)                       # [N*num_attrs]

    valid = labels_flat != ignore_index
    if not valid.any():
        return None

    loss = F.cross_entropy(
        logits_flat[valid],
        labels_flat[valid],
        reduction="mean",
    )
    return loss


def _bce_dice_loss(
    pred: torch.Tensor,     # [B, 1, H, W]  (logits)
    target: torch.Tensor,    # [B, 1, H, W]  float, values in [0, 1]
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Combined BCE + Dice loss for dense boundary maps.

    *target* is automatically resized to *pred* spatial resolution
    via ``_ensure_target_like_pred(…, mode="area")``, so GT
    boundary maps at full image resolution are safely downsampled
    to the coarse prediction grid.  The resulting soft target
    preserves 0–1 boundary values and is directly compatible with
    both BCE and Dice.
    """
    if pred is None or target is None:
        return None
    if pred.numel() == 0 or target.numel() == 0:
        return None

    # Resize target to match pred spatial shape (soft area downsample)
    target = _ensure_target_like_pred(target, pred, mode="area")
    if target is None:
        return None

    # BCE
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")

    # Dice (soft target compatible — no thresholding)
    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * target).sum()
    union = pred_prob.sum() + target.sum() + eps
    dice = 1.0 - (2.0 * intersection + eps) / union

    return bce_weight * bce + dice_weight * dice


def compute_structure_attr_loss(
    structure_attr_logits: torch.Tensor,   # [B, 5, 3]
    structure_attr_labels: torch.Tensor,    # [B, 5] long
) -> Optional[torch.Tensor]:
    """CE loss for patch-level structure attributes."""
    return _ce_loss(structure_attr_logits, structure_attr_labels)


def compute_boundary_attr_loss(
    boundary_attr_logits: torch.Tensor,    # [B, 4, 3]
    boundary_attr_labels: torch.Tensor,     # [B, 4] long
) -> Optional[torch.Tensor]:
    """CE loss for patch-level boundary attributes."""
    return _ce_loss(boundary_attr_logits, boundary_attr_labels)


def compute_dense_boundary_loss(
    dense_boundary_maps: Dict[str, torch.Tensor],   # Dict of 4 maps [B, 1, H, W]
    dense_boundary_map_gt: torch.Tensor,             # [B, 4, H, W] — GT dense boundary maps
) -> Dict[str, Optional[torch.Tensor]]:
    """Per-map BCE+Dice loss for 4 dense boundary maps.

    Returns dict keyed by dense boundary name
    (e.g. 'boundary_map', 'touching_region', 'small_nuclei', 'hv_gradient').

    Note: target is resized inside ``_bce_dice_loss`` via
    ``_ensure_target_like_pred`` so pred/target spatial dimensions
    are always aligned.
    """
    losses: Dict[str, Optional[torch.Tensor]] = {}
    if dense_boundary_maps is None or dense_boundary_map_gt is None:
        return losses

    for idx, name in enumerate(DenseBoundaryHead.DENSE_BOUNDARY_NAMES):
        pred_map = dense_boundary_maps.get(name, None)            # [B, 1, H, W]
        target_map = dense_boundary_map_gt[:, idx:idx + 1, :, :]  # [B, 1, H, W]
        losses[name] = _bce_dice_loss(pred_map, target_map)
    return losses


def compute_instance_attr_loss(
    instance_attr_logits: Dict[str, Any],          # output from InstanceMorphologyHead
    per_instance_attr_labels,                       # List[Tensor[N_i, 6]] or Tensor[B,N,6] or Tensor[N,6]
) -> Tuple[Optional[torch.Tensor], int, int]:
    """CE loss for instance-level morphology attributes.

    The instance_attr_logits dict provides per-instance logits
    (list of [N_kept_i, 6, 3]) and kept_positions that map into
    per_instance_attr_labels rows.  Loss is averaged across all
    valid (kept, non-ignore) instances in the batch.

    IMPORTANT: This function preserves the computation graph by
    accumulating ``loss_i`` as a tensor (not detaching via .item()).
    The returned loss tensor is directly connected to
    ``InstanceMorphologyHead.mlp`` parameters, enabling gradient flow.

    Returns:
        ``(loss, valid_instances, valid_attr_items)`` — *loss* is ``None``
        if no valid instances, otherwise a scalar Tensor connected to
        the computation graph.
        *valid_instances* counts the number of image-level instance groups
        that contributed to the loss; *valid_attr_items* counts total
        individual attribute entries (used for metric denominator).
    """
    if instance_attr_logits is None or per_instance_attr_labels is None:
        return None, 0, 0

    # --- Normalize labels to per-image list ---
    labels_list = _normalize_per_image_instance_labels(per_instance_attr_labels)
    if labels_list is None or len(labels_list) == 0:
        return None, 0, 0

    logits_list = instance_attr_logits.get("logits", None)               # List[Tensor[N_kept_i, 6, 3]]
    kept_indices = instance_attr_logits.get("kept_indices", None)         # List[Tensor[N_kept_i]]
    batch_splits = instance_attr_logits.get("batch_splits", None)         # List[int]

    if logits_list is None:
        return None, 0, 0

    B = len(labels_list)

    # total_loss must remain a tensor (connected to graph) for gradient flow
    total_loss = torch.tensor(0.0, device=labels_list[0].device if len(labels_list) > 0 else None)
    total_valid = 0
    total_valid_attrs = 0

    for b in range(B):
        if b >= len(logits_list):
            continue
        logits_i = logits_list[b]          # [N_kept_i, 6, 3]
        if logits_i.numel() == 0:
            continue

        labels_i = labels_list[b]          # [N_i, 6]
        if labels_i.numel() == 0:
            continue

        # --- Determine kept positions ---
        if kept_indices is not None and b < len(kept_indices):
            kept_pos = kept_indices[b]     # [N_kept_i]
            if kept_pos.numel() == 0:
                continue
            # Safety: kept_pos indices must be within labels_i range
            if kept_pos.max() >= labels_i.shape[0]:
                raise RuntimeError(
                    f"[PHASE_B_INST_SHAPE] kept_positions OOB at image {b}: "
                    f"max_kept_pos={kept_pos.max().item()}, "
                    f"labels_rows={labels_i.shape[0]}. "
                    f"logits_shape={tuple(logits_i.shape)}, "
                    f"labels_shape={tuple(labels_i.shape)}."
                )
            kept_labels = labels_i[kept_pos]     # [N_kept_i, 6]
        else:
            # No alignment info — assume logits N_kept == labels N_i
            kept_labels = labels_i

        # Shape diagnosis: logits N_kept should match kept_labels N_kept
        if logits_i.shape[0] != kept_labels.shape[0]:
            raise RuntimeError(
                f"[PHASE_B_INST_SHAPE] Instance count mismatch at image {b}: "
                f"logits.shape={tuple(logits_i.shape)} vs "
                f"kept_labels.shape={tuple(kept_labels.shape)}. "
                f"Expected dim-0 to match."
            )

        loss_i = _ce_loss(logits_i, kept_labels)
        if loss_i is not None:
            # IMPORTANT: Do NOT use .item() here — must preserve gradient graph.
            # Multiply by instance count so weighted averaging by instances works.
            total_loss = total_loss + loss_i * logits_i.shape[0]
            total_valid += logits_i.shape[0]
            # Count valid (non-ignore) attribute entries
            valid_mask = kept_labels != -1
            total_valid_attrs += valid_mask.sum().item()

    if total_valid > 0:
        # total_loss / total_valid preserves the gradient graph
        return total_loss / total_valid, total_valid, total_valid_attrs
    return None, 0, 0


# ====================================================================
# 2. Metric Computation
# ====================================================================

def _compute_attr_accuracy(
    logits: torch.Tensor,    # [N, num_attrs, num_classes]
    labels: torch.Tensor,    # [N, num_attrs] long
    ignore_index: int = -1,
) -> Dict[str, float]:
    """Compute per-attribute accuracy.

    Returns dict like ``{"attr_0": 0.85, "attr_1": 0.73, ...}``.
    """
    acc: Dict[str, float] = {}
    if logits is None or labels is None:
        return acc
    if logits.numel() == 0 or labels.numel() == 0:
        return acc

    preds = torch.argmax(logits, dim=-1)   # [N, num_attrs]
    for attr_idx in range(logits.shape[1]):
        valid = labels[:, attr_idx] != ignore_index
        if valid.sum() == 0:
            acc[f"attr_{attr_idx}"] = float("nan")
        else:
            correct = (preds[valid, attr_idx] == labels[valid, attr_idx]).sum().item()
            total = valid.sum().item()
            acc[f"attr_{attr_idx}"] = correct / max(total, 1)
    return acc


def _compute_attr_f1(
    logits: torch.Tensor,       # [N, num_attrs, num_classes]
    labels: torch.Tensor,       # [N, num_attrs] long
    num_classes: int = 3,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """Compute macro-averaged F1 per attribute (3-class).

    For each attribute column, computes per-class precision/recall/F1
    and returns macro F1 (average across classes).  Returns dict like
    ``{"attr_0": 0.72, "attr_1": 0.65, ...}``.
    """
    f1_per_attr: Dict[str, float] = {}
    if logits is None or labels is None:
        return f1_per_attr
    if logits.numel() == 0 or labels.numel() == 0:
        return f1_per_attr

    preds = torch.argmax(logits, dim=-1)  # [N, num_attrs]
    for attr_idx in range(logits.shape[1]):
        valid = labels[:, attr_idx] != ignore_index
        if valid.sum() == 0:
            f1_per_attr[f"attr_{attr_idx}"] = float("nan")
            continue
        y_true = labels[valid, attr_idx]
        y_pred = preds[valid, attr_idx]
        per_class_f1 = []
        for c in range(num_classes):
            tp = ((y_pred == c) & (y_true == c)).sum().item()
            fp = ((y_pred == c) & (y_true != c)).sum().item()
            fn = ((y_pred != c) & (y_true == c)).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
            per_class_f1.append(f1)
        f1_per_attr[f"attr_{attr_idx}"] = float(np.mean(per_class_f1))
    return f1_per_attr


def _compute_label_histogram(
    labels: torch.Tensor,       # [N, num_attrs] long, values 0/1/2 or -1 (ignore)
    num_classes: int = 3,
    ignore_index: int = -1,
) -> Dict[str, List[int]]:
    """Compute class-count histogram per attribute column.

    Returns dict like ``{"attr_0": [34, 120, 55], "attr_1": [12, 98, 89], ...}``
    where each list has counts for classes [0, 1, 2] (ignoring -1 entries).
    """
    hist: Dict[str, List[int]] = {}
    if labels is None or labels.numel() == 0:
        return hist
    for attr_idx in range(labels.shape[1]):
        col = labels[:, attr_idx]
        valid = col != ignore_index
        if valid.sum() == 0:
            hist[f"attr_{attr_idx}"] = [0, 0, 0]
        else:
            counts = []
            for c in range(num_classes):
                counts.append(int((col[valid] == c).sum().item()))
            hist[f"attr_{attr_idx}"] = counts
    return hist


def _compute_dense_boundary_f1(
    pred_map: torch.Tensor,    # [B, 1, H, W] logits
    target_map: torch.Tensor,   # [B, 1, H, W] float [0,1]
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Per-pixel F1 score for a single dense boundary map.

    *target_map* is automatically resized to *pred_map* spatial
    resolution via ``_ensure_target_like_pred(…, mode="area")``
    to ensure shape alignment before metric computation.
    """
    if pred_map is None or target_map is None:
        return float("nan")
    # Resize target to match pred spatial shape
    target_map = _ensure_target_like_pred(target_map, pred_map, mode="area")
    if target_map is None:
        return float("nan")
    pred_bin = (torch.sigmoid(pred_map) > threshold).float()
    target_bin = (target_map > threshold).float()
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return f1


def compute_dense_boundary_metrics(
    dense_boundary_maps: Dict[str, torch.Tensor],
    dense_boundary_map_gt: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-map F1 for all 4 dense boundary maps."""
    metrics: Dict[str, float] = {}
    if dense_boundary_maps is None or dense_boundary_map_gt is None:
        return metrics
    for idx, name in enumerate(DenseBoundaryHead.DENSE_BOUNDARY_NAMES):
        pred = dense_boundary_maps.get(name, None)
        target = dense_boundary_map_gt[:, idx:idx + 1, :, :]
        metrics[f"{name}_f1"] = _compute_dense_boundary_f1(pred, target)
    return metrics


# ====================================================================
# 3. Train One Epoch
# ====================================================================

# Default loss weight configuration for Phase B
DEFAULT_PHASE_B_LOSS_WEIGHTS = {
    "structure_attr": 1.0,
    "boundary_attr": 1.0,
    "dense_boundary": 1.0,
    "instance_attr": 1.0,
}


def _to_device(batch_input, device):
    """Recursively move tensors in a nested dict/list to device."""
    if isinstance(batch_input, torch.Tensor):
        return batch_input.to(device)
    elif isinstance(batch_input, dict):
        return {k: _to_device(v, device) for k, v in batch_input.items()}
    elif isinstance(batch_input, (list, tuple)):
        return [_to_device(v, device) for v in batch_input]
    else:
        return batch_input


def train_one_epoch_multilevel_attr(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: Any,
    logger: Any,
    writer: Any,
    rank: int,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Train Phase B — compute structure/boundary/instance attribute loss.

    Uses batch-level ``multilevel_attr_logits`` output.
    Returns dict of training metrics.
    """
    model.train()

    # ── None-safe loss_weights ──
    loss_weights = dict(loss_weights or {})  # type: ignore[arg-type]

    if rank == 0:
        _lw_struct = float(loss_weights.get("structure_attr", 1.0))
        _lw_bound = float(loss_weights.get("boundary_attr", 1.0))
        _lw_dense = float(loss_weights.get("dense_boundary", 1.0))
        _lw_inst = float(loss_weights.get("instance_attr", 1.0))
        if logger is not None:
            logger.info(
                f"[PHASE_B_ML_LOSS_WEIGHTS] structure_attr={_lw_struct} | "
                f"boundary_attr={_lw_bound} | dense_boundary={_lw_dense} | "
                f"instance_attr={_lw_inst}"
            )

    # --- Batch limit logic: debug_max_* takes priority over max_* ---
    _debug_max_train = getattr(args, "debug_max_train_batches", None)
    _max_train = getattr(args, "max_train_batches", None)
    if _debug_max_train is not None and int(_debug_max_train) > 0:
        train_limit = int(_debug_max_train)
        _limit_source = "debug_max_train_batches"
    elif _max_train is not None and int(_max_train) > 0:
        train_limit = int(_max_train)
        _limit_source = "max_train_batches"
    else:
        train_limit = None
        _limit_source = "none"

    effective_total = min(len(data_loader), train_limit) if train_limit else len(data_loader)

    if rank == 0:
        if logger is not None:
            logger.info(
                f"[PHASE_B_ML_LIMIT] train_limit={train_limit} | source={_limit_source} | "
                f"effective_total={effective_total} | total_dataloader={len(data_loader)}"
            )
        from tqdm import tqdm
        pbar = tqdm(data_loader, desc=f"Phase B Ep {epoch + 1} Train", total=effective_total)
    else:
        pbar = data_loader

    # Accumulators
    loss_struct_list: List[float] = []
    loss_bound_list: List[float] = []
    loss_dense_list: Dict[str, List[float]] = {}  # name -> list of values
    loss_inst_list: List[float] = []
    loss_total_list: List[float] = []

    # Per-attribute accuracy accumulators
    struct_correct: Dict[int, int] = {}
    struct_total: Dict[int, int] = {}
    bound_correct: Dict[int, int] = {}
    bound_total: Dict[int, int] = {}
    inst_correct: Dict[int, int] = {}
    inst_total: Dict[int, int] = {}

    # Dense boundary F1 accumulators
    dense_f1_accum: Dict[str, float] = {}
    dense_f1_count: Dict[str, int] = {}

    # ── Delta audit accumulators (per-submodule param change per step) ──
    _delta_audit_accum: Dict[str, float] = {}
    _delta_audit_steps: int = 0
    _raw_model_for_delta = unwrap_model(model)
    _ml_heads_for_delta = getattr(_raw_model_for_delta, "multilevel_attr_heads", None)
    if _ml_heads_for_delta is not None:
        for _sub_name in ("patch_structure_head", "dense_boundary_head", "instance_morph_head"):
            _sub_mod = getattr(_ml_heads_for_delta, _sub_name, None)
            if _sub_mod is not None and any(p.requires_grad for p in _sub_mod.parameters()):
                _delta_audit_accum[_sub_name] = 0.0

    optimizer.zero_grad(set_to_none=True)

    # Reset per-epoch audit flags (local variables, not module-level globals)
    dense_shape_audited = False
    instance_shape_audited = False
    instance_label_hist_audited = False

    for batch_idx, batched_input in enumerate(pbar):
        if train_limit is not None and batch_idx >= train_limit:
            break

        # Move all tensors to device
        batched_input = _to_device(batched_input, device)
        images = batched_input["image"]

        # ----- Build model_input (same format as train_one_epoch) -----
        organ_ids = batched_input.get("organ_id", None)
        attr_labels = batched_input.get("attr_labels", None)
        dynamic_text = batched_input.get("text_prompt", ["Cell nuclei"] * len(images))
        dynamic_attr_text = batched_input.get("attribute_text", ["Cell nuclei"] * len(images))

        model_input = []
        for i in range(len(images)):
            curr_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val
            entry = {
                "image": images[i],
                "original_size": (args.image_size, args.image_size),
                "organ_id": curr_id,
                "attribute_text": dynamic_attr_text[i],
                "text_prompt": dynamic_text[i],
                "attr_labels": attr_labels[i] if attr_labels is not None else None,
            }
            # Phase B: pass instance mask + attribute labels through for MultiLevelAttributeHeads
            lbl_inst = batched_input.get("label_inst", None)
            if lbl_inst is not None:
                entry["label_inst"] = lbl_inst[i]
            s_labels = batched_input.get("structure_attr_labels", None)
            if s_labels is not None:
                entry["structure_attr_labels"] = s_labels[i]
            b_labels = batched_input.get("boundary_attr_labels", None)
            if b_labels is not None:
                entry["boundary_attr_labels"] = b_labels[i]
            # per_instance_attr_labels passed for instance alignment (used by Phase C, but also available)
            pi_labels = batched_input.get("per_instance_attr_labels", None)
            if pi_labels is not None:
                entry["per_instance_attr_labels"] = pi_labels[i]
            model_input.append(entry)

        # ----- GT labels from batched_input -----
        structure_attr_labels = batched_input.get("structure_attr_labels", None)      # [B, 5]
        boundary_attr_labels = batched_input.get("boundary_attr_labels", None)        # [B, 4]
        dense_boundary_map_gt = batched_input.get("dense_boundary_map", None)         # [B, 4, H, W]
        per_instance_attr_labels = batched_input.get("per_instance_attr_labels", None)  # [B, N_i, 6]

        with _autocast_context(args):
            outputs = model(model_input, multimask_output=True)

            # All outputs share the same common_output (batch-level tensors).
            # The multilevel_attr_logits dict is shared across all samples.
            ml_out = outputs[0].get("multilevel_attr_logits", None) if outputs else None

            # Batch-level loss accumulators
            batch_struct_loss = torch.tensor(0.0, device=device)
            batch_bound_loss = torch.tensor(0.0, device=device)
            batch_dense_loss = torch.tensor(0.0, device=device)
            batch_inst_loss = torch.tensor(0.0, device=device)
            batch_total_loss = torch.tensor(0.0, device=device)

            if ml_out is not None:
                # ----- 1. Structure Attr Loss (CE) [B, 5, 3] -----
                s_logits = ml_out.get("structure_attr_logits", None)  # [B, 5, 3]
                if s_logits is not None and structure_attr_labels is not None:
                    s_loss = compute_structure_attr_loss(s_logits, structure_attr_labels)
                    if s_loss is not None:
                        w_struct = loss_weights.get("structure_attr", 1.0)
                        batch_struct_loss = batch_struct_loss + s_loss * w_struct
                        loss_struct_list.append(s_loss.item())

                        # Per-attribute accuracy
                        s_preds = torch.argmax(s_logits, dim=-1)  # [B, 5]
                        for attr_idx in range(s_logits.shape[1]):
                            valid = structure_attr_labels[:, attr_idx] >= 0
                            n_valid = valid.sum().item()
                            if n_valid > 0:
                                struct_total[attr_idx] = struct_total.get(attr_idx, 0) + n_valid
                                n_correct = (s_preds[valid, attr_idx] == structure_attr_labels[valid, attr_idx]).sum().item()
                                struct_correct[attr_idx] = struct_correct.get(attr_idx, 0) + n_correct

                # ----- 2. Boundary Attr Loss (CE) [B, 4, 3] -----
                b_logits = ml_out.get("boundary_attr_logits", None)  # [B, 4, 3]
                if b_logits is not None and boundary_attr_labels is not None:
                    b_loss = compute_boundary_attr_loss(b_logits, boundary_attr_labels)
                    if b_loss is not None:
                        w_bound = loss_weights.get("boundary_attr", 1.0)
                        batch_bound_loss = batch_bound_loss + b_loss * w_bound
                        loss_bound_list.append(b_loss.item())

                        # Per-attribute accuracy
                        b_preds = torch.argmax(b_logits, dim=-1)  # [B, 4]
                        for attr_idx in range(b_logits.shape[1]):
                            valid = boundary_attr_labels[:, attr_idx] >= 0
                            n_valid = valid.sum().item()
                            if n_valid > 0:
                                bound_total[attr_idx] = bound_total.get(attr_idx, 0) + n_valid
                                n_correct = (b_preds[valid, attr_idx] == boundary_attr_labels[valid, attr_idx]).sum().item()
                                bound_correct[attr_idx] = bound_correct.get(attr_idx, 0) + n_correct

                # ----- 3. Dense Boundary Loss (BCE+Dice) per map -----
                dense_maps = ml_out.get("dense_boundary_maps", None)  # Dict[str, [B, 1, H, W]]
                if dense_maps is not None and dense_boundary_map_gt is not None:
                    # ---- Shape audit (rank0-only, first batch of epoch) ----
                    if rank == 0 and not dense_shape_audited:
                        sample_pred = None
                        for name in DenseBoundaryHead.DENSE_BOUNDARY_NAMES:
                            sample_pred = dense_maps.get(name, None)
                            if sample_pred is not None:
                                break
                        if sample_pred is not None:
                            target_raw = dense_boundary_map_gt[:, 0:1, :, :]
                            target_resized = _ensure_target_like_pred(target_raw, sample_pred, mode="area")
                            print(
                                f"[PHASE_B_ML_DENSE_AUDIT] pred_shape={tuple(sample_pred.shape)} | "
                                f"target_shape_raw={tuple(dense_boundary_map_gt[:, 0:1, :, :].shape)} | "
                                f"target_shape_resized={tuple(target_resized.shape) if target_resized is not None else 'N/A'}"
                            )
                        dense_shape_audited = True

                    dense_losses = compute_dense_boundary_loss(dense_maps, dense_boundary_map_gt)
                    for name, d_loss in dense_losses.items():
                        if d_loss is not None:
                            w_dense = loss_weights.get("dense_boundary", 1.0)
                            batch_dense_loss = batch_dense_loss + d_loss * w_dense
                            if name not in loss_dense_list:
                                loss_dense_list[name] = []
                            loss_dense_list[name].append(d_loss.item())

                    # Dense boundary F1 metrics (for logging)
                    for idx, name in enumerate(DenseBoundaryHead.DENSE_BOUNDARY_NAMES):
                        pred_map = dense_maps.get(name, None)
                        if pred_map is not None:
                            target_map = dense_boundary_map_gt[:, idx:idx + 1, :, :]
                            f1 = _compute_dense_boundary_f1(pred_map, target_map)
                            if not np.isnan(f1):
                                dense_f1_accum[name] = dense_f1_accum.get(name, 0.0) + f1
                                dense_f1_count[name] = dense_f1_count.get(name, 0) + 1

                # ----- 4. Instance Morphology Loss (CE) per-instance -----
                inst_out = ml_out.get("instance_attr_logits", None)
                if inst_out is not None and per_instance_attr_labels is not None:
                    # ---- Instance shape audit (rank0-only, first batch of epoch) ----
                    if rank == 0 and not instance_shape_audited:
                        labels_type = type(per_instance_attr_labels).__name__
                        if isinstance(per_instance_attr_labels, (list, tuple)):
                            label_shapes = [tuple(t.shape) for t in per_instance_attr_labels]
                        elif isinstance(per_instance_attr_labels, torch.Tensor):
                            label_shapes = list(per_instance_attr_labels.shape)
                        else:
                            label_shapes = "unknown"
                        logits_obj = inst_out.get("logits", None)
                        logits_shape_info = (
                            f"list_len={len(logits_obj)}" if isinstance(logits_obj, list)
                            else str(tuple(logits_obj.shape)) if logits_obj is not None
                            else "None"
                        )
                        kept_pos = inst_out.get("kept_indices", None)
                        kept_type = type(kept_pos).__name__ if kept_pos is not None else "None"
                        print(
                            f"[PHASE_B_ML_INSTANCE_AUDIT] labels_type={labels_type} | "
                            f"batch={len(per_instance_attr_labels) if isinstance(per_instance_attr_labels, (list, tuple)) else (per_instance_attr_labels.shape[0] if isinstance(per_instance_attr_labels, torch.Tensor) else '?')} | "
                            f"label_shapes={label_shapes} | "
                            f"logits_shape={logits_shape_info} | "
                            f"kept_positions_type={kept_type}"
                        )
                        instance_shape_audited = True

                    # Normalize labels to per-image list for subsequent processing
                    normed_labels = _normalize_per_image_instance_labels(per_instance_attr_labels)
                    i_loss, valid_inst, valid_attrs = compute_instance_attr_loss(inst_out, per_instance_attr_labels)
                    if i_loss is not None:
                        w_inst = loss_weights.get("instance_attr", 1.0)
                        batch_inst_loss = batch_inst_loss + i_loss * w_inst
                        loss_inst_list.append(i_loss.item())

                        # ---- Label histogram audit (rank0-only, first batch) ----
                        if rank == 0 and not instance_label_hist_audited and normed_labels is not None:
                            _all_inst_labels = []
                            for nl in normed_labels:
                                if nl.numel() > 0:
                                    _all_inst_labels.append(nl)
                            if _all_inst_labels:
                                _cat_labels = torch.cat(_all_inst_labels, dim=0)  # [total_N, 6]
                                _hist = _compute_label_histogram(_cat_labels)
                                print(f"[PHASE_B_ML_INSTANCE_LABEL_HIST] per-attribution class counts [c0,c1,c2]:")
                                for _attr_name, _counts in _hist.items():
                                    _total = sum(_counts)
                                    _pct = [f"{c/max(_total,1)*100:.1f}%" for c in _counts]
                                    print(f"  {_attr_name}: {_counts} ({', '.join(_pct)})")
                                # Also log valid_inst / valid_attrs
                                print(f"[PHASE_B_ML_INSTANCE_VALID] valid_instances={valid_inst} | valid_attr_items={valid_attrs}")
                            instance_label_hist_audited = True

                        # Per-attribute accuracy and F1 for instances (using normalized labels)
                        logits_list = inst_out.get("logits", None)
                        kept_indices = inst_out.get("kept_indices", None)
                        if logits_list is not None and normed_labels is not None:
                            for b in range(min(len(logits_list), len(normed_labels))):
                                logits_i = logits_list[b]
                                if logits_i.numel() == 0:
                                    continue
                                labels_i = normed_labels[b]
                                if labels_i.numel() == 0:
                                    continue
                                if kept_indices is not None and b < len(kept_indices):
                                    kept_idx = kept_indices[b]
                                    if kept_idx.numel() == 0:
                                        continue
                                    if kept_idx.max() < labels_i.shape[0]:
                                        kept_labels = labels_i[kept_idx]
                                    else:
                                        continue
                                else:
                                    kept_labels = labels_i
                                if logits_i.shape[0] != kept_labels.shape[0]:
                                    continue
                                i_preds = torch.argmax(logits_i, dim=-1)
                                for attr_idx in range(logits_i.shape[1]):
                                    valid = kept_labels[:, attr_idx] >= 0
                                    if valid.sum() > 0:
                                        inst_total[attr_idx] = inst_total.get(attr_idx, 0) + valid.sum().item()
                                        n_correct = (i_preds[valid, attr_idx] == kept_labels[valid, attr_idx]).sum().item()
                                        inst_correct[attr_idx] = inst_correct.get(attr_idx, 0) + n_correct

            # ----- Combine losses and backward -----
            batch_total_loss = batch_struct_loss + batch_bound_loss + batch_dense_loss + batch_inst_loss

            if batch_total_loss.item() > 0:
                # Normalize by batch size so gradient scale is comparable across batch sizes
                B = len(images)
                scaled_loss = batch_total_loss / max(B, 1)
                scaled_loss.backward()

        # Gradient clipping (same as train_one_epoch)
        if hasattr(args, "grad_clip") and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # ── Delta audit: snapshot params before optimizer.step() ──
        if _delta_audit_accum and _ml_heads_for_delta is not None:
            _delta_before: Dict[str, Dict[str, torch.Tensor]] = {}
            for _sub_name in _delta_audit_accum:
                _sub_mod = getattr(_ml_heads_for_delta, _sub_name, None)
                if _sub_mod is not None:
                    _delta_before[_sub_name] = {
                        n: p.data.clone()
                        for n, p in _sub_mod.named_parameters()
                        if p.requires_grad
                    }

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Delta audit: compute delta after optimizer.step() ──
        if _delta_audit_accum and _ml_heads_for_delta is not None and _delta_before:
            for _sub_name, _before_dict in _delta_before.items():
                _sub_mod = getattr(_ml_heads_for_delta, _sub_name, None)
                if _sub_mod is not None:
                    _step_delta = 0.0
                    for n, p in _sub_mod.named_parameters():
                        if p.requires_grad and n in _before_dict:
                            _step_delta += (p.data - _before_dict[n]).norm().item()
                    _delta_audit_accum[_sub_name] += _step_delta
            _delta_audit_steps += 1
            # Periodic per-step logging (first 10 steps, then every 500)
            if rank == 0 and (_delta_audit_steps <= 10 or _delta_audit_steps % 500 == 0):
                _step_msg = " | ".join(
                    f"{_name}_delta={_delta_audit_accum[_name]/max(_delta_audit_steps, 1):.8f}"
                    for _name in sorted(_delta_audit_accum.keys())
                )
                print(f"[PHASE_B_ML_DELTA_AUDIT][step={batch_idx}] {_step_msg}")

        loss_total_list.append(batch_total_loss.item())

        # Writer logging every 10 batches
        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            step = epoch * len(data_loader) + batch_idx
            if loss_struct_list:
                _write_scalar_if_finite(writer, "PhaseB_Train/structure_attr_loss", loss_struct_list[-1], step)
            if loss_bound_list:
                _write_scalar_if_finite(writer, "PhaseB_Train/boundary_attr_loss", loss_bound_list[-1], step)
            _write_scalar_if_finite(writer, "PhaseB_Train/total_loss", batch_total_loss.item(), step)

    # ----- Epoch summary -----
    epoch_stats: Dict[str, float] = {}
    if loss_struct_list:
        epoch_stats["structure_attr_loss"] = float(np.mean(loss_struct_list))
    if loss_bound_list:
        epoch_stats["boundary_attr_loss"] = float(np.mean(loss_bound_list))
    for name, vals in loss_dense_list.items():
        if vals:
            epoch_stats[f"dense_boundary_{name}_loss"] = float(np.mean(vals))
    if loss_inst_list:
        epoch_stats["instance_attr_loss"] = float(np.mean(loss_inst_list))
    if loss_total_list:
        epoch_stats["total_loss"] = float(np.mean(loss_total_list))

    # Per-attribute accuracies
    for attr_idx, total in struct_total.items():
        epoch_stats[f"struct_acc_{attr_idx}"] = struct_correct.get(attr_idx, 0) / max(total, 1)
    for attr_idx, total in bound_total.items():
        epoch_stats[f"bound_acc_{attr_idx}"] = bound_correct.get(attr_idx, 0) / max(total, 1)
    for attr_idx, total in inst_total.items():
        epoch_stats[f"inst_acc_{attr_idx}"] = inst_correct.get(attr_idx, 0) / max(total, 1)

    # Dense boundary F1 averages
    for name, accum in dense_f1_accum.items():
        cnt = dense_f1_count.get(name, 0)
        epoch_stats[f"dense_{name}_f1"] = accum / max(cnt, 1)

    # ----- Aggregate metrics -----
    # MLStructAcc: mean of per-attribute structure accuracy
    _struct_acc_vals = [epoch_stats.get(f"struct_acc_{i}", float("nan")) for i in range(5)]
    epoch_stats["MLStructAcc"] = float(np.nanmean(_struct_acc_vals)) if any(not np.isnan(v) for v in _struct_acc_vals) else float("nan")

    # MLBoundaryAcc: mean of per-attribute boundary accuracy
    _bound_acc_vals = [epoch_stats.get(f"bound_acc_{i}", float("nan")) for i in range(4)]
    epoch_stats["MLBoundaryAcc"] = float(np.nanmean(_bound_acc_vals)) if any(not np.isnan(v) for v in _bound_acc_vals) else float("nan")

    # InstanceAcc: mean of per-attribute instance accuracy
    _inst_acc_vals = [epoch_stats.get(f"inst_acc_{i}", float("nan")) for i in range(6)]
    epoch_stats["InstanceAcc"] = float(np.nanmean(_inst_acc_vals)) if any(not np.isnan(v) for v in _inst_acc_vals) else float("nan")

    # DenseBoundaryF1: mean of all dense F1 scores
    _dense_f1_vals = [epoch_stats.get(f"dense_{name}_f1", float("nan")) for name in DenseBoundaryHead.DENSE_BOUNDARY_NAMES]
    epoch_stats["DenseBoundaryF1"] = float(np.nanmean(_dense_f1_vals)) if any(not np.isnan(v) for v in _dense_f1_vals) else float("nan")

    # AttrComboScore: composite across all submodule metrics
    _all_metrics = _struct_acc_vals + _bound_acc_vals + _inst_acc_vals + _dense_f1_vals
    epoch_stats["AttrComboScore"] = float(np.nanmean(_all_metrics)) if any(not np.isnan(v) for v in _all_metrics) else float("nan")

    if rank == 0 and writer is not None:
        for k, v in epoch_stats.items():
            _write_scalar_if_finite(writer, f"PhaseB_Epoch/{k}", v, epoch)

    # ── Delta audit epoch summary ──
    if _delta_audit_steps > 0 and rank == 0:
        _delta_msg = " | ".join(
            f"{_name}_mean_delta={_delta_audit_accum[_name]/max(_delta_audit_steps, 1):.8f}"
            for _name in sorted(_delta_audit_accum.keys())
        )
        print(
            f"[PHASE_B_ML_DELTA_AUDIT][epoch={epoch + 1}] "
            f"mean_step_delta_norm: {_delta_msg} (over {_delta_audit_steps} steps)"
        )
        if logger is not None:
            logger.info(
                f"[PHASE_B_ML_DELTA_AUDIT][epoch={epoch + 1}] "
                f"mean_step_delta_norm: {_delta_msg} (over {_delta_audit_steps} steps)"
            )

    return epoch_stats


# ====================================================================
# 4. Validation
# ====================================================================

@torch.no_grad()
def validate_one_epoch_multilevel_attr(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: Any,
    logger: Any,
    writer: Any,
    rank: int,
) -> Dict[str, float]:
    """Validate Phase B — compute structure/boundary/instance attribute metrics.

    Uses batch-level ``multilevel_attr_logits`` output.
    Returns dict of validation metrics.
    """
    model.eval()
    eval_model = unwrap_model(model)

    # --- Batch limit logic: debug_max_* takes priority over max_* ---
    _debug_max_val = getattr(args, "debug_max_val_batches", None)
    _max_val = getattr(args, "max_val_batches", None)
    if _debug_max_val is not None and int(_debug_max_val) > 0:
        val_limit = int(_debug_max_val)
        _limit_source = "debug_max_val_batches"
    elif _max_val is not None and int(_max_val) > 0:
        val_limit = int(_max_val)
        _limit_source = "max_val_batches"
    else:
        val_limit = None
        _limit_source = "none"

    effective_total = min(len(data_loader), val_limit) if val_limit else len(data_loader)

    if rank == 0:
        if logger is not None:
            logger.info(
                f"[PHASE_B_ML_LIMIT] val_limit={val_limit} | source={_limit_source} | "
                f"effective_total={effective_total} | total_dataloader={len(data_loader)}"
            )
        from tqdm import tqdm
        pbar = tqdm(data_loader, desc=f"Phase B Ep {epoch + 1} Val", total=effective_total)
    else:
        pbar = data_loader

    # Accumulators
    loss_struct_list: List[float] = []
    loss_bound_list: List[float] = []
    loss_inst_list: List[float] = []

    struct_correct: Dict[int, int] = {}
    struct_total: Dict[int, int] = {}
    bound_correct: Dict[int, int] = {}
    bound_total: Dict[int, int] = {}
    inst_correct: Dict[int, int] = {}
    inst_total: Dict[int, int] = {}
    dense_f1_accum: Dict[str, float] = {}
    dense_f1_count: Dict[str, int] = {}

    # Reset per-epoch audit flags (local variables, not module-level globals)
    dense_shape_audited = False
    instance_shape_audited = False
    instance_label_hist_audited = False

    for batch_idx, batched_input in enumerate(pbar):
        if val_limit is not None and batch_idx >= val_limit:
            break
        batched_input = _to_device(batched_input, device)
        images = batched_input["image"]

        structure_attr_labels = batched_input.get("structure_attr_labels", None)
        boundary_attr_labels = batched_input.get("boundary_attr_labels", None)
        dense_boundary_map_gt = batched_input.get("dense_boundary_map", None)
        per_instance_attr_labels = batched_input.get("per_instance_attr_labels", None)

        # Build model_input
        organ_ids = batched_input.get("organ_id", None)
        dynamic_text = batched_input.get("text_prompt", ["Cell nuclei"] * len(images))
        dynamic_attr_text = batched_input.get("attribute_text", ["Cell nuclei"] * len(images))

        model_input = []
        for i in range(len(images)):
            curr_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val
            entry = {
                "image": images[i],
                "original_size": (args.image_size, args.image_size),
                "organ_id": curr_id,
                "attribute_text": dynamic_attr_text[i],
                "text_prompt": dynamic_text[i],
            }
            lbl_inst = batched_input.get("label_inst", None)
            if lbl_inst is not None:
                entry["label_inst"] = lbl_inst[i]
            s_labels = batched_input.get("structure_attr_labels", None)
            if s_labels is not None:
                entry["structure_attr_labels"] = s_labels[i]
            b_labels = batched_input.get("boundary_attr_labels", None)
            if b_labels is not None:
                entry["boundary_attr_labels"] = b_labels[i]
            pi_labels = batched_input.get("per_instance_attr_labels", None)
            if pi_labels is not None:
                entry["per_instance_attr_labels"] = pi_labels[i]
            model_input.append(entry)

        with torch.inference_mode(), _autocast_context(args):
            outputs = eval_model(model_input, multimask_output=True)
            ml_out = outputs[0].get("multilevel_attr_logits", None) if outputs else None

        if ml_out is None:
            continue

        # ----- Structure Attr -----
        s_logits = ml_out.get("structure_attr_logits", None)
        if s_logits is not None and structure_attr_labels is not None:
            s_loss = compute_structure_attr_loss(s_logits, structure_attr_labels)
            if s_loss is not None:
                loss_struct_list.append(s_loss.item())
                s_preds = torch.argmax(s_logits, dim=-1)
                for attr_idx in range(s_logits.shape[1]):
                    valid = structure_attr_labels[:, attr_idx] >= 0
                    n_valid = valid.sum().item()
                    if n_valid > 0:
                        struct_total[attr_idx] = struct_total.get(attr_idx, 0) + n_valid
                        n_correct = (s_preds[valid, attr_idx] == structure_attr_labels[valid, attr_idx]).sum().item()
                        struct_correct[attr_idx] = struct_correct.get(attr_idx, 0) + n_correct

        # ----- Boundary Attr -----
        b_logits = ml_out.get("boundary_attr_logits", None)
        if b_logits is not None and boundary_attr_labels is not None:
            b_loss = compute_boundary_attr_loss(b_logits, boundary_attr_labels)
            if b_loss is not None:
                loss_bound_list.append(b_loss.item())
                b_preds = torch.argmax(b_logits, dim=-1)
                for attr_idx in range(b_logits.shape[1]):
                    valid = boundary_attr_labels[:, attr_idx] >= 0
                    n_valid = valid.sum().item()
                    if n_valid > 0:
                        bound_total[attr_idx] = bound_total.get(attr_idx, 0) + n_valid
                        n_correct = (b_preds[valid, attr_idx] == boundary_attr_labels[valid, attr_idx]).sum().item()
                        bound_correct[attr_idx] = bound_correct.get(attr_idx, 0) + n_correct

        # ----- Dense Boundary Maps -----
        dense_maps = ml_out.get("dense_boundary_maps", None)
        if dense_maps is not None and dense_boundary_map_gt is not None:
            # ---- Shape audit (rank0-only, first batch of val epoch) ----
            if rank == 0 and not dense_shape_audited:
                sample_pred = None
                for name in DenseBoundaryHead.DENSE_BOUNDARY_NAMES:
                    sample_pred = dense_maps.get(name, None)
                    if sample_pred is not None:
                        break
                if sample_pred is not None:
                    target_raw = dense_boundary_map_gt[:, 0:1, :, :]
                    target_resized = _ensure_target_like_pred(target_raw, sample_pred, mode="area")
                    print(
                        f"[PHASE_B_ML_DENSE_AUDIT][VAL] pred_shape={tuple(sample_pred.shape)} | "
                        f"target_shape_raw={tuple(dense_boundary_map_gt[:, 0:1, :, :].shape)} | "
                        f"target_shape_resized={tuple(target_resized.shape) if target_resized is not None else 'N/A'}"
                    )
                dense_shape_audited = True

            for idx, name in enumerate(DenseBoundaryHead.DENSE_BOUNDARY_NAMES):
                pred_map = dense_maps.get(name, None)
                if pred_map is not None:
                    target_map = dense_boundary_map_gt[:, idx:idx + 1, :, :]
                    f1 = _compute_dense_boundary_f1(pred_map, target_map)
                    if not np.isnan(f1):
                        dense_f1_accum[name] = dense_f1_accum.get(name, 0.0) + f1
                        dense_f1_count[name] = dense_f1_count.get(name, 0) + 1

        # ----- Instance Morphology -----
        inst_out = ml_out.get("instance_attr_logits", None)
        if inst_out is not None and per_instance_attr_labels is not None:
            # ---- Instance audit (rank0-only, first batch of val epoch) ----
            if rank == 0 and not instance_shape_audited:
                labels_type = type(per_instance_attr_labels).__name__
                if isinstance(per_instance_attr_labels, (list, tuple)):
                    label_shapes = [tuple(t.shape) for t in per_instance_attr_labels]
                elif isinstance(per_instance_attr_labels, torch.Tensor):
                    label_shapes = list(per_instance_attr_labels.shape)
                else:
                    label_shapes = "unknown"
                logits_obj = inst_out.get("logits", None)
                logits_shape_info = (
                    f"list_len={len(logits_obj)}" if isinstance(logits_obj, list)
                    else str(tuple(logits_obj.shape)) if logits_obj is not None
                    else "None"
                )
                kept_pos = inst_out.get("kept_indices", None)
                kept_type = type(kept_pos).__name__ if kept_pos is not None else "None"
                print(
                    f"[PHASE_B_ML_INSTANCE_AUDIT][VAL] labels_type={labels_type} | "
                    f"batch={len(per_instance_attr_labels) if isinstance(per_instance_attr_labels, (list, tuple)) else (per_instance_attr_labels.shape[0] if isinstance(per_instance_attr_labels, torch.Tensor) else '?')} | "
                    f"label_shapes={label_shapes} | "
                    f"logits_shape={logits_shape_info} | "
                    f"kept_positions_type={kept_type}"
                )
                instance_shape_audited = True

            # ---- Label histogram audit (rank0-only, first batch of val epoch) ----
            if rank == 0 and not instance_label_hist_audited:
                normed_labels_for_hist = _normalize_per_image_instance_labels(per_instance_attr_labels)
                if normed_labels_for_hist is not None:
                    _all_inst_labels = [nl for nl in normed_labels_for_hist if nl.numel() > 0]
                    if _all_inst_labels:
                        _cat_labels = torch.cat(_all_inst_labels, dim=0)
                        _hist = _compute_label_histogram(_cat_labels)
                        print(f"[PHASE_B_ML_INSTANCE_LABEL_HIST][VAL] per-attribution class counts [c0,c1,c2]:")
                        for _attr_name, _counts in _hist.items():
                            _total = sum(_counts)
                            _pct = [f"{c/max(_total,1)*100:.1f}%" for c in _counts]
                            print(f"  {_attr_name}: {_counts} ({', '.join(_pct)})")
                instance_label_hist_audited = True

            # Normalize labels to per-image list for subsequent processing
            normed_labels = _normalize_per_image_instance_labels(per_instance_attr_labels)
            logits_list = inst_out.get("logits", None)
            kept_indices = inst_out.get("kept_indices", None)
            if logits_list is not None and normed_labels is not None:
                for b in range(min(len(logits_list), len(normed_labels))):
                    logits_i = logits_list[b]
                    if logits_i.numel() == 0:
                        continue
                    labels_i = normed_labels[b]
                    if labels_i.numel() == 0:
                        continue
                    if kept_indices is not None and b < len(kept_indices):
                        kept_idx = kept_indices[b]
                        if kept_idx.numel() == 0:
                            continue
                        if kept_idx.max() < labels_i.shape[0]:
                            kept_labels = labels_i[kept_idx]
                        else:
                            continue
                    else:
                        kept_labels = labels_i
                    if logits_i.shape[0] != kept_labels.shape[0]:
                        continue
                    i_loss = _ce_loss(logits_i, kept_labels)
                    if i_loss is not None:
                        loss_inst_list.append(i_loss.item())
                        i_preds = torch.argmax(logits_i, dim=-1)
                        for attr_idx in range(logits_i.shape[1]):
                            valid = kept_labels[:, attr_idx] >= 0
                            if valid.sum() > 0:
                                inst_total[attr_idx] = inst_total.get(attr_idx, 0) + valid.sum().item()
                                n_correct = (i_preds[valid, attr_idx] == kept_labels[valid, attr_idx]).sum().item()
                                inst_correct[attr_idx] = inst_correct.get(attr_idx, 0) + n_correct

    # Aggregate
    val_stats: Dict[str, float] = {}
    if loss_struct_list:
        val_stats["val_structure_attr_loss"] = float(np.mean(loss_struct_list))
    if loss_bound_list:
        val_stats["val_boundary_attr_loss"] = float(np.mean(loss_bound_list))
    if loss_inst_list:
        val_stats["val_instance_attr_loss"] = float(np.mean(loss_inst_list))
    for attr_idx, total in struct_total.items():
        val_stats[f"val_struct_acc_{attr_idx}"] = struct_correct.get(attr_idx, 0) / max(total, 1)
    for attr_idx, total in bound_total.items():
        val_stats[f"val_bound_acc_{attr_idx}"] = bound_correct.get(attr_idx, 0) / max(total, 1)
    for attr_idx, total in inst_total.items():
        val_stats[f"val_inst_acc_{attr_idx}"] = inst_correct.get(attr_idx, 0) / max(total, 1)
    for name, accum in dense_f1_accum.items():
        val_stats[f"val_dense_{name}_f1"] = accum / max(dense_f1_count.get(name, 0), 1)

    # ----- Aggregate validation metrics -----
    # val_MLStructAcc
    _val_struct_acc_vals = [val_stats.get(f"val_struct_acc_{i}", float("nan")) for i in range(5)]
    val_stats["val_MLStructAcc"] = float(np.nanmean(_val_struct_acc_vals)) if any(not np.isnan(v) for v in _val_struct_acc_vals) else float("nan")

    # val_MLBoundaryAcc
    _val_bound_acc_vals = [val_stats.get(f"val_bound_acc_{i}", float("nan")) for i in range(4)]
    val_stats["val_MLBoundaryAcc"] = float(np.nanmean(_val_bound_acc_vals)) if any(not np.isnan(v) for v in _val_bound_acc_vals) else float("nan")

    # val_InstanceAcc
    _val_inst_acc_vals = [val_stats.get(f"val_inst_acc_{i}", float("nan")) for i in range(6)]
    val_stats["val_InstanceAcc"] = float(np.nanmean(_val_inst_acc_vals)) if any(not np.isnan(v) for v in _val_inst_acc_vals) else float("nan")

    # val_DenseBoundaryF1
    _val_dense_f1_vals = [val_stats.get(f"val_dense_{name}_f1", float("nan")) for name in DenseBoundaryHead.DENSE_BOUNDARY_NAMES]
    val_stats["val_DenseBoundaryF1"] = float(np.nanmean(_val_dense_f1_vals)) if any(not np.isnan(v) for v in _val_dense_f1_vals) else float("nan")

    # val_AttrComboScore: composite across all submodule metrics
    _val_all_metrics = _val_struct_acc_vals + _val_bound_acc_vals + _val_inst_acc_vals + _val_dense_f1_vals
    val_stats["val_AttrComboScore"] = float(np.nanmean(_val_all_metrics)) if any(not np.isnan(v) for v in _val_all_metrics) else float("nan")

    # Composite score: mean of structure acc + mean of instance acc + mean of dense F1
    # (keep existing val_composite_score for backward compatibility)
    struct_accs = [val_stats.get(f"val_struct_acc_{i}", float("nan")) for i in range(5)]
    bound_accs = [val_stats.get(f"val_bound_acc_{i}", float("nan")) for i in range(4)]
    inst_accs = [val_stats.get(f"val_inst_acc_{i}", float("nan")) for i in range(6)]
    dense_f1s = [
        val_stats.get(f"val_dense_{name}_f1", float("nan"))
        for name in DenseBoundaryHead.DENSE_BOUNDARY_NAMES
    ]
    val_stats["val_composite_score"] = float(
        np.nanmean(struct_accs + bound_accs + inst_accs + dense_f1s)
    )

    if rank == 0 and writer is not None:
        for k, v in val_stats.items():
            if not np.isnan(v):
                writer.add_scalar(f"PhaseB_Val/{k}", v, epoch)

    return val_stats


# ====================================================================
# 5. Best-Metric Tracking
# ====================================================================

def update_phase_b_best_metrics(
    val_stats: Dict[str, float],
    best_metrics: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    """Update best Phase B metrics and return whether a new best was achieved.

    ``best_metrics`` is mutated in place.
    Returns ``(best_metrics, updated)``.
    """
    updated = False
    composite = val_stats.get("val_composite_score", float("-inf"))
    if composite > best_metrics.get("val_composite_score", float("-inf")):
        best_metrics["val_composite_score"] = composite
        for k, v in val_stats.items():
            if k.startswith("val_"):
                best_metrics[k] = v
        updated = True
    return best_metrics, updated


def init_phase_b_best_metrics() -> Dict[str, float]:
    """Initialize the best-metrics tracker for Phase B."""
    return {
        "val_composite_score": float("-inf"),
    }


# ====================================================================
# 6. Checkpoint
# ====================================================================

def save_phase_b_checkpoint_if_needed(
    model: nn.Module,
    epoch: int,
    args: Any,
    val_stats: Dict[str, float],
    best_metrics: Dict[str, float],
    updated: bool,
    logger: Any,
    rank: int,
) -> None:
    """Save Phase B checkpoint — latest after every epoch, best on improvement.

    Files:
      - ``latest_multilevel_attr_model.pth``
      - ``best_multilevel_attr_model.pth``
    """
    if rank != 0:
        return

    raw_model = unwrap_model(model)
    model_save_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(model_save_dir, exist_ok=True)

    checkpoint_dict = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "phase": args.phase,
        "architecture_version": getattr(args, "architecture_version", "unknown"),
        "asr_variant": getattr(args, "asr_variant", "legacy"),
        "asr_regression": getattr(args, "asr_regression", False),
        "best_metrics": best_metrics,
        "val_stats": val_stats,
        "args": vars(args),
    }

    # Latest
    latest_path = os.path.join(model_save_dir, "latest_multilevel_attr_model.pth")
    torch.save(checkpoint_dict, latest_path)

    if logger is not None:
        logger.info(f"[PhaseB_CKPT] Latest model saved: {latest_path} (epoch {epoch})")

    # Best
    if updated:
        best_path = os.path.join(model_save_dir, "best_multilevel_attr_model.pth")
        torch.save(checkpoint_dict, best_path)
        if logger is not None:
            logger.info(
                f"[PhaseB_CKPT] New best composite score: {best_metrics['val_composite_score']:.6f}. "
                f"Best model saved: {best_path}"
            )
