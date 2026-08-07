"""
Phase C: Semantic Alignment — Attribute-Text Alignment training/validation module.

Extracted from train.py to reduce complexity while preserving 1-batch sanity behavior.
Train / val / best metric update / checkpoint saving / sanity checks are isolated here.

NOTE: This module must NOT import from train.py to avoid circular dependency.
All shared utilities are imported from training.common instead.
"""

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from training.common import (
    _autocast_context,
    _is_debug_run,
    _save_checkpoint_file,
    _to_float_or_nan,
    _write_scalar_if_finite,
    unwrap_model,
)

# ====================================================================
# Utility: zero-like loss tensor
# ====================================================================
def _zero_like_loss(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


# ====================================================================
# 1. Unified batch input builder (shared by train & val)
# ====================================================================
def build_phase_c_batch_inputs(
    batch: Dict[str, Any],
    args,
    device: torch.device,
    is_train: bool,
) -> List[Dict[str, Any]]:
    """Build model input list for Phase C semantic_alignment.

    Unified for both train and validation to prevent field omission.
    Phase C unconditionally requires GT structure / boundary / per-instance
    attribute labels (not gated on sb_guidance_mode).

    Args:
        batch: Raw batch dict from DataLoader.
        args: Parsed command-line arguments.
        device: Target device.
        is_train: Whether this is a training call (True) or validation (False).

    Returns:
        List of per-sample dicts ready for TextSam.forward().

    Raises:
        RuntimeError: If structure_attr_labels or boundary_attr_labels are missing.
    """
    images = batch["image"]
    # ------------------------------------------------------------------
    # GT structure / boundary labels (REQUIRED for Phase C)
    # ------------------------------------------------------------------
    structure_labels = batch.get("structure_attr_labels", None)
    boundary_labels = batch.get("boundary_attr_labels", None)

    # Phase C must NOT proceed without these labels — raise immediately.
    if structure_labels is None:
        raise RuntimeError(
            "[PHASE_C_INPUT_AUDIT] structure_attr_labels missing from batch. "
            "Phase C (semantic_alignment) requires GT structure attribute labels. "
            "Ensure --use_structure_boundary_attrs and --structure_boundary_attr_path are set."
        )
    if boundary_labels is None:
        raise RuntimeError(
            "[PHASE_C_INPUT_AUDIT] boundary_attr_labels missing from batch. "
            "Phase C (semantic_alignment) requires GT boundary attribute labels."
        )

    if structure_labels.device != device:
        structure_labels = structure_labels.to(device)
    if boundary_labels.device != device:
        boundary_labels = boundary_labels.to(device)

    # ------------------------------------------------------------------
    # Per-instance labels & IDs (optional — if no valid instance, skip branch)
    # ------------------------------------------------------------------
    per_instance_attr_labels: Optional[List] = batch.get("per_instance_attr_labels", None)
    per_instance_ids: Optional[List] = batch.get("per_instance_ids", None)

    # Instance mask (for InstanceMorphologyHead)
    label_inst: Optional[torch.Tensor] = batch.get("label_inst", None)
    if label_inst is not None and label_inst.device != device:
        label_inst = label_inst.to(device)

    # ------------------------------------------------------------------
    # Common fields
    # ------------------------------------------------------------------
    organ_ids = batch.get("organ_id", None)
    attr_labels = batch.get("attr_labels", None)
    dynamic_text = batch.get("text_prompt", ["Cell nuclei"] * len(images))
    dynamic_attr_text = batch.get("attribute_text", dynamic_text)
    sample_id = batch.get("sample_id", batch.get("name", None))

    model_input: List[Dict[str, Any]] = []

    for i in range(len(images)):
        # Organ ID
        curr_id = 20  # default (PanNuke background/unknown)
        if organ_ids is not None:
            _val = organ_ids[i]
            curr_id = _val.item() if isinstance(_val, torch.Tensor) else int(_val)

        entry: Dict[str, Any] = {
            "image": images[i],
            "original_size": (args.image_size, args.image_size),
            "organ_id": curr_id,
            "attribute_text": dynamic_attr_text[i],
            "text_prompt": dynamic_text[i],
            "attr_labels": (
                attr_labels[i].to(device) if attr_labels is not None and torch.is_tensor(attr_labels[i])
                else (attr_labels[i] if attr_labels is not None else None)
            ),
            # GT labels — unconditionally passed for Phase C internal alignment computation
            "structure_attr_labels": structure_labels[i],
            "boundary_attr_labels": boundary_labels[i],
        }

        # Instance mask (for instance morphology pooling in alignment)
        if label_inst is not None:
            entry["label_inst"] = label_inst[i]

        # Per-instance attribute labels (for per-instance text alignment)
        _has_per_inst = (
            per_instance_attr_labels is not None
            and i < len(per_instance_attr_labels)
            and per_instance_attr_labels[i] is not None
        )
        if _has_per_inst:
            entry["per_instance_attr_labels"] = per_instance_attr_labels[i]

        # Per-instance IDs (for id→position mapping)
        if per_instance_ids is not None and i < len(per_instance_ids):
            entry["per_instance_ids"] = per_instance_ids[i]

        # Sample identifier (for traceability)
        if sample_id is not None and i < len(sample_id):
            entry["sample_id"] = sample_id[i]

        model_input.append(entry)

    return model_input


# ====================================================================
# 2. Training one epoch (Phase C only)
# ====================================================================
def train_one_epoch_semantic_alignment(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    logger=None,
    writer=None,
    rank: int = 0,
) -> Dict[str, float]:
    """Train Phase C semantic_alignment for one epoch.

    Only attr-text alignment loss is computed; no segmentation losses,
    no Dice / IoU / AJI / PQ, no Phase B diagnostic logs.

    Returns:
        dict with averaged train metrics:
            "total_loss", "attr_text_align_loss", "s_sim", "b_sim", "i_sim",
            "text_norm_mean", "visual_norm_mean", "valid_instance_count"
    """
    model.train()
    stage = args.phase  # expected: "semantic_alignment"
    _debug_audit = bool(getattr(args, "debug_phase_c_audit", False))
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
                f"[PHASE_C_LIMIT] train_limit={train_limit} | source={_limit_source} | "
                f"effective_total={effective_total} | total_dataloader={len(data_loader)}"
            )
        pbar = tqdm(data_loader, desc=f"Ep {epoch + 1} Train [{stage}]", total=effective_total)
    else:
        pbar = data_loader

    # Accumulators
    accum_loss = 0.0
    accum_align = 0.0
    accum_s_sim = 0.0
    accum_b_sim = 0.0
    accum_i_sim = 0.0
    accum_tn = 0.0
    accum_vn = 0.0
    accum_valid_inst = 0.0
    num_batches = 0

    # Attempt gradient scaling (may be None if not using AMP)
    scaler = getattr(args, "scaler", None)
    use_scaler = scaler is not None and getattr(args, "use_amp", False)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        if train_limit is not None and batch_idx >= train_limit:
            break

        batch = _to_device(batch, device)

        # ----------------------------------------------------------
        # Build model input (unified builder — same for train & val)
        # ----------------------------------------------------------
        model_input = build_phase_c_batch_inputs(batch, args, device, is_train=True)

        # ----------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------
        with _autocast_context(args):
            outputs = model(model_input, multimask_output=True)

            # Phase C only: read attr_text_alignment_loss from outputs
            _align_loss_val = outputs[0].get("attr_text_align_loss", None)
            if not torch.is_tensor(_align_loss_val):
                _align_loss_val = _zero_like_loss(device)

            # --- Gradient audit (debug only) ---
            if rank == 0 and logger is not None and _debug_audit:
                logger.info(
                    f"[PHASE_C_GRAD_AUDIT] _align_loss_val.requires_grad="
                    f"{_align_loss_val.requires_grad} "
                    f"_align_loss_val.grad_fn={_align_loss_val.grad_fn} "
                    f"_align_loss_val.dtype={_align_loss_val.dtype} "
                    f"_align_loss_val.device={_align_loss_val.device}"
                )

            if not _align_loss_val.requires_grad:
                _err_msg = (
                    "[PHASE_C_GRAD_AUDIT][FATAL] _align_loss_val.requires_grad=False. "
                    "Alignment loss is detached from autograd graph. "
                    "Root cause: attr_text_align_loss is detached in TextSam.forward() "
                    "before being stored in diagnostics. Check that:\n"
                    "  1. TextSam.forward() does NOT .detach() attr_text_align_loss\n"
                    "  2. diagnostics['attr_text_align_loss'] is NOT detached\n"
                    "  3. _compute_attr_text_alignment_loss returns raw gradient tensor\n"
                    "  4. No torch.no_grad() wraps Phase C projection/loss computation"
                )
                if logger is not None:
                    logger.error(_err_msg)
                raise RuntimeError(_err_msg)

            final_loss = _align_loss_val

            # --- Gradient audit for final_loss ---
            if rank == 0 and logger is not None and _debug_audit:
                logger.info(
                    f"[PHASE_C_GRAD_AUDIT] final_loss.requires_grad={final_loss.requires_grad} "
                    f"final_loss.grad_fn={final_loss.grad_fn}"
                )

            if not final_loss.requires_grad:
                _err_msg = (
                    "[PHASE_C_GRAD_AUDIT][FATAL] final_loss.requires_grad=False. "
                    "Alignment loss is detached. Check:\n"
                    "  1. outputs[0]['attr_text_align_loss'] is NOT .detach()'ed\n"
                    "  2. _compute_attr_text_alignment_loss returns raw loss tensor\n"
                    "  3. No torch.no_grad() wraps Phase C projection/loss computation"
                )
                if logger is not None:
                    logger.error(_err_msg)
                raise RuntimeError(_err_msg)

        # ----------------------------------------------------------
        # Backward pass
        # ----------------------------------------------------------
        if use_scaler:
            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            # ---- grad_norm audit (debug) ----
            if _debug_audit and rank == 0:
                _audit_projection_grad_norms(model, logger, "scaler branch")
            # --------------------------------
            scaler.step(optimizer)
            scaler.update()
        else:
            final_loss.backward()
            # ---- grad_norm audit (debug) ----
            if _debug_audit and rank == 0:
                _audit_projection_grad_norms(model, logger, "non-scaler branch")
            # --------------------------------
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------
        # Accumulate metrics
        # ----------------------------------------------------------
        _align_float = float(_align_loss_val.detach().item()) if torch.is_tensor(_align_loss_val) else 0.0
        accum_align += _align_float
        accum_loss += _align_float

        _s_sim = _to_float_or_nan(outputs[0].get("s_sim", None))
        _b_sim = _to_float_or_nan(outputs[0].get("b_sim", None))
        _i_sim = _to_float_or_nan(outputs[0].get("i_sim", None))
        _tn = _to_float_or_nan(outputs[0].get("text_norm_mean", None))
        _vn = _to_float_or_nan(outputs[0].get("visual_norm_mean", None))
        _valid_inst = int(outputs[0].get("valid_instance_count", 0) or 0)

        accum_s_sim += _s_sim if not np.isnan(_s_sim) else 0.0
        accum_b_sim += _b_sim if not np.isnan(_b_sim) else 0.0
        accum_i_sim += _i_sim if not np.isnan(_i_sim) else 0.0
        accum_tn += _tn if not np.isnan(_tn) else 0.0
        accum_vn += _vn if not np.isnan(_vn) else 0.0
        accum_valid_inst += _valid_inst
        num_batches += 1

        # ----------------------------------------------------------
        # Writer logging (every 10 batches)
        # ----------------------------------------------------------
        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(data_loader) + batch_idx
            _write_scalar_if_finite(writer, "Train/AttrTextAlignLoss", accum_align / max(num_batches, 1), global_step)
            _write_scalar_if_finite(writer, "Train/StructSim", _s_sim, global_step)
            _write_scalar_if_finite(writer, "Train/BoundSim", _b_sim, global_step)
            _write_scalar_if_finite(writer, "Train/InstSim", _i_sim, global_step)

        # ----------------------------------------------------------
        # pbar update (Phase C compact: L, Align, Ssim, Bsim, Isim, TN, VN)
        # ----------------------------------------------------------
        if rank == 0:
            pbar.set_postfix(
                L=f"{accum_loss / max(num_batches, 1):.6f}",
                Align=f"{accum_align / max(num_batches, 1):.6f}",
                Ssim=f"{_s_sim:.4f}" if not np.isnan(_s_sim) else "nan",
                Bsim=f"{_b_sim:.4f}" if not np.isnan(_b_sim) else "nan",
                Isim=f"{_i_sim:.4f}" if _valid_inst > 0 and not np.isnan(_i_sim) else "skip",
                TN=f"{_tn:.4f}" if not np.isnan(_tn) else "?",
                VN=f"{_vn:.4f}" if not np.isnan(_vn) else "?",
            )

    # ----------------------------------------------------------
    # Average metrics
    # ----------------------------------------------------------
    n = max(num_batches, 1)
    train_metrics = {
        "total_loss": accum_loss / n,
        "attr_text_align_loss": accum_align / n,
        "s_sim": accum_s_sim / n,
        "b_sim": accum_b_sim / n,
        "i_sim": accum_i_sim / n,
        "text_norm_mean": accum_tn / n,
        "visual_norm_mean": accum_vn / n,
        "valid_instance_count": int(accum_valid_inst),
    }

    return train_metrics


# ====================================================================
# 3. Validation one epoch (Phase C only)
# ====================================================================
def validate_one_epoch_semantic_alignment(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    args,
    logger=None,
    writer=None,
    rank: int = 0,
) -> Dict[str, float]:
    """Validate Phase C semantic_alignment for one epoch.

    Runs under torch.no_grad(). Reads attr_text_alignment_loss and similarity
    scores from model outputs. Does NOT compute Dice / IoU / AJI / PQ.

    Returns:
        dict with averaged validation metrics:
            "val_align_loss", "val_s_sim", "val_b_sim", "val_i_sim",
            "val_text_norm_mean", "val_visual_norm_mean", "valid_instance_count"
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    _debug_audit = bool(getattr(args, "debug_phase_c_audit", False))

    # --- Batch limit logic: debug_max_* takes priority over max_* ---
    _debug_max_val = getattr(args, "debug_max_val_batches", None)
    _max_val = getattr(args, "max_val_batches", None)
    _raw_val = _debug_max_val if _debug_max_val is not None else _max_val

    if _raw_val is not None and int(_raw_val) > 0:
        val_limit = int(_raw_val)
        _limit_source = "debug_max_val_batches" if _debug_max_val is not None else "max_val_batches"
    else:
        val_limit = None
        _limit_source = "none"

    total_val_batches = len(data_loader)
    if val_limit is not None:
        limit_batches = min(val_limit, total_val_batches)
    else:
        val_fraction = float(getattr(args, "val_fraction", 0.4))
        if val_fraction <= 0.0 or total_val_batches <= 0:
            return {
                "val_align_loss": 0.0,
                "val_s_sim": 0.0,
                "val_b_sim": 0.0,
                "val_i_sim": 0.0,
                "val_text_norm_mean": 0.0,
                "val_visual_norm_mean": 0.0,
                "valid_instance_count": 0,
            }
        if val_fraction >= 1.0:
            limit_batches = total_val_batches
        else:
            limit_batches = max(1, min(int(np.ceil(total_val_batches * val_fraction)), total_val_batches))

    val_percent = 100.0 * limit_batches / max(total_val_batches, 1)
    if rank == 0:
        if logger is not None:
            logger.info(
                f"[PHASE_C_LIMIT] val_limit={val_limit} | source={_limit_source} | "
                f"effective_total={limit_batches} | total_dataloader={total_val_batches}"
            )
        pbar = tqdm(data_loader, desc=f"Ep {epoch + 1} Val [semantic_alignment] ({val_percent:.1f}%)", total=limit_batches)
    else:
        pbar = data_loader

    eval_model = unwrap_model(model)

    # Accumulators
    val_align_loss_sum = 0.0
    val_align_loss_count = 0
    val_s_sim_sum = 0.0
    val_b_sim_sum = 0.0
    val_i_sim_sum = 0.0
    val_sim_count = 0
    val_text_norm_sum = 0.0
    val_visual_norm_sum = 0.0
    val_norm_count = 0
    val_valid_inst_count = 0

    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= limit_batches:
            break

        batch = _to_device(batch, device)

        # Build model input (unified builder)
        model_input = build_phase_c_batch_inputs(batch, args, device, is_train=False)

        # ----------------------------------------------------------
        # Forward pass (no gradient)
        # ----------------------------------------------------------
        with torch.inference_mode(), _autocast_context(args):
            outputs = eval_model(model_input, multimask_output=True)

        # ----------------------------------------------------------
        # Extract Phase C metrics from outputs
        # ----------------------------------------------------------
        _diag = outputs[0].get("diagnostics", {}) if isinstance(outputs[0], dict) else {}

        # Primary source: diagnostics dict; fallback: top-level keys
        _v_align_loss = _to_float_or_nan(_diag.get("attr_text_align_loss"))
        if np.isnan(_v_align_loss):
            _v_align_loss = _to_float_or_nan(outputs[0].get("attr_text_align_loss"))

        _v_s_sim = _to_float_or_nan(_diag.get("s_sim"))
        if np.isnan(_v_s_sim):
            _v_s_sim = _to_float_or_nan(outputs[0].get("s_sim"))

        _v_b_sim = _to_float_or_nan(_diag.get("b_sim"))
        if np.isnan(_v_b_sim):
            _v_b_sim = _to_float_or_nan(outputs[0].get("b_sim"))

        _v_i_sim = _to_float_or_nan(_diag.get("i_sim"))
        if np.isnan(_v_i_sim):
            _v_i_sim = _to_float_or_nan(outputs[0].get("i_sim"))

        _v_tn = _to_float_or_nan(_diag.get("text_norm_mean"))
        if np.isnan(_v_tn):
            _v_tn = _to_float_or_nan(outputs[0].get("text_norm_mean"))

        _v_vn = _to_float_or_nan(_diag.get("visual_norm_mean"))
        if np.isnan(_v_vn):
            _v_vn = _to_float_or_nan(outputs[0].get("visual_norm_mean"))

        _v_valid_inst = int(outputs[0].get("valid_instance_count", 0) or 0)

        # --- Phase C validation grad audit (non-fatal for val) ---
        if rank == 0 and logger is not None and _debug_audit:
            _is_tensor = torch.is_tensor(outputs[0].get("attr_text_align_loss"))
            if _is_tensor:
                logger.info(
                    "[PHASE_C_GRAD_AUDIT][VAL_NO_GRAD] "
                    f"attr_text_align_loss (tensor, val mode — no grad expected)"
                )
            else:
                logger.info(
                    "[PHASE_C_GRAD_AUDIT][VAL_NO_GRAD] "
                    f"attr_text_align_loss={_v_align_loss} (float scalar, val mode)"
                )

        # --- Accumulate ---
        if not np.isnan(_v_align_loss):
            val_align_loss_sum += _v_align_loss
            val_align_loss_count += 1

        if not np.isnan(_v_s_sim):
            val_s_sim_sum += _v_s_sim
            val_b_sim_sum += _v_b_sim
            val_i_sim_sum += _v_i_sim
            val_sim_count += 1

        if not np.isnan(_v_tn):
            val_text_norm_sum += _v_tn
            val_visual_norm_sum += _v_vn
            val_norm_count += 1

        val_valid_inst_count += _v_valid_inst

        # --- pbar ---
        if rank == 0:
            pbar.set_postfix(
                ValAlign=f"{_v_align_loss:.6f}" if not np.isnan(_v_align_loss) else "nan",
                ValSsim=f"{_v_s_sim:.4f}" if not np.isnan(_v_s_sim) else "nan",
                ValBsim=f"{_v_b_sim:.4f}" if not np.isnan(_v_b_sim) else "nan",
                ValIsim=f"{_v_i_sim:.4f}" if not np.isnan(_v_i_sim) else "nan",
                TN=f"{_v_tn:.4f}" if not np.isnan(_v_tn) else "?",
                VN=f"{_v_vn:.4f}" if not np.isnan(_v_vn) else "?",
            )

    # --- Average results ---
    phase_c_results = {
        "val_align_loss": val_align_loss_sum / max(val_align_loss_count, 1),
        "val_s_sim": val_s_sim_sum / max(val_sim_count, 1),
        "val_b_sim": val_b_sim_sum / max(val_sim_count, 1),
        "val_i_sim": val_i_sim_sum / max(val_sim_count, 1),
        "val_text_norm_mean": val_text_norm_sum / max(val_norm_count, 1),
        "val_visual_norm_mean": val_visual_norm_sum / max(val_norm_count, 1),
        "valid_instance_count": val_valid_inst_count,
    }

    # --- Raise if alignment loss was never computed ---
    if val_align_loss_count == 0:
        raise RuntimeError(
            "[PHASE_C_VAL][FATAL] No valid attr_text_alignment_loss values computed during validation. "
            "The alignment loss is 0 or NaN. Check that the model produces attr_text_alignment_loss "
            "in its forward output (either at top-level or inside diagnostics)."
        )

    # --- Logging ---
    if rank == 0 and logger is not None:
        logger.info(
            f"[PHASE_C_VAL] AlignLoss={phase_c_results['val_align_loss']:.6f} | "
            f"SSim={phase_c_results['val_s_sim']:.4f} | "
            f"BSim={phase_c_results['val_b_sim']:.4f} | "
            f"ISim={phase_c_results['val_i_sim']:.4f}"
        )

    if rank == 0 and writer is not None:
        _write_scalar_if_finite(writer, "Val/AttrTextAlignLoss", phase_c_results["val_align_loss"], epoch)
        _write_scalar_if_finite(writer, "Val/StructSim", phase_c_results["val_s_sim"], epoch)
        _write_scalar_if_finite(writer, "Val/BoundSim", phase_c_results["val_b_sim"], epoch)
        _write_scalar_if_finite(writer, "Val/InstSim", phase_c_results["val_i_sim"], epoch)

    return phase_c_results


# ====================================================================
# 4. Best metric update (decoupled from checkpoint saving)
# ====================================================================
def update_phase_c_best_metrics(
    current_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    """Update best Phase C metrics based on val_align_loss (lower is better).

    Args:
        current_metrics: Validation results dict from validate_one_epoch_semantic_alignment().
        best_metrics: Current best metrics dict (mutated in-place).

    Returns:
        Tuple of (best_metrics, updated) where updated is True if best_metrics changed.
    """
    _cur_align = float(current_metrics.get("val_align_loss", float("inf")))
    _cur_s_sim = float(current_metrics.get("val_s_sim", float("-inf")))
    _cur_b_sim = float(current_metrics.get("val_b_sim", float("-inf")))
    _cur_i_sim = float(current_metrics.get("val_i_sim", float("-inf")))

    _prev_best_align = float(best_metrics.get("val_align_loss", float("inf")))
    updated = _cur_align < _prev_best_align

    if updated:
        best_metrics["val_align_loss"] = _cur_align
        best_metrics["val_s_sim"] = _cur_s_sim
        best_metrics["val_b_sim"] = _cur_b_sim
        best_metrics["val_i_sim"] = _cur_i_sim

    return best_metrics, updated


# ====================================================================
# 5. Checkpoint saving (Phase C only, decoupled from best metric update)
# ====================================================================
def save_phase_c_checkpoint_if_needed(
    model: torch.nn.Module,
    epoch: int,
    args,
    val_res: Dict[str, float],
    best_metrics: Dict[str, float],
    updated: bool,
    logger=None,
    rank: int = 0,
):
    """Save Phase C alignment checkpoint.

    In debug mode, checkpoint saving is skipped by default (unless
    --debug_allow_checkpoint_save is set). Best metric update is NOT
    skipped in debug mode.

    Saves:
        - latest_align_model.pth (always, unless debug skip)
        - best_align_model.pth (when val_align_loss improved)
    """
    if rank != 0:
        return

    _debug_mode = _is_debug_run(args) and not getattr(args, "debug_allow_checkpoint_save", False)
    _phase_c_debug_skip = _debug_mode

    # --- Build checkpoint dict ---
    _raw_model = unwrap_model(model)
    _phase_b_ckpt_path = str(getattr(args, "resume", ""))

    _s_vis_sd = (
        _raw_model.attr_align_vis_proj_structure.state_dict()
        if hasattr(_raw_model, "attr_align_vis_proj_structure") and _raw_model.attr_align_vis_proj_structure is not None
        else {}
    )
    _s_bound_sd = (
        _raw_model.attr_align_vis_proj_boundary.state_dict()
        if hasattr(_raw_model, "attr_align_vis_proj_boundary") and _raw_model.attr_align_vis_proj_boundary is not None
        else {}
    )
    _s_inst_sd = (
        _raw_model.attr_align_vis_proj_instance.state_dict()
        if hasattr(_raw_model, "attr_align_vis_proj_instance") and _raw_model.attr_align_vis_proj_instance is not None
        else {}
    )
    _s_text_sd = (
        _raw_model.attr_align_text_proj.state_dict()
        if hasattr(_raw_model, "attr_align_text_proj") and _raw_model.attr_align_text_proj is not None
        else {}
    )

    # ── Full model state dict: includes multilevel_attr_heads.* so Phase D can resume ──
    _full_model_sd = _raw_model.state_dict()

    checkpoint_dict = {
        "epoch": epoch,
        "epoch_num": epoch + 1,
        "model_state_dict": _full_model_sd,
        "attr_align_vis_proj_structure": _s_vis_sd,
        "attr_align_vis_proj_boundary": _s_bound_sd,
        "attr_align_vis_proj_instance": _s_inst_sd,
        "attr_align_text_proj": _s_text_sd,
        "val_align_loss": float(val_res.get("val_align_loss", float("inf"))),
        "val_structure_sim": float(val_res.get("val_s_sim", float("-inf"))),
        "val_boundary_sim": float(val_res.get("val_b_sim", float("-inf"))),
        "val_instance_sim": float(val_res.get("val_i_sim", float("-inf"))),
        "phase_b_checkpoint_path": _phase_b_ckpt_path,
        "attr_text_source": "structure_head_feature+dense_boundary_feature+instance_morphology_pre_mlp",
        "projection_dim": int(getattr(args, "attr_text_alignment_visual_dim", 256)),
        "phase": args.phase,
        "architecture_version": getattr(args, "architecture_version", ""),
        "args": vars(args),
    }

    model_dir = os.path.join(args.work_dir, "models", args.run_name)

    # --- Log best metric status ---
    if logger is not None:
        logger.info(
            f"[PHASE_C_BEST] updated={updated} | "
            f"best_val_align_loss={best_metrics['val_align_loss']:.6f} | "
            f"best_s_sim={best_metrics['val_s_sim']:.4f} | "
            f"best_b_sim={best_metrics['val_b_sim']:.4f} | "
            f"best_i_sim={best_metrics['val_i_sim']:.4f}"
        )

    # --- Checkpoint saving (may be skipped in debug mode) ---
    if _phase_c_debug_skip:
        if logger is not None:
            logger.info(
                "[DEBUG] Skip checkpoint saving in debug run (semantic_alignment mode). "
                "Use --debug_allow_checkpoint_save to override."
            )
            logger.info(f"[PHASE_C_CKPT] debug_skip_save=True | best_metrics_updated={updated}")
    else:
        if logger is not None:
            logger.info(
                f"[PHASE_C_CKPT] debug_skip_save=False | saving checkpoint ... | "
                f"best_metrics_updated={updated}"
            )

        # 1) Always save latest_align_model.pth
        latest_align_path = os.path.join(model_dir, "latest_align_model.pth")
        _save_checkpoint_file(
            checkpoint_dict,
            latest_align_path,
            logger=logger,
            message="[PHASE_C_CKPT] saved latest_align_model.pth",
        )

        # 2) Save best_align_model.pth when val_align_loss improves
        if updated:
            best_align_path = os.path.join(model_dir, "best_align_model.pth")
            _save_checkpoint_file(
                checkpoint_dict,
                best_align_path,
                logger=logger,
                message=(
                    f"[PHASE_C_CKPT] saved best_align_model.pth | "
                    f"val_align_loss={val_res.get('val_align_loss', float('inf')):.6f}"
                ),
            )


# ====================================================================
# 6. Sanity checks (debug run only)
# ====================================================================
def run_phase_c_sanity_checks(
    current_val_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    args,
    logger=None,
):
    """Run Phase C sanity checks on current validation metrics and best metrics.

    These checks verify that alignment values are finite, positive (loss > 0),
    and norms are reasonable. Designed for debug / 1-batch sanity runs.

    Args:
        current_val_metrics: The latest validation results dict from
            validate_one_epoch_semantic_alignment(). Must contain
            "val_text_norm_mean" and "val_visual_norm_mean" (these are
            per-epoch diagnostics, not best-tracked).
        best_metrics: Updated best metrics (never inf/-inf after update).
        args: Parsed command-line arguments.
        logger: Optional logger.
    """
    if logger is None:
        return

    if not _is_debug_run(args):
        if logger is not None:
            logger.info("[PHASE_C_SANITY] skipped: non-debug run")
        return

    # Use best metrics for val_align_loss / s_sim / b_sim / i_sim (already updated)
    _val_text_norm = float(current_val_metrics.get("val_text_norm_mean", 0.0))
    _val_visual_norm = float(current_val_metrics.get("val_visual_norm_mean", 0.0))

    _final_align = float(best_metrics.get("val_align_loss", float("inf")))
    _final_ssim = float(best_metrics.get("val_s_sim", float("-inf")))
    _final_bsim = float(best_metrics.get("val_b_sim", float("-inf")))
    _final_isim = float(best_metrics.get("val_i_sim", float("-inf")))

    _sanity_checks = [
        ("val_align_loss finite", not (np.isinf(_final_align) or np.isnan(_final_align))),
        ("val_align_loss > 0 (real alignment computed)", _final_align > 0),
        ("val_s_sim finite", not (np.isinf(_final_ssim) or np.isnan(_final_ssim))),
        ("val_b_sim finite", not (np.isinf(_final_bsim) or np.isnan(_final_bsim))),
        ("val_i_sim finite", not (np.isinf(_final_isim) or np.isnan(_final_isim))),
        ("val_text_norm_mean > 0", _val_text_norm > 0),
        ("val_visual_norm_mean > 0", _val_visual_norm > 0),
    ]

    _all_passed = all(v for _, v in _sanity_checks)
    _fail_count = sum(1 for _, v in _sanity_checks if not v)

    logger.info("=" * 50)
    logger.info("[PHASE_C_SANITY][VAL] Phase C validation metric sanity checks (finite only):")
    for _check_name, _check_result in _sanity_checks:
        _status = "PASS" if _check_result else "FAIL"
        logger.info(f"  [{_status}] {_check_name}")
    logger.info(
        "[PHASE_C_SANITY][VAL] Train-side grad audit is handled separately "
        "by PHASE_C_GRAD_AUDIT in sam.py (requires_grad check only when "
        "torch.is_grad_enabled())."
    )
    if _all_passed:
        logger.info("[PHASE_C_SANITY][VAL] All validation checks PASSED. Phase C sanity test passed.")
    else:
        logger.warning(
            f"[PHASE_C_SANITY][VAL] {_fail_count}/{len(_sanity_checks)} checks FAILED. "
            "Review logs above for details."
        )
    logger.info("=" * 50)


# ====================================================================
# Internal helpers
# ====================================================================
def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to device. Handles dict and list-of-dict inputs."""
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return batch


def _audit_projection_grad_norms(model: torch.nn.Module, logger, branch_name: str):
    """Log grad norms of Phase C projection heads (debug audit only)."""
    _pc_proj_names = [
        "attr_align_vis_proj_structure",
        "attr_align_vis_proj_boundary",
        "attr_align_vis_proj_instance",
        "attr_align_text_proj",
    ]
    _pc_grad_msgs = []
    for _pn in _pc_proj_names:
        _mod = getattr(model, _pn, None)
        if _mod is None:
            _mod = getattr(getattr(model, "module", None), _pn, None)
        if _mod is None:
            continue
        _total_norm = 0.0
        _n_params = 0
        for _p in _mod.parameters():
            if _p.grad is not None:
                _total_norm += _p.grad.norm(p=2).item() ** 2
                _n_params += 1
        _total_norm = _total_norm ** 0.5 if _n_params > 0 else 0.0
        _pc_grad_msgs.append(f"    {_pn}: grad_norm={_total_norm:.6f}")

    if _pc_grad_msgs and logger is not None:
        logger.info(
            f"[PHASE_C_GRAD_AUDIT] Projection head grad norms ({branch_name}):\n"
            + "\n".join(_pc_grad_msgs)
        )
