"""
Shared utility functions used across training modules.

Extracted from train.py to break the circular dependency between
training.phase_c_semantic_alignment and train.py.

These functions are copied (not moved) from train.py so that train.py
can continue to define them locally.  Both the originals in train.py
and these copies must be kept in sync.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


# ──────────────────────────────────────────────────────────────────────
# _autocast_context
# ──────────────────────────────────────────────────────────────────────
def _autocast_context(args):
    """Create an autocast context for mixed-precision training.

    Mirrors the definition in train.py.
    """
    if not torch.cuda.is_available():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    return autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16)


# ──────────────────────────────────────────────────────────────────────
# _is_nonnegative_int — safe None-compatible check
# ──────────────────────────────────────────────────────────────────────
def _is_nonnegative_int(value) -> bool:
    """Return True if value is a non-negative integer (>= 0), safely handling None.

    Compatible with argparse defaults of both None and -1.
    """
    try:
        return value is not None and int(value) >= 0
    except (TypeError, ValueError):
        return False


# ──────────────────────────────────────────────────────────────────────
# _is_debug_run
# ──────────────────────────────────────────────────────────────────────
def _is_debug_run(args) -> bool:
    """Return True if the current run is a debug run (any debug flag is set).

    Only debug_* flags trigger debug mode:
      - debug_train_audit / debug_audit_params_only (bool)
      - debug_max_train_batches / debug_max_val_batches (int, None-safe)

    max_train_batches / max_val_batches do NOT trigger debug mode.
    """
    if getattr(args, "debug_train_audit", False):
        return True
    if getattr(args, "debug_audit_params_only", False):
        return True
    if _is_nonnegative_int(getattr(args, "debug_max_train_batches", None)):
        return True
    if _is_nonnegative_int(getattr(args, "debug_max_val_batches", None)):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# _save_checkpoint_file
# ──────────────────────────────────────────────────────────────────────
def _save_checkpoint_file(
    checkpoint_dict: Dict,
    path: str,
    logger=None,
    message: Optional[str] = None,
):
    """Save a checkpoint dict to disk.

    Mirrors the definition in train.py.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint_dict, path)
    if logger is not None and message:
        logger.info(message)


# ──────────────────────────────────────────────────────────────────────
# _to_float_or_nan
# ──────────────────────────────────────────────────────────────────────
def _to_float_or_nan(value):
    """Safely convert a value to float, returning nan on failure.

    Mirrors the definition in train.py.
    """
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────
# _write_scalar_if_finite
# ──────────────────────────────────────────────────────────────────────
def _write_scalar_if_finite(writer, tag: str, value: float, step: int):
    """Write a scalar to TensorBoard if the value is finite (not nan/inf).

    Mirrors the definition in train.py.
    """
    if value is None:
        return
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return
    writer.add_scalar(tag, value, step)


# ──────────────────────────────────────────────────────────────────────
# unwrap_model
# ──────────────────────────────────────────────────────────────────────
def unwrap_model(model):
    """Unwrap DistributedDataParallel wrapper to get the raw module.

    Mirrors the definition in train.py.
    """
    return model.module if hasattr(model, "module") else model
