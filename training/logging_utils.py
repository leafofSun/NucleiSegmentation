"""
DDP rank0-only logging utilities.

Provides:
  - rank0_print: only print on rank 0 (DDP main process)
  - rank0_tqdm:  return tqdm on rank 0, raw iterable otherwise
  - is_main_process: check if current rank is 0
  - rank0_logger_info: logger.info on rank 0 only
  - AuditMode & audit helpers: structured audit log gating

Usage:
    from training.logging_utils import rank0_print, rank0_tqdm, is_main_process, AuditMode, audit_print

    rank0_print(rank, "Hello from main process")
    for batch in rank0_tqdm(loader, rank, desc="Training"):
        ...
"""

import os
from typing import Any, Dict, Iterable, Optional


# ── Audit Mode Constants ──────────────────────────────────────────────
AUDIT_OFF = "off"
AUDIT_BASIC = "basic"
AUDIT_DEBUG = "debug"
VALID_AUDIT_MODES = (AUDIT_OFF, AUDIT_BASIC, AUDIT_DEBUG)


class AuditMode:
    """Audit log level gate for Stage D 'mainline slim-down'.

    Usage::

        audit = AuditMode.from_args(args)   # at startup
        audit.print_debug("[PNUDP_DENSE_TRAIN_AUDIT] ...")
        audit.print_basic("[SB_ATTR_DATA_AUDIT] ...")
    """

    def __init__(self, mode: str = AUDIT_BASIC, pnudp_audit_interval: int = 0):
        assert mode in VALID_AUDIT_MODES, f"Invalid audit_mode={mode!r}, must be one of {VALID_AUDIT_MODES}"
        self._mode = mode
        self._pnudp_audit_interval = pnudp_audit_interval

    # ── factory ────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls, args) -> "AuditMode":
        mode = str(getattr(args, "audit_mode", AUDIT_BASIC)).strip().lower()
        interval = int(getattr(args, "pnudp_audit_interval", 0))
        return cls(mode=mode, pnudp_audit_interval=interval)

    # ── convenience booleans ────────────────────────────────────────────
    @property
    def is_off(self) -> bool:
        return self._mode == AUDIT_OFF

    @property
    def is_basic(self) -> bool:
        return self._mode == AUDIT_BASIC

    @property
    def is_debug(self) -> bool:
        return self._mode == AUDIT_DEBUG

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def interval(self) -> int:
        return self._pnudp_audit_interval

    # ── printing helpers ────────────────────────────────────────────────
    def _should_rank0(self, rank: int) -> bool:
        return int(os.environ.get("RANK", "0")) == 0 or rank == 0

    def print_off(self, *args, rank: int = 0, **kwargs):
        """Always printed regardless of audit mode (off/basic/debug)."""
        if self._should_rank0(rank):
            print(*args, **kwargs)

    def print_basic(self, *args, rank: int = 0, **kwargs):
        """Printed only when audit_mode is basic or debug."""
        if self._should_rank0(rank) and self._mode in (AUDIT_BASIC, AUDIT_DEBUG):
            print(*args, **kwargs)

    def print_debug(self, *args, rank: int = 0, **kwargs):
        """Printed only when audit_mode is debug."""
        if self._should_rank0(rank) and self._mode == AUDIT_DEBUG:
            print(*args, **kwargs)

    def print_batch_debug(self, batch_idx: int, *args, rank: int = 0, **kwargs):
        """Printed only when audit_mode=debug, respecting pnudp_audit_interval.

        - If interval == 0: only batch 0 is printed.
        - If interval > 0:  printed every `interval` batches.
        """
        if self._should_rank0(rank) and self._mode == AUDIT_DEBUG:
            if self._pnudp_audit_interval <= 0:
                if batch_idx == 0:
                    print(*args, **kwargs)
            elif batch_idx % self._pnudp_audit_interval == 0:
                print(*args, **kwargs)

    def logger_off(self, logger, rank: int, msg: str, *args, **kwargs):
        """Always logged via logger.info regardless of audit mode."""
        if self._should_rank0(rank) and logger is not None:
            logger.info(msg, *args, **kwargs)

    def logger_basic(self, logger, rank: int, msg: str, *args, **kwargs):
        """Logged via logger.info only when audit_mode is basic or debug."""
        if self._should_rank0(rank) and logger is not None and self._mode in (AUDIT_BASIC, AUDIT_DEBUG):
            logger.info(msg, *args, **kwargs)

    def logger_debug(self, logger, rank: int, msg: str, *args, **kwargs):
        """Logged via logger.info only when audit_mode is debug."""
        if self._should_rank0(rank) and logger is not None and self._mode == AUDIT_DEBUG:
            logger.info(msg, *args, **kwargs)


# ── legacy audit_print function (drop-in compatible) ──────────────────
_AUDIT_MODE_GLOBAL: Optional[AuditMode] = None

# Track keys that have been printed once (for _ADAPTER_ONCE_KEYS one-shot gating)
_ONCE_KEYS_CALLED: set = set()


def set_global_audit_mode(audit: AuditMode):
    """Set module-level global AuditMode for use by audit_print()."""
    global _AUDIT_MODE_GLOBAL
    _AUDIT_MODE_GLOBAL = audit


def audit_print(audit_key: str, *args, batch_idx: Optional[int] = None, rank: int = 0, **kwargs):
    """Unified audit print gating by audit key.

    This implements the mapping defined in the Stage D spec:

    Keys requiring audit_mode=debug:
      TEST_FORWARD_AUDIT, V3_INJECTION_ABLATION_AUDIT,
      PNUDP_DENSE_FUSION_CONSISTENCY, PNUDP_DENSE_PER_CHANNEL_AUDIT,
      PNUDP_DENSE_LOSS_CONSISTENCY, PNUDP_DENSE_TRAIN_SHAPE_AUDIT,
      PNUDP_DENSE_OUTPUT_KEY_AUDIT, PNUDP_DENSE_FUSION_DTYPE_AUDIT,
      PNUDP_DENSE_EVAL_SCALE_AUDIT, PNUDP_DENSE_TRAIN_AUDIT

    Keys printed once (batch 0) in basic mode, always in debug:
      SB_ATTR_BATCH_AUDIT

    Keys printed once in basic mode (first occurrence), always in debug:
      SB_ATTR_DATA_AUDIT, PNUDP_TEXT_PROMPT_SOURCE_AUDIT,
      PNUDP_SAMPLE_ATTR_PROMPT_AUDIT, PNUDP_DENSE_LOGIT_PROJ_INIT,
      PNUDP_DENSE_TRAIN_INIT, PNUDP_DENSE_TRAIN_OPTIMIZER_GROUPS,
      PNUDP_DENSE_TRAIN_CKPT, PNUDP_DENSE_TRAIN_RESUME_AUDIT

    Adapter one-shot keys (no batch_idx needed; basic=first call, debug=always):
      PROMPTNU_GUIDED_V3_ADAPTER_SHAPE_AUDIT
    """
    audit = _AUDIT_MODE_GLOBAL
    if audit is None:
        # Fallback: no global audit configured, print everything
        if rank == 0:
            print(*args, **kwargs)
        return

    if audit.is_off:
        return

    # ── Keys requiring audit_mode=debug ──
    _DEBUG_KEYS = {
        "TEST_FORWARD_AUDIT",
        "V3_INJECTION_ABLATION_AUDIT",
        "PNUDP_DENSE_FUSION_CONSISTENCY",
        "PNUDP_DENSE_PER_CHANNEL_AUDIT",
        "PNUDP_DENSE_LOSS_CONSISTENCY",
        "PNUDP_DENSE_TRAIN_SHAPE_AUDIT",
        "PNUDP_DENSE_OUTPUT_KEY_AUDIT",
        "PNUDP_DENSE_FUSION_DTYPE_AUDIT",
        "PNUDP_DENSE_EVAL_SCALE_AUDIT",
        "PNUDP_DENSE_TRAIN_AUDIT",
        "PROMPTNU_GUIDED_V3_FWD",
        "PROMPTNU_GUIDED_V3_EFFECT_AUDIT",
        "PROMPTNU_GUIDED_V3_GRAPH_AUDIT",
        "PROMPTNU_GUIDED_V3_ALIGN_STABILITY_AUDIT",
        "PROMPTNU_GUIDED_V3_RETAIN_GRAD_SKIP",
        "PNUDP_FUSION_INCONSISTENT",
    }

    # ── Keys printed once (batch 0) in basic+ ──
    _BASIC_ONCE_KEYS = {
        "SB_ATTR_BATCH_AUDIT",
    }

    # ── Adapter one-shot keys: basic=first call only, debug=always ──
    _ADAPTER_ONCE_KEYS = {
        "PROMPTNU_GUIDED_V3_ADAPTER_SHAPE_AUDIT",
    }

    if audit_key in _DEBUG_KEYS:
        if not audit.is_debug:
            return
        if batch_idx is not None:
            audit.print_batch_debug(batch_idx, *args, rank=rank, **kwargs)
        else:
            audit.print_debug(*args, rank=rank, **kwargs)
        return

    if audit_key in _BASIC_ONCE_KEYS:
        if audit.is_basic:
            # Only batch 0 in basic mode
            if batch_idx is not None and batch_idx > 0:
                return
        # In debug mode, always print (batch 0)
        if batch_idx is not None and batch_idx > 0 and not audit.is_debug:
            return
        audit.print_basic(*args, rank=rank, **kwargs)
        return

    # ── Adapter one-shot keys ──
    if audit_key in _ADAPTER_ONCE_KEYS:
        if not audit.is_debug and not audit.is_basic:
            return
        if audit.is_debug:
            # debug mode: always print
            audit.print_basic(*args, rank=rank, **kwargs)
            return
        # basic mode: one-shot (first call only)
        if audit_key in _ONCE_KEYS_CALLED:
            return
        _ONCE_KEYS_CALLED.add(audit_key)
        audit.print_basic(*args, rank=rank, **kwargs)
        return

    # Default: print in basic or debug
    audit.print_basic(*args, rank=rank, **kwargs)


def is_main_process(rank: int) -> bool:
    """Return True if *rank* is 0 (main process in DDP)."""
    return rank == 0


def rank0_print(rank: int, *args: Any, **kwargs: Any) -> None:
    """Only print when *rank* is 0.  Signature matches built-in ``print``."""
    if rank == 0:
        print(*args, **kwargs)


def rank0_tqdm(iterable: Iterable, rank: int, **kwargs: Any):
    """
    Wrap *iterable* with tqdm on rank 0; return the raw iterable on other ranks.

    Usage::

        for batch in rank0_tqdm(dataloader, rank, desc=f"Ep {epoch} Train"):
            ...
    """
    if rank == 0:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    return iterable


def rank0_logger_info(logger: Any, rank: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """Call ``logger.info(msg, *args, **kwargs)`` only on rank 0."""
    if rank == 0 and logger is not None:
        logger.info(msg, *args, **kwargs)


def rank0_logger_warning(logger: Any, rank: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """Call ``logger.warning(msg, *args, **kwargs)`` only on rank 0."""
    if rank == 0 and logger is not None:
        logger.warning(msg, *args, **kwargs)


def rank0_logger_error(logger: Any, rank: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """Call ``logger.error(msg, *args, **kwargs)`` only on rank 0."""
    if rank == 0 and logger is not None:
        logger.error(msg, *args, **kwargs)
