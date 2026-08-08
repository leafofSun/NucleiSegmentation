"""Standard instance-segmentation metrics used by the P0.3 audit.

The primary implementations in this module are ports of the public HoVer-Net
``metrics/stats_utils.py`` implementation at commit
``67e2ce5e3f1a64a2ece77ad1c24233653a9e0901``.  The PQ kernel is also
byte-for-byte equivalent in its matching logic to PanNuke-metrics ``utils.py``
at commit ``c00014d766ca1be142b81bea19d9ef4315cde65a``.

Only explicit empty-input handling and typed return values have been added
around the upstream algorithms.  ``pq_independent`` is intentionally separate:
it is the audit's independently written cross-check, not the reported primary
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


HOVERNET_COMMIT = "67e2ce5e3f1a64a2ece77ad1c24233653a9e0901"
HOVERNET_STATS_UTILS_SHA256 = (
    "34dd46f6ed9692a4c74ac723c73ebfd2f88397e4f7bad538b11257d6a17c0c68"
)
PANNuke_METRICS_COMMIT = "c00014d766ca1be142b81bea19d9ef4315cde65a"
PANNuke_UTILS_SHA256 = (
    "53890787f039e98e1d2b64a5421de8b89aee42a9f6608a388dc2aa7dbc6044a4"
)
PANNuke_RUN_SHA256 = (
    "506c50f6295a6d96f58ab574d9e23b682e4d896a0f12d36b1ee1576e93f5313e"
)


@dataclass(frozen=True)
class PQResult:
    """PQ and the sufficient statistics needed for global aggregation."""

    dq: float
    sq: float
    pq: float
    tp: int
    fp: int
    fn: int
    matched_iou_sum: float


def _validate_instance_maps(true: np.ndarray, pred: np.ndarray) -> None:
    if true.ndim != 2 or pred.ndim != 2:
        raise ValueError("instance maps must both be 2-D")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: true={true.shape}, pred={pred.shape}")
    if np.any(true < 0) or np.any(pred < 0):
        raise ValueError("instance IDs must be non-negative")
    if not np.issubdtype(true.dtype, np.integer):
        raise TypeError(f"true must have integer dtype, got {true.dtype}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise TypeError(f"pred must have integer dtype, got {pred.dtype}")


def remap_label(instance_map: np.ndarray) -> np.ndarray:
    """Port of HoVer-Net ``remap_label(..., by_size=False)``.

    Non-zero IDs are renamed to contiguous IDs in ascending original-ID order.
    """

    instance_ids = list(np.unique(instance_map))
    if 0 in instance_ids:
        instance_ids.remove(0)
    remapped = np.zeros(instance_map.shape, dtype=np.int32)
    for new_id, old_id in enumerate(instance_ids, start=1):
        remapped[instance_map == old_id] = new_id
    return remapped


def _official_pairwise_matrices(
    true: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build intersection and union matrices using the upstream loop order."""

    true_ids = list(np.unique(true))
    pred_ids = list(np.unique(pred))
    true_masks = [None]
    pred_masks = [None]
    for true_id in true_ids[1:]:
        true_masks.append(np.asarray(true == true_id, dtype=np.uint8))
    for pred_id in pred_ids[1:]:
        pred_masks.append(np.asarray(pred == pred_id, dtype=np.uint8))

    pairwise_inter = np.zeros(
        (len(true_ids) - 1, len(pred_ids) - 1), dtype=np.float64
    )
    pairwise_union = np.zeros_like(pairwise_inter)
    for true_id in true_ids[1:]:
        true_mask = true_masks[true_id]
        overlapping_pred_ids = np.unique(pred[true_mask > 0])
        for pred_id in overlapping_pred_ids:
            if pred_id == 0:
                continue
            pred_mask = pred_masks[pred_id]
            total = (true_mask + pred_mask).sum()
            intersection = (true_mask * pred_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = intersection
            pairwise_union[true_id - 1, pred_id - 1] = total - intersection
    return pairwise_inter, pairwise_union


def pq_official(
    true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5
) -> PQResult:
    """PanNuke/HoVer-Net PQ port, using the strict ``IoU > threshold`` rule.

    Empty convention added by this audit: both empty maps score 1; a one-sided
    empty map scores 0.  The P0.3 dataset-level PanNuke mean separately skips
    GT-empty images, matching the upstream PanNuke ``run.py`` protocol.
    """

    _validate_instance_maps(true, pred)
    if match_iou < 0:
        raise ValueError("match_iou cannot be negative")
    true = remap_label(true)
    pred = remap_label(pred)
    true_count = int(true.max())
    pred_count = int(pred.max())
    if true_count == 0 and pred_count == 0:
        return PQResult(1.0, 1.0, 1.0, 0, 0, 0, 0.0)
    if true_count == 0:
        return PQResult(0.0, 0.0, 0.0, 0, pred_count, 0, 0.0)
    if pred_count == 0:
        return PQResult(0.0, 0.0, 0.0, 0, 0, true_count, 0.0)

    pairwise_inter, pairwise_union = _official_pairwise_matrices(true, pred)
    pairwise_iou = np.divide(
        pairwise_inter,
        pairwise_union,
        out=np.zeros_like(pairwise_inter),
        where=pairwise_union > 0,
    )
    if match_iou >= 0.5:
        paired_true, paired_pred = np.nonzero(pairwise_iou > match_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        valid = paired_iou > match_iou
        paired_true = paired_true[valid]
        paired_pred = paired_pred[valid]
        paired_iou = paired_iou[valid]

    tp = int(len(paired_true))
    fp = pred_count - tp
    fn = true_count - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denominator) if denominator else 1.0
    matched_iou_sum = float(paired_iou.sum())
    # Upstream uses tp + 1e-6. Its only observable effect is when tp == 0,
    # where both forms return zero; using the exact quotient avoids biasing SQ.
    sq = matched_iou_sum / tp if tp else 0.0
    return PQResult(dq, sq, dq * sq, tp, fp, fn, matched_iou_sum)


def aji_kumar_greedy(true: np.ndarray, pred: np.ndarray) -> float:
    """Port of HoVer-Net ``get_fast_aji`` (one best-IoU pred per GT)."""

    _validate_instance_maps(true, pred)
    true = remap_label(true)
    pred = remap_label(pred)
    true_count = int(true.max())
    pred_count = int(pred.max())
    if true_count == 0 and pred_count == 0:
        return 1.0
    if true_count == 0 or pred_count == 0:
        return 0.0

    pairwise_inter, pairwise_union = _official_pairwise_matrices(true, pred)
    # Preserve the 1e-6 denominator from HoVer-Net's public AJI routine.
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    best_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(best_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    overall_inter = float(pairwise_inter[paired_true, paired_pred].sum())
    overall_union = float(pairwise_union[paired_true, paired_pred].sum())

    paired_true_ids = set((paired_true + 1).tolist())
    paired_pred_ids = set((paired_pred + 1).tolist())
    for true_id in range(1, true_count + 1):
        if true_id not in paired_true_ids:
            overall_union += float((true == true_id).sum())
    for pred_id in range(1, pred_count + 1):
        if pred_id not in paired_pred_ids:
            overall_union += float((pred == pred_id).sum())
    return overall_inter / overall_union if overall_union else 1.0


def aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    """Port of HoVer-Net ``get_fast_aji_plus`` (Hungarian on IoU)."""

    _validate_instance_maps(true, pred)
    true = remap_label(true)
    pred = remap_label(pred)
    true_count = int(true.max())
    pred_count = int(pred.max())
    if true_count == 0 and pred_count == 0:
        return 1.0
    if true_count == 0 or pred_count == 0:
        return 0.0

    pairwise_inter, pairwise_union = _official_pairwise_matrices(true, pred)
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    paired_iou = pairwise_iou[paired_true, paired_pred]
    valid = paired_iou > 0.0
    paired_true = paired_true[valid]
    paired_pred = paired_pred[valid]
    overall_inter = float(pairwise_inter[paired_true, paired_pred].sum())
    overall_union = float(pairwise_union[paired_true, paired_pred].sum())
    paired_true_ids = set((paired_true + 1).tolist())
    paired_pred_ids = set((paired_pred + 1).tolist())
    for true_id in range(1, true_count + 1):
        if true_id not in paired_true_ids:
            overall_union += float((true == true_id).sum())
    for pred_id in range(1, pred_count + 1):
        if pred_id not in paired_pred_ids:
            overall_union += float((pred == pred_id).sum())
    return overall_inter / overall_union if overall_union else 1.0


def binary_dice(true: np.ndarray, pred: np.ndarray) -> float:
    """Traditional binary foreground Dice, matching HoVer-Net ``get_dice_1``."""

    _validate_instance_maps(true, pred)
    true_fg = true > 0
    pred_fg = pred > 0
    denominator = int(true_fg.sum() + pred_fg.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(true_fg, pred_fg).sum() / denominator)


def pq_independent(
    true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5
) -> PQResult:
    """Independent contingency-table PQ implementation for cross-check only."""

    _validate_instance_maps(true, pred)
    true_ids = np.unique(true)
    pred_ids = np.unique(pred)
    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]
    true_count = len(true_ids)
    pred_count = len(pred_ids)
    if true_count == 0 and pred_count == 0:
        return PQResult(1.0, 1.0, 1.0, 0, 0, 0, 0.0)
    if true_count == 0:
        return PQResult(0.0, 0.0, 0.0, 0, pred_count, 0, 0.0)
    if pred_count == 0:
        return PQResult(0.0, 0.0, 0.0, 0, 0, true_count, 0.0)

    true_index = np.searchsorted(true_ids, true)
    pred_index = np.searchsorted(pred_ids, pred)
    overlap = (true > 0) & (pred > 0)
    encoded = true_index[overlap] * pred_count + pred_index[overlap]
    intersections = np.bincount(
        encoded, minlength=true_count * pred_count
    ).reshape(true_count, pred_count)
    true_areas = np.asarray([(true == value).sum() for value in true_ids])
    pred_areas = np.asarray([(pred == value).sum() for value in pred_ids])
    unions = true_areas[:, None] + pred_areas[None, :] - intersections
    iou = intersections / unions
    if match_iou >= 0.5:
        rows, columns = np.nonzero(iou > match_iou)
    else:
        rows, columns = linear_sum_assignment(-iou)
        valid = iou[rows, columns] > match_iou
        rows, columns = rows[valid], columns[valid]
    matched_iou_sum = float(iou[rows, columns].sum())
    tp = int(len(rows))
    fp = int(pred_count - tp)
    fn = int(true_count - tp)
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denominator) if denominator else 1.0
    sq = matched_iou_sum / tp if tp else 0.0
    return PQResult(dq, sq, dq * sq, tp, fp, fn, matched_iou_sum)


def pq_from_global_counts(
    tp: int, fp: int, fn: int, matched_iou_sum: float
) -> PQResult:
    """Compute dataset-global PQ from accumulated sufficient statistics."""

    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denominator) if denominator else 1.0
    sq = float(matched_iou_sum / tp) if tp else (1.0 if denominator == 0 else 0.0)
    return PQResult(dq, sq, dq * sq, tp, fp, fn, float(matched_iou_sum))
