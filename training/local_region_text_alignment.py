"""L1-A local-region text alignment primitives.

This module contains only supervision-side logic.  It never modifies image
embeddings or segmentation logits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from skimage.measure import regionprops
from torch import nn
from torch.nn import functional as F


ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "nuclear_density",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "boundary_irregularity",
    "nuclear_elongation",
)

ATTRIBUTE_ALIASES = {
    "density": "nuclear_density",
    "size_heterogeneity": "nuclear_size_heterogeneity",
    "crowding": "spatial_crowding",
    "boundary_irregularity": "boundary_irregularity",
    "elongation": "nuclear_elongation",
}

PROMPT_BANK: Dict[str, Tuple[str, str, str]] = {
    "nuclear_density": (
        "a sparse local nuclear region",
        "a moderately populated local nuclear region",
        "a densely populated local nuclear region",
    ),
    "nuclear_size_heterogeneity": (
        "a local region with uniformly sized nuclei",
        "a local region with moderately variable nuclear sizes",
        "a local region with highly heterogeneous nuclear sizes",
    ),
    "spatial_crowding": (
        "a local region with low nuclear crowding",
        "a local region with moderate nuclear crowding",
        "a local region with high nuclear crowding",
    ),
    "boundary_irregularity": (
        "a local region containing nuclei with smooth boundaries",
        "a local region containing nuclei with moderately irregular boundaries",
        "a local region containing nuclei with highly irregular boundaries",
    ),
    "nuclear_elongation": (
        "a local region containing mostly round nuclei",
        "a local region containing moderately elongated nuclei",
        "a local region containing highly elongated nuclei",
    ),
}

DEFAULT_WINDOW_SIZE = 192
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MODEL_SIZE = 512
DEFAULT_FEATURE_SIZE = 32


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_attribute_names(spec: str | Sequence[str]) -> Tuple[str, ...]:
    parts = spec.split(",") if isinstance(spec, str) else list(spec)
    names = tuple(ATTRIBUTE_ALIASES.get(str(x).strip(), str(x).strip()) for x in parts)
    if names != ATTRIBUTE_NAMES:
        raise ValueError(
            "L1-A attribute set and order are preregistered as "
            f"{ATTRIBUTE_NAMES}, got {names}"
        )
    return names


def region_coordinates(
    image_size: int = DEFAULT_IMAGE_SIZE,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Tuple[Tuple[int, int, int, int], ...]:
    if image_size != 256 or window_size != 192:
        raise ValueError("L1-A is preregistered for image_size=256 and window_size=192")
    starts = (0, image_size - window_size)
    return tuple(
        (x0, y0, x0 + window_size, y0 + window_size)
        for y0 in starts
        for x0 in starts
    )


def feature_region_coordinates(
    image_size: int = DEFAULT_IMAGE_SIZE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    feature_size: int = DEFAULT_FEATURE_SIZE,
) -> Tuple[Tuple[int, int, int, int], ...]:
    if image_size % feature_size:
        raise ValueError("image_size must divide exactly into feature_size")
    stride = image_size // feature_size
    if window_size % stride:
        raise ValueError("window_size must align to the feature grid")
    return tuple(
        (x0 // stride, y0 // stride, x1 // stride, y1 // stride)
        for x0, y0, x1, y1 in region_coordinates(image_size, window_size)
    )


def load_l0_thresholds(path: str | Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train":
        raise ValueError("L1-A thresholds must be fitted on the train split")
    attrs = payload.get("attributes", {})
    result: Dict[str, Dict[str, float]] = {}
    for name in ATTRIBUTE_NAMES:
        record = attrs.get(name)
        if not isinstance(record, dict):
            raise KeyError(f"Missing L0 threshold for {name}")
        if record.get("fit_split") != "train":
            raise ValueError(f"Threshold for {name} is not train-fitted")
        result[name] = {
            "lower": float(record["low_upper_exclusive"]),
            "upper": float(record["medium_upper_inclusive"]),
        }
    return result


def assign_bin(value: float, lower: float, upper: float) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("bool must not be used as a numeric local-attribute label")
    if not math.isfinite(float(value)):
        return -1
    if value < lower:
        return 0
    if value <= upper:
        return 1
    return 2


def _instance_properties(mask: np.ndarray) -> Dict[str, np.ndarray]:
    props = list(regionprops(mask.astype(np.int32, copy=False)))
    areas = np.asarray([float(p.area) for p in props], dtype=np.float64)
    perimeters = np.asarray([float(p.perimeter) for p in props], dtype=np.float64)
    major = np.asarray([float(p.axis_major_length) for p in props], dtype=np.float64)
    minor = np.asarray([float(p.axis_minor_length) for p in props], dtype=np.float64)
    elongation = major / np.maximum(minor, 1e-6)
    irregularity = perimeters / np.maximum(2.0 * np.sqrt(np.pi * areas), 1e-6)
    centroids = np.asarray(
        [[float(p.centroid[1]), float(p.centroid[0])] for p in props],
        dtype=np.float64,
    ).reshape(-1, 2)
    bboxes = np.asarray(
        [[int(p.bbox[1]), int(p.bbox[0]), int(p.bbox[3]), int(p.bbox[2])] for p in props],
        dtype=np.int32,
    ).reshape(-1, 4)
    height, width = mask.shape
    original_partial = (
        (bboxes[:, 0] <= 0)
        | (bboxes[:, 1] <= 0)
        | (bboxes[:, 2] >= width)
        | (bboxes[:, 3] >= height)
    ) if len(props) else np.zeros(0, dtype=bool)
    return {
        "areas": areas,
        "irregularity": irregularity,
        "elongation": elongation,
        "centroids": centroids,
        "bboxes": bboxes,
        "original_partial": original_partial,
    }


def compute_local_region_targets(
    mask: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Any]:
    """Recompute five L1-A labels on the post-geometric-augmentation mask."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"instance mask must be 2-D, got {array.shape}")
    if array.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        raise ValueError(
            "local labels must be computed before resize on the 256x256 "
            f"post-geometric mask, got {array.shape}"
        )
    props = _instance_properties(array)
    coords = region_coordinates(DEFAULT_IMAGE_SIZE, window_size)
    labels = np.full((len(coords), len(ATTRIBUTE_NAMES)), -1, dtype=np.int64)
    valid = np.zeros_like(labels, dtype=bool)
    values = np.full(labels.shape, np.nan, dtype=np.float64)
    complete_counts = np.zeros(len(coords), dtype=np.int64)
    foreground_counts = np.zeros(len(coords), dtype=np.int64)

    bboxes = props["bboxes"]
    partial = props["original_partial"]
    for region_index, (x0, y0, x1, y1) in enumerate(coords):
        within = (
            (bboxes[:, 0] >= x0)
            & (bboxes[:, 1] >= y0)
            & (bboxes[:, 2] <= x1)
            & (bboxes[:, 3] <= y1)
        ) if len(bboxes) else np.zeros(0, dtype=bool)
        complete = within & ~partial
        indices = np.flatnonzero(complete)
        count = int(indices.size)
        complete_counts[region_index] = count
        foreground = int(np.count_nonzero(array[y0:y1, x0:x1]))
        foreground_counts[region_index] = foreground

        attr_values: Dict[str, float] = {}
        attr_valid = {
            "nuclear_density": foreground > 0,
            "nuclear_size_heterogeneity": count >= 2,
            "spatial_crowding": count >= 2,
            "boundary_irregularity": count >= 1,
            "nuclear_elongation": count >= 1,
        }
        attr_values["nuclear_density"] = count / float(window_size * window_size) * 10000.0
        if count:
            areas = props["areas"][indices]
            attr_values["nuclear_size_heterogeneity"] = float(
                np.std(areas) / max(float(np.mean(areas)), 1e-6)
            )
            attr_values["boundary_irregularity"] = float(
                np.mean(props["irregularity"][indices])
            )
            attr_values["nuclear_elongation"] = float(
                np.mean(props["elongation"][indices])
            )
        if count >= 2:
            centers = props["centroids"][indices]
            delta = centers[:, None, :] - centers[None, :, :]
            distances = np.sqrt(np.sum(delta * delta, axis=2))
            np.fill_diagonal(distances, np.inf)
            median_nearest = float(np.median(np.min(distances, axis=1)))
            attr_values["spatial_crowding"] = 100.0 / max(median_nearest, 1e-6)

        for attr_index, name in enumerate(ATTRIBUTE_NAMES):
            if not attr_valid[name]:
                continue
            value = float(attr_values[name])
            threshold = thresholds[name]
            code = assign_bin(value, float(threshold["lower"]), float(threshold["upper"]))
            if code < 0:
                continue
            values[region_index, attr_index] = value
            labels[region_index, attr_index] = code
            valid[region_index, attr_index] = True

    return {
        "coordinates": coords,
        "labels": labels,
        "valid": valid,
        "values": values,
        "complete_instance_count": complete_counts,
        "foreground_pixel_count": foreground_counts,
        "region_count": len(coords),
        "policy": "complete_only",
    }


def validate_no_eval_gt(
    training: bool,
    batched_input: Sequence[Mapping[str, Any]],
) -> None:
    if training:
        return
    forbidden = ("local_region_attr_labels", "local_region_attr_valid")
    for item in batched_input:
        leaked = [key for key in forbidden if key in item]
        if leaked:
            raise RuntimeError(
                "[L1A_GT_LEAKAGE] validation/test forward received local GT keys: "
                + ",".join(leaked)
            )


class LocalVisualProjector(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features).float(), dim=-1, eps=1e-8)


class LocalRegionTextAlignment(nn.Module):
    """Read-only ROI supervision branch; segmentation features are never changed."""

    def __init__(
        self,
        prototype_bank: torch.Tensor,
        temperature: float = 0.07,
        input_dim: int = 256,
        hidden_dim: int = 256,
        attribute_names: Sequence[str] = ATTRIBUTE_NAMES,
    ):
        super().__init__()
        names = tuple(attribute_names)
        if names != ATTRIBUTE_NAMES:
            raise ValueError(f"Unexpected L1-A attributes: {names}")
        bank = torch.as_tensor(prototype_bank, dtype=torch.float32)
        if bank.ndim != 3 or tuple(bank.shape[:2]) != (len(names), 3):
            raise ValueError(f"prototype bank must be [5,3,D], got {tuple(bank.shape)}")
        bank = F.normalize(bank, dim=-1, eps=1e-8)
        self.register_buffer("text_prototypes", bank, persistent=True)
        self.attribute_names = names
        self.temperature = float(temperature)
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        self.projectors = nn.ModuleDict({
            name: LocalVisualProjector(input_dim, int(bank.shape[-1]), hidden_dim)
            for name in names
        })
        self.feature_injection = False

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        temperature: float = 0.07,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ) -> "LocalRegionTextAlignment":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            bank = payload.get("embeddings", payload.get("text_prototypes"))
            names = tuple(payload.get("attribute_names", ATTRIBUTE_NAMES))
        else:
            bank = payload
            names = ATTRIBUTE_NAMES
        return cls(bank, temperature, input_dim, hidden_dim, names)

    @staticmethod
    def pool_regions(image_embeddings: torch.Tensor) -> torch.Tensor:
        if image_embeddings.ndim != 4 or tuple(image_embeddings.shape[1:]) != (256, 32, 32):
            raise ValueError(
                "image embeddings must be [B,256,32,32], got "
                f"{tuple(image_embeddings.shape)}"
            )
        regions = []
        for x0, y0, x1, y1 in feature_region_coordinates():
            roi = image_embeddings[:, :, y0:y1, x0:x1]
            if tuple(roi.shape[-2:]) != (24, 24):
                raise RuntimeError(f"ROI must be 24x24, got {tuple(roi.shape[-2:])}")
            regions.append(roi.mean(dim=(-2, -1)))
        return torch.stack(regions, dim=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, Any]:
        if labels.dtype == torch.bool:
            raise TypeError("bool labels are forbidden")
        labels = labels.long()
        valid = valid.bool()
        batch = image_embeddings.shape[0]
        expected = (batch, 4, len(self.attribute_names))
        if tuple(labels.shape) != expected or tuple(valid.shape) != expected:
            raise ValueError(
                f"labels/valid must both be {expected}, got "
                f"{tuple(labels.shape)} and {tuple(valid.shape)}"
            )
        regions = self.pool_regions(image_embeddings)
        losses: Dict[str, torch.Tensor] = {}
        accuracies: Dict[str, torch.Tensor] = {}
        logits_by_attr: Dict[str, torch.Tensor] = {}
        valid_counts: Dict[str, int] = {}
        active_losses = []
        all_zero_losses = []
        for index, name in enumerate(self.attribute_names):
            projected = self.projectors[name](regions)
            prototypes = self.text_prototypes[index].to(projected)
            logits = torch.einsum("brd,cd->brc", projected, prototypes)
            logits = logits / self.temperature
            mask = valid[:, :, index]
            target = labels[:, :, index]
            zero = projected.sum() * 0.0
            all_zero_losses.append(zero)
            count = int(mask.sum().detach().item())
            valid_counts[name] = count
            if count:
                loss = F.cross_entropy(logits[mask].float(), target[mask])
                accuracy = (logits[mask].argmax(dim=-1) == target[mask]).float().mean()
                active_losses.append(loss)
            else:
                loss = zero
                accuracy = zero.detach()
            losses[name] = loss
            accuracies[name] = accuracy
            logits_by_attr[name] = logits
        if active_losses:
            total = torch.stack(active_losses).mean()
        else:
            total = torch.stack(all_zero_losses).sum()
        return {
            "local_region_text_loss": total,
            "local_region_attribute_losses": losses,
            "local_region_attribute_accuracies": accuracies,
            "local_region_valid_counts": valid_counts,
            "local_region_text_logits": logits_by_attr,
            "local_region_features_shape": tuple(regions.shape),
            "local_region_feature_injection": False,
        }


def maybe_compute_local_alignment(
    module: Optional[LocalRegionTextAlignment],
    image_embeddings: torch.Tensor,
    labels: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
) -> Optional[Dict[str, Any]]:
    if module is None:
        return None
    if labels is None or valid is None:
        return None
    return module(image_embeddings, labels, valid)
