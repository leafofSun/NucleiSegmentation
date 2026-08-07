"""RSGR-1 region-semantic grounding primitives (model-agnostic, CPU testable)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from training.rsgr_local5 import (
    DEFAULT_SCHEMA_PATH, attributes_for_group, load_local5_schema,
    local5_classification_stats, sha256_file,
)

RSGR_MODES = ("no_local", "correct_local", "shuffled_region", "random_prototype")
_LOCAL5_SCHEMA = load_local5_schema(DEFAULT_SCHEMA_PATH)
STRUCTURE_ATTR_NAMES = tuple(
    row["name"] for row in attributes_for_group(_LOCAL5_SCHEMA, "structure")
)
BOUNDARY_ATTR_NAMES = tuple(
    row["name"] for row in attributes_for_group(_LOCAL5_SCHEMA, "boundary")
)
DERANGEMENTS = ((1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))


@dataclass(frozen=True)
class RegionMetadata:
    index: int
    name: str
    original_xyxy: Tuple[int, int, int, int]
    mapped_xyxy: Tuple[int, int, int, int]


class FixedOverlappingRegionLayout:
    """Four 192/256 overlapping windows with centralized coordinate mapping."""
    NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")

    def __init__(self, image_size: int = 256, region_size: int = 192, num_regions: int = 4):
        if (image_size, region_size, num_regions) != (256, 192, 4):
            raise ValueError("RSGR-1 requires image_size=256, region_size=192, num_regions=4")
        self.image_size = image_size
        self.region_size = region_size
        self.num_regions = num_regions

    @property
    def original_coordinates(self) -> Tuple[Tuple[int, int, int, int], ...]:
        offset = self.image_size - self.region_size
        return ((0, 0, 192, 192), (offset, 0, 256, 192),
                (0, offset, 192, 256), (offset, offset, 256, 256))

    def coordinates_for_size(self, target_height: int, target_width: Optional[int] = None):
        width = int(target_width if target_width is not None else target_height)
        height = int(target_height)
        if height <= 0 or width <= 0:
            raise ValueError("target dimensions must be positive")
        mapped = []
        for x0, y0, x1, y1 in self.original_coordinates:
            mapped.append((
                round(x0 * width / self.image_size), round(y0 * height / self.image_size),
                round(x1 * width / self.image_size), round(y1 * height / self.image_size),
            ))
        return tuple(mapped)

    def metadata(self, target_height: int, target_width: Optional[int] = None):
        mapped = self.coordinates_for_size(target_height, target_width)
        return tuple(RegionMetadata(i, self.NAMES[i], self.original_coordinates[i], mapped[i])
                     for i in range(4))

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got {tuple(features.shape)}")
        coords = self.coordinates_for_size(features.shape[-2], features.shape[-1])
        regions = [features[..., y0:y1, x0:x1].mean(dim=(-2, -1))
                   for x0, y0, x1, y1 in coords]
        return torch.stack(regions, dim=1)


def soft_prototype_mixture(probabilities: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Probability-weight each attribute's 3 classes, then mean over attributes."""
    if probabilities.ndim != 4 or prototypes.ndim != 3:
        raise ValueError("expected probabilities [B,R,A,3], prototypes [A,3,D]")
    if probabilities.shape[2:4] != prototypes.shape[:2]:
        raise ValueError("probability/prototype attribute dimensions differ")
    per_attribute = torch.einsum("brac,acd->brad", probabilities, prototypes)
    return per_attribute.mean(dim=2)


def statistics_matched_random_bank(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate a seeded random direction for each prototype with its exact norm."""
    if reference.ndim != 3 or reference.shape[1] != 3:
        raise ValueError("reference prototype bank must be [A,3,D]")
    cpu_reference = reference.detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(cpu_reference.shape, generator=generator, dtype=torch.float32)
    random = F.normalize(random, dim=-1)
    random = random * cpu_reference.norm(dim=-1, keepdim=True)
    if torch.equal(random, cpu_reference):
        raise RuntimeError("seeded random prototype generation did not change the bank")
    return random.to(dtype=reference.dtype, device=reference.device)


def deterministic_derangement(batch_size: int, seed: int, device=None) -> torch.Tensor:
    choices = [DERANGEMENTS[(int(seed) + i) % len(DERANGEMENTS)] for i in range(batch_size)]
    return torch.tensor(choices, dtype=torch.long, device=device)


def apply_region_permutation(vectors: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 3 or permutation.shape != vectors.shape[:2]:
        raise ValueError("vectors/permutation shapes must be [B,R,D] and [B,R]")
    return vectors.gather(1, permutation.unsqueeze(-1).expand_as(vectors))


class RegionSemanticMapBuilder(nn.Module):
    def __init__(self, layout: Optional[FixedOverlappingRegionLayout] = None):
        super().__init__()
        self.layout = layout or FixedOverlappingRegionLayout()

    def forward(self, region_vectors: torch.Tensor, height: int, width: int):
        if region_vectors.ndim != 3 or region_vectors.shape[1] != 4:
            raise ValueError("region_vectors must be [B,4,C]")
        b, _, c = region_vectors.shape
        result = region_vectors.new_zeros((b, c, height, width))
        weights = region_vectors.new_zeros((b, 1, height, width))
        for index, (x0, y0, x1, y1) in enumerate(self.layout.coordinates_for_size(height, width)):
            result[..., y0:y1, x0:x1] += region_vectors[:, index, :, None, None]
            weights[..., y0:y1, x0:x1] += 1.0
        if torch.any(weights == 0):
            raise RuntimeError("fixed RSGR windows did not cover the full feature map")
        return result / weights, weights.reciprocal()


def bounded_residual(raw_delta: torch.Tensor, base_feature: torch.Tensor,
                     max_ratio: float, injection_scale: float):
    if max_ratio < 0 or injection_scale < 0:
        raise ValueError("residual bounds must be non-negative")
    base_rms = base_feature.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
    bounded = torch.tanh(raw_delta.float()) * base_rms * float(max_ratio)
    injected = bounded * float(injection_scale)
    delta_rms = injected.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    ratio = delta_rms / base_rms
    return injected.to(raw_delta.dtype), ratio


def local_attribute_ce(logits: torch.Tensor, labels: Optional[torch.Tensor]):
    if labels is None:
        return logits.sum() * 0.0, 0
    if labels.shape != logits.shape[:-1]:
        raise ValueError(f"label/logit shape mismatch: {tuple(labels.shape)} vs {tuple(logits.shape)}")
    flat_labels = labels.reshape(-1).long()
    valid = flat_labels >= 0
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    loss = F.cross_entropy(logits.reshape(-1, 3)[valid], flat_labels[valid])
    return loss, count


def prototype_hash(structure: torch.Tensor, boundary: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (structure, boundary):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_prototype_banks(
    path: str | Path,
    metadata_path: Optional[str | Path] = None,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
):
    """Strictly load a schema-bound formal Local-5 CONCH prototype bank."""
    bank_path = Path(path)
    meta_path = Path(metadata_path) if metadata_path is not None else bank_path.with_suffix(".metadata.json")
    schema = load_local5_schema(schema_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "rsgr_local5_conch_bank_v1":
        raise ValueError("unsupported Local-5 prototype metadata version")
    if metadata.get("backend") != "conch":
        raise ValueError("Local-5 prototype metadata backend must be conch")
    if metadata.get("bank_sha256") != sha256_file(bank_path):
        raise ValueError("Local-5 prototype bank SHA256 mismatch")
    if metadata.get("schema_sha256") != sha256_file(schema_path):
        raise ValueError("Local-5 prototype schema SHA256 mismatch")
    expected_names = {
        group: [row["name"] for row in attributes_for_group(schema, group)]
        for group in ("structure", "boundary")
    }
    expected_classes = list(schema["classes"])
    if metadata.get("attribute_names") != expected_names:
        raise ValueError("Local-5 prototype attribute name/order mismatch")
    if metadata.get("class_names") != expected_classes:
        raise ValueError("Local-5 prototype level/class order mismatch")
    if "level_order" in metadata and metadata.get("level_order") != expected_classes:
        raise ValueError("Local-5 prototype level/class order mismatch")
    payload = torch.load(str(bank_path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "rsgr_local5_conch_bank_v1"
        or payload.get("backend") != "conch"
    ):
        raise ValueError("prototype file must be a formal CONCH mapping")
    if payload.get("attribute_names") != expected_names:
        raise ValueError("Local-5 payload attribute name/order mismatch")
    if payload.get("class_names") != expected_classes:
        raise ValueError("Local-5 payload level/class order mismatch")
    banks = []
    for group in ("structure", "boundary"):
        value = payload.get(f"{group}_prototypes")
        expected_shape = (len(expected_names[group]), 3, int(metadata["embedding_dim"]))
        if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
            raise ValueError(f"invalid {group} prototype shape")
        value = value.detach().float()
        if not torch.isfinite(value).all():
            raise ValueError(f"{group} prototypes contain non-finite values")
        norms = value.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
            raise ValueError(f"{group} prototypes are not unit-normalized")
        banks.append(value)
    return banks[0], banks[1]


class RegionSemanticGrounding(nn.Module):
    """Predicted local attributes -> frozen prototype maps -> bounded branch deltas."""
    def __init__(self, input_dim: int, structure_channels: int, boundary_channels: int,
                 structure_prototypes: torch.Tensor, boundary_prototypes: torch.Tensor,
                 mode: str = "no_local", injection_scale: float = 0.05,
                 max_injection_ratio: float = 0.02, random_seed: int = 42,
                 attr_detach: bool = False, prototype_detach: bool = True):
        super().__init__()
        if mode not in RSGR_MODES:
            raise ValueError(f"rsgr_mode must be one of {RSGR_MODES}, got {mode}")
        if not prototype_detach:
            raise ValueError("RSGR-1 prototypes must remain detached/frozen")
        if structure_prototypes.shape[:2] != (len(STRUCTURE_ATTR_NAMES), 3) or boundary_prototypes.shape[:2] != (len(BOUNDARY_ATTR_NAMES), 3):
            raise ValueError("RSGR Local-5 prototypes must be [3,3,D] and [2,3,D]")
        if structure_prototypes.shape[-1] != boundary_prototypes.shape[-1]:
            raise ValueError("prototype dimensions must match")
        self.mode = mode
        self.injection_scale = float(injection_scale)
        self.max_injection_ratio = float(max_injection_ratio)
        self.random_seed = int(random_seed)
        self.attr_detach = bool(attr_detach)
        self.layout = FixedOverlappingRegionLayout()
        self.local5_structure_predictor = nn.Linear(input_dim, len(STRUCTURE_ATTR_NAMES) * 3)
        self.local5_boundary_predictor = nn.Linear(input_dim, len(BOUNDARY_ATTR_NAMES) * 3)
        text_dim = structure_prototypes.shape[-1]
        self.structure_text_norm = nn.LayerNorm(text_dim)
        self.boundary_text_norm = nn.LayerNorm(text_dim)
        self.structure_text_projector = nn.Linear(text_dim, structure_channels)
        self.boundary_text_projector = nn.Linear(text_dim, boundary_channels)
        self.structure_region_adapter = nn.Conv2d(structure_channels, structure_channels, 1)
        self.boundary_region_adapter = nn.Conv2d(boundary_channels, boundary_channels, 1)
        self.map_builder = RegionSemanticMapBuilder(self.layout)
        self.register_buffer("structure_prototypes", structure_prototypes.detach().float().clone(), persistent=True)
        self.register_buffer("boundary_prototypes", boundary_prototypes.detach().float().clone(), persistent=True)
        self.register_buffer("random_structure_prototypes",
                             statistics_matched_random_bank(structure_prototypes.float(), random_seed), persistent=True)
        self.register_buffer("random_boundary_prototypes",
                             statistics_matched_random_bank(boundary_prototypes.float(), random_seed + 1), persistent=True)

    @property
    def injection_enabled(self):
        return self.mode != "no_local"

    def _banks(self):
        if self.mode == "random_prototype":
            return self.random_structure_prototypes, self.random_boundary_prototypes
        return self.structure_prototypes, self.boundary_prototypes

    @staticmethod
    def assert_no_eval_gt(training: bool, local_structure_labels=None, local_boundary_labels=None,
                          gt_mask=None, instance_ids=None):
        if not training and any(value is not None for value in
                                (local_structure_labels, local_boundary_labels, gt_mask, instance_ids)):
            raise RuntimeError("[RSGR_GT_GUARD] eval/test forward received forbidden local GT input")

    def forward(self, structure_feature: torch.Tensor, boundary_feature: Optional[torch.Tensor] = None,
                local_structure_labels=None, local_boundary_labels=None, gt_mask=None, instance_ids=None):
        boundary_feature = structure_feature if boundary_feature is None else boundary_feature
        self.assert_no_eval_gt(self.training, local_structure_labels, local_boundary_labels, gt_mask, instance_ids)
        s_regions = self.layout.pool(structure_feature)
        b_regions = self.layout.pool(boundary_feature)
        b, r, channels = s_regions.shape
        if b_regions.shape != (b, r, channels):
            raise ValueError("structure/boundary region feature shapes are incompatible")
        s_logits = self.local5_structure_predictor(s_regions.reshape(b * r, channels)).view(
            b, r, len(STRUCTURE_ATTR_NAMES), 3
        )
        b_logits = self.local5_boundary_predictor(b_regions.reshape(b * r, channels)).view(
            b, r, len(BOUNDARY_ATTR_NAMES), 3
        )
        s_prob, b_prob = torch.softmax(s_logits, -1), torch.softmax(b_logits, -1)
        if self.attr_detach:
            s_prob, b_prob = s_prob.detach(), b_prob.detach()
        s_bank, b_bank = self._banks()
        s_sem = self.structure_text_norm(soft_prototype_mixture(s_prob, s_bank.detach()))
        b_sem = self.boundary_text_norm(soft_prototype_mixture(b_prob, b_bank.detach()))
        permutation = torch.arange(4, device=s_sem.device).expand(s_sem.shape[0], 4)
        if self.mode == "shuffled_region":
            permutation = deterministic_derangement(s_sem.shape[0], self.random_seed, s_sem.device)
            s_sem = apply_region_permutation(s_sem, permutation)
            b_sem = apply_region_permutation(b_sem, permutation)
        s_region_delta = self.structure_text_projector(s_sem)
        b_region_delta = self.boundary_text_projector(b_sem)
        s_map, s_overlap = self.map_builder(s_region_delta, structure_feature.shape[-2], structure_feature.shape[-1])
        b_map, b_overlap = self.map_builder(b_region_delta, boundary_feature.shape[-2], boundary_feature.shape[-1])
        s_raw = self.structure_region_adapter(s_map)
        b_raw = self.boundary_region_adapter(b_map)
        s_injected, s_ratio = bounded_residual(s_raw, structure_feature, self.max_injection_ratio, self.injection_scale)
        b_injected, b_ratio = bounded_residual(b_raw, boundary_feature, self.max_injection_ratio, self.injection_scale)
        if not self.injection_enabled:
            s_injected = s_injected * 0.0
            b_injected = b_injected * 0.0
            s_ratio = s_ratio * 0.0
            b_ratio = b_ratio * 0.0
        s_loss, s_valid = local_attribute_ce(s_logits, local_structure_labels if self.training else None)
        b_loss, b_valid = local_attribute_ce(b_logits, local_boundary_labels if self.training else None)
        s_stats = local5_classification_stats(s_logits, local_structure_labels) if self.training and local_structure_labels is not None else None
        b_stats = local5_classification_stats(b_logits, local_boundary_labels) if self.training and local_boundary_labels is not None else None
        diagnostics = {
            "RSGRActive": True, "RSGRInjectionEnabled": self.injection_enabled,
            "local_attr_source": "pred", "GTUsedForInjection": False,
            "gt_labels_used_for_aux_loss": bool(self.training and (s_valid + b_valid) > 0),
            "local5_attribute_names": {
                "structure": list(STRUCTURE_ATTR_NAMES),
                "boundary": list(BOUNDARY_ATTR_NAMES),
            },
            "prototype_hash": prototype_hash(s_bank, b_bank),
            "shuffle_permutation": permutation.detach().cpu().tolist(),
            "StructAttrEntropy": float((-(s_prob * s_prob.clamp_min(1e-8).log()).sum(-1).mean()).detach()),
            "BoundAttrEntropy": float((-(b_prob * b_prob.clamp_min(1e-8).log()).sum(-1).mean()).detach()),
            "StructSpatialStd": float(s_map.detach().float().std()),
            "BoundSpatialStd": float(b_map.detach().float().std()),
            "StructActualDeltaRatio": float(s_ratio.detach().max()),
            "BoundActualDeltaRatio": float(b_ratio.detach().max()),
            "LocalStructLoss": float(s_loss.detach()), "LocalBoundLoss": float(b_loss.detach()),
            "valid_structure_labels": s_valid, "valid_boundary_labels": b_valid,
            "local5_structure_metrics": s_stats, "local5_boundary_metrics": b_stats,
        }
        return {
            "local_structure_logits": s_logits, "local_boundary_logits": b_logits,
            "structure_probabilities": s_prob, "boundary_probabilities": b_prob,
            "structure_semantics": s_sem, "boundary_semantics": b_sem,
            "structure_semantic_map": s_map, "boundary_semantic_map": b_map,
            "structure_overlap_reciprocal": s_overlap, "boundary_overlap_reciprocal": b_overlap,
            "structure_delta": s_injected, "boundary_delta": b_injected,
            "structure_ratio": s_ratio, "boundary_ratio": b_ratio,
            "local_structure_loss": s_loss, "local_boundary_loss": b_loss,
            "valid_structure_labels": s_valid, "valid_boundary_labels": b_valid,
            "permutation": permutation, "diagnostics": diagnostics,
        }


def parameter_name_hash(module: nn.Module) -> str:
    names = sorted(name for name, parameter in module.named_parameters() if parameter.requires_grad)
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def parameter_name_shape_hash(module: nn.Module) -> str:
    rows = sorted(
        f"{name}:{tuple(parameter.shape)}" for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def checkpoint_compatibility_report(model_keys: Sequence[str], checkpoint_keys: Sequence[str]) -> Dict[str, List[str]]:
    """Classify old-checkpoint key differences without loading tensors or CUDA."""
    model, checkpoint = set(model_keys), set(checkpoint_keys)
    missing, unexpected = sorted(model - checkpoint), sorted(checkpoint - model)
    return {
        "rsgr_missing": [key for key in missing if key.startswith("rsgr.")],
        "non_rsgr_missing": [key for key in missing if not key.startswith("rsgr.")],
        "rsgr_unexpected": [key for key in unexpected if key.startswith("rsgr.")],
        "non_rsgr_unexpected": [key for key in unexpected if not key.startswith("rsgr.")],
    }


def optimizer_group_spec(module: RegionSemanticGrounding, lr: float):
    groups = {
        "rsgr_local_attr_predictor": sorted(
            [f"local5_structure_predictor.{name}" for name, _ in module.local5_structure_predictor.named_parameters()]
            + [f"local5_boundary_predictor.{name}" for name, _ in module.local5_boundary_predictor.named_parameters()]
        ),
        "rsgr_text_projector": sorted(
            [f"structure_text_norm.{name}" for name, _ in module.structure_text_norm.named_parameters()]
            + [f"boundary_text_norm.{name}" for name, _ in module.boundary_text_norm.named_parameters()]
            + [f"structure_text_projector.{name}" for name, _ in module.structure_text_projector.named_parameters()]
            + [f"boundary_text_projector.{name}" for name, _ in module.boundary_text_projector.named_parameters()]
        ),
        "rsgr_region_adapter": sorted(
            [f"structure_region_adapter.{name}" for name, _ in module.structure_region_adapter.named_parameters()]
            + [f"boundary_region_adapter.{name}" for name, _ in module.boundary_region_adapter.named_parameters()]
        ),
    }
    payload = [{"name": name, "lr": float(lr), "parameters": names} for name, names in groups.items()]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload, digest
