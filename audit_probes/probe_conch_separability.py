#!/usr/bin/env python3
"""Pre-registered CONCH semantic-prototype separability probe (inference only).

The numerical path deliberately depends only on NumPy.  PyTorch and CONCH are
imported lazily only when raw text must be encoded or a selected bank is
explicitly frozen.  This makes it possible to encode once on the server and
re-run/audit all geometry on a CPU-only machine via ``--embeddings-input``.

The CONCH path mirrors the project's audited production path:

* ``tools/build_l1a_text_prototype_bank.py:53-73``
* ``segment_anything/modeling/sam.py:3115-3134,3226-3234``

No training code, optimizer, dataset, or segmentation metric is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by dependency check
    raise SystemExit(
        "probe_conch_separability.py requires NumPy for its CPU-only geometry. "
        "Install the project's env_files/requirements_my_env.txt environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "training/rsgr_local5_schema.json"
DEFAULT_GLOBAL27_PATH = (
    PROJECT_ROOT / "workdir/attr_stats/structure_boundary_prompt_templates.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "audit_probes/outputs"
DEFAULT_SERVER_CONCH_CHECKPOINT = Path(
    "/hy-tmp/NuSeg/hf_cache/hub/models--MahmoodLab--conch/"
    "snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin"
)
DEFAULT_SERVER_CONCH_CACHE = Path("/hy-tmp/NuSeg/hf_cache/hub")
CONCH_HF_SOURCE = "hf_hub:MahmoodLab/conch"
CONCH_MODEL_NAME = "conch_ViT-B-16"
ENCODER_SOURCE = "audit_probes/probe_conch_separability.py:encode_with_project_conch_path"
PROJECT_ENCODER_REFERENCE = "tools/build_l1a_text_prototype_bank.py:53-73"
TRAINING_ENCODER_SOURCE = "segment_anything/modeling/sam.py:3115-3134,3226-3234"

LEVEL_ORDER: Tuple[str, ...] = ("low", "mid", "high")
SCHEMA_LEVEL_ORDER: Tuple[str, ...] = ("low", "medium", "high")
LOCAL5_ATTRIBUTE_ORDER: Tuple[str, ...] = (
    "nuclear_density",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "nuclear_irregularity",
    "nuclear_elongation",
)
LOCAL5_GROUP_ORDER: Tuple[str, ...] = (
    "structure",
    "structure",
    "structure",
    "boundary",
    "boundary",
)
GLOBAL27_STRUCTURE_ORDER: Tuple[str, ...] = (
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
)
GLOBAL27_BOUNDARY_ORDER: Tuple[str, ...] = (
    "boundary_density",
    "nuclear_irregularity",
    "nuclear_elongation",
    "small_nuclei_ratio",
)
LITERAL_PROMPTS: Tuple[str, str] = (
    "high nuclear density",
    "low nuclear density",
)

# Pre-registered thresholds.  Do not make these command-line configurable.
PREREGISTERED_THRESHOLDS: Mapping[str, Mapping[str, Any]] = {
    "C1": {"metric": "intra_attr_cos", "operator": ">", "value": 0.95},
    "C2": {"metric": "eff_rank_95", "operator": "<", "value": 5},
    "C3": {"metric": "level_axis_alignment", "operator": ">", "value": 0.90},
    "C4": {"metric": "monotonic_ratio", "operator": "<", "value": 0.8},
    "C5": {"metric": "separation", "operator": ">=", "value": 0.0},
}

VARIANT_ORDER: Tuple[str, ...] = (
    "V0",
    "V1",
    "V2_k1",
    "V2_k2",
    "V3",
    "V4",
)


# Appendix A, copied verbatim and validated at import time (5 * 3 * 4 = 60).
SET_B_PROMPTS: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    "nuclear_density": {
        "low": (
            "scattered individual nuclei separated by abundant stroma",
            "sparse nuclei within a paucicellular fibrous background",
            "occasional nuclei dispersed across largely acellular tissue",
            "hypocellular region with widely spaced nuclei",
        ),
        "mid": (
            "moderately cellular tissue with evenly distributed nuclei",
            "intermediate cellularity, nuclei regularly spaced throughout",
            "a moderately populated field of nuclei",
            "tissue of average cellular density",
        ),
        "high": (
            "hypercellular area where nuclei form confluent sheets",
            "densely packed nuclei with minimal intervening stroma",
            "markedly increased cellularity, nuclei crowded together",
            "a densely nucleated region with sheet-like growth",
        ),
    },
    "nuclear_size_heterogeneity": {
        "low": (
            "a monomorphic population of nuclei of uniform size",
            "nuclei of consistent caliber throughout the field",
            "uniformly sized nuclei without appreciable variation",
            "homogeneous nuclear dimensions across the region",
        ),
        "mid": (
            "mild anisonucleosis with some variation in nuclear size",
            "moderate variability in nuclear caliber",
            "nuclei showing slight to moderate size differences",
            "somewhat heterogeneous nuclear sizes",
        ),
        "high": (
            "marked anisonucleosis, pleomorphic nuclei varying several-fold in size",
            "striking variation in nuclear size with bizarre enlarged forms",
            "severe nuclear pleomorphism and size disparity",
            "highly heterogeneous nuclei ranging from small to markedly enlarged",
        ),
    },
    "spatial_crowding": {
        "low": (
            "well separated nuclei with wide intervening cytoplasm",
            "nuclei set far apart with generous separation",
            "loosely arranged nuclei showing no contact",
            "widely spaced nuclei with clear intercellular gaps",
        ),
        "mid": (
            "nuclei in close proximity but not in contact",
            "moderately crowded nuclei with narrow separation",
            "nuclei approaching one another without overlap",
            "intermediate crowding, occasional nuclei touching",
        ),
        "high": (
            "overlapping nuclei with nuclear molding and indistinct borders",
            "severely crowded nuclei that abut and deform one another",
            "tightly apposed nuclei showing molding and overlap",
            "nuclei piled upon each other with obscured boundaries",
        ),
    },
    "nuclear_irregularity": {
        "low": (
            "smooth round contours with even nuclear membranes",
            "nuclei with regular unbroken outlines",
            "evenly contoured nuclear membranes without indentation",
            "smoothly circumscribed nuclear borders",
        ),
        "mid": (
            "slightly undulating nuclear membranes with mild contour irregularity",
            "nuclei showing minor membrane wrinkling",
            "mildly irregular nuclear outlines",
            "subtle contour variation along the nuclear membrane",
        ),
        "high": (
            "deeply indented and notched membranes with highly irregular contours",
            "markedly convoluted nuclear outlines with grooves and clefts",
            "severely irregular nuclear membranes showing prominent infolding",
            "angulated nuclei with jagged and interrupted borders",
        ),
    },
    "nuclear_elongation": {
        "low": (
            "round to oval nuclei with low aspect ratio",
            "predominantly circular nuclei",
            "nuclei that are essentially round in profile",
            "equidimensional nuclei without axial elongation",
        ),
        "mid": (
            "ovoid nuclei somewhat longer than wide",
            "mildly elongated nuclei of oval outline",
            "nuclei showing moderate axial elongation",
            "elliptical nuclei of intermediate aspect ratio",
        ),
        "high": (
            "markedly spindled and elongated nuclei",
            "cigar-shaped nuclei with high length-to-width ratio",
            "strikingly elongate fusiform nuclei",
            "slender attenuated nuclei arranged along their long axis",
        ),
    },
}


@dataclass(frozen=True)
class PromptBundle:
    """Ordered raw prompts and their 3-level prototype aggregation groups."""

    prompt_set: str
    display_name: str
    attribute_names: Tuple[str, ...]
    attribute_groups: Tuple[str, ...]
    raw_prompt_ids: Tuple[str, ...]
    raw_prompt_texts: Tuple[str, ...]
    prototype_raw_indices: Tuple[Tuple[int, ...], ...]
    ignored_source_keys: Tuple[str, ...] = ()

    @property
    def prototype_ids(self) -> Tuple[str, ...]:
        return tuple(
            f"{attribute}_{level}"
            for attribute in self.attribute_names
            for level in LEVEL_ORDER
        )

    @property
    def prototype_attribute_indices(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.attribute_names), dtype=np.int64), 3)

    def validate(self) -> None:
        expected_prototypes = len(self.attribute_names) * 3
        if len(self.attribute_groups) != len(self.attribute_names):
            raise ValueError("attribute_names and attribute_groups length mismatch")
        if len(self.prototype_raw_indices) != expected_prototypes:
            raise ValueError("prototype aggregation map must have exactly 3 groups per attribute")
        if len(self.raw_prompt_ids) != len(self.raw_prompt_texts):
            raise ValueError("raw prompt ids/text length mismatch")
        if len(set(self.raw_prompt_ids)) != len(self.raw_prompt_ids):
            raise ValueError("raw prompt ids are not unique")
        covered: List[int] = []
        for indices in self.prototype_raw_indices:
            if not indices:
                raise ValueError("empty prototype prompt group")
            covered.extend(indices)
        if sorted(covered) != list(range(len(self.raw_prompt_texts))):
            raise ValueError("prototype groups must partition raw prompts exactly once")
        if any(not isinstance(text, str) or not text.strip() for text in self.raw_prompt_texts):
            raise ValueError("all prompts must be non-empty strings; fallback is forbidden")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_and_sha(path: Path) -> Dict[str, str]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved) if resolved.is_file() else "NOT_FOUND",
    }


def callable_source(function: Any) -> Dict[str, Any]:
    """Return an exact, hash-bound source location for audit provenance."""

    source_path = Path(inspect.getsourcefile(function) or __file__).resolve()
    return {
        "path": str(source_path),
        "line": int(inspect.getsourcelines(function)[1]),
        "sha256": sha256_file(source_path),
        "callable": function.__name__,
    }


def project_encoding_references() -> Dict[str, Dict[str, Any]]:
    builder = (PROJECT_ROOT / "tools/build_l1a_text_prototype_bank.py").resolve()
    training = (PROJECT_ROOT / "segment_anything/modeling/sam.py").resolve()
    return {
        "offline_bank_builder": {
            **path_and_sha(builder),
            "lines": "53-73",
            "contract": "tokenizer max_length=77; encode_text().float(); F.normalize(eps=1e-8)",
        },
        "training_text_path": {
            **path_and_sha(training),
            "lines": "3115-3134,3226-3234",
            "contract": "tokenizer max_length=77; model.encode_text(...).float()",
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_local5_schema(schema: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    rows = schema.get("attributes")
    if not isinstance(rows, list):
        raise ValueError("schema.attributes must be a list")
    names = tuple(row.get("name") for row in rows if isinstance(row, Mapping))
    if names != LOCAL5_ATTRIBUTE_ORDER:
        raise ValueError(
            "Local-5 attribute order mismatch: "
            f"expected {list(LOCAL5_ATTRIBUTE_ORDER)}, got {list(names)}"
        )
    groups = tuple(row.get("group") for row in rows)
    if groups != LOCAL5_GROUP_ORDER:
        raise ValueError(
            "Local-5 group/order mismatch: "
            f"expected {list(LOCAL5_GROUP_ORDER)}, got {list(groups)}"
        )
    if tuple(schema.get("classes", ())) != SCHEMA_LEVEL_ORDER:
        raise ValueError(
            f"schema.classes must be exactly {list(SCHEMA_LEVEL_ORDER)}; "
            f"got {schema.get('classes')!r}"
        )
    for row in rows:
        if int(row.get("class_count", -1)) != 3:
            raise ValueError(f"{row['name']}: class_count must equal 3")
        if tuple(row.get("values", ())) != SCHEMA_LEVEL_ORDER:
            raise ValueError(
                f"{row['name']}: values must be exactly {list(SCHEMA_LEVEL_ORDER)}"
            )
    irregularity = rows[3]
    if irregularity.get("label_source_name") != "boundary_irregularity":
        raise ValueError(
            "nuclear_irregularity must retain source label name boundary_irregularity"
        )
    return tuple(rows)


def load_set_a(schema_path: Path) -> PromptBundle:
    if not schema_path.is_file():
        raise FileNotFoundError(f"Set-A schema NOT_FOUND: {schema_path.resolve()}")
    rows = _validate_local5_schema(read_json(schema_path))
    ids: List[str] = []
    texts: List[str] = []
    aggregation: List[Tuple[int, ...]] = []
    for row in rows:
        prompts = row.get("prompt_texts")
        if not isinstance(prompts, list) or len(prompts) != 3:
            raise ValueError(f"{row['name']}: prompt_texts must contain exactly 3 strings")
        for level, text in zip(LEVEL_ORDER, prompts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{row['name']}/{level}: missing prompt; no fallback allowed")
            ids.append(f"{row['name']}__{level}")
            texts.append(text)
            aggregation.append((len(texts) - 1,))
    bundle = PromptBundle(
        prompt_set="A",
        display_name="Set-A",
        attribute_names=LOCAL5_ATTRIBUTE_ORDER,
        attribute_groups=LOCAL5_GROUP_ORDER,
        raw_prompt_ids=tuple(ids),
        raw_prompt_texts=tuple(texts),
        prototype_raw_indices=tuple(aggregation),
    )
    bundle.validate()
    return bundle


def load_set_b(schema_path: Path) -> PromptBundle:
    if not schema_path.is_file():
        raise FileNotFoundError(f"Set-B order schema NOT_FOUND: {schema_path.resolve()}")
    _validate_local5_schema(read_json(schema_path))
    if tuple(SET_B_PROMPTS) != LOCAL5_ATTRIBUTE_ORDER:
        raise AssertionError("Appendix-A Set-B attribute order was modified")
    ids: List[str] = []
    texts: List[str] = []
    aggregation: List[Tuple[int, ...]] = []
    for attribute in LOCAL5_ATTRIBUTE_ORDER:
        level_map = SET_B_PROMPTS[attribute]
        if tuple(level_map) != LEVEL_ORDER:
            raise AssertionError(f"Set-B level order mismatch for {attribute}")
        for level in LEVEL_ORDER:
            four = level_map[level]
            if len(four) != 4:
                raise AssertionError(f"Set-B {attribute}/{level} must contain 4 prompts")
            indices: List[int] = []
            for template_index, text in enumerate(four, start=1):
                if not text.strip():
                    raise AssertionError(f"Set-B {attribute}/{level} contains empty prompt")
                ids.append(f"{attribute}__{level}__template_{template_index}")
                texts.append(text)
                indices.append(len(texts) - 1)
            aggregation.append(tuple(indices))
    bundle = PromptBundle(
        prompt_set="B",
        display_name="Set-B",
        attribute_names=LOCAL5_ATTRIBUTE_ORDER,
        attribute_groups=LOCAL5_GROUP_ORDER,
        raw_prompt_ids=tuple(ids),
        raw_prompt_texts=tuple(texts),
        prototype_raw_indices=tuple(aggregation),
    )
    bundle.validate()
    if len(bundle.raw_prompt_texts) != 60:
        raise AssertionError("Set-B must contain exactly 60 raw prompts")
    return bundle


def _strict_prompt_group(
    payload: Mapping[str, Any],
    group_key: str,
    expected_attributes: Sequence[str],
    allowed_extra_attributes: Sequence[str] = (),
) -> Tuple[List[str], List[str], List[Tuple[int, ...]], List[str]]:
    group = payload.get(group_key)
    if not isinstance(group, Mapping):
        raise ValueError(f"global27 missing mapping {group_key!r}; no fallback allowed")
    missing_attributes = [name for name in expected_attributes if name not in group]
    extra_attributes = [name for name in group if name not in expected_attributes]
    unexpected_attributes = [
        name for name in extra_attributes if name not in allowed_extra_attributes
    ]
    if missing_attributes or unexpected_attributes:
        raise ValueError(
            f"global27 {group_key} fixed attribute contract mismatch: "
            f"missing={missing_attributes}, unexpected={unexpected_attributes}; "
            "no fallback allowed"
        )
    ids: List[str] = []
    texts: List[str] = []
    aggregation: List[Tuple[int, ...]] = []
    ignored = [f"{group_key}.{name}" for name in extra_attributes]
    for attribute in expected_attributes:
        levels = group[attribute]
        if not isinstance(levels, Mapping):
            raise ValueError(
                f"global27 {attribute} must be a mapping; no fallback allowed"
            )
        missing_levels = [level for level in LEVEL_ORDER if level not in levels]
        extra_level_keys = [key for key in levels if key not in LEVEL_ORDER]
        unexpected_level_keys = [key for key in extra_level_keys if key != "description"]
        if missing_levels or unexpected_level_keys:
            raise ValueError(
                f"global27 {attribute} level contract mismatch: "
                f"missing={missing_levels}, unexpected={unexpected_level_keys}; "
                "no fallback allowed"
            )
        ignored.extend(
            f"{group_key}.{attribute}.{key}" for key in extra_level_keys
        )
        for level in LEVEL_ORDER:
            value = levels[level]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"global27 missing {attribute}/{level}; no fallback allowed"
                )
            ids.append(f"{attribute}__{level}")
            texts.append(value)
            aggregation.append((len(texts) - 1,))
    return ids, texts, aggregation, ignored


def load_global27(global_path: Path) -> PromptBundle:
    if not global_path.is_file():
        raise FileNotFoundError(f"global27 template NOT_FOUND: {global_path.resolve()}")
    payload = read_json(global_path)
    if not isinstance(payload, Mapping):
        raise ValueError("global27 template root must be a JSON object")
    s_ids, s_texts, s_groups, s_ignored = _strict_prompt_group(
        payload, "structure_prompts", GLOBAL27_STRUCTURE_ORDER
    )
    b_ids, b_texts, b_groups_local, b_ignored = _strict_prompt_group(
        payload,
        "boundary_prompts",
        GLOBAL27_BOUNDARY_ORDER,
        allowed_extra_attributes=("touching_or_crowding_difficulty",),
    )
    offset = len(s_texts)
    b_groups = [tuple(index + offset for index in group) for group in b_groups_local]
    attributes = GLOBAL27_STRUCTURE_ORDER + GLOBAL27_BOUNDARY_ORDER
    groups = ("structure",) * len(GLOBAL27_STRUCTURE_ORDER) + (
        "boundary",
    ) * len(GLOBAL27_BOUNDARY_ORDER)
    bundle = PromptBundle(
        prompt_set="global27",
        display_name="global27",
        attribute_names=attributes,
        attribute_groups=groups,
        raw_prompt_ids=tuple(s_ids + b_ids),
        raw_prompt_texts=tuple(s_texts + b_texts),
        prototype_raw_indices=tuple(s_groups + b_groups),
        ignored_source_keys=tuple(s_ignored + b_ignored),
    )
    bundle.validate()
    if len(bundle.raw_prompt_texts) != 27:
        raise AssertionError("global27 strict loader must produce exactly 27 prompts")
    return bundle


def load_prompt_bundle(
    prompt_set: str,
    schema_path: Path,
    global27_path: Path,
) -> PromptBundle:
    if prompt_set == "A":
        return load_set_a(schema_path)
    if prompt_set == "B":
        return load_set_b(schema_path)
    if prompt_set == "global27":
        return load_global27(global27_path)
    raise ValueError(f"unsupported prompt set: {prompt_set}")


def l2_normalize(array: np.ndarray, *, reject_zero: bool = False) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("embedding matrix contains NaN or infinity")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if reject_zero and bool(np.any(norms <= 1e-12)):
        bad = np.flatnonzero(norms[:, 0] <= 1e-12).tolist()
        raise ValueError(f"zero-norm raw embeddings at rows {bad}")
    return np.divide(value, norms, out=np.zeros_like(value), where=norms > 1e-12)


def aggregate_raw_embeddings(
    bundle: PromptBundle,
    raw_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_embeddings, dtype=np.float64)
    expected = len(bundle.raw_prompt_texts)
    if raw.ndim != 2 or raw.shape[0] != expected:
        raise ValueError(
            f"raw embedding shape mismatch: expected [{expected}, D], got {raw.shape}"
        )
    raw_normalized = l2_normalize(raw, reject_zero=True)
    prototypes = np.stack(
        [raw_normalized[list(indices)].mean(axis=0) for indices in bundle.prototype_raw_indices],
        axis=0,
    )
    prototypes = l2_normalize(prototypes, reject_zero=True)
    expected_shape = (len(bundle.attribute_names) * 3, raw.shape[1])
    if prototypes.shape != expected_shape:
        raise AssertionError(
            f"prototype shape mismatch: expected {expected_shape}, got {prototypes.shape}"
        )
    return raw_normalized, prototypes


def transform_variant(
    prototypes: np.ndarray,
    num_attributes: int,
    variant: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    base = l2_normalize(prototypes, reject_zero=True)
    extras: Dict[str, np.ndarray] = {}
    if variant == "V0":
        transformed = base.copy()
    elif variant == "V1":
        mean = base.mean(axis=0, keepdims=True)
        extras["global_mean"] = mean
        extras["centered_before_normalization"] = base - mean
        transformed = l2_normalize(base - mean, reject_zero=True)
    elif variant in ("V2_k1", "V2_k2"):
        k = 1 if variant == "V2_k1" else 2
        mean = base.mean(axis=0, keepdims=True)
        centered = base - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal_components = vt[:k]
        projected = (centered @ principal_components.T) @ principal_components
        residual = centered - projected
        extras["global_mean"] = mean
        extras["centered_before_projection"] = centered
        extras["principal_components"] = principal_components
        extras["removed_projection"] = projected
        transformed = l2_normalize(residual, reject_zero=True)
    elif variant == "V3":
        shaped = base.reshape(num_attributes, 3, base.shape[1])
        means = shaped.mean(axis=1, keepdims=True)
        extras["attribute_means"] = means[:, 0, :]
        extras["centered_before_normalization"] = (shaped - means).reshape(base.shape)
        transformed = l2_normalize(
            (shaped - means).reshape(base.shape), reject_zero=True
        )
    elif variant == "V4":
        shaped = base.reshape(num_attributes, 3, base.shape[1])
        axes = l2_normalize(
            shaped[:, 2, :] - shaped[:, 0, :], reject_zero=True
        )
        alpha = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
        transformed = (axes[:, None, :] * alpha[None, :, None]).reshape(base.shape)
        mid = shaped[:, 1, :]
        mid_norm_sq = np.sum(mid * mid, axis=1)
        projection_length = np.sum(mid * axes, axis=1)
        residual_ratio = 1.0 - np.divide(
            projection_length * projection_length,
            mid_norm_sq,
            out=np.zeros_like(mid_norm_sq),
            where=mid_norm_sq > 1e-12,
        )
        extras["attribute_axes_from_original_space"] = axes
        extras["v4_alpha"] = alpha
        extras["v4_mid_residual_ratio"] = np.clip(residual_ratio, 0.0, 1.0)
    else:
        raise ValueError(f"unknown variant: {variant}")
    if not np.isfinite(transformed).all():
        raise ValueError(f"{variant} produced non-finite embeddings")
    return transformed, extras


def _effective_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = np.square(singular_values)
    total = float(energy.sum())
    if total <= 1e-24:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _four_significant(values: Iterable[float]) -> List[str]:
    return [format(float(value), ".4g") for value in values]


def evaluate_criteria(flat_metrics: Mapping[str, Any]) -> Dict[str, bool]:
    """Evaluate only the immutable pre-registered inequalities."""

    return {
        "C1": float(flat_metrics["intra_attr_cos"]) > 0.95,
        "C2": int(flat_metrics["eff_rank_95"]) < 5,
        "C3": float(flat_metrics["level_axis_alignment"]) > 0.90,
        "C4": float(flat_metrics["monotonic_ratio"]) < 0.8,
        "C5": float(flat_metrics["separation"]) >= 0.0,
    }


def compute_metrics(
    embeddings: np.ndarray,
    bundle: PromptBundle,
    variant: str,
    extras: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    expected_rows = len(bundle.attribute_names) * 3
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            f"expected [{expected_rows}, D] variant matrix, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("variant matrix contains NaN or infinity")

    # V4 intentionally contains zero mid rows.  Safe normalization preserves
    # those rows and therefore defines every cosine involving the origin as 0.
    cosine_ready = l2_normalize(matrix)
    cosine = cosine_ready @ cosine_ready.T
    attr_index = bundle.prototype_attribute_indices
    row, col = np.triu_indices(expected_rows, k=1)
    same_attr = attr_index[row] == attr_index[col]
    intra_values = cosine[row[same_attr], col[same_attr]]
    inter_values = cosine[row[~same_attr], col[~same_attr]]
    intra = float(intra_values.mean())
    inter = float(inter_values.mean())

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    singular_energy = np.square(singular_values)
    total_energy = float(singular_energy.sum())
    s1_energy_ratio = (
        float(singular_energy[0] / total_energy) if total_energy > 1e-24 else 0.0
    )

    shaped = matrix.reshape(len(bundle.attribute_names), 3, matrix.shape[1])
    level_axes = l2_normalize(shaped[:, 2, :] - shaped[:, 0, :])
    axis_cosine = level_axes @ level_axes.T
    axis_row, axis_col = np.triu_indices(len(bundle.attribute_names), k=1)
    level_axis_alignment = float(axis_cosine[axis_row, axis_col].mean())

    monotonic_t: Dict[str, float] = {}
    for attribute_index, attribute in enumerate(bundle.attribute_names):
        low, mid, high = shaped[attribute_index]
        direction = high - low
        denominator = float(np.dot(direction, direction)) + 1e-12
        monotonic_t[attribute] = float(np.dot(mid - low, direction) / denominator)
    monotonic_ratio = float(
        sum(0.0 < value < 1.0 for value in monotonic_t.values())
        / len(monotonic_t)
    )

    flat = {
        "intra_attr_cos": intra,
        "inter_attr_cos": inter,
        "separation": inter - intra,
        "eff_rank_95": _effective_rank(singular_values, 0.95),
        "eff_rank_90": _effective_rank(singular_values, 0.90),
        "s1_energy_ratio": s1_energy_ratio,
        "level_axis_alignment": level_axis_alignment,
        "monotonic_ratio": monotonic_ratio,
    }
    criteria = evaluate_criteria(flat)
    payload: Dict[str, Any] = {
        "prompt_set": bundle.display_name,
        "variant": variant,
        "embedding_count": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "D1": {
            "intra_attr_cos": intra,
            "inter_attr_cos": inter,
            "separation": inter - intra,
            "intra_pair_count": int(intra_values.size),
            "inter_pair_count": int(inter_values.size),
        },
        "D2": {
            "singular_values": [float(value) for value in singular_values],
            "singular_values_4_significant_digits": _four_significant(singular_values),
            "eff_rank_95": flat["eff_rank_95"],
            "eff_rank_90": flat["eff_rank_90"],
            "s1_energy_ratio": s1_energy_ratio,
        },
        "D3": {
            "level_axis_alignment": level_axis_alignment,
            "attribute_order": list(bundle.attribute_names),
            "axis_zero_norm_count": int(
                np.sum(np.linalg.norm(level_axes, axis=1) <= 1e-12)
            ),
        },
        "D4": {
            "t_by_attribute": monotonic_t,
            "monotonic_ratio": monotonic_ratio,
        },
        "flat_metrics": flat,
        "criteria": criteria,
        "passes_C1_through_C5": not any(criteria.values()),
        "zero_vector_count": int(np.sum(np.linalg.norm(matrix, axis=1) <= 1e-12)),
    }
    if variant == "V4":
        if extras is None or "v4_mid_residual_ratio" not in extras:
            raise ValueError("V4 metrics require v4_mid_residual_ratio")
        residual = np.asarray(extras["v4_mid_residual_ratio"], dtype=np.float64)
        residual_by_attribute = {
            attribute: float(residual[index])
            for index, attribute in enumerate(bundle.attribute_names)
        }
        payload["V4_extra"] = {
            "mid_residual_ratio_by_attribute": residual_by_attribute,
            "max_mid_residual_ratio": float(residual.max()),
            "attributes_over_50_percent": [
                attribute
                for attribute, value in residual_by_attribute.items()
                if value > 0.5
            ],
            "suitable_for_adoption": bool(np.all(residual <= 0.5)),
        }
    arrays = {
        "cosine_matrix": cosine,
        "singular_values": singular_values,
        "level_axes": level_axes,
        "level_axis_cosine_matrix": axis_cosine,
    }
    return payload, arrays


def write_matrix_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prompt_id", *labels])
            for label, row in zip(labels, matrix):
                writer.writerow([label, *(format(float(value), ".10g") for value in row)])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _heat_color(value: float) -> str:
    """Diverging blue/white/red map on the fixed cosine range [-1, 1]."""

    clipped = max(-1.0, min(1.0, float(value)))
    blue = (49, 54, 149)
    white = (255, 255, 255)
    red = (165, 0, 38)
    if clipped < 0.0:
        fraction = clipped + 1.0
        rgb = tuple(round(blue[i] + fraction * (white[i] - blue[i])) for i in range(3))
    else:
        fraction = clipped
        rgb = tuple(round(white[i] + fraction * (red[i] - white[i])) for i in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def write_heatmap_svg(
    path: Path,
    labels: Sequence[str],
    matrix: np.ndarray,
    title: str,
) -> None:
    """Write a dependency-free, deterministic SVG heatmap."""

    count = len(labels)
    cell = 27 if count <= 15 else 21
    left = 300
    top = 300
    legend = 90
    width = left + count * cell + legend
    height = top + count * cell + 40
    escaped_title = html.escape(title)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="16" y="28" font-family="sans-serif" font-size="18">{escaped_title}</text>',
    ]
    for index, label in enumerate(labels):
        escaped = html.escape(str(label))
        y = top + index * cell + cell * 0.7
        x = left + index * cell + cell * 0.5
        lines.append(
            f'<text x="{left - 8}" y="{y:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="10">{escaped}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top - 8}" text-anchor="start" '
            f'transform="rotate(-55 {x:.1f} {top - 8})" '
            f'font-family="monospace" font-size="10">{escaped}</text>'
        )
    show_values = count <= 15
    for row_index in range(count):
        for column_index in range(count):
            value = float(matrix[row_index, column_index])
            x = left + column_index * cell
            y = top + row_index * cell
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_heat_color(value)}" stroke="#dddddd" stroke-width="0.4"/>'
            )
            if show_values:
                foreground = "white" if abs(value) > 0.65 else "black"
                lines.append(
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell * 0.65:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="7" '
                    f'fill="{foreground}">{value:.2f}</text>'
                )
    legend_x = left + count * cell + 25
    for step in range(101):
        value = 1.0 - 2.0 * step / 100.0
        lines.append(
            f'<rect x="{legend_x}" y="{top + step * count * cell / 101:.2f}" '
            f'width="18" height="{count * cell / 101 + 0.5:.2f}" '
            f'fill="{_heat_color(value)}" stroke="none"/>'
        )
    lines.extend(
        [
            f'<text x="{legend_x + 24}" y="{top + 7}" font-size="10">+1</text>',
            f'<text x="{legend_x + 24}" y="{top + count * cell / 2 + 4:.1f}" font-size="10">0</text>',
            f'<text x="{legend_x + 24}" y="{top + count * cell:.1f}" font-size="10">-1</text>',
            "</svg>",
        ]
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def _npz_scalar_string(container: Mapping[str, np.ndarray], key: str) -> Optional[str]:
    if key not in container:
        return None
    value = np.asarray(container[key])
    if value.size != 1:
        raise ValueError(f"pre-encoded metadata {key!r} must be scalar")
    return str(value.reshape(-1)[0])


def load_preencoded_embeddings(
    path: Path,
    bundle: PromptBundle,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"embeddings input NOT_FOUND: {path.resolve()}")
    if path.suffix.lower() != ".npz":
        raise ValueError("--embeddings-input must be a .npz file")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "prompt_embeddings",
            "prompt_ids",
            "prompt_texts",
            "literal_embeddings",
            "literal_texts",
            "checkpoint_path",
            "checkpoint_sha256",
            "encoding_function_source",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"pre-encoded npz missing required keys: {missing}")
        ids = tuple(str(value) for value in np.asarray(data["prompt_ids"]).tolist())
        texts = tuple(str(value) for value in np.asarray(data["prompt_texts"]).tolist())
        literal_texts = tuple(
            str(value) for value in np.asarray(data["literal_texts"]).tolist()
        )
        if ids != bundle.raw_prompt_ids:
            raise ValueError("pre-encoded prompt_ids do not exactly match requested prompt set/order")
        if texts != bundle.raw_prompt_texts:
            raise ValueError("pre-encoded prompt_texts do not exactly match requested prompt set/order")
        if literal_texts != LITERAL_PROMPTS:
            raise ValueError("pre-encoded literal_texts do not match the two registered literals")
        prompt_embeddings = np.asarray(data["prompt_embeddings"], dtype=np.float64)
        literal_embeddings = np.asarray(data["literal_embeddings"], dtype=np.float64)
        metadata = {
            key: _npz_scalar_string(data, key)
            for key in (
                "checkpoint_path",
                "checkpoint_sha256",
                "encoding_function_source",
            )
        }
        if any(value is None or not value.strip() for value in metadata.values()):
            raise ValueError("pre-encoded provenance fields must be non-empty scalars")
    if prompt_embeddings.ndim != 2 or prompt_embeddings.shape[0] != len(texts):
        raise ValueError("pre-encoded prompt_embeddings has invalid shape")
    if literal_embeddings.shape != (2, prompt_embeddings.shape[1]):
        raise ValueError(
            "pre-encoded literal_embeddings must have shape [2, D] matching prompt D"
        )
    if not np.isfinite(prompt_embeddings).all() or not np.isfinite(literal_embeddings).all():
        raise ValueError("pre-encoded embeddings contain NaN or infinity")
    return prompt_embeddings, literal_embeddings, metadata


def _load_conch_low_memory_mmap(
    torch_module: Any,
    checkpoint_path: Path,
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the identical CoCa weights without a checkpoint/model double copy.

    The official factory materializes a float32 model and then a second float32
    state dict, which exceeds the server's 2 GiB cgroup in NO_GPU_MODE.  This
    path changes only storage construction: parameters are created on ``meta``
    and assigned directly from a read-only mmap of the same checkpoint.
    """

    if str(device) != "cpu":
        raise ValueError("--low-memory-mmap is registered only for CPU execution")
    from conch.open_clip_custom.coca_model import CoCa, resize_pos_embed
    from conch.open_clip_custom.factory import CFG_DIR

    config_path = Path(CFG_DIR) / f"{CONCH_MODEL_NAME}.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config.pop("custom_text", None)
    with torch_module.device("meta"):
        model = CoCa(**model_config)

    state_dict = torch_module.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("CONCH checkpoint did not contain a non-empty state dict")
    first_key = next(iter(state_dict))
    if str(first_key).startswith("module"):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    resize_pos_embed(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    del state_dict
    remaining_meta = [name for name, value in model.state_dict().items() if value.is_meta]
    disallowed_meta = [
        name for name in remaining_meta if not name.startswith("text_decoder.")
    ]
    if disallowed_meta:
        raise RuntimeError(
            "low-memory CONCH load left required meta tensors: "
            + ", ".join(disallowed_meta[:10])
        )
    # The released CONCH checkpoint intentionally omits all caption decoder
    # weights (the official factory also ignores them with strict=False).
    # encode_text never reads this module, so discard only that known-unused
    # meta subtree while retaining and validating every text-tower tensor.
    if remaining_meta:
        if any(not name.startswith("text_decoder.") for name in missing):
            raise RuntimeError("checkpoint is missing non-decoder CONCH parameters")
        model.text_decoder = None
    rebuilt_nonpersistent_buffers: List[str] = []
    # TextTransformer.attn_mask is registered with persistent=False, so it is
    # intentionally absent from both the released checkpoint and state_dict().
    # Meta construction therefore leaves it on meta; rebuild the same causal
    # mask on CPU before encode_text.
    if model.text.attn_mask.is_meta:
        model.text.attn_mask = model.text.build_attention_mask()
        rebuilt_nonpersistent_buffers.append("text.attn_mask")
    remaining_text_meta_buffers = [
        name for name, value in model.text.named_buffers() if value.is_meta
    ]
    if remaining_text_meta_buffers:
        raise RuntimeError(
            "low-memory CONCH load left required text buffers on meta: "
            + ", ".join(remaining_text_meta_buffers[:10])
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "mode": "meta_init_plus_read_only_mmap_assign",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "rebuilt_nonpersistent_buffers": rebuilt_nonpersistent_buffers,
    }


def encode_with_project_conch_path(
    texts: Sequence[str],
    checkpoint_path: Optional[Path],
    cache_path: Optional[Path],
    device: str,
    hf_hub_offline: bool,
    low_memory_mmap: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inference-only encoding using the exact audited project CONCH contract."""

    try:
        import torch
        from torch.nn import functional as F
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    except ImportError as exc:  # pragma: no cover - depends on server environment
        raise RuntimeError(
            "CONCH encoding requires torch and conch; alternatively provide "
            "--embeddings-input from a server encoding run"
        ) from exc

    if hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if cache_path is not None:
        cache_resolved = cache_path.expanduser().resolve()
        os.environ["HF_HOME"] = str(cache_resolved)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_resolved)

    if checkpoint_path is not None:
        resolved_checkpoint = checkpoint_path.expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"CONCH checkpoint NOT_FOUND: {resolved_checkpoint}")
        source = str(resolved_checkpoint)
        checkpoint_sha = sha256_file(resolved_checkpoint)
    elif DEFAULT_SERVER_CONCH_CHECKPOINT.is_file():
        resolved_checkpoint = DEFAULT_SERVER_CONCH_CHECKPOINT.resolve()
        source = str(resolved_checkpoint)
        checkpoint_sha = sha256_file(resolved_checkpoint)
    else:
        resolved_checkpoint = None
        source = CONCH_HF_SOURCE
        checkpoint_sha = "NOT_FOUND"

    if low_memory_mmap:
        if resolved_checkpoint is None:
            raise ValueError("--low-memory-mmap requires a local checkpoint file")
        model, load_audit = _load_conch_low_memory_mmap(
            torch, resolved_checkpoint, device
        )
    else:
        kwargs: Dict[str, Any] = {"device": device}
        if cache_path is not None:
            kwargs["cache_dir"] = str(cache_path.expanduser().resolve())
        if source == CONCH_HF_SOURCE:
            kwargs["hf_auth_token"] = os.environ.get("HF_TOKEN")
        model, _ = create_model_from_pretrained(CONCH_MODEL_NAME, source, **kwargs)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        load_audit = {
            "mode": "project_create_model_from_pretrained",
            "missing_keys": "NOT_REPORTED_BY_FACTORY",
            "unexpected_keys": "NOT_REPORTED_BY_FACTORY",
        }
    tokenizer = get_tokenizer()
    tokenized = tokenizer(
        list(texts),
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokens = tokenized.input_ids
    elif isinstance(tokenized, Mapping) and "input_ids" in tokenized:
        tokens = tokenized["input_ids"]
    else:
        tokens = tokenized
    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens)
    tokens = tokens.to(device)
    with torch.inference_mode():
        embeddings = model.encode_text(tokens).float()
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
    result = embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
    metadata = {
        "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint else source,
        "checkpoint_sha256": checkpoint_sha,
        "encoding_function_source": (
            f"{callable_source(encode_with_project_conch_path)['path']}:"
            f"{callable_source(encode_with_project_conch_path)['line']}"
        ),
        "encoding_function": callable_source(encode_with_project_conch_path),
        "project_encoding_references": project_encoding_references(),
        "normalization_contract": "torch.float32 F.normalize(dim=-1, eps=1e-8)",
        "model_load_audit": load_audit,
    }
    return result, metadata


def write_encoding_request(path: Path, bundle: PromptBundle, schema_path: Path, global_path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite encoding request: {path}")
    write_json(
        path,
        {
            "schema_version": "conch_separability_encoding_request_v1",
            "prompt_set": bundle.display_name,
            "prompt_ids": list(bundle.raw_prompt_ids),
            "prompt_texts": list(bundle.raw_prompt_texts),
            "literal_texts": list(LITERAL_PROMPTS),
            "expected_npz_keys": [
                "prompt_embeddings",
                "prompt_ids",
                "prompt_texts",
                "literal_embeddings",
                "literal_texts",
                "checkpoint_path",
                "checkpoint_sha256",
                "encoding_function_source",
            ],
            "schema": path_and_sha(schema_path),
            "global27_templates": path_and_sha(global_path),
            "encoding_function_source": (
                f"{callable_source(encode_with_project_conch_path)['path']}:"
                f"{callable_source(encode_with_project_conch_path)['line']}"
            ),
            "encoding_function": callable_source(encode_with_project_conch_path),
            "project_encoding_references": project_encoding_references(),
            "training_encoding_function_source": TRAINING_ENCODER_SOURCE,
            "tokenizer_contract": {
                "padding": "max_length",
                "max_length": 77,
                "truncation": True,
                "return_tensors": "pt",
            },
        },
    )


def selected_variants(argument: str) -> Tuple[str, ...]:
    if argument == "all":
        return VARIANT_ORDER
    if argument == "V2":
        return ("V2_k1", "V2_k2")
    if argument in VARIANT_ORDER:
        return (argument,)
    raise ValueError(f"unsupported variant selector: {argument}")


def select_best_variant(metrics_by_variant: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    eligible: List[str] = []
    for variant, metrics in metrics_by_variant.items():
        if any(bool(value) for value in metrics["criteria"].values()):
            continue
        if variant == "V4" and not metrics.get("V4_extra", {}).get(
            "suitable_for_adoption", False
        ):
            continue
        eligible.append(variant)
    if not eligible:
        return None
    order_index = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    return min(
        eligible,
        key=lambda variant: (
            -int(metrics_by_variant[variant]["flat_metrics"]["eff_rank_95"]),
            float(metrics_by_variant[variant]["flat_metrics"]["separation"]),
            order_index[variant],
        ),
    )


def classify_set_a_v0(metrics_by_variant: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply the registered failure hierarchy without inventing a new threshold."""

    if "V0" not in metrics_by_variant:
        return {"classification": "NOT_EVALUATED_WITHOUT_V0", "basis": {}}
    base = metrics_by_variant["V0"]
    flags = base["criteria"]
    if not any(flags.values()):
        return {"classification": "PASS", "basis": {"V0_criteria": flags}}
    v2 = [
        metrics_by_variant[name]
        for name in ("V2_k1", "V2_k2")
        if name in metrics_by_variant
    ]
    basis: Dict[str, Any] = {
        "V0_criteria": flags,
        "V2_variants_available": [item["variant"] for item in v2],
    }
    if not v2:
        return {"classification": "NEEDS_V2_FOR_REGISTERED_CLASSIFICATION", "basis": basis}

    # F1 > F3 > F2 > F4, exactly as pre-registered.  For statements about
    # "V2", both registered k values must retain the defect; a single repaired
    # k is sufficient to establish that a repair exists.
    if flags["C2"] and all(
        int(item["flat_metrics"]["eff_rank_95"]) < 3 for item in v2
    ):
        classification = "F1"
    elif flags["C3"]:
        classification = "F3"
    elif flags["C1"] and all(item["criteria"]["C1"] for item in v2):
        classification = "F2"
    elif flags["C1"] and any(
        not item["criteria"]["C1"]
        and not item["criteria"]["C2"]
        and float(item["flat_metrics"]["monotonic_ratio"]) >= 0.8
        for item in v2
    ):
        classification = "F4"
    else:
        classification = "UNCLASSIFIED_REGISTERED_FAILURE_COMBINATION"
    basis["V2"] = {
        item["variant"]: {
            "criteria": item["criteria"],
            "eff_rank_95": item["flat_metrics"]["eff_rank_95"],
            "monotonic_ratio": item["flat_metrics"]["monotonic_ratio"],
        }
        for item in v2
    }
    return {"classification": classification, "basis": basis}


def _prepare_new_directory(path: Path, description: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"{description} path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty {description} directory: {path}"
            )
    else:
        path.mkdir(parents=True)


def _flat_freeze_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(metrics["flat_metrics"])
    result["criteria"] = dict(metrics["criteria"])
    if "V4_extra" in metrics:
        result["V4_extra"] = metrics["V4_extra"]
    return result


def freeze_selected_bank(
    freeze_dir: Path,
    bundle: PromptBundle,
    variant: str,
    embeddings: np.ndarray,
    metrics: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if bundle.prompt_set not in ("A", "B"):
        raise ValueError("only Set-A or Set-B can be frozen as an RSGR Local-5 bank")
    if tuple(bundle.attribute_names) != LOCAL5_ATTRIBUTE_ORDER:
        raise AssertionError("freeze attribute order differs from Local-5 schema")
    if tuple(bundle.attribute_groups) != LOCAL5_GROUP_ORDER:
        raise AssertionError("freeze group order differs from Local-5 schema")
    if tuple(LEVEL_ORDER) != ("low", "mid", "high"):
        raise AssertionError("freeze level order was modified")
    if any(metrics["criteria"].values()):
        raise ValueError(f"refusing to freeze {variant}: it triggers C1-C5")
    if variant == "V4" and not metrics.get("V4_extra", {}).get(
        "suitable_for_adoption", False
    ):
        raise ValueError("refusing to freeze V4: a mid-axis residual exceeds 50%")
    if checkpoint_metadata.get("checkpoint_sha256") in (None, "", "NOT_FOUND"):
        raise ValueError("refusing to freeze without an exact CONCH checkpoint SHA256")
    if checkpoint_metadata.get("encoding_function_source") in (None, "", "NOT_FOUND"):
        raise ValueError("refusing to freeze without encoding-function provenance")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local machine lacks torch
        raise RuntimeError("freezing .pt banks requires torch") from exc

    _prepare_new_directory(freeze_dir, "freeze")
    shaped = np.asarray(embeddings, dtype=np.float32).reshape(5, 3, -1)
    structure = torch.from_numpy(shaped[:3].copy())
    boundary = torch.from_numpy(shaped[3:].copy())
    if tuple(structure.shape[:2]) != (3, 3) or tuple(boundary.shape[:2]) != (2, 3):
        raise AssertionError("frozen bank shapes do not match [3,3,D] and [2,3,D]")

    prompts_path = freeze_dir / "prompts_frozen.json"
    structure_path = freeze_dir / "structure_bank.pt"
    boundary_path = freeze_dir / "boundary_bank.pt"
    manifest_path = freeze_dir / "bank_manifest.json"
    write_json(
        prompts_path,
        {
            "prompt_set": bundle.display_name,
            "attribute_order": list(bundle.attribute_names),
            "level_order": list(SCHEMA_LEVEL_ORDER),
            "diagnostic_level_aliases": list(LEVEL_ORDER),
            "raw_prompt_ids": list(bundle.raw_prompt_ids),
            "raw_prompt_texts": list(bundle.raw_prompt_texts),
            "prototype_raw_indices": [list(indices) for indices in bundle.prototype_raw_indices],
            "ignored_source_keys": list(bundle.ignored_source_keys),
        },
    )
    torch.save(structure, structure_path)
    torch.save(boundary, boundary_path)
    manifest = {
        "prompt_set": bundle.display_name,
        "geometric_variant": variant,
        "conch_checkpoint_path": checkpoint_metadata["checkpoint_path"],
        "conch_checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "encoding_function_source": checkpoint_metadata["encoding_function_source"],
        "embeddings_input": checkpoint_metadata.get("embeddings_input", "DIRECT_INFERENCE"),
        "prompts_sha256": sha256_file(prompts_path),
        "structure_bank_sha256": sha256_file(structure_path),
        "boundary_bank_sha256": sha256_file(boundary_path),
        "embedding_dim": int(shaped.shape[-1]),
        "attribute_order": list(bundle.attribute_names),
        "level_order": list(SCHEMA_LEVEL_ORDER),
        "diagnostic_level_aliases": list(LEVEL_ORDER),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_at_freeze": _flat_freeze_metrics(metrics),
    }
    write_json(manifest_path, manifest)
    return manifest


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    schema_path = Path(args.schema).expanduser().resolve()
    global_path = Path(args.global27_templates).expanduser().resolve()
    bundle = load_prompt_bundle(args.prompt_set, schema_path, global_path)

    if args.write_encoding_request:
        request_path = Path(args.write_encoding_request).expanduser().resolve()
        write_encoding_request(request_path, bundle, schema_path, global_path)
        result = {"encoding_request": str(request_path), "sha256": sha256_file(request_path)}
        print("[PROBE_ENCODING_REQUEST] " + json.dumps(result, ensure_ascii=False))
        return result

    if args.embeddings_input:
        embeddings_input = Path(args.embeddings_input).expanduser().resolve()
        raw_embeddings, literal_embeddings, checkpoint_metadata = load_preencoded_embeddings(
            embeddings_input, bundle
        )
        embedding_source = {
            "mode": "preencoded_npz",
            "path": str(embeddings_input),
            "sha256": sha256_file(embeddings_input),
        }
        checkpoint_metadata["embeddings_input"] = dict(embedding_source)
        if args.conch_checkpoint_path:
            supplied = Path(args.conch_checkpoint_path).expanduser().resolve()
            if not supplied.is_file():
                raise FileNotFoundError(f"CONCH checkpoint NOT_FOUND: {supplied}")
            supplied_sha = sha256_file(supplied)
            recorded_sha = checkpoint_metadata["checkpoint_sha256"]
            if recorded_sha != supplied_sha:
                raise ValueError("pre-encoded checkpoint SHA256 differs from supplied checkpoint")
            checkpoint_metadata["checkpoint_verification_path"] = str(supplied)
    else:
        checkpoint_path = (
            Path(args.conch_checkpoint_path) if args.conch_checkpoint_path else None
        )
        cache_path = Path(args.conch_cache_path) if args.conch_cache_path else None
        combined, checkpoint_metadata = encode_with_project_conch_path(
            tuple(bundle.raw_prompt_texts) + LITERAL_PROMPTS,
            checkpoint_path,
            cache_path,
            args.device,
            args.hf_hub_offline,
            bool(getattr(args, "low_memory_mmap", False)),
        )
        raw_count = len(bundle.raw_prompt_texts)
        raw_embeddings = combined[:raw_count]
        literal_embeddings = combined[raw_count:]
        embedding_source = {"mode": "conch_inference", "device": args.device}

    raw_normalized, prototypes = aggregate_raw_embeddings(bundle, raw_embeddings)
    literal_normalized = l2_normalize(literal_embeddings, reject_zero=True)
    literal_cosine = float(np.dot(literal_normalized[0], literal_normalized[1]))
    variants = selected_variants(args.variant)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_new_directory(output_dir, "probe output")
    config: Dict[str, Any] = {
        "probe_schema_version": "conch_separability_probe_v1",
        "prompt_set": bundle.display_name,
        "conch_checkpoint_path": checkpoint_metadata["checkpoint_path"],
        "conch_checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "schema": path_and_sha(schema_path),
        "global27_templates": path_and_sha(global_path),
        "raw_prompt_count": len(bundle.raw_prompt_texts),
        "prototype_count": len(bundle.attribute_names) * 3,
        "literal_prompt_count": 2,
        "embedding_dim": int(prototypes.shape[1]),
        "encoding_function_source": checkpoint_metadata["encoding_function_source"],
        "encoding_function": checkpoint_metadata.get("encoding_function", "NOT_RECORDED"),
        "project_encoding_references": checkpoint_metadata.get(
            "project_encoding_references", project_encoding_references()
        ),
        "normalization_contract": checkpoint_metadata.get(
            "normalization_contract", "RECORDED_IN_PREENCODED_SOURCE"
        ),
        "model_load_audit": checkpoint_metadata.get(
            "model_load_audit", "RECORDED_IN_PREENCODED_SOURCE"
        ),
        "training_encoding_function_source": TRAINING_ENCODER_SOURCE,
        "embedding_source": embedding_source,
        "variants": list(variants),
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "segmentation_metrics_used": False,
        "training_started": False,
    }
    print("[PROBE_CONFIG] " + json.dumps(config, ensure_ascii=False, sort_keys=True))
    write_json(output_dir / "probe_config.json", config)
    prompt_manifest = {
        "prompt_set": bundle.display_name,
        "attribute_order": list(bundle.attribute_names),
        "attribute_groups": list(bundle.attribute_groups),
        "schema_level_order": list(SCHEMA_LEVEL_ORDER),
        "diagnostic_level_order": list(LEVEL_ORDER),
        "raw_prompt_ids": list(bundle.raw_prompt_ids),
        "raw_prompt_texts": list(bundle.raw_prompt_texts),
        "prototype_ids": list(bundle.prototype_ids),
        "prototype_raw_indices": [list(indices) for indices in bundle.prototype_raw_indices],
        "ignored_source_keys": list(bundle.ignored_source_keys),
        "literal_prompts": list(LITERAL_PROMPTS),
    }
    write_json(output_dir / "prompts.json", prompt_manifest)
    write_npz(
        output_dir / "raw_embeddings.npz",
        prompt_embeddings=np.asarray(raw_embeddings, dtype=np.float32),
        prompt_embeddings_l2=np.asarray(raw_normalized, dtype=np.float32),
        prompt_ids=np.asarray(bundle.raw_prompt_ids),
        prompt_texts=np.asarray(bundle.raw_prompt_texts),
        literal_embeddings=np.asarray(literal_embeddings, dtype=np.float32),
        literal_embeddings_l2=np.asarray(literal_normalized, dtype=np.float32),
        literal_texts=np.asarray(LITERAL_PROMPTS),
        checkpoint_path=np.asarray(checkpoint_metadata["checkpoint_path"]),
        checkpoint_sha256=np.asarray(checkpoint_metadata["checkpoint_sha256"]),
        encoding_function_source=np.asarray(
            checkpoint_metadata.get("encoding_function_source", ENCODER_SOURCE)
        ),
    )
    write_npz(
        output_dir / "prototype_embeddings.npz",
        prototype_embeddings=np.asarray(prototypes, dtype=np.float32),
        prototype_ids=np.asarray(bundle.prototype_ids),
        attribute_names=np.asarray(bundle.attribute_names),
        level_order=np.asarray(LEVEL_ORDER),
    )
    literal_payload = {
        "prompts": list(LITERAL_PROMPTS),
        "cosine": literal_cosine,
    }
    write_json(output_dir / "literal_density_cosine.json", literal_payload)

    metrics_by_variant: Dict[str, Dict[str, Any]] = {}
    transformed_by_variant: Dict[str, np.ndarray] = {}
    for variant in variants:
        transformed, extras = transform_variant(
            prototypes, len(bundle.attribute_names), variant
        )
        metrics, arrays = compute_metrics(transformed, bundle, variant, extras)
        variant_dir = output_dir / variant
        variant_dir.mkdir()
        write_json(variant_dir / "metrics.json", metrics)
        write_matrix_csv(
            variant_dir / "cosine_matrix.csv", bundle.prototype_ids, arrays["cosine_matrix"]
        )
        write_heatmap_svg(
            variant_dir / "cosine_heatmap.svg",
            bundle.prototype_ids,
            arrays["cosine_matrix"],
            f"{bundle.display_name} / {variant} cosine similarity",
        )
        write_matrix_csv(
            variant_dir / "level_axis_cosine_matrix.csv",
            bundle.attribute_names,
            arrays["level_axis_cosine_matrix"],
        )
        write_npz(
            variant_dir / "intermediates.npz",
            source_prototype_embeddings=np.asarray(prototypes, dtype=np.float32),
            transformed_embeddings=np.asarray(transformed, dtype=np.float32),
            **{key: np.asarray(value) for key, value in arrays.items()},
            **{f"transform_{key}": np.asarray(value) for key, value in extras.items()},
        )
        metrics_by_variant[variant] = metrics
        transformed_by_variant[variant] = transformed

    selected = select_best_variant(metrics_by_variant)
    classification = (
        classify_set_a_v0(metrics_by_variant)
        if bundle.prompt_set == "A"
        else {"classification": "PRIMARY_CLASSIFICATION_IS_SET_A_ONLY", "basis": {}}
    )
    summary: Dict[str, Any] = {
        "prompt_set": bundle.display_name,
        "variant_order": list(variants),
        "metrics_by_variant": {
            variant: metrics_by_variant[variant]["flat_metrics"] for variant in variants
        },
        "criteria_by_variant": {
            variant: metrics_by_variant[variant]["criteria"] for variant in variants
        },
        "best_eligible_variant": selected or "NONE",
        "variant_selection_rule": (
            "Among variants triggering none of C1-C5 (and excluding V4 when any mid "
            "residual ratio > 0.5), maximize eff_rank_95; ties minimize separation."
        ),
        "set_a_v0_failure_classification": classification,
        "literal_density_cosine": literal_cosine,
        "cross_encoder_control": "SKIPPED_NO_ACCESS",
    }

    if args.freeze_dir:
        if selected is None:
            raise ValueError("no eligible variant exists; refusing to freeze a bank")
        freeze_manifest = freeze_selected_bank(
            Path(args.freeze_dir).expanduser().resolve(),
            bundle,
            selected,
            transformed_by_variant[selected],
            metrics_by_variant[selected],
            checkpoint_metadata,
        )
        summary["freeze_manifest"] = freeze_manifest

    artifact_hashes: Dict[str, str] = {}
    for artifact in sorted(path for path in output_dir.rglob("*") if path.is_file()):
        artifact_hashes[str(artifact.relative_to(output_dir))] = sha256_file(artifact)
    summary["artifact_sha256"] = artifact_hashes
    write_json(output_dir / "summary.json", summary)
    print(
        "[PROBE_RESULT] "
        + json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "best_eligible_variant": selected or "NONE",
                "classification": classification["classification"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-only CONCH semantic prototype separability probe"
    )
    parser.add_argument(
        "--prompt_set",
        "--prompt-set",
        choices=("A", "B", "global27"),
        required=True,
        help="A=schema Local-5, B=Appendix-A 60 prompts, global27=strict template JSON",
    )
    parser.add_argument(
        "--variant",
        choices=("V0", "V1", "V2", "V2_k1", "V2_k2", "V3", "V4", "all"),
        default="all",
        help="V2 runs both registered k=1 and k=2 variants",
    )
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--global27_templates",
        "--global27-templates",
        default=str(DEFAULT_GLOBAL27_PATH),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="Must be absent or empty; existing results are never overwritten",
    )
    parser.add_argument(
        "--conch_checkpoint_path",
        "--conch-checkpoint-path",
        default=None,
        help="Local CONCH pytorch_model.bin; hashed before inference",
    )
    parser.add_argument(
        "--conch_cache_path",
        "--conch-cache-path",
        default=str(DEFAULT_SERVER_CONCH_CACHE),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf_hub_offline", "--hf-hub-offline", action="store_true")
    parser.add_argument(
        "--low_memory_mmap",
        "--low-memory-mmap",
        action="store_true",
        help=(
            "CPU-only exact-float32 loader using meta initialization plus a read-only "
            "checkpoint mmap; avoids the factory's checkpoint/model double allocation"
        ),
    )
    parser.add_argument(
        "--embeddings_input",
        "--embeddings-input",
        default=None,
        help="Strict pre-encoded .npz contract; skips all torch/CONCH imports",
    )
    parser.add_argument(
        "--write_encoding_request",
        "--write-encoding-request",
        default=None,
        help="Write ordered prompt request JSON and exit without encoding",
    )
    parser.add_argument(
        "--freeze_dir",
        "--freeze-dir",
        default=None,
        help="Optionally freeze the best eligible Set-A/Set-B variant into a new empty directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / f"{args.prompt_set}_{timestamp}")
    if args.write_encoding_request and args.embeddings_input:
        parser.error("--write-encoding-request and --embeddings-input are mutually exclusive")
    if args.low_memory_mmap and args.embeddings_input:
        parser.error("--low-memory-mmap is incompatible with --embeddings-input")
    if args.low_memory_mmap and args.device != "cpu":
        parser.error("--low-memory-mmap requires --device cpu")
    run_probe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
