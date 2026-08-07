"""Model-agnostic evaluation protocol and no-padding eval sampling helpers."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


CANONICAL_PROTOCOL_NAME = "canonical_pannuke_v1"
TRAIN_VALIDATION_PROTOCOL_NAME = "train_validation_pannuke_v1"


@dataclasses.dataclass(frozen=True)
class EvaluationProtocol:
    protocol_name: str
    protocol_role: str
    comparable_to_canonical_full_test: bool
    difference_from_canonical: List[str]
    mask_threshold: float
    object_threshold: float
    min_object_size: int
    image_size: int
    patch_size: int
    sliding_overlap: Optional[float]
    semantic_mode: str
    FREQPATH_ABLATION: str
    use_asr: bool
    asr_variant: str
    use_pnurl: bool
    use_sga_sb: bool
    use_pnudp_dense: bool
    validation_fraction: Optional[float] = None
    validation_subset_seed: Optional[int] = None
    validation_subset_fixed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def protocol_from_train_args(args: Any) -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_name=TRAIN_VALIDATION_PROTOCOL_NAME,
        protocol_role="train_validation",
        comparable_to_canonical_full_test=False,
        difference_from_canonical=[
            "whole resized crop instead of sliding-window 8x-TTA inference",
            "mask/marker thresholds are 0.45/0.40 instead of canonical 0.40/0.45",
            "marker minimum is 10 instead of canonical 12; fallback has no size-15 filter",
        ],
        mask_threshold=0.45,
        object_threshold=0.40,
        min_object_size=10,
        image_size=int(args.image_size),
        patch_size=int(args.crop_size),
        sliding_overlap=None,
        semantic_mode=str(getattr(args, "eval_prompt_mode", "base")),
        FREQPATH_ABLATION=os.environ.get("FREQPATH_ABLATION", "UNSET"),
        use_asr=bool(getattr(args, "use_asr", False)),
        asr_variant=str(getattr(args, "asr_variant", "legacy")),
        use_pnurl=bool(getattr(args, "use_pnurl", False)),
        use_sga_sb=str(getattr(args, "spatial_sb_mode", "none")) != "none",
        use_pnudp_dense=bool(getattr(args, "enable_pnudp_dense_train", False)),
        validation_fraction=float(getattr(args, "val_fraction", 1.0)),
        validation_subset_seed=int(getattr(args, "val_subset_seed", 42)),
        validation_subset_fixed=True,
    )


def protocol_from_test_args(args: Any) -> EvaluationProtocol:
    differences: List[str] = []
    canonical_values = {
        "image_size": 512,
        "patch_size": 256,
        "overlap": 0.8,
        "prob_thresh": 0.40,
        "marker_thresh": 0.45,
        "min_marker_size": 12,
        "final_min_object_size": 15,
    }
    for name, canonical in canonical_values.items():
        actual = getattr(args, name)
        if actual != canonical:
            differences.append(f"{name}={actual!r} (canonical={canonical!r})")
    return EvaluationProtocol(
        protocol_name=CANONICAL_PROTOCOL_NAME if not differences else "custom_pannuke_full_test",
        protocol_role="full_test",
        comparable_to_canonical_full_test=not differences,
        difference_from_canonical=differences,
        mask_threshold=float(args.prob_thresh),
        object_threshold=float(args.marker_thresh),
        min_object_size=int(args.final_min_object_size),
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        sliding_overlap=float(args.overlap),
        semantic_mode=str(getattr(args, "prompt_mode", "organ_static")),
        FREQPATH_ABLATION=os.environ.get("FREQPATH_ABLATION", "UNSET"),
        use_asr=bool(getattr(args, "use_asr", False)),
        asr_variant=str(getattr(args, "asr_variant", "legacy")),
        use_pnurl=bool(getattr(args, "use_pnurl", False)),
        use_sga_sb=str(getattr(args, "spatial_sb_mode", "none")) != "none",
        use_pnudp_dense=bool(getattr(args, "enable_pnudp_dense", False)),
    )


def write_evaluation_protocol(protocol: EvaluationProtocol, run_dir: str) -> str:
    path = Path(run_dir) / "evaluation_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(protocol.to_json() + "\n", encoding="utf-8")
    return str(path)


def fixed_subset_indices(size: int, fraction: float, seed: int) -> List[int]:
    """Return one deterministic global subset independent of rank/world size."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("val_fraction must be in (0, 1]")
    if size == 0:
        return []
    target = size if fraction >= 1.0 else max(1, int(np.ceil(size * fraction)))
    if target == size:
        return list(range(size))
    rng = np.random.RandomState(seed)
    return sorted(int(i) for i in rng.permutation(size)[:target])


class DistributedEvalSampler(Sampler[int]):
    """Deterministic strided eval shard without padding or duplicates."""

    def __init__(self, indices: Sequence[int], num_replicas: int = 1, rank: int = 0):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid num_replicas/rank")
        self.indices = list(indices)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        remaining = len(self.indices) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def audit_unique_sample_ids(sample_ids_by_rank: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Require globally unique, non-empty stable IDs and return audit counts."""
    seen: Dict[str, int] = {}
    total = 0
    duplicates: List[str] = []
    for rank_ids in sample_ids_by_rank:
        for raw_sample_id in rank_ids:
            sample_id = str(raw_sample_id)
            if not sample_id:
                raise ValueError("sample_id must be non-empty")
            total += 1
            seen[sample_id] = seen.get(sample_id, 0) + 1
            if seen[sample_id] == 2:
                duplicates.append(sample_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:8])
        raise RuntimeError(
            "duplicate sample_id detected across evaluation ranks: "
            f"count={total - len(seen)}, ids=[{preview}]"
        )
    return {
        "global_seen_before_dedup": total,
        "global_unique": len(seen),
        "duplicate_sample_count": 0,
    }


def actual_delta_ratio(delta_norm: float, base_norm: float, eps: float = 1e-12) -> float:
    """ActualDeltaRatio := DeltaNorm / (BaseNorm + eps)."""
    return float(delta_norm) / (float(base_norm) + float(eps))


class MetricAccumulator:
    """Per-image macro-metric accumulator with stable sample-ID deduplication."""

    def __init__(self, metric_names: Sequence[str]):
        self.metric_names = tuple(metric_names)
        self._records: Dict[str, Dict[str, float]] = {}
        self.seen_before_dedup = 0

    def add(self, sample_id: str, metrics: Mapping[str, float]) -> bool:
        sample_id = str(sample_id)
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        self.seen_before_dedup += 1
        if sample_id in self._records:
            return False
        self._records[sample_id] = {
            name: float(metrics[name]) for name in self.metric_names if name in metrics
        }
        return True

    @property
    def unique_count(self) -> int:
        return len(self._records)

    @property
    def duplicates_removed(self) -> int:
        return self.seen_before_dedup - self.unique_count

    def sums_counts(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for name in self.metric_names:
            values = [row[name] for row in self._records.values() if name in row]
            result[name] = {"sum": float(sum(values)), "count": len(values)}
        return result

    def records(self) -> Dict[str, Dict[str, float]]:
        return {sample_id: dict(row) for sample_id, row in self._records.items()}

    @classmethod
    def merge(cls, metric_names: Sequence[str], records_by_rank: Iterable[Mapping[str, Mapping[str, float]]]):
        merged = cls(metric_names)
        for records in records_by_rank:
            for sample_id, metrics in records.items():
                merged.add(sample_id, metrics)
        return merged


DEFAULT_SOURCE_FILES = (
    "train.py", "test.py", "metrics.py", "DataLoader.py", "evaluation_audit.py",
    "segment_anything/modeling/sam.py", "segment_anything/modeling/mask_decoder.py",
    "segment_anything/modeling/pnurl.py", "segment_anything/build_sam.py",
    "training/phase_b_multilevel_attr.py", "training/phase_c_semantic_alignment.py",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(project_root: Path) -> Dict[str, Any]:
    """Read Git metadata without initializing or modifying a repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"git_available": True, "git_commit": commit, "git_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_available": False, "git_commit": "UNKNOWN", "git_dirty": "UNKNOWN"}


def write_run_manifests(
    run_dir: str,
    run_name: str,
    args: Any,
    protocol: EvaluationProtocol,
    project_root: str,
    parent_checkpoint: Optional[str],
    evaluation_context: Mapping[str, Any],
    source_files: Iterable[str] = DEFAULT_SOURCE_FILES,
) -> Dict[str, str]:
    """Write run/source lineage; never initializes Git or mutates a checkpoint."""
    output = Path(run_dir)
    root = Path(project_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in source_files:
        source = root / relative
        if source.is_file():
            info = source.stat()
            rows.append((relative, info.st_size, info.st_mtime, sha256_file(source)))
    tsv = "file\tsize\tmtime\tsha256\n" + "".join(
        f"{name}\t{size}\t{mtime:.6f}\t{digest}\n"
        for name, size, mtime, digest in rows
    )
    source_manifest = output / "source_manifest.tsv"
    source_hashes = output / "source_snapshot_hashes.tsv"
    source_manifest.write_text(tsv, encoding="utf-8")
    source_hashes.write_text(tsv, encoding="utf-8")

    checkpoint_path = Path(parent_checkpoint) if parent_checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_info = {"path": parent_checkpoint or "", "size": None, "mtime": None, "sha256": ""}
    if checkpoint_path is not None and checkpoint_path.is_file():
        info = checkpoint_path.stat()
        checkpoint_info.update(size=info.st_size, mtime=info.st_mtime, sha256=sha256_file(checkpoint_path))

    metrics_path = root / "metrics.py"
    manifest = {
        "run_name": run_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root": str(root),
        **_git_metadata(root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "command": shlex.join(sys.argv),
        "args": vars(args),
        "environment_overrides": {key: os.environ[key] for key in (
            "FREQPATH_ABLATION", "ALLOW_EVAL_SAMPLE_ATTRIBUTES", "ORGAN_DROPOUT_PROB",
            "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
        ) if key in os.environ},
        "parent_checkpoint": checkpoint_info,
        "data_split": evaluation_context.get("data_split", "UNKNOWN"),
        "sample_count": evaluation_context.get("sample_count", 0),
        "unique_sample_count": evaluation_context.get("unique_sample_count", 0),
        "duplicate_sample_count": evaluation_context.get("duplicate_sample_count", 0),
        "world_size": evaluation_context.get("world_size", 1),
        "sampler_type": evaluation_context.get("sampler_type", "UNKNOWN"),
        "metric_implementation": "metrics.py:SegMetrics per-image macro mean",
        "metric_implementation_version": "nuseg_segmetrics_per_image_macro_v1",
        "metric_implementation_sha256": sha256_file(metrics_path) if metrics_path.is_file() else "UNKNOWN",
        "evaluation_protocol": protocol.to_dict(),
        "semantic_inference_config": {
            "semantic_mode": protocol.semantic_mode,
            "FREQPATH_ABLATION": protocol.FREQPATH_ABLATION,
            "use_asr": protocol.use_asr,
            "asr_variant": protocol.asr_variant,
            "use_pnurl": protocol.use_pnurl,
            "use_sga_sb": protocol.use_sga_sb,
            "use_pnudp_dense": protocol.use_pnudp_dense,
        },
        "seed": int(getattr(args, "seed", 42)),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"run_manifest": str(manifest_path), "source_manifest": str(source_manifest), "source_snapshot_hashes": str(source_hashes)}
