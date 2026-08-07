#!/usr/bin/env python3
"""Materialize the frozen P1-2 RSGR tensors into the formal loader format.

This utility performs tensor movement and metadata wrapping only.  It never
imports or calls CONCH, and it refuses any input whose SHA256 differs from the
P1-2 frozen artifact set.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rsgr_local5 import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    attributes_for_group,
    load_local5_schema,
    sha256_file,
)


BANK_SCHEMA_VERSION = "rsgr_local5_conch_bank_v1"
EXPECTED_INPUT_SHA256 = {
    "prompts_frozen.json": "de4413374061d3886fc87288ff48c46ea5f07d00268aaf191c7328d74f55eaa3",
    "structure_bank.pt": "ca28900b8650ec49974da776bdc2bef0e9408f42421e6f7aee5d4a32a34786a8",
    "boundary_bank.pt": "cb5cfb2d79d05cbeeef28efa5a25bb1252b287ed497c231929d8447308aeea0d",
    "bank_manifest.json": "a10944ad06cffdf70742c93ed2c6570ec32b8810ea77ac013faacd04c0cab7f1",
}
EXPECTED_SCHEMA_SHA256 = "01c8dfc779811592207df7b678b84bb192a42aebd00b18748eb09e24d0126e79"
ATTRIBUTE_ORDER = (
    "nuclear_density",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
    "nuclear_irregularity",
    "nuclear_elongation",
)
GROUPED_ATTRIBUTE_NAMES = {
    "structure": list(ATTRIBUTE_ORDER[:3]),
    "boundary": list(ATTRIBUTE_ORDER[3:]),
}
SCHEMA_LEVEL_ORDER = ("low", "medium", "high")
DIAGNOSTIC_LEVEL_ALIASES = ("low", "mid", "high")
EXPECTED_SHAPES = {
    "structure": (3, 3, 512),
    "boundary": (2, 3, 512),
}
EXPECTED_METRICS = {
    "intra_attr_cos": 0.4338474906449287,
    "inter_attr_cos": -0.15165889644610442,
    "separation": -0.5855063870910331,
    "eff_rank_95": 10,
    "level_axis_alignment": 0.0023018233564638365,
    "monotonic_ratio": 1.0,
}


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _validate_frozen_inputs(input_dir: Path, schema_path: Path):
    actual_schema_sha256 = sha256_file(schema_path)
    if actual_schema_sha256 != EXPECTED_SCHEMA_SHA256:
        raise ValueError(
            "canonical RSGR Local-5 schema SHA256 mismatch: "
            f"expected {EXPECTED_SCHEMA_SHA256}, got {actual_schema_sha256}"
        )
    paths = {name: input_dir / name for name in EXPECTED_INPUT_SHA256}
    for name, expected_sha in EXPECTED_INPUT_SHA256.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"frozen RSGR input NOT_FOUND: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"frozen RSGR input SHA256 mismatch for {path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    schema = load_local5_schema(schema_path)
    grouped_names = {
        group: [row["name"] for row in attributes_for_group(schema, group)]
        for group in ("structure", "boundary")
    }
    _require_equal(grouped_names, GROUPED_ATTRIBUTE_NAMES, "schema attribute order")
    _require_equal(tuple(schema["classes"]), SCHEMA_LEVEL_ORDER, "schema level order")

    prompts = _read_json(paths["prompts_frozen.json"])
    _require_equal(prompts.get("prompt_set"), "Set-A", "prompt set")
    _require_equal(tuple(prompts.get("attribute_order", ())), ATTRIBUTE_ORDER, "prompt attribute order")
    _require_equal(tuple(prompts.get("level_order", ())), SCHEMA_LEVEL_ORDER, "prompt level order")
    _require_equal(
        tuple(prompts.get("diagnostic_level_aliases", ())),
        DIAGNOSTIC_LEVEL_ALIASES,
        "diagnostic level aliases",
    )
    schema_prompt_texts = [
        prompt
        for row in schema["attributes"]
        for prompt in row["prompt_texts"]
    ]
    _require_equal(prompts.get("raw_prompt_texts"), schema_prompt_texts, "schema/frozen prompt text order")
    _require_equal(
        prompts.get("prototype_raw_indices"),
        [[index] for index in range(15)],
        "Set-A prototype aggregation order",
    )

    manifest = _read_json(paths["bank_manifest.json"])
    _require_equal(manifest.get("prompt_set"), "Set-A", "manifest prompt set")
    _require_equal(manifest.get("geometric_variant"), "V1", "manifest geometric variant")
    _require_equal(manifest.get("prompts_sha256"), EXPECTED_INPUT_SHA256["prompts_frozen.json"], "manifest prompt SHA256")
    _require_equal(manifest.get("structure_bank_sha256"), EXPECTED_INPUT_SHA256["structure_bank.pt"], "manifest structure SHA256")
    _require_equal(manifest.get("boundary_bank_sha256"), EXPECTED_INPUT_SHA256["boundary_bank.pt"], "manifest boundary SHA256")
    _require_equal(manifest.get("embedding_dim"), 512, "manifest embedding dimension")
    _require_equal(tuple(manifest.get("attribute_order", ())), ATTRIBUTE_ORDER, "manifest attribute order")
    _require_equal(tuple(manifest.get("level_order", ())), SCHEMA_LEVEL_ORDER, "manifest level order")
    _require_equal(
        tuple(manifest.get("diagnostic_level_aliases", ())),
        DIAGNOSTIC_LEVEL_ALIASES,
        "manifest diagnostic level aliases",
    )
    frozen_metrics = manifest.get("metrics_at_freeze")
    if not isinstance(frozen_metrics, Mapping):
        raise ValueError("manifest metrics_at_freeze is missing or invalid")
    for name, expected in EXPECTED_METRICS.items():
        actual = frozen_metrics.get(name)
        if name == "eff_rank_95":
            _require_equal(actual, expected, f"manifest metric {name}")
        elif actual is None or abs(float(actual) - float(expected)) > 1e-6:
            raise ValueError(
                f"manifest metric {name} mismatch: expected {expected!r}, got {actual!r}"
            )

    banks = {}
    for group in ("structure", "boundary"):
        value = torch.load(paths[f"{group}_bank.pt"], map_location="cpu", weights_only=True)
        if not torch.is_tensor(value):
            raise ValueError(f"frozen {group} bank must contain one tensor")
        if tuple(value.shape) != EXPECTED_SHAPES[group]:
            raise ValueError(
                f"frozen {group} bank shape mismatch: "
                f"expected {EXPECTED_SHAPES[group]}, got {tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"frozen {group} bank dtype must be torch.float32, got {value.dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"frozen {group} bank contains NaN or Inf")
        norms = value.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
            raise ValueError(f"frozen {group} bank is not unit-normalized")
        moved = value.detach().cpu().contiguous()
        if not torch.equal(moved, value):
            raise AssertionError(f"tensor movement changed {group} bank values")
        banks[group] = moved

    return paths, schema, prompts, manifest, banks


def _torch_save_bytes(payload: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _publish_atomic_exclusive(path: Path, data: bytes) -> None:
    """Publish complete bytes without overwriting; preserve staging on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging.", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        os.unlink(temporary_name)
    except Exception as error:
        raise RuntimeError(
            f"atomic publish failed; staging evidence retained at {temporary_name}"
        ) from error


def materialize(input_dir: Path, schema_path: Path, output_path: Path) -> Mapping[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only RSGR materializer")
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() and metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite complete existing output pair: "
            f"{output_path}, {metadata_path}"
        )

    paths, schema, prompts, manifest, banks = _validate_frozen_inputs(input_dir, schema_path)
    grouped_prompts = {
        row["name"]: list(row["prompt_texts"])
        for row in schema["attributes"]
    }
    payload = {
        "schema_version": BANK_SCHEMA_VERSION,
        "backend": "conch",
        "prompt_set": "Set-A",
        "geometric_variant": "V1",
        "attribute_names": GROUPED_ATTRIBUTE_NAMES,
        "class_names": list(SCHEMA_LEVEL_ORDER),
        "structure_prototypes": banks["structure"],
        "boundary_prototypes": banks["boundary"],
        "source_manifest_sha256": EXPECTED_INPUT_SHA256["bank_manifest.json"],
    }
    bank_bytes = _torch_save_bytes(payload)
    if bank_bytes != _torch_save_bytes(payload):
        raise RuntimeError("torch serialization was not byte-deterministic in this runtime")

    bank_sha256 = hashlib.sha256(bank_bytes).hexdigest()
    metadata = {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_name": "rsgr_local5_conch_bank_v1",
        "backend": "conch",
        "encoder": "CONCH",
        "embedding_dim": 512,
        "attribute_names": GROUPED_ATTRIBUTE_NAMES,
        "class_names": list(SCHEMA_LEVEL_ORDER),
        "level_order": list(SCHEMA_LEVEL_ORDER),
        "diagnostic_level_aliases": list(DIAGNOSTIC_LEVEL_ALIASES),
        "prompts": grouped_prompts,
        "shapes": {group: list(EXPECTED_SHAPES[group]) for group in ("structure", "boundary")},
        "dtype": "torch.float32",
        "finite": True,
        "normalized": True,
        "bank_sha256": bank_sha256,
        "schema_path": "training/rsgr_local5_schema.json",
        "schema_sha256": sha256_file(schema_path),
        "source_files": {
            name: {"path": name, "sha256": expected_sha}
            for name, expected_sha in EXPECTED_INPUT_SHA256.items()
        },
        "source_prompt_set": manifest["prompt_set"],
        "source_geometric_variant": manifest["geometric_variant"],
        "source_created_utc": manifest.get("created_utc", "UNVERIFIED"),
        "conch_checkpoint_path": manifest.get("conch_checkpoint_path", "UNVERIFIED"),
        "conch_checkpoint_sha256": manifest.get("conch_checkpoint_sha256", "UNVERIFIED"),
        "encoding_function_source": manifest.get("encoding_function_source", "UNVERIFIED"),
        "materialization": "tensor_movement_and_metadata_wrapping_only",
        "conch_reencoded": False,
        "geometry_transform_applied": False,
        "normalization_applied": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "training_started": False,
        "frozen_prompts_sha256": sha256_file(paths["prompts_frozen.json"]),
        "frozen_metrics": {name: manifest["metrics_at_freeze"][name] for name in EXPECTED_METRICS},
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")

    partial_recovery = output_path.exists() or metadata_path.exists()
    if output_path.exists() and output_path.read_bytes() != bank_bytes:
        raise FileExistsError(
            f"refusing partial recovery because existing bank bytes differ: {output_path}"
        )
    if metadata_path.exists() and metadata_path.read_bytes() != metadata_bytes:
        raise FileExistsError(
            f"refusing partial recovery because existing metadata bytes differ: {metadata_path}"
        )
    if not output_path.exists():
        _publish_atomic_exclusive(output_path, bank_bytes)
    if not metadata_path.exists():
        _publish_atomic_exclusive(metadata_path, metadata_bytes)
    result = {
        "training_started": False,
        "conch_reencoded": False,
        "geometry_transform_applied": False,
        "partial_output_recovered": partial_recovery,
        "output_bank": str(output_path),
        "output_bank_sha256": bank_sha256,
        "output_metadata": str(metadata_path),
        "output_metadata_sha256": sha256_file(metadata_path),
    }
    print("[RSGR_BANK_MATERIALIZED] " + json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrap the hash-frozen P1-2 tensors for the formal RSGR loader"
    )
    parser.add_argument("--input-dir", default="workdir/rsgr_bank")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--output",
        default="workdir/rsgr_bank/rsgr_local5_conch_bank_v1.pt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    materialize(
        Path(args.input_dir).expanduser().resolve(),
        Path(args.schema).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
