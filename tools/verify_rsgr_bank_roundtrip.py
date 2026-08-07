#!/usr/bin/env python3
"""CPU-only exact round-trip and frozen-geometry verification for RSGR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything.modeling.rsgr import load_prototype_banks  # noqa: E402
from tools.materialize_rsgr_bank import (  # noqa: E402
    ATTRIBUTE_ORDER,
    BANK_SCHEMA_VERSION,
    DIAGNOSTIC_LEVEL_ALIASES,
    EXPECTED_INPUT_SHA256,
    EXPECTED_METRICS,
    EXPECTED_SCHEMA_SHA256,
    EXPECTED_SHAPES,
    GROUPED_ATTRIBUTE_NAMES,
    SCHEMA_LEVEL_ORDER,
)
from training.rsgr_local5 import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    attributes_for_group,
    load_local5_schema,
    sha256_file,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _effective_rank(singular_values: np.ndarray, threshold: float = 0.95) -> int:
    energy = np.square(singular_values)
    total = float(energy.sum())
    if total <= 1e-24:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / total, threshold, side="left") + 1)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-12)


def compute_frozen_metrics(matrix: np.ndarray) -> Mapping[str, Any]:
    """Recompute the P1-2 metrics with the same float64 formulae."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (15, 512) or not np.isfinite(value).all():
        raise ValueError(f"expected a finite [15,512] matrix, got {value.shape}")
    cosine_ready = _l2_normalize(value)
    cosine = cosine_ready @ cosine_ready.T
    attribute_index = np.repeat(np.arange(5, dtype=np.int64), 3)
    row, column = np.triu_indices(15, k=1)
    same_attribute = attribute_index[row] == attribute_index[column]
    intra = float(cosine[row[same_attribute], column[same_attribute]].mean())
    inter = float(cosine[row[~same_attribute], column[~same_attribute]].mean())

    centered = value - value.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    shaped = value.reshape(5, 3, 512)
    level_axes = _l2_normalize(shaped[:, 2, :] - shaped[:, 0, :])
    axis_cosine = level_axes @ level_axes.T
    axis_row, axis_column = np.triu_indices(5, k=1)
    alignment = float(axis_cosine[axis_row, axis_column].mean())
    monotonic_values = []
    for low, medium, high in shaped:
        direction = high - low
        denominator = float(np.dot(direction, direction)) + 1e-12
        monotonic_values.append(float(np.dot(medium - low, direction) / denominator))
    return {
        "intra_attr_cos": intra,
        "inter_attr_cos": inter,
        "separation": inter - intra,
        "eff_rank_95": _effective_rank(singular_values),
        "level_axis_alignment": alignment,
        "monotonic_ratio": float(sum(0.0 < item < 1.0 for item in monotonic_values) / 5.0),
    }


def _write_report_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify(
    input_dir: Path,
    schema_path: Path,
    bank_path: Path,
    report_path: Path | None = None,
) -> Mapping[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only round-trip verifier")
    if sha256_file(schema_path) != EXPECTED_SCHEMA_SHA256:
        raise ValueError(
            "verifier schema differs from the canonical production Local-5 schema"
        )
    metadata_path = bank_path.with_suffix(".metadata.json")
    for path in (bank_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"formal RSGR artifact NOT_FOUND: {path}")
    for name, expected_sha in EXPECTED_INPUT_SHA256.items():
        source_path = input_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(f"frozen RSGR input NOT_FOUND: {source_path}")
        actual_sha = sha256_file(source_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"frozen RSGR input SHA256 mismatch for {source_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    schema = load_local5_schema(schema_path)
    expected_names = {
        group: [row["name"] for row in attributes_for_group(schema, group)]
        for group in ("structure", "boundary")
    }
    if expected_names != GROUPED_ATTRIBUTE_NAMES:
        raise ValueError("schema attribute order differs from the frozen Local-5 order")
    if tuple(schema["classes"]) != SCHEMA_LEVEL_ORDER:
        raise ValueError("schema level order differs from low/medium/high")

    metadata = _read_json(metadata_path)
    if metadata.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError("formal metadata schema_version mismatch")
    if metadata.get("backend") != "conch":
        raise ValueError("formal metadata backend must be conch")
    if metadata.get("attribute_names") != expected_names:
        raise ValueError("formal metadata attribute order mismatch")
    if tuple(metadata.get("level_order", ())) != SCHEMA_LEVEL_ORDER:
        raise ValueError("formal metadata level order mismatch")
    if tuple(metadata.get("class_names", ())) != SCHEMA_LEVEL_ORDER:
        raise ValueError("formal metadata class_names mismatch")
    if tuple(metadata.get("diagnostic_level_aliases", ())) != DIAGNOSTIC_LEVEL_ALIASES:
        raise ValueError("formal metadata diagnostic level aliases mismatch")
    if metadata.get("schema_sha256") != sha256_file(schema_path):
        raise ValueError("formal metadata schema SHA256 mismatch")
    if metadata.get("bank_sha256") != sha256_file(bank_path):
        raise ValueError("formal metadata bank SHA256 mismatch")

    payload = torch.load(bank_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("backend") != "conch":
        raise ValueError("formal payload must be a CONCH mapping")
    if payload.get("attribute_names") != expected_names:
        raise ValueError("formal payload attribute order mismatch")
    if tuple(payload.get("class_names", ())) != SCHEMA_LEVEL_ORDER:
        raise ValueError("formal payload level order mismatch")

    loaded_structure, loaded_boundary = load_prototype_banks(
        bank_path,
        metadata_path=metadata_path,
        schema_path=schema_path,
    )
    original_structure = torch.load(
        input_dir / "structure_bank.pt", map_location="cpu", weights_only=True
    )
    original_boundary = torch.load(
        input_dir / "boundary_bank.pt", map_location="cpu", weights_only=True
    )

    banks = {
        "structure": (original_structure, loaded_structure),
        "boundary": (original_boundary, loaded_boundary),
    }
    checks = {}
    for group, (original, loaded) in banks.items():
        expected_shape = EXPECTED_SHAPES[group]
        if tuple(original.shape) != expected_shape or tuple(loaded.shape) != expected_shape:
            raise ValueError(f"{group} shape round-trip mismatch")
        maximum_difference = float((loaded - original).abs().max().item())
        exact = bool(torch.equal(loaded, original))
        finite = bool(torch.isfinite(loaded).all().item())
        norms = loaded.norm(dim=-1)
        normalized = bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)
        )
        if maximum_difference != 0.0 or not exact:
            raise AssertionError(
                f"{group} round-trip changed tensor values: max_abs_diff={maximum_difference}"
            )
        if not finite or not normalized:
            raise AssertionError(f"{group} round-trip failed finite/unit-norm validation")
        checks[group] = {
            "shape": list(loaded.shape),
            "max_abs_diff": maximum_difference,
            "exact_equal": exact,
            "finite": finite,
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
            "unit_normalized_atol": 1e-5,
        }

    flat = torch.cat((loaded_structure, loaded_boundary), dim=0).reshape(15, 512)
    metrics = compute_frozen_metrics(flat.detach().cpu().numpy())
    metric_checks = {}
    for name, expected in EXPECTED_METRICS.items():
        actual = metrics[name]
        if name == "eff_rank_95":
            passed = int(actual) == int(expected)
            absolute_error = abs(int(actual) - int(expected))
        else:
            absolute_error = abs(float(actual) - float(expected))
            passed = absolute_error <= 1e-6
        metric_checks[name] = {
            "actual": actual,
            "expected": expected,
            "absolute_error": absolute_error,
            "passed": passed,
        }
        if not passed:
            raise AssertionError(
                f"frozen metric {name} mismatch: expected {expected}, got {actual}"
            )

    result = {
        "training_started": False,
        "device": "cpu",
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "real_loader_called": "segment_anything.modeling.rsgr.load_prototype_banks",
        "status": "PASS",
        "bank_path": str(bank_path),
        "bank_sha256": sha256_file(bank_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "backend": payload["backend"],
        "attribute_order": list(ATTRIBUTE_ORDER),
        "grouped_attribute_names": expected_names,
        "level_order": list(SCHEMA_LEVEL_ORDER),
        "diagnostic_level_aliases": list(DIAGNOSTIC_LEVEL_ALIASES),
        "roundtrip_checks": checks,
        "geometry_metric_checks": metric_checks,
    }
    if report_path is not None:
        _write_report_exclusive(report_path, result)
    print("[RSGR_BANK_ROUNDTRIP] " + json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify exact CPU round-trip through the production RSGR loader"
    )
    parser.add_argument("--input-dir", default="workdir/rsgr_bank")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--bank",
        default="workdir/rsgr_bank/rsgr_local5_conch_bank_v1.pt",
    )
    parser.add_argument("--report", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    verify(
        Path(args.input_dir).expanduser().resolve(),
        Path(args.schema).expanduser().resolve(),
        Path(args.bank).expanduser().resolve(),
        Path(args.report).expanduser().resolve() if args.report else None,
    )


if __name__ == "__main__":
    main()
