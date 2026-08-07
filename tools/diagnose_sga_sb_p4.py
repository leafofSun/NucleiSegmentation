#!/usr/bin/env python3
"""Dynamic SGA-SB tensor diagnostics without a hard-coded model loader.

The tool analyses four counterfactual modes (none, structure_only,
boundary_only, both) supplied either as a captured ``.npz`` bundle or by a
future GPU backend factory.  Gradient cosine consumes already-computed
gradient tensors; this module never calls autograd or ``backward`` itself.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np


MODES = ("none", "structure_only", "boundary_only", "both")
SHIFT_RE = re.compile(r"^shift_logits_both_dx(?P<dx>[+-]?\d+)_dy(?P<dy>[+-]?\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-bundle", type=Path)
    source.add_argument("--backend-factory", help="module:function returning Mapping[str, ndarray]")
    source.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--backend-config-json", default="{}")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else math.nan


def cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"gradient shape mismatch: {a.shape} vs {b.shape}")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(float(denominator), eps))


def dice_from_logits(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left) > 0
    b = np.asarray(right) > 0
    denominator = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 1.0


def center_of_mass_abs(value: np.ndarray) -> list[float] | None:
    array = np.abs(np.asarray(value, dtype=np.float64))
    while array.ndim > 2:
        array = array.mean(axis=0)
    total = float(array.sum())
    if total <= 0:
        return None
    coords = np.indices(array.shape)
    return [float((coords[axis] * array).sum() / total) for axis in range(2)]


def analyse(bundle: Mapping[str, np.ndarray]) -> tuple[dict[str, object], list[dict[str, object]]]:
    missing = [f"logits_{mode}" for mode in MODES if f"logits_{mode}" not in bundle]
    if missing:
        raise KeyError(f"missing required counterfactual tensors: {missing}")
    logits = {mode: np.asarray(bundle[f"logits_{mode}"]) for mode in MODES}
    shapes = {mode: list(value.shape) for mode, value in logits.items()}
    if len({tuple(shape) for shape in shapes.values()}) != 1:
        raise ValueError(f"counterfactual logit shape mismatch: {shapes}")

    structure_effect = logits["structure_only"] - logits["none"]
    boundary_effect = logits["boundary_only"] - logits["none"]
    both_effect = logits["both"] - logits["none"]
    interaction = logits["both"] - logits["structure_only"] - logits["boundary_only"] + logits["none"]
    rows: list[dict[str, object]] = []
    for name, value in bundle.items():
        rows.append({"section": "tensor_rms", "name": name, "value": rms(np.asarray(value))})

    counterfactual = {
        "structure_effect_rms": rms(structure_effect),
        "boundary_effect_rms": rms(boundary_effect),
        "both_effect_rms": rms(both_effect),
        "interaction_rms": rms(interaction),
        "interaction_over_both_effect": rms(interaction) / max(rms(both_effect), 1e-12),
        "structure_boundary_effect_cosine": cosine(structure_effect, boundary_effect),
    }
    rows.extend({"section": "counterfactual", "name": key, "value": value} for key, value in counterfactual.items())

    gradient = None
    if "grad_structure" in bundle and "grad_boundary" in bundle:
        gradient = {
            "cosine": cosine(bundle["grad_structure"], bundle["grad_boundary"]),
            "structure_rms": rms(bundle["grad_structure"]),
            "boundary_rms": rms(bundle["grad_boundary"]),
        }
        rows.extend({"section": "gradient", "name": key, "value": value} for key, value in gradient.items())

    impulse: dict[str, object] = {}
    if "impulse_none" in bundle:
        baseline = np.asarray(bundle["impulse_none"])
        for branch in ("structure", "boundary", "both"):
            key = f"impulse_{branch}"
            if key not in bundle:
                continue
            response = np.asarray(bundle[key]) - baseline
            impulse[branch] = {
                "response_rms": rms(response),
                "response_max_abs": float(np.max(np.abs(response))),
                "response_center_of_mass": center_of_mass_abs(response),
            }
            rows.append({"section": "impulse", "name": f"{branch}_response_rms", "value": rms(response)})

    shift_sweep: list[dict[str, object]] = []
    for key, value in bundle.items():
        match = SHIFT_RE.match(key)
        if not match:
            continue
        shifted = np.asarray(value)
        row = {
            "dx": int(match.group("dx")),
            "dy": int(match.group("dy")),
            "logit_rms_change": rms(shifted - logits["both"]),
            "binary_dice_vs_unshifted": dice_from_logits(shifted, logits["both"]),
        }
        shift_sweep.append(row)
        rows.append({"section": "shift_sweep", "name": f"dx={row['dx']},dy={row['dy']}", "value": row["logit_rms_change"]})
    shift_sweep.sort(key=lambda row: (row["dy"], row["dx"]))

    report: dict[str, object] = {
        "schema_version": "sga_sb_p4_dynamic_v1",
        "modes": MODES,
        "logit_shapes": shapes,
        "tensor_rms": {name: rms(np.asarray(value)) for name, value in bundle.items()},
        "counterfactual_logit_interaction": counterfactual,
        "gradient_cosine": gradient,
        "impulse_test": impulse,
        "integer_shift_sweep": shift_sweep,
        "autograd_called_by_this_tool": False,
    }
    return report, rows


def synthetic_bundle(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(2, 1, 8, 8)).astype(np.float32)
    structure = rng.normal(scale=0.1, size=base.shape).astype(np.float32)
    boundary = rng.normal(scale=0.03, size=base.shape).astype(np.float32)
    interaction = 0.05 * structure * boundary
    bundle: dict[str, np.ndarray] = {
        "logits_none": base,
        "logits_structure_only": base + structure,
        "logits_boundary_only": base + boundary,
        "logits_both": base + structure + boundary + interaction,
        "grad_structure": structure,
        "grad_boundary": boundary,
        "impulse_none": np.zeros((1, 1, 9, 9), dtype=np.float32),
        "impulse_structure": np.pad(np.ones((1, 1, 1, 1), dtype=np.float32), ((0, 0), (0, 0), (4, 4), (4, 4))),
        "impulse_boundary": np.pad(np.ones((1, 1, 1, 1), dtype=np.float32) * 0.5, ((0, 0), (0, 0), (4, 4), (4, 4))),
        "impulse_both": np.pad(np.ones((1, 1, 1, 1), dtype=np.float32) * 1.5, ((0, 0), (0, 0), (4, 4), (4, 4))),
    }
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            bundle[f"shift_logits_both_dx{dx:+d}_dy{dy:+d}"] = np.roll(
                bundle["logits_both"], shift=(dy, dx), axis=(-2, -1)
            )
    return bundle


def load_backend(spec: str, config_json: str) -> Mapping[str, np.ndarray]:
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--backend-factory must be module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    result = factory(json.loads(config_json))
    if not isinstance(result, Mapping):
        raise TypeError("backend factory must return Mapping[str, ndarray]")
    return result


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        bundle = synthetic_bundle()
    elif args.input_bundle:
        with np.load(args.input_bundle, allow_pickle=False) as loaded:
            bundle = {name: loaded[name] for name in loaded.files}
    else:
        bundle = dict(load_backend(args.backend_factory, args.backend_config_json))
    report, rows = analyse(bundle)
    if args.synthetic_self_test:
        assert report["counterfactual_logit_interaction"]["interaction_rms"] > 0
        assert len(report["integer_shift_sweep"]) == 25
        print(json.dumps({"result": "PASS", "synthetic_only": True, "autograd": False}))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=("section", "name", "value"))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
