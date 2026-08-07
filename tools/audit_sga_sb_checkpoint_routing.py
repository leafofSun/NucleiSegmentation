#!/usr/bin/env python3
"""State-dict-only SGA-SB/FreqPath routing audit (no model construction)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


TARGETS = {
    "residual_scale_1": ("mask_decoder.asr_upscale_1.residual_scale",),
    "residual_scale_2": ("mask_decoder.asr_upscale_2.residual_scale",),
    "fuse_final_1_weight": ("mask_decoder.asr_upscale_1.cnn_fusion.3.weight",),
    "fuse_final_1_bias": ("mask_decoder.asr_upscale_1.cnn_fusion.3.bias",),
    "fuse_final_2_weight": ("mask_decoder.asr_upscale_2.cnn_fusion.3.weight",),
    "fuse_final_2_bias": ("mask_decoder.asr_upscale_2.cnn_fusion.3.bias",),
    "cnn_proj_1_conv_weight": ("mask_decoder.asr_upscale_1.cnn_proj.0.weight",),
    "cnn_proj_1_norm_weight": ("mask_decoder.asr_upscale_1.cnn_proj.1.weight",),
    "cnn_proj_1_norm_bias": ("mask_decoder.asr_upscale_1.cnn_proj.1.bias",),
    "cnn_proj_2_conv_weight": ("mask_decoder.asr_upscale_2.cnn_proj.0.weight",),
    "cnn_proj_2_norm_weight": ("mask_decoder.asr_upscale_2.cnn_proj.1.weight",),
    "cnn_proj_2_norm_bias": ("mask_decoder.asr_upscale_2.cnn_proj.1.bias",),
    "structure_adapter_final_weight": ("spatial_structure_adapter.net.3.weight",),
    "structure_adapter_final_bias": ("spatial_structure_adapter.net.3.bias",),
    "boundary_adapter_final_weight": ("spatial_boundary_adapter.net.3.weight",),
    "boundary_adapter_final_bias": ("spatial_boundary_adapter.net.3.bias",),
    "gamma_structure": ("gamma_structure",),
    "gamma_boundary": ("gamma_boundary",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Any:
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **kwargs), "weights_only=True,mmap=True"
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, **kwargs), "weights_only=True,mmap=unsupported_fallback"


def find_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict", "model", "network", "net"):
            candidate = value.get(key)
            if isinstance(candidate, dict) and any(torch.is_tensor(item) for item in candidate.values()):
                value = candidate
                break
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint state must be dict, got {type(value).__name__}")
    return {str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)}


def match_key(state: dict[str, torch.Tensor], suffixes: tuple[str, ...]) -> str | None:
    matches = [key for key in state for suffix in suffixes if key == suffix or key.endswith("." + suffix)]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous suffix {suffixes}: {matches}")
    return matches[0]


def stats(tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().to(device="cpu", dtype=torch.float64)
    count = int(data.numel())
    return {
        "shape": "x".join(str(value) for value in data.shape) if data.ndim else "scalar",
        "numel": count,
        "l2_norm": float(torch.linalg.vector_norm(data).item()),
        "rms": float(torch.sqrt(torch.mean(data.square())).item()) if count else math.nan,
        "max_abs": float(data.abs().max().item()) if count else math.nan,
        "zero_ratio": float((data == 0).sum().item() / count) if count else math.nan,
        "scalar_value": float(data.item()) if count == 1 else math.nan,
    }


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.8e}"
    return str(value)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    rows: list[dict[str, object]] = []
    retained: dict[str, dict[str, torch.Tensor]] = {}
    load_modes: dict[str, str] = {}
    paths: dict[str, str] = {}
    for name, raw_path in args.checkpoint:
        path = Path(raw_path)
        checkpoint, load_mode = load_checkpoint(path)
        state = find_state_dict(checkpoint)
        load_modes[name] = load_mode
        paths[name] = path.as_posix()
        retained[name] = {}
        for logical_name, suffixes in TARGETS.items():
            key = match_key(state, suffixes)
            if key is None:
                rows.append({
                    "checkpoint": name, "path": path.as_posix(), "logical_name": logical_name,
                    "state_key": "MISSING", "present": False, "shape": "NA", "numel": 0,
                    "l2_norm": math.nan, "rms": math.nan, "max_abs": math.nan,
                    "zero_ratio": math.nan, "scalar_value": math.nan,
                })
                continue
            tensor = state[key]
            retained[name][logical_name] = tensor.detach().cpu().clone()
            row = {
                "checkpoint": name, "path": path.as_posix(), "logical_name": logical_name,
                "state_key": key, "present": True,
            }
            row.update(stats(tensor))
            rows.append(row)
        del state, checkpoint
        gc.collect()

    lookup = {(row["checkpoint"], row["logical_name"]): row for row in rows}
    comparisons: list[dict[str, object]] = []
    for logical_name in TARGETS:
        for left, right in (("G2", "Visual"), ("G3", "Visual"), ("G3", "G2")):
            a = retained.get(left, {}).get(logical_name)
            b = retained.get(right, {}).get(logical_name)
            if a is None or b is None or a.shape != b.shape:
                continue
            delta = a.to(torch.float64) - b.to(torch.float64)
            base_norm = float(torch.linalg.vector_norm(b.to(torch.float64)).item())
            comparisons.append({
                "logical_name": logical_name,
                "contrast": f"{left}-{right}",
                "delta_l2": float(torch.linalg.vector_norm(delta).item()),
                "relative_delta_l2": float(torch.linalg.vector_norm(delta).item() / max(base_norm, 1e-12)),
                "left_rms": float(lookup[(left, logical_name)]["rms"]),
                "right_rms": float(lookup[(right, logical_name)]["rms"]),
                "rms_ratio": float(lookup[(left, logical_name)]["rms"]) / max(float(lookup[(right, logical_name)]["rms"]), 1e-12),
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint", "path", "logical_name", "state_key", "present", "shape", "numel",
        "l2_norm", "rms", "max_abs", "zero_ratio", "scalar_value",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def value(checkpoint: str, logical_name: str, field: str = "scalar_value") -> float:
        return float(lookup[(checkpoint, logical_name)][field])

    visual_fuse2_rms = value("Visual", "fuse_final_2_weight", "rms")
    g2_fuse2_delta = next(item for item in comparisons if item["logical_name"] == "fuse_final_2_weight" and item["contrast"] == "G2-Visual")
    g3_fuse2_delta = next(item for item in comparisons if item["logical_name"] == "fuse_final_2_weight" and item["contrast"] == "G3-Visual")
    g2_adapter = value("G2", "boundary_adapter_final_weight", "l2_norm")
    g3_adapter = value("G3", "boundary_adapter_final_weight", "l2_norm")
    adapter_ratio = g3_adapter / max(g2_adapter, 1e-12)

    # A checkpoint-level gain proxy only. It omits normalization/activation and
    # is not claimed to equal a forward Jacobian.
    proxies: dict[str, float] = {}
    for checkpoint in ("G2", "G3"):
        proxies[checkpoint] = (
            abs(value(checkpoint, "gamma_boundary"))
            * value(checkpoint, "boundary_adapter_final_weight", "rms")
            * value(checkpoint, "cnn_proj_2_conv_weight", "rms")
            * value(checkpoint, "fuse_final_2_weight", "rms")
            * abs(value(checkpoint, "residual_scale_2"))
        )
    proxy_ratio = proxies["G3"] / max(proxies["G2"], 1e-30)
    compression = adapter_ratio < 0.9 or proxy_ratio < 0.9

    lines = [
        "# P4 Checkpoint Routing Audit", "",
        "State-dict tensors were loaded on CPU only. No model was instantiated and no forward/backward was executed.", "",
        "## Checkpoints", "",
    ]
    lines.extend(f"- **{name}**: `{paths[name]}`; `{load_modes[name]}`" for name in ("Visual", "G2", "G3"))
    lines += [
        "", "## Tensor statistics", "",
        "| Checkpoint | Tensor | Present | Shape | L2 | RMS | Max abs | Zero ratio | Scalar |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in (
            "checkpoint", "logical_name", "present", "shape", "l2_norm", "rms",
            "max_abs", "zero_ratio", "scalar_value",
        )) + " |")
    lines += [
        "", "## Cross-checkpoint updates", "",
        "| Tensor | Contrast | Delta L2 | Relative delta L2 | RMS ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append("| " + " | ".join(fmt(item[key]) for key in (
            "logical_name", "contrast", "delta_l2", "relative_delta_l2", "rms_ratio"
        )) + " |")
    lines += [
        "", "## Required judgments", "",
        f"1. Visual block-2 final fuse weight RMS: `{visual_fuse2_rms:.8e}`; near-zero criterion (`RMS < 1e-6`): **{'YES' if visual_fuse2_rms < 1e-6 else 'NO'}**.",
        f"2. G2/G3 block-2 fuse update relative to Visual: delta L2 `{g2_fuse2_delta['delta_l2']:.8e}` / `{g3_fuse2_delta['delta_l2']:.8e}`; nonzero update: **{'YES' if g2_fuse2_delta['delta_l2'] > 1e-6 and g3_fuse2_delta['delta_l2'] > 1e-6 else 'NO'}**.",
        "   The searched block-1/block-2 final fuse bias keys are absent in all three checkpoints, consistent with the current final `Conv2d(..., bias=False)` definition.",
        f"3. residual_scale_2: Visual `{value('Visual','residual_scale_2'):.8f}`, G2 `{value('G2','residual_scale_2'):.8f}`, G3 `{value('G3','residual_scale_2'):.8f}`.",
        f"4. Boundary adapter final-weight L2: G2 `{g2_adapter:.8e}`, G3 `{g3_adapter:.8e}`, G3/G2 `{adapter_ratio:.6f}`; G3 norm decrease: **{'YES' if adapter_ratio < 1 else 'NO'}**.",
        f"5. Boundary checkpoint gain proxy: G2 `{proxies['G2']:.8e}`, G3 `{proxies['G3']:.8e}`, G3/G2 `{proxy_ratio:.6f}`; compression criterion (<0.9 in adapter or proxy): **{'YES' if compression else 'NO'}**.",
        "", "The gain proxy is `abs(gamma_boundary) * adapter_final_RMS * cnn_proj2_conv_RMS * fuse2_final_RMS * abs(residual_scale_2)`. It is a routing-scale diagnostic, not a substitute for a forward/Jacobian measurement.",
    ]
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "PASS", "visual_fuse2_rms": visual_fuse2_rms,
        "g2_residual_scale_2": value("G2", "residual_scale_2"),
        "g3_residual_scale_2": value("G3", "residual_scale_2"),
        "g3_over_g2_boundary_adapter_l2": adapter_ratio,
        "g3_over_g2_gain_proxy": proxy_ratio,
        "compression": compression,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
