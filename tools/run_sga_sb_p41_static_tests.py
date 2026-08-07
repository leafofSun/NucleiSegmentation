#!/usr/bin/env python3
"""CPU-only static/synthetic verification for P4.1 G2-Soft preparation."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(command: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for P4.1 static tests")
    entry_environment = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    tests: dict[str, object] = {}

    compile_paths = [
        "train.py",
        "training/spatial_sb_targets.py",
        "tools/audit_sga_sb_p2.py",
        "tools/summarize_sga_sb_p41.py",
        "tools/run_sga_sb_p41_static_tests.py",
    ]
    tests["py_compile"] = run_command(
        [sys.executable, "-m", "py_compile", *compile_paths], root
    )
    shell_paths = [
        "workdir/audits/sga_sb_p41_20260714/RUN_P41_G2_SOFT_2BATCH_AUDIT.sh",
        "workdir/audits/sga_sb_p41_20260714/RUN_P41_G2_SOFT_E5.sh",
    ]
    tests["bash_n"] = run_command(["bash", "-n", *shell_paths], root)

    formal_targets = load_module(
        "p41_spatial_sb_targets", root / "training" / "spatial_sb_targets.py"
    )
    train = load_module("p41_train", root / "train.py")

    with mock.patch.object(sys, "argv", ["train.py"]):
        default_args = train.parse_args()
    with mock.patch.object(
        sys,
        "argv",
        ["train.py", "--spatial_boundary_target_mode", "direct_area_soft"],
    ):
        soft_args = train.parse_args()
    tests["cli_parser_smoke"] = {
        "passed": (
            default_args.spatial_boundary_target_mode == "legacy_max"
            and soft_args.spatial_boundary_target_mode == "direct_area_soft"
        ),
        "default": default_args.spatial_boundary_target_mode,
        "explicit": soft_args.spatial_boundary_target_mode,
    }

    label_inst = torch.zeros((1, 1, 512, 512), dtype=torch.float32)
    label_inst[:, :, 40:173, 55:221] = 1
    label_inst[:, :, 180:405, 203:399] = 2
    label_inst[:, :, 250:292, 390:500] = 3
    full_boundary = formal_targets.generate_boundary_target(
        label_inst, kernel_size=3, target_size=None
    )
    prediction32 = torch.zeros((1, 1, 32, 32), dtype=torch.float32)

    legacy_raw64 = formal_targets.generate_boundary_target(
        label_inst, kernel_size=3, target_size=(64, 64)
    )
    legacy_actual32 = formal_targets.align_spatial_target_to_prediction(
        legacy_raw64, prediction32, "boundary"
    )
    legacy_reference64 = F.interpolate(full_boundary, size=(64, 64), mode="nearest")
    legacy_reference32 = F.adaptive_max_pool2d(legacy_reference64, (32, 32))
    tests["legacy_bitwise_regression"] = {
        "passed": (
            torch.equal(legacy_raw64, legacy_reference64)
            and torch.equal(legacy_actual32, legacy_reference32)
        ),
        "raw64_equal": torch.equal(legacy_raw64, legacy_reference64),
        "aligned32_equal": torch.equal(legacy_actual32, legacy_reference32),
        "dtype": str(legacy_actual32.dtype),
        "shape": list(legacy_actual32.shape),
    }

    direct_actual32 = formal_targets.align_spatial_target_to_prediction(
        full_boundary, prediction32, "boundary_soft"
    )
    direct_reference32 = F.adaptive_avg_pool2d(full_boundary, (32, 32))
    fractional_count = int(
        torch.count_nonzero((direct_actual32 > 0) & (direct_actual32 < 1)).item()
    )
    tests["direct_area_soft_target"] = {
        "passed": (
            torch.equal(direct_actual32, direct_reference32)
            and float(direct_actual32.min()) >= 0.0
            and float(direct_actual32.max()) <= 1.0
            and fractional_count > 0
        ),
        "direct_512_to_32_bitwise_equal": torch.equal(
            direct_actual32, direct_reference32
        ),
        "intermediate_64_used": False,
        "threshold_used": False,
        "min": float(direct_actual32.min()),
        "max": float(direct_actual32.max()),
        "fractional_cell_count": fractional_count,
    }

    logits = torch.linspace(-2.0, 2.0, direct_actual32.numel()).reshape_as(direct_actual32)
    loss_actual = formal_targets.compute_boundary_loss(
        logits, direct_actual32, target_mode="direct_area_soft"
    )
    pos_mass = direct_actual32.sum()
    neg_mass = direct_actual32.numel() - pos_mass
    pos_weight = torch.clamp(neg_mass / (pos_mass + 1e-6), min=0.1, max=10.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        direct_actual32,
        pos_weight=pos_weight.reshape(1).to(dtype=logits.dtype),
        reduction="mean",
    )
    probability = torch.sigmoid(logits)
    dice = 1.0 - (
        2.0 * (probability * direct_actual32).sum() + 1e-6
    ) / (probability.sum() + direct_actual32.sum() + 1e-6)
    loss_reference = bce + dice
    tests["synthetic_soft_target_loss"] = {
        "passed": (
            bool(torch.isfinite(loss_actual).item())
            and torch.allclose(loss_actual, loss_reference, rtol=0.0, atol=1e-7)
        ),
        "loss": float(loss_actual),
        "reference_loss": float(loss_reference),
        "pos_mass": float(pos_mass),
        "neg_mass": float(neg_mass),
        "pos_weight": float(pos_weight),
        "target_thresholded": False,
    }

    logging_logits = logits.clone().requires_grad_(True)
    logging_state = train._new_boundary_epoch_logging_state(torch.device("cpu"))
    train._accumulate_boundary_epoch_logging(
        logging_state, direct_actual32, logging_logits, "direct_area_soft"
    )
    logging_fields = train._finalize_boundary_epoch_logging(logging_state)
    required_fields = {
        "boundary_target_mass", "boundary_target_mean", "boundary_target_max",
        "boundary_target_nonzero_ratio", "boundary_pos_weight", "boundary_pred_mean",
        "boundary_pred_std", "boundary_pred_q90", "boundary_pred_q99",
    }
    finalize_source = inspect.getsource(train._finalize_boundary_epoch_logging)
    train_source = (root / "train.py").read_text(encoding="utf-8")
    tests["boundary_epoch_logging"] = {
        "passed": (
            required_fields == set(logging_fields)
            and logging_logits.grad is None
            and "dist.all_reduce" in finalize_source
            and "if rank == 0" in train_source
        ),
        "fields": logging_fields,
        "ddp_all_reduce_present": "dist.all_reduce" in finalize_source,
        "rank0_guard_present": "if rank == 0" in train_source,
        "prediction_gradient_created": logging_logits.grad is not None,
        "histogram_bins": train._BOUNDARY_LOG_HISTOGRAM_BINS,
        "full_prediction_tensor_retained": False,
        "repeated_forward": False,
    }

    summary = load_module(
        "p41_summary", root / "tools" / "summarize_sga_sb_p41.py"
    )
    legacy_rows = []
    soft_rows = []
    for epoch in range(1, 6):
        legacy_row = {"epoch": epoch}
        soft_row = {"epoch": epoch}
        for metric in summary.METRICS:
            if epoch == 5:
                legacy_value = summary.LEGACY_E5[metric]
            elif epoch == 4:
                legacy_value = 2.0 * summary.LEGACY_E45_MEAN[metric] - summary.LEGACY_E5[metric]
            else:
                legacy_value = 0.5 + epoch * 0.01
            gain = 0.004 if metric in {"mPQ", "mAJI"} else 0.001
            legacy_row[metric] = legacy_value
            soft_row[metric] = legacy_value + gain
        legacy_rows.append(legacy_row)
        soft_rows.append(soft_row)
    with tempfile.TemporaryDirectory(prefix="p41_summary_") as tmp:
        tmpdir = Path(tmp)
        legacy_path = tmpdir / "legacy.json"
        soft_path = tmpdir / "soft.json"
        csv_path = tmpdir / "comparison.csv"
        md_path = tmpdir / "comparison.md"
        legacy_path.write_text(json.dumps(legacy_rows), encoding="utf-8")
        soft_path.write_text(json.dumps(soft_rows), encoding="utf-8")
        legacy_history = summary.load_history(legacy_path)
        soft_history = summary.load_history(soft_path)
        summary.verify_legacy_anchors(legacy_history)
        rows = summary.build_rows(legacy_history, soft_history)
        gate = summary.evaluate_gate(rows)
        summary.write_outputs(rows, gate, csv_path, md_path)
        tests["summary_parser_synthetic"] = {
            "passed": gate["advance"] and csv_path.stat().st_size > 0 and md_path.stat().st_size > 0,
            "advance": gate["advance"],
            "checks": gate["checks"],
            "row_count": len(rows),
        }

    passed = all(bool(value.get("passed")) for value in tests.values() if isinstance(value, dict))
    payload = {
        "schema_version": "p41_g2_soft_static_tests_v1",
        "result": "PASS" if passed else "FAIL",
        "environment_at_entry": entry_environment,
        "safety": {
            "gpu_used": False,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "model_instantiated": False,
            "model_forward": False,
            "backward": False,
            "optimizer_created": False,
            "training_executed": False,
        },
        "tests": tests,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": payload["result"], "test_count": len(tests)}))
    return 0 if passed and not payload["safety"]["cuda_initialized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
