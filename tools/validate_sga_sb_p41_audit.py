#!/usr/bin/env python3
"""Strict JSON gate for the P4.1 G2-Soft two-batch audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CANONICAL_SCHEMA_VERSION = "p41_g2_soft_audit_v1"
CANONICAL_CASE = "g2_soft_2batch_details"
NUMERIC_METRIC_CHECKS = {
    "target_mass_conservation_error": ("<=", 1e-6),
    "boundary_head_grad_norm": (">", 0.0),
    "boundary_adapter_grad_norm": (">", 0.0),
    "gamma_boundary_grad_abs": (">", 0.0),
    "boundary_delta_norm": (">", 0.0),
    "boundary_injection_ratio": (">", 0.0),
    "boundary_prediction_std": (">", 1e-6),
}
BOOLEAN_METRIC = "boundary_prediction_all_constant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--audit-exit-code", type=int, default=0)
    return parser.parse_args()


def is_finite_json_number(value: Any) -> bool:
    """Return true only for a finite JSON number; bool and strings are invalid."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def json_safe(value: Any) -> Any:
    """Make failed observations serializable as standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def extract_metrics(payload: dict[str, Any], failures: list[str]) -> tuple[str, dict[str, Any]]:
    schema_version = payload.get("schema_version")
    if schema_version == CANONICAL_SCHEMA_VERSION:
        source = "canonical.metrics"
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            failures.append(f"metrics={metrics!r}, expected object")
            return source, {}
        return source, metrics
    if schema_version is None:
        source = "legacy.summary"
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            failures.append(f"legacy summary={summary!r}, expected object")
            return source, {}
        return source, summary
    failures.append(
        f"schema_version={schema_version!r}, expected {CANONICAL_SCHEMA_VERSION!r} or legacy without version"
    )
    return "unknown", {}


def validate_payload(payload: Any, audit_exit_code: int = 0) -> dict[str, Any]:
    failures: list[str] = []
    observed: dict[str, Any] = {"audit_exit_code": audit_exit_code}
    if not isinstance(payload, dict):
        failures.append(f"audit_payload_type={type(payload).__name__}, expected object")
        payload = {}

    metrics_source, metrics = extract_metrics(payload, failures)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    optimizer = payload.get("optimizer") if isinstance(payload.get("optimizer"), dict) else {}
    observed.update({
        "schema_version": payload.get("schema_version"),
        "metrics_source": metrics_source,
        "audit_result": payload.get("result"),
        "audit_failures": payload.get("failures"),
        "case": payload.get("case"),
        "mode": payload.get("mode"),
        "branch": payload.get("branch"),
        "target_mode": payload.get("spatial_boundary_target_mode"),
        "batch_count": payload.get("batch_count"),
        "metrics": metrics,
        "summary": summary,
        "optimizer": optimizer,
    })

    if audit_exit_code != 0:
        failures.append(f"audit_process_exit_code={audit_exit_code}")
    if payload.get("case") != CANONICAL_CASE:
        failures.append(f"case={payload.get('case')!r}, expected {CANONICAL_CASE!r}")
    if payload.get("result") != "PASS":
        failures.append(f"audit_result={payload.get('result')!r}")
    audit_failures = payload.get("failures")
    if not isinstance(audit_failures, list):
        failures.append(f"audit_failures={audit_failures!r}, expected list")
    elif audit_failures:
        failures.append(f"audit_failures must be empty for PASS: {audit_failures!r}")
    if payload.get("mode") != "guidance":
        failures.append(f"spatial_sb_mode={payload.get('mode')!r}, expected 'guidance'")
    if payload.get("branch") != "boundary":
        failures.append(f"spatial_sb_branch={payload.get('branch')!r}, expected 'boundary'")
    if payload.get("spatial_boundary_target_mode") != "direct_area_soft":
        failures.append(
            f"spatial_boundary_target_mode={payload.get('spatial_boundary_target_mode')!r}, "
            "expected 'direct_area_soft'"
        )
    if payload.get("batch_count") != 2 or isinstance(payload.get("batch_count"), bool):
        failures.append(f"batch_count={payload.get('batch_count')!r}, expected integer 2")

    boolean_checks = {
        "loss_finite": summary.get("loss_finite"),
        "target_range_valid": summary.get("target_range_valid", summary.get("target_valid")),
    }
    for name, value in boolean_checks.items():
        observed[name] = value
        if value is not True:
            failures.append(f"{name}={value!r}")

    optimizer_checks = {
        "optimizer_duplicate_parameter_count": optimizer.get("duplicate_parameter_count"),
        "optimizer_missing_parameter_count": optimizer.get("trainable_missing_count"),
    }
    for name, value in optimizer_checks.items():
        observed[name] = value
        if not is_finite_json_number(value):
            failures.append(f"{name}={value!r} is not a finite JSON number")
        elif value != 0:
            failures.append(f"{name}={value!r} != 0")

    for name, (operator, threshold) in NUMERIC_METRIC_CHECKS.items():
        value = metrics.get(name)
        observed[name] = value
        if not is_finite_json_number(value):
            failures.append(f"{name}={value!r} is not a finite JSON number")
        elif operator == "<=" and value > threshold:
            failures.append(f"{name}={value!r} > {threshold}")
        elif operator == ">" and value <= threshold:
            failures.append(f"{name}={value!r} <= {threshold}")

    all_constant = metrics.get(BOOLEAN_METRIC)
    observed[BOOLEAN_METRIC] = all_constant
    if not isinstance(all_constant, bool):
        failures.append(f"{BOOLEAN_METRIC}={all_constant!r} is not a JSON boolean")
    elif all_constant:
        failures.append(f"{BOOLEAN_METRIC}=True")

    return {
        "schema_version": "p41_g2_soft_audit_gate_v1",
        "passed": not failures,
        "failures": failures,
        "observed": json_safe(observed),
    }


def main() -> int:
    args = parse_args()
    try:
        payload: Any = json.loads(args.audit_json.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = {}
        result = validate_payload(payload, args.audit_exit_code)
        result["failures"].insert(0, f"audit_json_unreadable: {type(exc).__name__}: {exc}")
        result["passed"] = False
    else:
        result = validate_payload(payload, args.audit_exit_code)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if result["failures"]:
        print("AUDIT_FAILED")
        for failure in result["failures"]:
            print(f"- {failure}")
        return 1
    print("AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
