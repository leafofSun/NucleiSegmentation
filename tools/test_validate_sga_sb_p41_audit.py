#!/usr/bin/env python3
"""CPU-only regression tests for the P4.1 audit schema contract."""

from __future__ import annotations

import copy
import math
import unittest

from tools import validate_sga_sb_p41_audit as gate


def complete_metrics() -> dict[str, object]:
    return {
        "target_mass_conservation_error": 0.0,
        "boundary_head_grad_norm": 0.3,
        "boundary_adapter_grad_norm": 0.01,
        "gamma_boundary_grad_abs": 0.0007,
        "boundary_delta_norm": 0.06,
        "boundary_injection_ratio": 0.0002,
        "boundary_prediction_std": 0.0001,
        "boundary_prediction_all_constant": False,
    }


def complete_payload(canonical: bool = True) -> dict[str, object]:
    metrics = complete_metrics()
    payload: dict[str, object] = {
        "case": gate.CANONICAL_CASE,
        "result": "PASS",
        "failures": [],
        "mode": "guidance",
        "branch": "boundary",
        "spatial_boundary_target_mode": "direct_area_soft",
        "batch_count": 2,
        "summary": {
            "loss_finite": True,
            "target_range_valid": True,
        },
        "optimizer": {
            "duplicate_parameter_count": 0,
            "trainable_missing_count": 0,
        },
    }
    if canonical:
        payload["schema_version"] = gate.CANONICAL_SCHEMA_VERSION
        payload["metrics"] = metrics
    else:
        payload["summary"].update(metrics)  # type: ignore[union-attr]
    return payload


class P41AuditSchemaTests(unittest.TestCase):
    def assert_gate_passes(self, payload: dict[str, object]) -> None:
        result = gate.validate_payload(payload)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["failures"], [])

    def assert_gate_fails(self, payload: dict[str, object]) -> None:
        result = gate.validate_payload(payload)
        self.assertFalse(result["passed"])
        self.assertTrue(result["failures"])

    def test_canonical_complete_pass(self) -> None:
        self.assert_gate_passes(complete_payload())

    def test_canonical_missing_one_metric_fails(self) -> None:
        payload = complete_payload()
        del payload["metrics"]["boundary_delta_norm"]  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_numeric_null_fails(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_head_grad_norm"] = None  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_numeric_nan_fails(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_head_grad_norm"] = math.nan  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_numeric_infinity_fails(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_head_grad_norm"] = math.inf  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_numeric_bool_true_fails(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_head_grad_norm"] = True  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_boolean_true_fails_semantic_check(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_prediction_all_constant"] = True  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_boolean_string_false_fails(self) -> None:
        payload = complete_payload()
        payload["metrics"]["boundary_prediction_all_constant"] = "False"  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_result_pass_but_metrics_missing_fails(self) -> None:
        payload = complete_payload()
        del payload["metrics"]
        self.assertEqual(payload["result"], "PASS")
        self.assert_gate_fails(payload)

    def test_result_pass_but_failures_nonempty_fails(self) -> None:
        payload = complete_payload()
        payload["failures"] = ["real audit failure"]
        self.assertEqual(payload["result"], "PASS")
        self.assert_gate_fails(payload)

    def test_complete_legacy_schema_passes(self) -> None:
        self.assert_gate_passes(complete_payload(canonical=False))

    def test_incomplete_legacy_schema_fails(self) -> None:
        payload = complete_payload(canonical=False)
        del payload["summary"]["boundary_adapter_grad_norm"]  # type: ignore[index]
        self.assert_gate_fails(payload)

    def test_input_payload_is_not_mutated(self) -> None:
        payload = complete_payload()
        before = copy.deepcopy(payload)
        gate.validate_payload(payload)
        self.assertEqual(payload, before)


if __name__ == "__main__":
    unittest.main()
