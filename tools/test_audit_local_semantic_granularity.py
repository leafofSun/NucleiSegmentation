#!/usr/bin/env python3
"""CPU-only tests for the L0 local semantic granularity audit."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from tools import audit_local_semantic_granularity as l0


def synthetic_image(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = 100 + (x % 40)
    image[..., 1] = 60 + (y % 50)
    image[..., 2] = 130
    return image


def window_inputs(image: np.ndarray, mask: np.ndarray):
    props = l0.extract_properties(image, mask, {})
    fg = (mask > 0).astype(np.float64)
    stain = props["stain"].astype(np.float64) * fg
    return (
        props,
        l0.integral_image(fg),
        l0.integral_image(stain),
        l0.integral_image(stain * props["stain"].astype(np.float64)),
    )


class L0AuditTests(unittest.TestCase):
    def test_single_instance_mask(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[20:40, 22:42] = 1
        props = l0.extract_properties(synthetic_image(), mask, {1: 2})
        self.assertEqual(props["labels"].tolist(), [1])
        self.assertGreater(props["areas"][0], 0)
        self.assertTrue(math.isfinite(props["irregularity"][0]))

    def test_two_touching_instances(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[20:40, 10:30] = 1
        mask[20:40, 30:50] = 2
        pairs = l0.touching_pairs_from_mask(mask, radius=2)
        self.assertIn((1, 2), pairs)
        props = l0.extract_properties(synthetic_image(), mask, {})
        self.assertEqual(props["touching_counts"].tolist(), [1, 1])

    def test_window_cut_instance(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[16:40, 20:44] = 1
        inputs = window_inputs(synthetic_image(), mask)
        _, complete, centroid, crossing = l0.attributes_for_window(*inputs, 0, 0, 32, 32)
        self.assertEqual(int(complete.sum()), 0)
        self.assertEqual(int(centroid.sum()), 1)
        self.assertEqual(int(crossing.sum()), 1)

    def test_empty_window(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[40:55, 40:55] = 1
        inputs = window_inputs(synthetic_image(), mask)
        values, complete, centroid, _ = l0.attributes_for_window(*inputs, 0, 0, 24, 24)
        self.assertEqual(int(complete.sum()), 0)
        self.assertEqual(int(centroid.sum()), 0)
        self.assertEqual(values["nuclear_area_fraction"], 0.0)
        self.assertEqual(values["nuclear_density"], 0.0)

    def test_border_touching_instance(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[0:20, 15:35] = 1
        props = l0.extract_properties(synthetic_image(), mask, {})
        self.assertTrue(bool(props["partial"][0]))

    def test_complete_instance_policy(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[5:20, 5:20] = 1
        mask[20:45, 20:45] = 2
        inputs = window_inputs(synthetic_image(), mask)
        _, complete, _, _ = l0.attributes_for_window(*inputs, 0, 0, 32, 32)
        self.assertEqual(int(complete.sum()), 1)
        selected_label = inputs[0]["labels"][np.flatnonzero(complete)[0]]
        self.assertEqual(int(selected_label), 1)

    def test_centroid_inside_policy(self) -> None:
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:42, 10:42] = 1
        inputs = window_inputs(synthetic_image(), mask)
        values, complete, centroid, crossing = l0.attributes_for_window(*inputs, 0, 0, 32, 32)
        self.assertEqual(int(complete.sum()), 0)
        self.assertEqual(int(centroid.sum()), 1)
        self.assertEqual(int(crossing.sum()), 1)
        self.assertGreater(values["centroid_nuclear_density"], values["nuclear_density"])

    def test_category_bin_assignment(self) -> None:
        values = np.array([-1.0, 1.0, 2.0, 2.1, np.nan])
        assigned = l0.assign_bin(values, 1.0, 2.0)
        self.assertEqual(assigned.tolist(), [0, 1, 1, 2, -1])
        self.assertEqual(l0.assign_bin(3.0, 1.0, 2.0), 2)

    def test_bootstrap_output_finite(self) -> None:
        lower, upper = l0.bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], repeats=200, seed=42)
        self.assertTrue(math.isfinite(lower))
        self.assertTrue(math.isfinite(upper))
        self.assertLessEqual(lower, upper)

    def test_output_json_has_no_nan_or_infinity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l0_json_test_") as tmp:
            path = Path(tmp) / "output.json"
            l0.write_json(path, {"nan": float("nan"), "inf": float("inf"), "ok": 1.0})
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            payload = json.loads(
                raw,
                parse_constant=lambda value: self.fail(f"non-standard constant {value}"),
            )
            self.assertIsNone(payload["nan"])
            self.assertIsNone(payload["inf"])

    def test_feature_grid_alignment_and_overlap(self) -> None:
        candidates = l0.build_window_candidates(14.0, 256, grid_step=8)
        self.assertTrue(all(row["window_size"] % 8 == 0 for row in candidates))
        positions = l0.window_positions(256, 64, 0.5, grid_step=8)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 192)
        self.assertTrue(all(position % 8 == 0 for position in positions))

    def test_cuda_and_torch_not_initialized(self) -> None:
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
