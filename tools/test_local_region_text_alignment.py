import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from training.local_region_text_alignment import (
    ATTRIBUTE_NAMES,
    LocalRegionTextAlignment,
    assign_bin,
    compute_local_region_targets,
    feature_region_coordinates,
    maybe_compute_local_alignment,
    region_coordinates,
    validate_no_eval_gt,
)


THRESHOLDS = {
    name: {"lower": 0.2 if name == "nuclear_size_heterogeneity" else 1.0,
           "upper": 0.6 if name == "nuclear_size_heterogeneity" else 3.0}
    for name in ATTRIBUTE_NAMES
}


def rectangle(mask, instance_id, x0, y0, x1, y1):
    mask[y0:y1, x0:x1] = instance_id


class LocalRegionAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bank = torch.nn.functional.normalize(torch.randn(5, 3, 16), dim=-1)

    def test_four_region_coordinates(self):
        self.assertEqual(region_coordinates(), (
            (0, 0, 192, 192), (64, 0, 256, 192),
            (0, 64, 192, 256), (64, 64, 256, 256),
        ))

    def test_feature_mapping(self):
        self.assertEqual(feature_region_coordinates(), (
            (0, 0, 24, 24), (8, 0, 32, 24),
            (0, 8, 24, 32), (8, 8, 32, 32),
        ))

    def test_flip_recomputes_labels(self):
        mask = np.zeros((256, 256), np.int32)
        rectangle(mask, 1, 10, 30, 30, 50)
        before = compute_local_region_targets(mask, THRESHOLDS)
        after = compute_local_region_targets(np.fliplr(mask).copy(), THRESHOLDS)
        self.assertFalse(before["valid"][1, 0])
        self.assertTrue(after["valid"][1, 0])

    def test_rotation_recomputes_labels(self):
        mask = np.zeros((256, 256), np.int32)
        rectangle(mask, 1, 20, 20, 40, 40)
        before = compute_local_region_targets(mask, THRESHOLDS)
        after = compute_local_region_targets(np.rot90(mask).copy(), THRESHOLDS)
        self.assertTrue(before["valid"][0, 0])
        self.assertTrue(after["valid"][2, 0])

    def test_empty_region_validity(self):
        target = compute_local_region_targets(np.zeros((256, 256), np.int32), THRESHOLDS)
        self.assertFalse(target["valid"].any())
        self.assertTrue((target["labels"] == -1).all())

    def test_single_instance_validity(self):
        mask = np.zeros((256, 256), np.int32)
        rectangle(mask, 1, 20, 20, 40, 40)
        target = compute_local_region_targets(mask, THRESHOLDS)
        self.assertTrue(target["valid"][0, 0])
        self.assertFalse(target["valid"][0, 1])
        self.assertFalse(target["valid"][0, 2])
        self.assertTrue(target["valid"][0, 3])
        self.assertTrue(target["valid"][0, 4])

    def test_two_instance_validity(self):
        mask = np.zeros((256, 256), np.int32)
        rectangle(mask, 1, 20, 20, 40, 40)
        rectangle(mask, 2, 80, 80, 110, 110)
        target = compute_local_region_targets(mask, THRESHOLDS)
        self.assertTrue(target["valid"][0, 1])
        self.assertTrue(target["valid"][0, 2])

    def test_complete_only_excludes_cut_instance(self):
        mask = np.zeros((256, 256), np.int32)
        rectangle(mask, 1, 180, 20, 210, 50)
        target = compute_local_region_targets(mask, THRESHOLDS)
        self.assertEqual(int(target["complete_instance_count"][0]), 0)
        self.assertTrue(target["valid"][0, 0])
        self.assertFalse(target["valid"][0, 3])

    def test_five_attribute_bin_assignment(self):
        for name in ATTRIBUTE_NAMES:
            low = THRESHOLDS[name]["lower"]
            high = THRESHOLDS[name]["upper"]
            self.assertEqual(assign_bin(low - 0.01, low, high), 0)
            self.assertEqual(assign_bin((low + high) / 2, low, high), 1)
            self.assertEqual(assign_bin(high + 0.01, low, high), 2)

    def test_prototype_shape_and_normalization(self):
        module = LocalRegionTextAlignment(self.bank)
        self.assertEqual(tuple(module.text_prototypes.shape), (5, 3, 16))
        self.assertTrue(torch.allclose(
            module.text_prototypes.norm(dim=-1), torch.ones(5, 3), atol=1e-6
        ))

    def test_prototypes_frozen(self):
        module = LocalRegionTextAlignment(self.bank)
        self.assertFalse(module.text_prototypes.requires_grad)
        self.assertNotIn(id(module.text_prototypes), {id(p) for p in module.parameters()})

    def test_masked_ce_no_valid_is_finite(self):
        module = LocalRegionTextAlignment(self.bank)
        feat = torch.randn(1, 256, 32, 32, requires_grad=True)
        labels = torch.full((1, 4, 5), -1, dtype=torch.long)
        valid = torch.zeros((1, 4, 5), dtype=torch.bool)
        result = module(feat, labels, valid)
        self.assertTrue(torch.isfinite(result["local_region_text_loss"]))
        result["local_region_text_loss"].backward()

    def test_bool_not_numeric_label(self):
        with self.assertRaises(TypeError):
            assign_bin(True, 1.0, 2.0)
        module = LocalRegionTextAlignment(self.bank)
        with self.assertRaises(TypeError):
            module(torch.randn(1, 256, 32, 32),
                   torch.zeros(1, 4, 5, dtype=torch.bool),
                   torch.ones(1, 4, 5, dtype=torch.bool))

    def test_disabled_alignment_is_identity(self):
        feat = torch.randn(1, 256, 32, 32)
        clone = feat.clone()
        self.assertIsNone(maybe_compute_local_alignment(None, feat, None, None))
        self.assertTrue(torch.equal(feat, clone))

    def test_eval_rejects_gt_local_attributes(self):
        with self.assertRaises(RuntimeError):
            validate_no_eval_gt(False, [{"local_region_attr_labels": torch.zeros(4, 5)}])
        validate_no_eval_gt(False, [{"image": torch.zeros(3, 512, 512)}])

    def test_json_has_no_nan_or_infinity(self):
        payload = {"shape": [5, 3, 16], "finite_loss": 0.0}
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)


if __name__ == "__main__":
    unittest.main()
