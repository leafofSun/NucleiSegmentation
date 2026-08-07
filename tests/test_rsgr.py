from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from segment_anything.modeling.rsgr import (
    BOUNDARY_ATTR_NAMES,
    DERANGEMENTS,
    STRUCTURE_ATTR_NAMES,
    FixedOverlappingRegionLayout,
    RegionSemanticGrounding,
    RegionSemanticMapBuilder,
    apply_region_permutation,
    bounded_residual,
    checkpoint_compatibility_report,
    deterministic_derangement,
    load_prototype_banks,
    optimizer_group_spec,
    parameter_name_hash,
    parameter_name_shape_hash,
    soft_prototype_mixture,
    statistics_matched_random_bank,
)
from training.local_region_text_alignment import (
    ATTRIBUTE_NAMES as L1A_ATTRIBUTE_NAMES,
    compute_local_region_targets,
)
from training.rsgr_local5 import (
    attributes_for_group,
    load_local5_schema,
    local5_classification_stats,
    split_local5_labels,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "workdir/text_banks/rsgr_local5_conch_v1.pt"
META = ROOT / "workdir/text_banks/rsgr_local5_conch_v1.metadata.json"


def banks(dim: int = 32):
    generator = torch.Generator().manual_seed(7)
    structure = torch.nn.functional.normalize(torch.randn(3, 3, dim, generator=generator), dim=-1)
    boundary = torch.nn.functional.normalize(torch.randn(2, 3, dim, generator=generator), dim=-1)
    return structure, boundary


def module(mode: str = "correct_local"):
    structure, boundary = banks()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return RegionSemanticGrounding(
            8, 5, 4, structure, boundary, mode=mode,
            injection_scale=0.05, max_injection_ratio=0.02, random_seed=42,
        )


class TestRSGRLocal5(unittest.TestCase):
    def test_01_structure_logits_shape(self):
        self.assertEqual(module()(torch.randn(2, 8, 32, 32))["local_structure_logits"].shape, (2, 4, 3, 3))

    def test_02_boundary_logits_shape(self):
        self.assertEqual(module()(torch.randn(2, 8, 32, 32))["local_boundary_logits"].shape, (2, 4, 2, 3))

    def test_03_schema_order(self):
        schema = load_local5_schema()
        self.assertEqual(tuple(x["name"] for x in attributes_for_group(schema, "structure")), STRUCTURE_ATTR_NAMES)
        self.assertEqual(tuple(x["name"] for x in attributes_for_group(schema, "boundary")), BOUNDARY_ATTR_NAMES)

    def test_04_undefined_attributes_excluded(self):
        names = set(STRUCTURE_ATTR_NAMES + BOUNDARY_ATTR_NAMES)
        self.assertTrue(names.isdisjoint({"nuclear_area_fraction", "mean_nuclear_size", "boundary_density", "small_nuclei_ratio"}))

    def test_05_label_generator_and_split_are_five_only(self):
        schema = load_local5_schema()
        thresholds = {
            row["label_source_name"]: {
                "lower": row["threshold_values"]["low_upper_exclusive"],
                "upper": row["threshold_values"]["medium_upper_inclusive"],
            }
            for row in schema["attributes"]
        }
        mask = np.zeros((256, 256), dtype=np.int32)
        mask[30:50, 30:50] = 1
        mask[80:105, 90:115] = 2
        result = compute_local_region_targets(mask, thresholds)
        self.assertEqual(result["labels"].shape, (4, 5))
        labels = torch.from_numpy(result["labels"]).unsqueeze(0)
        structure, boundary = split_local5_labels(labels, L1A_ATTRIBUTE_NAMES, schema)
        self.assertEqual((structure.shape, boundary.shape), ((1, 4, 3), (1, 4, 2)))

    def test_06_formal_bank_shapes(self):
        structure, boundary = load_prototype_banks(BANK)
        self.assertEqual((structure.shape, boundary.shape), ((3, 3, 512), (2, 3, 512)))

    def test_07_metadata_schema_and_bank_hash(self):
        structure, boundary = load_prototype_banks(BANK, META)
        self.assertTrue(torch.isfinite(structure).all() and torch.isfinite(boundary).all())
        with tempfile.TemporaryDirectory() as directory:
            bad_meta = Path(directory) / "bank.metadata.json"
            payload = json.loads(META.read_text())
            payload["schema_sha256"] = "0" * 64
            bad_meta.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "schema SHA256"):
                load_prototype_banks(BANK, bad_meta)
            payload = json.loads(META.read_text())
            payload["class_names"] = ["high", "medium", "low"]
            bad_meta.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "level/class order"):
                load_prototype_banks(BANK, bad_meta)

    def test_08_soft_weighted_sum(self):
        probability = torch.tensor([[[[0.25, 0.25, 0.5]]]])
        bank = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]])
        self.assertTrue(torch.allclose(soft_prototype_mixture(probability, bank), torch.tensor([[[3.5, 2.5]]])))

    def test_09_correct_shuffle_semantic_multiset(self):
        correct, shuffled = module("correct_local"), module("shuffled_region")
        shuffled.load_state_dict(correct.state_dict())
        value = torch.randn(2, 8, 32, 32)
        a, b = correct(value), shuffled(value)
        self.assertTrue(torch.allclose(a["structure_semantics"].sort(dim=1).values, b["structure_semantics"].sort(dim=1).values))
        self.assertTrue(torch.allclose(a["boundary_semantics"].sort(dim=1).values, b["boundary_semantics"].sort(dim=1).values))

    def test_10_shuffle_has_no_fixed_point(self):
        permutation = deterministic_derangement(9, 42)
        self.assertFalse(torch.any(permutation == torch.arange(4).expand_as(permutation)))
        self.assertTrue(all(tuple(row.tolist()) in DERANGEMENTS for row in permutation))

    def test_11_random_bank_norm_and_statistics(self):
        reference = load_prototype_banks(BANK)[0]
        random = statistics_matched_random_bank(reference, 42)
        self.assertTrue(torch.allclose(random.norm(dim=-1), reference.norm(dim=-1), atol=1e-6, rtol=0))
        self.assertLess(abs(float(random.mean() - reference.mean())), 0.01)
        self.assertLess(abs(float(random.std() - reference.std())), 0.01)

    def test_12_random_seed_contract(self):
        reference = banks()[0]
        self.assertTrue(torch.equal(statistics_matched_random_bank(reference, 3), statistics_matched_random_bank(reference, 3)))
        self.assertFalse(torch.equal(statistics_matched_random_bank(reference, 3), statistics_matched_random_bank(reference, 4)))

    def test_13_normalized_overlap(self):
        layout, builder = FixedOverlappingRegionLayout(), RegionSemanticMapBuilder()
        _, reciprocal = builder(torch.ones(1, 4, 1), 32, 32)
        summed = torch.zeros(1, 1, 32, 32)
        for x0, y0, x1, y1 in layout.coordinates_for_size(32):
            summed[..., y0:y1, x0:x1] += reciprocal[..., y0:y1, x0:x1]
        self.assertTrue(torch.allclose(summed, torch.ones_like(summed)))

    def test_14_bounded_residual(self):
        delta, ratio = bounded_residual(torch.full((2, 5, 8, 8), 100.0), torch.randn(2, 8, 8, 8), 0.02, 0.05)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(float(ratio.max()), 0.02 * 0.05 + 1e-7)

    def test_15_no_local_zero_but_predicts(self):
        output = module("no_local")(torch.randn(2, 8, 32, 32))
        self.assertEqual(torch.count_nonzero(output["structure_delta"]), 0)
        self.assertEqual(output["local_structure_logits"].shape, (2, 4, 3, 3))

    def test_16_correct_injection_nonzero(self):
        output = module("correct_local")(torch.randn(2, 8, 32, 32))
        self.assertGreater(torch.count_nonzero(output["structure_delta"]), 0)
        self.assertGreater(torch.count_nonzero(output["boundary_delta"]), 0)

    def test_17_parameter_name_shape_count_parity(self):
        models = [module(mode) for mode in ("no_local", "correct_local", "shuffled_region", "random_prototype")]
        self.assertEqual(len({parameter_name_hash(item) for item in models}), 1)
        self.assertEqual(len({parameter_name_shape_hash(item) for item in models}), 1)
        self.assertEqual(len({sum(p.numel() for p in item.parameters() if p.requires_grad) for item in models}), 1)

    def test_18_optimizer_parity_and_prototypes_excluded(self):
        specs = [optimizer_group_spec(module(mode), 1e-4) for mode in ("no_local", "correct_local", "shuffled_region", "random_prototype")]
        self.assertEqual(len({digest for _, digest in specs}), 1)
        self.assertFalse(any("prototype" in name for spec, _ in specs for row in spec for name in row["parameters"]))

    def test_19_prototypes_frozen(self):
        model = module()
        output = model(torch.randn(1, 8, 32, 32, requires_grad=True))
        output["structure_delta"].sum().backward()
        self.assertFalse(model.structure_prototypes.requires_grad)
        self.assertIsNone(model.structure_prototypes.grad)

    def test_20_eval_gt_guards(self):
        model = module().eval()
        with self.assertRaisesRegex(RuntimeError, "RSGR_GT_GUARD"):
            model(torch.randn(1, 8, 32, 32), local_structure_labels=torch.zeros(1, 4, 3, dtype=torch.long))
        with self.assertRaisesRegex(RuntimeError, "RSGR_GT_GUARD"):
            model(torch.randn(1, 8, 32, 32), gt_mask=torch.zeros(1, 256, 256))

    def test_21_cpu_forward_backward_finite(self):
        model = module().train()
        value = torch.randn(2, 8, 32, 32, requires_grad=True)
        output = model(value, local_structure_labels=torch.randint(0, 3, (2, 4, 3)), local_boundary_labels=torch.randint(0, 3, (2, 4, 2)))
        loss = output["structure_delta"].square().mean() + output["boundary_delta"].square().mean() + output["local_structure_loss"] + output["local_boundary_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(value.grad).all())
        self.assertEqual(output["diagnostics"]["local5_structure_metrics"]["valid_count"], 24)

    def test_22_stable_parameter_namespace(self):
        names = set(dict(module().named_parameters()))
        self.assertIn("local5_structure_predictor.weight", names)
        self.assertIn("local5_boundary_predictor.weight", names)

    def test_23_exp5_key_compatibility_helper(self):
        report = checkpoint_compatibility_report(["image.weight", "rsgr.local5_structure_predictor.weight"], ["image.weight"])
        self.assertEqual(report["rsgr_missing"], ["rsgr.local5_structure_predictor.weight"])
        self.assertEqual(report["non_rsgr_missing"], [])
        self.assertEqual(report["non_rsgr_unexpected"], [])

    def test_24_missing_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "bank.pt"
            missing.write_bytes(BANK.read_bytes())
            with self.assertRaises(FileNotFoundError):
                load_prototype_banks(missing)

    def test_25_no_random_fallback_and_cuda_uninitialized(self):
        source = (ROOT / "segment_anything/modeling/sam.py").read_text()
        self.assertIn("all RSGR Local-5 modes require --rsgr_prototype_path and metadata", source)
        self.assertNotIn("deterministic_synthetic_reference", source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
