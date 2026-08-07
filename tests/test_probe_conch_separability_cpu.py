"""Lightweight CPU-only tests for the pre-registered CONCH geometry probe."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_probes.probe_conch_separability import (
    GLOBAL27_BOUNDARY_ORDER,
    GLOBAL27_STRUCTURE_ORDER,
    LEVEL_ORDER,
    LITERAL_PROMPTS,
    LOCAL5_ATTRIBUTE_ORDER,
    PROJECT_ROOT,
    aggregate_raw_embeddings,
    compute_metrics,
    evaluate_criteria,
    load_global27,
    load_preencoded_embeddings,
    load_set_a,
    load_set_b,
    run_probe,
    transform_variant,
)


SCHEMA = PROJECT_ROOT / "training/rsgr_local5_schema.json"


class ProbeConchSeparabilityCpuTests(unittest.TestCase):
    def test_registered_prompt_sets_and_set_b_aggregation(self) -> None:
        set_a = load_set_a(SCHEMA)
        set_b = load_set_b(SCHEMA)
        self.assertEqual(set_a.attribute_names, LOCAL5_ATTRIBUTE_ORDER)
        self.assertEqual(len(set_a.raw_prompt_texts), 15)
        self.assertEqual(len(set_b.raw_prompt_texts), 60)
        self.assertTrue(all(len(group) == 4 for group in set_b.prototype_raw_indices))

        raw = np.zeros((60, 15), dtype=np.float64)
        for prototype_index, group in enumerate(set_b.prototype_raw_indices):
            raw[list(group), prototype_index] = 1.0
        raw_l2, prototypes = aggregate_raw_embeddings(set_b, raw)
        self.assertEqual(prototypes.shape, (15, 15))
        np.testing.assert_allclose(np.linalg.norm(raw_l2, axis=1), 1.0)
        np.testing.assert_allclose(prototypes, np.eye(15), atol=1e-12)

    def test_global27_is_strict_and_has_no_fallback(self) -> None:
        payload = {"structure_prompts": {}, "boundary_prompts": {}}
        for group_key, attributes in (
            ("structure_prompts", GLOBAL27_STRUCTURE_ORDER),
            ("boundary_prompts", GLOBAL27_BOUNDARY_ORDER),
        ):
            for attribute in attributes:
                payload[group_key][attribute] = {
                    level: f"{attribute} {level}" for level in LEVEL_ORDER
                }
                payload[group_key][attribute]["description"] = (
                    f"description of {attribute}"
                )
        payload["boundary_prompts"]["touching_or_crowding_difficulty"] = {
            "description": "intentionally outside the fixed nine",
            "low": "ignored low",
            "mid": "ignored mid",
            "high": "ignored high",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "global.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bundle = load_global27(path)
            self.assertEqual(len(bundle.raw_prompt_texts), 27)
            self.assertIn(
                "boundary_prompts.touching_or_crowding_difficulty",
                bundle.ignored_source_keys,
            )
            self.assertIn(
                "structure_prompts.nuclear_density.description",
                bundle.ignored_source_keys,
            )
            del payload["boundary_prompts"][GLOBAL27_BOUNDARY_ORDER[-1]]["high"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no fallback allowed"):
                load_global27(path)

    def test_all_six_variant_shapes_and_v4_contract(self) -> None:
        bundle = load_set_a(SCHEMA)
        rng = np.random.default_rng(17)
        base = rng.normal(size=(15, 12))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        for variant in ("V0", "V1", "V2_k1", "V2_k2", "V3", "V4"):
            transformed, extras = transform_variant(base, 5, variant)
            self.assertEqual(transformed.shape, base.shape)
            self.assertTrue(np.isfinite(transformed).all())
            metrics, arrays = compute_metrics(transformed, bundle, variant, extras)
            self.assertEqual(arrays["cosine_matrix"].shape, (15, 15))
            self.assertEqual(arrays["level_axis_cosine_matrix"].shape, (5, 5))
            self.assertEqual(len(metrics["D4"]["t_by_attribute"]), 5)
        v4, extras = transform_variant(base, 5, "V4")
        np.testing.assert_allclose(v4.reshape(5, 3, 12)[:, 1, :], 0.0)
        self.assertEqual(extras["v4_mid_residual_ratio"].shape, (5,))

    def test_metric_formulas_and_strict_threshold_boundaries(self) -> None:
        bundle = load_set_a(SCHEMA)
        matrix = np.zeros((15, 5), dtype=np.float64)
        for attribute_index in range(5):
            matrix[3 * attribute_index] = -np.eye(5)[attribute_index]
            matrix[3 * attribute_index + 2] = np.eye(5)[attribute_index]
        metrics, arrays = compute_metrics(matrix, bundle, "V4", {
            "v4_mid_residual_ratio": np.zeros(5, dtype=np.float64)
        })
        self.assertAlmostEqual(metrics["D3"]["level_axis_alignment"], 0.0)
        self.assertAlmostEqual(metrics["D4"]["monotonic_ratio"], 1.0)
        for value in metrics["D4"]["t_by_attribute"].values():
            self.assertAlmostEqual(value, 0.5, places=10)
        np.testing.assert_allclose(arrays["level_axis_cosine_matrix"], np.eye(5))

        flags = evaluate_criteria({
            "intra_attr_cos": 0.95,
            "eff_rank_95": 5,
            "level_axis_alignment": 0.90,
            "monotonic_ratio": 0.8,
            "separation": -1e-15,
        })
        self.assertEqual(flags, {name: False for name in ("C1", "C2", "C3", "C4", "C5")})
        self.assertTrue(evaluate_criteria({
            "intra_attr_cos": 0.0,
            "eff_rank_95": 5,
            "level_axis_alignment": 0.0,
            "monotonic_ratio": 1.0,
            "separation": 0.0,
        })["C5"])

    def test_preencoded_contract_rejects_prompt_reordering(self) -> None:
        bundle = load_set_a(SCHEMA)
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoded.npz"
            wrong_ids = list(bundle.raw_prompt_ids)
            wrong_ids[0], wrong_ids[1] = wrong_ids[1], wrong_ids[0]
            np.savez_compressed(
                path,
                prompt_embeddings=rng.normal(size=(15, 8)),
                prompt_ids=np.asarray(wrong_ids),
                prompt_texts=np.asarray(bundle.raw_prompt_texts),
                literal_embeddings=rng.normal(size=(2, 8)),
                literal_texts=np.asarray(LITERAL_PROMPTS),
            )
            with self.assertRaisesRegex(ValueError, "prompt_ids"):
                load_preencoded_embeddings(path, bundle)

    def test_offline_end_to_end_writes_all_audit_artifacts(self) -> None:
        bundle = load_set_a(SCHEMA)
        rng = np.random.default_rng(20260807)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoded = root / "encoded.npz"
            output = root / "result"
            np.savez_compressed(
                encoded,
                prompt_embeddings=rng.normal(size=(15, 9)),
                prompt_ids=np.asarray(bundle.raw_prompt_ids),
                prompt_texts=np.asarray(bundle.raw_prompt_texts),
                literal_embeddings=rng.normal(size=(2, 9)),
                literal_texts=np.asarray(LITERAL_PROMPTS),
                checkpoint_path=np.asarray("/offline/conch.bin"),
                checkpoint_sha256=np.asarray("NOT_FOUND"),
            )
            args = argparse.Namespace(
                prompt_set="A",
                variant="all",
                schema=str(SCHEMA),
                global27_templates=str(root / "NOT_FOUND.json"),
                output_dir=str(output),
                conch_checkpoint_path=None,
                conch_cache_path=None,
                device="cpu",
                hf_hub_offline=True,
                embeddings_input=str(encoded),
                write_encoding_request=None,
                freeze_dir=None,
            )
            summary = run_probe(args)
            self.assertTrue((output / "probe_config.json").is_file())
            self.assertTrue((output / "raw_embeddings.npz").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(len(summary["metrics_by_variant"]), 6)
            for variant in ("V0", "V1", "V2_k1", "V2_k2", "V3", "V4"):
                self.assertTrue((output / variant / "metrics.json").is_file())
                self.assertTrue((output / variant / "cosine_matrix.csv").is_file())
                self.assertTrue((output / variant / "cosine_heatmap.svg").is_file())
                self.assertTrue((output / variant / "intermediates.npz").is_file())


if __name__ == "__main__":
    unittest.main()
