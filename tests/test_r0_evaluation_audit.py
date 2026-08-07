import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluation_audit import (
    DistributedEvalSampler,
    EvaluationProtocol,
    MetricAccumulator,
    actual_delta_ratio,
    fixed_subset_indices,
    write_evaluation_protocol,
    write_run_manifests,
)


METRICS = ("dice", "iou", "mAJI", "mPQ")


def row(value):
    return {name: float(value) for name in METRICS}


class R0EvaluationAuditTests(unittest.TestCase):
    def test_duplicate_sample_id_removed_across_unequal_ranks(self):
        rank0 = MetricAccumulator(METRICS)
        rank1 = MetricAccumulator(METRICS)
        for sample_id, value in (("a", 1.0), ("c", 3.0), ("e", 5.0)):
            self.assertTrue(rank0.add(sample_id, row(value)))
        for sample_id, value in (("b", 2.0), ("d", 4.0), ("e", 999.0)):
            self.assertTrue(rank1.add(sample_id, row(value)))

        merged = MetricAccumulator.merge(METRICS, (rank0.records(), rank1.records()))
        self.assertEqual(merged.seen_before_dedup, 6)
        self.assertEqual(merged.unique_count, 5)
        self.assertEqual(merged.duplicates_removed, 1)
        for values in merged.sums_counts().values():
            self.assertEqual(values["count"], 5)
            self.assertEqual(values["sum"], 15.0)

    def test_single_process_equals_simulated_ddp_after_dedup(self):
        single = MetricAccumulator(METRICS)
        for sample_id, value in zip("abcde", (1, 2, 3, 4, 5)):
            single.add(sample_id, row(value))
        ddp = MetricAccumulator.merge(
            METRICS,
            (
                {"a": row(1), "c": row(3), "e": row(5)},
                {"b": row(2), "d": row(4), "e": row(999)},
            ),
        )
        self.assertEqual(single.sums_counts(), ddp.sums_counts())

    def test_no_padding_sampler_has_complete_disjoint_shards(self):
        shards = [list(DistributedEvalSampler(range(7), 3, rank)) for rank in range(3)]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(7)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [3, 2, 2])

    def test_fixed_subset_is_global_and_deterministic(self):
        first = fixed_subset_indices(11, 0.4, 42)
        second = fixed_subset_indices(11, 0.4, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_protocol_and_manifest_serialization(self):
        protocol = EvaluationProtocol(
            protocol_name="canonical_pannuke_v1",
            protocol_role="full_test",
            comparable_to_canonical_full_test=True,
            difference_from_canonical=[],
            mask_threshold=0.4,
            object_threshold=0.45,
            min_object_size=15,
            image_size=512,
            patch_size=256,
            sliding_overlap=0.8,
            semantic_mode="organ_static",
            FREQPATH_ABLATION="both",
            use_asr=True,
            asr_variant="freqpath",
            use_pnurl=False,
            use_sga_sb=False,
            use_pnudp_dense=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            run = Path(tmp) / "run"
            root.mkdir()
            (root / "metrics.py").write_text("# synthetic metric source\n", encoding="utf-8")
            protocol_path = write_evaluation_protocol(protocol, str(run))
            paths = write_run_manifests(
                run_dir=str(run),
                run_name="synthetic",
                args=Namespace(seed=42),
                protocol=protocol,
                project_root=str(root),
                parent_checkpoint=None,
                evaluation_context={
                    "data_split": "synthetic",
                    "sample_count": 5,
                    "unique_sample_count": 5,
                    "duplicate_sample_count": 0,
                    "world_size": 2,
                    "sampler_type": "DistributedEvalSampler(no_padding)",
                },
                source_files=("metrics.py",),
            )
            protocol_json = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
            manifest = json.loads(Path(paths["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(protocol_json["protocol_name"], "canonical_pannuke_v1")
            required = {
                "git_commit", "parent_checkpoint", "data_split", "sample_count",
                "unique_sample_count", "duplicate_sample_count", "world_size",
                "sampler_type", "metric_implementation_version", "evaluation_protocol",
                "semantic_inference_config", "seed",
            }
            self.assertFalse(required - set(manifest))
            self.assertEqual(manifest["unique_sample_count"], 5)
            self.assertTrue(Path(paths["source_manifest"]).is_file())

    def test_actual_delta_ratio_definition(self):
        self.assertAlmostEqual(actual_delta_ratio(2.0, 4.0), 0.5, places=10)


if __name__ == "__main__":
    unittest.main()
