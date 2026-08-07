"""Two-rank CPU/Gloo smoke test for the R0 evaluation aggregation contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist

from evaluation_audit import DistributedEvalSampler, audit_unique_sample_ids


def _all_gather_object(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized(), "CUDA was initialized before the Gloo test"
    assert int(os.environ["WORLD_SIZE"]) == 2, "this smoke test requires exactly two ranks"

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    try:
        assert dist.get_backend() == "gloo"
        assert dist.get_world_size() == 2
        assert not torch.cuda.is_initialized(), "Gloo setup initialized CUDA"

        # Unequal shards (3 vs 2 samples): reduce sums/counts, never rank means.
        values = [1.0, 1.0, 9.0, 1.0, 9.0]
        local_indices = list(DistributedEvalSampler(range(len(values)), 2, rank))
        local_values = [values[index] for index in local_indices]
        totals = torch.tensor(
            [sum(local_values), float(len(local_values))], dtype=torch.float64
        )
        local_mean = sum(local_values) / len(local_values)
        rank_means = _all_gather_object(local_mean)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        global_macro = (totals[0] / totals[1]).item()
        assert totals.tolist() == [21.0, 5.0]
        assert abs(global_macro - 4.2) < 1e-12
        assert abs(sum(rank_means) / 2.0 - global_macro) > 1e-3
        if rank == 0:
            assert global_macro == 4.2

        # Non-divisible dataset: no padding and every stable ID appears once.
        local_ids = [
            f"stable-sample-{index}"
            for index in DistributedEvalSampler(range(7), 2, rank)
        ]
        ids_by_rank = _all_gather_object(local_ids)
        audit = audit_unique_sample_ids(ids_by_rank)
        assert audit == {
            "global_seen_before_dedup": 7,
            "global_unique": 7,
            "duplicate_sample_count": 0,
        }
        assert sorted(len(ids) for ids in ids_by_rank) == [3, 4]

        # Artificial cross-rank duplication must fail loudly, never silently dedup.
        duplicate_ids = [f"rank-{rank}-unique", "shared-duplicate"]
        duplicate_ids_by_rank = _all_gather_object(duplicate_ids)
        try:
            audit_unique_sample_ids(duplicate_ids_by_rank)
        except RuntimeError as error:
            message = str(error)
            assert "duplicate sample_id detected" in message
            assert "shared-duplicate" in message
        else:
            raise AssertionError("duplicate stable sample ID was silently accepted")

        # Only global rank zero is eligible to write a best-checkpoint artifact.
        save_best_eligible = rank == 0
        save_flags = _all_gather_object(save_best_eligible)
        assert save_flags == [True, False]
        if rank != 0:
            assert not save_best_eligible

        assert not torch.cuda.is_initialized(), "CPU assertions initialized CUDA"
        print(
            f"[R0_5_GLOO_PASS] rank={rank} local_count={len(local_values)} "
            f"global_count={int(totals[1].item())} global_macro={global_macro:.12f}",
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    assert not dist.is_initialized(), "process group was not destroyed"
    assert not torch.cuda.is_initialized(), "CUDA was initialized during the smoke test"


if __name__ == "__main__":
    main()
