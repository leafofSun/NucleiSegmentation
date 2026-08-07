#!/usr/bin/env python3
"""Safely extract scheduler-related metadata from registered NuSeg checkpoints."""

import argparse
import json
from pathlib import Path


CHECKPOINTS = {
    "visual_baseline": "workdir/models/Visual_baseline/best_model.pth",
    "exp5_best_pq": "workdir/models/exp5_numeric_attr_route_10ep_reinit1e4_v1/best_pq_model.pth",
    "p3_old_n0_latest": "workdir/models/sga_sb_p3_n0_seed42_e5/latest_model.pth",
}
ARG_KEYS = (
    "epochs", "start_epoch", "lr", "min_lr", "warmup_epochs",
    "weight_decay", "seed", "phase", "run_name",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/hy-tmp/NuSeg")
    parser.add_argument(
        "--output",
        default="workdir/audits/sga_sb_p3_20260713/P3_1_CHECKPOINT_METADATA_AUDIT.json",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    import torch

    records = {}
    for role, relative in CHECKPOINTS.items():
        path = root / relative
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        embedded_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(embedded_args, dict):
            embedded_args = {}
        scheduler = checkpoint.get("scheduler") if isinstance(checkpoint, dict) else None
        records[role] = {
            "path": relative,
            "top_level_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            "epoch_zero_based": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
            "args": {key: embedded_args.get(key) for key in ARG_KEYS},
            "optimizer_state_present": isinstance(checkpoint.get("optimizer"), dict),
            "scheduler_state_present": isinstance(scheduler, dict),
            "scheduler_state": scheduler if isinstance(scheduler, dict) else None,
        }
        del checkpoint

    payload = {
        "load_policy": "torch.load(weights_only=True, mmap=True); metadata only; no model construction",
        "checkpoints": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
