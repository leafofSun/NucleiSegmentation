#!/usr/bin/env python3
"""Real CPU-only Exp5 parent checkpoint dry-load for RSGR Local-5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything import sam_model_registry
from segment_anything.modeling.rsgr import checkpoint_compatibility_report
from training.rsgr_local5 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="workdir/models/exp5_numeric_attr_route_10ep_reinit1e4_v1/best_pq_model.pth",
    )
    parser.add_argument("--prototype-bank", default="workdir/text_banks/rsgr_local5_conch_v1.pt")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    expected_sha = "3543568e4fedecdfecc8dd76c2009ddae71f8b33aa7c1fc593dd92cb58641c50"
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise ValueError("canonical Exp5 best-PQ checkpoint SHA256 mismatch")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = checkpoint["model"]
    saved_args = dict(checkpoint["args"])
    saved_args.update({
        "checkpoint": None,
        "sam_checkpoint": None,
        "enable_rsgr": True,
        "rsgr_mode": "correct_local",
        "rsgr_prototype_source": "conch",
        "rsgr_prototype_path": str(args.prototype_bank),
        "rsgr_num_regions": 4,
        "rsgr_region_size": 192,
        "rsgr_prototype_detach": True,
        "rsgr_attr_detach": False,
        "rsgr_shuffle_scope": "within_sample",
        "rsgr_random_seed": 42,
        "rsgr_overlap_blend": "normalized",
        "spatial_sb_mode": "none",
        "enable_conch_text_encoder": False,
        "use_checkpoint_text_bank_without_conch": False,
    })
    model = sam_model_registry[saved_args["model_type"]](SimpleNamespace(**saved_args))
    compatibility = checkpoint_compatibility_report(model.state_dict().keys(), state.keys())
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    expected_empty = {
        key for key in ("_structure_text_bank_buffer", "_boundary_text_bank_buffer")
        if key in model.state_dict() and model.state_dict()[key].numel() == 0
    }
    non_rsgr_missing = [
        key for key in missing if not key.startswith("rsgr.") and key not in expected_empty
    ]
    non_rsgr_unexpected = [key for key in unexpected if not key.startswith("rsgr.")]
    rsgr_missing = [key for key in missing if key.startswith("rsgr.")]
    if non_rsgr_missing or non_rsgr_unexpected:
        raise RuntimeError(
            f"Exp5 compatibility failure: missing={non_rsgr_missing}, "
            f"unexpected={non_rsgr_unexpected}"
        )
    result = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_sha,
        "checkpoint_size": checkpoint_path.stat().st_size,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_pq": checkpoint.get("best_pq"),
        "checkpoint_state_keys": len(state),
        "load_map_location": "cpu",
        "mmap": True,
        "rsgr_enabled": bool(model.enable_rsgr),
        "rsgr_missing_count": len(rsgr_missing),
        "rsgr_missing_keys": rsgr_missing,
        "non_rsgr_missing_keys": non_rsgr_missing,
        "non_rsgr_unexpected_keys": non_rsgr_unexpected,
        "compatibility_helper": compatibility,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "checkpoint_written": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
