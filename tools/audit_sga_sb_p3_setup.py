#!/usr/bin/env python3
"""Constructor-only P3 policy and initialization audit (no forward/data/train)."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_SHA256 = "7321ee6652c53469d77e2e1af8c9d57e6772936ad12a2cea6a437c8421264f9c"
CASES = {
    "N0": ("none", "both"),
    "S1": ("supervision_only", "both"),
    "G1": ("guidance", "structure"),
    "G2": ("guidance", "boundary"),
    "G3": ("guidance", "both"),
}
SGA_PREFIXES = (
    "spatial_structure_head.", "spatial_boundary_head.",
    "spatial_structure_adapter.", "spatial_boundary_adapter.",
    "gamma_structure", "gamma_boundary",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(model, prefix):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        if name == prefix or name.startswith(prefix + "."):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
            count += value.numel()
    return digest.hexdigest() if count else None, count


def module_max_abs(model, prefix):
    values = [
        tensor.detach().float().abs().max().item()
        for name, tensor in model.state_dict().items()
        if name == prefix or name.startswith(prefix + ".")
    ]
    return max(values) if values else None


def configure_args(formal, checkpoint, mode, branch):
    saved = sys.argv[:]
    try:
        sys.argv = ["train.py"]
        args = formal.parse_args()
    finally:
        sys.argv = saved
    args.phase = "vision"
    args.device = "cpu"
    args.seed = 42
    args.sam_checkpoint = str(checkpoint)
    args.checkpoint = str(checkpoint)
    args.prompt_mode = args.eval_prompt_mode = "base"
    args.use_asr = True
    args.asr_variant = "freqpath"
    args.use_pnurl = False
    args.use_coop = False
    args.enable_conch_text_encoder = False
    args.disable_conch_text_encoder = True
    args.clip_text_encoder = False
    args.enable_attr_text_alignment = False
    args.disable_attr_text_alignment = True
    args.enable_promptnu_lite_align = False
    args.disable_promptnu_lite_align = True
    args.enable_promptnu_guided_v3 = False
    args.disable_promptnu_guided_v3 = True
    args.enable_pnudp_dense_train = False
    args.enable_numeric_attr_freqpath_guidance = False
    args.enable_multilevel_attr_heads = False
    args.spatial_instance_attr_mode = "none"
    args.spatial_sb_mode = mode
    args.spatial_sb_branch = branch
    args.spatial_structure_loss_weight = 0.1
    args.spatial_boundary_loss_weight = 0.1
    args.spatial_structure_guidance_init = 0.05
    args.spatial_boundary_guidance_init = 0.05
    args.lr = 1.0e-4
    args.weight_decay = 1.0e-4
    return args


def classify(name):
    if not name.startswith(SGA_PREFIXES):
        return "non_sga"
    if "_head." in name:
        return "head"
    if "_adapter." in name:
        return "adapter"
    if name.startswith("gamma_"):
        return "gamma"
    raise ValueError(f"Unclassified SGA-SB parameter: {name}")


def write_reports(result, out_dir):
    rows = result["cases"]
    policy = [
        "# P3 Trainable Policy Comparison", "",
        f"Result: **{result['result']}**", "",
        "| Case | Total trainable | Non-SGA | SGA head | SGA adapter | SGA gamma | Optimizer groups (lr) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in CASES:
        row = rows[case]
        groups = ", ".join(f"{g['name']} ({g['lr']:.6g})" for g in row["optimizer_groups"])
        policy.append(
            f"| {case} | {row['counts']['total']} | {row['counts']['non_sga']} | "
            f"{row['counts']['head']} | {row['counts']['adapter']} | {row['counts']['gamma']} | {groups} |"
        )
    policy += ["", f"Five-case non-SGA parameter-name sets identical: **{str(result['checks']['non_sga_sets_identical']).upper()}**."]
    (out_dir / "P3_TRAINABLE_POLICY_COMPARISON.md").write_text("\n".join(policy) + "\n", encoding="utf-8")

    init = [
        "# P3 Initialization Audit", "",
        "Only constructor-time tensors were compared; no forward or optimizer step was executed.", "",
        "| Required comparison | Equal | Left hash/value | Right hash/value |",
        "|---|---|---|---|",
    ]
    for check in result["initialization_checks"]:
        init.append(f"| {check['name']} | {str(check['equal']).upper()} | `{check['left']}` | `{check['right']}` |")
    init += ["", f"Overall initialization equality: **{str(result['checks']['initialization_identical']).upper()}**."]
    (out_dir / "P3_INITIALIZATION_AUDIT.md").write_text("\n".join(init) + "\n", encoding="utf-8")


def verify(manifest, checkpoint):
    if not manifest.is_file():
        raise SystemExit(f"P3 preflight manifest missing: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    actual = sha256_file(checkpoint)
    if data.get("result") != "PASS" or actual != EXPECTED_SHA256:
        raise SystemExit(f"P3 preflight verification failed: result={data.get('result')} sha256={actual}")
    print(f"P3 preflight PASS; parent_sha256={actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/hy-tmp/NuSeg")
    parser.add_argument("--checkpoint", default="workdir/models/Visual_baseline/best_model.pth")
    parser.add_argument("--output-dir", default="workdir/audits/sga_sb_p3_20260713")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    manifest = out_dir / "P3_PREFLIGHT_MANIFEST.json"
    if args.verify_only:
        verify(manifest, checkpoint)
        return

    actual_sha = sha256_file(checkpoint)
    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(f"Parent checkpoint SHA-256 mismatch: expected={EXPECTED_SHA256} actual={actual_sha}")
    sys.path.insert(0, str(root))
    import torch
    import train as formal
    from segment_anything import sam_model_registry

    case_rows = {}
    for case, (mode, branch) in CASES.items():
        formal.setup_seed(42)
        cfg = configure_args(formal, checkpoint, mode, branch)
        model = sam_model_registry[cfg.model_type](cfg)
        formal.apply_stage_policy(model, cfg.phase, args=cfg, logger=None, rank=0)
        optimizer = formal.build_optimizer_by_stage(model, cfg.phase, args=cfg, logger=None, rank=0, audit=None)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        non_sga = sorted(name for name in trainable if not name.startswith(SGA_PREFIXES))
        counts = {"total": 0, "non_sga": 0, "head": 0, "adapter": 0, "gamma": 0}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            counts["total"] += param.numel()
            counts[classify(name)] += param.numel()
        hashes = {}
        for prefix in ("spatial_structure_head", "spatial_boundary_head", "spatial_structure_adapter", "spatial_boundary_adapter", "gamma_structure", "gamma_boundary"):
            hashes[prefix], _ = tensor_hash(model, prefix)
        values = {
            name: (float(getattr(model, name).detach().cpu().item()) if getattr(model, name, None) is not None else None)
            for name in ("gamma_structure", "gamma_boundary")
        }
        zero_init = {
            "spatial_structure_adapter_final": module_max_abs(model, "spatial_structure_adapter.net.3"),
            "spatial_boundary_adapter_final": module_max_abs(model, "spatial_boundary_adapter.net.3"),
        }
        case_rows[case] = {
            "mode": mode, "branch": branch, "counts": counts,
            "non_sga_trainable_names": non_sga,
            "optimizer_groups": [{"name": g.get("name", "unnamed"), "lr": float(g["lr"]), "numel": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
            "initial_hashes": hashes, "initial_values": values,
            "zero_init_max_abs": zero_init,
        }
        del optimizer, model
        gc.collect()

    baseline_set = case_rows["N0"]["non_sga_trainable_names"]
    non_sga_equal = all(row["non_sga_trainable_names"] == baseline_set for row in case_rows.values())
    comparisons = [
        ("G1 structure head == G3 structure head", "G1", "G3", "spatial_structure_head", "initial_hashes"),
        ("G1 structure adapter == G3 structure adapter", "G1", "G3", "spatial_structure_adapter", "initial_hashes"),
        ("G1 gamma_structure == G3 gamma_structure", "G1", "G3", "gamma_structure", "initial_values"),
        ("G2 boundary head == G3 boundary head", "G2", "G3", "spatial_boundary_head", "initial_hashes"),
        ("G2 boundary adapter == G3 boundary adapter", "G2", "G3", "spatial_boundary_adapter", "initial_hashes"),
        ("G2 gamma_boundary == G3 gamma_boundary", "G2", "G3", "gamma_boundary", "initial_values"),
        ("G1 structure adapter final projection zero-init", "G1", "G1", "spatial_structure_adapter_final", "zero_init_max_abs"),
        ("G3 structure adapter final projection zero-init", "G3", "G3", "spatial_structure_adapter_final", "zero_init_max_abs"),
        ("G2 boundary adapter final projection zero-init", "G2", "G2", "spatial_boundary_adapter_final", "zero_init_max_abs"),
        ("G3 boundary adapter final projection zero-init", "G3", "G3", "spatial_boundary_adapter_final", "zero_init_max_abs"),
    ]
    init_checks = []
    for name, left_case, right_case, key, source in comparisons:
        left = case_rows[left_case][source][key]
        right = case_rows[right_case][source][key]
        equal = left is not None and left == right
        if source == "zero_init_max_abs":
            equal = equal and left == 0.0
        init_checks.append({"name": name, "left": left, "right": right, "equal": equal})
    init_equal = all(item["equal"] for item in init_checks)
    result = {
        "result": "PASS" if non_sga_equal and init_equal else "FAIL",
        "parent_checkpoint": str(checkpoint), "parent_sha256": actual_sha,
        "checks": {"non_sga_sets_identical": non_sga_equal, "initialization_identical": init_equal},
        "initialization_checks": init_checks, "cases": case_rows,
        "prohibited_execution": {"forward": False, "training": False, "validation": False, "full_test": False},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_reports(result, out_dir)
    print(json.dumps({"result": result["result"], "checks": result["checks"], "manifest": str(manifest)}))
    if result["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
