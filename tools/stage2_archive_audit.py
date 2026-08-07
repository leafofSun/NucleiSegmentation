#!/usr/bin/env python3
"""Stage-2 research asset classification and archive-manifest generator."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path("/hy-tmp/NuSeg").resolve()
MODELS = ROOT / "workdir" / "models"
AUDIT = ROOT / "workdir" / "storage_audit"
ARCHIVE_META = {"RUN_ARCHIVE_MANIFEST.json", "RUN_ARCHIVE_MANIFEST.txt", "SHA256SUMS"}
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx"}
DATE = "2026-07-27"

P3_RUNS = {
    "sga_sb_p3_g1_seed42_e5_schedfix_v1",
    "sga_sb_p3_g2_seed42_e5_schedfix_v1",
    "sga_sb_p3_g3_seed42_e5_schedfix_v1",
    "sga_sb_p3_s1_seed42_e5_schedfix_v1",
}
L1A_RUNS = {
    "local_region_text_l1a_c0_seed42_e5_v1",
    "local_region_text_l1a_c0_seed42_e5_v1_failed_attempt_1",
    "local_region_text_l1a_conch_seed42_e5_v1",
}
CORE_RUNS = {
    "Visual_baseline",
    "phaseB_ml_instancefix_from_visual_3gpu_30ep_v1",
    "phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD",
    "sga_sb_p41_g2_soft_seed42_e5_v1",
}
CURRENT_REFERENCED_RUNS = {
    "exp5_numeric_attr_route_10ep_reinit1e4_v1",
    "exp6_phaseC_text_both_10ep_reinit1e4_v1",
}

ASSOCIATED = {
    "p3": [
        "workdir/audits/sga_sb_p3_20260713",
        "workdir/audits/sga_sb_p4_20260714/P3_FINAL_SCREENING_REPORT.md",
        "workdir/audits/sga_sb_p4_20260714/P4_CHECKPOINT_ROUTING_AUDIT.md",
    ],
    "l1a": [
        "workdir/audits/local_region_text_l1a_20260722/L1A_FINAL_REPORT.md",
        "workdir/audits/local_region_text_l1a_20260722/L1A_FINAL_SUMMARY.json",
        "workdir/audits/local_region_text_l1a_20260722/L1A_ADVANCE_GATE.json",
        "workdir/audits/local_region_text_l1a_20260722/L1A_EXPERIMENT_MANIFEST.json",
        "workdir/audits/local_region_text_l1a_20260722/L1A_C0_COMMAND.txt",
        "workdir/audits/local_region_text_l1a_20260722/L1A_LOCAL_TEXT_COMMAND.txt",
        "workdir/audits/local_region_text_l1a_20260722/L1A_C0_TRAIN.log",
        "workdir/audits/local_region_text_l1a_20260722/L1A_C0_TRAIN_FAILED_ATTEMPT_1.log",
        "workdir/audits/local_region_text_l1a_20260722/L1A_LOCAL_TEXT_TRAIN.log",
    ],
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def payload_files(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name not in ARCHIVE_META
    )


def tree_size(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in payload_files(run_dir))


def metrics(run_dir: Path):
    path = run_dir / "metrics_history.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            keys = ("epoch", "Dice", "IoU", "mAJI", "mPQ", "val_align_loss", "val_composite_score")
            return sanitize({key: data[-1].get(key) for key in keys if key in data[-1]})
        return sanitize(data)
    except Exception as exc:
        return {"parse_error": str(exc)}


def classification(name: str) -> tuple[str, str, str, str, str]:
    if name in CORE_RUNS:
        stage = {
            "Visual_baseline": "Phase A visual baseline",
            "phaseB_ml_instancefix_from_visual_3gpu_30ep_v1": "Phase B canonical",
            "phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD": "Phase C canonical",
            "sga_sb_p41_g2_soft_seed42_e5_v1": "P4.1 current mainline",
        }[name]
        return "KEEP_FULL_LOCAL", stage, "canonical/current", "PROTECTED", "Required local canonical asset"
    if name in P3_RUNS:
        return (
            "ARCHIVE_THEN_DELETE_LOCAL", "P3 five-epoch screening", "historical ablation",
            "COMPLETED_GATE_FAIL", "Superseded by P4.1/P41; archive required before local deletion",
        )
    if name in L1A_RUNS:
        status = "FAILED_ATTEMPT_PRESERVED" if "failed_attempt" in name else "COMPLETED_BRANCH_FAIL"
        return (
            "FAILED_BRANCH_ARCHIVE_THEN_DELETE_LOCAL", "L1-A local text alignment",
            "failed branch", status, "L1A_ADVANCE=FAIL; archive required before local deletion",
        )
    if name in CURRENT_REFERENCED_RUNS:
        status = "CURRENT_REPRO_REFERENCE" if name.startswith("exp5") else "HISTORICAL_BUT_CURRENTLY_REFERENCED"
        return (
            "KEEP_FULL_LOCAL", "Exp5/Exp6 reproducibility line", "referenced branch",
            status, "Referenced by current scripts/documents or pending protocol reconciliation",
        )
    return (
        "UNKNOWN_DO_NOT_DELETE", "unknown/historical", "undetermined",
        "UNKNOWN", "Scientific value or supersession cannot be proven",
    )


def parent_checkpoint(name: str):
    if name in P3_RUNS:
        return "workdir/models/Visual_baseline/best_model.pth"
    if name in L1A_RUNS:
        return "workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth"
    if name == "exp6_phaseC_text_both_10ep_reinit1e4_v1":
        return "workdir/models/phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD/best_align_full_model.pth"
    return None


def associated_assets(name: str) -> list[str]:
    raw = ASSOCIATED["p3"] if name in P3_RUNS else ASSOCIATED["l1a"] if name in L1A_RUNS else []
    result = []
    for item in raw:
        path = ROOT / item
        if path.exists():
            result.append(item)
    return result


def command_sources(name: str) -> tuple[list[str], list[str]]:
    if name in P3_RUNS:
        tag = name.split("_p3_", 1)[1].split("_", 1)[0].upper()
        train_paths = [
            f"workdir/audits/sga_sb_p3_20260713/RUN_P3_{tag}.sh",
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_CASES_COMMON.sh",
        ]
        return [p for p in train_paths if (ROOT / p).exists()], []
    if name == "local_region_text_l1a_conch_seed42_e5_v1":
        return ["workdir/audits/local_region_text_l1a_20260722/L1A_LOCAL_TEXT_COMMAND.txt"], []
    if name in L1A_RUNS:
        return ["workdir/audits/local_region_text_l1a_20260722/L1A_C0_COMMAND.txt"], []
    return [], []


def build_run_manifest(run_dir: Path, class_row: dict) -> dict:
    files = payload_files(run_dir)
    entries = []
    for path in files:
        st = path.stat()
        entries.append({
            "relative_path": str(path.relative_to(run_dir)),
            "size_bytes": st.st_size,
            "mtime": iso(st.st_mtime),
            "sha256": sha256(path),
        })
    dirs = [path for path in run_dir.rglob("*") if path.is_dir() and not path.is_symlink()]
    checkpoint_entries = [
        item for item in entries if Path(item["relative_path"]).suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    config_entries = [
        item["relative_path"] for item in entries
        if Path(item["relative_path"]).suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    ]
    log_entries = [
        item["relative_path"] for item in entries
        if Path(item["relative_path"]).suffix.lower() == ".log" or "events.out.tfevents" in item["relative_path"]
    ]
    metric_entries = [
        item["relative_path"] for item in entries
        if "metric" in item["relative_path"].lower()
    ]
    mtimes = [path.stat().st_mtime for path in files] or [run_dir.stat().st_mtime]
    train_cmds, test_cmds = command_sources(run_dir.name)
    manifest = {
        "schema_version": "nuseg_run_archive_v1",
        "run_name": run_dir.name,
        "original_local_path": str(run_dir),
        "experiment_stage": class_row["experiment_stage"],
        "experiment_conclusion": class_row["final_status"],
        "created_time_approx": iso(min(mtimes)),
        "last_modified": iso(max(mtimes)),
        "payload_file_count": len(entries),
        "payload_directory_count": len(dirs) + 1,
        "payload_total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
        "checkpoint_list": checkpoint_entries,
        "config_list": config_entries,
        "log_list": log_entries,
        "metrics_list": metric_entries,
        "associated_reports_and_logs": associated_assets(run_dir.name),
        "parent_checkpoint": parent_checkpoint(run_dir.name),
        "training_command_sources": train_cmds,
        "test_command_sources": test_cmds,
        "code_version_information": "GIT_METADATA_STATUS=ABSENT",
        "manifest_generated_at": now(),
        "archive_time": None,
        "remote_archive_location": f"oss://<BUCKET>/NuSeg-archive/{DATE}/{run_dir.name}/",
        "archive_status": "REMOTE_UNAVAILABLE",
        "archive_object_plan": {
            "payload_prefix": "files/",
            "metadata_objects": sorted(ARCHIVE_META),
            "planned_object_count": len(entries) + 3,
        },
        "credential_material_included": False,
    }
    return manifest


def write_run_metadata(run_dir: Path, manifest: dict) -> None:
    json_path = run_dir / "RUN_ARCHIVE_MANIFEST.json"
    txt_path = run_dir / "RUN_ARCHIVE_MANIFEST.txt"
    sums_path = run_dir / "SHA256SUMS"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path.write_text(
        "".join(f"{item['sha256']}  {item['relative_path']}\n" for item in manifest["files"]),
        encoding="utf-8",
    )
    txt_path.write_text(
        "\n".join([
            f"RUN_NAME={manifest['run_name']}",
            f"ORIGINAL_LOCAL_PATH={manifest['original_local_path']}",
            f"EXPERIMENT_STAGE={manifest['experiment_stage']}",
            f"EXPERIMENT_CONCLUSION={manifest['experiment_conclusion']}",
            f"PAYLOAD_FILE_COUNT={manifest['payload_file_count']}",
            f"PAYLOAD_DIRECTORY_COUNT={manifest['payload_directory_count']}",
            f"PAYLOAD_TOTAL_BYTES={manifest['payload_total_bytes']}",
            f"CHECKPOINT_COUNT={len(manifest['checkpoint_list'])}",
            f"CODE_VERSION={manifest['code_version_information']}",
            f"ARCHIVE_STATUS={manifest['archive_status']}",
            f"REMOTE_LOCATION={manifest['remote_archive_location']}",
            "CREDENTIAL_MATERIAL_INCLUDED=false",
        ]) + "\n", encoding="utf-8",
    )


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(path for path in MODELS.iterdir() if path.is_dir() and not path.is_symlink()):
        category, stage, branch, status, recommendation = classification(run_dir.name)
        files = payload_files(run_dir)
        checkpoints = [p for p in files if p.suffix.lower() in CHECKPOINT_SUFFIXES]
        mtimes = [p.stat().st_mtime for p in files] or [run_dir.stat().st_mtime]
        row = {
            "run_name": run_dir.name,
            "local_path": str(run_dir),
            "size_bytes": sum(p.stat().st_size for p in files),
            "last_modified": iso(max(mtimes)),
            "experiment_stage": stage,
            "experiment_goal": (
                "SGA-SB causal five-case screening" if run_dir.name in P3_RUNS
                else "L1-A local-region text supervision matched comparison" if run_dir.name in L1A_RUNS
                else stage
            ),
            "final_status": status,
            "mainline_or_branch": branch,
            "superseded_by": "sga_sb_p41_g2_soft_seed42_e5_v1" if run_dir.name in P3_RUNS else None,
            "referenced_by_current_scripts": run_dir.name == "exp6_phaseC_text_both_10ep_reinit1e4_v1",
            "referenced_by_current_documents": (
                run_dir.name in P3_RUNS or run_dir.name in L1A_RUNS or run_dir.name in CURRENT_REFERENCED_RUNS
            ),
            "checkpoint_count": len(checkpoints),
            "checkpoint_names": [p.name for p in checkpoints],
            "best_metrics": metrics(run_dir),
            "full_test_completed": False,
            "resume_still_needed": run_dir.name in CORE_RUNS,
            "failure_scene_value": "HIGH" if run_dir.name in L1A_RUNS else "NORMAL",
            "archive_recommendation": category,
            "local_retention_recommendation": (
                "KEEP_UNTIL_VERIFIED_REMOTE_ARCHIVE"
                if category in {"ARCHIVE_THEN_DELETE_LOCAL", "FAILED_BRANCH_ARCHIVE_THEN_DELETE_LOCAL"}
                else recommendation
            ),
            "classification_evidence": recommendation,
            "associated_assets": associated_assets(run_dir.name),
            "parent_checkpoint": parent_checkpoint(run_dir.name),
        }
        rows.append(row)

    archive_rows = [
        row for row in rows
        if row["archive_recommendation"] in {"ARCHIVE_THEN_DELETE_LOCAL", "FAILED_BRANCH_ARCHIVE_THEN_DELETE_LOCAL"}
    ]
    manifests = []
    for row in archive_rows:
        run_dir = Path(row["local_path"])
        manifest = build_run_manifest(run_dir, row)
        write_run_metadata(run_dir, manifest)
        manifests.append({
            "run_name": row["run_name"],
            "local_path": row["local_path"],
            "payload_file_count": manifest["payload_file_count"],
            "payload_directory_count": manifest["payload_directory_count"],
            "payload_total_bytes": manifest["payload_total_bytes"],
            "checkpoint_count": len(manifest["checkpoint_list"]),
            "manifest_paths": [str(run_dir / name) for name in sorted(ARCHIVE_META)],
            "planned_remote_location": manifest["remote_archive_location"],
            "archive_status": "REMOTE_UNAVAILABLE",
        })

    remote_rows = [{
        "run_name": item["run_name"],
        "ARCHIVE_UPLOAD_STATUS": "REMOTE_UNAVAILABLE",
        "REMOTE_FILE_COUNT": None,
        "REMOTE_TOTAL_BYTES": None,
        "CHECKSUM_VERIFICATION_METHOD": None,
        "CHECKSUM_VERIFIED": False,
        "CRITICAL_FILES_VERIFIED": False,
        "local_deletion_authorized": False,
        "reason": "rclone is installed but has no config and zero configured remotes.",
    } for item in manifests]

    (AUDIT / "STAGE2_RUN_CLASSIFICATION.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT / "STAGE2_ARCHIVE_MANIFEST.json").write_text(
        json.dumps({
            "generated_at": now(),
            "archive_status": "REMOTE_UNAVAILABLE",
            "remote_location_template": f"oss://<BUCKET>/NuSeg-archive/{DATE}/<run_name>/",
            "runs": manifests,
            "total_planned_payload_bytes": sum(item["payload_total_bytes"] for item in manifests),
            "local_tar_created": False,
            "credentials_recorded": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT / "STAGE2_REMOTE_VERIFICATION.json").write_text(
        json.dumps({
            "checked_at": now(),
            "remote_tool": "rclone",
            "remote_config_present": False,
            "configured_remote_count": 0,
            "ARCHIVE_STATUS": "REMOTE_UNAVAILABLE",
            "runs": remote_rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT / "STAGE2_LOCAL_DELETE_MANIFEST.json").write_text(
        json.dumps({
            "generated_at": now(),
            "authorized_entries": [],
            "deleted_entries": [],
            "reason": "No run has a verified remote archive; local checkpoint deletion is prohibited.",
            "deleted_file_count": 0,
            "deleted_directory_count": 0,
            "deleted_bytes": 0,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT / "STAGE2_DELETE_ERRORS.log").touch()
    print(json.dumps({
        "classified_runs": len(rows),
        "archive_planned_runs": len(manifests),
        "archive_planned_bytes": sum(item["payload_total_bytes"] for item in manifests),
        "remote_status": "REMOTE_UNAVAILABLE",
        "delete_authorized_count": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
