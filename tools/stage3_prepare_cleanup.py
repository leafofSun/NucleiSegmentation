#!/usr/bin/env python3
"""Prepare lightweight research records and exact Stage-3 delete manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path


ROOT = Path("/hy-tmp/NuSeg").resolve()
MODELS = ROOT / "workdir" / "models"
AUDIT = ROOT / "workdir" / "storage_audit"
REGISTRY = ROOT / "workdir" / "deleted_run_registry"
CLASSIFICATION = AUDIT / "STAGE2_RUN_CLASSIFICATION.json"
DELETE_MANIFEST = AUDIT / "STAGE3_LOCAL_FINAL_DELETE_MANIFEST.json"
ERROR_LOG = AUDIT / "STAGE3_LOCAL_DELETE_ERRORS.log"
ALLOWED_CLASSES = {"ARCHIVE_THEN_DELETE_LOCAL", "FAILED_BRANCH_ARCHIVE_THEN_DELETE_LOCAL"}
APPROVED_NAMES = {
    "sga_sb_p3_g3_seed42_e5_schedfix_v1",
    "sga_sb_p3_s1_seed42_e5_schedfix_v1",
    "sga_sb_p3_g2_seed42_e5_schedfix_v1",
    "sga_sb_p3_g1_seed42_e5_schedfix_v1",
    "local_region_text_l1a_c0_seed42_e5_v1",
    "local_region_text_l1a_c0_seed42_e5_v1_failed_attempt_1",
    "local_region_text_l1a_conch_seed42_e5_v1",
}
META_NAMES = ("RUN_ARCHIVE_MANIFEST.json", "RUN_ARCHIVE_MANIFEST.txt", "SHA256SUMS")

RUN_DETAILS = {
    "sga_sb_p3_g1_seed42_e5_schedfix_v1": {
        "modules": ["SGA-SB structure head", "structure adapter", "gamma_structure", "structure guidance"],
        "command_sources": [
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_G1.sh",
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_CASES_COMMON.sh",
        ],
        "log": "workdir/audits/sga_sb_p3_20260713/P3_G1_SCHEDFIX_V1_TRAIN.log",
    },
    "sga_sb_p3_g2_seed42_e5_schedfix_v1": {
        "modules": ["SGA-SB boundary head", "boundary adapter", "gamma_boundary", "boundary guidance"],
        "command_sources": [
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_G2.sh",
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_CASES_COMMON.sh",
        ],
        "log": "workdir/audits/sga_sb_p3_20260713/P3_G2_SCHEDFIX_V1_TRAIN.log",
    },
    "sga_sb_p3_g3_seed42_e5_schedfix_v1": {
        "modules": ["SGA-SB structure+boundary heads", "both adapters", "both gammas", "combined guidance"],
        "command_sources": [
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_G3.sh",
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_CASES_COMMON.sh",
        ],
        "log": "workdir/audits/sga_sb_p3_20260713/P3_G3_SCHEDFIX_V1_TRAIN.log",
    },
    "sga_sb_p3_s1_seed42_e5_schedfix_v1": {
        "modules": ["SGA-SB structure+boundary auxiliary supervision", "feature guidance disabled"],
        "command_sources": [
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_S1.sh",
            "workdir/audits/sga_sb_p3_20260713/RUN_P3_CASES_COMMON.sh",
        ],
        "log": "workdir/audits/sga_sb_p3_20260713/P3_S1_SCHEDFIX_V1_TRAIN.log",
    },
    "local_region_text_l1a_c0_seed42_e5_v1": {
        "modules": [
            "Numeric attribute FreqPath guidance", "multilevel attribute heads",
            "local region text alignment disabled (matched C0 control)",
        ],
        "command_sources": ["workdir/audits/local_region_text_l1a_20260722/L1A_C0_COMMAND.txt"],
        "log": "workdir/audits/local_region_text_l1a_20260722/L1A_C0_TRAIN.log",
    },
    "local_region_text_l1a_c0_seed42_e5_v1_failed_attempt_1": {
        "modules": [
            "Numeric attribute FreqPath guidance", "multilevel attribute heads",
            "local region text alignment disabled (failed C0 serialization attempt)",
        ],
        "command_sources": ["workdir/audits/local_region_text_l1a_20260722/L1A_C0_COMMAND.txt"],
        "log": "workdir/audits/local_region_text_l1a_20260722/L1A_C0_TRAIN_FAILED_ATTEMPT_1.log",
    },
    "local_region_text_l1a_conch_seed42_e5_v1": {
        "modules": [
            "Numeric attribute FreqPath guidance", "multilevel attribute heads",
            "L1-A local region text alignment", "frozen CONCH prototype bank",
            "supervision-only local text loss",
        ],
        "command_sources": ["workdir/audits/local_region_text_l1a_20260722/L1A_LOCAL_TEXT_COMMAND.txt"],
        "log": "workdir/audits/local_region_text_l1a_20260722/L1A_LOCAL_TEXT_TRAIN.log",
    },
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_stats(path: Path) -> tuple[int, int, int]:
    size = files = dirs = 0
    for base, _, names in os.walk(path, followlinks=False):
        dirs += 1
        for name in names:
            item = Path(base) / name
            if not item.is_symlink():
                size += item.stat().st_size
                files += 1
    return size, files, dirs


def load_metrics(run_dir: Path) -> tuple[list[dict], str | None]:
    for name in ("metrics_history.json", "metrics_history.json.tmp"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return (data if isinstance(data, list) else []), None
        except Exception as exc:
            return [], f"{name} is incomplete/unparseable: {exc}"
    return [], "metrics_history not found"


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def best_metric(rows: list[dict], key: str):
    valid = [row for row in rows if finite_number(row.get(key))]
    if not valid:
        return {"status": "UNKNOWN", "value": None, "epoch": None}
    row = max(valid, key=lambda item: float(item[key]))
    return {"status": "AVAILABLE", "value": float(row[key]), "epoch": row.get("epoch")}


def epoch_metrics(rows: list[dict]) -> list[dict]:
    return [{
        key: (None if isinstance(row.get(key), float) and not math.isfinite(row[key]) else row.get(key))
        for key in ("epoch", "Dice", "IoU", "mAJI", "mPQ")
    } for row in rows]


def command_text(sources: list[str]) -> str:
    chunks = []
    for relative in sources:
        path = ROOT / relative
        if path.exists():
            chunks.append(f"# SOURCE: {relative}\n{path.read_text(encoding='utf-8', errors='replace').strip()}")
    return "\n\n".join(chunks) if chunks else "UNKNOWN"


def error_excerpt(relative: str) -> list[str]:
    path = ROOT / relative
    if not path.exists():
        return [f"LOG_NOT_FOUND={relative}"]
    pattern = re.compile(r"traceback|error|exception|failed|outofmemory|out of memory|nan|inf", re.I)
    matches = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            clean = line.replace("\r", "").strip()
            if clean and pattern.search(clean):
                matches.append(clean[:2000])
                if len(matches) >= 40:
                    break
    return matches or ["NO_ERROR_PATTERN_FOUND"]


def conclusions(name: str) -> tuple[str, str, str, str]:
    if name.startswith("sga_sb_p3_"):
        return (
            "FAIL",
            "P3 completed; minimum and priority P4 advance gates both failed.",
            "sga_sb_p41_g2_soft_seed42_e5_v1",
            "Completed historical screening was superseded by the P4.1/P41 current route.",
        )
    if "failed_attempt" in name:
        return (
            "L1A_ADVANCE=FAIL; ATTEMPT_FAILED",
            "C0 attempt stopped after epoch 1 because strict JSON serialization rejected a non-finite inactive meter.",
            "sga_sb_p41_g2_soft_seed42_e5_v1 (project mainline; L1-A branch terminated)",
            "Failure is fully documented in the retained L1-A report/log; checkpoint recovery is explicitly waived.",
        )
    return (
        "FAIL",
        "L1A_ADVANCE=FAIL; E5 delta mAJI did not reach the +0.003 advance threshold.",
        "sga_sb_p41_g2_soft_seed42_e5_v1 (project mainline; L1-A branch terminated)",
        "L1-A branch conclusion is fixed and user explicitly waived checkpoint recovery.",
    )


def summary_markdown(summary: dict) -> str:
    def value(item):
        return "UNKNOWN" if item is None else str(item)

    lines = [
        f"# Deleted Run Research Summary: {summary['run_name']}",
        "",
        f"- run_name: `{summary['run_name']}`",
        f"- original_path: `{summary['original_path']}`",
        f"- experiment_stage: {summary['experiment_stage']}",
        f"- random_seed: {summary['random_seed']}",
        f"- starting_checkpoint: `{value(summary['starting_checkpoint'])}`",
        f"- enabled_modules: {', '.join(summary['enabled_modules'])}",
        f"- training_epochs: {summary['training_epochs']}",
        f"- best_validation_mAJI: {summary['best_validation_mAJI']}",
        f"- best_validation_mPQ: {summary['best_validation_mPQ']}",
        "- full_test_status=NOT_AVAILABLE",
        "- full_test_Dice: UNKNOWN",
        "- full_test_IoU: UNKNOWN",
        "- full_test_mAJI: UNKNOWN",
        "- full_test_mPQ: UNKNOWN",
        f"- advance_gate: {summary['advance_gate']}",
        f"- final_conclusion: {summary['final_conclusion']}",
        f"- superseded_by: {summary['superseded_by']}",
        f"- original_total_bytes: {summary['original_total_bytes']}",
        f"- original_file_count: {summary['original_file_count']}",
        f"- original_directory_count: {summary['original_directory_count']}",
        f"- deletion_reason: {summary['deletion_reason']}",
        f"- deletion_time: {summary['deletion_time']}",
        f"- deletion_success: {str(summary['deletion_success']).lower()}",
        "- recoverable: false",
        "",
        "## Checkpoints",
        "",
    ]
    for item in summary["checkpoint_files"]:
        lines.append(f"- `{item['relative_path']}` — {item['size_bytes']} B — SHA256 `{item['sha256']}`")
    lines.extend([
        "",
        "## Epoch Validation Metrics",
        "",
        "| Epoch | Dice | IoU | mAJI | mPQ |",
        "|---:|---:|---:|---:|---:|",
    ])
    if summary["epoch_validation_metrics"]:
        for row in summary["epoch_validation_metrics"]:
            lines.append(
                f"| {value(row.get('epoch'))} | {value(row.get('Dice'))} | {value(row.get('IoU'))} | "
                f"{value(row.get('mAJI'))} | {value(row.get('mPQ'))} |"
            )
    else:
        lines.append("| UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |")
    lines.extend([
        "",
        "## Key Launch Command",
        "",
        "```text",
        summary["key_launch_command"],
        "```",
        "",
        "## Evidence",
        "",
        f"- Stage 2 manifest: `RUN_ARCHIVE_MANIFEST.json`",
        f"- Original SHA list: `SHA256SUMS`",
        f"- Key log excerpt: `KEY_LOG_SUMMARY.txt`",
        f"- Metric parse note: {value(summary['metrics_parse_note'])}",
        "",
    ])
    return "\n".join(lines)


def create_run_registry(row: dict) -> tuple[dict, dict]:
    run_dir = Path(row["local_path"]).resolve(strict=True)
    manifest_path = run_dir / "RUN_ARCHIVE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = REGISTRY / row["run_name"]
    target.mkdir(parents=True, exist_ok=True)
    for name in META_NAMES:
        source = run_dir / name
        if not source.is_file():
            raise ValueError(f"{row['run_name']}: missing {name}")
        shutil.copy2(source, target / name)

    config_dir = target / "config_and_metrics"
    config_dir.mkdir(exist_ok=True)
    copied = []
    for source in list(run_dir.iterdir()):
        if source.is_file() and source.name not in META_NAMES and source.stat().st_size <= 10 * 1024 * 1024:
            if source.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".toml", ".tmp"}:
                shutil.copy2(source, config_dir / source.name)
                copied.append(str((config_dir / source.name).relative_to(target)))
    details = RUN_DETAILS[row["run_name"]]
    for relative in details["command_sources"]:
        source = ROOT / relative
        if source.is_file():
            destination = config_dir / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(target)))

    rows, parse_note = load_metrics(run_dir)
    advance, conclusion, superseded, reason = conclusions(row["run_name"])
    epochs = 1 if "failed_attempt" in row["run_name"] else 5
    launch = command_text(details["command_sources"])
    errors = error_excerpt(details["log"])
    current_size, current_files, current_dirs = tree_stats(run_dir)
    summary = {
        "schema_version": "nuseg_deleted_run_registry_v1",
        "run_name": row["run_name"],
        "original_path": str(run_dir),
        "experiment_stage": row["experiment_stage"],
        "random_seed": 42,
        "starting_checkpoint": row.get("parent_checkpoint"),
        "key_launch_parameters": launch,
        "key_launch_command": launch,
        "enabled_modules": details["modules"],
        "training_epochs": epochs,
        "best_validation_mAJI": best_metric(rows, "mAJI"),
        "best_validation_mPQ": best_metric(rows, "mPQ"),
        "epoch_validation_metrics": epoch_metrics(rows),
        "final_validation_metrics": epoch_metrics(rows)[-1] if rows else {
            "epoch": None, "Dice": None, "IoU": None, "mAJI": None, "mPQ": None,
        },
        "full_test_status": "NOT_AVAILABLE",
        "full_test_Dice": None,
        "full_test_IoU": None,
        "full_test_mAJI": None,
        "full_test_mPQ": None,
        "full_test_completed": False,
        "advance_gate": advance,
        "final_conclusion": conclusion,
        "superseded_by": superseded,
        "original_total_bytes": manifest["payload_total_bytes"],
        "original_file_count": manifest["payload_file_count"],
        "original_directory_count": manifest["payload_directory_count"],
        "current_tree_bytes_before_delete": current_size,
        "current_tree_file_count_before_delete": current_files,
        "current_tree_directory_count_before_delete": current_dirs,
        "checkpoint_files": manifest["checkpoint_list"],
        "retained_small_files": copied,
        "source_log": details["log"],
        "log_error_excerpt": errors,
        "metrics_parse_note": parse_note,
        "deletion_reason": reason,
        "deletion_time": "PENDING_EXECUTION",
        "deletion_success": False,
        "recoverable": False,
        "user_approved_no_remote_backup": True,
        "created_at": now(),
    }
    (target / "RUN_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "RUN_SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
    (target / "KEY_LOG_SUMMARY.txt").write_text(
        "\n".join([
            f"RUN_NAME={row['run_name']}",
            f"SOURCE_LOG={details['log']}",
            f"ADVANCE_GATE={advance}",
            f"FINAL_CONCLUSION={conclusion}",
            f"BEST_VALIDATION_MAJI={summary['best_validation_mAJI']}",
            f"BEST_VALIDATION_MPQ={summary['best_validation_mPQ']}",
            "FULL_TEST_STATUS=NOT_AVAILABLE",
            "",
            "EPOCH_METRICS_JSON=",
            json.dumps(summary["epoch_validation_metrics"], ensure_ascii=False, indent=2),
            "",
            "ERROR_EXCERPT=",
            "\n".join(errors),
        ]) + "\n", encoding="utf-8"
    )
    required = [
        target / "RUN_SUMMARY.md", target / "RUN_SUMMARY.json",
        target / "RUN_ARCHIVE_MANIFEST.json", target / "RUN_ARCHIVE_MANIFEST.txt",
        target / "SHA256SUMS", target / "KEY_LOG_SUMMARY.txt",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError(f"{row['run_name']}: incomplete registry")
    registry_record = {
        "run_name": row["run_name"],
        "registry_path": str(target),
        "registry_complete": True,
        "summary_json_sha256": sha256(target / "RUN_SUMMARY.json"),
        "stage2_manifest_sha256": sha256(target / "RUN_ARCHIVE_MANIFEST.json"),
    }
    delete_entry = {
        "run_name": row["run_name"],
        "local_path": str(run_dir),
        "classification": row["archive_recommendation"],
        "registry_path": str(target),
        "expected_payload_file_count": manifest["payload_file_count"],
        "expected_payload_directory_count": manifest["payload_directory_count"],
        "expected_payload_bytes": manifest["payload_total_bytes"],
        "current_tree_bytes": current_size,
        "current_tree_file_count": current_files,
        "current_tree_directory_count": current_dirs,
        "stage2_manifest_sha256": registry_record["stage2_manifest_sha256"],
        "user_approved_no_remote_backup": True,
        "recoverable": False,
    }
    return registry_record, delete_entry


def validation_summary(run_dir: Path):
    rows, _ = load_metrics(run_dir)
    return {
        "best_mAJI": best_metric(rows, "mAJI"),
        "best_mPQ": best_metric(rows, "mPQ"),
    } if rows else "UNKNOWN"


def model_entry(role, checkpoint, run, stage, parent, modules, status, metrics, resume, paper, reason):
    path = ROOT / checkpoint
    return {
        "model_role": role,
        "checkpoint_path": checkpoint,
        "checkpoint_exists": path.is_file(),
        "run_name": run,
        "stage": stage,
        "parent_checkpoint": parent,
        "main_modules": modules,
        "training_status": status,
        "validation_metrics": metrics,
        "same_test_pipeline_metrics": "UNKNOWN",
        "resume_needed": resume,
        "paper_comparison_needed": paper,
        "retention_reason": reason,
    }


def create_model_registry() -> list[dict]:
    n0 = MODELS / "sga_sb_p3_n0_seed42_e5"
    entries = [
        model_entry(
            "Visual baseline", "workdir/models/Visual_baseline/best_model.pth", "Visual_baseline",
            "Phase A", "workdir/models/sam-med2d_b.pth", ["visual segmentation baseline"],
            "COMPLETED_CANONICAL", validation_summary(MODELS / "Visual_baseline"), True, True,
            "Mandatory canonical visual baseline",
        ),
        model_entry(
            "Phase B canonical", "workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth",
            "phaseB_ml_instancefix_from_visual_3gpu_30ep_v1", "Phase B",
            "workdir/models/Visual_baseline/best_model.pth", ["multilevel attribute heads"],
            "COMPLETED_CANONICAL", "UNKNOWN", True, True, "Mandatory Phase B parent checkpoint",
        ),
        model_entry(
            "Phase C canonical", "workdir/models/phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD/best_align_full_model.pth",
            "phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD", "Phase C",
            "workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth",
            ["attribute-text alignment"], "COMPLETED_CANONICAL", "UNKNOWN", True, True,
            "Mandatory canonical Phase C checkpoint",
        ),
        model_entry(
            "Exp5 numeric representative", "workdir/models/exp5_numeric_attr_route_10ep_reinit1e4_v1/best_aji_model.pth",
            "exp5_numeric_attr_route_10ep_reinit1e4_v1", "Exp5 semantic injection",
            "workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth",
            ["numeric attribute FreqPath guidance", "no CONCH/PG3"], "COMPLETED_REPRO_REFERENCE",
            "UNKNOWN", False, True, "Current no-text numeric-route comparison asset",
        ),
        model_entry(
            "Exp6 CONCH/PG3 representative", "workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth",
            "exp6_phaseC_text_both_10ep_reinit1e4_v1", "Exp6 semantic injection",
            "workdir/models/phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD/best_align_full_model.pth",
            ["CONCH/text bank", "PG3", "attribute-text alignment"], "COMPLETED_REFERENCED",
            "UNKNOWN", False, True, "Explicitly protected and still referenced by current scripts/P0 plan",
        ),
        model_entry(
            "P41 current mainline", "workdir/models/sga_sb_p41_g2_soft_seed42_e5_v1/latest_model.pth",
            "sga_sb_p41_g2_soft_seed42_e5_v1", "P4.1",
            "workdir/models/Visual_baseline/best_model.pth", ["soft boundary SGA-SB guidance"],
            "CURRENT_MAINLINE", validation_summary(MODELS / "sga_sb_p41_g2_soft_seed42_e5_v1"),
            True, True, "Current mainline; mandatory local retention",
        ),
        model_entry(
            "Current highest retained validation mAJI candidate",
            "workdir/models/sga_sb_p3_n0_seed42_e5/best_model.pth", "sga_sb_p3_n0_seed42_e5",
            "P3 N0 control", "workdir/models/Visual_baseline/best_model.pth",
            ["no SGA-SB guidance control"], "COMPLETED_RETAINED",
            validation_summary(n0), False, True,
            "Highest observed validation mAJI among retained runs with parseable metrics; same-test result UNKNOWN",
        ),
        model_entry(
            "Current highest retained validation mPQ candidate",
            "workdir/models/sga_sb_p3_n0_seed42_e5/best_model.pth", "sga_sb_p3_n0_seed42_e5",
            "P3 N0 control", "workdir/models/Visual_baseline/best_model.pth",
            ["no SGA-SB guidance control"], "COMPLETED_RETAINED",
            validation_summary(n0), False, True,
            "Highest observed validation mPQ among retained runs with parseable metrics; same-test result UNKNOWN",
        ),
    ]
    if any(not item["checkpoint_exists"] for item in entries):
        missing = [item["checkpoint_path"] for item in entries if not item["checkpoint_exists"]]
        raise ValueError(f"model registry checkpoint missing: {missing}")
    (ROOT / "workdir" / "MODEL_REGISTRY.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# NuSeg Model Registry",
        "",
        "Generated before Stage-3 irreversible historical-run deletion.",
        "",
        "| Role | Checkpoint | Stage | Status | Resume | Paper comparison | Retention reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in entries:
        lines.append(
            f"| {item['model_role']} | `{item['checkpoint_path']}` | {item['stage']} | "
            f"{item['training_status']} | {item['resume_needed']} | {item['paper_comparison_needed']} | "
            f"{item['retention_reason']} |"
        )
    lines.extend([
        "",
        "Validation and same-test metrics are recorded as `UNKNOWN` where evidence is not comparable or unavailable.",
        "The two “highest retained” roles refer only to parseable validation histories among models retained after Stage 3; they are not full-test claims.",
        "",
    ])
    (ROOT / "workdir" / "MODEL_REGISTRY.md").write_text("\n".join(lines), encoding="utf-8")
    return entries


def main() -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.write_text("", encoding="utf-8")
    rows = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    selected = [row for row in rows if row["archive_recommendation"] in ALLOWED_CLASSES]
    names = {row["run_name"] for row in selected}
    if len(selected) != 7 or names != APPROVED_NAMES:
        raise SystemExit(f"Stage-2 exact authorization mismatch: count={len(selected)} names={sorted(names)}")
    model_registry = create_model_registry()
    registry_records, delete_entries, failures = [], [], []
    for row in sorted(selected, key=lambda item: item["run_name"]):
        try:
            registry_record, delete_entry = create_run_registry(row)
            registry_records.append(registry_record)
            delete_entries.append(delete_entry)
        except Exception as exc:
            failure = {"run_name": row["run_name"], "error": str(exc)}
            failures.append(failure)
            with ERROR_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    data = {
        "schema_version": "nuseg_stage3_local_final_delete_v1",
        "generated_at": now(),
        "project_root": str(ROOT),
        "models_root": str(MODELS),
        "registry_root": str(REGISTRY),
        "user_approved_no_remote_backup": True,
        "user_accepts_irrecoverability": True,
        "approved_exact_run_count": 7,
        "approved_exact_run_names": sorted(APPROVED_NAMES),
        "authorized_entries": delete_entries if not failures and len(delete_entries) == 7 else [],
        "registry_records": registry_records,
        "registry_failures": failures,
        "model_registry_entry_count": len(model_registry),
        "expected_release_bytes": sum(item["current_tree_bytes"] for item in delete_entries),
        "execute_status": "PENDING",
    }
    DELETE_MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "registry_complete_count": len(registry_records),
        "registry_failure_count": len(failures),
        "delete_authorized_count": len(data["authorized_entries"]),
        "expected_release_bytes": data["expected_release_bytes"],
        "model_registry_entries": len(model_registry),
    }, ensure_ascii=False, indent=2))
    if failures or len(data["authorized_entries"]) != 7:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
