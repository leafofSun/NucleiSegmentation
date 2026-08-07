#!/usr/bin/env python3
"""Explicit user-approved, no-remote Stage-3 deletion mode."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path

from stage3_prepare_cleanup import summary_markdown


ROOT = Path("/hy-tmp/NuSeg").resolve()
MODELS = (ROOT / "workdir" / "models").resolve()
AUDIT = ROOT / "workdir" / "storage_audit"
EXPECTED_MANIFEST = (AUDIT / "STAGE3_LOCAL_FINAL_DELETE_MANIFEST.json").resolve()
EXPECTED_REGISTRY = (ROOT / "workdir" / "deleted_run_registry").resolve()
APPROVED = {
    "sga_sb_p3_g3_seed42_e5_schedfix_v1",
    "sga_sb_p3_s1_seed42_e5_schedfix_v1",
    "sga_sb_p3_g2_seed42_e5_schedfix_v1",
    "sga_sb_p3_g1_seed42_e5_schedfix_v1",
    "local_region_text_l1a_c0_seed42_e5_v1",
    "local_region_text_l1a_c0_seed42_e5_v1_failed_attempt_1",
    "local_region_text_l1a_conch_seed42_e5_v1",
}
ALLOWED_CLASSES = {"ARCHIVE_THEN_DELETE_LOCAL", "FAILED_BRANCH_ARCHIVE_THEN_DELETE_LOCAL"}
PROTECTED = {
    "Visual_baseline",
    "phaseB_ml_instancefix_from_visual_3gpu_30ep_v1",
    "phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD",
    "sga_sb_p41_g2_soft_seed42_e5_v1",
    "exp6_phaseC_text_both_10ep_reinit1e4_v1",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


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


def open_paths() -> set[Path]:
    result = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            links = list(fd_dir.iterdir())
        except OSError:
            continue
        for link in links:
            try:
                target = Path(os.readlink(link))
                if target.is_absolute() and (target == MODELS or MODELS in target.parents):
                    result.add(target)
            except OSError:
                pass
    return result


def validate_registry(registry: Path, entry: dict) -> None:
    required = (
        "RUN_SUMMARY.md", "RUN_SUMMARY.json", "RUN_ARCHIVE_MANIFEST.json",
        "RUN_ARCHIVE_MANIFEST.txt", "SHA256SUMS", "KEY_LOG_SUMMARY.txt",
    )
    if any(not (registry / name).is_file() or (registry / name).stat().st_size == 0 for name in required):
        raise ValueError("deleted-run registry is incomplete or empty")
    summary = json.loads((registry / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    if summary.get("run_name") != entry["run_name"]:
        raise ValueError("registry run name mismatch")
    if summary.get("recoverable") is not False or summary.get("full_test_status") != "NOT_AVAILABLE":
        raise ValueError("registry recovery/full-test status mismatch")
    if digest(registry / "RUN_ARCHIVE_MANIFEST.json") != entry.get("stage2_manifest_sha256"):
        raise ValueError("registry Stage-2 manifest checksum mismatch")


def run_stage3(args) -> None:
    if not args.user_approved_no_remote:
        raise SystemExit("--user-approved-no-remote is required")
    if not args.run_manifest or not args.registry_root:
        raise SystemExit("--run-manifest and --registry-root are required")
    manifest_path = Path(args.run_manifest).resolve(strict=True)
    registry_root = Path(args.registry_root).resolve(strict=True)
    if manifest_path != EXPECTED_MANIFEST:
        raise SystemExit("run manifest is not the exact protected Stage-3 manifest")
    if registry_root != EXPECTED_REGISTRY:
        raise SystemExit("registry root mismatch")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("authorized_entries", [])
    names = {entry.get("run_name") for entry in entries}
    if (
        data.get("user_approved_no_remote_backup") is not True
        or data.get("user_accepts_irrecoverability") is not True
        or len(entries) != 7
        or names != APPROVED
    ):
        raise SystemExit("exact seven-run authorization mismatch")

    opened = open_paths()
    validated, failures = [], []
    for entry in entries:
        try:
            raw = entry.get("local_path")
            if not raw or not Path(raw).is_absolute():
                raise ValueError("missing or non-absolute path")
            unresolved = Path(raw)
            if unresolved.is_symlink():
                raise ValueError("symlink target prohibited")
            path = unresolved.resolve(strict=True)
            if path.parent != MODELS or path == MODELS:
                raise ValueError("path is not an immediate models child")
            if path.name != entry.get("run_name") or path.name not in APPROVED:
                raise ValueError("path not in exact approved set")
            if path.name in PROTECTED:
                raise ValueError("strong-protected run")
            if entry.get("classification") not in ALLOWED_CLASSES:
                raise ValueError("classification not authorized")
            if entry.get("user_approved_no_remote_backup") is not True or entry.get("recoverable") is not False:
                raise ValueError("irreversible approval missing")
            members = [path] + list(path.rglob("*"))
            if any(member.is_symlink() for member in members):
                raise ValueError("run contains symlink")
            if any(member.resolve(strict=False) in opened for member in members):
                raise ValueError("run contains open file")
            registry = Path(entry.get("registry_path", "")).resolve(strict=True)
            if registry.parent != registry_root or registry.name != path.name:
                raise ValueError("registry path mismatch")
            validate_registry(registry, entry)
            stage2 = json.loads((path / "RUN_ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
            if (
                stage2.get("payload_file_count") != entry.get("expected_payload_file_count")
                or stage2.get("payload_directory_count") != entry.get("expected_payload_directory_count")
                or stage2.get("payload_total_bytes") != entry.get("expected_payload_bytes")
            ):
                raise ValueError("Stage-2 payload count/byte mismatch")
            for item in stage2.get("files", []):
                source = path / item["relative_path"]
                if not source.is_file() or source.is_symlink() or source.stat().st_size != item["size_bytes"]:
                    raise ValueError(f"payload missing/changed: {item['relative_path']}")
                if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
                    raise ValueError(f"payload SHA missing: {item['relative_path']}")
            validated.append((entry, path, registry, tree_stats(path)))
        except Exception as exc:
            failures.append({"run_name": entry.get("run_name"), "path": entry.get("local_path"), "error": str(exc)})

    result = {
        "mode": "dry-run" if args.dry_run else "execute",
        "delete_mode": "LOCAL_FINAL_USER_APPROVED_NO_REMOTE",
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "expected_exact_count": 7,
        "validated_count": len(validated),
        "validated": [str(path) for _, path, _, _ in validated],
        "validation_failures": failures,
        "deleted": [],
        "delete_failures": [],
    }
    if failures or len(validated) != 7 or {path.name for _, path, _, _ in validated} != APPROVED:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    if args.execute:
        error_log = AUDIT / "STAGE3_LOCAL_DELETE_ERRORS.log"
        executed = AUDIT / "STAGE3_LOCAL_EXECUTED_DELETE_MANIFEST.json"
        for entry, path, registry, stats in validated:
            try:
                shutil.rmtree(path)
                deleted_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
                summary_path = registry / "RUN_SUMMARY.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary.update({
                    "deletion_time": deleted_at,
                    "deletion_success": True,
                    "deleted_tree_bytes": stats[0],
                    "deleted_tree_file_count": stats[1],
                    "deleted_tree_directory_count": stats[2],
                })
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (registry / "RUN_SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
                result["deleted"].append({
                    "run_name": path.name,
                    "local_path": str(path),
                    "registry_path": str(registry),
                    "deleted_at": deleted_at,
                    "deleted_bytes": stats[0],
                    "deleted_file_count": stats[1],
                    "deleted_directory_count": stats[2],
                    "recoverable": False,
                })
            except Exception as exc:
                failure = {"run_name": path.name, "path": str(path), "error": str(exc)}
                result["delete_failures"].append(failure)
                with error_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        data["execute_status"] = "SUCCESS" if len(result["deleted"]) == 7 and not result["delete_failures"] else "PARTIAL"
        data["executed_at"] = result["timestamp"]
        data["deleted_entries"] = result["deleted"]
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        executed.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

