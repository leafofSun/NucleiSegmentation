#!/usr/bin/env python3
"""Delete only model runs with fully verified remote archives."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/hy-tmp/NuSeg").resolve()
MODELS = (ROOT / "workdir" / "models").resolve()
AUDIT = ROOT / "workdir" / "storage_audit"
INDEX = ROOT / "workdir" / "archived_run_index"
DELETE_MANIFEST = AUDIT / "STAGE2_LOCAL_DELETE_MANIFEST.json"
REMOTE_VERIFY = AUDIT / "STAGE2_REMOTE_VERIFICATION.json"
ERROR_LOG = AUDIT / "STAGE2_DELETE_ERRORS.log"
PROTECTED = {
    "Visual_baseline",
    "phaseB_ml_instancefix_from_visual_3gpu_30ep_v1",
    "phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD",
    "sga_sb_p41_g2_soft_seed42_e5_v1",
}
META_FILES = ("RUN_ARCHIVE_MANIFEST.json", "RUN_ARCHIVE_MANIFEST.txt", "SHA256SUMS")


def inside_models(path: Path) -> bool:
    return path != MODELS and MODELS in path.parents


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


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--local-final-delete", action="store_true")
    parser.add_argument("--user-approved-no-remote", action="store_true")
    parser.add_argument("--run-manifest")
    parser.add_argument("--registry-root")
    args = parser.parse_args()
    if args.local_final_delete:
        from stage3_cleanup_mode import run_stage3
        run_stage3(args)
        return

    delete_data = json.loads(DELETE_MANIFEST.read_text(encoding="utf-8"))
    remote_data = json.loads(REMOTE_VERIFY.read_text(encoding="utf-8"))
    remote = {row["run_name"]: row for row in remote_data.get("runs", [])}
    opened = open_paths()
    validated, failures = [], []
    for entry in delete_data.get("authorized_entries", []):
        try:
            raw = entry.get("local_path")
            if not raw:
                raise ValueError("empty path")
            path = Path(raw)
            if path.is_symlink():
                raise ValueError("symlink run prohibited")
            resolved = path.resolve(strict=True)
            if not inside_models(resolved):
                raise ValueError("path is outside models or is models root")
            if resolved.name in PROTECTED:
                raise ValueError("strong-protected run")
            verification = remote.get(resolved.name)
            if not verification:
                raise ValueError("missing remote verification")
            if not (
                verification.get("ARCHIVE_UPLOAD_STATUS") == "SUCCESS"
                and verification.get("CHECKSUM_VERIFIED") is True
                and verification.get("CRITICAL_FILES_VERIFIED") is True
                and verification.get("local_deletion_authorized") is True
            ):
                raise ValueError("remote archive is not fully verified")
            members = [resolved] + [p.resolve(strict=False) for p in resolved.rglob("*")]
            if any(p.is_symlink() for p in [resolved] + list(resolved.rglob("*"))):
                raise ValueError("run contains symlink")
            if any(member in opened for member in members):
                raise ValueError("run contains open file")
            for name in META_FILES:
                if not (resolved / name).is_file():
                    raise ValueError(f"missing local archive metadata: {name}")
            validated.append((entry, resolved, tree_stats(resolved)))
        except Exception as exc:
            failures.append({"path": entry.get("local_path"), "error": str(exc)})

    result = {
        "mode": "dry-run" if args.dry_run else "execute",
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "validated": [str(path) for _, path, _ in validated],
        "validation_failures": failures,
        "deleted": [],
        "delete_failures": [],
    }
    if args.execute:
        INDEX.mkdir(parents=True, exist_ok=True)
        for entry, path, stats in validated:
            try:
                run_index = INDEX / path.name
                run_index.mkdir(parents=True, exist_ok=True)
                for name in META_FILES:
                    shutil.copy2(path / name, run_index / name)
                summary = {
                    "run_name": path.name,
                    "original_local_path": str(path),
                    "remote_location": entry.get("remote_location", "oss://<BUCKET>/NuSeg-archive/<REDACTED>"),
                    "experiment_conclusion": entry.get("experiment_conclusion"),
                    "deleted_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
                    "deleted_size_bytes": stats[0],
                }
                (run_index / "ARCHIVED_RUN_INDEX.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                shutil.rmtree(path)
                result["deleted"].append({**summary, "file_count": stats[1], "directory_count": stats[2]})
            except Exception as exc:
                failure = {"path": str(path), "error": str(exc)}
                result["delete_failures"].append(failure)
                with ERROR_LOG.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        delete_data["deleted_entries"] = result["deleted"]
        delete_data["execution_timestamp"] = result["timestamp"]
        DELETE_MANIFEST.write_text(json.dumps(delete_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
