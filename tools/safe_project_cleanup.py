#!/usr/bin/env python3
"""Delete only pre-audited LOW-risk NuSeg artifacts with path guards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/hy-tmp/NuSeg").resolve()
AUDIT = ROOT / "workdir" / "storage_audit"
MANIFEST = AUDIT / "DRY_RUN_DELETE_MANIFEST.json"
PROTECTED = AUDIT / "PROTECTED_ASSETS.json"
EXECUTED = AUDIT / "EXECUTED_DELETE_MANIFEST.json"
ERRORS = AUDIT / "DELETE_ERRORS.log"
ALLOWED_TYPES = {
    "SAFE_WHITELIST_TEMP_FILE",
    "CACHE_DIRECTORY",
    "EMPTY_TORCHELASTIC_TEMP_DIRECTORY",
}


def fail(message: str) -> None:
    raise ValueError(message)


def is_within_root(path: Path) -> bool:
    return path != ROOT and ROOT in path.parents


def git_tracked() -> set[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    if proc.returncode:
        return set()
    return {(ROOT / p.decode()).resolve() for p in proc.stdout.split(b"\0") if p}


def open_paths() -> set[Path]:
    result: set[Path] = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            links = list(fd_dir.iterdir())
        except OSError:
            continue
        for link in links:
            try:
                target = Path(os.readlink(link))
                if target.is_absolute() and (target == ROOT or ROOT in target.parents):
                    result.add(target)
            except OSError:
                pass
    return result


def symlink_targets() -> set[Path]:
    result = set()
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=False)
                if target == ROOT or ROOT in target.parents:
                    result.add(target)
            except OSError:
                pass
    return result


def tree_stats(path: Path) -> tuple[int, int, int]:
    if path.is_file():
        return path.stat().st_size, 1, 0
    size = files = dirs = 0
    for base, names, filenames in os.walk(path, followlinks=False):
        dirs += 1
        base_path = Path(base)
        for name in filenames:
            item = base_path / name
            if item.is_symlink():
                continue
            try:
                size += item.stat().st_size
                files += 1
            except OSError:
                pass
    return size, files, dirs


def validate(
    entry: dict, protected: set[Path], tracked: set[Path],
    opened: set[Path], targets: set[Path],
) -> tuple[Path, tuple[int, int, int]]:
    raw = entry.get("path")
    if not raw:
        fail("empty path")
    path = Path(raw)
    if not path.is_absolute():
        fail(f"non-absolute path: {raw}")
    if path.is_symlink():
        fail(f"symlink deletion prohibited: {path}")
    resolved = path.resolve(strict=True)
    if not is_within_root(resolved):
        fail(f"path outside root or root itself: {resolved}")
    if resolved == AUDIT or AUDIT in resolved.parents:
        fail(f"audit records are protected: {resolved}")
    if entry.get("risk_level") != "LOW" or entry.get("auto_delete_allowed") is not True:
        fail(f"entry is not approved LOW risk: {resolved}")
    if entry.get("candidate_type") not in ALLOWED_TYPES:
        fail(f"candidate type not whitelisted: {resolved}")
    members = [resolved]
    if resolved.is_dir():
        members.extend(p.resolve(strict=False) for p in resolved.rglob("*"))
    if any(member in protected for member in members):
        fail(f"protected asset inside target: {resolved}")
    if any(member in tracked for member in members):
        fail(f"Git tracked asset inside target: {resolved}")
    if any(member in opened for member in members):
        fail(f"open file/path inside target: {resolved}")
    if any(member in targets for member in members):
        fail(f"symlink target inside target: {resolved}")
    if any(member.is_symlink() for member in members):
        fail(f"target tree contains a symlink: {resolved}")
    return resolved, tree_stats(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protected_data = json.loads(PROTECTED.read_text(encoding="utf-8"))
    protected = {Path(item["path"]).resolve(strict=False) for item in protected_data}
    tracked = git_tracked()
    opened = open_paths()
    targets = symlink_targets()
    validated, failures = [], []
    for entry in data.get("entries", []):
        try:
            path, stats = validate(entry, protected, tracked, opened, targets)
            validated.append((entry, path, stats))
        except Exception as exc:
            failures.append({"path": entry.get("path"), "error": str(exc)})

    expected = data.get("counts", {})
    calculated = {
        "files": sum(stats[1] for _, _, stats in validated),
        "directories": sum(stats[2] for _, _, stats in validated),
        "bytes": sum(stats[0] for _, _, stats in validated),
    }
    # Manifest target counts count top-level objects; tree counts are intentionally reported separately.
    output = {
        "mode": "dry-run" if args.dry_run else "execute",
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "manifest_target_counts": {
            "files": expected.get("files", 0),
            "directories": expected.get("directories", 0),
            "bytes": expected.get("bytes", 0),
        },
        "validated_tree_counts": calculated,
        "validated_targets": [str(path) for _, path, _ in validated],
        "validation_failures": failures,
        "deleted": [],
        "delete_failures": [],
    }
    if args.execute:
        for entry, path, stats in validated:
            record = {
                "path": str(path), "size_bytes": stats[0], "file_count": stats[1],
                "directory_count": stats[2], "mtime": dt.datetime.fromtimestamp(
                    path.stat().st_mtime, dt.timezone.utc
                ).astimezone().isoformat(),
                "candidate_type": entry["candidate_type"], "reason": entry["reason"],
            }
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                record["deleted_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
                output["deleted"].append(record)
            except Exception as exc:
                failure = {"path": str(path), "error": str(exc)}
                output["delete_failures"].append(failure)
                with ERRORS.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        EXECUTED.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
