#!/usr/bin/env python3
"""Conservative, project-local storage audit for NuSeg."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path


ROOT = Path("/hy-tmp/NuSeg").resolve()
AUDIT = ROOT / "workdir" / "storage_audit"
MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx"}
SOURCE_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".c", ".cpp", ".cu", ".h", ".hpp",
    ".toml", ".ini", ".cfg", ".yaml", ".yml",
}
REPRO_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".log", ".md", ".txt", ".patch", ".diff"}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar")
CACHE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", "htmlcov",
}
TEMP_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".tmp", ".temp", ".part", "~")
PROTECTED_DIR_TOKENS = {
    "data", "datasets", "pannuke", "images", "masks", "labels", "annotations",
    "splits", "train", "val", "test", "models", "logs", "results", "audits",
    "attr_stats", "hf_cache",
}
EXPERIMENT_PARENTS = ("models", "runs", "logs", "results")


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone().isoformat()


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.strip()


def project_size() -> int:
    total = 0
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        base_path = Path(base)
        if base_path == AUDIT:
            dirs[:] = []
            continue
        for name in files:
            path = base_path / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                pass
    return total


def snapshot() -> dict:
    usage = shutil.disk_usage(ROOT)
    git_status = run(["git", "status", "--short", "--branch"])
    git_root = run(["git", "rev-parse", "--show-toplevel"])
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "project_size_bytes_logical": project_size(),
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes": usage.free,
        "git_status_exit": git_status[0],
        "git_status": git_status[1],
        "git_root_exit": git_root[0],
        "git_root": git_root[1],
    }


def git_tracked() -> set[Path]:
    code, output = run(["git", "ls-files", "-z"])
    if code:
        return set()
    return {(ROOT / item).resolve() for item in output.split("\0") if item}


def active_processes() -> list[dict]:
    found = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        low = raw.lower()
        if raw and ("torchrun" in low or "train.py" in low or "test.py" in low or "nuseg" in low):
            if "storage_audit.py" not in raw and "safe_project_cleanup.py" not in raw:
                found.append({"pid": int(entry.name), "cmdline": raw})
    return found


def open_paths() -> set[Path]:
    paths: set[Path] = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            links = list(fd_dir.iterdir())
        except OSError:
            continue
        for link in links:
            try:
                target = Path(os.readlink(link))
                if target.is_absolute() and (target == ROOT or ROOT in target.parents):
                    paths.add(target)
            except OSError:
                pass
    return paths


def symlink_targets() -> tuple[list[dict], set[Path]]:
    links, targets = [], set()
    for path in ROOT.rglob("*"):
        if AUDIT == path or AUDIT in path.parents:
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=False)
                links.append({"path": str(path), "target": str(target)})
                if target == ROOT or ROOT in target.parents:
                    targets.add(target)
            except OSError as exc:
                links.append({"path": str(path), "target": None, "error": str(exc)})
    return links, targets


def is_checkpoint(path: Path) -> bool:
    low = path.name.lower()
    return path.suffix.lower() in MODEL_SUFFIXES or any(
        token in low for token in ("checkpoint", "optimizer", "scheduler", "scaler", "ema")
    )


def source_like(path: Path) -> bool:
    low = path.name.lower()
    return (
        path.suffix.lower() in SOURCE_SUFFIXES
        or low.startswith(("dockerfile", "makefile", "requirements", "environment"))
    )


def protection_reason(path: Path, tracked: set[Path], link_targets: set[Path]) -> str:
    parts = {part.lower() for part in path.relative_to(ROOT).parts}
    low = path.name.lower()
    if path in tracked:
        return "GIT_TRACKED_FILE"
    if path in link_targets:
        return "SYMLINK_TARGET"
    if source_like(path):
        return "SOURCE_SCRIPT_OR_CONFIG"
    if is_checkpoint(path):
        return "MODEL_OR_CHECKPOINT"
    if parts & {"data", "datasets", "pannuke", "images", "masks", "labels", "annotations", "splits"}:
        return "DATASET_OR_ANNOTATION"
    if "hf_cache" in parts or "clip" in parts or ".conda" in parts:
        return "MODEL_CACHE_OR_RESEARCH_DEPENDENCY"
    if path.suffix.lower() in REPRO_SUFFIXES:
        return "REPRODUCIBILITY_ASSET"
    if parts & {"models", "logs", "results", "audits", "attr_stats"}:
        return "EXPERIMENT_ASSET"
    if any(token in low for token in ("prompt", "text_bank", "knowledge", "report", "readme")):
        return "RESEARCH_KNOWLEDGE_OR_DOCUMENTATION"
    return "UNKNOWN_DO_NOT_DELETE"


def is_failed_scene(path: Path) -> bool:
    low = str(path.relative_to(ROOT)).lower()
    return any(token in low for token in ("failed", "error", "diagnos", "audit"))


def low_file_candidate(path: Path) -> bool:
    name = path.name
    low = name.lower()
    if name in {".DS_Store", "Thumbs.db", ".coverage"}:
        return True
    if low.endswith(TEMP_SUFFIXES):
        return True
    return False


def directory_sizes(files: list[dict]) -> dict[Path, int]:
    sizes: collections.Counter[Path] = collections.Counter()
    for item in files:
        path = Path(item["path"])
        size = item["size_bytes"]
        parent = path.parent
        while parent == ROOT or ROOT in parent.parents:
            sizes[parent] += size
            if parent == ROOT:
                break
            parent = parent.parent
    return dict(sizes)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_groups(files: list[dict]) -> list[dict]:
    by_size: dict[int, list[Path]] = collections.defaultdict(list)
    for item in files:
        if item["size_bytes"] > 0:
            by_size[item["size_bytes"]].append(Path(item["path"]))
    exact = []
    for size, paths in sorted(by_size.items(), reverse=True):
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = collections.defaultdict(list)
        for path in paths:
            try:
                by_hash[sha256(path)].append(path)
            except OSError:
                continue
        for digest, matches in by_hash.items():
            if len(matches) > 1:
                exact.append({
                    "size_bytes_each": size,
                    "sha256": digest,
                    "paths": [str(path) for path in sorted(matches)],
                    "contains_checkpoint": any(is_checkpoint(path) for path in matches),
                    "contains_protected_type": any(
                        is_checkpoint(path) or source_like(path) or path.suffix.lower() in REPRO_SUFFIXES
                        or bool({p.lower() for p in path.relative_to(ROOT).parts} & PROTECTED_DIR_TOKENS)
                        for path in matches
                    ),
                    "auto_delete_allowed": False,
                    "reason": "Exact duplicates are reported only; protected/research assets are never auto-deleted.",
                })
    return exact


def archive_review(paths: list[Path]) -> list[dict]:
    review = []
    for path in paths:
        overview = ""
        try:
            low = path.name.lower()
            if low.endswith(".zip"):
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    overview = f"{len(names)} entries: {', '.join(names[:8])}"
            elif low.endswith((".tar", ".tar.gz", ".tgz")):
                with tarfile.open(path) as archive:
                    names = archive.getnames()
                    overview = f"{len(names)} entries: {', '.join(names[:8])}"
            else:
                overview = "Listing unsupported without optional archive tools."
        except Exception as exc:
            overview = f"Could not inspect safely: {exc}"
        st = path.stat()
        review.append({
            "path": str(path), "size_bytes": st.st_size, "mtime": iso(st.st_mtime),
            "content_overview": overview, "extracted_directory": None,
            "recommendation": "REPORT_ONLY_SOURCE_OR_PURPOSE_UNCLEAR",
            "auto_delete_safe": False,
        })
    return review


def experiment_review(dir_sizes: dict[Path, int], active: list[dict]) -> list[dict]:
    rows = []
    active_text = "\n".join(item["cmdline"] for item in active)
    for parent_name in EXPERIMENT_PARENTS:
        parent = ROOT / "workdir" / parent_name
        if not parent.exists():
            continue
        children = [p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()]
        if parent_name == "models":
            children += [parent]
        for child in sorted(set(children)):
            files = [p for p in child.rglob("*") if p.is_file() and not p.is_symlink()]
            checkpoint_count = sum(is_checkpoint(p) for p in files)
            names = [p.name.lower() for p in files]
            active_use = str(child) in active_text or child.name in active_text
            if active_use:
                category = "ACTIVE_RUN_PROTECTED"
            elif child.name == "Visual_baseline":
                category = "CORE_BASELINE_PROTECTED"
            elif checkpoint_count and any(name.startswith("best") for name in names):
                category = "BEST_RESULT_PROTECTED"
            elif is_failed_scene(child) and files:
                category = "FAILED_DEBUG_SCENE_PROTECTED"
            elif checkpoint_count or any(p.suffix.lower() in REPRO_SUFFIXES for p in files):
                category = "REPRODUCIBLE_EXPERIMENT_PROTECTED"
            elif not files:
                category = "STALE_RUN_CANDIDATE"
            else:
                category = "UNKNOWN_DO_NOT_DELETE"
            mtimes = [p.stat().st_mtime for p in files] or [child.stat().st_mtime]
            rows.append({
                "experiment_dir": str(child),
                "parent_type": parent_name,
                "size_bytes": dir_sizes.get(child, sum(p.stat().st_size for p in files)),
                "file_count": len(files),
                "checkpoint_count": checkpoint_count,
                "has_best_checkpoint": any(is_checkpoint(p) and p.name.lower().startswith("best") for p in files),
                "has_latest_checkpoint": any(is_checkpoint(p) and p.name.lower().startswith(("latest", "last")) for p in files),
                "has_config": any(source_like(p) and p.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"} for p in files),
                "has_log": any(p.suffix.lower() == ".log" or "events.out.tfevents" in p.name for p in files),
                "has_result_json": any(p.suffix.lower() in {".json", ".jsonl"} for p in files),
                "has_test_result": any("test" in p.name.lower() for p in files),
                "last_modified": iso(max(mtimes)),
                "active": active_use,
                "success_confirmed": bool(checkpoint_count and any(name.startswith("best") for name in names)),
                "superseded_confirmed": False,
                "category": category,
            })
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not ROOT.is_dir() or ROOT == Path("/"):
        raise SystemExit("Unsafe or missing project root")
    AUDIT.mkdir(parents=True, exist_ok=True)
    tracked = git_tracked()
    active = active_processes()
    opened = open_paths()
    links, link_targets = symlink_targets()
    files = []
    archives = []
    checkpoints = []
    scan_errors = []
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        base_path = Path(base)
        if base_path == AUDIT:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not (base_path / d).is_symlink()]
        for name in names:
            path = base_path / name
            try:
                if path.is_symlink():
                    continue
                st = path.stat()
                if not stat.S_ISREG(st.st_mode):
                    continue
                item = {"path": str(path), "size_bytes": st.st_size, "mtime": iso(st.st_mtime)}
                files.append(item)
                if path.name.lower().endswith(ARCHIVE_SUFFIXES):
                    archives.append(path)
                if is_checkpoint(path):
                    checkpoints.append(item)
            except OSError as exc:
                scan_errors.append({"path": str(path), "error": str(exc)})

    sizes = directory_sizes(files)
    duplicates = duplicate_groups(files)
    experiments = experiment_review(sizes, active)
    protected = []
    candidates = []
    protected_paths: set[Path] = set()

    for item in files:
        path = Path(item["path"])
        reason = protection_reason(path, tracked, link_targets)
        if low_file_candidate(path):
            if is_failed_scene(path) or reason in {
                "SOURCE_SCRIPT_OR_CONFIG", "MODEL_OR_CHECKPOINT", "DATASET_OR_ANNOTATION",
                "REPRODUCIBILITY_ASSET", "MODEL_CACHE_OR_RESEARCH_DEPENDENCY",
            }:
                candidates.append({
                    **item, "candidate_type": "TEMP_NAME_IN_PROTECTED_CONTEXT",
                    "reason": "Temporary-looking name is inside a protected research/debug context.",
                    "risk_level": "HIGH", "auto_delete_allowed": False,
                    "evidence": reason, "retained_copy": None,
                })
            elif path in opened or path in link_targets:
                candidates.append({
                    **item, "candidate_type": "OPEN_OR_SYMLINK_TARGET",
                    "reason": "Would match a whitelist but is open or a symlink target.",
                    "risk_level": "HIGH", "auto_delete_allowed": False,
                    "evidence": "OPEN_FILE_OR_SYMLINK_TARGET", "retained_copy": None,
                })
            else:
                candidates.append({
                    **item, "candidate_type": "SAFE_WHITELIST_TEMP_FILE",
                    "reason": "Explicit temporary/cache filename whitelist.",
                    "risk_level": "LOW", "auto_delete_allowed": True,
                    "evidence": "Filename whitelist; not tracked/open/protected/symlink target.",
                    "retained_copy": None,
                })
                continue
        protected.append({**item, "type": "file", "protection_reason": reason, "sha256": None})
        protected_paths.add(path)

    # The audit directory is always protected, even though generated outputs are excluded from the scan.
    protected.append({
        "path": str(AUDIT), "size_bytes": 0, "type": "directory",
        "protection_reason": "STORAGE_AUDIT_RECORDS", "mtime": iso(AUDIT.stat().st_mtime), "sha256": None,
    })

    # Explicit cache/build directories: installed environments and package metadata remain report-only.
    for path in ROOT.rglob("*"):
        if path == AUDIT or AUDIT in path.parents or path.is_symlink() or not path.is_dir():
            continue
        if path.name in CACHE_DIR_NAMES:
            contained = [p for p in path.rglob("*") if p.is_file()]
            if any(p in protected_paths or p in opened or p in link_targets for p in contained):
                risk, allowed, evidence = "HIGH", False, "Contains a protected/open/symlink-target file."
            else:
                risk, allowed, evidence = "LOW", True, "Explicit cache-directory whitelist; contents are unprotected."
            candidates.append({
                "path": str(path), "size_bytes": sizes.get(path, 0),
                "candidate_type": "CACHE_DIRECTORY", "reason": "Explicit cache directory whitelist.",
                "risk_level": risk, "auto_delete_allowed": allowed,
                "evidence": evidence, "retained_copy": None,
                "mtime": iso(path.stat().st_mtime),
            })
        elif path.name in {"build", "dist"} or path.name.endswith(".egg-info"):
            candidates.append({
                "path": str(path), "size_bytes": sizes.get(path, 0),
                "candidate_type": "BUILD_OR_PACKAGE_METADATA_REVIEW",
                "reason": "Name resembles a build artifact, but it is inside an installed environment/dependency tree.",
                "risk_level": "MEDIUM", "auto_delete_allowed": False,
                "evidence": "Runtime dependence cannot be excluded.", "retained_copy": None,
                "mtime": iso(path.stat().st_mtime),
            })

    # Root temporary source directories are protected; the empty torchelastic tree is safe.
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir() or path.is_symlink():
            continue
        if re.fullmatch(r"tmp[a-z0-9_]+", path.name):
            candidates.append({
                "path": str(path), "size_bytes": sizes.get(path, 0),
                "candidate_type": "TEMP_DIRECTORY_WITH_PROTECTED_SOURCE",
                "reason": "Temporary directory contains protected .py source.",
                "risk_level": "HIGH", "auto_delete_allowed": False,
                "evidence": "Absolute source-file protection rule.", "retained_copy": None,
                "mtime": iso(path.stat().st_mtime),
            })
        elif path.name.startswith("torchelastic_"):
            contained_files = [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
            if not contained_files and str(path) not in "\n".join(p["cmdline"] for p in active):
                candidates.append({
                    "path": str(path), "size_bytes": sizes.get(path, 0),
                    "candidate_type": "EMPTY_TORCHELASTIC_TEMP_DIRECTORY",
                    "reason": "Old project-local torch elastic temp tree contains no files or diagnostics.",
                    "risk_level": "LOW", "auto_delete_allowed": True,
                    "evidence": "No files, logs, checkpoints, active process, or open path.", "retained_copy": None,
                    "mtime": iso(path.stat().st_mtime),
                })

    for row in experiments:
        if row["category"] == "STALE_RUN_CANDIDATE":
            candidates.append({
                "path": row["experiment_dir"], "size_bytes": row["size_bytes"],
                "candidate_type": "STALE_RUN_CANDIDATE",
                "reason": "Empty experiment directory; scientific intent/supersession is not proven.",
                "risk_level": "MEDIUM", "auto_delete_allowed": False,
                "evidence": "No files, but conservative experiment-directory policy applies.",
                "retained_copy": None, "mtime": row["last_modified"],
            })

    existing_candidate_paths = {item["path"] for item in candidates}
    for group in duplicates:
        retained = group["paths"][0]
        for duplicate in group["paths"][1:]:
            if duplicate in existing_candidate_paths:
                continue
            path = Path(duplicate)
            risk = "HIGH" if group["contains_protected_type"] else "MEDIUM"
            candidates.append({
                "path": duplicate, "size_bytes": group["size_bytes_each"],
                "candidate_type": "EXACT_DUPLICATE_CANDIDATE",
                "reason": "Byte-identical duplicate; deletion is not automatically safe due to path/research semantics.",
                "risk_level": risk, "auto_delete_allowed": False,
                "evidence": f"size and SHA256 match retained copy; sha256={group['sha256']}",
                "retained_copy": retained, "mtime": iso(path.stat().st_mtime),
            })

    top_files = sorted(files, key=lambda x: x["size_bytes"], reverse=True)[:100]
    top_dirs = [
        {"path": str(path), "size_bytes": size}
        for path, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:50]
    ]
    archive_rows = archive_review(archives)
    pre = snapshot()
    summary = {
        "root": str(ROOT),
        "active_processes": active,
        "open_project_paths_count": len(opened),
        "symlinks": links,
        "git_tracked_count": len(tracked),
        "git_metadata_present": (ROOT / ".git").exists(),
        "file_count": len(files),
        "directory_count": len(sizes),
        "checkpoint_count": len(checkpoints),
        "archive_count": len(archive_rows),
        "duplicate_group_count": len(duplicates),
        "scan_errors": scan_errors,
    }
    write_json(AUDIT / "PRE_CLEANUP_SNAPSHOT.json", pre)
    write_json(AUDIT / "AUDIT_SUMMARY.json", summary)
    write_json(AUDIT / "PROTECTED_ASSETS.json", protected)
    write_json(AUDIT / "DELETE_CANDIDATES.json", candidates)
    write_json(AUDIT / "DRY_RUN_DELETE_MANIFEST.json", {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "root": str(ROOT),
        "entries": [item for item in candidates if item["auto_delete_allowed"]],
        "counts": {
            "files": sum(Path(item["path"]).is_file() for item in candidates if item["auto_delete_allowed"]),
            "directories": sum(Path(item["path"]).is_dir() for item in candidates if item["auto_delete_allowed"]),
            "bytes": sum(item["size_bytes"] for item in candidates if item["auto_delete_allowed"]),
            "LOW": sum(item["risk_level"] == "LOW" for item in candidates),
            "MEDIUM": sum(item["risk_level"] == "MEDIUM" for item in candidates),
            "HIGH": sum(item["risk_level"] == "HIGH" for item in candidates),
        },
    })
    (AUDIT / "PROTECTED_ASSETS.txt").write_text(
        "\n".join(f"{item['size_bytes']}\t{item['protection_reason']}\t{item['path']}" for item in protected) + "\n",
        encoding="utf-8",
    )
    (AUDIT / "DELETE_CANDIDATES.txt").write_text(
        "\n".join(
            f"{item['risk_level']}\t{item['auto_delete_allowed']}\t{item['size_bytes']}\t"
            f"{item['candidate_type']}\t{item['path']}\t{item['reason']}"
            for item in candidates
        ) + "\n", encoding="utf-8",
    )
    (AUDIT / "LARGE_FILES_TOP100.txt").write_text(
        "\n".join(f"{item['size_bytes']}\t{human(item['size_bytes'])}\t{item['mtime']}\t{item['path']}" for item in top_files) + "\n",
        encoding="utf-8",
    )
    (AUDIT / "LARGE_DIRECTORIES_TOP50.txt").write_text(
        "\n".join(f"{item['size_bytes']}\t{human(item['size_bytes'])}\t{item['path']}" for item in top_dirs) + "\n",
        encoding="utf-8",
    )
    (AUDIT / "DUPLICATE_FILE_CANDIDATES.txt").write_text(
        "\n\n".join(
            f"SIZE={group['size_bytes_each']} ({human(group['size_bytes_each'])})\n"
            f"SHA256={group['sha256']}\nAUTO_DELETE_ALLOWED=false\n" + "\n".join(group["paths"])
            for group in duplicates
        ) + ("\n" if duplicates else "No exact duplicate groups found.\n"),
        encoding="utf-8",
    )
    (AUDIT / "ARCHIVE_REVIEW.txt").write_text(
        "\n\n".join(
            f"PATH={row['path']}\nSIZE={row['size_bytes']} ({human(row['size_bytes'])})\n"
            f"MTIME={row['mtime']}\nCONTENTS={row['content_overview']}\n"
            f"EXTRACTED_DIRECTORY={row['extracted_directory']}\nRECOMMENDATION={row['recommendation']}\n"
            f"AUTO_DELETE_SAFE={str(row['auto_delete_safe']).lower()}"
            for row in archive_rows
        ) + ("\n" if archive_rows else "No archives found.\n"), encoding="utf-8",
    )
    (AUDIT / "EXPERIMENT_DIRECTORY_REVIEW.txt").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in experiments) + "\n",
        encoding="utf-8",
    )
    (AUDIT / "CHECKPOINT_INVENTORY.txt").write_text(
        "\n".join(f"{item['size_bytes']}\t{item['mtime']}\t{item['path']}" for item in sorted(checkpoints, key=lambda x: x["size_bytes"], reverse=True)) + "\n",
        encoding="utf-8",
    )
    (AUDIT / "CACHE_DIRECTORY_INVENTORY.txt").write_text(
        "\n".join(
            f"{item['risk_level']}\t{item['size_bytes']}\t{item['path']}"
            for item in candidates if "CACHE" in item["candidate_type"] or "BUILD" in item["candidate_type"]
        ) + "\n", encoding="utf-8",
    )
    (AUDIT / "DELETE_ERRORS.log").touch()
    print(json.dumps({"summary": summary, "pre": pre, "candidate_counts": collections.Counter(i["risk_level"] for i in candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
