#!/usr/bin/env python3
"""Atomic status-file helper for the P4.1 GPU pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--path", type=Path, required=True)
    init.add_argument("--target-mode", default="direct_area_soft")
    init.add_argument("--seed", type=int, default=42)
    update = sub.add_parser("update")
    update.add_argument("--path", type=Path, required=True)
    update.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    gate = sub.add_parser("apply-gate")
    gate.add_argument("--path", type=Path, required=True)
    gate.add_argument("--gate-json", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "init":
        payload = {
            "schema_version": "p41_gpu_pipeline_status_v1",
            "audit_started": False,
            "audit_passed": False,
            "audit_failures": [],
            "formal_training_started": False,
            "formal_training_completed": False,
            "formal_training_exit_code": None,
            "target_mode": args.target_mode,
            "seed": args.seed,
            "interrupted": False,
            "current_stage": "initialized",
        }
        atomic_write(args.path, payload)
        return 0

    if args.command == "update":
        payload = json.loads(args.path.read_text(encoding="utf-8")) if args.path.exists() else {}
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"Invalid --set value: {item!r}")
            key, raw = item.split("=", 1)
            payload[key] = parse_value(raw)
        atomic_write(args.path, payload)
        return 0

    payload = json.loads(args.path.read_text(encoding="utf-8")) if args.path.exists() else {}
    gate_payload = json.loads(args.gate_json.read_text(encoding="utf-8"))
    payload["audit_passed"] = bool(gate_payload.get("passed", False))
    payload["audit_failures"] = list(gate_payload.get("failures", []))
    payload["audit_gate"] = gate_payload
    payload["current_stage"] = "audit_passed" if payload["audit_passed"] else "audit_failed"
    atomic_write(args.path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
