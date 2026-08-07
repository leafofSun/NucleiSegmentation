#!/usr/bin/env python3
"""Extract the formal RSGR Local-5 bank from the audited L1-A CONCH bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rsgr_local5 import (
    DEFAULT_SCHEMA_PATH,
    attributes_for_group,
    load_local5_schema,
    sha256_file,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_json_save(payload: object, path: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--source-bank",
        default="workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.pt",
    )
    parser.add_argument(
        "--source-metadata",
        default="workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.json",
    )
    parser.add_argument("--output", default="workdir/text_banks/rsgr_local5_conch_v1.pt")
    args = parser.parse_args()

    schema_path = Path(args.schema)
    source_path = Path(args.source_bank)
    source_meta_path = Path(args.source_metadata)
    output_path = Path(args.output)
    output_meta_path = output_path.with_suffix(".metadata.json")
    schema = load_local5_schema(schema_path)
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if source_meta.get("pt_sha256") != sha256_file(source_path):
        raise ValueError("audited L1-A source bank SHA256 mismatch")
    if source.get("backend") != "conch" or source_meta.get("backend") != "conch":
        raise ValueError("source prototype bank is not CONCH")
    embeddings = source.get("embeddings")
    if not torch.is_tensor(embeddings) or tuple(embeddings.shape) != (5, 3, 512):
        raise ValueError("unexpected L1-A source embedding shape")
    source_names = list(source.get("attribute_names", ()))
    if source_names != source_meta.get("attribute_names"):
        raise ValueError("source bank and metadata attribute order differ")
    source_index = {name: index for index, name in enumerate(source_names)}

    banks = {}
    grouped_names = {}
    prompts = {}
    for group in ("structure", "boundary"):
        rows = attributes_for_group(schema, group)
        grouped_names[group] = [row["name"] for row in rows]
        indices = []
        for row in rows:
            source_name = row["label_source_name"]
            if source_name not in source_index:
                raise KeyError(f"source bank lacks {source_name}")
            if list(row["prompt_texts"]) != source_meta["prompts"][source_name]:
                raise ValueError(f"schema/source prompts differ for {row['name']}")
            indices.append(source_index[source_name])
            prompts[row["name"]] = list(row["prompt_texts"])
        banks[group] = embeddings[indices].detach().float().contiguous()

    payload = {
        "schema_version": "rsgr_local5_conch_bank_v1",
        "backend": "conch",
        "model_name": source["model_name"],
        "attribute_names": grouped_names,
        "class_names": list(schema["classes"]),
        "structure_prototypes": banks["structure"],
        "boundary_prototypes": banks["boundary"],
    }
    atomic_torch_save(payload, output_path)
    metadata = {
        "schema_version": "rsgr_local5_conch_bank_v1",
        "bank_name": "rsgr_local5_conch_v1",
        "schema_name": schema["schema_name"],
        "backend": "conch",
        "encoder": "CONCH",
        "model_name": source["model_name"],
        "embedding_dim": 512,
        "attribute_names": grouped_names,
        "class_names": list(schema["classes"]),
        "prompts": prompts,
        "prompt_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "shapes": {
            "structure": list(banks["structure"].shape),
            "boundary": list(banks["boundary"].shape),
        },
        "dtype": "torch.float32",
        "finite": bool(all(torch.isfinite(bank).all() for bank in banks.values())),
        "normalized": True,
        "norm_min": float(min(bank.norm(dim=-1).min() for bank in banks.values())),
        "norm_max": float(max(bank.norm(dim=-1).max() for bank in banks.values())),
        "tensor_sha256": {group: tensor_sha256(bank) for group, bank in banks.items()},
        "bank_sha256": sha256_file(output_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "source_bank_path": str(source_path),
        "source_bank_sha256": sha256_file(source_path),
        "source_metadata_path": str(source_meta_path),
        "source_metadata_sha256": sha256_file(source_meta_path),
        "encoder_source": source["checkpoint_path"],
        "encoder_sha256": source["checkpoint_sha256"],
        "encoder_checkpoint_sha256": source["checkpoint_sha256"],
        "tokenizer": "CONCH tokenizer from audited L1-A source",
        "prompt_file": str(schema_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "internet_used": False,
        "gpu_used": False,
        "extraction": "verified index extraction; no encoder execution",
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if not metadata["finite"] or metadata["norm_min"] < 1.0 - 1e-5 or metadata["norm_max"] > 1.0 + 1e-5:
        raise ValueError("extracted Local-5 bank failed finite/norm validation")
    atomic_json_save(metadata, output_meta_path)
    print(json.dumps({"bank": str(output_path), "metadata": str(output_meta_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
