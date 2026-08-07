#!/usr/bin/env python3
"""Build the frozen L1-A CONCH text prototype bank once, offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn import functional as F

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from training.local_region_text_alignment import (
    ATTRIBUTE_NAMES,
    PROMPT_BANK,
    sha256_file,
)


def strict_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cache-dir", default="/hy-tmp/NuSeg/hf_cache/hub")
    parser.add_argument(
        "--checkpoint",
        default="/hy-tmp/NuSeg/hf_cache/hub/models--MahmoodLab--conch/"
        "snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16",
        str(checkpoint),
        device=args.device,
        cache_dir=args.cache_dir,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen_parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = get_tokenizer()
    prompts = [text for name in ATTRIBUTE_NAMES for text in PROMPT_BANK[name]]
    tokenized = tokenizer(
        prompts, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    )
    tokens = tokenized["input_ids"].to(args.device)
    with torch.inference_mode():
        embeddings = model.encode_text(tokens).float()
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
    embeddings = embeddings.reshape(len(ATTRIBUTE_NAMES), 3, -1).cpu()
    if tuple(embeddings.shape[:2]) != (5, 3):
        raise RuntimeError(f"unexpected bank shape: {tuple(embeddings.shape)}")
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0):
        raise RuntimeError("prototype normalization failed")

    output_pt = Path(args.output_pt)
    output_json = Path(args.output_json)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "l1a_text_prototype_bank_v1",
        "backend": "conch",
        "model_name": "conch_ViT-B-16",
        "source": "local_checkpoint",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "attribute_names": ATTRIBUTE_NAMES,
        "prompts": PROMPT_BANK,
        "embeddings": embeddings,
        "normalized": True,
        "requires_grad": False,
        "frozen_conch_parameter_count": frozen_parameter_count,
        "trainable_conch_parameter_count": trainable_parameter_count,
    }
    torch.save(payload, output_pt)
    tensor_sha = hashlib.sha256(
        embeddings.contiguous().numpy().tobytes()
    ).hexdigest()
    metadata = {
        "schema_version": "l1a_text_prototype_bank_v1",
        "backend": "conch",
        "model_name": "conch_ViT-B-16",
        "source": "local_checkpoint",
        "offline": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "attribute_names": list(ATTRIBUTE_NAMES),
        "prompts": {name: list(PROMPT_BANK[name]) for name in ATTRIBUTE_NAMES},
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "requires_grad": False,
        "tensor_sha256": tensor_sha,
        "frozen_conch_parameter_count": frozen_parameter_count,
        "trainable_conch_parameter_count": trainable_parameter_count,
    }
    strict_json(output_json, metadata)
    metadata["pt_sha256"] = sha256_file(output_pt)
    strict_json(output_json, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
