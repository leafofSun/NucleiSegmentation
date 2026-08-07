"""Schema-driven RSGR Local-5 label and prototype utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("rsgr_local5_schema.json")
EXPECTED_GROUP_COUNTS = {"structure": 3, "boundary": 2}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_local5_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> Dict[str, Any]:
    schema_path = Path(path)
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != "rsgr_local5_v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported RSGR Local-5 schema version")
    attributes = payload.get("attributes")
    if not isinstance(attributes, list) or len(attributes) != 5:
        raise ValueError("RSGR Local-5 schema must define exactly five attributes")
    names = [row.get("name") for row in attributes]
    if len(set(names)) != len(names):
        raise ValueError("RSGR Local-5 attribute names must be unique")
    required = {
        "name", "label_source_name", "group", "class_count", "label_generator",
        "threshold_source", "thresholds", "values", "train_split", "code",
        "audit", "prompt_texts", "threshold_values",
        "training_split_source", "existing_code_location", "existing_audit_source",
    }
    counts = {key: 0 for key in EXPECTED_GROUP_COUNTS}
    for row in attributes:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Local-5 attribute is missing fields: {missing}")
        group = row["group"]
        if group not in counts:
            raise ValueError(f"unsupported Local-5 group: {group}")
        counts[group] += 1
        if row["class_count"] != 3 or len(row["values"]) != 3 or len(row["prompt_texts"]) != 3:
            raise ValueError(f"Local-5 attribute {row['name']} must have three classes")
        if row["train_split"] != "train":
            raise ValueError("Local-5 thresholds must be train-fitted")
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"Local-5 group counts must be {EXPECTED_GROUP_COUNTS}, got {counts}")
    return payload


def attributes_for_group(schema: Mapping[str, Any], group: str) -> Tuple[Mapping[str, Any], ...]:
    return tuple(row for row in schema["attributes"] if row["group"] == group)


def split_local5_labels(
    labels: torch.Tensor,
    source_names: Sequence[str],
    schema: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the schema-declared 3+2 labels without dummy heads."""
    if labels.ndim != 3 or labels.shape[1] != 4 or labels.shape[2] != len(source_names):
        raise ValueError("local labels must be [B,4,len(source_names)]")
    if len(set(source_names)) != len(source_names):
        raise ValueError("source label names must be unique")
    source_index = {name: index for index, name in enumerate(source_names)}
    grouped = []
    for group in ("structure", "boundary"):
        indices = []
        for row in attributes_for_group(schema, group):
            source_name = row["label_source_name"]
            if source_name not in source_index:
                raise KeyError(f"Local-5 label source is unavailable: {source_name}")
            indices.append(source_index[source_name])
        grouped.append(labels[..., indices].long())
    return grouped[0], grouped[1]


def local5_classification_stats(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Per-attribute valid count, accuracy, and macro-F1 for logging/audit."""
    if labels.shape != logits.shape[:-1] or logits.shape[-1] != 3:
        raise ValueError("Local-5 metric label/logit shapes differ")
    rows = []
    prediction = logits.detach().argmax(dim=-1)
    for index in range(logits.shape[2]):
        target = labels[..., index].detach()
        valid = target >= 0
        count = int(valid.sum().item())
        if not count:
            rows.append({"valid_count": 0, "accuracy": None, "macro_f1": None})
            continue
        y_true = target[valid]
        y_pred = prediction[..., index][valid]
        accuracy = float((y_true == y_pred).float().mean().item())
        f1_values = []
        for class_index in range(3):
            tp = int(((y_true == class_index) & (y_pred == class_index)).sum().item())
            fp = int(((y_true != class_index) & (y_pred == class_index)).sum().item())
            fn = int(((y_true == class_index) & (y_pred != class_index)).sum().item())
            denominator = 2 * tp + fp + fn
            f1_values.append(0.0 if denominator == 0 else (2.0 * tp / denominator))
        rows.append({"valid_count": count, "accuracy": accuracy, "macro_f1": sum(f1_values) / 3.0})
    return {"attributes": rows, "valid_count": sum(row["valid_count"] for row in rows)}
