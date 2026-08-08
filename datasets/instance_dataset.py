#!/usr/bin/env python3
"""Lossless PanNuke instance-NPZ dataset, parallel to the legacy JSON reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch

import DataLoader as legacy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_and_remap(
    inst_map: np.ndarray,
    inst_type: np.ndarray,
    minimum_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    if minimum_area < 0:
        raise ValueError("min_instance_area must be >= 0")
    count = len(inst_type)
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    if not np.array_equal(ids, np.arange(1, count + 1)):
        raise ValueError("stored inst_map IDs are not continuous 1..N")
    areas = np.bincount(inst_map.ravel(), minlength=count + 1)[1:]
    keep = areas >= minimum_area if minimum_area > 0 else np.ones(count, dtype=bool)
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[np.flatnonzero(keep) + 1] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[inst_map], inst_type[keep].astype(np.int32, copy=False)


def stack_instance_dict_batched(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy-compatible collate with variable-length class vectors."""
    output: dict[str, Any] = {}
    variable = {
        "per_instance_attr_labels",
        "per_instance_attr_values",
        "per_instance_ids",
        "inst_type",
    }
    for key, value in batch[0].items():
        if key in variable:
            output[key] = [sample[key] for sample in batch]
        elif isinstance(value, torch.Tensor):
            output[key] = torch.stack([sample[key] for sample in batch])
        else:
            output[key] = [sample[key] for sample in batch]
    return output


class InstanceNPZDataset(legacy.UniversalDataset):
    """Read R1 NPZ samples without polygonization or data-layer area loss.

    The legacy JSON class remains untouched.  This subclass reuses its prompt,
    augmentation, and target helpers while implementing an independent NPZ
    sample index and item reader.
    """

    def __init__(self, data_root: str, knowledge_path: str, mode: str = "train", **kwargs: Any):
        root = Path(data_root)
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "pannuke-instance-npz-v1":
            raise ValueError(f"unsupported instance dataset format: {manifest.get('format')}")

        # Initialize all unchanged prompt/target/augmentation machinery.  The
        # manifest-backed sample list replaces the legacy scan immediately.
        super().__init__(
            data_root=data_root,
            knowledge_path=knowledge_path,
            mode=mode,
            **kwargs,
        )
        canonical_mode = str(mode).lower().strip()
        folds = (1, 2) if canonical_mode == "train" else (3,)
        selected = [entry for entry in manifest["samples"] if int(entry["fold"]) in folds]
        self.samples = [
            {
                "npz_path": str(root / entry["relative_path"]),
                "data": {
                    "organ_idx": int(entry["tissue_id"]),
                    "organ_id": str(entry["tissue_name"]),
                },
                "rel_path": str(entry["relative_path"]),
                "sample_id": str(entry["sample_id"]),
                "instance_count": int(entry["instance_count"]),
            }
            for entry in selected
        ]
        self.dataset_manifest_sha256 = _sha256_file(manifest_path)
        self.dataset_folds = folds
        self.stored_instance_count = sum(int(entry["instance_count"]) for entry in selected)
        legacy._dataloader_print(
            "[DATA_CONFIG] "
            + json.dumps(
                {
                    "format": "instance_npz",
                    "fold": list(folds),
                    "sample_count": len(self.samples),
                    "instance_count": self.stored_instance_count,
                    "manifest_sha256": self.dataset_manifest_sha256,
                    "min_instance_area": self.min_instance_area,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        json_data = item["data"]
        with np.load(item["npz_path"], allow_pickle=False) as stored:
            image = stored["image"].astype(np.uint8, copy=False)
            raw_inst_map = stored["inst_map"].astype(np.int32, copy=False)
            raw_inst_type = stored["inst_type"].astype(np.int32, copy=False)
        mask, filtered_inst_type = _filter_and_remap(
            raw_inst_map, raw_inst_type, self.min_instance_area
        )

        h, w = image.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            target_h = max(h, self.crop_size)
            target_w = max(w, self.crop_size)
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        geometric = self.geometry_transform(image=image, mask=mask)
        local_aug_mask_inst = np.asarray(geometric["mask"]).astype(np.int32)
        augmented = self.transform(image=geometric["image"], mask=local_aug_mask_inst)
        img_tensor = augmented["image"].float()
        aug_mask_inst = augmented["mask"].numpy().astype(np.int32)
        aug_mask = (aug_mask_inst > 0).astype(np.uint8)

        # Remove instances lost entirely at crop time, then restore continuous
        # IDs and aligned class vectors.  This is augmentation behavior, not a
        # data conversion/filtering rule.
        present = np.unique(aug_mask_inst)
        present = present[present > 0]
        crop_lookup = np.zeros(len(filtered_inst_type) + 1, dtype=np.int32)
        crop_lookup[present] = np.arange(1, len(present) + 1, dtype=np.int32)
        aug_mask_inst = crop_lookup[aug_mask_inst]
        crop_inst_type = filtered_inst_type[present - 1]
        type_lookup = np.zeros(len(crop_inst_type) + 1, dtype=np.uint8)
        if len(crop_inst_type):
            type_lookup[1:] = crop_inst_type.astype(np.uint8)
        aug_type_map = type_lookup[aug_mask_inst]
        aug_mask = (aug_mask_inst > 0).astype(np.uint8)

        local_region_targets = None
        if self.enable_local_region_text_alignment:
            local_region_targets = legacy.compute_local_region_targets(
                local_aug_mask_inst,
                self.local_region_thresholds,
                window_size=self.local_region_window_size,
            )

        area_scale = 1.0
        task_type = "generic"
        text_suffix = ""
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        crop_analysis = legacy.analyze_physical_attributes(
            image=img_np,
            mask=aug_mask,
            config=self.attr_config,
            area_scale=area_scale,
        )
        organ_id, organ_name = self._resolve_organ(json_data)
        organ_dropout_applied = False
        if self.mode == "train" and self.organ_dropout_prob > 0.0:
            if random.random() < self.organ_dropout_prob:
                organ_id = 20
                organ_name = "Generic"
                organ_dropout_applied = True

        if self.prompt_mode == "base":
            attr_labels_np = legacy._to_int_list(
                json_data.get("attr_labels"), default=legacy._default_attr_labels()
            )
            prompt_visuals = legacy._normalise_visual_stats_from_entry(json_data)
            text_prompt, attribute_text, morphology_text = legacy._build_base_prompts()
            attr_source = "base_prompt_no_text_attribute"
        elif self.prompt_mode == "dynamic_gt":
            prompt_visuals = crop_analysis["visuals"]
            attr_labels_np = crop_analysis["labels"]
            text_prompt, attribute_text, morphology_text = legacy.build_pathology_prompts(
                base_prompt=legacy.STRICT_BASE_PROMPT,
                organ_name=organ_name,
                visuals=prompt_visuals,
                task_type=task_type,
                text_suffix=text_suffix,
                prompt_mode=self.prompt_mode,
            )
            attr_source = "crop_dynamic_gt"
        else:
            if organ_dropout_applied:
                labels, prompt_visuals, prompts, attr_source = self._get_organ_prior_payload("Generic")
                attr_labels_np = labels
                text_prompt, attribute_text, morphology_text = prompts
                attr_source += "_after_organ_dropout"
            elif self.is_v2:
                labels, prompt_visuals, prompts, attr_source = self._get_organ_prior_payload(organ_name)
                attr_labels_np = labels
                text_prompt, attribute_text, morphology_text = prompts
            else:
                full_analysis = legacy.analyze_physical_attributes(
                    image=image,
                    mask=(mask > 0).astype(np.uint8),
                    config=self.attr_config,
                    area_scale=1.0,
                )
                prompt_visuals = full_analysis["visuals"]
                attr_labels_np = full_analysis["labels"]
                text_prompt, attribute_text, morphology_text = legacy.build_pathology_prompts(
                    base_prompt=legacy.STRICT_BASE_PROMPT,
                    organ_name=organ_name,
                    visuals=prompt_visuals,
                    task_type=task_type,
                    text_suffix=text_suffix,
                    prompt_mode=self.prompt_mode,
                )
                attr_source = "fallback_full_image_physical"

        label_tensor = torch.from_numpy(aug_mask).long().unsqueeze(0)
        label_inst_tensor = torch.from_numpy(aug_mask_inst).long().unsqueeze(0)
        gt_heatmap = legacy.generate_adaptive_density(
            aug_mask, image_size=(self.image_size, self.image_size)
        )
        gt_heatmap_tensor = torch.from_numpy(gt_heatmap).float().unsqueeze(0)
        gt_hv_map_tensor = torch.from_numpy(legacy.generate_hv_map(aug_mask_inst)).float()
        boundary_radius = max(
            2, int(round(2.0 * float(self.image_size) / float(self.crop_size)))
        )
        structure_targets = legacy.generate_boundary_uncertainty_targets(
            inst_mask=aug_mask_inst,
            boundary_radius=boundary_radius,
            contact_radius=boundary_radius,
        )
        dense_maps = legacy.generate_dense_boundary_maps(aug_mask_inst)
        inst_morph = legacy.compute_instance_morphology_attrs(
            aug_mask_inst,
            min_instance_area=0,
            max_instances_per_image=self.max_instances_per_image,
        )

        invalid = legacy.INVALID_ATTR_LABEL
        structure_labels = [invalid] * len(legacy.STRUCTURE_ATTR_NAMES)
        boundary_labels = [invalid] * len(legacy.BOUNDARY_ATTR_NAMES)
        structure_values = [0.0] * len(legacy.STRUCTURE_ATTR_NAMES)
        boundary_values = [0.0] * len(legacy.BOUNDARY_ATTR_NAMES)
        result: dict[str, Any] = {
            "image": img_tensor,
            "label": label_tensor,
            "label_inst": label_inst_tensor,
            "type_map": torch.from_numpy(aug_type_map).long().unsqueeze(0),
            "inst_type": torch.from_numpy(crop_inst_type).long(),
            "gt_heatmap": gt_heatmap_tensor,
            "gt_hv_map": gt_hv_map_tensor,
            "fg_target": torch.from_numpy(structure_targets["fg_target"]).float().unsqueeze(0),
            "bg_target": torch.from_numpy(structure_targets["bg_target"]).float().unsqueeze(0),
            "boundary_target": torch.from_numpy(structure_targets["boundary_target"]).float().unsqueeze(0),
            "uncertain_target": torch.from_numpy(structure_targets["uncertain_target"]).float().unsqueeze(0),
            "organ_id": int(organ_id),
            "text_prompt": text_prompt,
            "attribute_text": attribute_text,
            "morphology_text": morphology_text,
            "attr_labels": torch.tensor(attr_labels_np, dtype=torch.long),
            "visual_attributes": prompt_visuals,
            "crop_visual_attributes": crop_analysis["visuals"],
            "metadata_visual_stats": {},
            "metadata_attr_labels": None,
            "attr_label_source": attr_source,
            "organ_dropout_applied": bool(organ_dropout_applied),
            "name": Path(item["npz_path"]).name,
            "rel_path": item["rel_path"],
            "original_size": (self.image_size, self.image_size),
            "task_type": task_type,
            "prompt_mode": self.prompt_mode,
            "requested_prompt_mode": self.requested_prompt_mode,
            "prompt_uses_gt_attributes": bool(legacy.prompt_uses_gt_attributes(self.prompt_mode)),
            "structure_attr_labels": torch.tensor(structure_labels, dtype=torch.long),
            "boundary_attr_labels": torch.tensor(boundary_labels, dtype=torch.long),
            "structure_attr_values": torch.tensor(structure_values, dtype=torch.float),
            "boundary_attr_values": torch.tensor(boundary_values, dtype=torch.float),
            "has_structure_boundary_attrs": False,
            "dense_boundary_map": torch.from_numpy(dense_maps["boundary_map"]).float().unsqueeze(0),
            "dense_touching_region": torch.from_numpy(dense_maps["touching_region"]).float().unsqueeze(0),
            "dense_small_nuclei": torch.from_numpy(dense_maps["small_nuclei"]).float().unsqueeze(0),
            "dense_hv_gradient": torch.from_numpy(dense_maps["hv_gradient"]).float().unsqueeze(0),
            "instance_attr_labels": torch.from_numpy(inst_morph["instance_attr_labels"]).long(),
            "instance_attr_values": torch.from_numpy(inst_morph["instance_attr_values"]).float(),
            "per_instance_attr_labels": torch.from_numpy(inst_morph["per_instance_attr_labels"]).long(),
            "per_instance_attr_values": torch.from_numpy(inst_morph["per_instance_attr_values"]).float(),
            "per_instance_ids": torch.from_numpy(inst_morph["per_instance_ids"]).long(),
        }
        if local_region_targets is not None:
            result.update(
                {
                    "local_region_attr_labels": torch.from_numpy(local_region_targets["labels"]).long(),
                    "local_region_attr_valid": torch.from_numpy(local_region_targets["valid"]).bool(),
                    "local_region_attr_values": torch.from_numpy(local_region_targets["values"]).float(),
                    "local_region_complete_counts": torch.from_numpy(
                        local_region_targets["complete_instance_count"]
                    ).long(),
                    "local_region_coordinates": torch.tensor(
                        local_region_targets["coordinates"], dtype=torch.long
                    ),
                }
            )
        return result
