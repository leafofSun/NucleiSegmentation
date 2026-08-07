import argparse
import datetime
import hashlib
import os
import math
import cv2
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from collections import defaultdict
from typing import Tuple
import multiprocessing as mp

# === 核心防死锁机制 ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 自动剔除所有非 ASCII 脏字符，避免环境变量里隐藏字符导致底层库异常
for _key, _value in list(os.environ.items()):
    if isinstance(_value, str):
        os.environ[_key] = "".join(c for c in _value if ord(c) < 128).strip()

_SEMANTIC_DIAG_PRINTED = False


def _sha256_file_for_audit(path):
    """Return a streaming SHA256 used only for immutable test provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensors_for_audit(tensors):
    """Hash tensor values in a fixed order without changing model state."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


import torch
import torch.nn.functional as F
import torch.distributed as dist

from evaluation_audit import (
    DistributedEvalSampler,
    protocol_from_test_args,
    write_evaluation_protocol,
    write_run_manifests,
)
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
from metrics import SegMetrics, get_fast_aji, get_fast_pq

from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as skimage_label


ORGAN_TO_ID = {
    "Adrenal_gland": 0,
    "Bile-duct": 1,
    "Bladder": 2,
    "Breast": 3,
    "Cervix": 4,
    "Colon": 5,
    "Esophagus": 6,
    "HeadNeck": 7,
    "Kidney": 8,
    "Liver": 9,
    "Lung": 10,
    "Ovarian": 11,
    "Pancreatic": 12,
    "Prostate": 13,
    "Skin": 14,
    "Stomach": 15,
    "Testis": 16,
    "Thyroid": 17,
    "Uterus": 18,
    "Brain": 19,
    "Generic": 20,
}

ID_TO_ORGAN = {v: k for k, v in ORGAN_TO_ID.items()}


# ==================================================================================================
# 0. Prompt 工具：支持 organ_static / dynamic_pred 测试
# ==================================================================================================
def format_organ_name(organ_name: str) -> str:
    if organ_name is None:
        return "generic"

    name = str(organ_name).strip()
    if not name:
        return "generic"

    special_map = {
        "Adrenal_gland": "adrenal gland",
        "Bile-duct": "bile duct",
        "HeadNeck": "head and neck",
        "Ovarian": "ovary",
        "Pancreatic": "pancreas",
        "Generic": "generic",
    }

    if name in special_map:
        return special_map[name]

    name = name.replace("_", " ").replace("-", " ")
    return name.lower()


# 这里的类别名必须和训练阶段 attr_labels 的语义大致一致。
# 当前项目的属性头是离散分类：[color, shape, arrange, size, density]。
# test 阶段没有 GT mask，因此 dynamic_pred 只能使用模型自己预测出来的 attr_logits。
ATTR_VALUE_TEXT = {
    "color": [
        "light or weak nuclear staining",
        "dark or strong nuclear staining",
    ],
    "shape": [
        "mostly round nuclear morphology",
        "elongated or oval nuclear morphology",
        "irregular nuclear morphology",
    ],
    "arrange": [
        "scattered nuclear arrangement",
        "clustered or crowded nuclear arrangement",
    ],
    "size": [
        "small nuclei",
        "medium-sized nuclei",
        "large nuclei",
    ],
    "density": [
        "low nuclear density",
        "moderate nuclear density",
        "high nuclear density",
    ],
}

ATTR_KEY_ORDER = ("color", "shape", "arrange", "size", "density")


def build_test_prompts(organ_name: str, prompt_mode: str = "organ_static"):
    """
    测试阶段 prompt。

    prompt_mode:
        base:
            全部使用 Cell nuclei。
        generic:
            使用通用病理图像上下文。
        organ_static:
            使用 organ-aware 静态 prompt，默认推荐。
        dynamic_pred:
            先回退为 organ_static；真正的 dynamic_pred 由 tta_inference_8x_batch 的两阶段推理生成。
            这样不会使用 GT mask-derived 属性，不存在测试泄漏。
    """
    prompt_mode = str(prompt_mode).lower().strip()
    if prompt_mode == "dynamic_pred":
        # 第一阶段用 organ_static 预测 attr_logits；第二阶段再用预测属性生成动态 prompt。
        prompt_mode = "organ_static"

    organ_text = format_organ_name(organ_name)
    base_prompt = "Cell nuclei"

    if prompt_mode == "base":
        return base_prompt, base_prompt, base_prompt

    if prompt_mode == "generic" or organ_text == "generic":
        text_prompt = "Cell nuclei in H&E-stained histopathology tissue."
        attribute_text = (
            "H&E-stained histopathology patch. "
            "The image contains cell nuclei. "
            "Focus on nuclear regions without using crop-level mask-derived attributes."
        )
        morphology_text = (
            "H&E-stained histopathology patch. "
            "Focus on nuclear boundaries, touching nuclei, and instance-level delineation."
        )
        return text_prompt, attribute_text, morphology_text

    text_prompt = f"Cell nuclei in {organ_text} tissue."
    attribute_text = (
        f"H&E-stained {organ_text} histopathology patch. "
        f"The image contains cell nuclei in {organ_text} tissue. "
        f"This prompt provides organ context without using crop-level mask-derived attributes."
    )
    morphology_text = (
        f"H&E-stained {organ_text} histopathology patch. "
        f"Focus on nuclear boundaries, touching nuclei, and instance-level delineation."
    )

    return text_prompt, attribute_text, morphology_text


def _safe_argmax_index(logits, key: str):
    """Return predicted class index from attr_logits[key]."""
    if not isinstance(logits, dict) or key not in logits:
        return None

    value = logits.get(key, None)
    if value is None or not torch.is_tensor(value):
        return None

    with torch.no_grad():
        value = value.detach().float().cpu()
        if value.numel() == 0:
            return None
        if value.dim() > 1:
            value = value.mean(dim=0)
        return int(torch.argmax(value).item())


def _aggregate_attr_logits_from_outputs(outputs):
    """Average attr_logits across 8x TTA outputs.

    TextSam usually returns out['attr_logits'] as a dict:
        color / shape / arrange / size / density -> logits.
    This function averages each attribute's logits over TTA views.
    """
    buckets = {key: [] for key in ATTR_KEY_ORDER}

    for out in outputs:
        attr_logits = out.get("attr_logits", None)
        if not isinstance(attr_logits, dict):
            continue
        for key in ATTR_KEY_ORDER:
            value = attr_logits.get(key, None)
            if value is None or not torch.is_tensor(value):
                continue
            value = value.detach().float().cpu()
            if value.numel() == 0:
                continue
            if value.dim() > 1:
                value = value.mean(dim=0)
            buckets[key].append(value)

    aggregated = {}
    for key, values in buckets.items():
        if len(values) > 0:
            aggregated[key] = torch.stack(values, dim=0).mean(dim=0)

    return aggregated


def decode_attr_logits_to_labels(attr_logits):
    """Decode predicted attr_logits into readable attribute strings."""
    decoded = {}
    for key in ATTR_KEY_ORDER:
        pred_idx = _safe_argmax_index(attr_logits, key)
        vocab = ATTR_VALUE_TEXT.get(key, [])
        if pred_idx is None or pred_idx < 0 or pred_idx >= len(vocab):
            continue
        decoded[key] = {
            "index": pred_idx,
            "text": vocab[pred_idx],
        }
    return decoded


def build_dynamic_pred_prompts(organ_name: str, attr_logits):
    """Build test-time dynamic prompts from model-predicted attributes.

    This is non-leaking dynamic prediction:
        image -> attr_logits -> dynamic text prompt -> segmentation.

    No GT mask / GT attr_labels are used.
    """
    decoded = decode_attr_logits_to_labels(attr_logits)

    if len(decoded) == 0:
        return build_test_prompts(organ_name=organ_name, prompt_mode="organ_static")

    organ_text = format_organ_name(organ_name)
    organ_phrase = "histopathology tissue" if organ_text == "generic" else f"{organ_text} tissue"

    color_text = decoded.get("color", {}).get("text", "unknown staining intensity")
    shape_text = decoded.get("shape", {}).get("text", "unknown nuclear morphology")
    arrange_text = decoded.get("arrange", {}).get("text", "unknown nuclear arrangement")
    size_text = decoded.get("size", {}).get("text", "unknown nuclear size")
    density_text = decoded.get("density", {}).get("text", "unknown nuclear density")

    text_prompt = f"Cell nuclei in {organ_phrase}."

    attribute_text = (
        f"H&E-stained {organ_phrase}. "
        f"The model predicts {density_text}, {arrange_text}, {size_text}, "
        f"{shape_text}, and {color_text}. "
        f"Use these predicted non-GT attributes as weak context for nuclei segmentation."
    )

    morphology_text = (
        f"H&E-stained {organ_phrase}. "
        f"Predicted morphology: {shape_text} and {size_text}. "
        f"Predicted spatial pattern: {density_text} with {arrange_text}. "
        f"Focus on nuclear boundaries, touching nuclei, and instance-level separation."
    )

    return text_prompt, attribute_text, morphology_text


def extract_organ_from_json(json_path: str):
    organ_name = "Generic"
    organ_id = 20

    if not os.path.exists(json_path):
        return organ_name, organ_id

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            if "organ_id" in data:
                organ_name = data.get("organ_id", "Generic")
                organ_id = ORGAN_TO_ID.get(organ_name, 20)
            elif "organ_idx" in data:
                organ_id = int(data.get("organ_idx", 20))
                organ_name = ID_TO_ORGAN.get(organ_id, "Generic")
    except Exception:
        organ_name = "Generic"
        organ_id = 20

    return organ_name, organ_id

# ==================================================================================================
# 1. Checkpoint / position embedding tools
# ==================================================================================================
def resize_pos_embed(state_dict, model_state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                if "pos_embed" in k:
                    v = v.permute(0, 3, 1, 2)
                    v = F.interpolate(
                        v,
                        size=model_state_dict[k].shape[1:3],
                        mode="bicubic",
                        align_corners=False,
                    )
                    v = v.permute(0, 2, 3, 1)
                elif "rel_pos" in k:
                    v = v.unsqueeze(0).permute(0, 2, 1)
                    target_len = model_state_dict[k].shape[0]
                    v = F.interpolate(
                        v,
                        size=target_len,
                        mode="linear",
                        align_corners=False,
                    )
                    v = v.permute(0, 2, 1).squeeze(0)

            new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict

    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt: dict) -> Tuple[dict, str]:
    """Extract the raw state dict from a checkpoint dict.

    Tries common keys: 'model', 'model_state_dict', 'state_dict'.
    Falls back to ckpt itself if none found (flat state_dict).
    Also strips 'module.' prefix if present.

    Returns:
        (state_dict, source_key): where source_key is the key name used.
    """
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            raw = ckpt[key]
            break
    else:
        raw = ckpt
        key = "flat"
    return strip_module_prefix(raw), key


def load_model_checkpoint(model, checkpoint_path, device,
                          filter_mismatch=True, verbose=True):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ---- 1. Extract, detect source, strip prefix, interpolate ----
    state_dict, sd_source = _extract_state_dict(ckpt)
    model_sd = model.state_dict()
    state_dict = resize_pos_embed(state_dict, model_sd)

    # ── Handle text_bank buffer shape mismatches (CONCHLESS mode) ──
    # The model registers _structure/boundary_text_bank_buffer as torch.zeros(0),
    # but the checkpoint has them as [5,3,512] / [4,3,512]. Re-register with
    # correct checkpoint shape before filtering to avoid load_state_dict error.
    _text_bank_buffers = ("_structure_text_bank_buffer", "_boundary_text_bank_buffer")
    for _buf_name in _text_bank_buffers:
        if _buf_name in state_dict and _buf_name in model_sd:
            _ckpt_shape = state_dict[_buf_name].shape
            _mdl_shape = model_sd[_buf_name].shape
            if _ckpt_shape != _mdl_shape:
                if verbose:
                    print(f"[TEXT_BANK_RESIZE] {_buf_name}: model={_mdl_shape} → checkpoint={_ckpt_shape}")
                model.register_buffer(_buf_name, torch.zeros(_ckpt_shape, device=device), persistent=True)

    # Re-capture model state dict after potential buffer resizing
    model_sd = model.state_dict()

    all_ckpt_keys = list(state_dict.keys())

    # ---- 2. Filter: keep only keys that exist AND have matching shape ----
    filtered_dict = {}
    loaded_count = 0
    skipped_missing = []  # key in checkpoint but NOT in model
    skipped_mismatch = []  # key exists but shape differs

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue
        if filter_mismatch and v.shape != model_sd[k].shape:
            skipped_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered_dict[k] = v
        loaded_count += 1

    # ---- 3. Safe load (strict=False only for remaining missing/unexpected) ----
    load_ret = model.load_state_dict(filtered_dict, strict=False)
    missing_keys = getattr(load_ret, "missing_keys", [])
    unexpected_keys = getattr(load_ret, "unexpected_keys", [])

    # ---- 4. Summary ----
    n_missing_skipped = len(skipped_missing)
    n_mismatch_skipped = len(skipped_mismatch)
    n_missing_after = len(missing_keys)
    n_unexpected = len(unexpected_keys)

    # ---- 5. [TEST_RESUME_AUDIT] ----
    # Module-level audit
    _ml_ckpt = [k for k in all_ckpt_keys if "multilevel_attr_heads" in k]
    _ml_loaded = [k for k in filtered_dict if "multilevel_attr_heads" in k]
    _aa_ckpt = [k for k in all_ckpt_keys if "attr_align" in k]
    _aa_loaded = [k for k in filtered_dict if "attr_align" in k]
    _sd_ckpt = [k for k in all_ckpt_keys if "semantic_delta_adapter" in k]
    _sd_loaded = [k for k in filtered_dict if "semantic_delta_adapter" in k]
    _sg_ckpt = [k for k in all_ckpt_keys if "semantic_channel_gate" in k]
    _sg_loaded = [k for k in filtered_dict if "semantic_channel_gate" in k]

    print("[TEST_RESUME_AUDIT]")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  state_dict_source={sd_source}")
    print(f"  loaded_keys={loaded_count}")
    print(f"  missing_keys={len(missing_keys)}")
    print(f"  unexpected_keys={n_unexpected}")
    print(f"  mismatch_skipped={n_mismatch_skipped}")
    print()
    print(f"  Module-level audit:")
    print(f"    multilevel_attr_heads loaded={len(_ml_loaded)}/{len(_ml_ckpt)}")
    print(f"    attr_align loaded={len(_aa_loaded)}/{len(_aa_ckpt)}")
    print(f"    semantic_delta_adapter loaded={len(_sd_loaded)}/{len(_sd_ckpt)}")
    print(f"    semantic_channel_gate loaded={len(_sg_loaded)}/{len(_sg_ckpt)}")

    # Error checks
    _enable_aa = getattr(model, "enable_attr_text_alignment", False)
    if _enable_aa and len(_aa_ckpt) == 0:
        raise RuntimeError(
            "[TEST_RESUME_AUDIT_ERROR] attr_align expected keys is 0 "
            "while enable_attr_text_alignment=True. "
            "Checkpoint may not have attr_align heads."
        )

    if isinstance(ckpt, dict):
        print(f"  Checkpoint architecture_version: {ckpt.get('architecture_version', 'N/A')}")
        print(
            "  Checkpoint phase="
            f"{ckpt.get('phase', 'N/A')} | "
            f"asr_variant={ckpt.get('asr_variant', 'N/A')} | "
            f"asr_regression={ckpt.get('asr_regression', 'N/A')}"
        )

    # ---- 6. Detailed shape-mismatch report (top 20) ----
    if n_mismatch_skipped > 0:
        print(f"  ⚠️  Shape mismatches ({n_mismatch_skipped} total, showing first 20):")
        for i, (k, ckpt_shape, model_shape) in enumerate(skipped_mismatch[:20]):
            print(f"      [{i}] {k}: checkpoint={ckpt_shape} vs model={model_shape}")
        if n_mismatch_skipped > 20:
            print(f"      ... and {n_mismatch_skipped - 20} more")

    print(f"  ✅ Loaded checkpoint: {checkpoint_path}")
    return model


# ==================================================================================================
# 2. 核心后处理
# ==================================================================================================
def hover_post_process(
    prob_map,
    hv_map,
    prob_thresh=0.40,
    marker_thresh=0.45,
    min_marker_size=12,
    final_min_object_size=15,
):
    mask = prob_map > prob_thresh
    mask = binary_fill_holes(mask)

    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    diff_v = np.gradient(v_map, axis=0)
    diff_h = np.gradient(h_map, axis=1)
    sobel_mag = np.sqrt(diff_v ** 2 + diff_h ** 2)

    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    marker_map = remove_small_objects(marker_map, min_size=int(min_marker_size))

    markers = skimage_label(marker_map).astype(np.int32)

    if markers.max() == 0:
        markers = skimage_label(mask).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)

    inst_map = remove_small_objects(inst_map, min_size=int(final_min_object_size))
    inst_map = inst_map.astype(np.int32)

    return inst_map


# ==================================================================================================
# 3. 8-fold TTA batch inference
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, organ_name, args):
    device = args.device
    input_size = (args.image_size, args.image_size)

    transforms = [
        (None, 0),
        (1, 0),
        (0, 0),
        (-1, 0),
        (None, 1),
        (1, 1),
        (0, 1),
        (-1, 1),
    ]

    img_list = []
    for f_code, r_k in transforms:
        img_t = image_rgb.copy()

        if f_code is not None:
            img_t = cv2.flip(img_t, f_code)

        if r_k > 0:
            img_t = np.rot90(img_t, k=r_k)

        img_t = cv2.resize(img_t, input_size)
        img_list.append(torch.from_numpy(img_t).permute(2, 0, 1).float())

    batch_img = torch.stack(img_list).to(device)

    autocast_enabled = bool(device.type == "cuda")

    def _forward_with_prompts(text_prompt, attribute_text, morphology_text):
        input_samples = []
        for i in range(len(transforms)):
            input_samples.append(
                {
                    "image": batch_img[i],
                    "original_size": input_size,
                    "organ_id": int(organ_id),
                    "text_prompt": text_prompt,
                    "attribute_text": attribute_text,
                    "morphology_text": morphology_text,
                    "attr_labels": None,
                }
            )
        return model(input_samples, multimask_output=True)

    prompt_mode = str(args.prompt_mode).lower().strip()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        if prompt_mode == "dynamic_pred":
            # Stage 1: use non-leaking static prompt to predict PNuRL attr_logits.
            bootstrap_mode = getattr(args, "dynamic_pred_bootstrap_prompt", "organ_static")
            boot_text, boot_attr, boot_morph = build_test_prompts(
                organ_name=organ_name,
                prompt_mode=bootstrap_mode,
            )
            bootstrap_outputs = _forward_with_prompts(boot_text, boot_attr, boot_morph)
            aggregated_attr_logits = _aggregate_attr_logits_from_outputs(bootstrap_outputs)

            # Stage 2: rebuild prompt from predicted attributes and run segmentation.
            text_prompt, attribute_text, morphology_text = build_dynamic_pred_prompts(
                organ_name=organ_name,
                attr_logits=aggregated_attr_logits,
            )
            outputs = _forward_with_prompts(text_prompt, attribute_text, morphology_text)
        else:
            text_prompt, attribute_text, morphology_text = build_test_prompts(
                organ_name=organ_name,
                prompt_mode=prompt_mode,
            )
            outputs = _forward_with_prompts(text_prompt, attribute_text, morphology_text)

        global _SEMANTIC_DIAG_PRINTED
        if not _SEMANTIC_DIAG_PRINTED and len(outputs) > 0:
            _SEMANTIC_DIAG_PRINTED = True
            _diag = outputs[0]
            _gate_mean = _diag.get("semantic_channel_gate_mean", None)
            _gate_std = _diag.get("semantic_channel_gate_std", None)
            _inj_norm = _diag.get("injected_delta_norm", None)
            _inj_ratio = _diag.get("injection_ratio", None)
            _delta_norm = _diag.get("semantic_delta_norm", None)
            _bias_print = getattr(args, "semantic_gate_bias_init", None)
            _scale_print = getattr(args, "semantic_injection_scale", 1.0)
            _runtime_attr = bool(getattr(model, "enable_attr_text_alignment", False))
            _runtime_pnurl_align = bool(getattr(model, "enable_promptnu_lite_align", False))
            _runtime_use_pnurl = bool(getattr(model, "use_pnurl", False))
            _gm = float(_gate_mean.cpu().item()) if torch.is_tensor(_gate_mean) else float("nan")
            _gs = float(_gate_std.cpu().item()) if torch.is_tensor(_gate_std) else float("nan")
            _dn = float(_delta_norm.cpu().item()) if torch.is_tensor(_delta_norm) else float("nan")
            _in = float(_inj_norm.cpu().item()) if torch.is_tensor(_inj_norm) else float("nan")
            _ir = float(_inj_ratio.cpu().item()) if torch.is_tensor(_inj_ratio) else float("nan")
            print(
                f"[DIAG] enable_attr_text_alignment={_runtime_attr} | "
                f"enable_promptnu_lite_align={_runtime_pnurl_align} | "
                f"use_pnurl={_runtime_use_pnurl} | "
                f"semantic_gate_bias_init={_bias_print} | "
                f"semantic_injection_scale={_scale_print} | "
                f"DeltaNorm={_dn:.6e} | "
                f"GateMean={_gm:.6e} | GateStd={_gs:.6e} | "
                f"InjectedNorm={_in:.6e} | InjRatio={_ir:.6e}"
            )
            # ── v3 runtime diagnostics from diagnostics dict ──
            _v3_enabled = _diag.get("enable_promptnu_guided_v3", None)
            if torch.is_tensor(_v3_enabled) and _v3_enabled.cpu().item() > 0.5:
                _v3_active = _diag.get("v3_active", None)
                _v3_skipped = _diag.get("v3_skipped", None)
                _v3_skip_reason = _diag.get("v3_skip_reason", "N/A")
                _td_std = _diag.get("promptnu_guided_v3_text_delta_std", None)
                _add_norm = _diag.get("promptnu_guided_v3_additive_delta_norm", None)
                _sd_before = _diag.get("semantic_delta_before_v3_norm", None)
                _sd_after = _diag.get("semantic_delta_after_v3_norm", None)
                _inj_norm_v3 = _diag.get("injected_delta_norm", None)
                _uses_guided = _diag.get("uses_guided_delta_for_injection", None)
                _v3_active_val = int(_v3_active.cpu().item()) if torch.is_tensor(_v3_active) else -1
                _v3_skipped_val = int(_v3_skipped.cpu().item()) if torch.is_tensor(_v3_skipped) else -1
                _td_std_val = float(_td_std.cpu().item()) if torch.is_tensor(_td_std) else float("nan")
                _add_norm_val = float(_add_norm.cpu().item()) if torch.is_tensor(_add_norm) else float("nan")
                _sd_before_val = float(_sd_before.cpu().item()) if torch.is_tensor(_sd_before) else float("nan")
                _sd_after_val = float(_sd_after.cpu().item()) if torch.is_tensor(_sd_after) else float("nan")
                _inj_norm_val = float(_inj_norm_v3.cpu().item()) if torch.is_tensor(_inj_norm_v3) else float("nan")
                _uses_guided_val = bool(_uses_guided.cpu().item()) if torch.is_tensor(_uses_guided) else False
                print(
                    f"[DIAG][v3_runtime] "
                    f"v3_active={_v3_active_val} | "
                    f"v3_skipped={_v3_skipped_val} | "
                    f"v3_skip_reason={_v3_skip_reason} | "
                    f"text_delta_std={_td_std_val:.8e} | "
                    f"additive_delta_norm={_add_norm_val:.8e} | "
                    f"semantic_delta_before_v3_norm={_sd_before_val:.8e} | "
                    f"semantic_delta_after_v3_norm={_sd_after_val:.8e} | "
                    f"injected_delta_norm={_inj_norm_val:.8e} | "
                    f"uses_guided_delta_for_injection={_uses_guided_val}"
                )

        all_probs = []
        all_hvs = []
        first_attr_logits = {}

        for i in range(len(transforms)):
            out = outputs[i]

            iou_predictions = out["iou_predictions"]
            if iou_predictions.ndim == 2:
                iou_predictions = iou_predictions.squeeze(0)

            best_idx = torch.argmax(iou_predictions).item()

            masks = out["masks"]
            if masks.dim() == 4:
                prob = torch.sigmoid(masks[0, best_idx])
            elif masks.dim() == 3:
                prob = torch.sigmoid(masks[best_idx])
            else:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")

            hv_raw = out.get("hv_logits", None)

            if hv_raw is not None:
                if hv_raw.dim() == 3:
                    hv_raw = hv_raw.unsqueeze(0)

                hv_raw = torch.tanh(hv_raw.float())
                hv = F.interpolate(
                    hv_raw,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

            if i == 0:
                first_attr_logits = out.get("attr_logits", {}) or {}
                # For dynamic_pred, keep the aggregated bootstrap logits for dynamic min-size.
                if prompt_mode == "dynamic_pred" and 'aggregated_attr_logits' in locals() and aggregated_attr_logits:
                    first_attr_logits = aggregated_attr_logits

            f_code, r_k = transforms[i]

            # Inverse rotation first because forward order is flip -> rotation.
            if r_k == 1:
                prob = torch.rot90(prob, k=-1, dims=[0, 1])
                hv = torch.rot90(hv, k=-1, dims=[1, 2])

                # HV vector inverse rotation.
                v_new = hv[1].clone()
                h_new = -hv[0].clone()
                hv[0], hv[1] = v_new, h_new

            # Inverse flip.
            if f_code == 1:
                prob = torch.flip(prob, [1])
                hv = torch.flip(hv, [2])
                hv[1] = -hv[1]
            elif f_code == 0:
                prob = torch.flip(prob, [0])
                hv = torch.flip(hv, [1])
                hv[0] = -hv[0]
            elif f_code == -1:
                prob = torch.flip(prob, [0, 1])
                hv = torch.flip(hv, [1, 2])
                hv = -hv

            all_probs.append(prob)
            all_hvs.append(hv)

    avg_prob = torch.stack(all_probs).mean(0).cpu().float().numpy()
    avg_hv = torch.stack(all_hvs).mean(0).cpu().float().numpy()

    return avg_prob, avg_hv, first_attr_logits


# ==================================================================================================
# 4. Sliding window inference
# ==================================================================================================
def get_gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))
    return kernel.astype(np.float32)


def sliding_window_inference(model, image_rgb, organ_id, organ_name, args, patch_size=256, overlap=0.8):
    h, w = image_rgb.shape[:2]

    stride = max(1, int(patch_size * (1 - overlap)))

    pad_h = 0 if h % stride == 0 else stride - (h % stride)
    pad_w = 0 if w % stride == 0 else stride - (w % stride)

    pad_h = max(pad_h, patch_size - h) if h < patch_size else pad_h
    pad_w = max(pad_w, patch_size - w) if w < patch_size else pad_w

    padded_img = np.pad(
        image_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )

    pad_h_full, pad_w_full = padded_img.shape[:2]

    canvas_prob = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    canvas_hv = np.zeros((2, pad_h_full, pad_w_full), dtype=np.float32)
    canvas_weight = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)

    weight_mask = get_gaussian_kernel(patch_size, sigma=0.33)
    accumulated_size_logits = None

    for y in range(0, pad_h_full - patch_size + 1, stride):
        for x in range(0, pad_w_full - patch_size + 1, stride):
            patch = padded_img[y:y + patch_size, x:x + patch_size, :]

            prob_512, hv_512, attr_logits = tta_inference_8x_batch(
                model=model,
                image_rgb=patch,
                organ_id=organ_id,
                organ_name=organ_name,
                args=args,
            )

            prob_patch = cv2.resize(
                prob_512,
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            hv_v_patch = cv2.resize(
                hv_512[0],
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            hv_h_patch = cv2.resize(
                hv_512[1],
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            canvas_prob[y:y + patch_size, x:x + patch_size] += prob_patch * weight_mask
            canvas_hv[0, y:y + patch_size, x:x + patch_size] += hv_v_patch * weight_mask
            canvas_hv[1, y:y + patch_size, x:x + patch_size] += hv_h_patch * weight_mask
            canvas_weight[y:y + patch_size, x:x + patch_size] += weight_mask

            if isinstance(attr_logits, dict) and "size" in attr_logits:
                size_logits = attr_logits["size"].detach().cpu()

                if accumulated_size_logits is None:
                    accumulated_size_logits = size_logits.clone()
                else:
                    accumulated_size_logits += size_logits

    canvas_prob /= (canvas_weight + 1e-8)
    canvas_hv /= (canvas_weight + 1e-8)

    final_prob = canvas_prob[:h, :w]
    final_hv = canvas_hv[:, :h, :w]

    dynamic_min_size = args.min_marker_size

    if accumulated_size_logits is not None:
        if accumulated_size_logits.ndim > 1:
            mean_logits = accumulated_size_logits.mean(dim=0)
        else:
            mean_logits = accumulated_size_logits

        pred_size_idx = torch.argmax(mean_logits).item()
        dynamic_min_size = {0: 12, 1: 25, 2: 38}.get(pred_size_idx, args.min_marker_size)

    return final_prob, final_hv, dynamic_min_size


# ==================================================================================================
# 5. GT loading
# ==================================================================================================
def load_filtered_gt(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        annotations = data.get("annotations", []) if isinstance(data, dict) else data

        if not annotations:
            return None

        h, w = None, None

        if isinstance(data, dict):
            h, w = data.get("height"), data.get("width")

        if h is None or w is None:
            first_seg = annotations[0].get("segmentation", {})
            if isinstance(first_seg, dict) and "size" in first_seg:
                h, w = first_seg["size"]
            else:
                h, w = 1000, 1000

        instance_map = np.zeros((int(h), int(w)), dtype=np.int32)

        for idx, ann in enumerate(annotations):
            seg = ann.get("segmentation")
            if not seg:
                continue

            if isinstance(seg, list):
                for poly in seg:
                    poly_np = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    if poly_np.shape[0] >= 3:
                        poly_np = np.round(poly_np).astype(np.int32)
                        cv2.fillPoly(instance_map, [poly_np], idx + 1)
            elif isinstance(seg, dict) and "counts" in seg:
                binary_mask = mask_utils.decode(seg)
                instance_map[binary_mask > 0] = idx + 1

        return instance_map

    except Exception as exc:
        print(f"[WARN] Failed to load GT for {img_path}: {exc}")
        return None


# ==================================================================================================
# 6. Prediction saving
# ==================================================================================================
def save_prediction(pred_mask, img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]

    npy_path = os.path.join(output_dir, f"{base}_inst.npy")
    png_path = os.path.join(output_dir, f"{base}_inst.png")

    np.save(npy_path, pred_mask.astype(np.int32))

    max_val = int(pred_mask.max())
    if max_val <= 65535:
        cv2.imwrite(png_path, pred_mask.astype(np.uint16))
    else:
        vis = (pred_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(png_path, vis)


# ==================================================================================================
# 7a. Shared model building helper (used by both mp.Pool and DDP)
# ==================================================================================================
def _build_test_model(args, device, rank=0, is_rank0=True):
    """Build TextSam model and load checkpoint with rank-aware logging.

    Extracted from process_chunk to avoid code duplication between
    single-GPU mp.Pool and DDP torchrun paths.
    """
    fine_tuned_ckpt = args.checkpoint

    if fine_tuned_ckpt is None or not os.path.isfile(fine_tuned_ckpt):
        raise FileNotFoundError(f"checkpoint not found: {fine_tuned_ckpt}")

    _rsgr_enabled = bool(getattr(args, "enable_rsgr", False))
    _rsgr_prototype_path = getattr(args, "rsgr_prototype_path", None)
    _rsgr_bank_sha256 = (
        _sha256_file_for_audit(_rsgr_prototype_path)
        if _rsgr_prototype_path and os.path.isfile(_rsgr_prototype_path)
        else "NOT_FOUND"
    )

    # ── [TEST_CONFIG] (rank0 only) ──
    if is_rank0:
        print("=" * 60)
        print("[TEST_CONFIG] Phase D / PromptNu-lite v2 Test Configuration")
        print(f"  model_type={args.model_type}")
        print(f"  resume={fine_tuned_ckpt}")
        print(f"  resume_filter_mismatch={getattr(args, 'resume_filter_mismatch', False)}")
        print(f"  enable_attr_text_alignment={getattr(args, 'enable_attr_text_alignment', False)}")
        print(f"  enable_multilevel_attr_heads={getattr(args, 'enable_multilevel_attr_heads', False)}")
        print(f"  enable_promptnu_lite_align={getattr(args, 'enable_promptnu_lite_align', False)}")
        print(f"  promptnu_lite_target={getattr(args, 'promptnu_lite_target', 'semantic_delta')}")
        print(f"  promptnu_lite_pool_mode={getattr(args, 'promptnu_lite_pool_mode', 'gap')}")
        print(f"  promptnu_lite_struct_weight={getattr(args, 'promptnu_lite_struct_weight', 0.0)}")
        print(f"  promptnu_lite_boundary_weight={getattr(args, 'promptnu_lite_boundary_weight', 0.0)}")
        print(f"  promptnu_lite_instance_weight={getattr(args, 'promptnu_lite_instance_weight', 0.0)}")
        print(f"  use_structure_boundary_attrs={getattr(args, 'use_structure_boundary_attrs', False)}")
        print(f"  structure_boundary_attr_path={getattr(args, 'structure_boundary_attr_path', None)}")
        print(f"  hf_hub_offline={getattr(args, 'hf_hub_offline', False)}")
        print(f"  conch_cache_path={getattr(args, 'conch_cache_path', None)}")
        print(f"  use_asr={args.use_asr}")
        print(f"  asr_variant={args.asr_variant}")
        print(f"  rsgr_enabled={_rsgr_enabled}")
        print(f"  rsgr_prototype_path={_rsgr_prototype_path}")
        print(f"  rsgr_bank_sha256={_rsgr_bank_sha256}")
        # ── PromptNu-guided v3 ──
        print(f"  enable_promptnu_guided_v3={getattr(args, 'enable_promptnu_guided_v3', False)}")
        print(f"  promptnu_guided_v3_struct_weight={getattr(args, 'promptnu_guided_v3_struct_weight', 1.0)}")
        print(f"  promptnu_guided_v3_boundary_weight={getattr(args, 'promptnu_guided_v3_boundary_weight', 1.0)}")
        print(f"  promptnu_guided_v3_text_weight={getattr(args, 'promptnu_guided_v3_text_weight', 0.01)}")
        print(f"  promptnu_guided_v3_embed_dim={getattr(args, 'promptnu_guided_v3_embed_dim', 256)}")
        print(f"  promptnu_guided_v3_hidden_dim={getattr(args, 'promptnu_guided_v3_hidden_dim', 128)}")
        print(f"  promptnu_guided_v3_vis_proj_dim={getattr(args, 'promptnu_guided_v3_vis_proj_dim', 512)}")
        print(f"  promptnu_guided_v3_align_loss_weight={getattr(args, 'promptnu_guided_v3_align_loss_weight', 0.1)}")
        # ── PromptNu-guided v3.3: Scale + Additive guidance ──
        print(f"  promptnu_guided_v3_guidance_mode={getattr(args, 'promptnu_guided_v3_guidance_mode', 'scale_add')}")
        print(f"  promptnu_guided_v3_scale_weight={getattr(args, 'promptnu_guided_v3_scale_weight', None)}")
        print(f"  promptnu_guided_v3_delta_weight={getattr(args, 'promptnu_guided_v3_delta_weight', 0.001)}")
        print(f"  promptnu_guided_v3_delta_init_std={getattr(args, 'promptnu_guided_v3_delta_init_std', 1e-5)}")
        print(f"  promptnu_guided_v3_max_guided_delta_ratio={getattr(args, 'promptnu_guided_v3_max_guided_delta_ratio', 0.0)}")
        # ── PromptNu-guided v3.1: CONCH text bank & GT align target ──
        print(f"  promptnu_guided_v3_use_text_bank={getattr(args, 'promptnu_guided_v3_use_text_bank', False)}")
        print(f"  promptnu_guided_v3_use_gt_align_target={getattr(args, 'promptnu_guided_v3_use_gt_align_target', False)}")
        print(f"  promptnu_guided_v3_semantic_dim={getattr(args, 'promptnu_guided_v3_semantic_dim', 256)}")
        print(f"  promptnu_guided_v3_text_dim={getattr(args, 'promptnu_guided_v3_text_dim', 512)}")
        print(f"  ablate_semantic_injection={getattr(args, 'ablate_semantic_injection', False)}")
        print(f"  ablate_pred_attr_guidance={getattr(args, 'ablate_pred_attr_guidance', False)}")
        # ── PromptNu-guided v3 diagnostic: prompt source & attr quality audit ──
        print(f"  promptnu_guided_v3_prompt_source={getattr(args, 'promptnu_guided_v3_prompt_source', 'pred_attr')}")
        print(f"  attr_quality_audit={getattr(args, 'attr_quality_audit', False)}")
        print(f"  promptnu_guided_v3_fixed_global_text={getattr(args, 'promptnu_guided_v3_fixed_global_text', 'cell nuclei with irregular nuclear morphology and dense chromatin')}")
        _freqpath_abl = os.environ.get("FREQPATH_ABLATION", "both")
        print(f"  FREQPATH_ABLATION={_freqpath_abl}")
        print("=" * 60)

    # ── [CONCHLESS_ARG_AUDIT] diagnostic logging before model build ──
    _conchless_flag = bool(getattr(args, "use_checkpoint_text_bank_without_conch", False))
    # ── [CLIP_BACKEND_ARG_AUDIT] CLIP text encoder replaces CONCH ──
    _clip_text_flag = bool(getattr(args, "clip_text_encoder", False))
    if is_rank0:
        print(f"[CONCHLESS_ARG_AUDIT:test.py] use_checkpoint_text_bank_without_conch={_conchless_flag}")
        if _clip_text_flag:
            print(f"[CLIP_BACKEND_ARG_AUDIT:test.py] clip_text_encoder=True, "
                  f"model={getattr(args, 'clip_text_encoder_model', 'ViT-B/32')}")

    # ── [CONCH_REQUIRED_AUDIT] Determine if CONCH text encoder is actually needed ──
    # CONCH is only needed when at least one CONCH-dependent module requires
    # real-time text encoding AND CLIP backend is NOT active.
    _conch_required = any([
        bool(getattr(args, "enable_attr_text_alignment", False)),
        bool(getattr(args, "enable_promptnu_lite_align", False)),
        (bool(getattr(args, "enable_promptnu_guided_v3", False))
         and bool(getattr(args, "promptnu_guided_v3_use_text_bank", False))
         and not _conchless_flag),
        (bool(getattr(args, "use_pnurl", False)) and not _conchless_flag),
    ]) and not _clip_text_flag  # CLIP backend replaces CONCH
    if is_rank0:
        print("[CONCH_REQUIRED_AUDIT]")
        print(f"  enable_attr_text_alignment={bool(getattr(args, 'enable_attr_text_alignment', False))}")
        print(f"  enable_promptnu_lite_align={bool(getattr(args, 'enable_promptnu_lite_align', False))}")
        print(f"  enable_promptnu_guided_v3={bool(getattr(args, 'enable_promptnu_guided_v3', False))}")
        print(f"  promptnu_guided_v3_use_text_bank={bool(getattr(args, 'promptnu_guided_v3_use_text_bank', False))}")
        print(f"  use_pnurl={bool(getattr(args, 'use_pnurl', False))}")
        print(f"  use_checkpoint_text_bank_without_conch={_conchless_flag}")
        print(f"  clip_text_encoder={_clip_text_flag}")
        print(f"  conch_required={_conch_required}")
        print(f"  enable_conch_text_encoder_final={_conch_required}")

    # 构建基础 SAM / TextSam 所需组件。
    # 注意：这里不要把 fine-tuned checkpoint 当成 base checkpoint 加载。
    original_checkpoint = args.checkpoint
    args.checkpoint = None

    # ── Override enable_conch_text_encoder based on conch_required ──
    # This propagates to sam_model_registry[args.model_type](args) → build_sam_vit_*(args)
    # → _build_sam(...) → TextSam(...).
    args.enable_conch_text_encoder = _conch_required

    vanilla_sam = sam_model_registry[args.model_type](args)

    args.checkpoint = original_checkpoint

    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name=args.clip_model,
        num_organs=args.num_organs,
        num_heads=args.num_heads,
        sg_epsilon=0.05,
        sg_iters=3,
        use_pnurl=args.use_pnurl,
        use_coop=args.use_coop,
        use_ot=False,
        use_asr=args.use_asr,
        asr_variant=args.asr_variant,
        asr_regression=args.asr_regression,
        max_semantic_gate=args.max_semantic_gate,
        max_delta_ratio=args.max_delta_ratio,
        init_delta_ratio=args.init_delta_ratio,
        semantic_gate_bias_init=getattr(args, "semantic_gate_bias_init", None),
        semantic_injection_scale=float(getattr(args, "semantic_injection_scale", 1.0)),
        # --- Structure & Boundary predicted direct guidance ---
        enable_structure_boundary_attr_heads=getattr(args, "enable_structure_boundary_attr_heads", False),
        sb_guidance_mode=str(getattr(args, "sb_guidance_mode", "none")),
        sb_guidance_weight=float(getattr(args, "sb_guidance_weight", 0.05)),
        sb_guidance_routing=str(getattr(args, "sb_guidance_routing", "structure_low_boundary_high")),
        # ── Phase B: MultiLevel Attribute Heads ──
        enable_multilevel_attr_heads=bool(getattr(args, "enable_multilevel_attr_heads", False)),
        # ── Phase C: Attribute-Text Alignment ──
        enable_attr_text_alignment=bool(getattr(args, "enable_attr_text_alignment", False)),
        # ── PromptNu-lite v2: Residual-Coupled Semantic Alignment ──
        enable_promptnu_lite_align=bool(getattr(args, "enable_promptnu_lite_align", False)),
        promptnu_lite_target=str(getattr(args, "promptnu_lite_target", "semantic_delta")),
        promptnu_lite_pool_mode=str(getattr(args, "promptnu_lite_pool_mode", "gap")),
        promptnu_lite_struct_weight=float(getattr(args, "promptnu_lite_struct_weight", 0.0)),
        promptnu_lite_boundary_weight=float(getattr(args, "promptnu_lite_boundary_weight", 0.0)),
        promptnu_lite_instance_weight=float(getattr(args, "promptnu_lite_instance_weight", 0.0)),
        promptnu_lite_detach_text=bool(getattr(args, "promptnu_lite_detach_text", True)),
        promptnu_lite_detach_visual=bool(getattr(args, "promptnu_lite_detach_visual", False)),
        promptnu_lite_proj_lr_mult=float(getattr(args, "promptnu_lite_proj_lr_mult", 0.5)),
        # ── PromptNu-guided v3 ──
        enable_promptnu_guided_v3=bool(getattr(args, "enable_promptnu_guided_v3", False)),
        promptnu_guided_v3_struct_weight=float(getattr(args, "promptnu_guided_v3_struct_weight", 1.0)),
        promptnu_guided_v3_boundary_weight=float(getattr(args, "promptnu_guided_v3_boundary_weight", 1.0)),
        promptnu_guided_v3_text_weight=float(getattr(args, "promptnu_guided_v3_text_weight", 0.01)),
        promptnu_guided_v3_embed_dim=int(getattr(args, "promptnu_guided_v3_embed_dim", 256)),
        promptnu_guided_v3_hidden_dim=int(getattr(args, "promptnu_guided_v3_hidden_dim", 128)),
        promptnu_guided_v3_vis_proj_dim=int(getattr(args, "promptnu_guided_v3_vis_proj_dim", 512)),
        promptnu_guided_v3_align_loss_weight=float(getattr(args, "promptnu_guided_v3_align_loss_weight", 0.1)),
        # ── PromptNu-guided v3.1: CONCH text bank & GT align target ──
        promptnu_guided_v3_use_text_bank=bool(getattr(args, "promptnu_guided_v3_use_text_bank", False)),
        promptnu_guided_v3_use_gt_align_target=bool(getattr(args, "promptnu_guided_v3_use_gt_align_target", False)),
        promptnu_guided_v3_semantic_dim=int(getattr(args, "promptnu_guided_v3_semantic_dim", 256)),
        promptnu_guided_v3_text_dim=int(getattr(args, "promptnu_guided_v3_text_dim", 512)),
        promptnu_guided_v3_strict_audit=bool(getattr(args, "promptnu_guided_v3_strict_audit", False)),
        # ── PromptNu-guided v3.3: Scale + Additive guidance ──
        promptnu_guided_v3_guidance_mode=str(getattr(args, "promptnu_guided_v3_guidance_mode", "scale_add")),
        promptnu_guided_v3_scale_weight=getattr(args, "promptnu_guided_v3_scale_weight", None),
        promptnu_guided_v3_delta_weight=float(getattr(args, "promptnu_guided_v3_delta_weight", 0.001)),
        promptnu_guided_v3_delta_init_std=float(getattr(args, "promptnu_guided_v3_delta_init_std", 1e-5)),
        promptnu_guided_v3_max_guided_delta_ratio=float(getattr(args, "promptnu_guided_v3_max_guided_delta_ratio", 0.0)),
        # ── PromptNu-guided v3.3 alignment stability ──
        promptnu_guided_v3_align_eps=float(getattr(args, "promptnu_guided_v3_align_eps", 1e-8)),
        promptnu_guided_v3_cosine_eps=float(getattr(args, "promptnu_guided_v3_cosine_eps", 1e-8)),
        promptnu_guided_v3_min_align_delta_norm=float(getattr(args, "promptnu_guided_v3_min_align_delta_norm", 0.0)),
        promptnu_guided_v3_align_low_norm_mode=str(getattr(args, "promptnu_guided_v3_align_low_norm_mode", "detach_guided")),
        ablate_semantic_injection=bool(getattr(args, "ablate_semantic_injection", False)),
        ablate_pred_attr_guidance=bool(getattr(args, "ablate_pred_attr_guidance", False)),
        # ── PromptNu-guided v3 diagnostic: prompt source ──
        promptnu_guided_v3_prompt_source=str(getattr(args, "promptnu_guided_v3_prompt_source", "pred_attr")),
        # ── Numeric Attribute → FreqPath guidance (Exp5) ──
        enable_numeric_attr_freqpath_guidance=bool(getattr(args, "enable_numeric_attr_freqpath_guidance", False)),
        numeric_attr_freqpath_hidden_dim=int(getattr(args, "numeric_attr_freqpath_hidden_dim", 128)),
        # ── Text encoder backend ──
        enable_conch_text_encoder=_conch_required,
        # ── CONCHLESS test mode ──
        use_checkpoint_text_bank_without_conch=bool(getattr(args, "use_checkpoint_text_bank_without_conch", False)),
        # ── CLIP text encoder backend (Exp7: CLIP_BACKEND_ABLATION) ──
        clip_text_encoder=bool(getattr(args, "clip_text_encoder", False)),
        clip_text_encoder_model=str(getattr(args, "clip_text_encoder_model", "ViT-B/32")),
        clip_text_encoder_cache_path=str(getattr(args, "clip_text_encoder_cache_path", "hf_cache/clip")),
        # ── SGA-SB v1 CORRECTION: Spatial Granularity-Aligned Structure/Boundary Guidance ──
        spatial_sb_mode=str(getattr(args, "spatial_sb_mode", "none")),
        spatial_sb_branch=str(getattr(args, "spatial_sb_branch", "both")),
        spatial_structure_guidance_init=float(getattr(args, "spatial_structure_guidance_init", 0.05)),
        spatial_boundary_guidance_init=float(getattr(args, "spatial_boundary_guidance_init", 0.05)),
        spatial_instance_attr_mode=str(getattr(args, "spatial_instance_attr_mode", "none")),
        # ── RSGR Local-5 inference wiring ──
        enable_rsgr=_rsgr_enabled,
        rsgr_mode=str(getattr(args, "rsgr_mode", "no_local")),
        rsgr_num_regions=int(getattr(args, "rsgr_num_regions", 4)),
        rsgr_region_size=int(getattr(args, "rsgr_region_size", 192)),
        rsgr_injection_scale=float(getattr(args, "rsgr_injection_scale", 0.05)),
        rsgr_max_injection_ratio=float(getattr(args, "rsgr_max_injection_ratio", 0.02)),
        rsgr_prototype_source=str(getattr(args, "rsgr_prototype_source", "conch")),
        rsgr_prototype_path=_rsgr_prototype_path,
        rsgr_prototype_detach=bool(getattr(args, "rsgr_prototype_detach", True)),
        rsgr_attr_detach=bool(getattr(args, "rsgr_attr_detach", False)),
        rsgr_shuffle_scope=str(getattr(args, "rsgr_shuffle_scope", "within_sample")),
        rsgr_random_seed=int(getattr(args, "rsgr_random_seed", 42)),
        rsgr_overlap_blend=str(getattr(args, "rsgr_overlap_blend", "normalized")),
    )

    _rsgr_module_built = getattr(model, "rsgr", None) is not None
    if _rsgr_enabled and not _rsgr_module_built:
        raise RuntimeError(
            "enable_rsgr=True but model.rsgr is None; refusing silent RSGR degradation"
        )
    _rsgr_expected_buffers = {}
    if _rsgr_module_built:
        _rsgr_guard_names = ["structure_prototypes", "boundary_prototypes"]
        if str(getattr(model.rsgr, "mode", "")) == "random_prototype":
            _rsgr_guard_names.extend(
                ["random_structure_prototypes", "random_boundary_prototypes"]
            )
        _rsgr_expected_buffers = {
            name: getattr(model.rsgr, name).detach().cpu().clone()
            for name in _rsgr_guard_names
        }

    model = model.to(device)

    del vanilla_sam

    # ── [TEST_MODEL_BUILD_AUDIT] (rank0 only) ──
    if is_rank0:
        _aa_named_params = [
            n for n, _ in model.named_parameters()
            if "attr_align" in n
        ]
        _aa_sd = {k: v for k, v in model.state_dict().items() if "attr_align" in k}
        _ml_named_params = [
            n for n, _ in model.named_parameters()
            if "multilevel_attr_heads" in n
        ]
        _ml_sd = {k: v for k, v in model.state_dict().items() if "multilevel_attr_heads" in k}
        _has_sd_adapter = hasattr(model, "pnurl") and hasattr(model.pnurl, "semantic_delta_adapter") and model.pnurl.semantic_delta_adapter is not None
        _has_sc_gate = hasattr(model, "pnurl") and hasattr(model.pnurl, "semantic_channel_gate") and model.pnurl.semantic_channel_gate is not None

        print("[TEST_MODEL_BUILD_AUDIT]")
        print(f"  attr_align_named_param_count={len(_aa_named_params)}")
        print(f"  attr_align_state_dict_count={len(_aa_sd)}")
        print(f"  multilevel_attr_named_param_count={len(_ml_named_params)}")
        print(f"  multilevel_attr_state_dict_count={len(_ml_sd)}")
        print(f"  has_semantic_delta_adapter={_has_sd_adapter}")
        print(f"  has_semantic_channel_gate={_has_sc_gate}")

        _aa_expected = 8  # 4 Linear modules × (weight + bias)
        if getattr(args, "enable_attr_text_alignment", False) and len(_aa_named_params) == 0:
            raise RuntimeError(
                "[TEST_MODEL_BUILD_ERROR] attr_align heads not created. "
                "Check enable_attr_text_alignment passing to TextSam constructor."
            )
        if getattr(args, "enable_attr_text_alignment", False):
            print(f"  [OK] enable_attr_text_alignment=True, attr_align count={len(_aa_named_params)} (expected={_aa_expected})")
        if getattr(args, "enable_multilevel_attr_heads", False):
            print(f"  [OK] enable_multilevel_attr_heads=True, multilevel_attr params={len(_ml_named_params)}")

        # ── PromptNu-guided v3 audit ──
        _v3_enabled = bool(getattr(args, "enable_promptnu_guided_v3", False))
        _v3_adapter = getattr(model, "promptnu_guided_adapter", None)
        _v3_has_adapter = _v3_adapter is not None
        _v3_has_vis_proj = hasattr(model, "promptnu_guided_v3_vis_proj") and model.promptnu_guided_v3_vis_proj is not None
        _v3_has_struct_text_proj = hasattr(model, "promptnu_guided_v3_struct_text_proj") and model.promptnu_guided_v3_struct_text_proj is not None
        _v3_has_bound_text_proj = hasattr(model, "promptnu_guided_v3_boundary_text_proj") and model.promptnu_guided_v3_boundary_text_proj is not None
        _v3_named_params = [n for n, _ in model.named_parameters() if "promptnu_guided" in n]
        print(f"  enable_promptnu_guided_v3={_v3_enabled}")
        print(f"  promptnu_guided_adapter_exists={_v3_has_adapter}")
        print(f"  promptnu_guided_v3_vis_proj_exists={_v3_has_vis_proj}")
        print(f"  promptnu_guided_v3_struct_text_proj_exists={_v3_has_struct_text_proj}")
        print(f"  promptnu_guided_v3_boundary_text_proj_exists={_v3_has_bound_text_proj}")
        print(f"  promptnu_guided_v3_named_param_count={len(_v3_named_params)}")
        if _v3_enabled and not _v3_has_adapter:
            raise RuntimeError(
                "[TEST_MODEL_BUILD_ERROR] enable_promptnu_guided_v3=True but "
                "promptnu_guided_adapter is None. Check v3 kwargs passing to TextSam constructor."
            )

    # --- SB-related module status before checkpoint load ---
    _has_sb_heads = getattr(model, "structure_boundary_attr_heads", None) is not None
    _has_sb_direct = getattr(model, "sb_direct_adapter", None) is not None
    if is_rank0:
        print(
            f"[SB_PRELOAD] structure_boundary_attr_heads_exists={_has_sb_heads} | "
            f"sb_direct_adapter_exists={_has_sb_direct}"
        )

    model = load_model_checkpoint(
        model, fine_tuned_ckpt, device,
        filter_mismatch=getattr(args, "resume_filter_mismatch", False),
    )

    # A checkpoint may contain persistent RSGR prototype buffers.  The CLI bank
    # and random seed are authoritative for this test invocation; reject any
    # silent post-construction override so the logged provenance remains true.
    _rsgr_buffer_mismatches = []
    for _name, _expected in _rsgr_expected_buffers.items():
        _actual = getattr(getattr(model, "rsgr", None), _name, None)
        if _actual is None or not torch.equal(_actual.detach().cpu(), _expected):
            _rsgr_buffer_mismatches.append(_name)
    if _rsgr_buffer_mismatches:
        raise RuntimeError(
            "checkpoint overrode CLI-selected RSGR prototype buffers: "
            f"{_rsgr_buffer_mismatches}; refusing false bank provenance"
        )
    _rsgr_module_built = getattr(model, "rsgr", None) is not None
    if _rsgr_enabled and not _rsgr_module_built:
        raise RuntimeError(
            "enable_rsgr=True but model.rsgr is None after checkpoint load"
        )
    _rsgr_active_prototype_sha256 = "NOT_APPLICABLE"
    if _rsgr_module_built:
        if str(getattr(model.rsgr, "mode", "")) == "random_prototype":
            _rsgr_active_tensors = (
                model.rsgr.random_structure_prototypes,
                model.rsgr.random_boundary_prototypes,
            )
        else:
            _rsgr_active_tensors = (
                model.rsgr.structure_prototypes,
                model.rsgr.boundary_prototypes,
            )
        _rsgr_active_prototype_sha256 = _sha256_tensors_for_audit(
            _rsgr_active_tensors
        )
    if is_rank0:
        print("[TEST_CONFIG] RSGR Runtime Configuration (post-checkpoint)")
        print(f"  rsgr_enabled={bool(getattr(model, 'enable_rsgr', False))}")
        print(f"  rsgr_module_built={_rsgr_module_built}")
        print(f"  rsgr_prototype_path={_rsgr_prototype_path}")
        print(f"  rsgr_bank_sha256={_rsgr_bank_sha256}")
        print(f"  rsgr_active_prototype_sha256={_rsgr_active_prototype_sha256}")
        print(f"  rsgr_checkpoint_buffer_match={not _rsgr_buffer_mismatches}")

    # ── [CONCHLESS] Validate checkpoint text_bank buffers ──
    _is_conchless = bool(getattr(args, "use_checkpoint_text_bank_without_conch", False))
    if _is_conchless and is_rank0:
        _struct_buf = getattr(model, "_structure_text_bank_buffer", None)
        _bound_buf = getattr(model, "_boundary_text_bank_buffer", None)
        _struct_ok = _struct_buf is not None and _struct_buf.numel() > 0
        _bound_ok = _bound_buf is not None and _bound_buf.numel() > 0
        print("[CONCHLESS_CKPT_VALIDATION]")
        print(f"  _structure_text_bank_buffer: {'✅' if _struct_ok else '❌'} "
              f"{tuple(_struct_buf.shape) if _struct_ok else 'EMPTY'}")
        print(f"  _boundary_text_bank_buffer:  {'✅' if _bound_ok else '❌'} "
              f"{tuple(_bound_buf.shape) if _bound_ok else 'EMPTY'}")
        if not _struct_ok or not _bound_ok:
            print("  ⚠️  Text bank buffers missing or empty in checkpoint!")
            print(f"  Run: python NuSeg/scripts/extract_text_bank_from_ckpt.py")
            raise RuntimeError(
                "[CONCHLESS_CKPT_VALIDATION] Checkpoint missing text bank buffers. "
                "Run NuSeg/scripts/extract_text_bank_from_ckpt.py first."
            )

    # ── [TEST_INFERENCE_MODE] Disable training-only forward paths ──
    # After building attr_align heads and loading checkpoint weights,
    # disable enable_attr_text_alignment and enable_promptnu_lite_align
    # so that test forward does not need GT structure/boundary labels.
    # The attr_align modules remain built and loaded; only the forward
    # branch that calls _get_attr_text_embeddings() is suppressed.
    if getattr(args, "disable_attr_text_alignment_forward_in_test", True):
        _attr_before = getattr(model, "enable_attr_text_alignment", False)
        _pnurl_before = getattr(model, "enable_promptnu_lite_align", False)
        if _attr_before:
            model.enable_attr_text_alignment = False
        if _pnurl_before:
            model.enable_promptnu_lite_align = False
        if is_rank0:
            print("[TEST_INFERENCE_MODE]")
            print(f"  attr_align_heads_created_for_ckpt=True")
            print(f"  attr_align_weights_loaded=True")
            print(f"  enable_attr_text_alignment_forward=False")
            print(f"  enable_promptnu_lite_align_forward=False")
            print(
                f"  reason=test inference does not use GT structure/boundary labels; "
                f"PromptNu-lite loss is training-only"
            )

    # --- sb_direct_adapter_init_std override (default 0.0 = preserve checkpoint) ---
    _sb_init_std = float(getattr(args, "sb_direct_adapter_init_std", 0.0))
    if _sb_init_std > 0.0 and _has_sb_direct:
        _adapter = model.sb_direct_adapter
        torch.nn.init.normal_(_adapter.structure_out.weight, mean=0.0, std=_sb_init_std)
        torch.nn.init.normal_(_adapter.structure_out.bias, mean=0.0, std=_sb_init_std)
        torch.nn.init.normal_(_adapter.boundary_out.weight, mean=0.0, std=_sb_init_std)
        torch.nn.init.normal_(_adapter.boundary_out.bias, mean=0.0, std=_sb_init_std)
        if is_rank0:
            print(
                f"[SB_DIRECT_ADAPTER_INIT] init_std={_sb_init_std:.3e} applied | "
                f"struct_out.weight norm={_adapter.structure_out.weight.detach().float().norm().item():.6e} | "
                f"bound_out.weight norm={_adapter.boundary_out.weight.detach().float().norm().item():.6e}"
            )
    elif _sb_init_std > 0.0 and not _has_sb_direct:
        if is_rank0:
            print(f"[SB_DIRECT_ADAPTER_INIT] init_std={_sb_init_std:.3e} specified but sb_direct_adapter is None")

    model.eval()

    # --- [TEST_SB_PRED_DIRECT] One-shot diagnostic (rank0 only) ---
    _sb_mode = str(getattr(args, "sb_guidance_mode", "none"))
    _sb_enabled = _sb_mode == "pred_direct"
    if is_rank0:
        if _sb_enabled:
            print(
                f"[TEST_SB_PRED_DIRECT] enabled=True | "
                f"sb_guidance_mode={_sb_mode} | "
                f"sb_guidance_weight={getattr(args, 'sb_guidance_weight', 0.05)} | "
                f"sb_guidance_routing={getattr(args, 'sb_guidance_routing', 'structure_low_boundary_high')} | "
                f"structure_boundary_attr_heads_exists={_has_sb_heads} | "
                f"sb_direct_adapter_exists={_has_sb_direct} | "
                f"sb_direct_adapter_init_std={_sb_init_std}"
            )
        else:
            print(
                f"[TEST_SB_PRED_DIRECT] enabled=False | "
                f"sb_guidance_mode={_sb_mode}"
            )

    # ── [NUMERIC_ATTR_ROUTE_TEST_CONFIG] One-shot diagnostic (rank0 only) ──
    _numeric_route_enabled = bool(getattr(args, "enable_numeric_attr_freqpath_guidance", False))
    _numeric_proj_exists = hasattr(model, "numeric_attr_freqpath_proj") and model.numeric_attr_freqpath_proj is not None
    if is_rank0:
        print(
            "[NUMERIC_ATTR_ROUTE_TEST_CONFIG]\n"
            f"  enable_numeric_attr_freqpath_guidance={_numeric_route_enabled}\n"
            f"  numeric_attr_freqpath_proj_exists={_numeric_proj_exists}\n"
            f"  enable_structure_boundary_attr_heads={getattr(args, 'enable_structure_boundary_attr_heads', False)}\n"
            f"  sb_guidance_mode={_sb_mode}\n"
            f"  sb_guidance_weight={getattr(args, 'sb_guidance_weight', 0.05)}\n"
            f"  sb_guidance_routing={getattr(args, 'sb_guidance_routing', 'structure_low_boundary_high')}\n"
            f"  use_structure_boundary_attrs={getattr(args, 'use_structure_boundary_attrs', False)}\n"
            f"  structure_boundary_attr_path={getattr(args, 'structure_boundary_attr_path', 'None')}\n"
            f"  conch_required={getattr(args, 'conch_required', _conch_required)}\n"
            f"  use_pnurl={getattr(args, 'use_pnurl', False)}\n"
            f"  enable_promptnu_guided_v3={getattr(args, 'enable_promptnu_guided_v3', False)}"
        )

    if is_rank0:
        print(
            f"[Rank {rank}] device={device} | "
            f"asr_variant={args.asr_variant} | "
            f"asr_regression={args.asr_regression} | "
            f"use_asr={args.use_asr} | use_pnurl={args.use_pnurl} | use_coop={args.use_coop} | "
            f"max_semantic_gate={args.max_semantic_gate} | "
            f"semantic_injection_scale={getattr(args, 'semantic_injection_scale', 1.0)} | "
            f"semantic_gate_bias_init={getattr(args, 'semantic_gate_bias_init', None)} | "
            f"enable_structure_boundary_attr_heads={getattr(args, 'enable_structure_boundary_attr_heads', False)} | "
            f"sb_guidance_mode={_sb_mode} | "
            f"sb_guidance_weight={getattr(args, 'sb_guidance_weight', 0.05)} | "
            f"enable_numeric_attr_freqpath_guidance={_numeric_route_enabled}"
        )

    # ── [TEST_FORWARD_AUDIT] Comprehensive forward-mode diagnostic (rank0 only) ──
    # This audit confirms which modules exist and which forward paths are active
    # during test inference, independent of the --disable_attr_text_alignment_forward_in_test flag.
    if is_rank0:
        _use_pnurl = bool(getattr(model, "use_pnurl", False))
        _pnurl_exists = hasattr(model, "pnurl") and model.pnurl is not None
        _sd_adapter_exists = (
            _pnurl_exists
            and hasattr(model.pnurl, "semantic_delta_adapter")
            and model.pnurl.semantic_delta_adapter is not None
        )
        _sc_gate_exists = hasattr(model, "_get_semantic_channel_gate")
        _ml_heads_val = bool(getattr(model, "enable_multilevel_attr_heads", False))
        _enable_attr_val = bool(getattr(model, "enable_attr_text_alignment", False))
        _enable_pnurl_align_val = bool(getattr(model, "enable_promptnu_lite_align", False))
        # semantic_delta_forward_active: True if use_pnurl=True → PNuRL.forward() runs
        # → semantic_delta_adapter.forward() is called unconditionally at pnurl.py:795
        _sd_fwd_active = _use_pnurl and _pnurl_exists and _sd_adapter_exists
        # semantic_gate_forward_active: True if use_pnurl=True → _get_semantic_channel_gate()
        # is called unconditionally at sam.py:3355
        _sg_fwd_active = _use_pnurl

        print("[TEST_FORWARD_AUDIT]")
        print(f"  use_pnurl                              = {_use_pnurl}")
        print(f"  pnurl_module_exists                    = {_pnurl_exists}")
        print(f"  has_semantic_delta_adapter             = {_sd_adapter_exists}")
        print(f"  has_semantic_channel_gate              = {_sc_gate_exists}")
        print(f"  enable_multilevel_attr_heads           = {_ml_heads_val}")
        print(f"  enable_attr_text_alignment             = {_enable_attr_val}")
        print(f"  enable_promptnu_lite_align             = {_enable_pnurl_align_val}")
        print(f"  ────────────────────────────────────────────────────────")
        print(f"  semantic_delta_forward_active          = {_sd_fwd_active}")
        print(f"    -> PNuRL.forward() runs when use_pnurl=True, which always")
        print(f"       calls semantic_delta_adapter (pnurl.py:795).")
        print(f"    -> NOT affected by enable_attr_text_alignment or")
        print(f"       enable_promptnu_lite_align (those gate training-only losses).")
        print(f"  semantic_gate_forward_active           = {_sg_fwd_active}")
        print(f"    -> _get_semantic_channel_gate() called unconditionally")
        print(f"       when use_pnurl=True (sam.py:3355).")
        print(f"    -> NOT affected by enable_attr_text_alignment or")
        print(f"       enable_promptnu_lite_align.")
        print(f"  ────────────────────────────────────────────────────────")
        print(f"  ────────────────────────────────────────────────────────")
        _v3_enabled_audit = bool(getattr(model, "enable_promptnu_guided_v3", False))
        _v3_adapter_audit = hasattr(model, "promptnu_guided_adapter") and model.promptnu_guided_adapter is not None
        _v3_vis_proj_audit = hasattr(model, "promptnu_guided_v3_vis_proj") and model.promptnu_guided_v3_vis_proj is not None
        _v3_struct_text_proj_audit = hasattr(model, "promptnu_guided_v3_struct_text_proj") and model.promptnu_guided_v3_struct_text_proj is not None
        _v3_bound_text_proj_audit = hasattr(model, "promptnu_guided_v3_boundary_text_proj") and model.promptnu_guided_v3_boundary_text_proj is not None
        print(f"  enable_promptnu_guided_v3                  = {_v3_enabled_audit}")
        print(f"  promptnu_guided_adapter_exists             = {_v3_adapter_audit}")
        print(f"  promptnu_guided_v3_vis_proj_exists         = {_v3_vis_proj_audit}")
        print(f"  promptnu_guided_v3_struct_text_proj_exists = {_v3_struct_text_proj_audit}")
        print(f"  promptnu_guided_v3_boundary_text_proj_exists = {_v3_bound_text_proj_audit}")
        _v3_guidance_mode = str(getattr(model, "promptnu_guided_v3_guidance_mode", "scale_add"))
        _v3_scale_weight = getattr(model, "promptnu_guided_v3_scale_weight", None)
        _v3_delta_weight = float(getattr(model, "promptnu_guided_v3_delta_weight", 0.001))
        _v3_delta_init_std = float(getattr(model, "promptnu_guided_v3_delta_init_std", 1e-5))
        _v3_max_delta_ratio = float(getattr(model, "promptnu_guided_v3_max_guided_delta_ratio", 0.0))
        print(f"  promptnu_guided_v3_guidance_mode            = {_v3_guidance_mode}")
        print(f"  promptnu_guided_v3_scale_weight             = {_v3_scale_weight}")
        print(f"  promptnu_guided_v3_delta_weight             = {_v3_delta_weight}")
        print(f"  promptnu_guided_v3_delta_init_std           = {_v3_delta_init_std}")
        print(f"  promptnu_guided_v3_max_guided_delta_ratio   = {_v3_max_delta_ratio}")
        if _v3_enabled_audit:
            print(f"    -> guidance_mode={_v3_guidance_mode}: guided = sd * (1 + w_scale*tanh(scale)) + w_delta*tanh(delta)") if _v3_guidance_mode == "scale_add" else None
            print(f"    -> Cosine alignment loss is training-only; NOT computed during test.")
        else:
            print(f"    -> v3 is disabled; semantic_delta passed through unmodified.")
        # ── PromptNu-guided v3 diagnostic: prompt source & attr quality audit ──
        _v3_prompt_source = str(getattr(model, "promptnu_guided_v3_prompt_source", "pred_attr"))
        _attr_quality_audit = bool(getattr(args, "attr_quality_audit", False))
        _fixed_global_text = str(getattr(args, "promptnu_guided_v3_fixed_global_text",
                                          "cell nuclei with irregular nuclear morphology and dense chromatin"))
        print(f"  ── [V3 PROMPT SOURCE] ──")
        print(f"  promptnu_guided_v3_prompt_source           = {_v3_prompt_source}")
        print(f"    -> pred_attr:    use predicted attr logits from multilevel_attr_heads (current default)")
        print(f"    -> fixed_global: use fixed global CONCH text embedding (bypasses attr heads)")
        print(f"    -> uniform_bank: use uniform weighting over CONCH text bank (requires --promptnu_guided_v3_use_text_bank)")
        print(f"    -> oracle_gt_attr: use GT attribute labels as oracle pseudo-logits (requires attr labels in test set)")
        print(f"  attr_quality_audit                          = {_attr_quality_audit}")
        if _attr_quality_audit:
            print(f"    -> Attribute quality audit ENABLED: predicted attr logits will be compared")
            print(f"       against GT attr labels during inference.")
        print(f"  promptnu_guided_v3_fixed_global_text        = \"{_fixed_global_text}\"")
        print(f"  ────────────────────────────────────────────────────────")
        print(f"  CONCLUSION: Semantic injection path IS fully active in test mode.")
        print(f"  The enable_attr_text_alignment=False and")
        print(f"  enable_promptnu_lite_align=False only disable training-only")
        print(f"  loss branches (Phase C alignment, PNurl alignment loss).")

    return model


# ==================================================================================================
# 7b. Parallel worker (mp.Pool path, uses _build_test_model internally)
# ==================================================================================================
def process_chunk(worker_id, image_files_chunk, args):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("CUDA is required for this testing pipeline, but no GPU was detected.")

    gpu_id = worker_id % num_gpus
    device = torch.device(f"cuda:{gpu_id}")
    args.device = device

    # Build model and load checkpoint via shared helper (rank-aware logging)
    model = _build_test_model(args, device, rank=worker_id, is_rank0=(worker_id == 0))
    chunk_metrics = defaultdict(list)

    pbar = tqdm(
        image_files_chunk,
        desc=f"Worker {worker_id} (GPU {gpu_id})",
        position=worker_id,
        leave=False,
    )

    for img_path in pbar:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        json_path = os.path.splitext(img_path)[0] + ".json"
        organ_name, organ_id = extract_organ_from_json(json_path)

        prob, hv, dynamic_min_size = sliding_window_inference(
            model=model,
            image_rgb=image_rgb,
            organ_id=organ_id,
            organ_name=organ_name,
            args=args,
            patch_size=args.patch_size,
            overlap=args.overlap,
        )

        pred_mask = hover_post_process(
            prob,
            hv,
            prob_thresh=args.prob_thresh,
            marker_thresh=args.marker_thresh,
            min_marker_size=dynamic_min_size,
            final_min_object_size=args.final_min_object_size,
        )

        if pred_mask.max() == 0:
            fallback_mask = prob > args.prob_thresh
            fallback_mask = binary_fill_holes(fallback_mask)
            pred_mask = skimage_label(fallback_mask).astype(np.int32)
            pred_mask = remove_small_objects(
                pred_mask, min_size=int(args.final_min_object_size)
            ).astype(np.int32)

        if args.save_pred:
            save_prediction(pred_mask, img_path, args.output_dir)

        gt_mask = load_filtered_gt(img_path)

        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint16),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                chunk_metrics[k].append(v)

    return dict(chunk_metrics)


# ==================================================================================================
# 8. Args
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FreqPath-SAM Inference & Testing")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth")

    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--output_dir", type=str, default="test_predictions")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.8)

    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--num_organs", type=int, default=21)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder_adapter", action="store_true", default=True)

    parser.add_argument("--use_pnurl", action="store_true", default=False)
    parser.add_argument("--use_coop", action="store_true", default=False)
    parser.add_argument("--use_asr", action="store_true", default=False)

    parser.add_argument(
        "--asr_variant",
        type=str,
        default="legacy",
        choices=["legacy", "freqpath"],
        help="ASR path used to build TextSam. Must match checkpoint architecture.",
    )

    parser.add_argument(
        "--asr_regression",
        action="store_true",
        default=False,
        help="Use pure visual ASR regression mode. Usually False for semantic/freqpath testing.",
    )

    parser.add_argument("--max_semantic_gate", type=float, default=0.03)
    parser.add_argument("--max_delta_ratio", type=float, default=0.02)
    parser.add_argument("--init_delta_ratio", type=float, default=0.005)
    parser.add_argument(
        "--semantic_gate_bias_init",
        type=float,
        default=None,
        help="Override final Conv2d bias of SemanticChannelGate. Must match training value for valid ablation.",
    )
    parser.add_argument(
        "--semantic_injection_scale",
        type=float,
        default=1.0,
        help="Multiplier for channel_gate * semantic_delta. Must match training scale for valid ablation.",
    )

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="organ_static",
        choices=["base", "generic", "organ_static", "dynamic_pred"],
        help=(
            "Testing prompt mode. dynamic_pred is non-leaking two-pass inference: "
            "first predict PNuRL attributes with a static prompt, then rebuild the prompt from predicted attributes."
        ),
    )
    parser.add_argument(
        "--dynamic_pred_bootstrap_prompt",
        type=str,
        default="organ_static",
        choices=["base", "generic", "organ_static"],
        help="Bootstrap prompt used in the first pass of --prompt_mode dynamic_pred.",
    )

    parser.add_argument("--prob_thresh", type=float, default=0.40)
    parser.add_argument("--marker_thresh", type=float, default=0.45)
    # --- Structure & Boundary predicted direct guidance ---
    parser.add_argument(
        "--enable_structure_boundary_attr_heads",
        action="store_true",
        default=False,
        help="Enable structure/boundary attribute auxiliary heads in TextSam",
    )
    # --- Numeric Attribute → FreqPath guidance (Exp5) ---
    parser.add_argument(
        "--enable_numeric_attr_freqpath_guidance",
        action="store_true",
        default=False,
        help="Enable NumericAttrFreqPathProj: project predicted structure/boundary logits "
             "directly into FreqPath prompt space via flatten+MLP (Exp5). "
             "Requires --enable_multilevel_attr_heads or --enable_structure_boundary_attr_heads.",
    )
    parser.add_argument(
        "--numeric_attr_freqpath_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for NumericAttrFreqPathProj MLP (default: 128)",
    )
    parser.add_argument("--enable_local_region_text_alignment", action="store_true", default=False)
    parser.add_argument("--local_region_window_size", type=int, default=192)
    parser.add_argument("--local_region_text_temperature", type=float, default=0.07)
    parser.add_argument("--local_region_text_attributes", default="density,size_heterogeneity,crowding,boundary_irregularity,elongation")
    parser.add_argument("--local_region_policy", choices=["complete_only"], default="complete_only")
    parser.add_argument("--local_region_text_backend", choices=["conch"], default="conch")
    parser.add_argument("--local_region_text_supervision_only", action="store_true", default=False)
    parser.add_argument("--local_region_text_prototype_path", default="workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.pt")
    parser.add_argument("--enable_rsgr", action="store_true", default=False)
    parser.add_argument("--rsgr_mode", choices=["no_local", "correct_local", "shuffled_region", "random_prototype"], default="no_local")
    parser.add_argument("--rsgr_num_regions", type=int, default=4)
    parser.add_argument("--rsgr_region_size", type=int, default=192)
    parser.add_argument("--rsgr_local_attr_weight", type=float, default=0.05)
    parser.add_argument("--rsgr_semantic_align_weight", type=float, default=0.0)
    parser.add_argument("--rsgr_injection_scale", type=float, default=0.05)
    parser.add_argument("--rsgr_max_injection_ratio", type=float, default=0.02)
    parser.add_argument("--rsgr_prototype_source", choices=["conch", "synthetic"], default="conch")
    parser.add_argument("--rsgr_prototype_path", default=None)
    parser.add_argument("--rsgr_prototype_detach", type=lambda value: str(value).lower() in ("1", "true", "yes", "on"), default=True)
    parser.add_argument("--rsgr_attr_detach", type=lambda value: str(value).lower() in ("1", "true", "yes", "on"), default=False)
    parser.add_argument("--rsgr_shuffle_scope", choices=["within_sample"], default="within_sample")
    parser.add_argument("--rsgr_random_seed", type=int, default=42)
    parser.add_argument("--rsgr_overlap_blend", choices=["normalized"], default="normalized")
    parser.add_argument(
        "--sb_guidance_mode",
        type=str,
        default="none",
        choices=["none", "pred_direct"],
        help="SB guidance mode: 'none' (disabled), 'pred_direct' (predicted logits → argmax → MLP → low/high injection)",
    )
    parser.add_argument(
        "--sb_guidance_weight",
        type=float,
        default=0.05,
        help="Weight multiplier for pred_direct guidance deltas (default: 0.05)",
    )
    parser.add_argument(
        "--sb_guidance_routing",
        type=str,
        default="structure_low_boundary_high",
        choices=["structure_low_boundary_high"],
        help="Routing strategy for SB guidance",
    )
    parser.add_argument(
        "--sb_direct_adapter_init_std",
        type=float,
        default=0.0,
        help="Normal init std for sb_direct_adapter output layers. Default 0.0 = preserve checkpoint weights.",
    )

    parser.add_argument("--min_marker_size", type=int, default=12)
    parser.add_argument(
        "--final_min_object_size",
        type=int,
        default=15,
        help="Final watershed/fallback minimum object size. Canonical PanNuke v1: 15.",
    )

    # ── Resume / checkpoint compatibility ──
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Alias for --checkpoint. If both are set, --checkpoint takes precedence.",
    )
    parser.add_argument(
        "--resume_filter_mismatch",
        action="store_true",
        default=False,
        help="Filter out checkpoint tensors whose shapes do not match the current model.",
    )

    # ── Phase C: Attribute-Text Alignment ──
    parser.add_argument(
        "--enable_attr_text_alignment",
        action="store_true",
        default=False,
        help="Enable Phase C attribute-text alignment projection heads",
    )
    parser.add_argument(
        "--enable_multilevel_attr_heads",
        action="store_true",
        default=False,
        help="Enable MultiLevelAttributeHeads for visual features",
    )

    # ── PromptNu-lite v2: Residual-Coupled Semantic Alignment ──
    parser.add_argument(
        "--enable_promptnu_lite_align",
        action="store_true",
        default=False,
        help="Enable PromptNu-lite v2 residual-coupled semantic alignment",
    )
    parser.add_argument(
        "--promptnu_lite_target",
        type=str,
        default="semantic_delta",
        choices=["semantic_delta", "injected_delta", "refined_feature"],
        help="PromptNu-lite v2 visual feature target for alignment",
    )
    parser.add_argument(
        "--promptnu_lite_pool_mode",
        type=str,
        default="gap",
        choices=["gap", "absmean", "rms"],
        help="Pooling mode for PromptNu-lite v2 target feature",
    )
    parser.add_argument(
        "--promptnu_lite_struct_weight",
        type=float,
        default=0.0,
        help="Weight for structure-text alignment loss in PromptNu-lite v2",
    )
    parser.add_argument(
        "--promptnu_lite_boundary_weight",
        type=float,
        default=0.0,
        help="Weight for boundary-text alignment loss in PromptNu-lite v2",
    )
    parser.add_argument(
        "--promptnu_lite_instance_weight",
        type=float,
        default=0.0,
        help="Instance alignment weight for PromptNu-lite v2",
    )
    parser.add_argument(
        "--promptnu_lite_detach_text",
        action="store_true",
        default=True,
        help="Detach text embeddings from gradient graph in PromptNu-lite v2",
    )
    parser.add_argument(
        "--promptnu_lite_detach_visual",
        action="store_true",
        default=False,
        help="Detach visual features from gradient graph in PromptNu-lite v2",
    )
    parser.add_argument(
        "--promptnu_lite_proj_lr_mult",
        type=float,
        default=0.5,
        help="LR multiplier for attr_align projection heads when PromptNu-lite v2 enabled",
    )

    # ── PromptNu-guided v3: Scale-Based Semantic Guidance ──
    parser.add_argument(
        "--enable_promptnu_guided_v3",
        action="store_true",
        default=False,
        help="Enable PromptNu-guided v3 scale-based semantic guidance",
    )
    parser.add_argument(
        "--promptnu_guided_v3_struct_weight",
        type=float,
        default=1.0,
        help="Structure attr weight for v3 PromptNuGuidedAdapter (default: 1.0)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_boundary_weight",
        type=float,
        default=1.0,
        help="Boundary attr weight for v3 PromptNuGuidedAdapter (default: 1.0)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_text_weight",
        type=float,
        default=0.01,
        help="Text guidance weight for v3 scale-based modulation (default: 0.01)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_embed_dim",
        type=int,
        default=256,
        help="Embedding dimension for v3 PromptNuGuidedAdapter (default: 256)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for v3 PromptNuGuidedAdapter MLP (default: 128)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_vis_proj_dim",
        type=int,
        default=512,
        help="Projection dimension for v3 cosine alignment visual/text projections (default: 512)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_align_loss_weight",
        type=float,
        default=0.1,
        help="Weight for v3 cosine alignment loss (default: 0.1)",
    )
    # ── PromptNu-guided v3.1: CONCH text bank & GT align target ──
    parser.add_argument(
        "--promptnu_guided_v3_use_text_bank",
        action="store_true",
        default=False,
        help="Enable CONCH text bank for v3.1: use CONCH-encoded attribute text embeddings as soft prompt to guide semantic_delta (default: False)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_use_gt_align_target",
        action="store_true",
        default=False,
        help="Use GT attribute labels as alignment target for v3 cosine alignment loss (default: False)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_semantic_dim",
        type=int,
        default=256,
        help="Semantic projection dimension for v3.1 text-bank guidance (default: 256)",
    )
    parser.add_argument(
        "--promptnu_guided_v3_text_dim",
        type=int,
        default=512,
        help="Text feature dimension for CONCH text bank (v3.1) (default: 512)",
    )
    # ── PromptNu-guided v3.3: Scale + Additive guidance ──
    parser.add_argument(
        "--promptnu_guided_v3_guidance_mode",
        type=str,
        default="scale_add",
        choices=["scale", "additive", "scale_add"],
        help="v3.3 guidance mode: 'scale' (v3.2 compat), 'additive', 'scale_add' (default).",
    )
    parser.add_argument(
        "--promptnu_guided_v3_scale_weight",
        type=float,
        default=None,
        help="Override scale weight for v3.3. If None, defaults to promptnu_guided_v3_text_weight. Default: None.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_delta_weight",
        type=float,
        default=0.001,
        help="Weight for v3.3 additive text-delta branch. Default: 0.001.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_delta_init_std",
        type=float,
        default=1e-5,
        help="Init std for v3.3 delta_head (additive branch). Default: 1e-5.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_max_guided_delta_ratio",
        type=float,
        default=0.0,
        help="Max additive_delta_norm / base_feat_norm ratio for v3.3 amplitude clamping. Default: 0.0.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_prompt_source",
        type=str,
        default="pred_attr",
        choices=["pred_attr", "fixed_global", "uniform_bank", "oracle_gt_attr"],
        help=(
            "Prompt source for v3 guidance in test inference. "
            "pred_attr (default): use predicted attr logits from multilevel_attr_heads. "
            "fixed_global: use a fixed global CONCH text embedding (bypasses attr logits). "
            "uniform_bank: use uniform weighting over CONCH text bank (requires --promptnu_guided_v3_use_text_bank). "
            "oracle_gt_attr: use GT attribute labels as oracle pseudo-logits (requires test set attr labels)."
        ),
    )
    parser.add_argument(
        "--attr_quality_audit",
        action="store_true",
        default=False,
        help=(
            "Enable attribute quality audit: compare predicted attr logits against GT attr labels "
            "during test inference. Prints per-attribute Top-1 accuracy. "
            "Requires test set to have attr labels (structure_boundary_attr_path with test split)."
        ),
    )
    parser.add_argument(
        "--promptnu_guided_v3_fixed_global_text",
        type=str,
        default="cell nuclei with irregular nuclear morphology and dense chromatin",
        help=(
            "Fixed global text prompt used when prompt_source=fixed_global. "
            "Default: a generic nuclei description covering irregular morphology and dense chromatin."
        ),
    )
    # ── PromptNu-guided v3.3 alignment stability ──
    parser.add_argument(
        "--promptnu_guided_v3_align_eps",
        type=float,
        default=1e-8,
        help="Epsilon for PG3 alignment loss RMS (inside sqrt as eps^2). Prevents gradient explosion when semantic_delta ≈ 0. Default: 1e-8.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_cosine_eps",
        type=float,
        default=1e-8,
        help="Epsilon for PG3 cosine_similarity to prevent 0/0 → NaN. Default: 1e-8.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_min_align_delta_norm",
        type=float,
        default=0.0,
        help="Min spatial norm of semantic_delta_before_v3 for PG3 alignment. Below this, warmup gate activates. 0.0 = no gate. Default: 0.0.",
    )
    parser.add_argument(
        "--promptnu_guided_v3_align_low_norm_mode",
        type=str,
        default="detach_guided",
        choices=["detach_semantic", "detach_guided"],
        help="PG3 low-norm fallback: 'detach_semantic' (no visual gradient), 'detach_guided' (use guided delta, richer). Default: detach_guided.",
    )
    parser.add_argument(
        "--ablate_semantic_injection",
        action="store_true",
        default=False,
        help="Ablation: zero out semantic_delta entirely, bypassing PNuRL modulation",
    )
    parser.add_argument(
        "--ablate_pred_attr_guidance",
        action="store_true",
        default=False,
        help="Ablation: disable predicted-attribute branch, using only text_scale from predicted attrs",
    )

    # ── Test inference mode: disable training-only forward paths ──
    parser.add_argument(
        "--disable_attr_text_alignment_forward_in_test",
        action="store_true",
        default=True,
        help=(
            "After building attr_align heads and loading checkpoint, disable "
            "enable_attr_text_alignment and enable_promptnu_lite_align forward "
            "during test inference. Test does not use GT structure/boundary labels, "
            "so the training-only alignment/loss forward would crash on missing labels. "
            "Default: True (safe for test)."
        ),
    )

    # ── Structure & Boundary attributes ──
    parser.add_argument(
        "--use_structure_boundary_attrs",
        action="store_true",
        default=False,
        help="Load GT structure/boundary attribute labels in DataLoader",
    )
    parser.add_argument(
        "--structure_boundary_attr_path",
        type=str,
        default=None,
        help="Path to gt_structure_boundary_attr_all.jsonl",
    )

    # ── HF Hub / CONCH offline ──
    parser.add_argument(
        "--hf_hub_offline",
        action="store_true",
        default=False,
        help="Set HF_HUB_OFFLINE=1 to prevent HuggingFace download attempts.",
    )
    parser.add_argument(
        "--conch_cache_path",
        type=str,
        default=None,
        help="Set HF_HOME / HUGGINGFACE_HUB_CACHE to this path for CONCH loading.",
    )

    # ── CLIP text encoder backend (Exp7: CLIP_BACKEND_ABLATION) ────────
    parser.add_argument(
        "--clip_text_encoder",
        action="store_true",
        default=False,
        help="Use CLIP text encoder instead of CONCH for text bank generation (Exp7). "
             "Mutually exclusive with CONCH loading.",
    )
    parser.add_argument(
        "--clip_text_encoder_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model variant for text encoder backend. Default: ViT-B/32.",
    )
    parser.add_argument(
        "--clip_text_encoder_cache_path",
        type=str,
        default="hf_cache/clip",
        help="Cache directory for CLIP model download. Default: hf_cache/clip.",
    )

    # ── CONCHLESS / CHECKPOINT_TEXT_BANK mode: use text_bank from checkpoint ──
    parser.add_argument(
        "--use_checkpoint_text_bank_without_conch",
        action="store_true",
        default=False,
        help=(
            "Use pre-computed text bank embeddings embedded in the checkpoint "
            "instead of loading a text encoder (CONCH or CLIP). "
            "Requires checkpoint to contain _structure_text_bank_buffer [5,3,D] "
            "and _boundary_text_bank_buffer [4,3,D]. "
            "Works with both CONCH and CLIP backends."
        ),
    )

    # ── DataLoader ──
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Crop/patch size for DataLoader (used if dataset loading requires it).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader subprocess workers.",
    )

    # ── Debug dry-run ──
    parser.add_argument(
        "--debug_max_test_batches",
        type=int,
        default=None,
        help="Limit test batches for debug/smoke test. Default: None (no limit).",
    )

    # ── Distributed test via torchrun ──
    parser.add_argument(
        "--distributed_test",
        action="store_true",
        default=False,
        help="Enable distributed test via torchrun. "
             "Automatically detected if WORLD_SIZE > 1. Not required for single-GPU testing.",
    )

    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Evaluation manifest seed record.")
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "mAJI", "mPQ"])

    # ── Instance metric audit (debug AJI/PQ issues) ──
    parser.add_argument(
        "--instance_metric_audit",
        action="store_true",
        default=False,
        help=(
            "Enable per-batch instance metric audit logging. "
            "Prints pred/gt instance map details for first 3 batches per rank."
        ),
    )

    # ── SGA-SB v1 CORRECTION: Spatial Granularity-Aligned Structure/Boundary Guidance ──
    parser.add_argument(
        "--spatial_sb_mode",
        type=str,
        default="none",
        choices=["none", "supervision_only", "guidance"],
        help="SGA-SB v1 CORRECTION: spatial structure/boundary guidance mode. 'none'=disabled, 'supervision_only'=predict+loss only, 'guidance'=predict+loss+gamma-scaled delta injection. Default: none.",
    )
    parser.add_argument(
        "--spatial_sb_branch",
        type=str,
        default="both",
        choices=["structure", "boundary", "both"],
        help="Which branch to guide. 'structure'=low-path only, 'boundary'=high-path only, 'both'=both. Default: both.",
    )
    parser.add_argument(
        "--spatial_structure_loss_weight",
        type=float,
        default=0.1,
        help="Weight for structure SmoothL1 loss. Default: 0.1.",
    )
    parser.add_argument(
        "--spatial_boundary_loss_weight",
        type=float,
        default=0.1,
        help="Weight for boundary BCE+Dice loss. Default: 0.1.",
    )
    parser.add_argument(
        "--spatial_structure_guidance_init",
        type=float,
        default=0.05,
        help="Initial value for learnable gamma_structure (scales structure delta). Default: 0.05.",
    )
    parser.add_argument(
        "--spatial_boundary_guidance_init",
        type=float,
        default=0.05,
        help="Initial value for learnable gamma_boundary (scales boundary delta). Default: 0.05.",
    )
    parser.add_argument(
        "--spatial_instance_attr_mode",
        type=str,
        default="none",
        choices=["none", "v1"],
        help="Legacy ablation: keep old 18ch SpatialInstanceAttrHead (v1 morphology attrs). 'none' = disabled (use new structure/boundary heads). Default: none.",
    )

    args = parser.parse_args()

    # ── Normalize metric name aliases ──
    # Accept both 'aji'/'mAJI' and 'pq'/'mPQ' from command line
    _METRIC_ALIAS = {'aji': 'mAJI', 'pq': 'mPQ', 'dq': 'mDQ', 'sq': 'mSQ'}
    args.metrics = [_METRIC_ALIAS.get(m, m) for m in args.metrics]

    # --resume as alias for --checkpoint
    if args.checkpoint is None and args.resume is not None:
        args.checkpoint = args.resume

    return args


# ==================================================================================================
# 9. Main
# ==================================================================================================
def _write_test_run_audit(args, image_files, world_size, sampler_type):
    protocol = protocol_from_test_args(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_audit_dir = os.path.join(args.output_dir, f"run_audit_{timestamp}")
    stable_ids = [os.path.relpath(path, args.data_path) for path in image_files]
    protocol_path = write_evaluation_protocol(protocol, run_audit_dir)
    manifest_paths = write_run_manifests(
        run_dir=run_audit_dir,
        run_name=os.path.basename(os.path.normpath(args.output_dir)) or "full_test",
        args=args,
        protocol=protocol,
        project_root=os.path.dirname(os.path.abspath(__file__)),
        parent_checkpoint=args.checkpoint,
        evaluation_context={
            "data_split": args.data_path,
            "sample_count": len(stable_ids),
            "unique_sample_count": len(set(stable_ids)),
            "duplicate_sample_count": len(stable_ids) - len(set(stable_ids)),
            "world_size": world_size,
            "sampler_type": sampler_type,
        },
    )
    print("[EVALUATION_PROTOCOL] " + protocol.to_json().replace("\n", " "), flush=True)
    print(f"[RUN_MANIFEST] protocol={protocol_path} files={manifest_paths}", flush=True)


def main(args):
    # ═══════════════════════════════════════════════════════════════
    # HF Hub / CONCH offline env application (for subprocess compatibility)
    # ═══════════════════════════════════════════════════════════════
    if getattr(args, "hf_hub_offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
    _conch_cache = getattr(args, "conch_cache_path", None)
    if _conch_cache is not None:
        os.environ["HF_HOME"] = _conch_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = _conch_cache

    # ═══════════════════════════════════════════════════════════════
    # CONCHLESS test mode diagnostic
    # ═══════════════════════════════════════════════════════════════
    _conchless = bool(getattr(args, "use_checkpoint_text_bank_without_conch", False))
    if _conchless:
        print("[CONCHLESS_TEST_CONFIG]")
        print(f"  use_checkpoint_text_bank_without_conch=True")
        print(f"  CONCH text encoder will NOT be loaded from HuggingFace.")
        print(f"  Text bank will be read from checkpoint buffers:")
        print(f"    _structure_text_bank_buffer: expected [5, 3, D]")
        print(f"    _boundary_text_bank_buffer:  expected [4, 3, D]")
        print(f"  ⚠️  Ensure checkpoint was processed with extract_text_bank_from_ckpt.py first.")

    # ═══════════════════════════════════════════════════════════════
    # Distributed detection: torchrun sets WORLD_SIZE.
    # If WORLD_SIZE > 1, enter DDP path; otherwise, classic mp.Pool path.
    # ═══════════════════════════════════════════════════════════════
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        # ── DDP path ──────────────────────────────────────────────
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        args.device = device
        is_rank0 = (rank == 0)

        print(f"[TEST_DDP_CONFIG]")
        print(f"  distributed=True")
        print(f"  world_size={world_size}")
        print(f"  rank={rank}")
        print(f"  local_rank={local_rank}")
        print(f"  device={device}")

        # ── Data validation (checked on all ranks) ──
        if not os.path.isdir(args.data_path):
            raise FileNotFoundError(f"data_path does not exist or is not a directory: {args.data_path}")
        if args.checkpoint is None or not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

        image_files = sorted([
            os.path.join(args.data_path, f)
            for f in os.listdir(args.data_path)
            if f.lower().endswith((".png", ".tif", ".tiff"))
        ])
        if len(image_files) == 0:
            raise RuntimeError(f"No image files found in {args.data_path}")

        # No-padding strided sharding: every stable global index appears exactly once.
        sampler = DistributedEvalSampler(
            range(len(image_files)), num_replicas=world_size, rank=rank
        )
        my_indices = list(iter(sampler))
        my_files = [image_files[i] for i in my_indices]

        # Apply any debug limit before sample audit/manifest accounting.
        _debug_max = getattr(args, "debug_max_test_batches", None)
        if _debug_max is not None and _debug_max > 0:
            _orig = len(my_files)
            my_files = my_files[:_debug_max]
            print(f"\n🔬 [DEBUG_MAX_TEST_BATCHES] Per-rank debug limit: {_debug_max} batches (was {_orig})")

        my_sample_ids = [os.path.relpath(path, args.data_path) for path in my_files]
        gathered_sample_ids = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_sample_ids, my_sample_ids)
        global_sample_ids = [sample_id for ids in gathered_sample_ids for sample_id in ids]
        global_unique_ids = set(global_sample_ids)
        duplicates_removed = len(global_sample_ids) - len(global_unique_ids)
        if duplicates_removed:
            raise RuntimeError(
                f"DDP test sample_id collision before metrics: duplicates={duplicates_removed}"
            )

        print(
            f"[DDP_SAMPLE_AUDIT] rank={rank} local_seen={len(my_sample_ids)} "
            f"global_seen_before_dedup={len(global_sample_ids)} "
            f"global_unique={len(global_unique_ids)} duplicates_removed={duplicates_removed}",
            flush=True,
        )
        print(f"  sampler=DistributedEvalSampler(no_padding)")
        if is_rank0:
            print(
                f"  Total images: {len(image_files)}, per-rank: "
                f"{[len(ids) for ids in gathered_sample_ids]}"
            )
            evaluated_files = [
                os.path.join(args.data_path, sample_id) for sample_id in global_sample_ids
            ]
            _write_test_run_audit(
                args, evaluated_files, world_size, "DistributedEvalSampler(no_padding)"
            )

        # ── Build model (rank-aware logging inside) ──
        try:
            model = _build_test_model(args, device, rank=rank, is_rank0=is_rank0)

            # ── Per-rank output directory ──
            if args.save_pred:
                rank_output_dir = os.path.join(args.output_dir, f"rank{rank}")
                os.makedirs(rank_output_dir, exist_ok=True)
            else:
                rank_output_dir = args.output_dir

            # ── Process images: accumulate local sums + counts ──
            local_sums = {k: 0.0 for k in args.metrics}
            local_counts = {k: 0 for k in args.metrics}

            pbar = tqdm(
                my_files,
                desc=f"Rank {rank}",
                position=0,
                leave=True,
            )

            _audit_counter = 0

            for img_path in pbar:
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                json_path = os.path.splitext(img_path)[0] + ".json"
                organ_name, organ_id = extract_organ_from_json(json_path)

                prob, hv, dynamic_min_size = sliding_window_inference(
                    model=model,
                    image_rgb=image_rgb,
                    organ_id=organ_id,
                    organ_name=organ_name,
                    args=args,
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                )

                pred_mask = hover_post_process(
                    prob, hv,
                    prob_thresh=args.prob_thresh,
                    marker_thresh=args.marker_thresh,
                    min_marker_size=dynamic_min_size,
                    final_min_object_size=args.final_min_object_size,
                )

                if pred_mask.max() == 0:
                    fallback_mask = prob > args.prob_thresh
                    fallback_mask = binary_fill_holes(fallback_mask)
                    pred_mask = skimage_label(fallback_mask).astype(np.int32)
                    pred_mask = remove_small_objects(
                        pred_mask, min_size=int(args.final_min_object_size)
                    ).astype(np.int32)

                if args.save_pred:
                    save_prediction(pred_mask, img_path, rank_output_dir)

                gt_mask = load_filtered_gt(img_path)

                # ── Instance metric audit (first 3 batches per rank) ──
                if getattr(args, "instance_metric_audit", False) and _audit_counter < 3:
                    _audit_batch_idx = _audit_counter
                    _audit_counter += 1
                    # Compute instance-level stats for audit
                    _pred_uniq = np.unique(pred_mask)
                    _pred_inst_ids = _pred_uniq[_pred_uniq > 0]
                    _gt_uniq = np.unique(gt_mask) if gt_mask is not None else np.array([], dtype=np.int32)
                    _gt_inst_ids = _gt_uniq[_gt_uniq > 0] if gt_mask is not None else np.array([], dtype=np.int32)

                    _img_name = os.path.basename(img_path)
                    print(
                        f"\n[INSTANCE_METRIC_AUDIT]"
                        f"\n  rank={rank}"
                        f"\n  batch_idx={_audit_batch_idx}"
                        f"\n  image_name={_img_name}"
                        f"\n  pred_mask_shape={pred_mask.shape}"
                        f"\n  pred_mask_unique={len(_pred_uniq)} (instances={len(_pred_inst_ids)})"
                        f"\n  pred_inst_shape={pred_mask.shape}"
                        f"\n  pred_inst_min={int(pred_mask.min())}"
                        f"\n  pred_inst_max={int(pred_mask.max())}"
                        f"\n  pred_inst_num_instances={len(_pred_inst_ids)}"
                        f"\n  prob_thresh={args.prob_thresh}"
                        f"\n  marker_thresh={args.marker_thresh}"
                        f"\n  min_marker_size={dynamic_min_size}"
                        f"\n  gt_mask_shape={gt_mask.shape if gt_mask is not None else 'None'}"
                        f"\n  gt_mask_unique={len(_gt_uniq) if gt_mask is not None else 0}"
                        f"\n  gt_inst_shape={gt_mask.shape if gt_mask is not None else 'None'}"
                        f"\n  gt_inst_min={int(gt_mask.min()) if gt_mask is not None else 'N/A'}"
                        f"\n  gt_inst_max={int(gt_mask.max()) if gt_mask is not None else 'N/A'}"
                        f"\n  gt_inst_num_instances={len(_gt_inst_ids) if gt_mask is not None else 0}",
                        flush=True,
                    )

                    if gt_mask is not None and len(_pred_inst_ids) > 0 and len(_gt_inst_ids) > 0:
                        # Compute AJI intersection/union for audit
                        _aji_val = get_fast_aji(gt_mask, pred_mask)
                        _pq_val, _dq_val, _sq_val = get_fast_pq(gt_mask, pred_mask)
                        print(
                            f"  aji_valid=True"
                            f"\n  pq_valid=True"
                            f"\n  aji_value={_aji_val:.6f}"
                            f"\n  pq_value={_pq_val:.6f}",
                            flush=True,
                        )
                    elif gt_mask is not None:
                        print(
                            f"  aji_valid=False (pred_inst={len(_pred_inst_ids)}, gt_inst={len(_gt_inst_ids)})"
                            f"\n  pq_valid=False (pred_inst={len(_pred_inst_ids)}, gt_inst={len(_gt_inst_ids)})",
                            flush=True,
                        )
                    else:
                        print(
                            f"  aji_valid=False (gt_mask=None)"
                            f"\n  pq_valid=False (gt_mask=None)",
                            flush=True,
                        )
                    print(flush=True)

                if gt_mask is not None:
                    if gt_mask.shape != pred_mask.shape:
                        pred_mask = cv2.resize(
                            pred_mask.astype(np.uint16),
                            (gt_mask.shape[1], gt_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(np.int32)

                    res = SegMetrics(pred_mask, gt_mask, args.metrics)
                    for k, v in res.items():
                        local_sums[k] += v
                        local_counts[k] += 1

            # ── All-reduce metrics across ranks ──
            metric_keys = list(args.metrics)
            local_vals_t = torch.tensor([local_sums[k] for k in metric_keys], device=device)
            local_nums_t = torch.tensor([local_counts[k] for k in metric_keys], device=device)

            dist.all_reduce(local_vals_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_nums_t, op=dist.ReduceOp.SUM)

            if is_rank0:
                print(f"\n[TEST_DDP_METRIC_REDUCE]")
                for i, k in enumerate(metric_keys):
                    g_val = local_vals_t[i].item() / max(local_nums_t[i].item(), 1)
                    print(f"  {k}: local_sum={local_sums[k]:.4f}, global_sum={local_vals_t[i].item():.4f}, "
                          f"local_count={local_counts[k]}, global_count={int(local_nums_t[i].item())}, "
                          f"global_mean={g_val:.4f}")

                print("\n" + "🌟" * 15)
                print("📊 Final Results (DDP aggregated):")
                for i, k in enumerate(metric_keys):
                    g_val = local_vals_t[i].item() / max(local_nums_t[i].item(), 1)
                    print(f"{k:>10}: {g_val:.4f}")
                print("🌟" * 15 + "\n")

        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return
    # ═══════════════════════════════════════════════════════════════
    # Original single-GPU / mp.Pool path (completely unchanged)
    # ═══════════════════════════════════════════════════════════════
    mp.set_start_method("spawn", force=True)

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"data_path does not exist or is not a directory: {args.data_path}")

    if args.checkpoint is None or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    image_files = [
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.lower().endswith((".png", ".tif", ".tiff"))
    ]

    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in {args.data_path}")

    # ── Debug dry-run: limit test batches ──
    _debug_max = getattr(args, "debug_max_test_batches", None)
    if _debug_max is not None and _debug_max > 0:
        # Each batch = 1 image in single-GPU mode. Limit total images processed.
        _orig_count = len(image_files)
        image_files = image_files[:_debug_max]
        print(f"\n🔬 [DEBUG_MAX_TEST_BATCHES] Limiting to {_debug_max} images (was {_orig_count})")

    _write_test_run_audit(args, image_files, 1, "single_process_sorted_full_list")
    print(
        f"[DDP_SAMPLE_AUDIT] rank=0 local_seen={len(image_files)} "
        f"global_seen_before_dedup={len(image_files)} "
        f"global_unique={len(set(image_files))} "
        f"duplicates_removed={len(image_files) - len(set(image_files))}",
        flush=True,
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("CUDA is required for this testing pipeline, but no GPU was detected.")

    workers_per_gpu = max(1, int(args.workers_per_gpu))
    num_workers = min(num_gpus * workers_per_gpu, len(image_files))

    chunk_size = math.ceil(len(image_files) / num_workers)
    chunks = [
        image_files[i:i + chunk_size]
        for i in range(0, len(image_files), chunk_size)
    ]

    print(f"\n🚀 System Detected {num_gpus} GPUs. Launching {len(chunks)} parallel Workers.")
    print(f"🔥 Testing Pipeline: overlap={args.overlap}, patch_size={args.patch_size}, TTA=8x, MultiMask=ON")
    print(f"🧠 Prompt mode: {args.prompt_mode}")
    if str(args.prompt_mode).lower().strip() == "dynamic_pred":
        print(f"🧠 dynamic_pred bootstrap prompt: {args.dynamic_pred_bootstrap_prompt}")
    print(
        f"🧩 Modules: use_asr={args.use_asr}, use_pnurl={args.use_pnurl}, "
        f"use_coop={args.use_coop}, use_ot=False"
    )
    print(
        f"🧬 ASR: asr_variant={args.asr_variant}, asr_regression={args.asr_regression}, "
        f"max_semantic_gate={args.max_semantic_gate}, "
        f"max_delta_ratio={args.max_delta_ratio}, init_delta_ratio={args.init_delta_ratio}"
    )
    print(
        f"🔬 Semantic Ablation: semantic_gate_bias_init={getattr(args, 'semantic_gate_bias_init', None)}, "
        f"semantic_injection_scale={getattr(args, 'semantic_injection_scale', 1.0)}"
    )
    print(f"� Checkpoint: {args.checkpoint}")
    print(f"📂 Data path: {args.data_path}")

    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append((i, chunk, args))

    all_metrics = defaultdict(list)

    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(process_chunk, tasks)

    for res in results:
        for k, v in res.items():
            all_metrics[k].extend(v)

    print("\n" + "🌟" * 15)
    print("📊 Final Results:")

    for k in args.metrics:
        values = all_metrics.get(k, [])
        if len(values) == 0:
            print(f"{k:>10}: N/A")
        else:
            print(f"{k:>10}: {np.mean(values):.4f}")

    print("🌟" * 15 + "\n")


if __name__ == "__main__":
    main(parse_args())
