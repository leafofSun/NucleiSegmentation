import argparse
import datetime
import gc
import os
import random
import traceback
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from skimage.measure import label as skimage_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from DataLoader import UniversalDataset, stack_dict_batched
from hover_loss import generate_hv_map_from_inst, msge_loss
from metrics import SegMetrics
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
from utils import FocalDiceloss_IoULoss, density_map_loss, get_logger, point_guidance_loss


ARCHITECTURE_VERSION = "freqpath_sam_asr_legacy_recovery_v1"
VALID_STAGES = ("vision", "pnurl_warmup", "semantic_injection")
VALID_ASR_VARIANTS = ("legacy", "freqpath")
VALID_ASR_REGRESSION_STAGES = ("freeze_decoder", "finetune_decoder")


# Speedup settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ==================================================================================================
# 1. Configuration
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FreqPath-SAM staged training with legacy ASR recovery switch"
    )

    parser.add_argument("--work_dir", type=str, default="workdir", help="Directory to save logs and models")
    parser.add_argument("--run_name", type=str, default="mp_sam_pannuke_final", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation")

    parser.add_argument("--data_path", type=str, default="data/PanNuke", help="Root directory of dataset")
    parser.add_argument(
        "--knowledge_path",
        type=str,
        default="data/PanNuke/medical_knowledge.json",
        help="Path to medical knowledge file",
    )
    parser.add_argument("--image_size", type=int, default=512, help="SAM input resolution")
    parser.add_argument("--crop_size", type=int, default=256, help="Patch size")
    parser.add_argument("--mask_num", type=int, default=1, help="Number of masks per proposal")

    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="Backbone")
    parser.add_argument("--sam_checkpoint", type=str, default="workdir/models/sam-med2d_b.pth", help="SAM checkpoint")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP/CONCH model name")
    parser.add_argument("--num_organs", type=int, default=21, help="Number of organ categories")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for prompt generator")

    parser.add_argument("--encoder_adapter", action="store_true", default=True, help="Use image encoder adapters")
    parser.add_argument("--use_pnurl", action="store_true", default=False, help="Enable PNuRL branch")
    parser.add_argument("--use_coop", action="store_true", default=False, help="Enable CoOp prompt learner")
    parser.add_argument("--use_asr", action="store_true", default=False, help="Enable CNN/ASR high-frequency branch")

    parser.add_argument(
        "--asr_variant",
        type=str,
        default="legacy",
        choices=list(VALID_ASR_VARIANTS),
        help="ASR variant: legacy for pure-visual ASR recovery, freqpath for low/high-frequency semantic branch",
    )
    parser.add_argument(
        "--asr_regression",
        action="store_true",
        default=False,
        help="Enable pure-visual ASR regression mode. Forces PNuRL/CoOp off and base prompt.",
    )
    parser.add_argument(
        "--asr_regression_stage",
        type=str,
        default="finetune_decoder",
        choices=list(VALID_ASR_REGRESSION_STAGES),
        help="ASR regression schedule: freeze_decoder first trains HV/heatmap/ASR, finetune_decoder unlocks decoder.",
    )

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="organ_static",
        choices=["base", "organ_static", "dynamic_gt", "dynamic_pred"],
        help="Training text prompt mode: base | organ_static | dynamic_gt | dynamic_pred",
    )
    parser.add_argument(
        "--eval_prompt_mode",
        type=str,
        default="base",
        choices=["base", "organ_static", "dynamic_gt", "dynamic_pred"],
        help="Validation prompt mode. ASR regression should use base.",
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="vision",
        choices=list(VALID_STAGES),
        help="Training stage: vision | pnurl_warmup | semantic_injection",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable AMP and use FP32")

    parser.add_argument("--mask_weight", type=float, default=10.0)
    parser.add_argument("--heatmap_weight", type=float, default=1.0)
    parser.add_argument("--hv_weight", type=float, default=10.0)
    parser.add_argument("--pnurl_weight", type=float, default=0.1, help="Weight for PNuRL attribute/classification loss")
    parser.add_argument("--attr_weight", type=float, default=None, help="Backward-compatible alias for pnurl_weight")
    parser.add_argument("--density_map_weight", type=float, default=0.5)

    parser.add_argument("--cnn_lr_ratio", type=float, default=None, help="LR ratio for ResNet/CNN high-frequency branch")
    parser.add_argument("--prompt_generator_lr_mult", type=float, default=None, help="LR multiplier for prompt generator")
    parser.add_argument("--adapter_lr_ratio", type=float, default=None, help="LR ratio for image encoder adapters")
    parser.add_argument("--asr_lr_ratio", type=float, default=1.0, help="LR ratio for GlobalASRUpsampler")

    parser.add_argument("--max_semantic_gate", type=float, default=0.10, help="Upper bound for SemanticChannelGate output")
    parser.add_argument("--max_delta_ratio", type=float, default=0.10, help="Upper bound for semantic_delta/base feature RMS ratio")
    parser.add_argument("--init_delta_ratio", type=float, default=0.02, help="Initial semantic_delta/base feature RMS ratio")
    parser.add_argument("--semantic_delta_reg_weight", type=float, default=1.0, help="Weight for semantic_delta relative-scale regularization")
    parser.add_argument("--injection_ratio_weight", type=float, default=10.0, help="Weight for injection_ratio hinge regularization")
    parser.add_argument("--max_injection_ratio", type=float, default=0.02, help="Soft upper bound for injected_delta_norm/base_feat_norm")
    parser.add_argument(
        "--stage_c_train_image_adapter",
        action="store_true",
        default=False,
        help="Unfreeze image encoder adapters during Stage C. Disabled by default for stable semantic injection.",
    )

    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--consistency_warmup_epochs", type=int, default=50)

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to load model weights from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from")
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "mAJI", "mPQ"], help="Metrics to evaluate")

    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        default=True,
        help="Use find_unused_parameters=True for staged training safety",
    )

    args = parser.parse_args()

    if args.attr_weight is not None:
        args.pnurl_weight = args.attr_weight


    if args.asr_regression:
        if args.phase != "vision":
            raise ValueError("--asr_regression can only be used with --phase vision.")
        args.use_asr = True
        args.use_pnurl = False
        args.use_coop = False
        args.prompt_mode = "base"
        args.eval_prompt_mode = "base"
        # 先排除 bf16/AMP 对 HV/heatmap 回归的干扰。
        args.use_amp = False

    if args.cnn_lr_ratio is None:
        args.cnn_lr_ratio = 0.1 if args.asr_regression else 0.5
    if args.prompt_generator_lr_mult is None:
        args.prompt_generator_lr_mult = 5.0 if args.asr_regression else 1.0
    if args.adapter_lr_ratio is None:
        args.adapter_lr_ratio = 1.0 if args.asr_regression else 0.1

    args.architecture_version = ARCHITECTURE_VERSION
    return args


# ==================================================================================================
# 2. Generic utilities and post-processing
# ==================================================================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is None:
            device_input[key] = value
        elif isinstance(value, torch.Tensor):
            device_input[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list):
            device_input[key] = value
        else:
            device_input[key] = value
    return device_input


def resize_pos_embed(state_dict, model_state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                if "pos_embed" in k:
                    v = v.permute(0, 3, 1, 2)
                    v = F.interpolate(v, size=model_state_dict[k].shape[1:3], mode="bicubic", align_corners=False)
                    v = v.permute(0, 2, 3, 1)
                elif "rel_pos" in k:
                    v = v.unsqueeze(0).permute(0, 2, 1)
                    target_len = model_state_dict[k].shape[0]
                    v = F.interpolate(v, size=target_len, mode="linear", align_corners=False)
                    v = v.permute(0, 2, 1).squeeze(0)
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def hover_post_process(prob_map, hv_map, prob_thresh=0.45, marker_thresh=0.4, min_marker_size=10):
    mask = prob_map > prob_thresh
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)
    diff_v = np.gradient(v_map, axis=0)
    diff_h = np.gradient(h_map, axis=1)
    sobel_mag = np.sqrt(diff_v ** 2 + diff_h ** 2)

    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    marker_map = remove_small_objects(marker_map, min_size=min_marker_size)
    markers = skimage_label(marker_map).astype(np.int32)
    inst_map = watershed(-prob_map, markers, mask=mask)
    return inst_map.astype(np.int32)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_requires_grad(module: Optional[nn.Module], flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


def set_named_requires_grad(module: Optional[nn.Module], flag: bool, include_keywords: Iterable[str]):
    if module is None:
        return
    include_keywords = tuple(include_keywords)
    for name, p in module.named_parameters():
        if any(key in name for key in include_keywords):
            p.requires_grad = flag


def set_semantic_gate_state(raw_model: nn.Module, stage: str):
    close_value = -12.0
    warm_value = -10.0
    open_value = -5.0
    gate_trainable = stage == "semantic_injection"

    pnurl = getattr(raw_model, "pnurl", None)

    if pnurl is not None and hasattr(pnurl, "residual_gate"):
        gate = pnurl.residual_gate
        if isinstance(gate, torch.nn.Parameter):
            with torch.no_grad():
                if stage == "vision":
                    gate.fill_(close_value)
                elif stage == "pnurl_warmup":
                    gate.fill_(warm_value)
                elif stage == "semantic_injection":
                    gate.fill_(open_value)
            gate.requires_grad = gate_trainable

    gate_modules = []
    if pnurl is not None and hasattr(pnurl, "semantic_channel_gate"):
        gate_modules.append(pnurl.semantic_channel_gate)

    for module_name in (
        "semantic_channel_gate",
        "channel_gate",
        "semantic_gate",
        "semantic_injection_gate",
    ):
        gate_module = getattr(raw_model, module_name, None)
        if gate_module is not None:
            gate_modules.append(gate_module)

    seen_gate_ids = set()
    for gate_module in gate_modules:
        if id(gate_module) in seen_gate_ids:
            continue
        seen_gate_ids.add(id(gate_module))

        set_requires_grad(gate_module, gate_trainable)

        if hasattr(gate_module, "reset_to_closed") and stage in ("vision", "pnurl_warmup"):
            gate_module.reset_to_closed()

        if hasattr(gate_module, "gate"):
            try:
                final_layer = gate_module.gate[-1]
                if hasattr(final_layer, "bias") and final_layer.bias is not None:
                    with torch.no_grad():
                        if stage == "vision":
                            final_layer.bias.fill_(close_value)
                        elif stage == "pnurl_warmup":
                            final_layer.bias.fill_(warm_value)
                        elif stage == "semantic_injection":
                            final_layer.bias.fill_(open_value)
            except Exception:
                pass

    raw_model.training_stage = stage
    raw_model.semantic_injection_enabled = gate_trainable


# ==================================================================================================
# 3. Stage policy and optimizer routing
# ==================================================================================================
def apply_stage_policy(model: nn.Module, stage: str, args=None, logger=None, rank: int = 0):
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage}. Expected one of {VALID_STAGES}.")

    raw_model = unwrap_model(model)

    set_requires_grad(raw_model, False)
    set_requires_grad(getattr(raw_model, "clip_model", None), False)
    set_requires_grad(getattr(raw_model, "prompt_encoder", None), False)

    if getattr(args, "asr_regression", False):
        if hasattr(raw_model, "use_pnurl"):
            raw_model.use_pnurl = False
        if hasattr(raw_model, "use_coop"):
            raw_model.use_coop = False
    else:
        if hasattr(raw_model, "use_pnurl"):
            raw_model.use_pnurl = stage in ("pnurl_warmup", "semantic_injection") and hasattr(raw_model, "pnurl")
        if hasattr(raw_model, "use_coop"):
            raw_model.use_coop = stage == "semantic_injection" and hasattr(raw_model, "prompt_learner")

    if stage == "vision":
        freeze_decoder = (
            args is not None
            and getattr(args, "asr_regression", False)
            and getattr(args, "asr_regression_stage", "finetune_decoder") == "freeze_decoder"
        )

        set_requires_grad(getattr(raw_model, "mask_decoder", None), not freeze_decoder)
        set_requires_grad(getattr(raw_model, "prompt_generator", None), True)
        set_requires_grad(getattr(raw_model, "basic_hv_head", None), True)

        if getattr(raw_model, "use_asr", False):
            set_requires_grad(getattr(raw_model, "cnn_stage0", None), True)
            set_requires_grad(getattr(raw_model, "cnn_stage1", None), True)
            set_requires_grad(getattr(raw_model, "cnn_stage2", None), True)
            set_requires_grad(getattr(raw_model, "global_asr_upsampler", None), True)

        if args is not None and getattr(args, "encoder_adapter", False):
            set_named_requires_grad(getattr(raw_model, "image_encoder", None), True, include_keywords=("Adapter", "adapter"))

        set_semantic_gate_state(raw_model, stage)
        if args is not None and getattr(args, "asr_regression", False):
            msg = (
                f"[ASR regression: {getattr(args, 'asr_regression_stage', 'finetune_decoder')}] "
                "Pure-visual legacy ASR. PNuRL/CoOp/OT disabled. "
                "Use base prompt and recover SamMed2D+ResNet+ASR baseline first."
            )
        else:
            msg = "[Stage A: vision] Train mask decoder / prompt generator / HV-ASR / image adapters. Freeze PNuRL and CoOp."

    elif stage == "pnurl_warmup":
        set_requires_grad(getattr(raw_model, "pnurl", None), True)
        set_requires_grad(getattr(raw_model, "prompt_learner", None), False)
        set_semantic_gate_state(raw_model, stage)
        msg = "[Stage B: pnurl_warmup] Train PNuRL only. No segmentation loss is routed to PNuRL. Gate remains closed."

    else:
        set_requires_grad(getattr(raw_model, "pnurl", None), True)
        set_requires_grad(getattr(raw_model, "prompt_learner", None), True)
        set_requires_grad(getattr(raw_model, "prompt_generator", None), True)
        set_requires_grad(getattr(raw_model, "mask_decoder", None), True)
        set_requires_grad(getattr(raw_model, "basic_hv_head", None), True)

        if getattr(raw_model, "use_asr", False):
            set_requires_grad(getattr(raw_model, "global_asr_upsampler", None), True)
            set_requires_grad(getattr(raw_model, "cnn_stage0", None), True)
            set_requires_grad(getattr(raw_model, "cnn_stage1", None), True)
            set_requires_grad(getattr(raw_model, "cnn_stage2", None), True)

        if (
            args is not None
            and getattr(args, "encoder_adapter", False)
            and getattr(args, "stage_c_train_image_adapter", False)
        ):
            set_named_requires_grad(getattr(raw_model, "image_encoder", None), True, include_keywords=("Adapter", "adapter"))

        set_semantic_gate_state(raw_model, stage)
        msg = (
            "[Stage C: semantic_injection] Open controlled SemanticChannelGate. "
            "Segmentation loss can update bounded PNuRL/CoOp/gate. "
            "Image adapters stay frozen unless --stage_c_train_image_adapter is set."
        )

    if rank == 0 and logger is not None:
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in raw_model.parameters())
        logger.info(msg)
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / max(total, 1):.2%})")

    return raw_model


def build_optimizer_by_stage(model: nn.Module, stage: str, args, logger=None, rank: int = 0):
    raw_model = unwrap_model(model)
    seen = set()
    param_groups = []

    def add_named_params(name: str, named_params: Iterable[Tuple[str, torch.nn.Parameter]], lr: float, weight_decay=None):
        params = []
        for _, p in named_params:
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
        if params:
            group = {"params": params, "lr": lr, "name": name}
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            param_groups.append(group)

    def module_named_params(module: Optional[nn.Module]):
        if module is None:
            return []
        return list(module.named_parameters())

    base_lr = args.lr

    if stage == "vision":
        add_named_params("mask_decoder", module_named_params(getattr(raw_model, "mask_decoder", None)), base_lr)
        add_named_params(
            "prompt_generator",
            module_named_params(getattr(raw_model, "prompt_generator", None)),
            base_lr * float(getattr(args, "prompt_generator_lr_mult", 1.0)),
        )
        add_named_params("basic_hv_head", module_named_params(getattr(raw_model, "basic_hv_head", None)), base_lr)
        add_named_params(
            "global_asr_upsampler",
            module_named_params(getattr(raw_model, "global_asr_upsampler", None)),
            base_lr * float(getattr(args, "asr_lr_ratio", 1.0)),
        )

        cnn_lr = base_lr * float(getattr(args, "cnn_lr_ratio", 0.1))
        add_named_params("cnn_stage0", module_named_params(getattr(raw_model, "cnn_stage0", None)), cnn_lr)
        add_named_params("cnn_stage1", module_named_params(getattr(raw_model, "cnn_stage1", None)), cnn_lr)
        add_named_params("cnn_stage2", module_named_params(getattr(raw_model, "cnn_stage2", None)), cnn_lr)

        adapter_params = []
        image_encoder = getattr(raw_model, "image_encoder", None)
        if image_encoder is not None:
            adapter_params = [
                (n, p)
                for n, p in image_encoder.named_parameters()
                if ("Adapter" in n or "adapter" in n) and p.requires_grad
            ]
        add_named_params(
            "image_encoder_adapter",
            adapter_params,
            base_lr * float(getattr(args, "adapter_lr_ratio", 0.1)),
        )

    elif stage == "pnurl_warmup":
        add_named_params("pnurl", module_named_params(getattr(raw_model, "pnurl", None)), base_lr)

    elif stage == "semantic_injection":
        add_named_params(
            "semantic_gate",
            _select_params_by_keywords(
                raw_model,
                (
                    "pnurl.semantic_channel_gate",
                    "semantic_channel_gate",
                    "channel_gate",
                    "semantic_gate",
                    "residual_gate",
                ),
            ),
            base_lr * 0.5,
        )
        add_named_params("pnurl", module_named_params(getattr(raw_model, "pnurl", None)), base_lr * 0.3)
        add_named_params("prompt_learner", module_named_params(getattr(raw_model, "prompt_learner", None)), base_lr * 0.5)
        add_named_params("prompt_generator", module_named_params(getattr(raw_model, "prompt_generator", None)), base_lr * 0.1)

        decoder_new, decoder_old = _split_decoder_params(getattr(raw_model, "mask_decoder", None))
        add_named_params("mask_decoder_new_semantic", decoder_new, base_lr * 0.3)
        add_named_params("mask_decoder_pretrained", decoder_old, base_lr * 0.05)

        add_named_params("basic_hv_head", module_named_params(getattr(raw_model, "basic_hv_head", None)), base_lr * 0.05)
        add_named_params("global_asr_upsampler", module_named_params(getattr(raw_model, "global_asr_upsampler", None)), base_lr * 0.05)
        add_named_params("cnn_stage0", module_named_params(getattr(raw_model, "cnn_stage0", None)), base_lr * 0.02)
        add_named_params("cnn_stage1", module_named_params(getattr(raw_model, "cnn_stage1", None)), base_lr * 0.02)
        add_named_params("cnn_stage2", module_named_params(getattr(raw_model, "cnn_stage2", None)), base_lr * 0.02)

        image_encoder = getattr(raw_model, "image_encoder", None)
        adapter_params = []
        if image_encoder is not None:
            adapter_params = [
                (n, p)
                for n, p in image_encoder.named_parameters()
                if ("Adapter" in n or "adapter" in n) and p.requires_grad
            ]
        add_named_params("image_encoder_adapter", adapter_params, base_lr * 0.01)

    if len(param_groups) == 0:
        raise RuntimeError(
            f"No trainable parameters were collected for stage={stage}. "
            "Check use_pnurl/use_coop/use_asr and module names."
        )

    if rank == 0 and logger is not None:
        logger.info("Optimizer parameter groups:")
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"  - {group.get('name', 'unnamed')}: lr={group['lr']:.3e}, params={n_params:,}")

    return optim.AdamW(param_groups, lr=base_lr, weight_decay=args.weight_decay)


def _select_params_by_keywords(module: nn.Module, keywords: Tuple[str, ...]):
    selected = []
    for name, p in module.named_parameters():
        if any(key in name for key in keywords):
            selected.append((name, p))
    return selected


def _split_decoder_params(mask_decoder: Optional[nn.Module]):
    if mask_decoder is None:
        return [], []

    decoder_new_keywords = (
        "low_freq_modulator",
        "high_freq_modulator",
        "high_freq_prompt_encoder",
        "cnn_residual_fusion",
        "low_freq_residual_scale",
        "high_freq_residual_scale",
        "residual_scale",
        "attr_modulator",
        "morphology_modulator",
        "morph_encoder",
        "asr_upscale",
    )

    new_params, old_params = [], []
    for name, p in mask_decoder.named_parameters():
        if any(key in name for key in decoder_new_keywords):
            new_params.append((name, p))
        else:
            old_params.append((name, p))
    return new_params, old_params


# ==================================================================================================
# 4. Loss helpers and diagnostics
# ==================================================================================================
def _zero_like_loss(device):
    return torch.tensor(0.0, device=device)


def _get_best_mask_and_iou(out: Dict, labels_i: torch.Tensor):
    iou_preds = out["iou_predictions"]
    if iou_preds.ndim == 2:
        iou_preds = iou_preds.squeeze(0)
    best_idx = torch.argmax(iou_preds).item()

    if out["masks"].ndim == 3:
        pred_mask = out["masks"][best_idx, :, :]
    else:
        pred_mask = out["masks"][0, best_idx]
    pred_iou = iou_preds[best_idx]

    gt_mask = labels_i.squeeze(0).float()
    gt_mask = (gt_mask > 0).float()
    if pred_mask.shape != gt_mask.shape:
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0),
            size=pred_mask.shape,
            mode="nearest",
        ).squeeze()
    return pred_mask, pred_iou, gt_mask


def _compute_mask_loss(criterion, pred_mask, pred_iou, gt_mask):
    loss_m, _ = criterion(
        pred_mask.unsqueeze(0).unsqueeze(0),
        gt_mask.unsqueeze(0).unsqueeze(0),
        pred_iou.unsqueeze(0),
    )
    return loss_m


def _compute_heatmap_loss(out: Dict, labels_i: torch.Tensor):
    pred_heatmap = out.get("heatmap_logits", None)
    if pred_heatmap is None:
        return _zero_like_loss(labels_i.device)

    with torch.no_grad():
        target_mask = labels_i.float().unsqueeze(0)
        gt_nuclei = F.interpolate(target_mask, size=pred_heatmap.shape[-2:], mode="nearest").squeeze(0)
        gt_nuclei[gt_nuclei == 255] = 0
    return point_guidance_loss(pred_heatmap, gt_nuclei.unsqueeze(0))


def _compute_hv_loss(out: Dict, labels_i: torch.Tensor, batched_input: Dict, sample_index: int):
    pred_hv = out.get("hv_logits", None)
    if pred_hv is None:
        return _zero_like_loss(labels_i.device)

    if pred_hv.dim() == 3:
        pred_hv = pred_hv.unsqueeze(0)
    pred_hv = torch.tanh(pred_hv)

    with torch.no_grad():
        inst_batch = batched_input.get("label_inst", None)
        gt_hv_map_batch = batched_input.get("gt_hv_map", None)
        gt_hv = None
        focus = None

        if gt_hv_map_batch is not None:
            gt_hv_full = gt_hv_map_batch[sample_index].to(pred_hv.device)
            if gt_hv_full.dim() == 3:
                gt_hv_full = gt_hv_full.unsqueeze(0)
            gt_hv = F.interpolate(gt_hv_full.float(), size=pred_hv.shape[-2:], mode="nearest")
        elif inst_batch is not None:
            inst_map = inst_batch[sample_index].float()
            if inst_map.dim() == 2:
                inst_map = inst_map.unsqueeze(0).unsqueeze(0)
            elif inst_map.dim() == 3:
                inst_map = inst_map.unsqueeze(0)
            inst_map_resized = F.interpolate(inst_map, size=pred_hv.shape[-2:], mode="nearest").squeeze().long()
            inst_map_resized[inst_map_resized == 255] = 0
            gt_hv = generate_hv_map_from_inst(inst_map_resized).unsqueeze(0)

        if gt_hv is not None:
            if inst_batch is not None:
                focus_full = (inst_batch[sample_index].squeeze(0) > 0).float().unsqueeze(0).unsqueeze(0)
            else:
                focus_full = (labels_i.squeeze(0) > 0).float().unsqueeze(0).unsqueeze(0)
            focus = F.interpolate(focus_full, size=pred_hv.shape[-2:], mode="nearest").squeeze(1)

    if gt_hv is None or focus is None:
        return _zero_like_loss(pred_hv.device)

    focus_exp = focus.unsqueeze(1)
    mse_map = F.mse_loss(pred_hv.float(), gt_hv.float(), reduction="none")
    loss_hv_mse = (mse_map * focus_exp).sum() / (focus_exp.sum() * pred_hv.shape[1] + 1e-8)
    loss_hv_grad = msge_loss(gt_hv, pred_hv, focus)
    return loss_hv_mse + 2.0 * loss_hv_grad


def _compute_density_loss(out: Dict, labels_i: torch.Tensor, pred_mask: torch.Tensor, stage: str, epoch: int):
    pred_density = out.get("density_map", None)
    if pred_density is None:
        return _zero_like_loss(pred_mask.device)

    if stage == "pnurl_warmup":
        density_reference = (labels_i.float() > 0).float()
        if density_reference.dim() == 2:
            density_reference = density_reference.unsqueeze(0).unsqueeze(0)
        elif density_reference.dim() == 3:
            density_reference = density_reference.unsqueeze(0)
        enable_iou = False
    else:
        density_reference = pred_mask.detach().unsqueeze(0).unsqueeze(0)
        enable_iou = epoch > 20

    loss_d, _, _ = density_map_loss(
        pred_density_map=pred_density,
        gt_mask=labels_i.float().unsqueeze(0),
        pred_mask=density_reference,
        mse_weight=1.0,
        iou_weight=0.5,
        enable_iou=enable_iou,
    )
    return loss_d


def _compute_semantic_stability_loss(out: Dict, args, device: torch.device):
    reg_loss = out.get("semantic_delta_reg_loss", None)
    if reg_loss is None:
        reg_loss = torch.tensor(0.0, device=device)
    elif not torch.is_tensor(reg_loss):
        reg_loss = torch.tensor(float(reg_loss), device=device)
    else:
        reg_loss = reg_loss.to(device=device).float().mean()

    injection_ratio = out.get("injection_ratio", None)
    if injection_ratio is None:
        injection_penalty = torch.tensor(0.0, device=device)
    elif not torch.is_tensor(injection_ratio):
        ratio_tensor = torch.tensor(float(injection_ratio), device=device)
        injection_penalty = F.relu(ratio_tensor - args.max_injection_ratio).pow(2)
    else:
        ratio_tensor = injection_ratio.to(device=device).float().mean()
        injection_penalty = F.relu(ratio_tensor - args.max_injection_ratio).pow(2)

    stability_loss = (
        args.semantic_delta_reg_weight * reg_loss
        + args.injection_ratio_weight * injection_penalty
    )
    return stability_loss, reg_loss.detach(), injection_penalty.detach()


def _to_float_or_nan(value):
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except Exception:
        return float("nan")


def _norm_or_nan(value):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return float("nan")
    return float(value.detach().float().norm(p=2).cpu().item())


def collect_semantic_diagnostics(outputs: List[Dict]):
    gate_vals = []
    delta_norms = []
    base_norms = []
    injected_norms = []
    injection_ratios = []
    delta_reg_losses = []
    delta_ratios = []
    raw_norms = []
    direction_norms = []

    for out in outputs:
        gate = None
        for key in ("semantic_channel_gate", "semantic_gate", "channel_gate", "gate"):
            if key in out and out[key] is not None:
                gate = out[key]
                break
        if torch.is_tensor(gate):
            gate_vals.append(gate.detach().float().reshape(-1))
        elif gate is not None:
            gate_vals.append(torch.as_tensor(gate).float().reshape(-1))

        if "semantic_delta_norm" in out:
            delta_norms.append(_to_float_or_nan(out.get("semantic_delta_norm")))
        else:
            delta_norms.append(_norm_or_nan(out.get("semantic_delta", None)))

        if "base_feat_norm" in out:
            base_norms.append(_to_float_or_nan(out.get("base_feat_norm")))
        else:
            base_feat = out.get("base_feat", None)
            if base_feat is None:
                base_feat = out.get("image_embeddings", None)
            base_norms.append(_norm_or_nan(base_feat))

        if "injected_delta_norm" in out:
            injected_norms.append(_to_float_or_nan(out.get("injected_delta_norm")))
        else:
            injected_norms.append(_norm_or_nan(out.get("injected_delta", None)))

        if "injection_ratio" in out:
            injection_ratios.append(_to_float_or_nan(out.get("injection_ratio")))

        if "semantic_delta_reg_loss" in out:
            delta_reg_losses.append(_to_float_or_nan(out.get("semantic_delta_reg_loss")))

        if "semantic_delta_ratio" in out:
            delta_ratios.append(_to_float_or_nan(out.get("semantic_delta_ratio")))

        if "semantic_delta_raw_norm" in out:
            raw_norms.append(_to_float_or_nan(out.get("semantic_delta_raw_norm")))

        if "semantic_delta_direction_norm" in out:
            direction_norms.append(_to_float_or_nan(out.get("semantic_delta_direction_norm")))

    if gate_vals:
        gate_all = torch.cat(gate_vals)
        gate_mean = float(gate_all.mean().cpu().item())
        gate_min = float(gate_all.min().cpu().item())
        gate_max = float(gate_all.max().cpu().item())
    else:
        gate_mean = gate_min = gate_max = float("nan")

    def finite_mean(values):
        values = [x for x in values if not np.isnan(x)]
        return float(np.mean(values)) if values else float("nan")

    return {
        "semantic_channel_gate_mean": gate_mean,
        "semantic_channel_gate_min": gate_min,
        "semantic_channel_gate_max": gate_max,
        "semantic_delta_norm": finite_mean(delta_norms),
        "base_feat_norm": finite_mean(base_norms),
        "injected_delta_norm": finite_mean(injected_norms),
        "injection_ratio": finite_mean(injection_ratios),
        "semantic_delta_reg_loss": finite_mean(delta_reg_losses),
        "semantic_delta_ratio": finite_mean(delta_ratios),
        "semantic_delta_raw_norm": finite_mean(raw_norms),
        "semantic_delta_direction_norm": finite_mean(direction_norms),
    }


def _write_scalar_if_finite(writer, tag: str, value: float, step: int):
    if value is None:
        return
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return
    writer.add_scalar(tag, value, step)


def _autocast_context(args):
    if not torch.cuda.is_available():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    return autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16)


# ==================================================================================================
# 5. Training logic
# ==================================================================================================
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, writer, rank):
    model.train()
    stage = args.phase

    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1} Train [{stage}]")
    else:
        pbar = train_loader

    meters = {
        "total": [],
        "mask": [],
        "heatmap": [],
        "hv": [],
        "pnurl": [],
        "density": [],
        "semantic_channel_gate_mean": [],
        "semantic_channel_gate_min": [],
        "semantic_channel_gate_max": [],
        "semantic_delta_norm": [],
        "base_feat_norm": [],
        "injected_delta_norm": [],
        "injection_ratio": [],
        "semantic_delta_reg_loss": [],
        "semantic_delta_ratio": [],
        "semantic_delta_raw_norm": [],
        "semantic_delta_direction_norm": [],
        "semantic_stability": [],
        "injection_penalty": [],
    }

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batched_input in enumerate(pbar):
        batched_input = to_device(batched_input, args.device)
        images = batched_input["image"]
        labels = batched_input["label"]

        if images.shape[-1] != args.image_size:
            images = F.interpolate(images, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            labels = F.interpolate(labels.float(), size=(args.image_size, args.image_size), mode="nearest").long()

        organ_ids = batched_input.get("organ_id", None)
        attr_labels = batched_input.get("attr_labels", None)
        dynamic_text = batched_input.get("text_prompt", ["Cell nuclei"] * len(images))
        dynamic_attr_text = batched_input.get("attribute_text", ["Cell nuclei"] * len(images))

        if getattr(args, "asr_regression", False):
            dynamic_text = ["Cell nuclei"] * len(images)
            dynamic_attr_text = ["Cell nuclei"] * len(images)
            attr_labels = None

        model_input = []
        for i in range(len(images)):
            curr_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_id = val.item() if isinstance(val, torch.Tensor) else val
            model_input.append(
                {
                    "image": images[i],
                    "original_size": (args.image_size, args.image_size),
                    "organ_id": curr_id,
                    "attribute_text": dynamic_attr_text[i],
                    "text_prompt": dynamic_text[i],
                    "attr_labels": attr_labels[i] if attr_labels is not None else None,
                }
            )

        with _autocast_context(args):
            outputs = model(model_input, multimask_output=True)

            loss_batch = torch.tensor(0.0, device=args.device)
            accum = {
                "mask": 0.0,
                "heatmap": 0.0,
                "hv": 0.0,
                "pnurl": 0.0,
                "density": 0.0,
                "semantic_stability": 0.0,
                "injection_penalty": 0.0,
            }

            for i, out in enumerate(outputs):
                pred_mask, pred_iou, gt_mask = _get_best_mask_and_iou(out, labels[i])

                loss_m = _compute_mask_loss(criterion, pred_mask, pred_iou, gt_mask)
                loss_h = _compute_heatmap_loss(out, labels[i])
                loss_hv = _compute_hv_loss(out, labels[i], batched_input, i)
                loss_pnurl = out.get("pnurl_loss", _zero_like_loss(args.device))
                loss_d = _compute_density_loss(out, labels[i], pred_mask, stage=stage, epoch=epoch)
                loss_semantic_stability, loss_delta_reg, loss_injection_penalty = _compute_semantic_stability_loss(
                    out,
                    args=args,
                    device=args.device,
                )

                if stage == "vision":
                    if (
                        getattr(args, "asr_regression", False)
                        and getattr(args, "asr_regression_stage", "finetune_decoder") == "freeze_decoder"
                    ):
                        loss_i = args.heatmap_weight * loss_h + args.hv_weight * loss_hv
                    else:
                        loss_i = (
                            args.mask_weight * loss_m
                            + args.heatmap_weight * loss_h
                            + args.hv_weight * loss_hv
                        )
                elif stage == "pnurl_warmup":
                    loss_i = (
                        args.pnurl_weight * loss_pnurl
                        + args.density_map_weight * loss_d
                    )
                else:
                    loss_i = (
                        args.mask_weight * loss_m
                        + args.heatmap_weight * loss_h
                        + args.hv_weight * loss_hv
                        + args.pnurl_weight * loss_pnurl
                        + args.density_map_weight * loss_d
                        + loss_semantic_stability
                    )

                loss_batch = loss_batch + loss_i
                accum["mask"] += float(loss_m.detach().item())
                accum["heatmap"] += float(loss_h.detach().item())
                accum["hv"] += float(loss_hv.detach().item())
                accum["pnurl"] += float(loss_pnurl.detach().item()) if torch.is_tensor(loss_pnurl) else float(loss_pnurl)
                accum["density"] += float(loss_d.detach().item())
                accum["semantic_stability"] += float(loss_semantic_stability.detach().item())
                accum["injection_penalty"] += float(loss_injection_penalty.detach().item())

            final_loss = loss_batch / max(len(images), 1)
            final_loss = final_loss / args.accumulation_steps

        is_accumulating = (batch_idx + 1) % args.accumulation_steps != 0 and (batch_idx + 1) != len(train_loader)

        if scaler:
            scaler.scale(final_loss).backward()
        else:
            final_loss.backward()

        if not is_accumulating:
            trainable_params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss_val = final_loss.detach().item() * args.accumulation_steps
        diag = collect_semantic_diagnostics(outputs)

        n = max(len(images), 1)
        meters["total"].append(current_loss_val)

        for key in ["mask", "heatmap", "hv", "pnurl", "density", "semantic_stability", "injection_penalty"]:
            meters[key].append(accum[key] / n)

        for key, val in diag.items():
            meters[key].append(val)

        if rank == 0 and writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Train/Loss_Total", current_loss_val, global_step)
            writer.add_scalar("Train/Loss_Mask", meters["mask"][-1], global_step)
            writer.add_scalar("Train/Loss_Heatmap", meters["heatmap"][-1], global_step)
            writer.add_scalar("Train/Loss_HV", meters["hv"][-1], global_step)
            writer.add_scalar("Train/Loss_PNuRL", meters["pnurl"][-1], global_step)
            writer.add_scalar("Train/Loss_Density", meters["density"][-1], global_step)
            writer.add_scalar("Train/Loss_Semantic_Stability", meters["semantic_stability"][-1], global_step)
            writer.add_scalar("Train/Loss_Injection_Penalty", meters["injection_penalty"][-1], global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)

            for name in [
                "semantic_channel_gate_mean",
                "semantic_channel_gate_min",
                "semantic_channel_gate_max",
                "semantic_delta_norm",
                "base_feat_norm",
                "injected_delta_norm",
                "injection_ratio",
                "semantic_delta_reg_loss",
                "semantic_delta_ratio",
                "semantic_delta_raw_norm",
                "semantic_delta_direction_norm",
            ]:
                _write_scalar_if_finite(writer, f"Diagnostics/{name}", diag[name], global_step)

        if rank == 0:
            pbar.set_postfix(
                L=f"{current_loss_val:.3f}",
                M=f"{meters['mask'][-1]:.3f}",
                H=f"{meters['heatmap'][-1]:.3f}",
                HV=f"{meters['hv'][-1]:.3f}",
                P=f"{meters['pnurl'][-1]:.3f}",
                D=f"{meters['density'][-1]:.3f}",
                S=f"{meters['semantic_stability'][-1]:.4f}",
                G=f"{diag['semantic_channel_gate_mean']:.4f}" if not np.isnan(diag["semantic_channel_gate_mean"]) else "nan",
                IR=f"{diag['injection_ratio']:.4f}" if not np.isnan(diag["injection_ratio"]) else "nan",
            )

        del batched_input, images, labels, model_input, outputs, final_loss

    return {k: float(np.nanmean(v)) if len(v) > 0 else 0.0 for k, v in meters.items()}


# ==================================================================================================
# 6. Validation logic. Kept as the original fast validation route.
# ==================================================================================================
@torch.no_grad()
def validate_one_epoch(args, model, val_loader, epoch, writer, rank):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()

    val_results = {k: [] for k in args.metrics}
    visualize_done = False

    total_val_batches = len(val_loader)
    limit_batches = int(total_val_batches * 0.4)
    if limit_batches < 1:
        limit_batches = 1

    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Ep {epoch + 1} Val (40%)", total=limit_batches)
    else:
        pbar = val_loader

    eval_model = unwrap_model(model)

    for batch, batched_input in enumerate(pbar):
        if batch >= limit_batches:
            break

        batched_input = to_device(batched_input, args.device)
        images = batched_input["image"]
        inst_labels = batched_input.get("label_inst", batched_input["label"]).cpu().numpy()
        organ_ids = batched_input.get("organ_id", None)

        dynamic_text = batched_input.get("text_prompt", ["Cell nuclei"] * len(images))
        dynamic_attr_text = batched_input.get("attribute_text", dynamic_text)
        attr_labels = batched_input.get("attr_labels", None)

        if getattr(args, "asr_regression", False):
            dynamic_text = ["Cell nuclei"] * len(images)
            dynamic_attr_text = ["Cell nuclei"] * len(images)
            attr_labels = None

        if images.shape[-1] != args.image_size:
            images = F.interpolate(images, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)

        for i in range(len(images)):
            curr_organ_id = 20
            if organ_ids is not None:
                val = organ_ids[i]
                curr_organ_id = val.item() if isinstance(val, torch.Tensor) else val

            with torch.inference_mode(), _autocast_context(args):
                model_input = [
                    {
                        "image": images[i],
                        "original_size": (args.image_size, args.image_size),
                        "text_prompt": dynamic_text[i],
                        "organ_id": curr_organ_id,
                        "attribute_text": dynamic_attr_text[i],
                        "attr_labels": attr_labels[i] if attr_labels is not None else None,
                    }
                ]
                out = eval_model(model_input, multimask_output=True)

            iou_preds = out[0]["iou_predictions"]
            if iou_preds.ndim == 2:
                iou_preds = iou_preds.squeeze(0)
            best_idx = torch.argmax(iou_preds).item()

            pred_logits = out[0]["masks"][0, best_idx]
            prob_map = torch.sigmoid(pred_logits)

            hv_logits = out[0].get("hv_logits", None)
            if hv_logits is not None:
                hv_map = torch.tanh(hv_logits)
                is_expanded = False
                if hv_map.dim() == 3:
                    hv_map = hv_map.unsqueeze(0)
                    is_expanded = True
                target_hw = prob_map.shape[-2:]
                hv_map = F.interpolate(hv_map.float(), size=target_hw, mode="bilinear", align_corners=False)
                if is_expanded:
                    hv_map = hv_map.squeeze(0)
                elif hv_map.dim() == 4 and hv_map.shape[0] == 1:
                    hv_map = hv_map.squeeze(0)
            else:
                target_hw = prob_map.shape[-2:]
                hv_map = torch.zeros((2, target_hw[0], target_hw[1]), device=args.device)

            prob_np = prob_map.float().cpu().numpy()
            hv_np = hv_map.float().cpu().numpy()

            pred_mask = hover_post_process(
                prob_np,
                hv_np,
                prob_thresh=0.45,
                marker_thresh=0.4,
                min_marker_size=10,
            )
            if pred_mask.max() == 0:
                pred_mask = skimage_label(prob_np > 0.5).astype(np.int32)

            gt = inst_labels[i]
            gt = gt[0] if gt.ndim == 3 else gt
            gt_valid = gt.copy()
            gt_valid[gt == 255] = 0

            if pred_mask.shape != gt_valid.shape:
                gt_valid = cv2.resize(
                    gt_valid,
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            res = SegMetrics(pred_mask, gt_valid, args.metrics)
            for k in args.metrics:
                if k in res:
                    val_results[k].append(res[k])

            if rank == 0 and writer is not None and not visualize_done:
                img_vis = images[i].cpu().numpy().transpose(1, 2, 0)
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
                writer.add_image("Val_Viz/Image", torch.tensor(img_vis.transpose(2, 0, 1)), epoch)
                writer.add_image("Val_Viz/GT", torch.tensor(gt_valid * 255).unsqueeze(0).to(torch.uint8), epoch)
                pred_binary = (pred_mask > 0).astype(np.uint8)
                writer.add_image("Val_Viz/Pred", torch.tensor(pred_binary * 255).unsqueeze(0).to(torch.uint8), epoch)
                visualize_done = True

            if rank == 0 and "mAJI" in args.metrics and len(val_results["mAJI"]) > 0:
                pbar.set_postfix(AJI=f"{val_results['mAJI'][-1]:.3f}")

    gc.collect()
    torch.cuda.empty_cache()

    avg_results = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_results.items()}
    if rank == 0 and writer is not None:
        for metric_name, metric_value in avg_results.items():
            writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
        writer.add_scalar(
            "Val/Weighted_Score",
            avg_results.get("mAJI", 0) * 0.6 + avg_results.get("dice", 0) * 0.4,
            epoch,
        )
    return avg_results


# ==================================================================================================
# 7. Checkpoint helpers
# ==================================================================================================
def load_sam_checkpoint_with_asr_mapping(raw_model: nn.Module, args, logger=None, rank: int = 0):
    """
    兼容旧 SAM/SamMed2D checkpoint：
    将原始 output_upscaling 的部分权重映射到 legacy/freqpath ASRBlock 的 structure_upsample。
    build_sam.py 可能已经做过一次 strict=False 加载；这里再执行一次映射，保证 ASR upscaling 能吃到旧权重。
    """
    if not args.sam_checkpoint or not os.path.exists(args.sam_checkpoint):
        return

    if rank == 0 and logger is not None:
        logger.info(f"Loading SAM checkpoint with ASR mapping: {args.sam_checkpoint}")

    try:
        ckpt = torch.load(args.sam_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        state_dict = resize_pos_embed(state_dict, raw_model.state_dict())

        key_mapping = {
            "mask_decoder.output_upscaling.0.weight": "mask_decoder.asr_upscale_1.structure_upsample.0.weight",
            "mask_decoder.output_upscaling.0.bias": "mask_decoder.asr_upscale_1.structure_upsample.0.bias",
            "mask_decoder.output_upscaling.1.weight": "mask_decoder.asr_upscale_1.structure_upsample.1.weight",
            "mask_decoder.output_upscaling.1.bias": "mask_decoder.asr_upscale_1.structure_upsample.1.bias",
            "mask_decoder.output_upscaling.3.weight": "mask_decoder.asr_upscale_2.structure_upsample.0.weight",
            "mask_decoder.output_upscaling.3.bias": "mask_decoder.asr_upscale_2.structure_upsample.0.bias",
        }
        mapped_state_dict = dict(state_dict)
        mapped_count = 0
        model_keys = raw_model.state_dict().keys()
        for old_key, new_key in key_mapping.items():
            if old_key in state_dict and new_key in model_keys:
                mapped_state_dict[new_key] = state_dict[old_key]
                mapped_count += 1

        missing_keys, unexpected_keys = raw_model.load_state_dict(mapped_state_dict, strict=False)
        if rank == 0 and logger is not None:
            logger.info(f"ASRBlock upscaling weights mapped: {mapped_count}/{len(key_mapping)}")
            logger.info(f"SAM checkpoint load: Missing keys={len(missing_keys)} | Unexpected keys={len(unexpected_keys)}")
    except Exception as e:
        if rank == 0 and logger is not None:
            logger.warning(f"SAM checkpoint mapping/loading failed: {e}")


# ==================================================================================================
# 8. Main entry
# ==================================================================================================
def main(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        args.device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if world_size > 1:
        dist.barrier()

    setup_seed(args.seed + rank)

    platform_tb_roots = ["/home/pod", "/root/shared-nvme/tensorboard/logs"]
    platform_tb_root = None
    for tb_path in platform_tb_roots:
        if os.path.exists(tb_path):
            platform_tb_root = tb_path
            break
    if platform_tb_root is None:
        platform_tb_root = os.path.join(args.work_dir, "runs")

    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_log_dir = os.path.join(platform_tb_root, f"{args.run_name}_{timestamp}")
        text_log_dir = os.path.join(args.work_dir, "logs")
        model_save_dir = os.path.join(args.work_dir, "models", args.run_name)
        os.makedirs(run_log_dir, exist_ok=True)
        os.makedirs(text_log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        logger = get_logger(os.path.join(text_log_dir, f"{args.run_name}_{timestamp}.log"))
        logger.info(
            f"[Start] FreqPath-SAM staged training | size={args.image_size} | "
            f"stage={args.phase} | arch={args.architecture_version}"
        )
        logger.info(f"GPUs: {world_size}, Batch/GPU: {args.batch_size}, Resume: {args.resume if args.resume else 'No'}")
        logger.info(
            f"ASR config | use_asr={args.use_asr} | asr_variant={args.asr_variant} | "
            f"asr_regression={args.asr_regression} | asr_regression_stage={args.asr_regression_stage} | "
            f"prompt_mode={args.prompt_mode} | eval_prompt_mode={args.eval_prompt_mode} | "
            f"use_amp={args.use_amp} | cnn_lr_ratio={args.cnn_lr_ratio} | "
            f"prompt_generator_lr_mult={args.prompt_generator_lr_mult} | adapter_lr_ratio={args.adapter_lr_ratio}"
        )
        writer = SummaryWriter(log_dir=run_log_dir, flush_secs=60)
    else:
        logger = None
        writer = None

    try:
        train_dataset = UniversalDataset(
            data_root=args.data_path,
            knowledge_path=args.knowledge_path,
            image_size=args.image_size,
            crop_size=args.crop_size,
            mode="train",
            prompt_mode=args.prompt_mode,
        )
        val_dataset = UniversalDataset(
            data_root=args.data_path,
            knowledge_path=args.knowledge_path,
            image_size=args.image_size,
            crop_size=args.crop_size,
            mode="test",
            prompt_mode=args.eval_prompt_mode,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=True) if world_size > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=4,
            collate_fn=stack_dict_batched,
            pin_memory=True,
            sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=(val_sampler is None),
            num_workers=4,
            collate_fn=stack_dict_batched,
            pin_memory=True,
            sampler=val_sampler,
            persistent_workers=True,
            prefetch_factor=2,
        )

        if rank == 0:
            logger.info(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

        args.checkpoint = args.sam_checkpoint
        built_model = sam_model_registry[args.model_type](args)

        if isinstance(built_model, TextSam):
            raw_model = built_model.to(args.device)
        else:
            raw_model = TextSam(
                image_encoder=built_model.image_encoder,
                prompt_encoder=built_model.prompt_encoder,
                mask_decoder=built_model.mask_decoder,
                clip_model_name=args.clip_model,
                num_organs=args.num_organs,
                num_heads=args.num_heads,
                sg_epsilon=0.05,
                sg_iters=3,
                use_pnurl=True if args.phase in ("pnurl_warmup", "semantic_injection") else args.use_pnurl,
                use_coop=True if args.phase == "semantic_injection" else args.use_coop,
                use_ot=False,
                use_asr=args.use_asr,
                asr_variant=args.asr_variant,
                asr_regression=args.asr_regression,
                max_semantic_gate=args.max_semantic_gate,
                max_delta_ratio=args.max_delta_ratio,
                init_delta_ratio=args.init_delta_ratio,
            ).to(args.device)
            del built_model

        load_sam_checkpoint_with_asr_mapping(raw_model, args=args, logger=logger, rank=rank)

        if args.encoder_adapter:
            for n, p in raw_model.image_encoder.named_parameters():
                if "Adapter" in n and "weight" in n:
                    torch.nn.init.zeros_(p)

        if args.resume and os.path.exists(args.resume):
            if rank == 0:
                logger.info(f"Loading model weights from: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
                state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                state_dict = resize_pos_embed(state_dict, raw_model.state_dict())
                missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=False)
                if rank == 0:
                    logger.info("Model weights loaded. Newly added modules remain initialized by current code.")
                    logger.info(f"Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
                    old_arch = checkpoint.get("architecture_version", "unknown") if isinstance(checkpoint, dict) else "unknown"
                    logger.info(f"Checkpoint architecture_version: {old_arch}")
                    if isinstance(checkpoint, dict):
                        logger.info(
                            f"Checkpoint phase={checkpoint.get('phase', 'unknown')} | "
                            f"asr_variant={checkpoint.get('asr_variant', 'unknown')} | "
                            f"asr_regression={checkpoint.get('asr_regression', 'unknown')}"
                        )
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Resume failed: {e}")

        apply_stage_policy(raw_model, args.phase, args=args, logger=logger, rank=rank)
        optimizer = build_optimizer_by_stage(raw_model, args.phase, args=args, logger=logger, rank=rank)

        if world_size > 1:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=args.ddp_find_unused_parameters,
            )
        else:
            model = raw_model

        criterion = FocalDiceloss_IoULoss(weight=20.0, iou_scale=1.0, ignore_index=255)

        if args.use_amp:
            try:
                scaler = GradScaler("cuda")
            except TypeError:
                scaler = GradScaler()
        else:
            scaler = None

        warmup_epochs = min(args.warmup_epochs, max(args.epochs, 1))
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - warmup_epochs),
                eta_min=getattr(args, "min_lr", 1e-6),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)

        best_aji = 0.0
        best_dice = 0.0

        for epoch in range(args.start_epoch, args.epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            if val_sampler:
                val_sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                args=args,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                epoch=epoch,
                criterion=criterion,
                scaler=scaler,
                writer=writer,
                rank=rank,
            )

            val_res = validate_one_epoch(
                args=args,
                model=model,
                val_loader=val_loader,
                epoch=epoch,
                writer=writer,
                rank=rank,
            )

            if rank == 0:
                dice = float(val_res.get("dice", 0.0))
                aji = float(val_res.get("mAJI", 0.0))
                pq = float(val_res.get("mPQ", 0.0))

                logger.info(
                    f"Ep {epoch + 1}/{args.epochs} | Stage: {args.phase} | "
                    f"ASR:{args.asr_variant}/{args.asr_regression_stage if args.asr_regression else 'normal'} | "
                    f"Loss: {train_stats['total']:.4f} "
                    f"(M:{train_stats['mask']:.3f}, H:{train_stats['heatmap']:.3f}, "
                    f"HV:{train_stats['hv']:.3f}, P:{train_stats['pnurl']:.3f}, D:{train_stats['density']:.3f}, "
                    f"S:{train_stats['semantic_stability']:.4f}) | "
                    f"GateMean:{train_stats['semantic_channel_gate_mean']:.6e} | "
                    f"DeltaNorm:{train_stats['semantic_delta_norm']:.6e} | "
                    f"BaseNorm:{train_stats['base_feat_norm']:.6e} | "
                    f"InjectedNorm:{train_stats['injected_delta_norm']:.6e} | "
                    f"InjRatio:{train_stats['injection_ratio']:.6e} | "
                    f"DeltaRatio:{train_stats['semantic_delta_ratio']:.6e} | "
                    f"Dice:{dice:.4f} | AJI:{aji:.4f} | PQ:{pq:.4f}"
                )

                checkpoint_dict = {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "best_aji": best_aji,
                    "best_dice": best_dice,
                    "phase": args.phase,
                    "architecture_version": args.architecture_version,
                    "asr_variant": args.asr_variant,
                    "asr_regression": args.asr_regression,
                    "asr_regression_stage": args.asr_regression_stage,
                    "args": vars(args),
                }

                latest_model_path = os.path.join(args.work_dir, "models", args.run_name, "latest_model.pth")
                torch.save(checkpoint_dict, latest_model_path)

                if aji > best_aji:
                    best_aji = aji
                    best_dice = max(best_dice, dice)
                    checkpoint_dict["best_aji"] = best_aji
                    checkpoint_dict["best_dice"] = best_dice
                    best_model_path = os.path.join(args.work_dir, "models", args.run_name, "best_model.pth")
                    torch.save(checkpoint_dict, best_model_path)
                    logger.info(f"New Best AJI: {best_aji:.4f}. Model saved.")

            if dist.is_initialized():
                dist.barrier()

            scheduler.step()

        if rank == 0:
            logger.info(f"Training finished. Best AJI: {best_aji:.4f}")
            if writer is not None:
                writer.close()

    except Exception as e:
        if rank == 0 and logger is not None:
            logger.error("\n" + "=" * 50)
            logger.error(f"Fatal training error: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error("=" * 50 + "\n")
        raise e
    finally:
        if rank == 0 and writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
