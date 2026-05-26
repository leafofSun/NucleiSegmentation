# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM builder for NucleiSegmentation.

This version is modified for the ASR regression / FreqPath-SAM workflow:

1. The builder instantiates TextSam from modeling/sam.py instead of the base Sam
   exported by modeling/__init__.py.
2. asr_variant is passed into both MaskDecoder and TextSam, so the command line
   can switch between:
      - legacy  : pure-visual ASR regression baseline
      - freqpath: low-frequency semantic + high-frequency morphology branch
3. asr_regression is passed into TextSam. In TextSam, this mode should force
   PNuRL=False, CoOp=False, OT=False, and base prompt "Cell nuclei".
4. The builder keeps checkpoint loading tolerant because architecture changes
   introduce expected missing/unexpected keys when switching ASR variants.
"""

from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from .modeling.image_encoder import ImageEncoderViT
from .modeling.mask_decoder import MaskDecoder
from .modeling.prompt_encoder import PromptEncoder
from .modeling.sam import TextSam
from .modeling.transformer import TwoWayTransformer


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _get_checkpoint(args: Any, prefer: str = "checkpoint") -> Optional[str]:
    """
    Keep compatibility with different scripts:
    - old scripts may use args.sam_checkpoint
    - current vit_b path often uses args.checkpoint
    """
    if prefer == "sam_checkpoint":
        return _get_arg(args, "sam_checkpoint", _get_arg(args, "checkpoint", None))
    return _get_arg(args, "checkpoint", _get_arg(args, "sam_checkpoint", None))


def _get_asr_variant(args: Any) -> str:
    variant = str(_get_arg(args, "asr_variant", "legacy")).lower().strip()
    if variant not in ("legacy", "freqpath"):
        raise ValueError(f"--asr_variant must be 'legacy' or 'freqpath', got {variant}")
    return variant


def _get_asr_regression(args: Any) -> Optional[bool]:
    """
    Return None when the argument is absent, so TextSam can apply its own default:
    legacy -> True, freqpath -> False.
    """
    return _get_arg(args, "asr_regression", None)


def _get_use_coop(args: Any) -> bool:
    """
    Support both names because different train.py versions use different flags.
    """
    return bool(_get_arg(args, "use_coop", _get_arg(args, "use_coop_prompt", False)))


def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="sam_checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", True),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="sam_checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", False),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=_get_checkpoint(args, prefer="checkpoint"),
        encoder_adapter=args.encoder_adapter,
        use_multimodal_prompt=_get_arg(args, "use_multimodal_prompt", False),
        clip_model_path=_get_arg(args, "clip_model_path", None),
        num_classes=_get_arg(args, "num_classes", 8),
        use_pnurl=bool(_get_arg(args, "use_pnurl", False)),
        use_coop=_get_use_coop(args),
        use_asr=bool(_get_arg(args, "use_asr", True)),
        asr_variant=_get_asr_variant(args),
        asr_regression=_get_asr_regression(args),
        max_semantic_gate=float(_get_arg(args, "max_semantic_gate", 0.10)),
        max_delta_ratio=float(_get_arg(args, "max_delta_ratio", 0.10)),
        init_delta_ratio=float(_get_arg(args, "init_delta_ratio", 0.02)),
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes,
    image_size: int,
    checkpoint: Optional[str],
    encoder_adapter: bool,
    use_multimodal_prompt: bool = True,
    clip_model_path: Optional[str] = None,
    num_classes: int = 8,
    use_pnurl: bool = False,
    use_coop: bool = False,
    use_asr: bool = True,
    asr_variant: str = "legacy",
    asr_regression: Optional[bool] = None,
    max_semantic_gate: float = 0.10,
    max_delta_ratio: float = 0.10,
    init_delta_ratio: float = 0.02,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # PNuRL output is aligned with SAM prompt embedding dimension.
    text_embed_dim = prompt_embed_dim if use_pnurl else None

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_asr=use_asr,
        asr_variant=asr_variant,
    )

    sam = TextSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train=encoder_adapter,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            use_multimodal_prompt=use_multimodal_prompt,
            clip_model_path=clip_model_path,
            num_classes=num_classes,
            text_embed_dim=text_embed_dim,
        ),
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        embed_dim=prompt_embed_dim,
        num_organs=_get_num_organs(num_classes),
        use_pnurl=use_pnurl,
        use_coop=use_coop,
        use_ot=False,
        use_asr=use_asr,
        asr_variant=asr_variant,
        asr_regression=asr_regression,
        max_semantic_gate=max_semantic_gate,
        max_delta_ratio=max_delta_ratio,
        init_delta_ratio=init_delta_ratio,
    )

    print(
        f"[build_sam] image_size={image_size} | use_asr={use_asr} | "
        f"asr_variant={asr_variant} | asr_regression={asr_regression} | "
        f"use_pnurl={use_pnurl} | use_coop={use_coop}"
    )

    if checkpoint is not None:
        _load_checkpoint_into_model(
            sam=sam,
            checkpoint=checkpoint,
            image_size=image_size,
            vit_patch_size=vit_patch_size,
            encoder_adapter=encoder_adapter,
        )

    return sam


def _get_num_organs(num_classes: int) -> int:
    """
    TextSam's DualPromptLearner defaults to 21 organs.
    num_classes in PromptEncoder is not always organ count, so keep the previous
    TextSam default unless a larger value is explicitly requested.
    """
    return max(int(num_classes), 21)


def _load_checkpoint_into_model(
    sam: torch.nn.Module,
    checkpoint: str,
    image_size: int,
    vit_patch_size: int,
    encoder_adapter: bool,
) -> None:
    with open(checkpoint, "rb") as f:
        # Training checkpoints can include optimizer / scheduler objects.
        state_dict = torch.load(f, map_location="cpu", weights_only=False)

    actual_state_dict = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict

    try:
        missing, unexpected = sam.load_state_dict(actual_state_dict, strict=False)
        print(f"*******load {checkpoint}")
        print(f"[checkpoint] missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")
        if len(missing) > 0:
            print(f"[checkpoint] first missing keys: {missing[:20]}")
        if len(unexpected) > 0:
            print(f"[checkpoint] first unexpected keys: {unexpected[:20]}")
        return
    except RuntimeError as exc:
        print(f"[checkpoint] direct non-strict load failed: {exc}")
        print("*******interpolate")

    new_state_dict = load_from(sam, actual_state_dict, image_size, vit_patch_size)
    missing, unexpected = sam.load_state_dict(new_state_dict, strict=False)
    print(f"*******load {checkpoint}")
    print(f"[checkpoint/interpolate] missing_keys={len(missing)} | unexpected_keys={len(unexpected)}")
    print(f"[checkpoint] encoder_adapter={encoder_adapter}")


def load_from(sam, state_dicts: Dict[str, torch.Tensor], image_size: int, vit_patch_size: int):
    sam_dict = sam.state_dict()
    except_keys = ["mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head"]

    new_state_dict = {
        k: v
        for k, v in state_dicts.items()
        if k in sam_dict.keys()
        and except_keys[0] not in k
        and except_keys[1] not in k
        and except_keys[2] not in k
    }

    if "image_encoder.pos_embed" not in new_state_dict:
        print(
            "Warning: 'image_encoder.pos_embed' not found in checkpoint. "
            f"Available keys: {list(new_state_dict.keys())[:10]}..."
        )
        sam_dict.update(new_state_dict)
        return sam_dict

    pos_embed = new_state_dict["image_encoder.pos_embed"]
    token_size = int(image_size // vit_patch_size)

    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            (token_size, token_size),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        new_state_dict["image_encoder.pos_embed"] = pos_embed

        rel_pos_keys = [k for k in sam_dict.keys() if "rel_pos" in k]
        global_rel_pos_keys = [
            k
            for k in rel_pos_keys
            if "2" in k
            or "5" in k
            or "7" in k
            or "8" in k
            or "11" in k
            or "13" in k
            or "15" in k
            or "23" in k
            or "31" in k
        ]

        for k in global_rel_pos_keys:
            if k not in new_state_dict:
                continue

            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)

            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(
                    rel_pos_params,
                    (h_check, w_check),
                    mode="bilinear",
                    align_corners=False,
                )

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict