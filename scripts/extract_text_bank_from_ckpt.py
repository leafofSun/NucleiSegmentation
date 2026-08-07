#!/usr/bin/env python3
"""
从 Exp6 checkpoint 中提取 text_bank（CONCH 编码的 prompt embeddings）并保存。

该脚本需要一次性的 CONCH 可访问环境（有 HuggingFace 缓存或网络）。
输出是与原 checkpoint 路径相同的 .pth 文件，但 state_dict 中会新增两个 buffer：
  - _structure_text_bank_buffer: [5, 3, 512]
  - _boundary_text_bank_buffer: [4, 3, 512]

用法:
  # 在有 HuggingFace 缓存/网络的机器上运行
  python NuSeg/scripts/extract_text_bank_from_ckpt.py

  # 或指定 CONCH 缓存路径
  HF_HOME=/hy-tmp/NuSeg/hf_cache python NuSeg/scripts/extract_text_bank_from_ckpt.py
"""
import os
import sys
import torch
import argparse

# ── 添加项目根目录到 path ──
PROJECT_ROOT = "/hy-tmp/NuSeg"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CKPT_PATH = "/hy-tmp/NuSeg/workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth"
OUTPUT_PATH = CKPT_PATH  # 直接修改原文件（会先备份）

# ── 导入所需模块 ──
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam

def extract_text_bank(ckpt_path: str, output_path: str, force: bool = False):
    """
    加载 checkpoint → 构建 TextSam → 加载权重 → 创建 text_bank → 保存。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── 1. 加载 checkpoint ──
    print(f"[EXTRACT] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    args_dict = ckpt.get("args", None)

    print(f"[EXTRACT] state_dict keys: {len(state_dict)}")
    print(f"[EXTRACT] args present: {args_dict is not None}")

    # ── 2. 检查是否已有 text_bank buffer ──
    has_struct_buf = any("_structure_text_bank_buffer" in k for k in state_dict.keys())
    has_bound_buf = any("_boundary_text_bank_buffer" in k for k in state_dict.keys())
    if has_struct_buf and has_bound_buf and not force:
        print(f"[EXTRACT] ✅ Text bank buffers already exist in checkpoint. Skipping.")
        return

    # ── 3. 构建 TextSam（需 CONCH 可访问）──
    # 使用 checkpoint args 中的配置参数
    if args_dict is not None:
        # 构建一个简单的 namespace
        class Args:
            pass
        args = Args()
        for k, v in args_dict.items():
            setattr(args, k, v)
        model_type = getattr(args, "model_type", "vit_b")
    else:
        args = None
        model_type = "vit_b"

    print(f"[EXTRACT] Building TextSam with model_type={model_type}")

    # 构建基础 SAM（清除 checkpoint 避免加载不存在的 base ckpt）
    if args is not None:
        _saved_ckpt = getattr(args, "checkpoint", None)
        args.checkpoint = None
    vanilla_sam = sam_model_registry[model_type](args)
    if args is not None:
        args.checkpoint = _saved_ckpt

    # ── 从 args 中提取 TextSam 参数 ──
    _get = lambda k, d: getattr(args, k, d) if args is not None else d

    text_sam_kwargs = {
        "image_encoder": vanilla_sam.image_encoder,
        "prompt_encoder": vanilla_sam.prompt_encoder,
        "mask_decoder": vanilla_sam.mask_decoder,
        "clip_model_name": _get("clip_model", "ViT-B/16"),
        "num_organs": _get("num_organs", 21),
        "num_heads": _get("num_heads", 8),
        "sg_epsilon": 0.05,
        "sg_iters": 3,
        "use_pnurl": _get("use_pnurl", True),
        "use_coop": _get("use_coop", True),
        "use_ot": False,
        "use_asr": _get("use_asr", True),
        "asr_variant": _get("asr_variant", "legacy"),
        "asr_regression": _get("asr_regression", None),
        "max_semantic_gate": _get("max_semantic_gate", 0.10),
        "max_delta_ratio": _get("max_delta_ratio", 0.10),
        "init_delta_ratio": _get("init_delta_ratio", 0.02),
        "semantic_gate_bias_init": _get("semantic_gate_bias_init", None),
        "semantic_injection_scale": float(_get("semantic_injection_scale", 1.0)),
        # Structure & Boundary
        "enable_structure_boundary_attr_heads": bool(_get("enable_structure_boundary_attr_heads", False)),
        "sb_guidance_mode": str(_get("sb_guidance_mode", "none")),
        "sb_guidance_weight": float(_get("sb_guidance_weight", 0.05)),
        "sb_guidance_routing": str(_get("sb_guidance_routing", "structure_low_boundary_high")),
        # MultiLevel (Phase B)
        "enable_multilevel_attr_heads": bool(_get("enable_multilevel_attr_heads", False)),
        # Phase C
        "enable_attr_text_alignment": bool(_get("enable_attr_text_alignment", False)),
        "attr_text_alignment_visual_dim": int(_get("attr_text_alignment_visual_dim", 256)),
        "attr_text_alignment_text_dim": int(_get("attr_text_alignment_text_dim", 512)),
        "debug_phase_c_audit": bool(_get("debug_phase_c_audit", False)),
        "debug_instance_align_audit": bool(_get("debug_instance_align_audit", False)),
        # Numeric Attr FreqPath
        "enable_numeric_attr_freqpath_guidance": bool(_get("enable_numeric_attr_freqpath_guidance", False)),
        "numeric_attr_freqpath_hidden_dim": int(_get("numeric_attr_freqpath_hidden_dim", 128)),
        "numeric_attr_freqpath_init": str(_get("numeric_attr_freqpath_init", "zero")),
        # PromptNu-lite v2
        "enable_promptnu_lite_align": bool(_get("enable_promptnu_lite_align", False)),
        "promptnu_lite_target": str(_get("promptnu_lite_target", "semantic_delta")),
        "promptnu_lite_struct_weight": float(_get("promptnu_lite_struct_weight", 0.0)),
        "promptnu_lite_boundary_weight": float(_get("promptnu_lite_boundary_weight", 0.0)),
        "promptnu_lite_instance_weight": float(_get("promptnu_lite_instance_weight", 0.0)),
        "promptnu_lite_detach_text": bool(_get("promptnu_lite_detach_text", True)),
        "promptnu_lite_detach_visual": bool(_get("promptnu_lite_detach_visual", False)),
        "promptnu_lite_proj_lr_mult": float(_get("promptnu_lite_proj_lr_mult", 0.5)),
        "promptnu_lite_pool_mode": str(_get("promptnu_lite_pool_mode", "gap")),
        # PromptNu-guided v3
        "enable_promptnu_guided_v3": bool(_get("enable_promptnu_guided_v3", False)),
        "promptnu_guided_v3_struct_weight": float(_get("promptnu_guided_v3_struct_weight", 1.0)),
        "promptnu_guided_v3_boundary_weight": float(_get("promptnu_guided_v3_boundary_weight", 1.0)),
        "promptnu_guided_v3_text_weight": float(_get("promptnu_guided_v3_text_weight", 0.01)),
        "promptnu_guided_v3_embed_dim": int(_get("promptnu_guided_v3_embed_dim", 256)),
        "promptnu_guided_v3_hidden_dim": int(_get("promptnu_guided_v3_hidden_dim", 128)),
        "promptnu_guided_v3_vis_proj_dim": int(_get("promptnu_guided_v3_vis_proj_dim", 512)),
        "promptnu_guided_v3_align_loss_weight": float(_get("promptnu_guided_v3_align_loss_weight", 0.1)),
        "ablate_semantic_injection": bool(_get("ablate_semantic_injection", False)),
        "ablate_pred_attr_guidance": bool(_get("ablate_pred_attr_guidance", False)),
        # v3.1: text bank
        "promptnu_guided_v3_use_text_bank": bool(_get("promptnu_guided_v3_use_text_bank", False)),
        "promptnu_guided_v3_use_gt_align_target": bool(_get("promptnu_guided_v3_use_gt_align_target", False)),
        "promptnu_guided_v3_semantic_dim": int(_get("promptnu_guided_v3_semantic_dim", 256)),
        "promptnu_guided_v3_text_dim": int(_get("promptnu_guided_v3_text_dim", 512)),
        "promptnu_guided_v3_strict_audit": bool(_get("promptnu_guided_v3_strict_audit", False)),
        "promptnu_guided_v3_prompt_source": str(_get("promptnu_guided_v3_prompt_source", "pred_attr")),
        # v3.3
        "promptnu_guided_v3_guidance_mode": str(_get("promptnu_guided_v3_guidance_mode", "scale_add")),
        "promptnu_guided_v3_scale_weight": _get("promptnu_guided_v3_scale_weight", None),
        "promptnu_guided_v3_delta_weight": float(_get("promptnu_guided_v3_delta_weight", 0.001)),
        "promptnu_guided_v3_delta_init_std": float(_get("promptnu_guided_v3_delta_init_std", 1e-5)),
        "promptnu_guided_v3_max_guided_delta_ratio": float(_get("promptnu_guided_v3_max_guided_delta_ratio", 0.0)),
        "promptnu_guided_v3_align_eps": float(_get("promptnu_guided_v3_align_eps", 1e-8)),
        "promptnu_guided_v3_cosine_eps": float(_get("promptnu_guided_v3_cosine_eps", 1e-8)),
        "promptnu_guided_v3_min_align_delta_norm": float(_get("promptnu_guided_v3_min_align_delta_norm", 0.0)),
        "promptnu_guided_v3_align_low_norm_mode": str(_get("promptnu_guided_v3_align_low_norm_mode", "detach_guided")),
        "promptnu_guided_v3_injection_ablation": str(_get("promptnu_guided_v3_injection_ablation", "default")),
        "promptnu_guided_v3_post_gate_alpha": float(_get("promptnu_guided_v3_post_gate_alpha", 1.0)),
        # PNuDP
        "enable_pnudp_diag": bool(_get("enable_pnudp_diag", False)),
        "pnudp_fusion_mode": str(_get("pnudp_fusion_mode", "none")),
        "pnudp_scale": float(_get("pnudp_scale", 20.0)),
        "enable_pnudp_dense_train": bool(_get("enable_pnudp_dense_train", False)),
        "pnudp_dense_alpha_init": float(_get("pnudp_dense_alpha_init", 0.0)),
        "pnudp_dense_logit_proj_init": str(_get("pnudp_dense_logit_proj_init", "zero")),
        "pnudp_dense_logit_proj_init_std": float(_get("pnudp_dense_logit_proj_init_std", 1.0)),
        "pnudp_dense_apply_in_eval": bool(_get("pnudp_dense_apply_in_eval", False)),
        "pnudp_dense_num_mask_channels": int(_get("pnudp_dense_num_mask_channels", 1)),
        # SB guidance
        "sb_prompt_template_path": os.path.join(PROJECT_ROOT, str(_get("sb_prompt_template_path", "workdir/attr_stats/structure_boundary_prompt_templates.json"))),
        "sb_direct_adapter_hidden_dim": int(_get("sb_direct_adapter_hidden_dim", 64)),
        "sb_conch_freeze": bool(_get("sb_conch_freeze", True)),
        # CONCH
        "enable_conch_text_encoder": True,  # 必须启用才能加载 clip_model
    }

    print(f"[EXTRACT] Building TextSam...")
    model = TextSam(**text_sam_kwargs)

    # ── 4. 加载 checkpoint 权重 ──
    # 跳过不匹配的 key（如 prompt_encoder 的某些权重可能因版本不同而略有差异）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[EXTRACT] Missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
    if unexpected:
        print(f"[EXTRACT] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected)-10} more")

    # ── 5. 确保 CONCH 已加载 ──
    if model.clip_model is None:
        raise RuntimeError(
            "CONCH model is None after checkpoint loading. "
            "This machine may not have HuggingFace CONCH cache. "
            "Try: HF_HOME=/hy-tmp/NuSeg/hf_cache python NuSeg/scripts/extract_text_bank_from_ckpt.py"
        )

    # ── 6. 初始化 CONCH cache 并创建 text_bank ──
    print(f"[EXTRACT] Initializing CONCH SB cache...")
    model.eval()
    device = next(model.clip_model.parameters()).device
    _ = model._init_sb_conch_cache()

    print(f"[EXTRACT] Creating text bank tensors...")
    with torch.no_grad():
        struct_text_bank = model._get_sb_text_bank("structure", device)  # [5, 3, 512]
        bound_text_bank = model._get_sb_text_bank("boundary", device)    # [4, 3, 512]

    print(f"[EXTRACT] structure_text_bank: {tuple(struct_text_bank.shape)}")
    print(f"[EXTRACT] boundary_text_bank:  {tuple(bound_text_bank.shape)}")

    # ── 7. 将 text_bank 添加到 state_dict ──
    state_dict["_structure_text_bank_buffer"] = struct_text_bank.cpu()
    state_dict["_boundary_text_bank_buffer"] = bound_text_bank.cpu()

    # 更新 checkpoint
    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"]["_structure_text_bank_buffer"] = struct_text_bank.cpu()
        ckpt["model_state_dict"]["_boundary_text_bank_buffer"] = bound_text_bank.cpu()
    elif "state_dict" in ckpt:
        ckpt["state_dict"]["_structure_text_bank_buffer"] = struct_text_bank.cpu()
        ckpt["state_dict"]["_boundary_text_bank_buffer"] = bound_text_bank.cpu()
    elif "model" in ckpt:
        ckpt["model"]["_structure_text_bank_buffer"] = struct_text_bank.cpu()
        ckpt["model"]["_boundary_text_bank_buffer"] = bound_text_bank.cpu()

    # ── 8. 备份原文件 ──
    backup_path = output_path + ".bak"
    if not os.path.exists(backup_path):
        print(f"[EXTRACT] Backing up original to: {backup_path}")
        os.rename(output_path, backup_path)
    else:
        print(f"[EXTRACT] Backup already exists: {backup_path}")

    # ── 9. 保存修改后的 checkpoint ──
    print(f"[EXTRACT] Saving modified checkpoint to: {output_path}")
    torch.save(ckpt, output_path)
    print(f"[EXTRACT] ✅ Done! Text bank buffers added to checkpoint.")
    print(f"[EXTRACT]    _structure_text_bank_buffer: {tuple(struct_text_bank.shape)}")
    print(f"[EXTRACT]    _boundary_text_bank_buffer:  {tuple(bound_text_bank.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text_bank from Exp6 checkpoint")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force re-extraction even if buffers already exist")
    args = parser.parse_args()

    extract_text_bank(CKPT_PATH, OUTPUT_PATH, force=args.force)
