#!/usr/bin/env python3
"""
Exp7: 使用 CLIP text encoder 编码 structure/boundary prompt templates，
生成 [5,3,512] 和 [4,3,512] text bank buffer，并注入到 Exp6 checkpoint 中。

与 extract_text_bank_from_ckpt.py 的区别：
  - 使用 CLIP (ViT-B/32) 而非 CONCH 作为 text encoder
  - 不加载 TextSam，不加载 CONCH
  - 支持两种 CLIP 后端：open_clip (HuggingFace) 或本地 OpenAI CLIP
  - 输出格式与 CONCH 版本完全一致（可直接用 --use_checkpoint_text_bank_without_conch 加载）
  - 同时也支持 --clip_text_encoder 模式加载（TextSam 将跳过 CONCH）

用法:
  # 默认使用 CLIP ViT-B/32（自动选择可用后端），注入到 Exp6 checkpoint
  python NuSeg/scripts/extract_clip_text_bank_from_prompts.py

  # 指定 CLIP 模型变体
  python NuSeg/scripts/extract_clip_text_bank_from_prompts.py --clip_model "ViT-B/16"

  # 使用本地 OpenAI CLIP 包（适用于无 HuggingFace 网络的环境）
  python NuSeg/scripts/extract_clip_text_bank_from_prompts.py --backend clip

  # 强制重新生成
  python NuSeg/scripts/extract_clip_text_bank_from_prompts.py --force

  # 从已有 CLIP text bank 的 checkpoint 加载（验证模式）
  python NuSeg/scripts/extract_clip_text_bank_from_prompts.py --verify
"""
import os
import sys
import json
import torch
import argparse
import numpy as np

# ── 添加项目根目录到 path ──
PROJECT_ROOT = "/hy-tmp/NuSeg"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加本地 CLIP 包路径
_LOCAL_CLIP_PATH = os.path.join(PROJECT_ROOT, "CLIP/CLIP-main")
if _LOCAL_CLIP_PATH not in sys.path:
    sys.path.insert(0, _LOCAL_CLIP_PATH)

# ── 默认路径 ──
CKPT_PATH = "/hy-tmp/NuSeg/workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth"
OUTPUT_PATH = CKPT_PATH  # 直接修改原文件（会先备份）
PROMPT_TEMPLATE_PATH = os.path.join(
    PROJECT_ROOT, "workdir/attr_stats/structure_boundary_prompt_templates.json"
)

# ── Attribute order (must match TextSam.STRUCTURE_ATTR_NAMES / BOUNDARY_ATTR_NAMES) ──
STRUCTURE_ATTR_NAMES = (
    "nuclear_density",
    "nuclear_area_fraction",
    "mean_nuclear_size",
    "nuclear_size_heterogeneity",
    "spatial_crowding",
)
BOUNDARY_ATTR_NAMES = (
    "boundary_density",
    "nuclear_irregularity",
    "nuclear_elongation",
    "small_nuclei_ratio",
)
LEVEL_NAMES = ("low", "mid", "high")
NUM_STRUCTURE_ATTRS = len(STRUCTURE_ATTR_NAMES)  # 5
NUM_BOUNDARY_ATTRS = len(BOUNDARY_ATTR_NAMES)    # 4
NUM_LEVELS = len(LEVEL_NAMES)                     # 3
TEXT_DIM = 512  # CLIP ViT-B/32 and ViT-B/16 both output 512-dim

# ── CLIP model name mapping ──
# open_clip uses "ViT-B/32", local CLIP uses "ViT-B/32"
CLIP_MODEL_MAP = {
    "ViT-B/32": "ViT-B/32",
    "ViT-B/16": "ViT-B/16",
}


def load_prompt_templates(template_path: str) -> dict:
    """Load structure/boundary prompt templates from JSON file."""
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    with open(template_path, "r") as f:
        templates = json.load(f)
    return templates


def build_all_prompt_texts(templates: dict):
    """
    构建所有 27 个 prompt 文本。

    Returns:
        structure_keys:   ["nuclear_density_low", ..., "spatial_crowding_high"]  (15 items)
        structure_texts:  [prompt_string, ...]                                    (15 items)
        boundary_texts:   [prompt_string, ...]                                    (12 items)
        boundary_keys:    ["boundary_density_low", ..., "small_nuclei_ratio_high"] (12 items)
    """
    struct_prompts = templates.get("structure_prompts", {})
    bound_prompts = templates.get("boundary_prompts", {})

    structure_keys = []
    structure_texts = []
    for attr_name in STRUCTURE_ATTR_NAMES:
        attr_data = struct_prompts.get(attr_name, {})
        for level_name in LEVEL_NAMES:
            key = f"{attr_name}_{level_name}"
            text = attr_data.get(level_name, "")
            if not text:
                print(f"⚠️  [CLIP] Missing template for {key}, using fallback.")
                text = f"{attr_name} {level_name}"
            structure_keys.append(key)
            structure_texts.append(text)

    boundary_keys = []
    boundary_texts = []
    for attr_name in BOUNDARY_ATTR_NAMES:
        attr_data = bound_prompts.get(attr_name, {})
        for level_name in LEVEL_NAMES:
            key = f"{attr_name}_{level_name}"
            text = attr_data.get(level_name, "")
            if not text:
                print(f"⚠️  [CLIP] Missing template for {key}, using fallback.")
                text = f"{attr_name} {level_name}"
            boundary_keys.append(key)
            boundary_texts.append(text)

    return structure_keys, structure_texts, boundary_texts, boundary_keys


def _load_clip_open_clip(model_name: str, device: torch.device, hf_cache: str = ""):
    """Load CLIP model via open_clip (HuggingFace weights)."""
    import open_clip

    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        os.environ["TORCH_HOME"] = hf_cache

    print(f"[CLIP_EXTRACT] Loading via open_clip: {model_name}")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai",
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    backend_name = f"open_clip/{model_name}"

    return model, tokenizer, backend_name


def _load_clip_local(model_name: str, device: torch.device):
    """Load CLIP model via OpenAI local CLIP package."""
    import clip

    # 本地 CLIP 包使用相同的模型名称
    print(f"[CLIP_EXTRACT] Loading via local CLIP: {model_name}")
    download_root = os.path.join(PROJECT_ROOT, "CLIP/CLIP-main")
    model, _ = clip.load(model_name, device=device, download_root=download_root)
    model.eval()

    # local CLIP tokenizer
    tokenizer = clip.tokenize
    backend_name = f"clip/{model_name}"

    return model, tokenizer, backend_name


def _encode_texts_open_clip(model, tokenizer, texts: list, device: torch.device) -> torch.Tensor:
    """Encode texts using open_clip model."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    return embeddings


def _encode_texts_local_clip(model, tokenizer, texts: list, device: torch.device) -> torch.Tensor:
    """Encode texts using local OpenAI CLIP model."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    return embeddings


def extract_clip_text_bank(
    ckpt_path: str,
    output_path: str,
    clip_model_name: str = "ViT-B/32",
    backend: str = "auto",
    hf_cache: str = "",
    force: bool = False,
    verify: bool = False,
):
    """
    使用 CLIP text encoder 编码 prompts → text bank → 注入 checkpoint。

    Args:
        ckpt_path:   Exp6 checkpoint 路径
        output_path: 输出路径（通常会覆盖原文件，先备份）
        clip_model_name: CLIP 模型名称 ("ViT-B/32" 或 "ViT-B/16")
        backend:     "auto" (自动选择), "open_clip", "clip" (本地 OpenAI CLIP)
        hf_cache:    HuggingFace 缓存路径（仅 open_clip 后端使用）
        force:       强制重新生成
        verify:      仅验证已有 text bank，不重新生成
    """
    # ── 1. 加载 checkpoint ──
    print(f"[CLIP_EXTRACT] Loading checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    print(f"[CLIP_EXTRACT] state_dict keys: {len(state_dict)}")

    # ── 2. 检查是否已有 text_bank buffer ──
    has_struct_buf = any("_structure_text_bank_buffer" in k for k in state_dict.keys())
    has_bound_buf = any("_boundary_text_bank_buffer" in k for k in state_dict.keys())

    if verify:
        if has_struct_buf and has_bound_buf:
            struct_buf = state_dict.get("_structure_text_bank_buffer", state_dict.get(
                next(k for k in state_dict if "_structure_text_bank_buffer" in k)))
            bound_buf = state_dict.get("_boundary_text_bank_buffer", state_dict.get(
                next(k for k in state_dict if "_boundary_text_bank_buffer" in k)))
            print(f"[CLIP_EXTRACT] ✅ Verification: text bank buffers found!")
            print(f"[CLIP_EXTRACT]    _structure_text_bank_buffer: {tuple(struct_buf.shape)}")
            print(f"[CLIP_EXTRACT]    _boundary_text_bank_buffer:  {tuple(bound_buf.shape)}")
            # 检查 norm
            with torch.no_grad():
                s_norm = struct_buf.float().norm(dim=-1).mean().item()
                b_norm = bound_buf.float().norm(dim=-1).mean().item()
            print(f"[CLIP_EXTRACT]    structure norm (mean): {s_norm:.4f}")
            print(f"[CLIP_EXTRACT]    boundary norm (mean):  {b_norm:.4f}")
        else:
            print(f"[CLIP_EXTRACT] ❌ Verification failed: text bank buffers not found.")
        return

    if has_struct_buf and has_bound_buf and not force:
        print(f"[CLIP_EXTRACT] ✅ Text bank buffers already exist in checkpoint. "
              f"Use --force to re-extract, or --verify to inspect.")
        return
    if has_struct_buf and has_bound_buf and force:
        print(f"[CLIP_EXTRACT] Force mode: re-extracting even though buffers exist.")

    # ── 3. 加载 CLIP text encoder ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP_EXTRACT] Using device: {device}")

    # 自动选择可用后端
    if backend == "auto":
        try:
            model, tokenizer, backend_name = _load_clip_open_clip(clip_model_name, device, hf_cache)
            print(f"[CLIP_EXTRACT] ✅ Using open_clip backend")
        except Exception as e:
            print(f"[CLIP_EXTRACT] open_clip failed: {e}")
            try:
                model, tokenizer, backend_name = _load_clip_local(clip_model_name, device)
                print(f"[CLIP_EXTRACT] ✅ Using local CLIP backend (fallback)")
            except Exception as e2:
                raise RuntimeError(
                    f"Both CLIP backends failed. open_clip: {e} | local CLIP: {e2}\n"
                    "Install open_clip: pip install open-clip-torch\n"
                    "Or ensure local CLIP is available at: CLIP/CLIP-main/"
                )
        encode_fn = _encode_texts_open_clip
    elif backend == "open_clip":
        model, tokenizer, backend_name = _load_clip_open_clip(clip_model_name, device, hf_cache)
        encode_fn = _encode_texts_open_clip
    elif backend == "clip":
        model, tokenizer, backend_name = _load_clip_local(clip_model_name, device)
        encode_fn = _encode_texts_local_clip
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'open_clip', or 'clip'.")

    # 验证输出维度
    with torch.no_grad():
        dummy_emb = encode_fn(model, tokenizer, ["test prompt"], device)
        actual_dim = dummy_emb.shape[-1]
        print(f"[CLIP_EXTRACT] CLIP text feature dim: {actual_dim} (backend={backend_name})")
        if actual_dim != TEXT_DIM:
            print(f"[CLIP_EXTRACT] ⚠️  CLIP output dim={actual_dim}, expected={TEXT_DIM}. "
                  f"Will pad/truncate to {TEXT_DIM}.")

    # ── 4. 加载 prompt templates ──
    print(f"[CLIP_EXTRACT] Loading prompt templates: {PROMPT_TEMPLATE_PATH}")
    templates = load_prompt_templates(PROMPT_TEMPLATE_PATH)
    structure_keys, structure_texts, boundary_texts, boundary_keys = build_all_prompt_texts(templates)
    print(f"[CLIP_EXTRACT] Structure prompts: {len(structure_texts)} "
          f"({NUM_STRUCTURE_ATTRS} attrs × {NUM_LEVELS} levels)")
    print(f"[CLIP_EXTRACT] Boundary prompts:  {len(boundary_texts)} "
          f"({NUM_BOUNDARY_ATTRS} attrs × {NUM_LEVELS} levels)")

    # ── 5. 编码所有 prompts ──
    print(f"[CLIP_EXTRACT] Encoding prompts with CLIP text encoder ({backend_name})...")
    all_texts = structure_texts + boundary_texts  # 15 + 12 = 27
    all_keys = structure_keys + boundary_keys

    all_embeddings = encode_fn(model, tokenizer, all_texts, device)  # [27, D]

    # L2 normalize (matching CONCH text bank behavior)
    all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)

    # 确保维度匹配 TEXT_DIM
    if all_embeddings.shape[-1] != TEXT_DIM:
        _current_dim = all_embeddings.shape[-1]
        if _current_dim > TEXT_DIM:
            all_embeddings = all_embeddings[:, :TEXT_DIM]
        else:
            pad = torch.zeros(all_embeddings.shape[0], TEXT_DIM - _current_dim, device=device)
            all_embeddings = torch.cat([all_embeddings, pad], dim=-1)

    print(f"[CLIP_EXTRACT] All embeddings shape: {tuple(all_embeddings.shape)}")

    for i, (key, emb) in enumerate(zip(all_keys, all_embeddings)):
        norm = emb.norm().item()
        print(f"  [{i:02d}] {key:40s} norm={norm:.4f}")

    # ── 6. 重构为 text bank 格式 ──
    # structure_text_bank: [5, 3, 512]
    struct_embeddings = all_embeddings[:len(structure_texts)]  # [15, 512]
    structure_text_bank = struct_embeddings.view(NUM_STRUCTURE_ATTRS, NUM_LEVELS, TEXT_DIM)
    print(f"[CLIP_EXTRACT] structure_text_bank: {tuple(structure_text_bank.shape)}")

    # boundary_text_bank: [4, 3, 512]
    bound_embeddings = all_embeddings[len(structure_texts):]  # [12, 512]
    boundary_text_bank = bound_embeddings.view(NUM_BOUNDARY_ATTRS, NUM_LEVELS, TEXT_DIM)
    print(f"[CLIP_EXTRACT] boundary_text_bank:  {tuple(boundary_text_bank.shape)}")

    # ── 7. 注入到 checkpoint ──
    state_dict["_structure_text_bank_buffer"] = structure_text_bank.cpu()
    state_dict["_boundary_text_bank_buffer"] = boundary_text_bank.cpu()

    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"]["_structure_text_bank_buffer"] = structure_text_bank.cpu()
        ckpt["model_state_dict"]["_boundary_text_bank_buffer"] = boundary_text_bank.cpu()
    elif "state_dict" in ckpt:
        ckpt["state_dict"]["_structure_text_bank_buffer"] = structure_text_bank.cpu()
        ckpt["state_dict"]["_boundary_text_bank_buffer"] = boundary_text_bank.cpu()
    elif "model" in ckpt:
        ckpt["model"]["_structure_text_bank_buffer"] = structure_text_bank.cpu()
        ckpt["model"]["_boundary_text_bank_buffer"] = boundary_text_bank.cpu()

    # 添加 CLIP 元数据到 checkpoint（便于审计）
    ckpt["clip_text_bank_metadata"] = {
        "source": backend_name,
        "text_dim": TEXT_DIM,
        "structure_shape": list(structure_text_bank.shape),
        "boundary_shape": list(boundary_text_bank.shape),
        "prompt_template_path": PROMPT_TEMPLATE_PATH,
    }

    # ── 8. 备份原文件 ──
    backup_path = output_path + ".clip_bak"
    if not os.path.exists(backup_path):
        print(f"[CLIP_EXTRACT] Backing up original to: {backup_path}")
        os.rename(output_path, backup_path)
    else:
        print(f"[CLIP_EXTRACT] Backup already exists: {backup_path}")

    # ── 9. 保存修改后的 checkpoint ──
    print(f"[CLIP_EXTRACT] Saving modified checkpoint to: {output_path}")
    torch.save(ckpt, output_path)
    print(f"[CLIP_EXTRACT] ✅ Done! CLIP text bank buffers added to checkpoint.")
    print(f"[CLIP_EXTRACT]    _structure_text_bank_buffer: {tuple(structure_text_bank.shape)}")
    print(f"[CLIP_EXTRACT]    _boundary_text_bank_buffer:  {tuple(boundary_text_bank.shape)}")
    print(f"[CLIP_EXTRACT]    metadata: {backend_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP text bank from prompt templates and inject into Exp6 checkpoint"
    )
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH,
                        help=f"Path to Exp6 checkpoint (default: {CKPT_PATH})")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH,
                        help="Output checkpoint path (default: same as ckpt_path)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16"],
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "open_clip", "clip"],
                        help="CLIP backend: auto (try open_clip → fallback), open_clip, clip (local)")
    parser.add_argument("--hf_cache", type=str, default="",
                        help="HuggingFace cache directory (only for open_clip backend)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force re-extraction even if buffers already exist")
    parser.add_argument("--verify", action="store_true", default=False,
                        help="Only verify existing text bank, don't re-extract")
    args = parser.parse_args()

    extract_clip_text_bank(
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        clip_model_name=args.clip_model,
        backend=args.backend,
        hf_cache=args.hf_cache,
        force=args.force,
        verify=args.verify,
    )
