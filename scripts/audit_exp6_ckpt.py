#!/usr/bin/env python3
"""
审计 Exp6 checkpoint 内容：检查 text_bank / PG3 / v3 相关 key。
用法: python NuSeg/scripts/audit_exp6_ckpt.py
"""
import torch
import os

CKPT_PATH = "/hy-tmp/NuSeg/workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth"

if not os.path.isfile(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location="cpu")

print("=" * 72)
print(f"📦 Checkpoint: {CKPT_PATH}")
print(f"📄 File size: {os.path.getsize(CKPT_PATH) / 1024 / 1024:.2f} MB")
print("=" * 72)

# 1. Top-level keys
print("\n🔑 Top-level keys:")
for k in ckpt.keys():
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"   {k}: dict[{len(v)} entries]")
    elif isinstance(v, torch.Tensor):
        print(f"   {k}: Tensor{tuple(v.shape)}")
    elif isinstance(v, (int, float, str, bool)):
        print(f"   {k}: {v}")
    elif v is None:
        print(f"   {k}: None")
    else:
        print(f"   {k}: {type(v).__name__}")

# 2. Search state_dict for relevant keys
state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
print(f"\n🔍 Searching state_dict ({len(state_dict)} keys) for v3 / text_bank / CONCH related keys...")

KEYWORDS = [
    "conch", "text_bank", "promptnu_guided", "guided_v3",
    "structure_text", "boundary_text", "text_proj", "prompt_bank",
    "attr_text", "structure_prompt", "boundary_prompt",
    "sb_structure_cache", "sb_boundary_cache", "attr_text_instance_cache",
    "clip_model", "tokenizer", "prompt_learner",
    "sb_conch", "pnudp", "promptnu_guided_adapter",
    "structure_boundary_attr_heads"
]

found_keys = {}
for kw in KEYWORDS:
    matched = [k for k in state_dict.keys() if kw.lower() in k.lower()]
    if matched:
        found_keys[kw] = matched

print(f"\n{'─' * 72}")
print(f"📊 Found {sum(len(v) for v in found_keys.values())} matched keys across {len(found_keys)} categories:")
print(f"{'─' * 72}")

for kw, keys in sorted(found_keys.items()):
    print(f"\n  [{kw}] ({len(keys)} keys):")
    for k in sorted(keys):
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            print(f"    • {k}: Tensor{tuple(v.shape)} | "
                  f"dtype={v.dtype} | "
                  f"requires_grad={v.requires_grad} | "
                  f"device={v.device}")
        else:
            print(f"    • {k}: {type(v).__name__} = {str(v)[:100]}")

# 3. Check if text_bank-like tensors are stored as buffers
print(f"\n{'═' * 72}")
print(f"🔬 DEEP INSPECTION: text_bank / embedding buffer candidates")
print(f"{'═' * 72}")

# Look for any key containing [5,3,D] or [4,3,D] shaped tensors (text_bank shape)
text_bank_candidates = []
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor) and v.dim() == 3:
        if v.shape[0] in (4, 5) and v.shape[1] == 3:
            text_bank_candidates.append((k, v.shape, v.dtype))

if text_bank_candidates:
    print(f"\n  ✅ Found {len(text_bank_candidates)} text_bank-shaped tensors [N,3,D]:")
    for k, shape, dtype in text_bank_candidates:
        print(f"    • {k}: {shape} ({dtype})")
else:
    print(f"\n  ❌ No text_bank-shaped [N,3,D] tensors found in state_dict")

# Also look for any 512-dim embeddings
embed_candidates = []
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[-1] == 512:
        if v.shape[0] <= 27:  # likely prompt embeddings
            embed_candidates.append((k, v.shape, v.dtype))

if embed_candidates:
    print(f"\n  Found {len(embed_candidates)} candidate embedding tensors (dim=512, N≤27):")
    for k, shape, dtype in sorted(embed_candidates, key=lambda x: x[0]):
        print(f"    • {k}: {shape} ({dtype})")

# 4. Check if promptnu_guided_adapter exists in state_dict
print(f"\n{'═' * 72}")
print(f"🧩 PROMPTNU_GUIDED_ADAPTER KEYS")
print(f"{'═' * 72}")
adapter_keys = [k for k in state_dict.keys() if "promptnu_guided_adapter" in k.lower()]
if adapter_keys:
    print(f"\n  Found {len(adapter_keys)} adapter keys:")
    for k in sorted(adapter_keys):
        v = state_dict[k]
        print(f"    • {k}: Tensor{tuple(v.shape) if isinstance(v, torch.Tensor) else '?'}")
else:
    print(f"\n  ❌ No promptnu_guided_adapter keys found in state_dict")

# 5. Check for structure_boundary_attr_heads
sb_keys = [k for k in state_dict.keys() if "structure_boundary_attr_heads" in k.lower()]
if sb_keys:
    print(f"\n  Found {len(sb_keys)} structure_boundary_attr_heads keys:")
    for k in sorted(sb_keys)[:10]:
        print(f"    • {k}")
    if len(sb_keys) > 10:
        print(f"    ... and {len(sb_keys)-10} more")

print(f"\n{'═' * 72}")
print("✅ 审计完成")
print(f"{'═' * 72}")
