#!/usr/bin/env python3
"""
pnudp_dense_ckpt_audit.py

Checkpoint parameter audit for PNuDP Dense Training (Stage D).

Reads best_aji_model.pth / latest_pnudp_dense_train_model.pth and prints:
  [PNUDP_DENSE_CKPT_PARAM_AUDIT]
    dense_alpha        value
    proj.weight        norm / std / absmax
    logit_proj.weight  norm / std / absmax
    num_loaded_pnudp_keys

Usage:
    python scripts/pnudp_dense_ckpt_audit.py \
        --checkpoint workdir/models/pnudp_dense_train_1ep_v1/best_aji_model.pth
"""

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PNuDP Dense Train checkpoint parameters."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the Stage D checkpoint (.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = args.checkpoint

    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", flush=True)
        sys.exit(1)

    # ── Load checkpoint ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state_dict
    state_dict = ckpt
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break
    # Strip 'module.' prefix if present
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # ── Filter PNuDP dense train keys ──
    pnudp_keys = {k: v for k, v in state_dict.items() if "pnudp_dense_train" in k}
    num_loaded_pnudp_keys = len(pnudp_keys)

    # ── Print audit ──
    print("[PNUDP_DENSE_CKPT_PARAM_AUDIT]", flush=True)
    print(f"  checkpoint_path={ckpt_path}", flush=True)
    print(f"  num_loaded_pnudp_keys={num_loaded_pnudp_keys}", flush=True)

    if num_loaded_pnudp_keys == 0:
        print("  [WARN] No pnudp_dense_train keys found in checkpoint!", flush=True)
        print("  Available keys containing 'pnudp':", flush=True)
        for k in state_dict.keys():
            if "pnudp" in k.lower():
                print(f"    {k}", flush=True)
        return

    # Print all pnudp keys found
    print("  pnudp_keys_found:", flush=True)
    for k in sorted(pnudp_keys.keys()):
        print(f"    {k}: shape={list(pnudp_keys[k].shape)}", flush=True)

    # ── dense_alpha ──
    alpha_key = "pnudp_dense_train.dense_alpha"
    if alpha_key in pnudp_keys:
        alpha_val = float(pnudp_keys[alpha_key].item())
        print(f"  dense_alpha={alpha_val:.12e}", flush=True)
    else:
        # Try any key containing "dense_alpha"
        alt_alpha = [k for k in pnudp_keys if "dense_alpha" in k]
        if alt_alpha:
            alpha_val = float(pnudp_keys[alt_alpha[0]].item())
            print(f"  dense_alpha (from '{alt_alpha[0]}')={alpha_val:.12e}", flush=True)
        else:
            print(f"  dense_alpha=NOT_FOUND", flush=True)

    # ── proj.weight ──
    proj_key = "pnudp_dense_train.proj.weight"
    if proj_key in pnudp_keys:
        w = pnudp_keys[proj_key].float()
        proj_norm = float(w.norm().item())
        proj_std = float(w.std().item())
        proj_absmax = float(w.abs().max().item())
        print(f"  proj.weight  norm={proj_norm:.8e}  std={proj_std:.8e}  absmax={proj_absmax:.8e}", flush=True)
    else:
        alt_proj = [k for k in pnudp_keys if "proj.weight" in k and "logit" not in k]
        if alt_proj:
            w = pnudp_keys[alt_proj[0]].float()
            proj_norm = float(w.norm().item())
            proj_std = float(w.std().item())
            proj_absmax = float(w.abs().max().item())
            print(f"  proj.weight (from '{alt_proj[0]}')  norm={proj_norm:.8e}  std={proj_std:.8e}  absmax={proj_absmax:.8e}", flush=True)
        else:
            print(f"  proj.weight=NOT_FOUND", flush=True)

    # ── logit_proj.weight ──
    logit_proj_key = "pnudp_dense_train.logit_proj.weight"
    if logit_proj_key in pnudp_keys:
        w = pnudp_keys[logit_proj_key].float()
        lp_norm = float(w.norm().item())
        lp_std = float(w.std().item())
        lp_absmax = float(w.abs().max().item())
        print(f"  logit_proj.weight  norm={lp_norm:.8e}  std={lp_std:.8e}  absmax={lp_absmax:.8e}", flush=True)
    else:
        alt_lp = [k for k in pnudp_keys if "logit_proj.weight" in k]
        if alt_lp:
            w = pnudp_keys[alt_lp[0]].float()
            lp_norm = float(w.norm().item())
            lp_std = float(w.std().item())
            lp_absmax = float(w.abs().max().item())
            print(f"  logit_proj.weight (from '{alt_lp[0]}')  norm={lp_norm:.8e}  std={lp_std:.8e}  absmax={lp_absmax:.8e}", flush=True)
        else:
            print(f"  logit_proj.weight=NOT_FOUND", flush=True)

    # ── Also check initial values for reference ──
    print("  [REFERENCE] Initial values:", flush=True)
    print("    proj.weight: Kaiming normal init (fan_out, relu)", flush=True)
    print("    logit_proj.weight: zero init (all zeros)", flush=True)
    print("    dense_alpha: 0.0", flush=True)

    # ── Additional context from checkpoint metadata ──
    if isinstance(ckpt, dict):
        phase = ckpt.get("phase", "N/A")
        epoch = ckpt.get("epoch", "N/A")
        best_aji = ckpt.get("best_aji", "N/A")
        print(f"  ckpt_phase={phase}  epoch={epoch}  best_aji={best_aji}", flush=True)


if __name__ == "__main__":
    main()
