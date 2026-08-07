# SGA-SB v1 CORRECTION: Spatial Granularity-Aligned Structure/Boundary Guidance

## 1. Problem Statement

### 1.1 Original Implementation (SGA-SB v1 — Flawed)

The original `SpatialAttrHead` output **18 channels** (6 morphology attributes × 3 classes), which were aggregated by `SpatialSBGuidance` into a single unified spatial guidance map. This map was then applied as a **global feature modulation**:

```python
fused_image_embeddings *= (1 + spatial_guidance_map)  # single global modulation
```

**Key flaws:**
1. **Channel mismatch**: Paper requires 27 channels (5 structure attrs × 3 + 4 boundary attrs × 3), not 18
2. **No separation**: Structure and boundary should be guided independently, not fused into a single map
3. **Wrong injection point**: Guidance should be injected into FreqPathASRBlock's **separate low/high paths**, not applied as a late global multiplier
4. **Wrong loss**: Used CE loss for 18-class attr classification instead of SmoothL1 (structure) + BCE+Dice (boundary)

### 1.2 Corrected Implementation (SGA-SB v1 CORRECTION)

| Aspect | Before (Flawed) | After (Corrected) |
|--------|-----------------|-------------------|
| Prediction heads | `SpatialAttrHead` (18ch, 6 attrs × 3 classes) | `SpatialStructureHead` (1ch) + `SpatialBoundaryHead` (1ch) |
| Aggregation | Single unified map via `SpatialSBGuidance` | Separate adapters → 256ch → gamma-scaled |
| Injection | `fused *= (1 + map)` — global modulation | Low-path: `x_up += gamma_s * adapter(prob)` |
| | | High-path: `cnn_feat += gamma_b * adapter(prob)` |
| Target | Per-instance 6-attr classification labels | Structure: local occupancy via `avg_pool2d(k=31)` |
| | | Boundary: per-instance `mask - erosion` |
| Loss | CE loss over 18 channels | Structure: SmoothL1(sigmoid, target) |
| | | Boundary: BCEWithLogitsLoss + Dice |
| GT leakage | Oracle maps accepted in forward | RuntimeError if oracle maps in eval mode |

---

## 2. Architecture Overview

### 2.1 Prediction Heads (in [`sam.py`](NuSeg/segment_anything/modeling/sam.py:1797))

```
SpatialStructureHead (in_dim=256, hidden_dim=128)
  └─ Conv2d(256,128,3) → GroupNorm(8,128) → ReLU → Conv2d(128,1,1)
  └─ Output: [B, 1, 64, 64] structure_logits

SpatialBoundaryHead (in_dim=256, hidden_dim=128)
  └─ Same architecture as SpatialStructureHead
  └─ Output: [B, 1, 64, 64] boundary_logits

SpatialStructureAdapter (in_dim=1, out_dim=256, hidden_dim=32)
  └─ Conv2d(1,32,3,padding=1) → ReLU → Conv2d(32,256,1)
  └─ Projects [B,1,64,64] → [B,256,64,64]

SpatialBoundaryAdapter (in_dim=1, out_dim=256, hidden_dim=32)
  └─ Same architecture as SpatialStructureAdapter
  └─ Projects [B,1,64,64] → [B,256,64,64]
```

All `conv_out` layers are **zero-initialized** for safe integration.

### 2.2 Injection Points (in [`mask_decoder.py`](NuSeg/segment_anything/modeling/mask_decoder.py:323))

**FreqPathASRBlock.forward()** now accepts `structure_delta` and `boundary_delta`:

```
Low-path (structure-upsampled):
  x_up = self.structure_upsample(x)
  x_up = x_up + _sd                    # structure delta injection
  x_up = self.low_freq_modulation(x_up)

High-path (CNN boundary):
  cnn_feat = self.cnn_layers(x)
  cnn_feat = cnn_feat + _bd            # boundary delta injection
  cnn_feat = self.cnn_proj(cnn_feat)
```

### 2.3 Gamma-Residual Scaling (in [`sam.py`](NuSeg/segment_anything/modeling/sam.py:6442))

```python
if structure_delta is not None and gamma_structure is not None:
    structure_delta = structure_delta * gamma_structure  # learnable, init=0.05
if boundary_delta is not None and gamma_boundary is not None:
    boundary_delta = boundary_delta * gamma_boundary     # learnable, init=0.05
```

### 2.4 Target Generation (in [`spatial_sb_targets.py`](NuSeg/training/spatial_sb_targets.py:34))

**Structure target** — local occupancy map:
```python
foreground = (label_inst > 0).float()
occupancy = F.avg_pool2d(foreground, kernel_size=31, stride=1, padding=15)
# Output: [B, 1, 64, 64] float occupancy in [0, 1]
```

**Boundary target** — per-instance erosion:
```python
for each instance:
    mask = (inst_map == inst_id).float()
    eroded = -F.max_pool2d(-mask, kernel_size=3, padding=1)
    boundary = (mask - eroded).clamp(0, 1)
    full_boundary = torch.maximum(full_boundary, boundary)
# Output: [B, 1, H, W] binary, preserving internal holes
```

---

## 3. Run Modes

| Mode | Description |
|------|-------------|
| `none` | Skip all spatial structure/boundary computation entirely |
| `supervision_only` | Predict structure+boundary logits, compute losses, **NO** feature injection |
| `guidance` | Predict + loss + gamma-scaled delta injection into FreqPathASRBlock |

### 3.1 Legacy Ablation

The old `spatial_instance_attr_mode="v1"` preserves the original 18-channel morphology head for ablation studies. When enabled, the old `SpatialInstanceAttrHead` + `SpatialInstanceSBGuidance` run alongside the new heads (kept for backward compat).

---

## 4. Control Group Ablation Commands

### 4.1 Baseline (no spatial guidance)

```bash
python train.py --spatial_sb_mode none
```

### 4.2 Supervision-only (predict + loss, no injection)

```bash
python train.py --spatial_sb_mode supervision_only --spatial_structure_loss_weight 0.1 --spatial_boundary_loss_weight 0.1
```

### 4.3 Full guidance (predict + loss + injection)

```bash
python train.py --spatial_sb_mode guidance --spatial_sb_branch both --spatial_structure_loss_weight 0.1 --spatial_boundary_loss_weight 0.1 --spatial_structure_guidance_init 0.05 --spatial_boundary_guidance_init 0.05
```

### 4.4 Structure-only guidance

```bash
python train.py --spatial_sb_mode guidance --spatial_sb_branch structure --spatial_structure_loss_weight 0.1 --spatial_structure_guidance_init 0.05
```

### 4.5 Boundary-only guidance

```bash
python train.py --spatial_sb_mode guidance --spatial_sb_branch boundary --spatial_boundary_loss_weight 0.1 --spatial_boundary_guidance_init 0.05
```

### 4.6 Legacy ablation (original v1 morphology attrs, no corrected heads)

```bash
python train.py --spatial_sb_mode none --spatial_instance_attr_mode v1
```

### 4.7 Combined (corrected guidance + legacy ablation)

```bash
python train.py --spatial_sb_mode guidance --spatial_sb_branch both --spatial_structure_loss_weight 0.1 --spatial_boundary_loss_weight 0.1 --spatial_instance_attr_mode v1
```

---

## 5. Files Modified / Created

| File | Status | Description |
|------|--------|-------------|
| [`training/spatial_sb_targets.py`](NuSeg/training/spatial_sb_targets.py) | **Rewritten** | New target generation (`generate_structure_target`, `generate_boundary_target`, `batch_generate_spatial_sb_targets`) and loss functions (`compute_structure_loss`, `compute_boundary_loss`) |
| [`segment_anything/modeling/sam.py`](NuSeg/segment_anything/modeling/sam.py) | **Modified** | Added `SpatialStructureHead`, `SpatialBoundaryHead`, `SpatialStructureAdapter`, `SpatialBoundaryAdapter` classes. Renamed old classes to `SpatialInstanceAttrHead`/`SpatialInstanceSBGuidance`. Updated `TextSam.__init__` with new params. Updated forward with corrected spatial guidance and output dict. Added GT leakage protection. Added SPATIAL_SB_FORWARD_AUDIT diagnostic logging. |
| [`segment_anything/modeling/mask_decoder.py`](NuSeg/segment_anything/modeling/mask_decoder.py) | **Modified** | Updated `FreqPathASRBlock.forward()` to accept and inject `structure_delta` (low-path) and `boundary_delta` (high-path). Updated `MaskDecoder.forward()` and `predict_masks()` to pass deltas. |
| [`segment_anything/build_sam.py`](NuSeg/segment_anything/build_sam.py) | **Modified** | Added new params to `_build_sam()` signature and `TextSam` constructor call. |
| [`train.py`](NuSeg/train.py) | **Modified** | Updated imports, argparse, target generation, and loss computation to use new functions. |
| [`test.py`](NuSeg/test.py) | **Modified** | Updated TextSam constructor call and argparse with new spatial_sb parameters. |
| [`scripts/inspect_spatial_sb_targets.py`](NuSeg/scripts/inspect_spatial_sb_targets.py) | **Rewritten** | Updated to visualize structure occupancy and boundary maps instead of old 6-channel attr targets. |
| [`scripts/audit_spatial_sb_channels.py`](NuSeg/scripts/audit_spatial_sb_channels.py) | **Created** | Channel audit confirming 18 vs 27 channel mismatch. |
| [`scripts/smoke_spatial_sb_forward.py`](NuSeg/scripts/smoke_spatial_sb_forward.py) | **Created** | Forward-friendly smoke test with 7 test cases. |

---

## 6. Key Design Decisions

1. **Zero-initialized conv_out**: All adapter heads start with zero output to avoid disrupting the pretrained feature distribution at initialization.

2. **Learnable gamma at 0.05**: Gamma parameters start small (0.05) to allow gradual ramp-up during training, avoiding sudden feature perturbations.

3. **SmoothL1 for structure, BCE+Dice for boundary**: Structure occupancy is a continuous [0,1] map → SmoothL1 is appropriate. Boundary is a sparse binary map → BCE with dynamic pos_weight + Dice handles class imbalance.

4. **Per-instance boundary with internal holes**: Using `mask - erosion` preserves holes inside nuclei (e.g., nucleoli), which the original uniform boundary approach would miss.

5. **Three modes**: `none`/`supervision_only`/`guidance` provides clean ablation — `supervision_only` measures head quality without injection confound, `guidance` measures full effect.

6. **GT leakage protection**: The model's `forward()` raises `RuntimeError` if oracle spatial maps (`structure_target`/`boundary_target`) are passed in `batched_input` during eval mode. These maps are exclusively generated and consumed by `train.py`'s loss computation.

---

## 7. Channel Audit Summary

The original `SpatialAttrHead` outputs 18 channels:
```
[6 morphology attrs × 3 classes (low/mid/high)] = 18ch
```

The corrected design uses:
```
SpatialStructureHead → 1 channel (continuous occupancy)
SpatialBoundaryHead  → 1 channel (binary boundary)
Total: 2 output channels (not 27 — paper's 27 is for GT attribute classification, not spatial guidance)
```

The 27-channel requirement from the paper (5 structure × 3 + 4 boundary × 3) is for the **StructureBoundaryAttrHeads** (a separate classification module), not the spatial guidance heads. The spatial guidance heads correctly use **1 channel each** for the dense prediction task.
