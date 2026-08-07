# EXP5 Numeric Attr Route Test Fix Report

## 问题描述

```text
test.py: error: unrecognized arguments: --enable_numeric_attr_freqpath_guidance
```

Exp5 训练的 checkpoint 使用 `--enable_numeric_attr_freqpath_guidance`，但 `test.py` 的 argparse 和 TextSam 构造器均未支持该参数，导致测试入口崩溃。

## 设计决策

### 关键发现：`enable_numeric_attr_freqpath_guidance` ≠ `sb_guidance_mode=pred_direct`

审计 [`sam.py`](NuSeg/segment_anything/modeling/sam.py:6326-6356) forward 逻辑后发现，二者是 **完全独立的前向路径**，且在 forward 触发条件中 **互斥**：

- **NumericAttrFreqPathProj** (line 6326-6356): 将 `structure_attr_logits [B,5,3]` / `boundary_attr_logits [B,4,3]` flatten → LayerNorm → MLP → [B,512]，直接注入 FreqPath 的 `attr_prompt` / `morph_feat`
- **SBDirectAttrGuidanceAdapter** (line 5593-5649): argmax → MLP → deltas，通过 SemanticDeltaAdapter 注入

触发条件（line 6333-6334）明确将 `sb_guidance_mode != "none"` 作为 **阻断条件**：

```python
elif self.sb_guidance_mode not in ("none",):
    _numeric_guidance_reason = f"sb_guidance_mode_not_none:{self.sb_guidance_mode}"
```

因此，**不能**将 `--enable_numeric_attr_freqpath_guidance` 做成 `sb_guidance_mode=pred_direct` 的别名。正确做法是作为独立参数直接传递。

### 训练侧配置审计结果

| 参数 | 值 |
|------|-----|
| `enable_conch_text_encoder` | `False` |
| `enable_attr_text_alignment` | `False` |
| `enable_promptnu_guided_v3` | `False` |
| `enable_multilevel_attr_heads` | `True` |
| `sb_guidance_mode` | `"none"` (未被覆盖) |
| `use_pnurl` | PNuRL optimizer group 存在但 forward 未启用 |

结论：训练侧使用 `enable_numeric_attr_freqpath_guidance` 作为独立路径，**非** `pred_direct` 别名。

### 修改文件

#### 1. [`test.py`](NuSeg/test.py)

- **argparse** (line 1553-1567): 添加 `--enable_numeric_attr_freqpath_guidance` 和 `--numeric_attr_freqpath_hidden_dim`
- **TextSam 构造器** (line 1096-1098): 传递两个参数到 model
- **`[NUMERIC_ATTR_ROUTE_TEST_CONFIG]`** (line 1259-1276): 一次性诊断日志，打印路由配置
- **rank0 print** (line 1290): 添加 `enable_numeric_attr_freqpath_guidance` 输出

#### 2. [`sam.py`](NuSeg/segment_anything/modeling/sam.py)

- **`[NUMERIC_ATTR_ROUTE_FORWARD_AUDIT]`** (line 6521-6543): 前向一次性诊断（eval only），报告 `numeric_attr_route_active`、structure/boundary attrs 可用性、sb_guidance_delta_norm 等

#### 3. `build_sam.py` — **未修改**

已在（line 121-122, 218-219, 313-314, 411-412, 557-558）支持 `enable_numeric_attr_freqpath_guidance`。test.py 直接调用 `TextSam(...)`，参数通过构造器传递即可。

### CONCH 安全审计

`enable_numeric_attr_freqpath_guidance` 不在 `_conch_required` 的 any[] 列表中（test.py:994-1001, sam.py:1939-1946），因此不会触发 CONCH 加载。smoke 测试已确认：
- `conch_required=False`
- `clip_model=None`
- `tokenizer=None`
- `prompt_learner=None`
- `[CONCH_SKIP] CONCH encoder skipped (conch_required=False)`
- `hf_hub_offline=True`

## 验证结果

### py_compile

```bash
python -m py_compile test.py segment_anything/build_sam.py segment_anything/modeling/sam.py
```

**通过** — exit code 0，零语法错误。

### 2-batch Smoke 测试

```bash
torchrun --nproc_per_node=2 test.py \
  --data_path data/PanNuke/test \
  --checkpoint workdir/models/exp5_numeric_attr_route_10ep_reinit1e4_v1/best_aji_model.pth \
  --image_size 512 --crop_size 256 --num_workers 4 --distributed_test --workers_per_gpu 4 \
  --use_asr --asr_variant freqpath \
  --enable_multilevel_attr_heads \
  --use_structure_boundary_attrs \
  --structure_boundary_attr_path workdir/attr_stats/gt_structure_boundary_attr_all.jsonl \
  --enable_numeric_attr_freqpath_guidance \
  --hf_hub_offline \
  --debug_max_test_batches 2 --metrics dice iou mAJI mPQ
```

**通过** — exit code 0。

### 8 项成功标准验证

| # | 标准 | 状态 | 证据 |
|---|------|------|------|
| 1 | `test.py -h` 能看到 `--enable_numeric_attr_freqpath_guidance` | ✅ | argparse 定义 at test.py:1554-1561，smoke 命令成功解析 |
| 2 | 不再出现 `unrecognized arguments` | ✅ | 无此类错误 |
| 3 | `conch_required=False` | ✅ | `[CONCH_REQUIRED_AUDIT] conch_required=False` |
| 4 | 不访问 HuggingFace | ✅ | `hf_hub_offline=True`, `[CONCH_SKIP]`, `clip_model=None` |
| 5 | `numeric_attr_route_active=True` | ✅ | `[NUMERIC_ATTR_ROUTE_FORWARD_AUDIT] numeric_attr_route_active=True` |
| 6 | structure/boundary attrs 已加载 | ✅ | `structure_attrs_available=True, boundary_attrs_available=True` |
| 7 | Dice/IoU/mAJI/mPQ 正常输出 | ✅ | dice=0.7987, iou=0.6724, mAJI=0.6305, mPQ=0.5654 |
| 8 | 未训练 | ✅ | 仅推理，无训练循环 |

### 诊断日志确认

```
[NUMERIC_ATTR_ROUTE_TEST_CONFIG]
  enable_numeric_attr_freqpath_guidance=True
  numeric_attr_freqpath_proj_exists=True
  enable_structure_boundary_attr_heads=False
  sb_guidance_mode=none
  ...

[NUMERIC_ATTR_ROUTE_FORWARD_AUDIT]
  numeric_attr_route_active=True
  structure_attrs_available=True
  boundary_attrs_available=True
  sb_guidance_delta_norm=0.00000000
  sb_guidance_weight=0.0500
  FREQPATH_ABLATION=both
```

## 结论

Exp5 numeric_attr_route 测试入口修复完成。`test.py` 现在完整支持 `--enable_numeric_attr_freqpath_guidance` 参数，且与训练侧真实前向路径一致（非 `pred_direct` 别名）。CONCH 不会被错误加载，所有指标正常输出，零训练。
