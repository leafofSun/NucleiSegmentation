# CONCH_REQUIRED_BASELINE_TEST_FIX_REPORT

## 问题描述

Visual baseline 测试（无任何语义模块）失败，因为即使所有语义模块都禁用，`TextSam` 仍然默认尝试加载 CONCH 文本编码器。

**错误路径**：
```
test.py::_build_test_model()
  → sam_model_registry[args.model_type](args)
    → build_sam_vit_*(args)
      → _build_sam(...)
        → TextSam(...)
          → TextSam.__init__()
            → create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", ...)
              → hf_hub_download(...)
                → Cannot reach https://huggingface.co/MahmoodLab/conch/resolve/main/meta.yaml
```

**根本原因**：`enable_conch_text_encoder` 在三级调用链中均默认为 `True`：
1. [`build_sam.py:119`](segment_anything/build_sam.py:119)：`enable_conch_text_encoder=bool(_get_arg(args, "enable_conch_text_encoder", True))`
2. [`sam.py:1876`](segment_anything/modeling/sam.py:1876)：`enable_conch_text_encoder: bool = True`
3. [`test.py`](test.py) 在直接 `TextSam(...)` 构造调用中未传递 `enable_conch_text_encoder`

## 修改原则

严格按照用户要求：
- ✅ **不改模型结构**
- ✅ **不改 checkpoint**
- ✅ **不改训练逻辑**
- ✅ **只修测试/构建阶段的 CONCH 加载条件**
- ✅ **只有真正需要文本语义时才加载 CONCH**

## 修改方案（双层保护）

### 第一层保护：test.py 计算 `_conch_required` 并覆盖 `enable_conch_text_encoder`

[`test.py:993-1021`](test.py:993) — `[CONCH_REQUIRED_AUDIT]` 诊断块：

```python
_conch_required = any([
    bool(getattr(args, "enable_attr_text_alignment", False)),
    bool(getattr(args, "enable_promptnu_lite_align", False)),
    (bool(getattr(args, "enable_promptnu_guided_v3", False))
     and bool(getattr(args, "promptnu_guided_v3_use_text_bank", False))
     and not _conchless_flag),
    (bool(getattr(args, "use_pnurl", False)) and not _conchless_flag),
])
args.enable_conch_text_encoder = _conch_required
```

该覆盖同时作用于两个路径：
1. `sam_model_registry[args.model_type](args)` — 通过 `_get_arg(args, "enable_conch_text_encoder", True)` 传播
2. 直接 `TextSam(...)` 构造 — 通过 [`test.py:1097`](test.py:1097) 显式传入 `enable_conch_text_encoder=_conch_required`

### 第二层保护：sam.py 添加 `self.conch_required` 硬保护

[`sam.py:1937-1946`](segment_anything/modeling/sam.py:1937) — `self.conch_required` 逻辑：

```python
self.conch_required = any([
    bool(enable_attr_text_alignment),
    bool(enable_promptnu_lite_align),
    (bool(enable_promptnu_guided_v3)
     and bool(promptnu_guided_v3_use_text_bank)
     and not bool(use_checkpoint_text_bank_without_conch)),
    (bool(use_pnurl) and not bool(use_checkpoint_text_bank_without_conch)),
])
```

[`sam.py:1962`](segment_anything/modeling/sam.py:1962) — 修改 CONCH 加载条件：

```python
_skip_conch = self.use_checkpoint_text_bank_without_conch or not self.conch_required
if self.enable_conch_text_encoder and self.conch_required and not _skip_conch:
    # ... create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", ...)
elif _skip_conch:
    # CONCHLESS mode or CONCH not required: clip_model = None, tokenizer = None
```

### 诊断日志

- [`test.py:1002-1011`](test.py:1002) — `[CONCH_REQUIRED_AUDIT]`: 打印所有 CONCH 依赖模块的状态和最终判断
- [`sam.py:1964-1968`](segment_anything/modeling/sam.py:1964) — `[CONCH_SKIP]`: 当 `conch_required=False` 时打印跳过原因
- [`sam.py:1964-1965`](segment_anything/modeling/sam.py:1964) — `[CONCHLESS_TEST]`: 当 `use_checkpoint_text_bank_without_conch=True` 时打印

## 三种场景的期望状态

| 场景 | `conch_required` | `enable_conch_text_encoder` 最终值 | `clip_model` | HuggingFace 访问 | 文本编码来源 |
|------|:-:|:-:|:-:|:-:|:-:|
| **Visual baseline**（无语义模块） | `False` | `False` | `None` | ❌ 不访问 | 无（不需要） |
| **Exp6 CONCHLESS**（checkpoint text bank） | `True` | `True`（但跳过 CONCH） | `None` | ❌ 不访问 | checkpoint buffer |
| **Normal Phase C**（完整语义训练） | `True` | `True` | ✅ loaded | ✅ 需要 | CONCH |

## 验证结果

### 1. `py_compile` 语法检查 ✅

```bash
python -m py_compile test.py segment_anything/build_sam.py segment_anything/modeling/sam.py
```
- 三个文件均无语法错误。

### 2. Visual baseline 2-batch smoke 测试 ✅

**命令**：
```bash
torchrun --nproc_per_node=2 test.py \
  --data_path data/PanNuke/test \
  --checkpoint workdir/models/Visual_baseline/best_model.pth \
  --image_size 512 --crop_size 256 --num_workers 4 --distributed_test --workers_per_gpu 4 \
  --use_asr --asr_variant freqpath \
  --debug_max_test_batches 2 --metrics dice iou mAJI mPQ
```

**关键输出**：
```
[CONCH_REQUIRED_AUDIT]
  enable_attr_text_alignment=False
  enable_promptnu_lite_align=False
  enable_promptnu_guided_v3=False
  promptnu_guided_v3_use_text_bank=False
  use_pnurl=False
  use_checkpoint_text_bank_without_conch=False
  conch_required=False                          # ✅ 正确判断
  enable_conch_text_encoder_final=False          # ✅ CONCH 被禁用

[CONCH_SKIP] CONCH encoder skipped (conch_required=False).  # ✅ 跳过消息

[CONCH_MODULE_AUDIT] clip_model=None | tokenizer=None | prompt_learner=None  # ✅ 无 CONCH

📊 Final Results: dice=0.7930, iou=0.6639, mAJI=0.6183, mPQ=0.5426  # ✅ 指标正常
```

**成功标准**：`conch_required=False`，无 `create_model_from_pretrained`，无 `hf_hub_download`，无 huggingface.co 访问，指标正常。

### 3. Exp6 CONCHLESS 2-batch smoke 测试 ✅

**命令**：
```bash
torchrun --nproc_per_node=2 test.py \
  --data_path data/PanNuke/test \
  --checkpoint workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model.pth \
  --image_size 512 --crop_size 256 --num_workers 4 --distributed_test --workers_per_gpu 4 \
  --use_asr --asr_variant freqpath \
  --use_pnurl \
  --use_checkpoint_text_bank_without_conch \
  --enable_attr_text_alignment --enable_multilevel_attr_heads \
  --enable_promptnu_guided_v3 --promptnu_guided_v3_use_text_bank \
  --debug_max_test_batches 2 --metrics dice iou mAJI mPQ
```

**关键输出**：
```
[CONCH_REQUIRED_AUDIT]
  enable_attr_text_alignment=True
  enable_promptnu_guided_v3=True
  use_pnurl=True
  use_checkpoint_text_bank_without_conch=True
  conch_required=True                           # ✅ 语义模块启用时正确判断

[CONCHLESS_TEST] CONCH encoder skipped (text_bank from checkpoint).  # ✅ CONCH 跳过
[CONCH_CONFIG][TextSam] source=checkpoint_text_bank                 # ✅ 来源正确
[CONCHLESS_KEEP_PNURL] use_pnurl=True (KEPT from command line).    # ✅ PNuRL 保留

📊 Final Results: dice=0.7600, iou=0.6238, mAJI=0.5997, mPQ=0.5095  # ✅ 指标正常
```

**成功标准**：CONCH 跳过但语义模块正常使用 checkpoint text bank，指标正常。

## 修改文件清单

| 文件 | 修改类型 | 关键行 |
|------|----------|--------|
| [`test.py`](test.py:993) | 新增 `[CONCH_REQUIRED_AUDIT]` + 覆盖 `enable_conch_text_encoder` | 993-1021, 1097 |
| [`sam.py`](segment_anything/modeling/sam.py:1937) | 新增 `self.conch_required` + 修改 CONCH 加载条件 | 1937-1946, 1962-2038 |

`build_sam.py` 未修改 — `test.py` 在调用 `sam_model_registry` 前通过 `args.enable_conch_text_encoder = _conch_required` 覆盖默认值 `True`，自然传播到 `_build_sam()` 中。

## 结论

✅ **修复完成**。Visual baseline 测试不再尝试加载 CONCH，Exp6 CONCHLESS 向后兼容不受影响。双层保护机制确保：
1. 上层（test.py）在构建前正确判断并设置 `enable_conch_text_encoder`
2. 下层（sam.py）在构造函数中独立验证 `conch_required`，提供硬保护
