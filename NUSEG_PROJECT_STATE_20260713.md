# NuSeg 项目事实状态（2026-07-13）

## 1. 审计边界与证据等级

本报告仅审计 `/hy-tmp/NuSeg` 当前代码、配置、日志、checkpoint 元数据和已有文档；未修改代码，未执行训练、验证或 full test。事实优先级为：当前代码 > checkpoint 安全元数据 > 原始日志/结果 > 历史文档 > 实验命名。项目目录不是有效 Git 工作树，因此无法用 commit/branch 固定源码版本。

用户给定的 full-test 固定数值在现有日志/结果中未找到完整原始输出，统一标记 `PROVIDED_CONTEXT_UNVERIFIED`，但不改写数值。

## 2. 项目目标与当前真实主线

目标：以 SAM/MedSAM 为视觉底座，结合 FreqPathASR、核属性监督及可选病理视觉语言语义，完成核实例分割。

截至当前文件快照，**有 checkpoint 和训练日志支撑的真实主线是 Exp5：图像级数值属性经 `numeric_attr_freqpath_proj` 路由到 FreqPathASR 的 structure/low 与 morphology/high 路径**。当前代码所谓 low/high frequency 是功能性特征路径，不是 FFT/DCT 或显式频谱分解。

最新 SGA-SB corrected 是候选下一阶段主线，但仅达到 `IMPLEMENTED_NOT_TRAINED`：

- `SpatialStructureHead`、`SpatialBoundaryHead`、两个 adapter、独立 gamma 和 low/high 注入代码存在；
- `train.py:4810-4844` 构造 `TextSam` 时没有传递任何 `spatial_sb_*` 参数，模型始终采用构造默认值 `spatial_sb_mode="none"`；
- `apply_stage_policy()` 首先冻结全模型，随后没有对新 head/adapter/gamma 解冻；
- `build_optimizer_by_stage()` 没有收集这些模块；
- 所有已审计 checkpoint 中对应键计数均为 0。

## 3. 当前真实架构图

```text
image -> SAM/MedSAM image encoder -> image_embeddings [B,256,64,64]
  |                                      |
  |                                      +-> Phase B multi-level attribute heads
  |                                            -> numeric_attr_freqpath_proj (Exp5)
  |                                            -> low attr_prompt / high morph_prompt
  |
  +-> CNN stage0/1/2 multi-scale detail -------+-> FreqPathASRBlock x2 -> mask logits
                                                     | low: structure_upsample -> semantic modulation
                                                     | high: CNN feature -> morphology modulation -> cnn_proj
                                                     +-> fusion -> instance mask/HV/heatmap outputs

Optional historical semantic path:
Phase C attr-text projections -> Phase D semantic_channel_gate / PromptNu-lite / PG3
                              -> CONCH or checkpoint text bank

Corrected SGA-SB code path (currently unreachable from train.py configuration):
image_embeddings -> StructureHead -> sigmoid -> StructureAdapter -> gamma_structure
                 -> low path after structure_upsample
image_embeddings -> BoundaryHead  -> sigmoid -> BoundaryAdapter  -> gamma_boundary
                 -> high path before cnn_proj
```

## 4. SGA-SB corrected 静态审计

| 项目 | 当前事实 | 结论 |
|---|---|---|
| Structure head | `sam.py:1797`，1-channel logits，末层 zero-init | 已实现 |
| Boundary head | `sam.py:1828`，1-channel logits，末层 zero-init | 已实现 |
| Structure adapter | `sam.py:1864`，1→256，输出层 zero-init | 已实现 |
| Boundary adapter | `sam.py:1887`，1→256，输出层 zero-init | 已实现 |
| gamma | `sam.py:2162/2172`，独立参数，构造默认 0.05 | 已实现但未进入训练 |
| low 注入 | `mask_decoder.py:384-392`，`structure_upsample` 后注入 | 已实现；代码中不存在名为 `low_freq_modulation` 的独立调用，实际是在 attr modulation 后加入 |
| high 注入 | `mask_decoder.py:419-427`，CNN feature 调制后、`cnn_proj` 前注入 | 已实现 |
| modes | none / supervision_only / guidance；legacy `v1` 映射 guidance | 已实现 |
| branches | structure / boundary / both | 已实现 |
| structure target/loss | local occupancy；bilinear resize；SmoothL1(sigmoid, target) | 已实现 |
| boundary target/loss | 每实例 mask-minus-erosion；nearest resize；BCEWithLogits + Dice | 已实现 |
| dtype/range | structure float [0,1]；boundary binary float | 静态代码正确；已有 target 图仅是可视化审计 |
| GT leakage | eval 时检测输入 oracle key；模型 forward 自身不消费 target | 保护存在；`test.py` 未传 target |
| train constructor | 未传 `spatial_sb_*` | **阻断** |
| stage/optimizer | 未解冻、未注册 optimizer | **阻断** |
| checkpoint | 所有关键 checkpoint 均无新模块键 | 未训练 |

`spatial_sb_mode=none` 在当前构造逻辑下不会创建新模块或新 forward 计算，静态上保持旧行为；但论文级“严格等价”仍需 P0 固定 checkpoint/test pipeline 后做输出哈希或逐张量比较。`supervision_only` 的 forward 不创建 delta，因而没有 feature injection；当前训练入口问题使该模式也无法实际启用。

## 5. Phase 与模块状态

状态词严格限定为任务给定集合。

| 模块 | 状态 | 事实依据 |
|---|---|---|
| Phase A visual baseline | VERIFIED_EFFECTIVE | 30 epoch 历史、checkpoint；checkpoint val Dice 0.812706、mAJI 0.619436、mPQ 0.561229 |
| Phase B multilevel attribute warmup | VERIFIED_EFFECTIVE | epoch 28 checkpoint，32 个 multilevel head 键及属性验证指标；不等于已证明分割增益 |
| Phase C attribute-text alignment | VERIFIED_EFFECTIVE | epoch 9 full-for-Phase-D checkpoint，8 个 attr-align 键；只证明 alignment 阶段完成 |
| 旧 Phase D semantic_channel_gate | VERIFIED_NO_CLEAR_GAIN | 多个训练 checkpoint 存在，但缺少严格同父 checkpoint 因果对照 |
| PromptNu-lite v2 | VERIFIED_NO_CLEAR_GAIN | 5-epoch 实验，best val mAJI 0.637797；无独立 full-test 原始结果 |
| PromptNu-guided v3 / PG3 | VERIFIED_NO_CLEAR_GAIN | Exp6 已训练；用户固定 full test 未超过 Exp5，且比较受父 checkpoint/模块差异影响 |
| Exp5 no-text route | VERIFIED_EFFECTIVE | 10-epoch checkpoint、日志、数值 route 参数和 optimizer 组明确；full-test 数值仍为用户记录 |
| Exp6 CONCH route | VERIFIED_NO_CLEAR_GAIN | 10-epoch checkpoint；与 Exp5 不严格可比 |
| Exp7 CLIP route | HISTORICAL_ABLATION | 未发现独立训练 checkpoint；同一 Exp6 权重被替换 text bank 后测试 |
| PNuDP Dense | SMOKE_ONLY | 多个 smoke/20-batch/1-epoch event；当前无 PNuDP checkpoint，未见 full test |
| PNuDP channel-specific support | IMPLEMENTED_NOT_TRAINED | `pnudp_dense_num_mask_channels` 默认 3 及 channel-specific projection 存在；无对应保留 checkpoint |
| 旧 SpatialAttrHead / SpatialInstanceAttrHead | DEPRECATED | 代码以 `spatial_instance_attr_mode=v1` 保留，默认 none |
| SGA-SB corrected heads | IMPLEMENTED_NOT_TRAINED | 静态代码存在但训练入口、冻结和 optimizer 三重断路 |

## 6. checkpoint 谱系

```text
sam-med2d_b.pth / earlier visual checkpoint
  -> Visual_baseline/best_model.pth (phase=vision, epoch=28, seed=42)
     -> phaseB_ml_instancefix.../best_multilevel_attr_model.pth (epoch=28)
        +-> Exp5 numeric no-text (10 ep; parent=Phase B)
        -> phaseC_align.../best_align_full_model.pth (epoch=9)
           +-> PromptNu-lite v2 (5 ep)
           +-> Exp6 CONCH/PG3 (10 ep)
                -> post-hoc text-bank replacement -> Exp7 CLIP-style test artifact
```

Exp6 目录中的 `best_aji_model.pth` 当前包含 `clip_text_bank_metadata.source=clip/ViT-B/32`；原 CONCH bank 版本保存在 `.clip_bak`，无 bank 的早期版本在 `.bak`。因此该目录当前 checkpoint 命名具有污染风险。

## 7. 必须回答的核心问题

1. Exp5 实际启用：Phase B multilevel heads、`numeric_attr_freqpath_proj`、FreqPathASR low/high prompt 路由、semantic-injection 阶段的分割相关模块，以及 post-resume FreqPath modulator reinit。不是 SGA-SB。
2. Exp5 checkpoint/log 明确 `enable_conch_text_encoder=False`、PG3/text bank false，且 checkpoint 无 text-bank 键；因此 Exp5 不依赖 CONCH、CLIP 或 text bank。
3. Exp6 与 Exp5 不仅差文本：父 checkpoint（Phase C vs Phase B）、`use_pnurl`、attr-align、PG3、prompt learner、optimizer 组均不同；epoch 均为 10、seed 均为 42，但不能严格比较。
4. Exp7 未发现独立训练；仅看到在 Exp6 epoch-3 best-AJI 权重上替换 CLIP bank。它不是“只替换 encoder 且其他训练条件严格一致”的训练实验。
5. Exp6≈Exp7 只能说明在该 post-hoc bank 替换测试里 encoder/bank 选择未造成明显差异，不能排除语义路径、prompt 质量、注入幅度或 pipeline 是瓶颈。
6. corrected code 将旧统一 spatial guidance 分离并默认关闭旧模块；但训练入口未接通，谈不上已完成运行时替代。
7. 旧 spatial instance 模块默认 none；其他历史模块由 phase/flags 控制。semantic-injection 会自动设置部分 PNuRL/CoOp 行为，必须记录完整 args，不能只看实验名。
8. `none` 静态上跳过 corrected heads；严格历史等价尚未做输出级验证。
9. `supervision_only` 静态上完全没有 delta 注入。
10. 新 heads/adapters/gamma 当前未进入 optimizer，且被 stage policy 冻结。
11. 从旧 checkpoint resume 时所有新 SGA-SB 参数都会按当前构造随机/零初始化；但当前 train 构造根本不创建它们。
12. `test.py` 完整声明并传递 SGA-SB 配置；`train.py` 声明参数但未传给 `TextSam`。
13. PNuDP Dense 将 dense text logits 投影后加到 mask logits，未直接耦合 FreqPath low/high 内部路径。
14. 受混杂影响最大：Exp5-vs-Exp6（父 checkpoint/模块/optimizer）；Exp6-vs-Exp7（checkpoint text-bank 后处理）；用户固定 full-test 数值（缺原始结果路径和精确命令）。
15. 有证据：FreqPath 功能性 low/high 路由、Exp5 no-text 完整训练。仅假设：SGA-SB 带来增益、CONCH 语义锚定有效、独立 gamma 有效、PNuDP Dense 有效。

## 8. 代码与历史文档冲突

- correction report 声称 train/build 已完整接通；当前 `train.py` 构造未传 SGA-SB 参数，结论不成立。
- correction report 给出可运行 ablation 命令；当前命令参数会被 parser 接收但不会影响训练模型。
- 历史实现报告描述 unified 18-channel spatial modulation；当前代码已改名为 legacy ablation，默认关闭。
- 历史文档多次把“实现/烟测”写成接近“完成”；本报告按 checkpoint 和 optimizer 事实降级。

