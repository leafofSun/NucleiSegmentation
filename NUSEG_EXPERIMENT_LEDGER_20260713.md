# NuSeg 实验账本（2026-07-13）

## 口径

- 主要指标：mAJI、mPQ；次要指标：Dice、IoU。
- `用户固定 full test`：保持用户给定值，若仓库没有原始结果则标记 `PROVIDED_CONTEXT_UNVERIFIED`。
- checkpoint 中的单次 validation 数值仅用于谱系核对，不用于宣称论文级提升。
- `FREQPATH_ABLATION` 在相关训练日志中为 `both`；无日志者记 UNKNOWN。

## 核心实验

| 实验 | parent checkpoint | phase / epoch / seed | 核心开关与文本 | FREQPATH_ABLATION | spatial_sb_mode | 可训练模块摘要 | Dice / IoU / mAJI / mPQ | 原始路径 | 严格可比 | 当前结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| Visual baseline | earlier `mp_sam_vision_phase/best_model.pth` | vision / 30（best epoch 28）/ 42 | use_asr=true, freqpath, no PNuRL/text | checkpoint args 未记录该环境变量 | 参数不存在（历史等价 none） | decoder/FreqPath/CNN/ASR/image adapters | 用户固定：0.8089 / 0.6963 / 0.6270 / 0.6034 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；checkpoint val：`workdir/models/Visual_baseline/best_model.pth`，metrics history 同目录 | 否；固定 full test 缺命令/结果文件 | 基线可用；固定 full-test 来源待补 |
| Phase B multilevel warmup | Visual baseline | multilevel_attr_warmup / 30（best epoch 28）/ 42 | multilevel heads=true；CONCH=false | UNKNOWN | 参数不存在 | multilevel + legacy structure/boundary attr heads | 非分割实验 | `workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth` | 不适用 | 属性头已训练；不证明分割增益 |
| Phase C alignment | Phase B | semantic_alignment / 10（best epoch 9）/ 42 | attr-text alignment=true；CONCH=true | UNKNOWN | 参数不存在 | 4 个 attr-align projections | 非分割实验 | `workdir/models/phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD/best_align_full_model.pth` | 不适用 | alignment 已训练；不证明语义注入增益 |
| PromptNu-lite v2 | Phase C | semantic_injection / 5（best epoch 3）/ 42 | lite=true；target=semantic_delta；struct/bound=0.05 | UNKNOWN | 参数不存在 | semantic injection + attr-align projections 等 | checkpoint best val Dice 0.809370；mAJI 0.637797；IoU/mPQ 未存顶层 | `workdir/models/promptnu_lite_v2_semantic_delta_rms_w005_5ep_v1/best_aji_model.pth` | 否 | 已训练但无明确严格增益 |
| Exp5 numeric no-text | Phase B | semantic_injection / 10（best AJI epoch 8）/ 42 | numeric route=true；small_normal；modulator reinit 1e-4；CONCH/PG3/text bank=false；use_pnurl=false | both | 参数不存在 | segmentation groups + numeric_attr_freqpath_proj | 用户固定：0.8172 / 0.7048 / 0.6361 / 0.6094 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；训练日志 `workdir/logs/exp5_numeric_attr_route_10ep_reinit1e4_v1_20260708_1214.log`；checkpoint 同名模型目录 | 与 baseline 仍缺严格同 pipeline 证据；与 Exp6 否 | 当前用户固定记录最好；真实主线 |
| Exp6 CONCH/PG3 | Phase C | semantic_injection / 10（best AJI epoch 3）/ 42 | use_pnurl=true；attr-align=true；PG3=true；CONCH/text bank=true；modulator reinit 1e-4 | both | 参数不存在 | Exp5 多数 segmentation groups + PNuRL/prompt learner/PG3 projections；无 numeric projection | 用户固定：0.8091 / 0.6945 / 0.6269 / 0.6047 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；训练日志 `workdir/logs/exp6_phaseC_text_both_10ep_reinit1e4_v1_20260708_1622.log` | 否 | 无清晰增益；不能归因于文本本身 |
| Exp7 CLIP/PG3 | Exp6 best-AJI epoch-3 权重 | post-hoc test artifact；无独立训练记录 | 将 text bank 替换为 CLIP ViT-B/32；checkpoint args 仍写 CONCH=true | test 默认可能为 both，精确命令未找到 | none | 无训练；仅替换 buffer | 用户记录：与 Exp6 基本一致 | `workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model_clip_textbank.pth`；结果文件未找到 | 否 | HISTORICAL_ABLATION；不能视为严格 encoder 训练对照 |
| PNuDP Dense | Phase C/Exp6 路线（按脚本和命名） | smoke、20-batch、1-epoch | dense projection/logit_add/channel-specific support | UNKNOWN | none | 仅 PNuDP dense proj/logit_proj/alpha（设计） | 无保留 full-test 数值 | `workdir/runs/pnudp_*`；当前 `workdir/models` 无 PNuDP checkpoint | 否 | SMOKE_ONLY |
| SGA-SB corrected | 计划父 checkpoint 未固定 | 未训练 | parser 有 none/supervision_only/guidance 与 branch | 计划 both | 训练构造实际恒为 none | 新模块未解冻、未进 optimizer | 无 | 代码与 target audit：`segment_anything/modeling/sam.py`、`mask_decoder.py`、`training/spatial_sb_targets.py`、`workdir/audits/spatial_sb_targets_corrected/` | 否 | IMPLEMENTED_NOT_TRAINED |

## Exp5 与 Exp6 optimizer 差异

Exp5 checkpoint 的 optimizer 含 `numeric_attr_freqpath_proj`（12 个参数张量），无 PG3/attr-align 组；Exp6 含 prompt learner、4 组 PG3 adapter/projection，且父 checkpoint 含 8 个 attr-align 键，无 numeric projection。两者不是单变量文本对照。

## checkpoint 污染记录

Exp6 目录：

- `best_aji_model.pth.bak`：没有 text-bank buffer；
- `best_aji_model.pth.clip_bak`：有原先 bank，但无 CLIP metadata；
- 当前 `best_aji_model.pth`：含 `clip/ViT-B/32` metadata；
- `best_aji_model_clip_textbank.pth`：与当前文件大小和元数据一致。

测试 Exp6/Exp7 前必须按哈希固定具体文件，不能只写 `best_aji_model.pth`。

## 固定数值来源状态

仓库全文搜索未找到四组 full-test 数值的完整原始输出。Visual baseline 的 `metrics_history.json` 只出现 Dice 0.8089036 的 validation 记录，其他固定指标不成套；这不足以认定为同一 full test。故四组固定记录均保留为 `PROVIDED_CONTEXT_UNVERIFIED`。

