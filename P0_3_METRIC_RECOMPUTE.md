# P0.3 标准指标重算与文献坐标定位

`training_started=false`

`inference_rerun=false`

本任务仅用 CPU 读取已存盘预测与 GT；未启动训练、未重跑推理、未改写任何预测文件。最终可复算产物位于远端：

```text
/hy-tmp/NuSeg/workdir/audits/p0_3_metric_recompute_20260808_v3/
```

## 0. 环境、输入哈希、.npy 内容确认

- 执行代码 commit：`66d697f765908056f690eff1d2d8efb02ecccb66`（远端 detached HEAD，执行后工作树干净）。
- 执行方式：`CUDA_VISIBLE_DEVICES="" python3 evaluation/recompute_from_npy.py ...`，纯 CPU。
- Visual baseline 输入：`/hy-tmp/NuSeg/workdir/audits/p0_20260713/results_1gpu_canonical/visual_baseline/`
  - 2607 `.npy` + 2607 `.png`
  - 内容摘要 SHA256：`d643be69bd8683a7456d31cd8e41e3bc79fc4bd8a46150eff1bedc3018204140`
- Exp5 输入：`/hy-tmp/NuSeg/workdir/audits/p0_20260713/results_1gpu_canonical/exp5_best_pq/`
  - 2607 `.npy` + 2607 `.png`
  - 内容摘要 SHA256：`fac6f9b444c5e3af9e375daa7fb559b832a5394e5dfd7754e117ae8b52af2776`
- GT 输入：`/hy-tmp/NuSeg/data/PanNuke/test/`
  - 2607 `.png` + 2607 `.json`
  - 内容摘要 SHA256：`22b95cb77c93527f16b1f815e55962ea381fa0bbfa92496e84f497cfa8ab6a47`

### 0.1 `.npy` 内容门禁

门禁结论：两组 `.npy` 都是最终实例标签图，本任务可行。

| 检查项 | Visual baseline | Exp5 |
|---|---:|---:|
| dtype | `int32` | `int32` |
| shape | 全部 `(256, 256)` | 全部 `(256, 256)` |
| 全目录值域 | `[0, 119]` | `[0, 120]` |
| 背景为 0 | 2607/2607 | 2607/2607 |
| 空预测图 | 17 | 8 |
| ID 不连续图 | 7 | 2 |
| 单图最大实例数 | 119（`sa_0005945`） | 120（`sa_0005945`） |
| PNG 与 NPY 像素级完全相同 | 2607/2607 | 2607/2607 |

`.png` 是无损的 `uint16` 标签图副本，不只是彩色可视化。保存逻辑见 `test.py:936-951`。指标计算只读 `.npy`；计算前对 GT 和预测的非零 ID 连续重编号。

## 1. GT 定位与类别标签可用性

- 绝对路径：`/hy-tmp/NuSeg/data/PanNuke/test/`。
- 格式：每个样本为同名 PNG 图像和 COCO/SA-1B 风格 JSON；JSON 多边形按当前测试入口 `test.py:880-926` 解码为 `int32` 实例图。
- 样本数：2607；预测键 `sa_NNNNNNN_inst.npy` 去掉 `_inst` 后，与 `sa_NNNNNNN.png/.json` 一一匹配，无缺失、无多余项。
- 范围：`sa_0004946` 至 `sa_0007552`；测试入口也按路径排序（`test.py:2521-2527`）。
- JSON 保留 19 类 `organ_type`，但所有实例只有 `category_id=1`，**没有五类核类别标签**。

数据准备脚本先加载原始五通道 mask，随后把通道 0–4 合并成二值前景，再转成轮廓多边形（`convertdata.py:68-100`）；因此当前转换 GT 不支持模型级五类 mPQ。结论：

```text
mPQ=NOT_APPLICABLE_NO_CLASSIFICATION_HEAD
five_class_labels_in_current_gt=false
```

## 2. 指标实现来源与移植说明

主实现移植自以下公开实现，并固定 commit 与源文件哈希：

- [HoVer-Net `metrics/stats_utils.py`](https://github.com/vqdang/hover_net/blob/67e2ce5e3f1a64a2ece77ad1c24233653a9e0901/metrics/stats_utils.py)
  - commit：`67e2ce5e3f1a64a2ece77ad1c24233653a9e0901`
  - SHA256：`34dd46f6ed9692a4c74ac723c73ebfd2f88397e4f7bad538b11257d6a17c0c68`
- [PanNuke-metrics](https://github.com/TIA-Lab/PanNuke-metrics/tree/c00014d766ca1be142b81bea19d9ef4315cde65a)
  - commit：`c00014d766ca1be142b81bea19d9ef4315cde65a`
  - `utils.py` SHA256：`53890787f039e98e1d2b64a5421de8b89aee42a9f6608a388dc2aa7dbc6044a4`
  - `run.py` SHA256：`506c50f6295a6d96f58ab574d9e23b682e4d896a0f12d36b1ee1576e93f5313e`

入口 `evaluation/recompute_from_npy.py` SHA256 为 `16dda201cc493f0dd9fe9027d6daaf035dc2f6f2c9d9e7d795fd7ad0b1985471`。本地移植 `evaluation/metrics_standard.py` SHA256 为 `6a050d7996924b4a2a947cadb4603d5bec5a56fcdafcf84949d680cb92601233`。对应代码位置：PQ `109-159`、Kumar AJI `162-193`、AJI+ `196-226`、Dice `229-238`、独立 PQ 交叉实现 `241-284`、全局聚合 `287-295`。

旧指标直接调用当前项目 `metrics.py`；其 SHA256 为 `5e78a57a248c9ab373407bcc28e932e0d1674f5b2c700eb31504eaec50d63181`。

## 3. 关键实现细节

1. **匹配阈值**：PQ 严格使用 `IoU > 0.5`，不是 `>=`（`metrics_standard.py:140-142`）；单元测试覆盖恰好 0.5 不匹配。
2. **空图**：PanNuke per-image/tissue bPQ 跳过 GT 空图；GT 非空而预测为空记 0。全局聚合不跳过，GT 空而有预测会贡献 FP。AJI/AJI+/Dice 的显式包装规则为双方空记 1、单方空记 0。本次转换 GT 2607/2607 均非空，因此 GT 空图规则不影响本次数值。
3. **平均方式**：
   - `bPQ (per-image avg)`：2607 张图的 PQ 算术平均；
   - `bPQ (global agg)`：先汇总 TP/FP/FN 和匹配 IoU，再计算一次；
   - 另算 `bPQ (PanNuke tissue macro)`：每种组织先做逐图均值，再对 19 种组织等权平均。这才与 PanNuke 官方文献表的聚合维度一致。
4. **实例重编号**：GT 与预测均在标准指标前重排为 `0,1,...,N`（`metrics_standard.py:62-74`）。原始预测不改写。
5. **边界实例**：不特殊排除；与当前 GT/官方 PQ 核心一致。
6. **最小面积**：指标侧不做面积过滤。GT 在数据转换阶段已过滤轮廓面积 `<10` 的对象并跳过空样本（`convertdata.py:28-32,94-101`）；测试的 `final_min_object_size=15` 是预测后处理侧过滤（`test.py:522-555,1756-1761`），不是评估侧过滤。

## 4. 交叉校验结果

### 4.1 旧指标复现

| 方法 | 指标 | 历史值 | 本次完整值 | 4 位小数 | 门禁 |
|---|---|---:|---:|---:|---|
| Visual baseline | `bPQ_img` | 0.6034 | 0.6034408644 | 0.6034 | PASS |
| Visual baseline | `AJI_custom` | 0.6270 | 0.6270132136 | 0.6270 | PASS |
| Exp5 | `bPQ_img` | 0.6094 | 0.6093660465 | 0.6094 | PASS |
| Exp5 | `AJI_custom` | 0.6361 | 0.6361007241 | 0.6361 | PASS |

### 4.2 双实现 bPQ

主实现保留官方 `SQ = matched_iou_sum / (TP + 1e-6)`；独立实现用列联表和精确 `TP` 分母。因此非零差值仅来自这个已记录的官方 epsilon 约定。

| 方法 | 2607 图最大绝对差 | 最大差样本 | `<1e-6` |
|---|---:|---|---|
| Visual baseline | `9.5633092134e-7` | `sa_0006196` | PASS |
| Exp5 | `9.1525332191e-7` | `sa_0006993` | PASS |

## 5. 主输出表

DQ/SQ、AJI、AJI+、Dice 在主表中均为逐图平均。

| 指标 | Visual baseline | Exp5 | Δ (Exp5−Visual) |
|---|---:|---:|---:|
| bPQ (per-image avg) | 0.603441 | 0.609366 | +0.005925 |
| bPQ (global agg) | 0.606379 | 0.611650 | +0.005271 |
| DQ (per-image avg) | 0.729711 | 0.736039 | +0.006329 |
| SQ (per-image avg) | 0.812073 | 0.816335 | +0.004263 |
| AJI (Kumar greedy) | 0.617242 | 0.627020 | +0.009778 |
| AJI+ (Hungarian IoU) | 0.626814 | 0.635956 | +0.009142 |
| Dice (binary) | 0.808852 | 0.817227 | +0.008375 |
| — 旧指标对照 — |  |  |  |
| bPQ_img（旧 mPQ） | 0.603441 | 0.609366 | +0.005925 |
| AJI_custom（旧 mAJI） | 0.627013 | 0.636101 | +0.009088 |

补充聚合：

| 指标 | Visual baseline | Exp5 | Δ |
|---|---:|---:|---:|
| DQ (global) | 0.734741 | 0.739864 | +0.005123 |
| SQ (global) | 0.825297 | 0.826705 | +0.001409 |
| bPQ (PanNuke tissue macro) | 0.621168 | 0.627487 | +0.006319 |

全局计数：Visual 为 TP=38,888、FP=12,470、FN=15,609；Exp5 为 TP=38,743、FP=11,490、FN=15,754。

## 6. 文献坐标定位与预注册判定

PanNuke 官方论文规定 bPQ/mPQ 先对每种组织计算，再对 19 种组织等权平均；官方代码进一步显示组织内是逐图平均。因此文献坐标必须使用本报告的 `bPQ (PanNuke tissue macro)`，不能使用 global agg。[PanNuke 论文与协议](https://arxiv.org/abs/2003.10778)、[官方指标代码](https://github.com/TIA-Lab/PanNuke-metrics/tree/c00014d766ca1be142b81bea19d9ef4315cde65a)。

| 方法 | PanNuke bPQ | PanNuke AJI | fold / 聚合 | TTA 等可比性信息 | 来源 |
|---|---:|---|---|---|---|
| HoVer-Net | 0.6596 | NOT_REPORTED | 官方三折平均；19 组织宏平均 | TTA=UNVERIFIED | [PanNuke benchmark](https://arxiv.org/abs/2003.10778)；[ECCV 表 1 复列](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| StarDist | 0.6692 | NOT_REPORTED | 官方三折平均；19 组织宏平均 | PromptNucSeg 表中对照；TTA=UNVERIFIED | [PromptNucSeg 表 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| CPP-Net | 0.6798 | NOT_REPORTED | 官方三折平均；19 组织宏平均 | TTA=UNVERIFIED | [PromptNucSeg 表 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| PointNu-Net | 0.6808 | NOT_REPORTED | 官方三折平均；19 组织宏平均 | TTA=UNVERIFIED | [PromptNucSeg 表 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| CellViT-H | 0.6793 ± 0.0318 | NOT_REPORTED | 官方三折 CV；19 组织宏平均 | 论文使用增强和定制采样；TTA=UNVERIFIED | [CellViT](https://arxiv.org/abs/2306.15350)；[ECCV 表 1 复列](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| PromptNucSeg-H | 0.6924 ± 0.0093 | NOT_REPORTED | 官方三折平均；19 组织宏平均 | 明确无 stain normalization、TTA、oversampling、辅助组织分类分支 | [ECCV 2024 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03925.pdf) |
| PromptNu (TMI 2025) | NOT_FOUND | NOT_FOUND | UNVERIFIED | 官方仓库的数据准备只列 CoNSeP/CPM17/MoNuSeg；未找到可核验的 PanNuke bPQ/AJI，不能用 Dice 代替 | [IEEE TMI 论文页](https://ieeexplore.ieee.org/document/11050438)、[官方代码](https://github.com/NucleiDet/PromptNu) |
| WeaveSeg (ICCV 2025) | NOT_APPLICABLE_NOT_EVALUATED_ON_PANNUKE | NOT_APPLICABLE | 不适用 | 原论文只评估 MoNuSeg、CoNSeP、CPM17 | [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/html/Li_WeaveSeg_Iterative_Contrast-weaving_and_Spectral_Feature-refining_for_Nuclei_Instance_Segmentation_ICCV_2025_paper.html) |

文献主流区间定义为 `[lo, hi] = [0.6596, 0.6924]`：取上述具备官方三折、19 组织宏平均且 bPQ 可核验的现代方法（HoVer-Net、StarDist、CPP-Net、PointNu-Net、CellViT-H、PromptNucSeg-H）的最小值与最大值；不纳入 `NOT_FOUND`、未评估 PanNuke 的方法，也不纳入早期 Mask R-CNN 低基线。

按预注册判据：

- Visual baseline tissue-macro bPQ=0.621168，低于 `lo` 0.038432（≥0.03）：**`BASELINE_WEAK`**。
- Exp5 tissue-macro bPQ=0.627487，低于 `lo` 0.032113；在同一坐标下仍低于主流区间。
- 若改用任务主表的逐图平均口径，Visual/Exp5 分别低于 `lo` 0.056159/0.050234，判定不变；但该口径不能与文献表直接比较。

**预注册提示：Exp5 的旧 `AJI_custom` 增量约 +0.0091 是在 `BASELINE_WEAK` 基线上取得的，其意义需要重新评估。** 这里只执行预注册坐标判定，不作额外方法论结论。

重要限制：本项目值来自一个经轮廓化、面积过滤并剔除空样本的 Fold 3 衍生集合，而文献值是完整官方三折平均。因此绝对数值不是严格同协议比较，坐标结论须附带此限制。

## 7. fold、阈值协议与 manifest

### 7.1 fold

`convertdata.py:11-15` 明确：训练使用 Fold 1 + Fold 2，测试使用 Fold 3。当前 2607 样本是该脚本输出的 Fold 3 衍生 test 集。风险项：

- 转换过程合并五类、轮廓化、过滤面积 `<10`、跳过空样本；因此不是原始 Fold 3 mask 的无损副本。
- 文件名使用跨 train/test 的全局递增 ID，不保留原始 Fold 3 数组索引。
- 远端未找到原始 `Fold 3/images.npy|masks.npy|types.npy`：`NOT_FOUND`。所以当前 2607 manifest 到官方原始 Fold 3 索引的逐项映射为 `UNVERIFIED`。

### 7.2 train-val / test 后处理协议错配

| 协议 | 实际代码值 | 代码位置 |
|---|---|---|
| train-val | `prob=0.45, marker=0.40, min_marker=10`；直接整图后处理，无 test 滑窗 | `train.py:5139-5147` |
| test | `prob=0.40, marker=0.45, min_marker=12, final_min_object=15` | `test.py:522-555,1682-1683,1756-1761` |
| test 推理几何 | 模型输入 `image_size=512`；空间窗口 `patch_size=256`、overlap=0.8；8× TTA | `test.py:561-563,788-848,1617-1619` |

`prob` 与 `marker` 阈值确实互换。所有历史 best checkpoint 在 train-val 协议下选择，而存盘 test 预测由另一协议产生；影响方向和幅度在“不重跑推理”的约束下为 `UNVERIFIED`。本任务未修改这些值。

### 7.3 测试集 manifest 与输出哈希

- `test_set_manifest.json`：2607 项，含排序索引、sample ID、图像文件及 SHA256、GT JSON 及 SHA256、organ type。
- manifest SHA256：`05fe7486608a42bdd1afa1f089e58f39a1017b23ed4fd841db5b30d5b6658d7d`。
- 两个 CSV 均为 2608 行（1 行表头 + 2607 样本）。
- `SHA256SUMS.txt` 已用 `sha256sum -c` 全部验证通过。

核心数据产物 SHA256：

| 文件 | SHA256 |
|---|---|
| `per_image_metrics_visual_baseline.csv` | `809f79f64ea19960bce9c832bd63ae8797df4d9fe41411f16d82eb31b2553f6a` |
| `per_image_metrics_exp5.csv` | `40afcbca88d7fba221df92f1968ba0703e4a4fe5b527b8affb900932f697a65c` |
| `test_set_manifest.json` | `05fe7486608a42bdd1afa1f089e58f39a1017b23ed4fd841db5b30d5b6658d7d` |
| `summary.json` | `7ab8bd1f5b679690c9fad87e5c87d929128310c831ef5bc787b9405d9aaa89e6` |
| `crosscheck.json` | `68dc293442e97eaabb92aad777546e185ed105ee45019c6886c7c82c1917210f` |
| `recompute_config.json` | `af8534d4c3741759a9f779a763d6a66e175f0b266636ab7fb201ad8adc8d47bd` |
| `input_file_manifest.json` | `dafaea92e8b25311edb1d9219a5b197f905a324d182966ada6a86df793463d1a` |

## 8. 对 P0.1 路线选择的输入

- 当前转换 GT **不支持五类 mPQ**：五通道原始类别已在转换时合并，实例 JSON 统一为 `category_id=1`。
- 数据侧增加分类头的前置条件：恢复原始 PanNuke Fold 1/2/3 的五通道 masks 与稳定原始索引，并重新生成保留类别的训练/测试标注及 manifest。
- 当前远端原始五通道 Fold 数据：`NOT_FOUND`；因此基于现有转换目录直接训练五类分类头不可行。恢复原始数据后的模型与工作量判断不属于本任务。
