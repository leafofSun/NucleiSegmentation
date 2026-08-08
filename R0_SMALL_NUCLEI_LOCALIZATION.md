# R0 小核丢失定位

## 执行边界

- 状态：`COMPLETE_DEGRADED_RAW_NOT_AVAILABLE`
- `training_started=false`
- `inference_started=false`
- `postprocessing_started=false`
- 执行日期：2026-08-08
- 远端代码 commit：`a7f120c2172671d690f70df9db1c4571f6f7262a`

本诊断仅使用固定版本 Parquet、当前训练 GT、D1 映射和已存最终实例图。没有启动训练、模型推理或后处理重算。

## 1. Raw 输出盘点

在 `workdir/audits/p0_20260713/` 与 `workdir/runs/` 中共找到 5,214 个张量文件，全部是最终 `*_inst.npy`：

- `visual_baseline`：2,607
- `exp5_best_pq`：2,607
- `.npz/.pt/.pth`：0
- probability / logits / heatmap / marker / HV raw：0

结论为 `RAW_NOT_AVAILABLE`。因此没有生成 `postproc_sweep_results.csv`，`prob_mean/prob_max/marker_max` 均为 `UNVERIFIED`。

## 2. 环节 A：训练信号缺失

Fold1+2 原始数据 5,179 张、123,090 个实例；当前训练数据 4,946 张均通过 RGB SHA256 映射回原始数据，逐像素 `max_abs_diff=0`。原始实例与当前训练实例采用最大 IoU 一对一匹配，只要交集大于 0 即计为保留。

| 原始实例面积 | 原始数 | 当前 GT 数（按当前面积） | 一对一保留数 | 一对一保留率 |
|---|---:|---:|---:|---:|
| `[1,10)` | 1,596 | 0 | 0 | 0.0000% |
| `[10,20)` | 1,384 | 61 | 65 | 4.6965% |
| `[20,50)` | 3,686 | 2,915 | 2,930 | 79.4900% |
| `[50,100)` | 7,241 | 5,664 | 5,860 | 80.9280% |
| `[100,200)` | 17,253 | 13,434 | 14,038 | 81.3656% |
| `[200,+∞)` | 91,930 | 79,738 | 78,915 | 85.8425% |

Fold1+2 有 233 张原始样本未进入旧训练集，其中 4 张为非空样本。`[1,10)` 完全丢失、`[10,20)` 仅保留 4.70%，确认训练目标系统性缺少最小实例。

## 3. 环节 B/C 的降级观测

以下统计覆盖 Fold3 中与旧 test 精确对应且有保存预测的 2,607 张；新增 115 张没有历史预测，未纳入分母。独立实例定义为与该 GT 的最佳预测实例 IoU > 0.5。

| 原始 GT 面积 | GT 数 | 任意预测像素覆盖率 | 独立实例率 | GT 像素覆盖率 |
|---|---:|---:|---:|---:|
| `[10,50)` | 2,564 | 19.3838% | 0.2340% | 17.3809% |
| `[50,100)` | 3,882 | 39.5415% | 12.3648% | 36.4045% |
| `[200,+∞)` | 49,996 | 93.1535% | 73.2079% | 86.3479% |

最终实例图显示小核位置的覆盖与独立实例形成率都远低于大核。不过最终图位于模型响应与后处理之后，无法用它区分“模型未响应”和“响应被后处理删除”。

## 4. 预注册归因

- 环节 A（训练信号缺失）：`CONFIRMED`。证据是 Fold1+2 `[1,10)` 保留率 0、`[10,20)` 保留率 4.6965%。
- 环节 B（模型未检出）：`UNVERIFIED`。最终图的小核覆盖率很低，但 raw probability/marker 未保存。
- 环节 C（后处理删除）：`UNVERIFIED`。同一原因，无法执行固定的 5×4 参数扫描。
- 仅调后处理可回收 bPQ：`UNVERIFIED`。
- 需重训才能回收的部分：`UNVERIFIED`。

## 5. 可复现性与哈希

输入：

- Fold1 Parquet：`84428c1abae5015baf6b324f4927fe8558bbb6610137eb047a335aae7d040f25`
- Fold2 Parquet：`a779daf86cd3ebd25e885e50ec131b7d05e53ad3a6ada21e387d4bc2f9d2b3d8`
- Fold3 Parquet：`5684f09517e81ff18e570608a54741e4e6715a93cbe08e32dbec3d60513457a0`
- Fold3 映射：`39847f31717690c9aaa9f1a84ff82a195b4ba3bc57cfe17653c8a72a21ecd4bb`
- 当前训练 PNG+JSON manifest：`cb5f763399adbbf925e44e5b0b51acd33a1c26f62164a24c9949e76941fbc70a`
- Visual 最终预测 manifest：`00b99be6fc116ab1199e093008d8235dd189e0503fd6aa8c2a93426780bbb70b`

提交的表：

- `train_gt_size_distribution.csv`：`52aa3fb889d283e48542be5c7215ca26c8817f0ab6c8cdf32b5844e6bafe9871`
- `small_nuclei_response_stats.csv`：`999490da3bcc533bd741f6ba0e16384634065e6291d1f47f16e0d0f675b6254d`

远端完整逐实例覆盖表与审计输出位于：

`/hy-tmp/NuSeg/workdir/audits/r0_small_nuclei_20260808/`
