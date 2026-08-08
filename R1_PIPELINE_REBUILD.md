# R1 PanNuke 实例数据管线重建

## 执行结果

- 状态：`PASS`
- `training_started=false`
- `inference_started=false`
- 执行日期：2026-08-08
- 远端代码 commit：`a7f120c2172671d690f70df9db1c4571f6f7262a`
- 新数据：`/hy-tmp/NuSeg/data/PanNuke_v2/`
- 新数据磁盘占用：1.3 GiB（7,901 个 `.npz`）
- 旧数据：`/hy-tmp/NuSeg/data/PanNuke/`（未改动）

新管线逐样本直接读取原始 instance mask，不轮廓化、不多边形化、不在构建层应用面积阈值。跨类别重叠严格按原始列表顺序（官方 channel 0→4、通道内 ID 升序）后写覆盖。

## 1. 数据格式与划分

每个 `.npz` 包含：

- `image`: `uint8[256,256,3]` RGB
- `inst_map`: `int32[256,256]`，背景 0，实例 ID `1..N`
- `type_map`: `uint8[256,256]`，背景 0，类别 `1..5`
- `inst_type`: `int32[N]`，与实例 ID 对齐
- `tissue_id`, `tissue_name`, `fold`, `orig_index`

Fold1+2 为 train，Fold3 为 test。文件名保留 fold 与原始行号，例如 `fold3/fold3_0000236.npz`。空样本原样保留。

## 2. Manifest 统计

| Fold | 样本 | 实例 | 空样本 |
|---|---:|---:|---:|
| 1 | 2,656 | 63,218 | 116 |
| 2 | 2,523 | 59,872 | 113 |
| 3 | 2,722 | 66,654 | 114 |
| 合计 | 7,901 | 189,744 | 343 |

五类实例分布：

| Fold | 类1 | 类2 | 类3 | 类4 | 类5 |
|---|---:|---:|---:|---:|---:|
| 1 | 26,201 | 10,820 | 16,388 | 967 | 8,842 |
| 2 | 22,731 | 10,631 | 16,756 | 884 | 8,870 |
| 3 | 28,471 | 10,825 | 17,441 | 1,057 | 8,860 |
| 合计 | 77,403 | 32,276 | 50,585 | 2,908 | 26,572 |

完整 tissue 分布、7,901 个数据文件 SHA、2,722 个 test 图像 SHA、旧 test 映射及新增 115 张清单均在 `dataset_manifest.json`。

## 3. 独立验证硬门

验证器重新从固定 Parquet 解码每一张图，并与写出的 NPZ 逐字段、逐像素比较。

| 检查 | 结果 |
|---|---|
| 样本数 2,656 / 2,523 / 2,722 | `PASS` |
| 实例数 63,218 / 59,872 / 66,654，总计 189,744 | `PASS` |
| 旧 test 重叠 2,607 张 | `PASS` |
| 旧 test 图像 `max_abs_diff=0` | `PASS` |
| 新增 test 115 张逐项列出 | `PASS` |
| 全局前景 IoU 精确 `1.0` | `PASS` |
| 每图实例数等于原始实例列表长度 | `PASS` |
| 五类分布与原始并行标签精确一致 | `PASS` |
| 每图非零 ID 连续为 `1..N` | `PASS` |
| 8 个可检查含洞实例的洞均保留 | `PASS` |

最终验证输出：

`/hy-tmp/NuSeg/workdir/audits/r1_pipeline_20260808_v2/`

第一次验证结果保留在 `r1_pipeline_20260808/`；它在加入 115 张 manifest 标记前已通过数据本体硬门，没有删除。

## 4. 兼容层

- 新类：`datasets/instance_dataset.py::InstanceNPZDataset`
- 开关：`--data_format {legacy_json,instance_npz}`
- 面积阈值：`--min_instance_area`，默认 `0`
- 面积过滤发生在训练读取时；原始 NPZ 永久保留所有实例
- `[DATA_CONFIG]` 输出格式、fold、样本数、实例数、manifest SHA256 和面积阈值
- 新 collate 仅额外处理变长的 `inst_type`；模型、损失、优化器与训练循环未改变

远端已完成一个完整 `InstanceNPZDataset.__getitem__` CPU smoke test：图像为 `[3,256,256]`、实例图为 `[1,256,256]`、实例数与 `inst_type` 长度一致，退出码 0。没有启动训练。

## 5. 哈希

- 数据 manifest：`0ce3ad621a88c58fe8982a8d2d2f9fd5e959fdb6ac305cd251eec8095075030a`
- 远端逐文件 `SHA256SUMS.txt`：`6b037410be41f8ddf989de9b1e0a3ded2bd9279ef1688c5912a6b98a03ebc741`
- 最终 verification summary：`b7ab00b4069da7fdee531af6cf664c18d601b5343dd148a3dd3ce70c86f6e2e0`
- 2,722 张 test 图像 SHA 清单：`444ae4c652da9e3e699b46f1739aab97d54bcba0f6242bbfd9521f822010e7d0`
- Fold1 source：`84428c1abae5015baf6b324f4927fe8558bbb6610137eb047a335aae7d040f25`
- Fold2 source：`a779daf86cd3ebd25e885e50ec131b7d05e53ad3a6ada21e387d4bc2f9d2b3d8`
- Fold3 source：`5684f09517e81ff18e570608a54741e4e6715a93cbe08e32dbec3d60513457a0`
- 旧 test 映射：`39847f31717690c9aaa9f1a84ff82a195b4ba3bc57cfe17653c8a72a21ecd4bb`

远端 `data/PanNuke_v2/SHA256SUMS.txt` 保存全部 7,901 个 NPZ 与 manifest 的逐文件 SHA256；提交侧 `SHA256SUMS.txt` 保存脚本、报告、CSV 与 manifest 的 SHA256。
