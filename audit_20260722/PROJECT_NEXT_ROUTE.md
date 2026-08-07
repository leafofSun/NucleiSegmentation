# NuSeg 下一路线与验收协议

## 路线裁决

`FINAL_DECISION=RESTRUCTURE`

主路线选择 **B：局部/区域语义粒度闭环**。路线 A（现有 SGA/FreqPath/soft-boundary）冻结为工程对照与负结果，不追加堆叠；路线 C（完整实例级动态文本）暂缓，直到区域级方案证明语义内容有效且能无 GT 推理。

## 唯一下一实验

名称建议：`RSGR-1 Region-Semantic Grounding Randomization`。

固定项：

- 固定 Exp5 的可核查父 checkpoint，不切换 Phase C/Exp6 父节点；
- 固定 PanNuke split、数据增强、epoch、batch、optimizer、学习率、trainable parameter list；
- 固定相同 decoder、SGA/FreqPath 状态，不同时引入新边界或频率模块；
- 修复并冻结验证协议：所有 rank 的 sum/count 汇总，明确是否全量验证；
- 固定 full-test 阈值、最小对象、滑窗 overlap 和 sampler 去重规则；
- 保留每次命令、环境、配置、随机种子、checkpoint 与机器可读 metrics。

唯一变量是区域语义条件：

| 组 | 条件 | 目的 |
|---|---|---|
| C0 no-local | 禁用局部语义残差 | 匹配父路线基线 |
| C1 correct-local | 区域与语义正确对应 | 测试目标机制 |
| C2 shuffled-region | 器官/批次内打乱对应，幅值和参数量不变 | 排除普通正则化和先验效应 |
| C3 random-prototype | 冻结随机原型，幅值和参数量不变 | 排除容量与噪声注入效应 |

推荐实现边界：区域由固定网格或无 GT 的视觉 proposal 产生；文本原型库冻结；注入采用已有有界 PNuRL residual，但记录 BaseNorm、InjectedNorm、真实 injected/base ratio。不得在正式 val/test 使用 GT mask、GT instance attr 或 dynamic_gt prompt。

## 两阶段执行与门槛

### Stage 1：单种子筛选

- 四组、同一 seed、5 epoch；预注册以 E5 为主，E4–E5 均值仅作稳定性辅证。
- ADVANCE：C1 相对 C0 的 mAJI `>= +.003`，mPQ 下降不超过 `.002`；且 C1 在 mAJI 上同时高于 C2、C3 至少 `.002`。
- HOLD：方向为正但未达阈值，或 E5 与末两 epoch 方向冲突；只允许一次诊断，不允许连续调参追 best。
- STOP：C1 不优于 C0，或 C2/C3 与 C1 等效，或收益仅来自验证而 canonical full-test 消失。

### Stage 2：多种子正式验证

- 仅在 Stage 1 ADVANCE 后运行 C0–C3 的至少 3 个预注册种子。
- 每个种子均跑同一 canonical full-test，报告全部种子，不只报告 best checkpoint。
- 最终 ADVANCE：C1 对 C0 的三种子平均 mAJI `>= +.003`、平均 mPQ 下降 `<= .002`，且 C1 显著/一致优于 C2 与 C3；至少 2/3 种子方向一致。
- 同时报告 Dice、IoU、mAJI、mPQ 的均值、标准差或 bootstrap CI，以及器官分层结果。

## 资源估计

- 筛选：4 组 × 5 epoch，建议 `2 × 24 GB` GPU；依据现存日志，单卡/每进程峰值约 16–19 GB，预计合计约 7–10 GPU 小时。
- 正式：4 组 × 3 seeds，预算按筛选实测吞吐线性估计；预留 checkpoint、日志和预测结果空间，不与其他训练并行争用显存。
- CPU/内存：数据审计与指标汇总可 CPU 执行；训练不建议 CPU。
- 本次审计没有启动上述资源，资源值是计划估计而非已执行记录。

## 实验前必须修复的审计阻塞

1. 将 validation 改为所有 rank 的严格 sum/count all-reduce，并明确全量或固定子集；禁止 rank0 本地均值选模。
2. 统一 train validation 与 canonical full-test 的阈值、最小对象和滑窗配置，或明确二者用途并禁止跨协议比较。
3. 为 DistributedSampler 的尾部重复样本做去重或使用可证明无重复的汇总。
4. 恢复 Git 版本元数据；每个实验记录 commit、dirty 状态、配置快照和源码 diff。
5. 把 FreqPath 更名为 low/high feature path，除非真正加入 FFT/DCT/频谱操作。
6. 明确定义 DeltaRatio 与 InjectedRatio；论文主表应使用真实注入范数比，不把 adapter 内部预测 RMS 比例混为实际注入量。

## 明确停止项

- 不追加 P4.1 soft-boundary full-test；其相对 legacy G2 的 E5 mAJI/mPQ 均下降。
- 不把 SGA G2 升级成正式主线；它未同时超过 N0 的 mAJI 与 mPQ。
- 不继续堆叠 CLIP/CONCH/PNuDP/新 loss 到同一实验；这会扩大归因混杂。
- 不把 Exp6 文档中的无日志指标作为论文表格正式结果。
- 不以 best epoch、单种子或 rank-local validation 宣称统计显著性。

## 预期决策

若 RSGR-1 通过，项目可围绕“密集核场景的区域语义正确性与有界残差注入”重写主线；若 correct-local 不胜 shuffled/random，应接受文本内容没有被证明有效，回退到 Exp5 视觉+数值属性配方，停止视觉语言主创新路线。
