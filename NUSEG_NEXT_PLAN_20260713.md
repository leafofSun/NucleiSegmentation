# NuSeg 下一阶段计划（2026-07-13）

## 总原则

- 主要指标：mAJI、mPQ；次要指标：Dice、IoU。
- 不以单次最高 validation 宣称提升；论文级结论至少 3 seeds。
- 每一步只改变一个因果变量；固定代码快照、父 checkpoint、数据划分、后处理、test 命令与 checkpoint selection 规则。
- 当前不引入 FFT/DCT、LLM、新数据集或其他复杂模块。
- 本文件是计划，不包含训练命令；本轮不自动开始。

## P0：固定代码、checkpoint 与 test pipeline

- 研究问题：现有 baseline/Exp5/Exp6/Exp7 数值是否来自同一评估协议？
- 唯一变化变量：无；只固定和复核资产。
- 父 checkpoint：分别记录当前四条谱系的精确 SHA-256；Exp6 明确选择 CONCH `.clip_bak` 或其他原始文件，禁止使用被 CLIP bank 覆盖的模糊文件名。
- 冻结/训练：全部冻结；不训练。
- 观察指标：代码哈希、checkpoint 哈希、args、test flags、数据 split、后处理、每张图输出数量、mAJI/mPQ/Dice/IoU 原始结果文件。
- 成功条件：每个固定数值均可追溯到唯一 checkpoint + 唯一 test 配置 + 原始结果。
- 失败停止条件：任一结果无法追溯，立即降级为 `PROVIDED_CONTEXT_UNVERIFIED`，不得进入跨实验结论。
- 是否进入下一步：只有 Exp5 与 baseline 的 pipeline 可复现一致后进入 P1。

## P1：SGA-SB 静态接线与 target 可视化

- 研究问题：corrected SGA-SB 是否从 CLI 到模型、stage policy、optimizer、loss、checkpoint save/load、test 全链路一致？
- 唯一变化变量：仅修复/确认 SGA-SB 接线，不改变模型数学设计。
- 父 checkpoint：P0 固定的 Exp5 或统一 visual/Phase-B 父 checkpoint；此阶段不实际训练。
- 冻结/训练：不训练；只核对预期 future trainable 集合。
- 观察指标：constructor 参数、mode/branch、head/adapter/gamma 参数名、target shape/dtype/range、旧模块默认关闭、eval GT leakage guard。
- 成功条件：`none` 不创建新模块；supervision/guidance 按 branch 创建；预期参数会被 stage policy 解冻并被 optimizer 收集；test 接受同一配置；target 可视化覆盖稀疏/密集/空前景边界情况。
- 失败停止条件：任何参数被 parser 接收但未传递，或参数不在 optimizer，或 eval 能消费 GT map。
- 是否进入下一步：全部静态检查通过才进入 P2。

## P2：2-batch forward、loss、gradient 与 norm audit

- 研究问题：新分支是否真的产生有限 loss、梯度和参数更新，且 guidance delta 非退化？
- 唯一变化变量：mode/branch 固定为一个待测组合；仅运行最小 2-batch 审计。
- 父 checkpoint：P0/P1 固定的同一个父 checkpoint。
- 冻结/训练：仅解冻计划中的 segmentation 模块与对应 SGA head/adapter/gamma；其他模块按统一 policy 冻结。
- 观察指标：structure/boundary loss、head/adapter/gamma grad norm、optimizer membership、step delta、structure/boundary delta norm、注入前后 feature norm、NaN/Inf。
- 成功条件：所有启用参数在 optimizer；loss 有限；梯度和 step delta 非零；`supervision_only` feature delta 严格为 0；guidance delta 小而非零。
- 失败停止条件：任一启用参数无梯度/无更新、adapter 永久零、gamma 无更新、出现 GT leakage/NaN。
- 是否进入下一步：通过后才允许短程消融。

## P3：none / supervision_only / structure / boundary / both 短程消融

- 研究问题：收益来自辅助监督还是 feature guidance；结构和边界各自贡献什么？
- 唯一变化变量：`spatial_sb_mode` 与 `spatial_sb_branch`；其余完全固定。
- 父 checkpoint：同一个经哈希固定的 checkpoint。
- 冻结/训练：每组相同；仅依据 branch 决定对应 head/adapter/gamma 是否存在和训练。
- 观察指标：validation mAJI/mPQ 为主，Dice/IoU 为辅；loss、delta norm、gamma trajectory；同一预设 epoch 的结果，不取组内任意最高点比较。
- 成功条件：both 至少相对 supervision_only 在 mAJI/mPQ 上方向一致；structure/boundary 的作用与 low/high 路径诊断一致。
- 失败停止条件：both 不优于 supervision_only，或差异只出现在 Dice/IoU，或结果对 checkpoint selection 高度敏感。
- 是否进入下一步：只有 guidance 对 supervision_only 显示可信信号才进入 P4。

## P4：完整训练、full test 与多随机种子

- 研究问题：SGA-SB 增益是否稳定且具有论文级效应量？
- 唯一变化变量：最终选定的 SGA-SB 配置相对统一 baseline/supervision_only。
- 父 checkpoint：P3 使用的同一父 checkpoint。
- 冻结/训练：严格沿用 P3 的 trainable policy；不得临时改变 optimizer、epoch 或初始化。
- 观察指标：3 seeds 的 full-test mAJI/mPQ mean±std；Dice/IoU；训练稳定性、参数 norm、失败率。
- 成功条件：mAJI 和 mPQ 均稳定改善，且提升超过 seed 方差；结论不依赖单一 best epoch。
- 失败停止条件：任一主指标不稳定、均值不增或方差覆盖效应量。
- 是否进入下一步：主线成立后才评估文本语义。

## P5：CONCH 因果消融

- 研究问题：真实文本语义是否提供超越数值/no-text 路线的因果增益？
- 唯一变化变量：no-text / real prompt / shuffled prompt / uniform prompt；同一 encoder 与同一 text-bank 形状约束。
- 父 checkpoint：P4 最佳但未接触 test 的统一 checkpoint，或预先固定的同一 Phase C checkpoint；所有组一致。
- 冻结/训练：完全相同模块和 optimizer；只改变 prompt 语义内容。
- 观察指标：3 seeds full-test mAJI/mPQ、real-vs-shuffled、real-vs-uniform、semantic delta/injection ratio。
- 成功条件：real prompt 在两项主指标上稳定超过 no-text、shuffled 和 uniform。
- 失败停止条件：real 不稳定超过 no-text，或 shuffled/uniform 与 real 等价。
- 是否进入下一步：失败则将 CONCH 从主贡献移除；成功才保留为扩展贡献。

## P6：PNuDP Dense 附加实验

- 研究问题：dense text-logit bias 是否在既定主线之上提供额外实例分割收益？
- 唯一变化变量：PNuDP off/on；channel-specific 只作为第二层消融。
- 父 checkpoint：P4/P5 已固定的最终主线 checkpoint。
- 冻结/训练：先仅训练 PNuDP projection/logit projection/alpha；主线冻结；若设计另有需要必须单列实验。
- 观察指标：3 seeds mAJI/mPQ、mask-channel bias norm、alpha、base-vs-fused logits 差异、计算/显存代价。
- 成功条件：两项主指标稳定增益且不是单一 channel 或后处理偶然性。
- 失败停止条件：无稳定增益、alpha 回到零、logit bias 退化或代价不成比例。
- 是否进入下一步：无稳定增益则只保留为负结果或附录，不再扩展。

## 最高优先级单一实验

在任何训练前，优先完成 **P0 的 Exp5 vs visual baseline 同 checkpoint/test pipeline 复核**。在当前 SGA-SB 训练入口断路且固定 full-test 数值无原始结果文件的状态下，直接启动 SGA-SB 消融无法形成可信基准。

