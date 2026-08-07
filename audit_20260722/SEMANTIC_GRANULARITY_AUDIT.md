# NuSeg 语义粒度专项审计

## 核心结论

项目的数据证据充分支持一个问题：同一 256×256 tile 内的核在类别、大小、拥挤度、边界不规则性和伸长度上存在显著异质性，而当前主 prompt 多为 tile/organ 级单个全局向量。因此全局语义不能可靠代表所有局部核。可是现有 L1A 仅在训练时用 GT mask/instance ID 抽取窗口做只读对齐，未把区域语义注入推理路径，也没有通过预注册 E5 性能门槛。故“粒度错配存在”成立，“现有局部方案解决了它”不成立。

## 四级粒度图

| 粒度 | 当前语义来源 | 空间单位 | 进入模型的方式 | 训练/推理可用性 | 主要风险 | 完成度 |
|---|---|---|---|---|---|---:|
| 数据集/器官级 | organ_static prior、train-split organ prior | organ/domain | 全局 prompt/先验 | train/val/test 可用；eval 避免 GT 动态泄漏 | 过粗，可能编码数据集偏差 | 3/5 |
| tile/sample 级 | 每 tile JSON、global SB vector、数值属性 | 256×256 tile（再 resize 512） | 一个共享向量/多级属性路由 | train/eval 可用 | tile 内所有实例共享描述，异质性被平均 | 4/5 工程，2/5 语义充分性 |
| region/window 级 | L1A 从增强后 GT mask 选择 4 个 192 窗口 | 局部窗口 | 训练期 alignment loss；feature injection=false | 仅训练；eval 有 GT guard | 监督与部署断裂，窗口仍含多个实例 | 3/5 工程，1/5 推理闭环 |
| instance 级 | 增强后重算的 GT instance attrs | 单核 | 属性监督/分析；无完整预测到文本路径 | 训练可用，正常推理不可用 | GT 依赖、匹配/成本、邻近实例混淆 | 2/5 |

## 数据证据

L0 审计覆盖 4946 个训练 tile 和 101812 个实例，推荐局部窗口为 192。关键统计包括：全局—局部类别不一致密度约 `.265`，大小异质性约 `.363`，拥挤异质性约 `.275`，边界不规则性约 `.313`，伸长度异质性约 `.289`。这些量支持“单一 tile 描述信息不足”的动机。

但这些统计不能直接证明：

- 文本或视觉语言模型一定优于普通区域特征；
- 192 是性能最优窗口，而非数据审计中的折中建议；
- GT 导出的局部属性可在部署时无损获得；
- 更多语义参数就是提升来源。

## 数据与代码链核对

1. PanNuke 训练以 tile 为单位，基础尺寸 256×256；训练采用 256 crop、翻转/旋转等几何增强，再 resize 至 512。
2. 实例属性在几何增强后根据 mask 重算，避免直接沿用增强前位置/形状，这是正确的。
3. 正常 val/test 不允许 dynamic GT prompt；相应配置回退至 organ_static/train-split prior，降低了显式标签泄漏风险。
4. tile 级 Prompt/global SB JSON 对局部区域共享同一语义向量，不能被称为实例级语义。
5. L1A 在后增强 mask 上构建四个局部窗口，在 32×32 feature map 上做 ROI/对齐；其 feature injection 为 false，且 eval 有 GT guard。
6. 当前没有完整的“预测实例/区域 → 生成或检索对应文本 → 空间匹配 → 注入 decoder → 输出实例”的部署路径。

## L1A 结果解释

匹配的 2-GPU、5-epoch screening 中：

| 组 | E5 mAJI | E5 mPQ | 相对 C0 |
|---|---:|---:|---|
| C0 | .628076 | .535493 | control |
| L1 | .627075 | .536647 | mAJI `-.001001`，mPQ `+.001154` |

预注册主门槛要求 E5 mAJI 至少 `+.003`，因此失败。E4–E5 均值约有 mAJI `+.001895`、mPQ `+.003535` 的弱正信号，可作为重新设计依据，但不能覆盖主判据，更不能被报告为稳定提升。两组指标还受 rank-local、默认部分验证和单种子限制。

## 泄漏与因果风险

- `dynamic_gt` 只能用于训练、oracle 或 debug；如果用于正式测试，会把标签信息带入 prompt。
- 训练期由 GT instance 产生的区域选择，若推理期没有等价预测器，会形成 train–test semantic gap。
- 正确文本若没有与 shuffled-region、frozen-random-prototype 比较，就无法排除额外损失/参数的普通正则化效应。
- 用 organ prior 评估可能利用数据集构成信息；需要跨器官或 leave-one-organ-out 分析其依赖程度。
- 同一窗口含多个密集实例时，region text 与单实例目标之间仍可能错配。

## 建议的闭环设计

推荐把目标限定为 **区域条件语义残差**：先由无 GT 的视觉提议或固定网格产生区域 token，再从冻结文本原型库检索/组合条件，利用有界 residual 注入 decoder。保持实例属性只作训练监督或分析，不在正式测试读取 GT。

必须包含四组：

1. no-local：无局部语义；
2. correct-local：区域与其语义正确对应；
3. shuffled-region：在 batch/器官内打乱区域—语义对应；
4. frozen-random-prototype：相同参数量与注入幅值，但原型无语义。

只有 correct-local 稳定优于三组，才支持“语义内容和空间粒度”而不是容量/正则化。筛选阶段可用 5 epoch，但正式结论必须回到统一 canonical full-test 和至少 3 个种子。

## 裁决

- 问题真实性：**SUPPORTED**。
- 当前 tile/global prompt 充分性：**CONTRADICTED**。
- L1A 工程实现：**SUPPORTED**。
- L1A 已解决粒度错配并提高性能：**CONTRADICTED**。
- 实例级部署闭环：**UNVERIFIED/INCOMPLETE**。
- 路线建议：保留粒度问题，重构方法；不要继续把训练期 GT 窗口对齐直接称为实例级语义推理。
