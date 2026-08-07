# NuSeg 项目完成度矩阵

评分：0=概念/缺失，1=代码雏形，2=可运行短程，3=训练链完整，4=正式单种子 full-test，5=多种子、匹配消融与统计闭环。`same pipeline` 仅表示当前证据能否确认同一评估流水线。

| 模块/阶段 | 目标 | 代码 | 默认启用 | 可训练参数/优化器 | checkpoint | 训练日志 | full-test | 指标 | same pipeline | 多种子 | 匹配消融 | 分数 | 等级 | 审计结论/首要阻塞 |
|---|---|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|---|
| Visual baseline | 纯视觉核实例分割 | 是 | 是/基线 | encoder adapter + decoder 路线可查 | 是 | 是 | 是 | `.8089/.696~/.6270/.6034` | 是 | 否 | 基线本身 | 4 | A- | 可复核正式基线；缺多种子 |
| FreqPath-none | 无 low/high 分支对照 | 是 | 依配置 | decoder 参数组 | 是/混合 | 部分 | 基线有 | 未形成专属表 | 不明 | 否 | 不足 | 2 | C | 未建立严格频率消融 |
| FreqPath-low | 低路径结构调制 | 是 | 依配置 | transposed conv/MLP/gate | 不明确 | 部分 | 无专属 | 无可信独立结果 | 否 | 否 | 不足 | 2 | C | “low”不是实频谱；缺匹配实验 |
| FreqPath-high | 高频细节路径 | 是 | 依配置 | CNN modulation/fusion | 不明确 | 部分 | 无专属 | 无可信独立结果 | 否 | 否 | 不足 | 2 | C | 缺专属 checkpoint/full-test |
| FreqPath-both | low/high 融合 | 是 | 依配置 | mask decoder 路由 | 混入阶段模型 | 有 | 无独立 | 归因不可能 | 不明 | 否 | 不足 | 2 | C | 需重命名并做 none/low/high/both |
| Phase B | 阶段性适配 | 是 | 历史阶段 | 可从训练代码推断 | 是 | 原始正式链不完整 | 无独立日志 | 历史声明 | 不明 | 否 | 否 | 3 | B- | 父节点存在，版本/原始日志断裂 |
| Phase C | 进一步语义阶段 | 是 | 历史阶段 | 可从训练代码推断 | 是 | 原始正式链不完整 | 无独立日志 | 历史声明 | 不明 | 否 | 否 | 3 | B- | 只宜作为谱系父节点 |
| Phase D | 后续阶段目标 | 零散/不清 | 否 | 未形成固定 protocol | 不明确 | 不足 | 无 | 无 | 否 | 否 | 否 | 1 | D | 定义、产物和验收均未闭环 |
| PromptNu-lite | 轻量文字辅助 | 是 | 依 flags | alignment/head 参数 | 是/混合 | 有若干 | 无独立 | 无单变量结果 | 否 | 否 | 不足 | 3 | B- | 工程完整度高于科学证据 |
| PromptNu-guided/PNuRL | 语义条件残差 | 是 | 否/依 flags | PNuRL adapter/head | 是（Exp6） | 是 | 无对应正式 | 文档声称未验证 | 否 | 否 | 否 | 3 | B | 推理路径真实；缺 canonical full-test |
| Multi-level numeric attrs | 多层级数值语义 | 是 | Exp5 开启 | 属性头/路由参数 | 是 | 是 | 是（组合） | `.8172/.7048/.6361/.6094` | 是 | 否 | 非单变量 | 4 | A- | 当前最佳整体配方，不能单独归因 |
| SGA | 结构引导 gamma 门控 | 是 | 依 flags | 显式 optimizer group | 是 | 是 | 否 | 5组 E5 validation | 是（组内） | 否 | 是（N0/S1/G1/G2/G3） | 3 | B | 真实执行，但 advance gate 失败 |
| PNuDP Dense | dense semantic bias 融合 | 是 | 否 | alpha/bias 路线可查 | 无正式 | smoke/短程 | 无 | 无正式 | 否 | 否 | 否 | 2 | C | 只能称辅助可行性 |
| L0 granularity audit | 量化 tile/region/instance 错配 | 审计脚本/产物 | N/A | N/A | N/A | N/A | N/A | 4946/101812 与异质性统计 | N/A | N/A | N/A | 4 | A- | 数据动机证据强，不是模型增益 |
| L1A local alignment | 局部窗口文字监督 | 是 | 训练 flag | alignment loss/head | 是 | 是 | 否 | C0/L1 E5 | 是（matched） | 否 | C0 vs L1 | 3 | B | 主门槛失败，且非推理期实例语义 |
| P4.1 soft boundary | soft 边界监督 | 是 | 实验 flag | boundary loss/相关参数 | 是 | 是 | 否 | 相对 G2 双降 | 是 | 否 | legacy vs soft | 3 | B | 明确 STOP，不建议追加 full-test |
| CLIP route | CLIP 文本编码 | 零散 | 否 | 不清 | 不足 | 不足 | 无 | 无 | 否 | 否 | 否 | 1 | D | Exp7 正式链缺失 |
| CONCH route | 病理视觉语言编码 | 是/Exp6 | 否/依 flags | encoder关联参数受配置控制 | 是（Exp6） | 是 | 无对应正式 | 声称值不可核验 | 否 | 否 | 否 | 3 | B | 可运行但效果不闭环 |
| dynamic_gt prompt | 由样本 GT 动态生成 prompt | 是 | 仅 train/oracle/debug | 数据侧，无独立 optimizer | 混入模型 | 有 | 正常 eval 禁用 | 不应作部署结果 | 否 | 否 | 否 | 2 | C | 防泄漏 guard 正确；不可用于常规测试 |
| dynamic_pred prompt | 预测驱动动态 prompt | 不完整 | 否 | 未形成端到端训练 | 无 | 无 | 无 | 无 | 否 | 否 | 否 | 1 | D | 实例/区域预测到文本的闭环缺失 |

## 总体分层

- 工程完成度较高：Visual baseline、Exp5 配方、SGA 执行链、CONCH/PNuRL 训练链。
- 科研证据中等：SGA、L1A、P4.1 有匹配短程对照，但都没有通过各自主推进门槛。
- 科研证据薄弱：FreqPath 独立贡献、PNuDP Dense、CLIP、dynamic_pred、Phase D。
- 论文级缺口：统一验证协议、固定父模型、三种子 canonical full-test、随机/打乱语义对照、统计不确定性和版本提交证据。

综合论文准备度为 **1/5**；工程实现约 **3/5**，数据审计约 **4/5**，正式基线/Exp5 复现证据约 **4/5**，主创新因果与多种子证据分别约 **1/5** 与 **0/5**。
