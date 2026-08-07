# NuSeg 实验谱系审计

审计日期：2026-07-22。谱系只采用现存代码、命令日志、指标历史和 checkpoint 文件名；由于 `.git` 缺失，任何“代码版本一致”均无法被提交哈希证明。

## 可信度规则

- A：训练命令/配置、checkpoint、canonical full-test 命令与结果齐全。
- B：训练日志、指标历史和 checkpoint 基本齐全，但缺 canonical full-test 或存在验证协议风险。
- C：仅 smoke/短程/辅助轨迹，或关键产物缺失。
- U：只有文档陈述，无法由当前工作区独立复核。

## 主谱系

| 节点 | 推定父节点 | 核心变化 | 训练证据 | checkpoint | canonical full-test | 当前结论 | 等级 |
|---|---|---|---|---|---|---|---|
| Visual baseline | SAM/视觉初始化 | 纯视觉实例分割基线 | 有历史训练/manifest 线索 | 有 | 有，`.8089/.696~/.6270/.6034` | 可作为现有正式基线 | A |
| Phase B | Visual baseline | 项目阶段性适配 | 历史指标/元数据存在 | 有 | 未发现独立正式日志 | 只能作为谱系父节点 | B-/U |
| Phase C | Phase B | 进一步阶段训练 | 历史指标/元数据存在 | 有 | 未发现独立正式日志 | 只能作为谱系父节点 | B-/U |
| Exp5 numeric route | Phase B（日志声明） | multi-level attrs + numeric route；无 CONCH、无 PNuRL | 完整训练日志，best AJI epoch 8 | 有 | 有，`.8172/.7048/.6361/.6094` | 当前最高可信整体配方 | A |
| Exp6 CONCH/PNuRL | Phase C（日志声明） | CONCH + PNuRL + attr alignment + PG3/text bank | 10 epoch 训练证据 | 有 | 未发现 | 文档声称结果不可升级为正式结论 | B |
| Exp7 CLIP | 不明 | CLIP 语义路线 | 不足 | 未形成可核查闭环 | 无 | 未验证 | U/C |

上述四元组均按 Dice/IoU/mAJI/mPQ 排列。Exp5 相对视觉基线约为 `+.0083/+.008~ /+.0091/+.0060`，但父 checkpoint、训练时长、启用参数与模块组合混杂，只能支持“整体配方有效”，不能支持某一模块的独立因果贡献。

## 结构—边界分支

| 节点 | 父节点/对照 | 实验范围 | E5 mAJI | E5 mPQ | 因果判断 | 去向 |
|---|---|---:|---:|---:|---|---|
| SGA N0 | matched control | 5 epoch validation | .635921 | .549545 | 对照 | 保留作 frozen control |
| SGA S1 | N0 | 单组门控 | .634983 | .546496 | 双指标不优 | HOLD |
| SGA G1 | N0 | gamma group 1 | .633129 | .544487 | 双指标不优 | STOP |
| SGA G2 | N0 | gamma group 2 | .636847 | .547004 | mAJI +.000926，mPQ -.002541 | HOLD，不进 full-test |
| SGA G3 | N0 | gamma group 3 | .631037 | .542174 | 双指标不优 | STOP |
| P4.1 soft boundary | legacy G2 | 5 epoch validation | legacy-relative -.006571 | legacy-relative -.005762 | E5 与末两 epoch 均无推进依据 | STOP |

SGA 的工程链条是真实的：参数在 forward 中参与 gamma 缩放，进入 mask decoder，且有明确 optimizer 参数组与 checkpoint。但五组均为单种子、短程、验证集结果；“真实执行”不能替代“性能贡献成立”。

## 语义与粒度分支

| 节点 | 父节点/对照 | 变化 | 证据 | 结果 | 判断 |
|---|---|---|---|---|---|
| PromptNu-lite | 视觉/阶段 checkpoint | tile/global prompt 辅助路线 | 代码与若干训练链 | 独立 canonical 因果结果不足 | 工程存在，贡献未闭环 |
| PromptNu-guided / PNuRL | Phase C/Exp6 | 条件语义 delta 注入 | 代码、训练日志、checkpoint | 无对应 formal full-test | 机制真实，效果未验证 |
| Multi-level numeric attrs | Phase B/Exp5 | 多级属性与数值路由 | 完整训练 + full-test | 当前最好可信整体配方 | 保留为下一轮固定父路线 |
| L0 | 数据本身 | 全局—局部语义错配审计 | 4946 tiles、101812 instances | 多项异质性/不一致率约 .265–.363 | 动机成立 |
| L1A C0 | matched control | 无 local-region alignment | 2-GPU、5 epoch validation | E5 `.628076/.535493` | 对照 |
| L1A L1 | C0 | 4 个 192 窗口、训练期只读对齐 | 2-GPU、5 epoch validation | E5 `.627075/.536647` | 预注册 mAJI 门槛失败 |
| PNuDP Dense | 语义分支 | dense bias 融合 | smoke、20-batch、1-epoch | 无正式 checkpoint/full-test | 辅助可行性，不得宣称有效 |

L1A 的 E4–E5 均值相对 C0 有弱正信号（约 mAJI `+.001895`、mPQ `+.003535`），但主判据是 E5 mAJI 至少 `+.003`，实际为 `-.001001`。此外它依赖 GT mask/instance ID 生成窗口监督，推理阶段没有对应实例级文本获取链，因此不能被包装为完整的实例级推理方法。

## 可比性断点

1. 工作区没有 Git 元数据，无法证明任意两次实验的代码提交相同。
2. `train.py` 的验证默认只取每个 rank 的约 40%，且没有跨 rank 聚合；rank0 本地均值用于选模。
3. 训练验证与 full-test 的阈值、最小对象参数和滑窗设置不同。
4. Exp5 与 Exp6 的父 checkpoint、模块组合和优化参数不同，不构成单变量消融。
5. 核心结果只有单种子，现存 `best` checkpoint 会放大选择偏差。
6. Phase B/C 的原始正式日志不完整，谱系边只能按历史元数据标为“推定”。

## 最小可复现实验谱系

下一轮应冻结 Exp5 的父 checkpoint、数据 split、训练 epoch、优化器、后处理与 canonical full-test 命令，只改变局部语义条件：`no-local / correct-local / shuffled-region / frozen-random-prototype`。筛选通过后用至少 3 个预注册种子重跑，并保留每个种子的训练命令、环境、checkpoint、full-test 日志和机器可读指标。
