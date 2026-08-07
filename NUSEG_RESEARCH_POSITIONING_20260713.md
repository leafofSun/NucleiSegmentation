# NuSeg 研究定位（2026-07-13）

## 1. 论文证据边界

远程项目中未发现 WeaveSeg、PromptNu、PromptNucSeg 或 APSeg 的 PDF/正式论文副本，也没有可核验的书目表。现有代码注释和历史 Markdown 只能证明“项目作者声称受到启发”，不能替代论文原文核对。因此以下映射只基于已有项目材料与当前实现，不把未提供论文中的细节当作事实。

## 2. 论文启发—项目实现—差异

| 来源线索 | 项目中的对应实现 | 关键差异/边界 |
|---|---|---|
| WeaveSeg：adaptive spectral refinement、结构/细节解耦、边界细化 | `FreqPathASRBlock` 的 structure upsample 与 CNN detail path；HV/heatmap/边界相关监督；计划中的 structure-to-low / boundary-to-high | 当前“frequency-aware”是结构/细节功能分路及可学习调制，**不是 FFT/DCT/Fourier 频率分解**；未发现 uncertainty 与 SGA-SB 的直接耦合证据 |
| PromptNu：核知识 prompt、视觉语言 alignment、属性指导实例识别 | Phase B 属性头、Phase C attr-text projection、PromptNu-lite、PG3、PNuDP Dense | 项目实现是自定义阶段式属性与 text-bank 路由；不能声称复现 PromptNu；Exp5 反而是不依赖文本的数值属性路线 |
| CONCH：病理视觉语言 encoder | 生成/缓存 structure 与 boundary text bank；Phase C/PG3 的 512-d text semantic | CONCH 是外部 encoder，不是本项目创新；目前无因果证据证明优于 no-text 或 shuffled prompt |
| PromptNucSeg/APSeg：自动 prompt、密度/分布知识、类别/形态语义 | 数据中的 density/population/shape 属性、自动 prompt 相关历史脚本、属性路由 | 未发现严格复现或单独验证；只能作为问题设定启发，不能写成已实现同等机制 |

## 3. 候选贡献分级

### 已有证据支持

- **FreqPath 功能性 structure/low 与 morphology/high 路由**：代码、训练日志和 Exp5 checkpoint 支持“路径真实启用并参与训练”。
- **无文本数值属性路由可以完整训练**：Exp5 的 CONCH/CLIP/text bank 均关闭，numeric projection 进入 optimizer。
- **阶段式训练基础设施存在**：Phase A/B/C checkpoint 谱系及各阶段 trainable policy 可核验。

注意：以上不等于已有多随机种子论文级增益证据。

### 合理但尚待验证

- **spatial granularity alignment**：dense occupancy 与 instance boundary target 合理，但训练入口未接通。
- **structure-to-low / boundary-to-high routing**：注入点符合设计语义，尚无训练结果。
- **independent gamma residual guidance**：独立参数和 zero-init adapter 存在，但未训练、未进 optimizer。
- **staged training 的总体收益**：阶段 checkpoint 存在，但缺少同预算 end-to-end/去阶段对照。

### 当前没有证据支持

- **CONCH attribute semantic anchoring 是主要增益来源**：Exp6 不优于用户固定 Exp5，且比较多重混杂。
- **CLIP 与 CONCH 相同可证明 encoder 不是瓶颈**：Exp7 是 post-hoc bank 替换，不能排除其他瓶颈。
- **SGA-SB corrected 已有效**：训练构造、冻结、optimizer 均断路。
- **独立 gamma=0.05 优于其他初始化**：无消融。

### 应降级为附加消融

- **PNuDP Dense**：当前仅 smoke/短程记录，无保留 checkpoint/full test；与主线 low/high path 没有内部耦合。
- **CONCH/CLIP encoder 选择**：在完成 no-text/real/shuffled/uniform 因果消融前，不应作为主贡献。
- **旧 unified SpatialInstanceAttrHead**：已 deprecated，仅保留历史 ablation。

## 4. 最可能的审稿质疑与所需实验

| 质疑 | 当前风险 | 必需证据 |
|---|---|---|
| “frequency-aware”是否只是命名 | 没有 FFT/DCT，容易被误解为频域算法 | 明确定义功能性 low/high feature routing；报告路径张量、注入点和 FREQPATH_ABLATION；不要声称频谱分解 |
| 提升是否来自辅助监督而非 guidance | 当前没有 supervision_only 对照 | none vs supervision_only vs guidance；同父 checkpoint/seed/预算 |
| structure/boundary 分离是否必要 | corrected 未训练 | structure-only、boundary-only、both；至少 3 seeds；主看 mAJI/mPQ |
| gamma/adapter 是否真实学习 | 当前未进 optimizer | 参数注册、grad norm、step delta、adapter 输出 norm 与 gamma trajectory |
| CONCH 是否只是更大 encoder | Exp6/Exp7 混杂 | no-text、real、shuffled、uniform prompt；完全相同 checkpoint/pipeline |
| Exp5/Exp6 是否公平 | 父 checkpoint 和 trainable modules 不同 | 统一 parent checkpoint、epochs、optimizer policy、test command 与 checkpoint selection |
| 结果是否挑单次最高 validation | 现有记录主要单 seed | 3 seeds，报告 mean±std；预先固定 selection metric，full test 一次性执行 |
| PNuDP 是否堆模块 | 只有 smoke | 主线完成后再做，若无稳定 mAJI/mPQ 增益则附录/负结果 |

## 5. 推荐论文主张边界

当前可安全陈述：项目实现了可训练的数值属性驱动 FreqPath feature routing，且已有单 seed 实验显示用户记录中的 Exp5 优于 visual baseline。

当前不可安全陈述：SGA-SB 有效、CONCH 提供稳定增益、CLIP/CONCH 等价、PNuDP Dense 改善实例分割、或方法进行了严格频域分解。

