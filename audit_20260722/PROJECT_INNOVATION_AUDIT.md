# NuSeg 创新性与论文主张审计

## 总结

当前项目存在三个可讨论的机制方向：多层级属性/数值语义、语义条件残差（PNuRL/PNuDP）、结构—边界门控（SGA）。但“代码中存在机制”与“论文级创新成立”之间仍缺关键证据：固定父模型的单变量消融、语义正确性对照、多种子 canonical full-test，以及推理期实例/区域语义闭环。因此当前不能把任一路线表述为已验证的主创新。

## 相关工作的可核查定位

| 工作 | 可核查来源 | 与 NuSeg 的关系 | 审计边界 |
|---|---|---|---|
| PromptNu，Yao et al., IEEE TMI 2025 | DOI: https://doi.org/10.1109/TMI.2025.3579214 | 已有工作把视觉语言提示用于核实例分割/分类；NuSeg 不能把“给核分割加入文本提示”本身作为新颖性 | 本次只核对公开元数据与项目引用语境，未做全文逐句 novelty search |
| WeaveSeg，Li et al., ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Li_WeaveSeg_Iterative_Contrast-weaving_and_Spectral_Feature-refining_for_Nuclei_Instance_Segmentation_ICCV_2025_paper.html | 真实 spectral feature refining 已存在；NuSeg 当前 FreqPath 没有 FFT/DCT，不能借“频域”措辞制造差异 | 应将当前模块准确称为 low/high feature path，或补真实频谱操作与消融 |
| CONCH，Lu et al., Nature Medicine 2024 | https://www.nature.com/articles/s41591-024-02856-4 | CONCH 是现成病理视觉语言基础模型；使用它属于组件选择，不自动构成方法创新 | 创新必须落到条件生成、粒度对齐或可验证的交互机制 |
| IVAAN，Jeong et al., CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/papers/Jeong_IVAAN_Instance-level_Vision-Language_Alignment_via_Attribute-Guided_Text_Prompts_Generation_for_CVPR_2026_paper.pdf | 直接覆盖实例级视觉语言对齐与属性引导文本；对 NuSeg 的实例属性/文本路线构成强邻近工作 | NuSeg 必须证明其局部区域机制、密集核场景和推理闭环的实质差异 |
| DyKo，Li et al., CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Li_Universal-to-Specific_Dynamic_Knowledge-Guided_Multiple_Instance_Learning_for_Few-Shot_Whole_Slide_CVPR_2026_paper.html | 动态知识从通用到特定的思想与 dynamic prompt 叙事邻近 | 任务是 WSI MIL，不等同于核实例分割；可用于界定而非直接比较指标 |
| MLLM-HWSI，Alawode et al., CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/papers/Alawode_MLLM-HWSI_A_Multimodal_Large_Language_Model_for_Hierarchical_Whole_Slide_CVPR_2026_paper.pdf | 层级病理视觉语言建模已成为明确方向 | NuSeg 的 tile/region/instance 层级必须给出真正的层级交互与实验证据 |

## 候选创新矩阵

| 候选主张 | 代码独特性 | 现有性能证据 | 与邻近工作区分度 | 当前可用表述 | 不得使用的表述 | 结论 |
|---|---:|---:|---:|---|---|---|
| 多层级数值属性路由改善核分割 | 中 | 中：Exp5 整体配方最好 | 中低 | “包含多级属性与数值路由的配方在单种子 full-test 上优于视觉基线” | “属性路由独立带来显著提升” | 保留为固定父路线，不作独立因果主张 |
| PNuRL 有界语义 delta 注入 | 中高 | 低：Exp6 无正式 full-test | 中 | “实现了受条件控制、幅值有界的残差注入” | “被证明提升泛化/实例分割” | 候选机制，需正确/打乱/随机语义对照 |
| PNuDP Dense bias | 中 | 很低 | 中 | “完成短程可运行性验证” | “完成训练并取得增益” | 暂不进入论文主线 |
| SGA gamma 结构门控 | 中 | 中低：五组短程对照均未联合取胜 | 中低 | “完成了真实执行与 matched screening” | “显著改善结构边界” | 冻结为负结果/控制，不继续堆叠 |
| P4.1 soft boundary | 低中 | 负 | 低中 | “预注册筛选未通过” | “更优的边界学习” | STOP |
| L1A local-region alignment | 中 | 中低：E5 主门槛失败 | 中，且受 IVAAN 邻近约束 | “数据审计支持粒度错配，首个局部监督实现未通过主门槛” | “实现了实例级推理语义对齐并提升性能” | 重新设计后再测 |
| FreqPath | 低（按频域主张） | 低 | 低 | “low/high feature modulation path” | “spectral/frequency-domain module” | 必须重命名或实质改造 |
| 动态 GT prompt | 低 | 不适合作部署证据 | 低 | “训练/oracle/debug 数据策略” | “测试时动态语义推理” | 只作训练辅助与上界分析 |

## 新颖性风险

1. **名词大于运算。** FreqPath 没有真实频谱变换；继续用“频域精炼”会与代码事实冲突，并在 WeaveSeg 等真实 spectral 方法面前暴露。
2. **组件组合不等于创新。** SAM、CONCH、属性 prompt、残差 adapter 与边界损失的组合需要一个可证伪的核心机制，不能靠模块数量形成贡献列表。
3. **粒度叙事尚未闭环。** L0 证明 tile 内异质性，L1A 却是训练期 GT 窗口辅助损失，没有推理期区域/实例语义生成；从动机到方法中间仍断裂。
4. **结果归因风险。** Exp5 是混合配方；Exp6 更换父节点与多个模块；验证还存在 rank-local/40% 问题。任何单模块提升措辞都超出现有证据。
5. **邻近工作压力。** PromptNu、IVAAN 和层级病理 VLM 已覆盖大方向。可辩护差异应落在“密集核场景中的区域条件语义、严格负对照和部署期闭环”，而非泛称视觉语言对齐。

## 可形成论文贡献的最低条件

- 明确定义一个主机制：建议为 **region-conditioned semantic residual for dense nuclei**，而不是同时主推 SGA、频率、属性、边界与多个 prompt 变体。
- 正确局部文本必须同时胜过 no-local、区域打乱文本和冻结随机原型；否则提升可能来自正则化或额外参数。
- 固定 Exp5 父 checkpoint、训练预算、后处理和 canonical full-test，至少 3 个预注册种子。
- 报告每种子的 Dice/IoU/mAJI/mPQ、均值、标准差/置信区间，以及失败种子；不得只报 best checkpoint。
- 给出推理期区域条件的来源、成本和无 GT 泄漏证明。
- 修复/统一 DDP validation 汇总和 train/test 后处理协议后再做模型选择。

## 创新审计裁决

当前创新准备度 **1/5**。可以写成内部技术报告或负结果/路线选择记录，但不具备把语义、SGA 或 FreqPath 宣称为已验证论文主创新的证据。最合理的动作是 **RESTRUCTURE**：收缩到局部语义粒度这一条主问题，以严格负对照和多种子 full-test 决定是否继续。
