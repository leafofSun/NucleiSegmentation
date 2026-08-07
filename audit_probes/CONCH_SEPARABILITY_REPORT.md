# CONCH 语义原型可分性诊断（P1-2）

本报告仅使用 CONCH 文本空间几何量；没有启动训练，也没有读取或使用 Dice、AJI、PQ 等分割指标。任务说明第 4 节的预注册阈值及失败优先级在执行中保持不变。主判定结果为 **PASS**，按预注册变体选择规则冻结 **Set-A + V1**，因此本诊断对 RSGR-1 的语义原型门给出 **GO**。

## 0. 环境与输入哈希

### 0.1 执行环境

| 项目 | 审计值 |
|---|---|
| GPU 主机 | `nuseg-server` |
| 远端主机名 | `I2a4700123c0020120a` |
| 远端仓库 | `/hy-tmp/NuSeg` |
| 审计代码 commit | `d9ef588ad2386067b29c45c3a8ebe24a5c37d33f` |
| 取证时间 | `2026-08-07T10:14:08Z` |
| Python | `3.12.13`，`/usr/local/miniconda3/envs/my_env/bin/python` |
| PyTorch | `2.9.1+cu128` |
| CONCH 源码 revision | `141cc09c7d4ff33d8eda562bd75169b457f71a62` |
| CUDA | `torch.cuda.is_available() == False`; device count `0` |
| `nvidia-smi` | `No devices were found`，按项目规则记为 **NO_GPU_MODE**，不是基础设施故障 |
| 容器资源 | cgroup v1 memory limit `2147483648` bytes（2 GiB），`memory.failcnt=166257`、`oom_kill=4`；CPU quota `100000/100000`（1 core） |
| 运行方式 | CPU、离线、纯推理；`training_started=false`，`segmentation_metrics_used=false` |

没有启动任何训练。三次正式输出均由 CPU 推理生成。初始完整 factory loader 在 2 GiB cgroup 下被 OOM 终止；失败日志和后续低内存加载失败证据均保留，见附录 D。

### 0.2 输入与编码路径

| 资源 | 实际绝对路径 | SHA256 |
|---|---|---|
| CONCH checkpoint（CLI/snapshot 路径） | `/hy-tmp/NuSeg/hf_cache/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin`（symlink） | `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c` |
| CONCH checkpoint | `/hy-tmp/NuSeg/hf_cache/hub/models--MahmoodLab--conch/blobs/40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c` | `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c` |
| RSGR Local-5 schema | `/hy-tmp/NuSeg/training/rsgr_local5_schema.json` | `01c8dfc779811592207df7b678b84bb192a42aebd00b18748eb09e24d0126e79` |
| global27 模板 | `/hy-tmp/NuSeg/workdir/attr_stats/structure_boundary_prompt_templates.json` | `4b166663963ac6063c1dd81846754da3fe51f5ba8bf63bb33840794ac87ba66d` |
| 本次实际编码 callable | `/hy-tmp/NuSeg/audit_probes/probe_conch_separability.py:1072`，`encode_with_project_conch_path` | 文件 `fa975bedd2ab35370f76ad89dc4c09d0338aeff2d8db173ce69ab0ab8e6fe87f` |
| 项目离线 bank 编码参考 | `/hy-tmp/NuSeg/tools/build_l1a_text_prototype_bank.py:53-73` | `f95db7e632129f877db83338878869dcca0570babfd30974f913344824d6d201` |
| 项目训练期文本路径参考 | `/hy-tmp/NuSeg/segment_anything/modeling/sam.py:3115-3134,3226-3234` | `5d4a198ef16ec951de2d3ca2844ac7e15a01e908a5fcbd21338a0d3db7f38f20` |
| CONCH model config | `/usr/local/miniconda3/envs/my_env/lib/python3.12/site-packages/conch/open_clip_custom/model_configs/conch_ViT-B-16.json` | `da350fdff87c831dc5181e2103db8bf52aca805ed20e1eb8ae7f7ec59afad1ab` |

本地工作区中的 global27 预期路径 `/Users/yizheng001/Developer/NuSeg/workdir/attr_stats/structure_boundary_prompt_templates.json` 为 `NOT_FOUND`；没有以相似文件替代，正式输入使用上表远端原文件。该 JSON 实际含 5 个 structure 项和 5 个 boundary 项；项目约定的 9 属性固定顺序只取 5+4，额外的 `boundary_prompts.touching_or_crowding_difficulty` 以及描述字段被显式列入 `ignored_source_keys`，没有静默回退或猜测。

编码严格沿用项目 contract：CONCH `get_tokenizer()`；Hugging Face tokenizer 使用 `padding="max_length"`、`max_length=77`、`truncation=True`；`model.encode_text(...).float()`；最后以 `torch.float32 F.normalize(dim=-1, eps=1e-8)` 归一化。项目离线 bank builder 与训练期 CONCH 路径均使用 77 token。需要显著说明：安装包的模型 config 声明 context length 128，CONCH 自带 convenience tokenizer helper 默认 `max_length=127`；本诊断没有采用该默认 helper，而是有意复用项目实际的 77-token 路径。L1A parity 结果进一步验证该选择，见附录 D。

### 0.3 低内存加载差异

2 GiB 限制下，模型存储加载改为：在 `meta` 上按同一 CONCH config 实例化，使用 `torch.load(weights_only=True, mmap=True)` 只读映射 checkpoint，再以 `load_state_dict(assign=True)` 绑定权重，并重建非持久 buffer `text.attn_mask`。checkpoint 相对完整 CoCa config 缺少的 315 个 key 全部且仅属于 `text_decoder.*`，unexpected key 为 0；这些 key 与 `encode_text` 无关，完整 factory loader 本来也以 `strict=False` 接受该 checkpoint。实现只允许这一已审计的 decoder 缺失集合，并对任何非 decoder 的 missing/meta tensor 硬失败。

这改变的是模型构造和权重驻留方式，不改变 prompt、tokenizer、77-token 截断、`encode_text` forward、float32 转换或 L2 归一化。成功运行的 `probe_config.json` 将加载模式记为 `meta_init_plus_read_only_mmap_assign`。与既有 L1A bank 的最大逐元素差为 `1.1920928955078125e-07`，支持其数值等价性。

### 0.4 固定判据

| 判据 | 固定触发条件 |
|---|---|
| C1 | `intra_attr_cos > 0.95` |
| C2 | `eff_rank_95 < 5` |
| C3 | `level_axis_alignment > 0.90` |
| C4 | `monotonic_ratio < 0.8` |
| C5 | `separation >= 0` |

失败模式并发时的固定优先级为 **F1 > F3 > F2 > F4**。变体选择规则固定为：在不触发 C1–C5 的变体中最大化 `eff_rank_95`；并列时选择 `separation` 最负者。V4 任一 mid 残差比大于 0.5 时不得采用。全文中的 `✓` 表示该判据被触发，`—` 表示未触发。

## 1. Set-A / V0 主判定

Set-A 从 schema 读取，属性顺序为 `nuclear_density, nuclear_size_heterogeneity, spatial_crowding, nuclear_irregularity, nuclear_elongation`，等级顺序为 schema 的 `low, medium, high`（诊断别名 `low, mid, high`），共 15 条 prompt、15 个 512 维原型。

### D1：余弦分离

- `intra_attr_cos = 0.7376895378192483`（15 对）
- `inter_attr_cos = 0.46671019119876694`（90 对）
- `separation = -0.27097934662048134`
- 完整 15×15 矩阵：`/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3/V0/cosine_matrix.csv`，SHA256 `4d568034ef0021bbc9cd76e9f3b2da3bc8856d2192c2980a036a07c364caccac`
- 热力图：同目录 `cosine_heatmap.svg`，SHA256 `ed2bfda38994d15c60014e3697910b5ddc04d4f1ac691cf9331fbcc2e1809924`

### D2：有效秩

- 奇异值（4 位有效数字）：`[1.376, 1.184, 1.003, 0.7726, 0.7403, 0.5966, 0.5663, 0.5142, 0.4212, 0.3218, 0.3057, 0.2809, 0.2511, 0.1503, 3.746e-16]`
- `eff_rank_95 = 10`
- `eff_rank_90 = 8`
- `s1_energy_ratio = 0.273254093484833`

### D3：等级轴一致性

`level_axis_alignment = -0.02124939197836279`，没有 zero-norm axis。完整 5×5 矩阵如下；原 CSV SHA256 为 `c494f35af57c18d45d58b3cc884fef3943ce24cb350e2b59f91d1d24d1335857`。

| 属性 | nuclear_density | nuclear_size_heterogeneity | spatial_crowding | nuclear_irregularity | nuclear_elongation |
|---|---:|---:|---:|---:|---:|
| nuclear_density | 1 | 0.01651723452 | -0.07552559418 | -0.03613487940 | 0.05352924442 |
| nuclear_size_heterogeneity | 0.01651723452 | 1 | -0.15957109560 | -0.02940297774 | -0.02005450671 |
| spatial_crowding | -0.07552559418 | -0.15957109560 | 1 | 0.03713081004 | -0.03128737657 |
| nuclear_irregularity | -0.03613487940 | -0.02940297774 | 0.03713081004 | 1 | 0.03230522141 |
| nuclear_elongation | 0.05352924442 | -0.02005450671 | -0.03128737657 | 0.03230522141 | 1 |

### D4：单调性

| 属性 | `t[a]` | `0 < t < 1` |
|---|---:|---|
| nuclear_density | 0.7175934694923486 | 是 |
| nuclear_size_heterogeneity | 0.5413188062801594 | 是 |
| spatial_crowding | 0.5764311233751173 | 是 |
| nuclear_irregularity | 0.9251571146162103 | 是 |
| nuclear_elongation | 0.8719551021234129 | 是 |

`monotonic_ratio = 1.0`。

### C1–C5 与分类

| 判据 | 实测与固定阈值 | 触发 |
|---|---|---|
| C1 | `0.7376895378192483 > 0.95` 为假 | — |
| C2 | `10 < 5` 为假 | — |
| C3 | `-0.02124939197836279 > 0.90` 为假 | — |
| C4 | `1.0 < 0.8` 为假 | — |
| C5 | `-0.27097934662048134 >= 0` 为假 | — |

结论：C1–C5 全部不触发，按预注册定义分类为 **PASS**；不进入 F1/F2/F3/F4 的优先级消歧。

## 2. 变体矩阵完整表（Set-A × V0–V4）

V2 的 `k=1` 和 `k=2` 分列，因此共有 6 行。`criteria` 列列出实际触发项。

| Variant | intra | inter | separation | rank95 | rank90 | s1 energy | axis align | monotonic | criteria |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 | 0.737689538 | 0.466710191 | -0.270979347 | 10 | 8 | 0.273254093 | -0.021249392 | 1.0 | none |
| V1 | 0.433847491 | -0.151658896 | -0.585506387 | 10 | 8 | 0.234700492 | 0.002301823 | 1.0 | none |
| V2_k1 | 0.257498141 | -0.124630141 | -0.382128282 | 9 | 8 | 0.263242668 | -0.041222244 | 1.0 | none |
| V2_k2 | 0.100839439 | -0.095129963 | -0.195969402 | 10 | 8 | 0.283481444 | -0.034346770 | 1.0 | none |
| V3 | -0.481935230 | -0.000522038 | 0.481413192 | 9 | 8 | 0.177805621 | -0.006457683 | 1.0 | C5 |
| V4 | -0.333333333 | -7.709882115e-20 | 0.333333333 | 5 | 5 | 0.240303317 | -0.021249392 | 1.0 | C5；V4 residual fail |

通过 C1–C5 的候选为 V0、V1、V2_k1、V2_k2。最高 `eff_rank_95=10` 的候选为 V0、V1、V2_k2；并列时 V1 的 separation `-0.5855063870910331` 最负，因此预注册规则选 **V1**。V3 和 V4 均触发 C5，不能为追求表观中心化而采用。

Set-A/V4 的 mid 残差比分别为：nuclear_density `0.9487110250433329`、nuclear_size_heterogeneity `0.9982784283498048`、spatial_crowding `0.9981774680984491`、nuclear_irregularity `0.9642642603767932`、nuclear_elongation `0.9271990693144141`；最大值 `0.9982784283498048`，5/5 均超过 0.5，故 **V4 不适合采用**。V4 的 5 个 mid 向量按定义为零向量；这也是其 `zero_vector_count=5` 的来源，而不是编码失败。

## 3. Set-B 完整表

Set-B 使用附录 A 的 60 条临床措辞 prompt，先逐条编码和归一化，再按 5 属性×3 等级将每组 4 条形成一个归一化均值原型，最终仍为 15×512。

| Variant | intra | inter | separation | rank95 | rank90 | s1 energy | axis align | monotonic | criteria |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 | 0.587545245 | 0.529752418 | -0.057792828 | 12 | 10 | 0.189995931 | 0.049402593 | 1.0 | none |
| V1 | 0.034128956 | -0.086727783 | -0.120856738 | 12 | 11 | 0.154382777 | 0.057310744 | 1.0 | none |
| V2_k1 | -0.071150500 | -0.070177984 | 0.000972516 | 12 | 10 | 0.163573301 | 0.081269229 | 1.0 | C5 |
| V2_k2 | -0.106559619 | -0.064442081 | 0.042117538 | 11 | 9 | 0.169316048 | 0.157281725 | 1.0 | C5 |
| V3 | -0.496412690 | -0.000110245 | 0.496302445 | 9 | 8 | 0.188582416 | 0.050971276 | 1.0 | C5 |
| V4 | -0.333333333 | -1.310679960e-18 | 0.333333333 | 5 | 4 | 0.279222251 | 0.049402593 | 1.0 | C5；V4 residual fail |

Set-B 的可用候选只有 V0、V1；两者 `eff_rank_95=12` 并列，V1 的 separation 更负，因此其集合内最佳候选也是 V1。但主 Set-A/V0 已经 PASS，预注册分支不要求改写 prompt，不能用 Set-B 替换 Set-A。

Set-B/V4 残差比：nuclear_density `0.9857351883151331`、nuclear_size_heterogeneity `0.9818342253733268`、spatial_crowding `0.9981895473658197`、nuclear_irregularity `0.9999260715042740`、nuclear_elongation `0.9868770128283844`；最大 `0.9999260715042740`，5/5 超过 0.5，故不适用。

## 4. 体系 A（27 条）结果

global27 严格读取模板中的固定 9 属性×3 等级，得到 27×512。其 V0 也不触发 C1–C5；但本任务的主判定仍只以 Set-A 为准。

| Variant | intra | inter | separation | rank95 | rank90 | s1 energy | axis align | monotonic | criteria |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 | 0.554902912 | 0.372978739 | -0.181924173 | 19 | 15 | 0.157702729 | 0.052686374 | 1.0 | none |
| V1 | 0.259570702 | -0.062807677 | -0.322378379 | 19 | 15 | 0.165538454 | 0.062257004 | 1.0 | none |
| V2_k1 | 0.189605269 | -0.056079113 | -0.245684383 | 19 | 15 | 0.149363152 | 0.057298469 | 1.0 | none |
| V2_k2 | 0.085351433 | -0.046203739 | -0.131555172 | 18 | 16 | 0.119069317 | 0.039984657 | 1.0 | none |
| V3 | -0.491425178 | 0.000296537 | 0.491721715 | 15 | 13 | 0.148946120 | 0.051983213 | 1.0 | C5 |
| V4 | -0.333333333 | -3.426614274e-19 | 0.333333333 | 8 | 7 | 0.220711750 | 0.052686374 | 1.0 | C5；V4 residual fail |

global27 的 V0、V1、V2_k1 都有最高 `eff_rank_95=19`，其中 V1 separation 最负，故集合内预注册最优也是 V1。global27/V4 的 9 个残差比为 `0.9924231918360884, 0.9992290412155200, 0.9832340321668743, 0.9998387418953420, 0.9753782974643080, 0.9636405687922510, 0.9814572655277840, 0.9999976845006310, 0.9990945485997463`（属性顺序见附录 C）；9/9 超过 0.5，最大 `0.9999976845006310`，故不适用。

三张表合计覆盖全部 18 个 set/variant aggregate，均完整列出 D1、D2 aggregate、D3 aggregate、D4 ratio 和实际触发判据。每行的全部奇异值和逐属性 `t` 在附录 A、B 给出。每个变体的完整 cosine 矩阵、D3 axis 矩阵及热力图路径分别为 `<输出目录>/<variant>/cosine_matrix.csv`、`<输出目录>/<variant>/level_axis_cosine_matrix.csv`、`<输出目录>/<variant>/cosine_heatmap.svg`；其精确 SHA256 均由该输出目录的 `summary.json` 锁定，summary 自身哈希见附录 D。

## 5. 字面量对照（"high/low nuclear density"）

按顺序编码 `"high nuclear density"` 与 `"low nuclear density"`：

- Set-A 批次：cosine `0.7108526033682374`
- Set-B 批次：cosine `0.7108525859450872`
- global27 批次：cosine `0.7108526033682374`

两种批次结果仅有约 `1.74e-08` 的 float32 数值漂移。结论是 CONCH 并未把 high/low density 字面概念编码为近乎相同（相对 C1 的 0.95 仅作尺度参照；C1 并不是这组 literal pair 的独立预注册阈值）；Set-A 的完整句式同样通过主判据，因此没有“密度概念本身完全不可分”的证据。主文件 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3/literal_density_cosine.json` 的 SHA256 为 `dfbe7198556b208edbb3c6ca272a2fc67b37c8a4d983bae42f8b9816e1582930`。

## 6. 跨编码器对照

`SKIPPED_NO_ACCESS`。环境中没有经本任务确认可用且有权重访问的 PLIP、BiomedCLIP 或 QuiltNet，因此没有伪造跨编码器结果，也不据此扩大结论。

## 7. 推荐方案

### 7.1 预注册选择

推荐 **Set-A + V1（全局中心化后逐原型 L2 归一化）**。理由只来自预注册规则：Set-A/V0 已 PASS；在所有不触发 C1–C5 的 Set-A 变体中，V1 与 V0、V2_k2 的 `eff_rank_95=10` 并列最高，而 V1 的 separation 最负。没有依据下游分割指标选择。

该冻结组合的全部 aggregate 指标为：

| 指标 | 值 |
|---|---:|
| intra_attr_cos | 0.4338474906449287 |
| inter_attr_cos | -0.15165889644610442 |
| separation | -0.5855063870910331 |
| eff_rank_95 | 10 |
| eff_rank_90 | 8 |
| s1_energy_ratio | 0.23470049170406895 |
| level_axis_alignment | 0.0023018233564638365 |
| monotonic_ratio | 1.0 |

V1 的 C1–C5 均为 false。逐属性 `t` 为 `0.7142903955344039, 0.6047715680051761, 0.7417910656060677, 0.9381326112734244, 0.8736329567634503`，顺序为五个 Local-5 属性；奇异值见附录 A。

### 7.2 冻结产物与验证

冻结目录：`/hy-tmp/NuSeg/workdir/rsgr_bank`。

| 文件 | 形状/作用 | SHA256 |
|---|---|---|
| `prompts_frozen.json` | Set-A prompt 全文 | `de4413374061d3886fc87288ff48c46ea5f07d00268aaf191c7328d74f55eaa3` |
| `structure_bank.pt` | `[3, 3, 512]` | `ca28900b8650ec49974da776bdc2bef0e9408f42421e6f7aee5d4a32a34786a8` |
| `boundary_bank.pt` | `[2, 3, 512]` | `cb5cfb2d79d05cbeeef28efa5a25bb1252b287ed497c231929d8447308aeea0d` |
| `bank_manifest.json` | prompt、几何、顺序、hash、freeze metrics | `a10944ad06cffdf70742c93ed2c6570ec32b8810ea77ac013faacd04c0cab7f1` |

manifest 记录 `prompt_set=Set-A`、`geometric_variant=V1`、embedding dim 512、checkpoint SHA256 `40a964…9ae7c`、created UTC `2026-08-07T10:10:16.269991+00:00`。属性顺序严格为：

`[nuclear_density, nuclear_size_heterogeneity, spatial_crowding, nuclear_irregularity, nuclear_elongation]`。

等级顺序按 schema 原文严格记录为 `[low, medium, high]`，并另记诊断别名 `[low, mid, high]`；这不是顺序变化。两个 bank 的原型范数范围为 `[0.9999999404, 1.0]`，`atol=1e-5` 单位范数检查为 true，数值均有限。冻结重放使用已审计的 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3/raw_embeddings.npz`（SHA256 `e485b9ef7504fe7a17ca1ee49d67d3c762db82f682ff88df9c830df95613b287`），重放 summary SHA256 为 `cc7508064e3822a17c2cee270f1ca3ea164cc458db4641144ccebe7f591a67b4`。

### 7.3 RSGR loader 集成缺口

冻结产物本身通过形状、顺序、hash 和归一化验证，但**当前训练/推理 loader 尚不能直接消费这四文件目录**。`segment_anything/modeling/rsgr.py:169-204` 的 `load_prototype_banks` 接受一个 formal mapping `.pt`，并默认读取同 basename 的 `.metadata.json`；它要求 `schema_version=rsgr_local5_conch_bank_v1`、`backend=conch`、schema/bank hash，以及同一 mapping 内的 `structure_prototypes` 和 `boundary_prototypes`。当前冻结格式是两个独立 tensor `.pt` 加 `bank_manifest.json`，因此不能把 `workdir/rsgr_bank` 或任一单独 tensor 文件直接传给 `--rsgr_prototype_path`。

进入 RSGR-1 前需要完成任务说明附录 C 的下一步集成：从这组已冻结、hash 锁定的文件物化 loader 所需的 formal mapping + `.metadata.json`（不得重新编码或改变几何），然后用 CPU round-trip 调用 `load_prototype_banks`，核对结构/边界 tensor 逐元素一致、属性/等级顺序一致、hash 一致。这个是封装集成缺口，不改变本次文本空间 **PASS/GO** 判定；在该 round-trip 通过前不要启动 RSGR-1 训练。

## 8. 对 RSGR-1 的结论

**GO**（语义原型可分性门）。

- Set-A/V0 的 C1–C5 全部不触发，失败分类为 PASS。
- 预注册变体选择规则选中同一 Set-A prompt 的 V1，而不是改写 prompt；因此不是 `GO_WITH_MODIFIED_PROMPTS`。
- 推荐 bank 已冻结并通过 hash、shape、顺序、finite 和单位范数验证。
- 本结论只授权进入下一步 bank 封装/loader round-trip；完成该集成前不应启动 RSGR-1。
- 本诊断没有训练、没有分割数据依赖、没有使用分割指标。

## 附录 A：每个集合/变体的全部奇异值（4 位有效数字）

### Set-A

- V0: `[1.376, 1.184, 1.003, 0.7726, 0.7403, 0.5966, 0.5663, 0.5142, 0.4212, 0.3218, 0.3057, 0.2809, 0.2511, 0.1503, 3.746e-16]`
- V1: `[1.873, 1.818, 1.531, 1.114, 0.9914, 0.9442, 0.8931, 0.7884, 0.6435, 0.5058, 0.4753, 0.4192, 0.3690, 0.2256, 3.601e-16]`
- V2_k1: `[1.986, 1.706, 1.339, 1.212, 1.096, 1.045, 0.9218, 0.7747, 0.6137, 0.5325, 0.4779, 0.4144, 0.2486, 7.617e-16, 3.174e-16]`
- V2_k2: `[2.058, 1.458, 1.393, 1.213, 1.155, 1.085, 0.9298, 0.7528, 0.6480, 0.6131, 0.5544, 0.3486, 1.024e-15, 7.532e-16, 4.407e-16]`
- V3: `[1.631, 1.521, 1.450, 1.348, 1.274, 1.186, 1.124, 0.9382, 0.7742, 0.5509, 5.607e-16, 4.994e-16, 4.073e-16, 3.615e-16, 1.707e-16]`
- V4: `[1.550, 1.452, 1.415, 1.354, 1.285, 6.024e-16, 4.543e-16, 4.040e-16, 2.095e-16, 8.235e-17, 6.157e-17, 1.967e-28, 1.681e-32, 1.534e-33, 0]`

### Set-B

- V0: `[1.109, 0.9668, 0.9250, 0.8079, 0.6976, 0.6443, 0.6266, 0.5802, 0.5491, 0.4937, 0.4466, 0.3965, 0.3807, 0.3434, 4.494e-16]`
- V1: `[1.520, 1.436, 1.407, 1.277, 1.102, 1.066, 0.9543, 0.9137, 0.8293, 0.7859, 0.7160, 0.6411, 0.6038, 0.5483, 3.385e-16]`
- V2_k1: `[1.566, 1.537, 1.359, 1.238, 1.113, 1.078, 1.004, 0.9315, 0.8059, 0.7703, 0.6721, 0.6694, 0.6123, 1.163e-15, 4.097e-16]`
- V2_k2: `[1.593, 1.497, 1.463, 1.227, 1.185, 1.083, 1.027, 0.8693, 0.8248, 0.7308, 0.7271, 0.6573, 1.555e-15, 9.914e-16, 3.108e-16]`
- V3: `[1.682, 1.616, 1.442, 1.301, 1.163, 1.118, 1.071, 0.8983, 0.7907, 0.7744, 5.255e-16, 4.019e-16, 3.167e-16, 2.549e-16, 1.660e-16]`
- V4: `[1.671, 1.583, 1.510, 1.201, 0.9901, 2.662e-16, 1.949e-16, 1.899e-16, 1.205e-16, 6.633e-17, 6.085e-17, 2.178e-29, 7.167e-33, 6.256e-35, 0]`

### global27

- V0: `[1.585, 1.414, 1.245, 1.110, 1.055, 0.9859, 0.9490, 0.9421, 0.8414, 0.7500, 0.6851, 0.6538, 0.6283, 0.5987, 0.5594, 0.5478, 0.4890, 0.4408, 0.4088, 0.3927, 0.3768, 0.3213, 0.2974, 0.2649, 0.2310, 0.1665, 6.065e-16]`
- V1: `[2.114, 1.822, 1.617, 1.439, 1.349, 1.268, 1.233, 1.165, 1.087, 0.9926, 0.8896, 0.8568, 0.8194, 0.7884, 0.7329, 0.6918, 0.6504, 0.5725, 0.5687, 0.5229, 0.4956, 0.4281, 0.3971, 0.3437, 0.3215, 0.2345, 4.243e-16]`
- V2_k1: `[2.007, 1.705, 1.608, 1.513, 1.353, 1.309, 1.272, 1.213, 1.095, 0.9724, 0.9487, 0.8859, 0.8644, 0.8268, 0.7632, 0.7067, 0.6454, 0.5984, 0.5775, 0.5469, 0.4594, 0.4411, 0.3995, 0.3617, 0.2792, 3.989e-15, 3.368e-16]`
- V2_k2: `[1.791, 1.744, 1.708, 1.474, 1.401, 1.363, 1.305, 1.165, 1.044, 1.016, 0.9963, 0.9387, 0.8884, 0.8064, 0.7587, 0.7431, 0.6948, 0.6226, 0.5906, 0.4959, 0.4804, 0.4402, 0.4099, 0.3089, 4.996e-15, 2.009e-15, 3.691e-16]`
- V3: `[2.004, 1.807, 1.589, 1.432, 1.396, 1.361, 1.298, 1.202, 1.182, 1.083, 1.054, 1.021, 0.8868, 0.8569, 0.8220, 0.7578, 0.6456, 0.5286, 6.908e-16, 6.351e-16, 5.046e-16, 4.563e-16, 3.869e-16, 3.680e-16, 3.373e-16, 2.802e-16, 2.441e-16]`
- V4: `[1.993, 1.684, 1.565, 1.550, 1.336, 1.310, 1.121, 0.9706, 0.8011, 6.322e-16, 4.874e-16, 4.014e-16, 3.632e-16, 3.420e-16, 2.444e-16, 1.917e-16, 1.591e-16, 1.104e-16, 9.707e-17, 2.142e-17, 4.736e-18, 2.656e-32, 1.640e-32, 5.563e-33, 3.579e-33, 7.903e-50, 0]`

## 附录 B：每个集合/变体的逐属性 `t`

Set-A/Set-B 顺序均为 `[nuclear_density, nuclear_size_heterogeneity, spatial_crowding, nuclear_irregularity, nuclear_elongation]`。

| Set | Variant | `t`（按上述属性顺序） | ratio |
|---|---|---|---:|
| Set-A | V0 | `[0.7175934694923486, 0.5413188062801594, 0.5764311233751173, 0.9251571146162103, 0.8719551021234129]` | 1.0 |
| Set-A | V1 | `[0.7142903955344039, 0.6047715680051761, 0.7417910656060677, 0.9381326112734244, 0.8736329567634503]` | 1.0 |
| Set-A | V2_k1 | `[0.6841298178770138, 0.7302750027161309, 0.7717514671055503, 0.9394419804174413, 0.9063292561210115]` | 1.0 |
| Set-A | V2_k2 | `[0.6972914224016279, 0.6794535480740356, 0.6950310520573989, 0.7966069098421400, 0.8828144010061224]` | 1.0 |
| Set-A | V3 | `[0.6810272623879475, 0.5238244451780330, 0.5422078173438474, 0.6899909354730891, 0.7458162218934764]` | 1.0 |
| Set-A | V4 | `[0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750]` | 1.0 |
| Set-B | V0 | `[0.3917737216330973, 0.6286939373893136, 0.5533876532957941, 0.5090318439039121, 0.3673238155516810]` | 1.0 |
| Set-B | V1 | `[0.4958036161397010, 0.6980784599957431, 0.6042260051699713, 0.5930439085485265, 0.4860598378958388]` | 1.0 |
| Set-B | V2_k1 | `[0.2280702117919678, 0.7163264039972684, 0.6272468895483517, 0.4043208159353809, 0.4864864604991843]` | 1.0 |
| Set-B | V2_k2 | `[0.3940691956154070, 0.7288382565586321, 0.7082365091282639, 0.3674964862148343, 0.4499375179046138]` | 1.0 |
| Set-B | V3 | `[0.4403030649135277, 0.5841666968527043, 0.5263309891436690, 0.5059476671854717, 0.4292241046167771]` | 1.0 |
| Set-B | V4 | `[0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750]` | 1.0 |

global27 顺序为 `[nuclear_density, nuclear_area_fraction, mean_nuclear_size, nuclear_size_heterogeneity, spatial_crowding, boundary_density, nuclear_irregularity, nuclear_elongation, small_nuclei_ratio]`。

| Set | Variant | `t`（按上述属性顺序） | ratio |
|---|---|---|---:|
| global27 | V0 | `[0.3422171185982217, 0.4709810393615627, 0.8249557886225733, 0.4876252132912209, 0.6743025807138044, 0.6703451151636025, 0.3805039990977790, 0.5016631531587690, 0.5339339831525268]` | 1.0 |
| global27 | V1 | `[0.3821966347881943, 0.5424114425009157, 0.7635677810571383, 0.6684254550356183, 0.4606014467298819, 0.6440778470585107, 0.3597199587379617, 0.4799229376630552, 0.7066334060844051]` | 1.0 |
| global27 | V2_k1 | `[0.4048303468998168, 0.6664923538588869, 0.8323976811989864, 0.6683250074369984, 0.4879482114132323, 0.6526801892308921, 0.4408673171126953, 0.4524870009605419, 0.6954700818288259]` | 1.0 |
| global27 | V2_k2 | `[0.4864644764761516, 0.6671930545507082, 0.7858012612636379, 0.4397208248338094, 0.4814604846146257, 0.6888147666731101, 0.4140485386751280, 0.4191357558581055, 0.7000019495452217]` | 1.0 |
| global27 | V3 | `[0.4451092022333833, 0.4857390453435347, 0.5716222749183207, 0.4935745160480610, 0.5602016017594484, 0.6087118279367058, 0.4277523230370358, 0.5005846342895507, 0.5150613695651276]` | 1.0 |
| global27 | V4 | `[0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750, 0.4999999999998750]` | 1.0 |

## 附录 C：V4 残差明细

| Set | 属性 | mid 残差比 | `> 0.5` |
|---|---|---:|---|
| Set-A | nuclear_density | 0.9487110250433329 | 是 |
| Set-A | nuclear_size_heterogeneity | 0.9982784283498048 | 是 |
| Set-A | spatial_crowding | 0.9981774680984491 | 是 |
| Set-A | nuclear_irregularity | 0.9642642603767932 | 是 |
| Set-A | nuclear_elongation | 0.9271990693144141 | 是 |
| Set-B | nuclear_density | 0.9857351883151331 | 是 |
| Set-B | nuclear_size_heterogeneity | 0.9818342253733268 | 是 |
| Set-B | spatial_crowding | 0.9981895473658197 | 是 |
| Set-B | nuclear_irregularity | 0.9999260715042740 | 是 |
| Set-B | nuclear_elongation | 0.9868770128283844 | 是 |
| global27 | nuclear_density | 0.9924231918360884 | 是 |
| global27 | nuclear_area_fraction | 0.9992290412155200 | 是 |
| global27 | mean_nuclear_size | 0.9832340321668743 | 是 |
| global27 | nuclear_size_heterogeneity | 0.9998387418953420 | 是 |
| global27 | spatial_crowding | 0.9753782974643080 | 是 |
| global27 | boundary_density | 0.9636405687922510 | 是 |
| global27 | nuclear_irregularity | 0.9814572655277840 | 是 |
| global27 | nuclear_elongation | 0.9999976845006310 | 是 |
| global27 | small_nuclei_ratio | 0.9990945485997463 | 是 |

三组 V4 都因 C5 和残差门失败，不应采用。V4 中 alpha(mid)=0，因而 zero-vector count 分别为 5、5、9。

## 附录 D：复现产物、哈希、日志与 parity

### D.1 输出目录与摘要

每个 `summary.json` 内含该目录全部 `metrics.json`、完整 cosine CSV、D3 matrix CSV、SVG 热力图和 NPZ intermediates 的逐文件 SHA256 索引。

| Prompt set | 绝对输出目录 | summary SHA256 |
|---|---|---|
| Set-A | `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3` | `315f2a2fb9b5a6c4286b9887f9de0d83828821d1def08309acfa1f1341c94d86` |
| Set-B | `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_b_v3` | `24f48e2a4ad938924a396e1253259bea9264f71277035018357fdc0f36ef73c7` |
| global27 | `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/global27_v3` | `10acdcef78b1077671670598e59b8ec195680b5d0ee1899f9e777b8f556b92fe` |
| freeze replay | `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_freeze_replay` | `cc7508064e3822a17c2cee270f1ca3ea164cc458db4641144ccebe7f591a67b4` |

各 aggregate 行对应的 `metrics.json` SHA256：

| Set | V0 | V1 | V2_k1 | V2_k2 | V3 | V4 |
|---|---|---|---|---|---|---|
| Set-A | `bd9d28a2866b…` | `e16659822a6b…` | `738ae303fab5…` | `cd6b7964c800…` | `15b30585d9a0…` | `10cad5671386…` |
| Set-B | `232b2fee8cbf…` | `be26c6bc13b5…` | `4c818270c817…` | `13ad35443fc6…` | `f30f10eea538…` | `ab0c840dea96…` |
| global27 | `1c882a895ca5…` | `ef8fc6dbf337…` | `11ee27321d7b…` | `acd5f9ba76f3…` | `149e8e912715…` | `eca4477773dc…` |

完整值分别为：

- Set-A: `bd9d28a2866bc4ae15013e4bd666e9907a96d73e433501fe99d648a625fae798`, `e16659822a6bd8f4bc0fd75fb0c449aca6e367f3e901854d7dc86571c5fe393d`, `738ae303fab58218c883686c5731b31aefceb5ef7a22bfdd15c1607e7ffcb8e1`, `cd6b7964c800579011efcb1106ec6c3bd8d9ffaa6bbd7a542088e2b2cb010ac8`, `15b30585d9a050b5a1debdb58aaa9dcadd6d6bbe3a8c4252e70e42d3dc2cd651`, `10cad56713864c1ba759f0a377fde75385684e5ad3ac031a677a33058e094cbe`。
- Set-B: `232b2fee8cbf7c9863866f130f21e71debc018a0d7e6c077ffd3a4d61ef55a21`, `be26c6bc13b5268f7dee8a59621e6ea5f610cb0d4b6912f28526f3365200bc1d`, `4c818270c817af3e520646c2f7457da332b16855a8aa7e57b7fc6eb234338c2b`, `13ad35443fc6316c62e7ddf07b7fd5d6d38c01df1f186b3161c6677d98eb4ca8`, `f30f10eea5380a519fb9d1829be761e5d2e672088b4b57e25d53cf68fe910270`, `ab0c840dea965631176ede7757c9fb54109d8ae9b53c70154389f6c0c474b808`。
- global27: `1c882a895ca596d3fba9433823ecec74d44f3fd574bfb07d07ba4db1dc73c470`, `ef8fc6dbf3376d5756ba54345a80eae54c8f2a26f9ea5bbdfe0c77481e5a94c6`, `11ee27321d7bfa54e2fe5c6a8141019c2642776062c49e5a1d70181ff5961ab7`, `acd5f9ba76f3bb1ce3c5da9aa956ed8e500f3da1b5492728a6e2d6e85cfc62f3`, `149e8e912715f3b6f070df31212edf1196afb375edd0841ee56ed29601239de2`, `eca4477773dc85f7420c3c23a48fc553fec4f186d868f9d3a379c2a965fae954`。

### D.2 日志

日志根目录：`/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/logs`。

| 日志 | 状态 | SHA256 |
|---|---|---|
| `set_a.log`, `set_b.log`, `global27.log` | 初始完整 factory load；OOM 后无 `[PROBE_RESULT]`，三文件内容相同，均保留 | `bb7033150454bc9e48dccab3f44668be1523ea60d494a1eb02921ffd05bba125` |
| `set_a_lowmem.log` | 失败证据：发现 decoder meta tensors | `18f1249dda86547a7790ddb67b30b36cf5875510bd6064b3e264464fcb7973c5` |
| `set_a_lowmem_v2.log` | 失败证据：发现 causal mask 的 meta/CPU mismatch | `6fba38c7578b37fdb0532ff75dbeaaa2804e7b960fa20f678606a4a249d77391` |
| `set_a_lowmem_v3.log` | 成功，`[PROBE_RESULT]` PASS/V1 | `cf504b8c29fd6ac3500dcbe56be751408e718a100cd72282ab1f2a8454021bd5` |
| `set_b_lowmem_v3.log` | 成功 | `8290dbb2714882bd06a78bd621deeb6d3fda9b1980f5887ac0d94f3d70e91dae` |
| `global27_lowmem_v3.log` | 成功 | `1665da29201f19ea89139efd1d8d84702fd32cf91dd9844c8256653fb4e5271e` |
| `set_a_freeze_replay.log` | 成功 | `e13ce188a0f0d576f6c04ea49479504ab54b0581f5aa67c14506108051a13efb` |
| `unit_tests.log` | 8 tests，OK | `4ef9868f1ba6a8d82c7879b82e1b344700a0309a015fb81bad48ab7716ad809b` |

### D.3 L1A parity

既有 bank：`/hy-tmp/NuSeg/workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.pt`，SHA256 `f02d6d99d3059a5b62aa096560c5289ae6f4d2036b28cd2ed36a2b301221dcb4`；metadata `/hy-tmp/NuSeg/workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.json`，SHA256 `09f212b72db5a0115c46519f15efe12391536032fe7f817b92eb80141c9da046`。

将本次 Set-A/V0 的 15×512 原型 reshape 为 `[5,3,512]` 与既有 L1A embeddings 对齐比较：

- shape 均为 `[5,3,512]`
- `exact_equal = false`（float32 最末位差异）
- `max_abs = 1.1920928955078125e-07`
- `mean_abs = 4.1066101630349294e-09`
- row cosine min `0.9999998807907104`
- row cosine max `1.0000001192092896`
- row cosine mean `1.0`

该误差量级与批处理/加载路径的 float32 舍入一致；没有 prompt、顺序、tokenizer 或归一化 contract 的实质偏离。
