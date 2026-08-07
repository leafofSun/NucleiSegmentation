# NuSeg 实验账本（2026-07-13）

## 口径

- 主要指标：mAJI、mPQ；次要指标：Dice、IoU。
- `用户固定 full test`：保持用户给定值，若仓库没有原始结果则标记 `PROVIDED_CONTEXT_UNVERIFIED`。
- checkpoint 中的单次 validation 数值仅用于谱系核对，不用于宣称论文级提升。
- `FREQPATH_ABLATION` 在相关训练日志中为 `both`；无日志者记 UNKNOWN。

## 核心实验

| 实验 | parent checkpoint | phase / epoch / seed | 核心开关与文本 | FREQPATH_ABLATION | spatial_sb_mode | 可训练模块摘要 | Dice / IoU / mAJI / mPQ | 原始路径 | 严格可比 | 当前结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| Visual baseline | earlier `mp_sam_vision_phase/best_model.pth` | vision / 30（best epoch 28）/ 42 | use_asr=true, freqpath, no PNuRL/text | checkpoint args 未记录该环境变量 | 参数不存在（历史等价 none） | decoder/FreqPath/CNN/ASR/image adapters | 用户固定：0.8089 / 0.6963 / 0.6270 / 0.6034 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；checkpoint val：`workdir/models/Visual_baseline/best_model.pth`，metrics history 同目录 | 否；固定 full test 缺命令/结果文件 | 基线可用；固定 full-test 来源待补 |
| Phase B multilevel warmup | Visual baseline | multilevel_attr_warmup / 30（best epoch 28）/ 42 | multilevel heads=true；CONCH=false | UNKNOWN | 参数不存在 | multilevel + legacy structure/boundary attr heads | 非分割实验 | `workdir/models/phaseB_ml_instancefix_from_visual_3gpu_30ep_v1/best_multilevel_attr_model.pth` | 不适用 | 属性头已训练；不证明分割增益 |
| Phase C alignment | Phase B | semantic_alignment / 10（best epoch 9）/ 42 | attr-text alignment=true；CONCH=true | UNKNOWN | 参数不存在 | 4 个 attr-align projections | 非分割实验 | `workdir/models/phaseC_align_from_phaseB30_3gpu_10ep_v1_full_for_phaseD/best_align_full_model.pth` | 不适用 | alignment 已训练；不证明语义注入增益 |
| PromptNu-lite v2 | Phase C | semantic_injection / 5（best epoch 3）/ 42 | lite=true；target=semantic_delta；struct/bound=0.05 | UNKNOWN | 参数不存在 | semantic injection + attr-align projections 等 | checkpoint best val Dice 0.809370；mAJI 0.637797；IoU/mPQ 未存顶层 | `workdir/models/promptnu_lite_v2_semantic_delta_rms_w005_5ep_v1/best_aji_model.pth` | 否 | 已训练但无明确严格增益 |
| Exp5 numeric no-text | Phase B | semantic_injection / 10（best AJI epoch 8）/ 42 | numeric route=true；small_normal；modulator reinit 1e-4；CONCH/PG3/text bank=false；use_pnurl=false | both | 参数不存在 | segmentation groups + numeric_attr_freqpath_proj | 用户固定：0.8172 / 0.7048 / 0.6361 / 0.6094 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；训练日志 `workdir/logs/exp5_numeric_attr_route_10ep_reinit1e4_v1_20260708_1214.log`；checkpoint 同名模型目录 | 与 baseline 仍缺严格同 pipeline 证据；与 Exp6 否 | 当前用户固定记录最好；真实主线 |
| Exp6 CONCH/PG3 | Phase C | semantic_injection / 10（best AJI epoch 3）/ 42 | use_pnurl=true；attr-align=true；PG3=true；CONCH/text bank=true；modulator reinit 1e-4 | both | 参数不存在 | Exp5 多数 segmentation groups + PNuRL/prompt learner/PG3 projections；无 numeric projection | 用户固定：0.8091 / 0.6945 / 0.6269 / 0.6047 | 固定值：`PROVIDED_CONTEXT_UNVERIFIED`；训练日志 `workdir/logs/exp6_phaseC_text_both_10ep_reinit1e4_v1_20260708_1622.log` | 否 | 无清晰增益；不能归因于文本本身 |
| Exp7 CLIP/PG3 | Exp6 best-AJI epoch-3 权重 | post-hoc test artifact；无独立训练记录 | 将 text bank 替换为 CLIP ViT-B/32；checkpoint args 仍写 CONCH=true | test 默认可能为 both，精确命令未找到 | none | 无训练；仅替换 buffer | 用户记录：与 Exp6 基本一致 | `workdir/models/exp6_phaseC_text_both_10ep_reinit1e4_v1/best_aji_model_clip_textbank.pth`；结果文件未找到 | 否 | HISTORICAL_ABLATION；不能视为严格 encoder 训练对照 |
| PNuDP Dense | Phase C/Exp6 路线（按脚本和命名） | smoke、20-batch、1-epoch | dense projection/logit_add/channel-specific support | UNKNOWN | none | 仅 PNuDP dense proj/logit_proj/alpha（设计） | 无保留 full-test 数值 | `workdir/runs/pnudp_*`；当前 `workdir/models` 无 PNuDP checkpoint | 否 | SMOKE_ONLY |
| SGA-SB corrected | 计划父 checkpoint 未固定 | 未训练 | parser 有 none/supervision_only/guidance 与 branch | 计划 both | 训练构造实际恒为 none | 新模块未解冻、未进 optimizer | 无 | 代码与 target audit：`segment_anything/modeling/sam.py`、`mask_decoder.py`、`training/spatial_sb_targets.py`、`workdir/audits/spatial_sb_targets_corrected/` | 否 | IMPLEMENTED_NOT_TRAINED |
| P1-2 CONCH 语义原型可分性 | 无（纯推理） | text-space diagnostic / 2026-08-07 / 无 seed | Set-A、Set-B、global27；固定 C1–C5；预注册规则选择 Set-A/V1 | 不适用 | 不适用 | 无可训练模块；`training_started=false` | 不适用；仅使用文本 embedding 几何量，`segmentation_metrics_used=false` | `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/`；冻结库 `/hy-tmp/NuSeg/workdir/rsgr_bank/` | 不适用；非分割实验 | Set-A/V0=`PASS`；`GO` 仅表示文本几何门通过；loader 接线及 Appendix-C 尚未执行 |

## P1-2 CONCH 语义原型可分性诊断（纯推理）

### 执行口径与输入

- 执行日期：2026-08-07；远端仓库：`/hy-tmp/NuSeg`，commit `d9ef588ad2386067b29c45c3a8ebe24a5c37d33f`。
- 状态锁：`training_started=false`、`segmentation_metrics_used=false`。本记录只依据文本 embedding 的余弦、有效秩、等级轴与单调性，不以任何分割指标选择 prompt 或几何变体。
- `nvidia-smi` 返回 `No devices were found`，按项目规则记为 `NO_GPU_MODE`；全部正式诊断均为 CPU 纯推理，没有启动训练。
- CONCH 权重：`/hy-tmp/NuSeg/hf_cache/hub/models--MahmoodLab--conch/blobs/40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`，SHA256 `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`。
- Local-5 schema：`/hy-tmp/NuSeg/training/rsgr_local5_schema.json`，SHA256 `01c8dfc779811592207df7b678b84bb192a42aebd00b18748eb09e24d0126e79`；global27 模板：`/hy-tmp/NuSeg/workdir/attr_stats/structure_boundary_prompt_templates.json`，SHA256 `4b166663963ac6063c1dd81846754da3fe51f5ba8bf63bb33840794ac87ba66d`。
- 编码复用项目契约：tokenizer `max_length=77`，`encode_text().float()` 后 `F.normalize(dim=-1, eps=1e-8)`；调用点 `/hy-tmp/NuSeg/audit_probes/probe_conch_separability.py:1072`，并核对 `tools/build_l1a_text_prototype_bank.py:53-73` 与 `segment_anything/modeling/sam.py:3115-3134,3226-3234`。

### 预注册判据与主判定

固定阈值未在结果后改动：C1=`intra_attr_cos > 0.95`；C2=`eff_rank_95 < 5`；C3=`level_axis_alignment > 0.90`；C4=`monotonic_ratio < 0.8`；C5=`separation >= 0`。多重失败按 F1 > F3 > F2 > F4 分类。

Set-A/V0：`intra=0.737689538`、`inter=0.466710191`、`separation=-0.270979347`、`rank95=10`、`rank90=8`、`s1_energy=0.273254093`、`axis_alignment=-0.021249392`、`monotonic_ratio=1.0`。C1=false、C2=false、C3=false、C4=false、C5=false，因此主判定为 `PASS`。其奇异值（4 位有效数字）为 `[1.376, 1.184, 1.003, 0.7726, 0.7403, 0.5966, 0.5663, 0.5142, 0.4212, 0.3218, 0.3057, 0.2809, 0.2511, 0.1503, 3.746e-16]`；五属性的 `t` 分别为 `0.717593469 / 0.541318806 / 0.576431123 / 0.925157115 / 0.871955102`。

按“先排除触发 C1–C5 的变体，再最大化 rank95；并列时取 separation 最负者”的固定规则，最终选择 **Set-A/V1**：`intra=0.433847491`、`inter=-0.151658896`、`separation=-0.585506387`、`rank95=10`、`rank90=8`、`s1_energy=0.234700492`、`axis_alignment=0.002301823`、`monotonic_ratio=1.0`，C1–C5 全为 false。

### 三套 prompt 的完整变体摘要

| prompt set | 变体 | intra | inter | separation | rank95 | rank90 | s1 energy | axis alignment | monotonic | 触发判据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Set-A | V0 | 0.737690 | 0.466710 | -0.270979 | 10 | 8 | 0.273254 | -0.021249 | 1.0 | 无 |
| Set-A | V1 | 0.433847 | -0.151659 | -0.585506 | 10 | 8 | 0.234700 | 0.002302 | 1.0 | 无 |
| Set-A | V2_k1 | 0.257498 | -0.124630 | -0.382128 | 9 | 8 | 0.263243 | -0.041222 | 1.0 | 无 |
| Set-A | V2_k2 | 0.100839 | -0.095130 | -0.195969 | 10 | 8 | 0.283481 | -0.034347 | 1.0 | 无 |
| Set-A | V3 | -0.481935 | -0.000522 | 0.481413 | 9 | 8 | 0.177806 | -0.006458 | 1.0 | C5 |
| Set-A | V4 | -0.333333 | -0.000000 | 0.333333 | 5 | 5 | 0.240303 | -0.021249 | 1.0 | C5 |
| Set-B | V0 | 0.587545 | 0.529752 | -0.057793 | 12 | 10 | 0.189996 | 0.049403 | 1.0 | 无 |
| Set-B | V1 | 0.034129 | -0.086728 | -0.120857 | 12 | 11 | 0.154383 | 0.057311 | 1.0 | 无 |
| Set-B | V2_k1 | -0.071150 | -0.070178 | 0.000973 | 12 | 10 | 0.163573 | 0.081269 | 1.0 | C5 |
| Set-B | V2_k2 | -0.106560 | -0.064442 | 0.042118 | 11 | 9 | 0.169316 | 0.157282 | 1.0 | C5 |
| Set-B | V3 | -0.496413 | -0.000110 | 0.496302 | 9 | 8 | 0.188582 | 0.050971 | 1.0 | C5 |
| Set-B | V4 | -0.333333 | -0.000000 | 0.333333 | 5 | 4 | 0.279222 | 0.049403 | 1.0 | C5 |
| global27 | V0 | 0.554903 | 0.372979 | -0.181924 | 19 | 15 | 0.157703 | 0.052686 | 1.0 | 无 |
| global27 | V1 | 0.259571 | -0.062808 | -0.322378 | 19 | 15 | 0.165538 | 0.062257 | 1.0 | 无 |
| global27 | V2_k1 | 0.189605 | -0.056079 | -0.245684 | 19 | 15 | 0.149363 | 0.057298 | 1.0 | 无 |
| global27 | V2_k2 | 0.085351 | -0.046204 | -0.131555 | 18 | 16 | 0.119069 | 0.039985 | 1.0 | 无 |
| global27 | V3 | -0.491425 | 0.000297 | 0.491722 | 15 | 13 | 0.148946 | 0.051983 | 1.0 | C5 |
| global27 | V4 | -0.333333 | -0.000000 | 0.333333 | 8 | 7 | 0.220712 | 0.052686 | 1.0 | C5 |

### 其余变体的 D2 奇异值与 D4 完整 t-vector

上文已完整记录 Set-A/V0。以下奇异值均来自各自 `metrics.json` 的 `singular_values_4_significant_digits`。Set-A/Set-B 的 t-vector 顺序固定为 `[nuclear_density, nuclear_size_heterogeneity, spatial_crowding, nuclear_irregularity, nuclear_elongation]`；global27 的顺序固定为 `[nuclear_density, nuclear_area_fraction, mean_nuclear_size, nuclear_size_heterogeneity, spatial_crowding, boundary_density, nuclear_irregularity, nuclear_elongation, small_nuclei_ratio]`。

- Set-A/V1：`s=[1.873, 1.818, 1.531, 1.114, 0.9914, 0.9442, 0.8931, 0.7884, 0.6435, 0.5058, 0.4753, 0.4192, 0.369, 0.2256, 3.601e-16]`；`t=[0.7142903955344039, 0.6047715680051761, 0.7417910656060677, 0.9381326112734244, 0.8736329567634503]`。
- Set-A/V2_k1：`s=[1.986, 1.706, 1.339, 1.212, 1.096, 1.045, 0.9218, 0.7747, 0.6137, 0.5325, 0.4779, 0.4144, 0.2486, 7.617e-16, 3.174e-16]`；`t=[0.6841298178770138, 0.7302750027161309, 0.7717514671055503, 0.9394419804174413, 0.9063292561210115]`。
- Set-A/V2_k2：`s=[2.058, 1.458, 1.393, 1.213, 1.155, 1.085, 0.9298, 0.7528, 0.648, 0.6131, 0.5544, 0.3486, 1.024e-15, 7.532e-16, 4.407e-16]`；`t=[0.6972914224016279, 0.6794535480740356, 0.6950310520573989, 0.79660690984214, 0.8828144010061224]`。
- Set-A/V3：`s=[1.631, 1.521, 1.45, 1.348, 1.274, 1.186, 1.124, 0.9382, 0.7742, 0.5509, 5.607e-16, 4.994e-16, 4.073e-16, 3.615e-16, 1.707e-16]`；`t=[0.6810272623879475, 0.523824445178033, 0.5422078173438474, 0.6899909354730891, 0.7458162218934764]`。
- Set-A/V4：`s=[1.55, 1.452, 1.415, 1.354, 1.285, 6.024e-16, 4.543e-16, 4.04e-16, 2.095e-16, 8.235e-17, 6.157e-17, 1.967e-28, 1.681e-32, 1.534e-33, 0]`；`t=[0.499999999999875, 0.49999999999987504, 0.499999999999875, 0.499999999999875, 0.49999999999987504]`。
- Set-B/V0：`s=[1.109, 0.9668, 0.925, 0.8079, 0.6976, 0.6443, 0.6266, 0.5802, 0.5491, 0.4937, 0.4466, 0.3965, 0.3807, 0.3434, 4.494e-16]`；`t=[0.39177372163309726, 0.6286939373893136, 0.5533876532957941, 0.5090318439039121, 0.36732381555168103]`。
- Set-B/V1：`s=[1.52, 1.436, 1.407, 1.277, 1.102, 1.066, 0.9543, 0.9137, 0.8293, 0.7859, 0.716, 0.6411, 0.6038, 0.5483, 3.385e-16]`；`t=[0.49580361613970103, 0.6980784599957431, 0.6042260051699713, 0.5930439085485265, 0.48605983789583884]`。
- Set-B/V2_k1：`s=[1.566, 1.537, 1.359, 1.238, 1.113, 1.078, 1.004, 0.9315, 0.8059, 0.7703, 0.6721, 0.6694, 0.6123, 1.163e-15, 4.097e-16]`；`t=[0.22807021179196776, 0.7163264039972684, 0.6272468895483517, 0.40432081593538094, 0.48648646049918426]`。
- Set-B/V2_k2：`s=[1.593, 1.497, 1.463, 1.227, 1.185, 1.083, 1.027, 0.8693, 0.8248, 0.7308, 0.7271, 0.6573, 1.555e-15, 9.914e-16, 3.108e-16]`；`t=[0.394069195615407, 0.7288382565586321, 0.7082365091282639, 0.3674964862148343, 0.44993751790461384]`。
- Set-B/V3：`s=[1.682, 1.616, 1.442, 1.301, 1.163, 1.118, 1.071, 0.8983, 0.7907, 0.7744, 5.255e-16, 4.019e-16, 3.167e-16, 2.549e-16, 1.66e-16]`；`t=[0.4403030649135277, 0.5841666968527043, 0.526330989143669, 0.5059476671854717, 0.4292241046167771]`。
- Set-B/V4：`s=[1.671, 1.583, 1.51, 1.201, 0.9901, 2.662e-16, 1.949e-16, 1.899e-16, 1.205e-16, 6.633e-17, 6.085e-17, 2.178e-29, 7.167e-33, 6.256e-35, 0]`；`t=[0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875]`。
- global27/V0：`s=[1.585, 1.414, 1.245, 1.11, 1.055, 0.9859, 0.949, 0.9421, 0.8414, 0.75, 0.6851, 0.6538, 0.6283, 0.5987, 0.5594, 0.5478, 0.489, 0.4408, 0.4088, 0.3927, 0.3768, 0.3213, 0.2974, 0.2649, 0.231, 0.1665, 6.065e-16]`；`t=[0.34221711859822174, 0.47098103936156266, 0.8249557886225733, 0.48762521329122094, 0.6743025807138044, 0.6703451151636025, 0.380503999097779, 0.501663153158769, 0.5339339831525268]`。
- global27/V1：`s=[2.114, 1.822, 1.617, 1.439, 1.349, 1.268, 1.233, 1.165, 1.087, 0.9926, 0.8896, 0.8568, 0.8194, 0.7884, 0.7329, 0.6918, 0.6504, 0.5725, 0.5687, 0.5229, 0.4956, 0.4281, 0.3971, 0.3437, 0.3215, 0.2345, 4.243e-16]`；`t=[0.3821966347881943, 0.5424114425009157, 0.7635677810571383, 0.6684254550356183, 0.4606014467298819, 0.6440778470585107, 0.3597199587379617, 0.47992293766305516, 0.7066334060844051]`。
- global27/V2_k1：`s=[2.007, 1.705, 1.608, 1.513, 1.353, 1.309, 1.272, 1.213, 1.095, 0.9724, 0.9487, 0.8859, 0.8644, 0.8268, 0.7632, 0.7067, 0.6454, 0.5984, 0.5775, 0.5469, 0.4594, 0.4411, 0.3995, 0.3617, 0.2792, 3.989e-15, 3.368e-16]`；`t=[0.4048303468998168, 0.6664923538588869, 0.8323976811989864, 0.6683250074369984, 0.48794821141323225, 0.6526801892308921, 0.4408673171126953, 0.4524870009605419, 0.6954700818288259]`。
- global27/V2_k2：`s=[1.791, 1.744, 1.708, 1.474, 1.401, 1.363, 1.305, 1.165, 1.044, 1.016, 0.9963, 0.9387, 0.8884, 0.8064, 0.7587, 0.7431, 0.6948, 0.6226, 0.5906, 0.4959, 0.4804, 0.4402, 0.4099, 0.3089, 4.996e-15, 2.009e-15, 3.691e-16]`；`t=[0.4864644764761516, 0.6671930545507082, 0.7858012612636379, 0.43972082483380937, 0.48146048461462565, 0.6888147666731101, 0.41404853867512803, 0.41913575585810553, 0.7000019495452217]`。
- global27/V3：`s=[2.004, 1.807, 1.589, 1.432, 1.396, 1.361, 1.298, 1.202, 1.182, 1.083, 1.054, 1.021, 0.8868, 0.8569, 0.822, 0.7578, 0.6456, 0.5286, 6.908e-16, 6.351e-16, 5.046e-16, 4.563e-16, 3.869e-16, 3.68e-16, 3.373e-16, 2.802e-16, 2.441e-16]`；`t=[0.4451092022333833, 0.4857390453435347, 0.5716222749183207, 0.49357451604806096, 0.5602016017594484, 0.6087118279367058, 0.4277523230370358, 0.5005846342895507, 0.5150613695651276]`。
- global27/V4：`s=[1.993, 1.684, 1.565, 1.55, 1.336, 1.31, 1.121, 0.9706, 0.8011, 6.322e-16, 4.874e-16, 4.014e-16, 3.632e-16, 3.42e-16, 2.444e-16, 1.917e-16, 1.591e-16, 1.104e-16, 9.707e-17, 2.142e-17, 4.736e-18, 2.656e-32, 1.64e-32, 5.563e-33, 3.579e-33, 7.903e-50, 0]`；`t=[0.49999999999987504, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875, 0.499999999999875]`。

V4 的 mid 残差结论为不适合采用：Set-A 五属性残差比分别为 `0.948711 / 0.998278 / 0.998177 / 0.964264 / 0.927199`（max `0.998278`）；Set-B 为 `0.985735 / 0.981834 / 0.998190 / 0.999926 / 0.986877`（max `0.999926`）；global27 为 `0.992423 / 0.999229 / 0.983234 / 0.999839 / 0.975378 / 0.963641 / 0.981457 / 0.999998 / 0.999095`（max `0.999998`）。三套集合的每个属性均超过预注册 50% 损失线。

字面量对照 `"high nuclear density"` 与 `"low nuclear density"` 的余弦为 `0.7108526033682374`。其他病理/医学文本编码器不可用，跨编码器对照记为 `SKIPPED_NO_ACCESS`。

### 低内存执行、等价性与失败证据

服务器 cgroup 内存上限为 2 GiB，标准 CONCH factory 加载的 Set-A、Set-B、global27 尝试在产生配置前被 OOM 终止；失败证据保留在 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/logs/{set_a,set_b,global27}.log`，三者 SHA256 均为 `bb7033150454bc9e48dccab3f44668be1523ea60d494a1eb02921ffd05bba125`。随后仅改变权重存储加载方式为 `meta_init_plus_read_only_mmap_assign`，编码、tokenizer、context length 与归一化契约不变。

两次低内存修正前失败也原样保留：`set_a_lowmem.log`（未加载的 `text_decoder.*` meta tensor，SHA256 `18f1249dda86547a7790ddb67b30b36cf5875510bd6064b3e264464fcb7973c5`）与 `set_a_lowmem_v2.log`（CPU/meta device mismatch，SHA256 `6fba38c7578b37fdb0532ff75dbeaaa2804e7b960fa20f678606a4a249d77391`）。最终只允许 checkpoint 本就不含且 `encode_text` 不使用的 `text_decoder.*` 缺失项，其他 missing/unexpected/meta 项继续硬失败。

最终 Set-A 原始 embedding 与既有 L1A bank `/hy-tmp/NuSeg/workdir/audits/local_region_text_l1a_20260722/L1A_TEXT_PROTOTYPE_BANK.pt`（SHA256 `f02d6d99d3059a5b62aa096560c5289ae6f4d2036b28cd2ed36a2b301221dcb4`）的 parity：shape 均为 `[5,3,512]`，`max_abs=1.1920928955078125e-07`、`mean_abs=4.1066101630349294e-09`、`cos_min=0.9999998807907104`、`cos_mean=1.0`。8 个 CPU 单元测试通过；日志 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/logs/unit_tests.log`，SHA256 `4ef9868f1ba6a8d82c7879b82e1b344700a0309a015fb81bad48ab7716ad809b`。

### 可复核产物与冻结库

- Set-A summary：`/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3/summary.json`，SHA256 `315f2a2fb9b5a6c4286b9887f9de0d83828821d1def08309acfa1f1341c94d86`；V0 matrix/metrics：`V0/cosine_matrix.csv` `4d568034ef0021bbc9cd76e9f3b2da3bc8856d2192c2980a036a07c364caccac`、`V0/metrics.json` `bd9d28a2866bc4ae15013e4bd666e9907a96d73e433501fe99d648a625fae798`；所选 V1 matrix/metrics：`V1/cosine_matrix.csv` `d74031a067b0806052dde35cc3c2df0121d69157b14debf980be2ecdafbc43a6`、`V1/metrics.json` `e16659822a6bd8f4bc0fd75fb0c449aca6e367f3e901854d7dc86571c5fe393d`。
- Set-B summary：`/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_b_v3/summary.json`，SHA256 `24f48e2a4ad938924a396e1253259bea9264f71277035018357fdc0f36ef73c7`；V0 matrix/metrics：`V0/cosine_matrix.csv` `7494a8b4c8c3365a2e501968c997f88c69ab31c3bfc168516836b471c6183bad`、`V0/metrics.json` `232b2fee8cbf7c9863866f130f21e71debc018a0d7e6c077ffd3a4d61ef55a21`。
- global27 summary：`/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/global27_v3/summary.json`，SHA256 `10acdcef78b1077671670598e59b8ec195680b5d0ee1899f9e777b8f556b92fe`；V0 matrix/metrics：`V0/cosine_matrix.csv` `42291baa64d502d97997e828086c33ee1618243e8f7132136f9b3eb6e6c50ea1`、`V0/metrics.json` `1c882a895ca596d3fba9433823ecec74d44f3fd574bfb07d07ba4db1dc73c470`。
- 全变体目录索引：Set-A 为 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_a_v3/{V0,V1,V2_k1,V2_k2,V3,V4}/`，Set-B 为 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/set_b_v3/{V0,V1,V2_k1,V2_k2,V3,V4}/`，global27 为 `/hy-tmp/NuSeg/workdir/audits/conch_separability_p1_2/global27_v3/{V0,V1,V2_k1,V2_k2,V3,V4}/`。每个变体目录均含 `metrics.json`、`level_axis_cosine_matrix.csv` 与完整 `cosine_matrix.csv`；逐文件 SHA256 位于对应 summary 的 `artifact_sha256`，以上三份固定 summary SHA256 构成不可变索引。
- 冻结路径 `/hy-tmp/NuSeg/workdir/rsgr_bank/`：`prompts_frozen.json` SHA256 `de4413374061d3886fc87288ff48c46ea5f07d00268aaf191c7328d74f55eaa3`；`structure_bank.pt` SHA256 `ca28900b8650ec49974da776bdc2bef0e9408f42421e6f7aee5d4a32a34786a8`；`boundary_bank.pt` SHA256 `cb5cfb2d79d05cbeeef28efa5a25bb1252b287ed497c231929d8447308aeea0d`；`bank_manifest.json` SHA256 `a10944ad06cffdf70742c93ed2c6570ec32b8810ea77ac013faacd04c0cab7f1`。manifest 固定 `prompt_set=Set-A`、`geometric_variant=V1`、embedding dim 512、schema 属性顺序与 `low/medium/high` 等级顺序。

冻结 loader 集成状态：`APPENDIX_C_LOADER_INTEGRATION_PENDING`。当前 `segment_anything/modeling/rsgr.py:169-205` 的 `load_prototype_banks` 要求单一 formal CONCH mapping `.pt`（同时含 `structure_prototypes` 与 `boundary_prototypes`）及同名 `.metadata.json`，而本次冻结交付是拆分的 `structure_bank.pt`、`boundary_bank.pt`、`prompts_frozen.json`、`bank_manifest.json` 四件套，不能直接作为现有 `--rsgr_prototype_path` 的输入。Appendix-C 启动前必须增加确定性的转换/接线并通过严格 loader round-trip 校验；本任务未做该接线，也未启动训练。

结论：**GO（仅表示 P1-2 文本几何先决门通过）**。Set-A/V0 通过主判定并按预注册规则冻结 Set-A/V1；这不等于 RSGR-1 训练入口已就绪。Appendix-C 的 loader 接线、因果训练与对照均尚未执行。

## Exp5 与 Exp6 optimizer 差异

Exp5 checkpoint 的 optimizer 含 `numeric_attr_freqpath_proj`（12 个参数张量），无 PG3/attr-align 组；Exp6 含 prompt learner、4 组 PG3 adapter/projection，且父 checkpoint 含 8 个 attr-align 键，无 numeric projection。两者不是单变量文本对照。

## checkpoint 污染记录

Exp6 目录：

- `best_aji_model.pth.bak`：没有 text-bank buffer；
- `best_aji_model.pth.clip_bak`：有原先 bank，但无 CLIP metadata；
- 当前 `best_aji_model.pth`：含 `clip/ViT-B/32` metadata；
- `best_aji_model_clip_textbank.pth`：与当前文件大小和元数据一致。

测试 Exp6/Exp7 前必须按哈希固定具体文件，不能只写 `best_aji_model.pth`。

## 固定数值来源状态

仓库全文搜索未找到四组 full-test 数值的完整原始输出。Visual baseline 的 `metrics_history.json` 只出现 Dice 0.8089036 的 validation 记录，其他固定指标不成套；这不足以认定为同一 full test。故四组固定记录均保留为 `PROVIDED_CONTEXT_UNVERIFIED`。
