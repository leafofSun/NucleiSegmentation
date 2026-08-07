# SGA-SB v1 代码审计报告

## 1. PanNuke Dataset 当前返回字段

[`UniversalDataset.__getitem__`](NuSeg/DataLoader.py:1734) 返回字典包含以下空间相关字段：

| 字段名 | 类型/形状 | 来源 | 说明 |
|--------|-----------|------|------|
| `label_inst` | `[1, H, W]` int | JSON mask 解码 | **实例 ID 图**（背景=0，前景=1..N） |
| `dense_boundary_map` | `[1, H, W]` float | [`generate_dense_boundary_maps`](NuSeg/DataLoader.py:903) | 边界距离图（Phase B） |
| `dense_touching_region` | `[1, H, W]` float | 同上 | 接触区域图 |
| `dense_small_nuclei` | `[1, H, W]` float | 同上 | 小核图 |
| `dense_hv_gradient` | `[1, H, W]` float | 同上 | H/V 梯度图 |
| `fg_target` | `[1, H, W]` float | [`generate_boundary_uncertainty_targets`](NuSeg/DataLoader.py:810) | 前景边界目标 |
| `bg_target` | `[1, H, W]` float | 同上 | 背景边界目标 |
| `boundary_target` | `[1, H, W]` float | 同上 | 边界区域目标 |
| `uncertain_target` | `[1, H, W]` float | 同上 | 不确定区域目标 |
| `structure_attr_labels` | `[5]` long | JSONL 文件 | **图像级**结构属性标签（5类×3级） |
| `boundary_attr_labels` | `[4]` long | JSONL 文件 | **图像级**边界属性标签（4类×3级） |
| `per_instance_attr_labels` | `[N_i, 6]` long | [`compute_instance_morphology_attrs`](NuSeg/DataLoader.py:1009) | **逐实例**形态属性标签（6属性×3级） |
| `per_instance_attr_values` | `[N_i, 6]` float | 同上 | **逐实例**形态属性连续值 |
| `per_instance_ids` | `[N_i]` long | 同上 | **逐实例**ID（对应 `label_inst`） |

### 关键发现

1. **结构/边界属性只有图像级** - `structure_attr_labels` (5类) 和 `boundary_attr_labels` (4类) 来自 JSONL，对整个图像聚合，没有逐实例信息
2. **逐实例属性只有 6 种形态属性** - `per_instance_attr_labels` (edge, shape, size, texture, color, population) 通过 [`compute_instance_morphology_attrs`](NuSeg/DataLoader.py:1009) 从实例 mask 计算
3. **`label_inst` + `per_instance_ids` 提供了从像素到属性的桥梁** - 对于 SGA-SB v1，可以用 `label_inst` 的实例 ID 查找 `per_instance_attr_labels` 生成逐像素目标

---

## 2. 当前 Mask Decoder / FreqPath Forward 架构

### FreqPathASRBlock 双路径设计

[`FreqPathASRBlock`](NuSeg/segment_anything/modeling/mask_decoder.py:147) 采用低频（结构）+ 高频（边界）双路径：

#### 低频/结构路径（全局向量 → 特征调制）
```
attr_prompt [B, 512] → attr_modulator(Linear+LN+ReLU+Linear+tanh) 
                      → gamma_low_raw [B, out_dim] → clamp [-2, 2]
                      → x_low = x * (low_strength * gamma_low) * spatial_gate + x
```
- `attr_modulator`: `Linear(512→512) → LayerNorm → ReLU → Linear(512→out_dim) → tanh`
- `low_spatial_gate`: `Conv2d(out_dim→hidden,3) → LN → ReLU → Conv2d(hidden→1,1) → Sigmoid`
- `gamma_low` 是 **逐通道（per-channel）** 但 **全局（global）** — 所有空间位置共享同一个 gamma
- `spatial_gate` 是唯一的逐像素组件，但它的输入 `x_before_low` 不含任何结构/边界语义

#### 高频/边界路径（全局向量 → CNN 特征调制）
```
morph_feat [B, 512] → morphology_modulator(Linear+LN+ReLU+Linear+tanh) 
                     → gamma_high [B, cnn_dim] → clamp [-2, 2]
                     → cnn_feat = cnn_feat * (1 + gamma_high)
```
- `morphology_modulator`: `Linear(cnn_dim→cnn_dim) → LN → ReLU → Linear(cnn_dim→cnn_dim) → tanh`
- `gamma_high` 同样是 **全局** 向量

### MaskDecoder.predict_masks

[`MaskDecoder.predict_masks`](NuSeg/segment_anything/modeling/mask_decoder.py:701)：

```
image_embeddings [B,256,64,64]
  → asr_upscale_1 (FreqPathASRBlock): 256→64, CNN dim=512
    → x_up [B,64,128,128] 
    → Low path: attr_prompt [B,512] → per-channel gamma → per-pixel spatial gate
    → High path: morph_feat [B,512] → per-channel gamma → CNN feature modulation
  → asr_upscale_2 (FreqPathASRBlock): 64→32, CNN dim=256
    → Same dual-path architecture
  → MLP → masks [B, 3, 256, 256]
```

### 核心问题

1. **全局向量 → 弱空间影响**：`attr_prompt` / `morph_feat` 通过 GAP 或文本编码产生 `[B,512]` 全局向量，每个空间位置获得相同的调制强度
2. **数值实验证实**：Exp5 中 test-time numeric route on/off 仅带来 ~0.0003-0.0008 Dice 差异
3. **`low_spatial_gate` 可复用**：已存在逐像素门控机制，但其输入不含语义信息

---

## 3. 可复用组件

### ✅ 可直接复用

| 组件 | 位置 | 说明 |
|------|------|------|
| [`FreqPathASRBlock.low_spatial_gate`](NuSeg/segment_anything/modeling/mask_decoder.py:226) | mask_decoder.py:226 | 逐像素卷积门控，可改造为接收空间引导图 |
| [`generate_dense_boundary_maps`](NuSeg/DataLoader.py:903) | DataLoader.py:903 | 生成 4 种密集边界图，可作为空间引导目标 |
| `label_inst` + `per_instance_attr_labels` | DataLoader __getitem__ | 实例 ID → 属性映射，用于逐像素目标生成 |
| `image_embeddings [B,256,64,64]` | sam.py:3804 | 16× 下采样特征，适合作为空间预测头的输入分辨率 |
| `fused_image_embeddings` | sam.py:6091 | 注入语义后的特征，可在 chunk loop 前做空间调制 |

### 🔧 需要扩展/新增

| 组件 | 说明 |
|------|------|
| **`SpatialAttrHead`** | 轻量 CNN：`[B,256,64,64] → [B,18,64,64]`（6属性×3类逐像素logits） |
| **`SpatialSBGuidance`** | 聚合 `[B,18,64,64]` → `[B,1,64,64]` 空间引导图 + Sigmoid |
| **逐像素 CE Loss** | 每个像素预测其所属核的属性类，忽略背景 |
| **空间引导注入** | 将 `[B,1,64,64]` 引导图乘到 `fused_image_embeddings` 上 |
| **无 oracle 泄漏** | GT spatial targets 仅用于 loss 监督，推理用模型预测图 |

### 架构对比

| 维度 | 旧方案（Global） | SGA-SB v1（Spatial） |
|------|-----------------|---------------------|
| 输入 | GAP → [B,512] 向量 | [B,256,64,64] → Conv → [B,18,64,64] |
| 调制方式 | 逐通道 gamma（全图共享） | 逐像素 + 逐通道（混合） |
| 空间分辨率 | 无 | 64×64 → 128×128 → 256×256 |
| 训练监督 | 图像级 CE（5+4 属性） | 逐像素 CE（6 × 3 类） |
| 推理输入 | 文本嵌入 / 全局 logits | 模型预测的逐像素 logits |
