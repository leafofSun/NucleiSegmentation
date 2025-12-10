# 全局标签转换说明

## 概述

`global_label_*.json` 文件（来自 PromptNu 项目）需要转换为 `attribute_info_*.json` 格式，以便 `DataLoader.py` 和 PNuRL 模块使用。

## 格式对比

### PromptNu 格式 (`global_label_*.json`)

```json
[
  {
    "id": ["image_00"],
    "color": ["deep purple"],
    "size": ["small"],
    "density": ["moderately dense"],
    "arrange": ["scattered"],
    "shape": ["elliptical/oval"]
  }
]
```

### PNuRL 格式 (`attribute_info_*.json`)

```json
{
  "image_00": {
    "attribute_prompts": [
      "deep purple",      // 颜色
      "elliptical/oval",  // 形状
      "scattered",        // 排列
      "small",           // 大小
      "moderately dense"  // 分布
    ],
    "attribute_labels": [
      [1, 0, 0],         // 颜色 one-hot (3类)
      [1, 0, 0, 0, 0],   // 形状 one-hot (5类)
      [1, 0, 0, 0],      // 排列 one-hot (4类)
      [1, 0, 0],         // 大小 one-hot (3类)
      [0, 1, 0]          // 分布 one-hot (3类)
    ]
  }
}
```

## 属性映射

| 属性维度 | PromptNu 键名 | PNuRL 顺序 | 类别数 | 示例值 |
|---------|--------------|-----------|--------|--------|
| **颜色** | `color` | 0 | 3 | "deep purple", "light pink" |
| **形状** | `shape` | 1 | 5 | "elliptical/oval", "spindle", "irregular" |
| **排列** | `arrange` | 2 | 4 | "scattered", "clustered", "linear" |
| **大小** | `size` | 3 | 3 | "small", "medium", "large" |
| **分布** | `density` | 4 | 3 | "sparsely distributed", "moderately dense", "densely packed" |

## 使用方法

### 1. 转换训练集标签

```bash
python convert_global_labels.py \
    --input data/global_labels/global_label_cpm17.json \
    --output data/cpm17/attribute_info_train.json \
    --num_classes 3 5 4 3 3
```

### 2. 转换测试集标签（如果有）

```bash
python convert_global_labels.py \
    --input data/global_labels/global_label_cpm17_test.json \
    --output data/cpm17/attribute_info_test.json \
    --num_classes 3 5 4 3 3
```

### 3. 在训练/测试中使用

转换后的文件会自动被 `DataLoader.py` 加载：

```bash
# 训练
python train.py \
    --data_path data/cpm17 \
    --use_pnurl \
    # attribute_info_train.json 会自动从 data/cpm17/ 目录加载

# 测试
python test.py \
    --data_path data/cpm17 \
    --use_pnurl \
    # attribute_info_test.json 会自动从 data/cpm17/ 目录加载
```

## 自定义属性映射

如果需要修改属性类别映射，请编辑 `convert_global_labels.py` 中的 `ATTRIBUTE_MAPPINGS` 字典：

```python
ATTRIBUTE_MAPPINGS = {
    'color': {
        'deep purple': 0,
        'light pink': 1,
        'other': 2,
    },
    # ... 其他属性
}
```

## 注意事项

1. **属性顺序**：`attribute_prompts` 和 `attribute_labels` 必须按照 `[颜色, 形状, 排列, 大小, 分布]` 的顺序
2. **类别数量**：必须与 PNuRL 的 `num_classes_per_attr` 参数匹配（默认 `[3, 5, 4, 3, 3]`）
3. **图像ID格式**：转换脚本会自动移除图像ID的扩展名（如 `image_00.png` → `image_00`）


