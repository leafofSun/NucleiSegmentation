# NucleiSegmentation

基于 SAM-Med2D 的细胞核分割项目，集成了 PNuRL (Prompting Nuclei Representation Learning) 和 CoOp (Context Optimization) 方法。

## 项目特性

- **SAM-Med2D**: 基于 Segment Anything Model 的医学图像分割框架
- **PNuRL**: 使用属性感知提示增强图像特征表示
- **CoOp集成**: 可学习的语义提示工程，替代或增强传统的点/框提示
- **多指标评估**: 支持 mDice, mAJI, mPQ, mDQ, mSQ 等多种评估指标

## 项目结构

```
.
├── segment_anything/          # SAM核心代码
│   ├── modeling/
│   │   ├── pnurl.py           # PNuRL模块
│   │   ├── pnurl_text_encoder.py  # CoOp文本编码器
│   │   ├── prompt_encoder.py  # 提示编码器（支持文本嵌入）
│   │   ├── sam_model.py       # SAM模型主文件
│   │   └── ...
│   └── ...
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── DataLoader.py              # 数据加载器
├── metrics.py                 # 评估指标
└── data/                      # 数据目录（不包含在git中）
```

## 安装依赖

```bash
pip install torch torchvision
pip install opencv-python
pip install scipy
pip install scikit-image
pip install tqdm
# 安装CLIP（用于CoOp）
cd CLIP/CLIP-main
pip install -e .
```

## 使用方法

### 训练

```bash
python train.py \
    --data_path data/cpm17 \
    --workdir workdir \
    --num_epochs 200 \
    --batch_size 4 \
    --use_pnurl \
    --pnurl_clip_path path/to/clip/weights \
    --pnurl_num_classes 10 \
    --use_coop_prompt \
    --metrics mDice mAJI mPQ mDQ mSQ
```

### 测试

```bash
python test.py \
    --checkpoint workdir/models/best_model.pth \
    --data_path data/cpm17 \
    --use_pnurl \
    --use_coop_prompt \
    --metrics mDice mAJI mPQ mDQ mSQ
```

### 恢复训练

```bash
python train.py \
    --resume workdir/models/checkpoint_epoch_50.pth \
    --data_path data/cpm17 \
    --workdir workdir
```

## 主要功能

### 1. PNuRL (Prompting Nuclei Representation Learning)
- 使用属性感知提示增强ViT特征
- 通过注意力机制对图像特征进行加权
- 支持多属性分类任务

### 2. CoOp集成 (Context Optimization)
- 可学习的语义提示向量
- 基于CLIP的文本编码
- 替代或增强SAM的点/框提示

### 3. 评估指标
- **mDice**: 平均Dice系数
- **mAJI**: 平均聚合Jaccard指数
- **mPQ**: 平均Panoptic Quality
- **mDQ**: 平均Detection Quality
- **mSQ**: 平均Segmentation Quality

## 数据格式

项目支持CPM17数据集格式：
- `data/cpm17/train/Images/`: 训练图像
- `data/cpm17/train/Labels/`: 训练标签（.mat格式）
- `data/cpm17/test/Images/`: 测试图像
- `data/cpm17/test/Labels/`: 测试标签

需要生成 `image2label_train.json` 和 `image2label_test.json` 文件来映射图像和标签。

## 模型保存

- 每50个epoch自动保存最佳模型到 `workdir/models/`
- 支持从checkpoint恢复训练
- Checkpoint包含模型权重、优化器状态和训练进度

## 注意事项

1. 数据文件（图像和掩码）不包含在git仓库中，需要单独下载
2. 模型权重文件（.pth, .pt, .ckpt）不包含在git仓库中
3. 确保有足够的GPU内存（建议至少8GB）
4. 训练和测试时确保数据预处理一致

## 许可证

请参考原始SAM-Med2D项目的许可证。

## 贡献

欢迎提交Issue和Pull Request！

