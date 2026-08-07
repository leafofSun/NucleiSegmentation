# NucleiSegmentation — 迁移说明 (my_env 版)

## 1. 新服务器环境准备

```bash
# 创建新 conda 环境（Python 3.12）
conda create -n nuseg python=3.12
conda activate nuseg
```

## 2. 安装 PyTorch（先于其他依赖）

优先复现旧服务器 `my_env` 环境：

```bash
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

> **关于 torchaudio**：`my_env` 中未安装 torchaudio（`NOT INSTALLED`），因此不是必需项。
> 如果后续代码需要音频相关功能，再手动安装：
> ```bash
> pip install torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
> ```
>
> 如果新服务器的 CUDA 版本与旧服务器（CUDA 12.8）不同，请到
> https://pytorch.org/get-started/previous-versions/ 选择对应版本。

## 3. 安装项目核心依赖

```bash
pip install -r requirements_my_env.txt
```

## 4. 安装特殊包

```bash
# CLIP（项目子目录 — 本地源码安装）
pip install -e ./CLIP/CLIP-main

# CONCH（外部 git 仓库 — 病理基础模型）
# 旧服务器安装版本: commit 141cc09
pip install git+https://github.com/Mahmoodlab/CONCH.git
```

## 5. 需要保留的项目文件/目录

从旧服务器复制以下目录到新服务器（保持相对路径一致）：

| 路径 | 说明 |
|------|------|
| `data/` | 训练/测试数据（PanNuke 等） |
| `workdir/models/` | 已训练的模型检查点 |
| `workdir/attr_stats/` | 属性统计数据 |
| `pretrained_pth/` | 预训练权重（如有） |

## 6. 文件说明

| 文件 | 用途 |
|------|------|
| `environment_snapshot_my_env.txt` | `my_env` 环境快照（仅供参考） |
| `requirements_lock_my_env.txt` | `my_env` 完整 `pip freeze`（**仅供参考，不建议直接安装**） |
| `requirements_my_env.txt` | 精简核心依赖（**推荐安装**） |
| `requirements_optional_my_env.txt` | 可选依赖说明（训练报缺时再安装） |
| `migration_notes_my_env.txt` | 本文件 — 迁移步骤 |

## 7. 补充说明

- **`requirements_lock_my_env.txt`** 仅作为旧环境完整依赖的参考快照，其中包含 nvidia-* CUDA 工具包等与项目无关的包，直接安装可能在新服务器上失败。
- **`requirements_my_env.txt`** 是扫描项目代码中所有 `import` 语句后整理的精简依赖，**推荐安装此文件**。
- **`requirements_optional_my_env.txt`** 中的包（如 `timm`、`h5py`）虽然在 `my_env` 中存在，但未被项目代码直接 import。如果训练时报 `ModuleNotFoundError: No module named 'timm'` 或 `'h5py'`，再从 `requirements_optional_my_env.txt` 中安装对应包即可。
- 不要复制 `.conda/` 目录 — 与项目无关。
