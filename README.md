# 3D-Shape-Tokenization-Repro

本项目是关于 3D Shape Tokenization 的复现工作，旨在通过 Latent Flow Matching 方法对 3D 形状进行编码和解码。

---

## 📁 项目结构

```
./
├── .gitignore
├── LICENSE
├── README.md
├── checkpoints                           # 模型权重
│   ├── best_estimator.pt                 
│   └── best_tokenizer.pt
├── configs
│   └── label.json                        # 数据集中子目录对应的类别
├── data
│   └── taxonomy.json                     # 原始数据集类别标签
├── datasets
│   ├── __init__.py
│   └── dataloader.py
├── models
│   ├── __init__.py
│   ├── shape_tokenizer.py
│   └── velocity_estimator.py
├── requirements.txt
├── tools
│   ├── eval_reconstruct.py               # 重建评估脚本
│   ├── train.py                          # 训练脚本
│   └── visualize.py                      # 可视化脚本
└── utils
    ├── __init__.py
    └── utils.py
```

---

## 🧠 模型简介

本项目复现了 Latent Flow Matching 方法，用于对 3D 形状进行编码和解码。具体来说，我们使用一个形状编码器将 3D 形状编码为潜在表示，然后使用一个速度估计器将潜在表示解码为形状的速度。这种方法可以有效地捕捉形状的动态特性，并在重建过程中保持形状的连续性和一致性。

---

## 📈 实验结果

TODO

---


## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:Sy-SU/3D-Shape-Tokenization.git
cd 3D-Shape-Tokenization
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

使用 ShapeNetCore.v2.PC15k 作为数据集, 下载链接: https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j

将数据集解压到 `data/` 目录下，目录结构如下：

```
./
├── data
│   ├── ShapeNetCore.v2.PC15k
│   │   ├── 02691156
```

### 4. 训练模型

```bash
python tools/train.py
```

### 5. 评估模型

```bash
python tools/eval_reconstruct.py --weights checkpoints/best_model.pth
```

---

## ⚙️ 配置说明 (TODO)

示例 `config.yaml`：

```yaml
model:
  name: resnet18
  pretrained: true

train:
  epochs: 50
  batch_size: 64
  lr: 0.001

data:
  dataset: CIFAR10
  path: ./data
```

---

## 📊 实验结果 (TODO)



---

## 🛠️ TODO

* [x] 基本训练脚本
* [ ] 完成基本调参, 实现完整的训练流程, 发布模型权重
* [ ] 支持 TensorBoard
* [ ] 将模型与 3d-CLIP 结合

