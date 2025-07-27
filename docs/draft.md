# 3D-Shape-Tokenization

## 总体结构

论文分为两个阶段：

1. **训练 Shape Tokenizer（VQ-VAE 模块）**
2. **训练 Transformer 作为 shape prior（建模 token 序列）**

这两个阶段的输入输出不一样，我们分别来看。

---

### ✅ 阶段一：Shape Tokenizer（VQ-VAE）部分

#### 📥 输入数据：

* 原始的 3D 形状，通常是：

  * mesh / point cloud → 转换为体素（voxel grid）或 Signed Distance Field（SDF）
* 示例格式：

  * 一个大小为 $32 \times 32 \times 32$ 的 voxel grid
  * 每个体素表示 occupancy 或距离

#### 📤 输出数据：

* 重建后的 voxel / SDF grid（与输入同尺寸）
* 中间产生的离散 token ID grid，例如：

  * 每个 voxel 被编码为一个 token index，大小为 $8 \times 8 \times 8$
  * 每个位置是一个整数 $\in [0, K-1]$，K 是 codebook 的大小（例如 512）

---

### ✅ 阶段二：Transformer 模块（建模 token）

#### 📥 输入数据：

* 一个 token grid，flatten 成序列，比如 $z = \{z_1, z_2, ..., z_{512}\}$
* 对于 **shape completion** 或 **denoising**：

  * 输入的是一个 **部分缺失或被 mask 的 token 序列**

#### 📤 输出数据：

* 预测的完整 token 序列（长度等于输入）
* 每个位置是 token 的概率分布或预测的 index（分类任务）

最终，这些 token 会被解码器还原成完整的 voxel shape。

---

### 🧪 举个例子：

| 模块                | 输入                          | 输出                                             |
| ----------------- | --------------------------- | ---------------------------------------------- |
| Tokenizer（VQ-VAE） | $32^3$ voxel grid（float）    | $8^3$ token index grid（int） + recon voxel grid |
| Transformer       | token index 序列（int, 长度 512） | token index 序列（int, 长度 512）                    |
| Decoder（VQ-VAE）   | token index grid（int）       | $32^3$ voxel grid（float）                       |

---

## Todo List

- Shape Tokenization
- transformer 模块, 对 token 建模
- 流匹配 ?

## frame work

- 实现 shape tokenizer 和 decoder, 计算编码和重建能力

##

```txt
your_project/
│
├── config/                # 配置文件（如 YAML，用于训练参数等）
│   └── train.yaml
│
├── data/                  # 数据目录（原始数据、预处理、缓存等）
│   ├── raw/
│   ├── processed/
│   └── splits/            # train/val/test 划分文件
│
├── datasets/              # 自定义 Dataset 类
│   └── voxel_dataset.py
│
├── models/                # 模型定义
│   └── voxelnet.py
│
├── train.py               # 训练主脚本
├── eval.py                # 评估脚本
├── predict.py             # 推理脚本
│
├── losses/                # 自定义 loss 函数
│   └── focal_loss.py
│
├── utils/                 # 工具函数（日志、metrics、可视化等）
│   ├── logger.py
│   ├── metrics.py
│   └── visualization.py
│
├── checkpoints/           # 保存模型参数
│   └── best_model.pth
│
├── logs/                  # TensorBoard 日志或训练输出
│
├── scripts/               # 启动脚本（如 bash/Slurm/Colab）
│   └── train_voxel.sh
│
├── requirements.txt       # Python依赖
├── .gitignore
├── README.md
```