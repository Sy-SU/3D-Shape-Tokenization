# 3D Shape Tokenization 复现报告

## 1. 实验背景

3D Shape Tokenization 是 Apple 公司在 ["3D Shape Tokenization via Latent Flow Matching"](https://arxiv.org/abs/2412.15618) 中提出的一种将 3D 形状转换为低维表示的方法，并称之为 Shape Token。

在传统的 3D 形状表示方法中，3D 形状通常被表示为点云、网格或者是体素等形式。但是这些表示方法难以同时满足连续、紧凑与预处理少这三个要求 (TODO: 为什么?)。于是，Apple 公司提出了一种通过flow matching学习3D形状的**共享潜空间**的方法来满足以上特性。根据论文报告，这种表示方法除了满足连续、紧凑等特性，同时只需要从3D形状的表面采样一定的点云，此外还具有一定的几何能力 (TODO: 具体是什么?)。

## 2. 数据集简介

### 2.1 ShapeNetCore

在本实验中，我们选取了 ShapeNetCore 作为数据集。该数据集具有约 50,000 个 3D 形状, 包括桌子、椅子、飞机等约 50 个类别，我们选用的是 [ShapeNetCore.v2.PC15k](https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j) 版本, 其中的 3D 形状以点云形式给出。我们按照数据集中的划分方式完成训练和测试.

### 2.2 Demo

在复现的初步阶段，为了在算力不足的情况下验证模块的效果，我们选取了其中的 4 个类别，约 20,000 个形状进行训练和测试。

### 2.3 数据集格式

数据集的格式如下：

```txt
ShapeNetCore.v2.PC15k/
│
├── 02691156/
│   ├── train/
│   │   ├── xxx.npy
│   │   ├── yyy.npy
│   │   └── ...
│   ├── val/
│   │   ├── ...
│   ├── test/
│   │   ├── ...
│   └── ...
│
├── 02958343/
│   ├── train/
│   ├── val/
│   ├── test/
│
└── ...
```

Demo 数据集包含的类别为: `02691156`, `02747177`, `03001627`, `04379243`。


## 3. 实验方法

### 3.1 Shape Tokenizer

#### 3.1.1 简介

Shape Tokenizer 相当于模型的编码器，它的目标是将 3D 形状 (本实验中为点云) 映射到低维的潜在表示，这些潜在表示也将用作后续 Velocity Estimator 的输入，它们用于描述 3D 形状的概率分布。

#### 3.1.2 模型结构

Shape Tokenizer 的结构如下图:

![Shape Tokenizer](assets/shape_tokenizer.png)

Shape Tokenizer 将点云编码为 $k$ 个 $d$ 维的 Shape Token。在 Shape Tokenizer 内部，Shape Token 的维度为 $d_f$。

初始状态下，我们随机生成 $k$ 个 $d_f$ 维的 Shape Token，同时通过位置编码器将输入点云中的 $n$ 个点编码为 $d'$ 维的向量。我们使用 Shape Tokens 作为交叉注意力模块中的查询，位置编码作为键值，从点云中提取全局几何特征。同时使用残差连接、归一化和 MLP 来增强训练的稳定性, 经过交叉注意力模块后, 我们得到 $k$ 个 $d_f$ 维的 Shape Token。接着我们采用 2 个堆叠的自注意力模块，建立 Token 之间的依赖关系. 我们选择重复上述模块 6 次以达到细化 Token 表示的目的。最终经过一个线性层，我们得到 $k$ 个 $d$ 维的 Shape Token。它将作为后续 Velocity Estimator 的输入。

### 3.2 Velocity Estimator

#### 3.2.1 简介

Velocity Estimator 是模型的解码器，它的目标是将 Shape Token 解码为 3D 形状的表面点云。Velocity Estimator 的输入为 Shape Token、经过位置编码的点以及经过位置编码的时间步。**这是因为 Velocity Estimator 本质上是在学习每个点随时间步的运动情况。**

#### 3.2.2 模型结构

Velocity Estimator 的结构如下图:

![Velocity Estimator](assets/velocity_estimator.png)

Velocity Estimator 接收 Shape Tokenizer 编码的 $k$ 个 $d$ 维的 Shape Token，同时通过位置编码器将当前时间步编码为 $d''$ 维的向量，将当前点的位置编码为 $d'$ 的向量。时间编码 $t_{emb}$ 首先经过线性层和 SiLU 激活函数，然后生成 Shift 向量和 Scale 向量，去指导位置编码 $x_{emb}$ 经过线性层与归一化层（自适应归一化层）后的调制，调制的结果作为交叉注意力模块的询问，Shape Tokens 作为键值，得到的结果经过 gating 后与归一化后的 $x_{emb}$ 进行残差连接，然后将交叉注意力模块换成一个多层感知机，重复相似过程。最终重复上述模块 3 次，得到在当前时间步下，特定点在 Shape Token 下的运动方向。

> 在实现中, $d''$ 与 $d$ 的维度相同。

> **自适应归一化层的结构仍需继续确认。**

### 3.3 推理过程

### 3.4 损失函数

## 4. 实验结果

## 5. 分析与讨论

## 6. 附录

### 6.1 复现指南

### 6.2 代码结构

### 6.3 参考资料