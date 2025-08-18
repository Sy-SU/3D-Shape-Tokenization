# 3D Shape Tokenization 复现报告

## 1. 实验背景

3D Shape Tokenization 是 Apple 公司在 ["3D Shape Tokenization via Latent Flow Matching"](https://arxiv.org/abs/2412.15618) 中提出的一种将 3D 形状转换为低维表示的方法，并称之为 Shape Token。

在传统的 3D 形状表示方法中，3D 形状通常被表示为点云、网格或者是体素等形式。但是这些表示方法难以同时满足连续、紧凑与预处理少这三个要求 (TODO: 为什么?)。于是，Apple 公司提出了一种通过flow matching学习3D形状的**共享潜空间**的方法来满足以上特性。根据论文报告，这种表示方法除了满足连续、紧凑等特性，同时只需要从3D形状的表面采样一定的点云，此外还具有一定的几何能力 (TODO: 具体是什么?)。

## 2. 数据集简介

### 2.1 ShapeNet



## 3. 实验方法

## 4. 实验结果

## 5. 分析与讨论

## 6. 附录

### 6.1 复现指南

### 6.2 代码结构

### 6.3 参考资料