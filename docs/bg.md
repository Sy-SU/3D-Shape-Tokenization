# 背景知识

## 位置编码器

### 动机

由于 MLP 与线性层对高频细节的感知能力低，我们需要一种方法提取位置信息，使得模型可以充分感知各种频段的信息。

### 方法

对于给定的位置 $x$ 以及频率的数量 $L$, 我们可以将 $x$ 映射到一个高维空间:

\[
\gamma(x) = (x, \sin(2^0\pi x), \cos(2^0\pi x), \ldots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x))
\]

> TODO: 合理性或许与傅里叶变换有关，待证明

---

## 时间编码器

## 动机

类似地，我们需要一种方法提取时间信息，使得模型可以充分感知时间信息。

## 方法

对于给定的时间 $t$ 以及频率的数量 $L$, 我们可以将 $t$ 映射到一个高维空间:

\[
\gamma(t) = (\sin(2^0\pi t), \cos(2^0\pi t), \ldots, \sin(2^{L-1}\pi t), \cos(2^{L-1}\pi t))
\]

> 在实现过程中，时间编码器与 `Linear+SiLU+Linear` 的结构共同封装在 `TimeEncoder` 内。

---

## 自适应归一化层

在流匹配中，输入不仅包含点的坐标特征 $x$，还依赖额外的条件变量，例如时间步 $t$。不同时间步下，数据的分布差异较大，若仅使用固定参数的标准 Layer Normalization，模型难以兼顾所有时间分布。因此，需要设计一种能够根据时间步动态调制特征分布的归一化方式。

### 标准 Layer Normalization

给定输入 $x \in \mathbb{R}^d$，标准 LayerNorm 计算过程为：

\[
\text{LN}(x) = \frac{x - \mu}{\sigma},
\]

其中 $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$，$\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$。  

为了增强表示能力，通常还会引入可学习的缩放与偏移参数 $\gamma, \beta$：

\[
y = \gamma \cdot \text{LN}(x) + \beta.
\]


### 自适应归一化层

在 AdaLN 中，$\gamma$ 和 $\beta$ 不再是固定的参数，而是由**时间嵌入向量** $t_{\text{emb}}$ 动态生成。  
具体实现方式为：

\[
\gamma = W_{\text{scale}} \, t_{\text{emb}}, \quad \beta = W_{\text{shift}} \, t_{\text{emb}},
\]

从而得到：

\[
\text{AdaLN}(x, t_{\text{emb}}) = \text{LN}(x) \cdot (1 + \gamma) + \beta.
\]

其中 $W_{\text{scale}}$ 与 $W_{\text{shift}}$ 是线性变换矩阵。这样，模型能够根据时间步 $t$ 的不同，灵活地调控特征的分布。

---

## Shift & Scale & Gating

在流匹配（Flow Matching）的 Velocity Estimator 中，除了基本的归一化外，还引入了 **Shift、Scale 和 Gating** 机制，用于根据时间步 $t$ 对特征进行动态调制。

### 1. Shift & Scale

对于输入特征 $x \in \mathbb{R}^d$，首先通过 LayerNorm 或 AdaLayerNorm 得到归一化结果 $x_{\text{norm}}$。  
接着利用时间嵌入 $t_{\text{emb}}$ 生成两个向量：
\[
\text{shift} = W_{\text{shift}} \, t_{\text{emb}}, 
\quad
\text{scale} = W_{\text{scale}} \, t_{\text{emb}},
\]

然后对归一化后的特征进行调制：
\[
x' = x_{\text{norm}} \cdot (1 + \text{scale}) + \text{shift}.
\]

这样，特征的分布会随着时间步 $t$ 的变化而自适应地调整。


### 2. Gating

除了 Shift & Scale 外，模型还引入了 **门控（Gating）机制**，即通过一个 sigmoid 激活函数将时间嵌入映射到 $[0,1]$ 范围：
\[
g = \sigma(W_{\text{gate}} \, t_{\text{emb}}),
\]

并将其作用在输出上：
\[
h = f(x') \cdot g,
\]

其中 $f(\cdot)$ 表示 Cross Attention 或 MLP 等子模块的输出。

门控机制的作用是 **控制信息流的强弱**：  
- $g \approx 0$ 时，抑制该时间步的特征；  
- $g \approx 1$ 时，完全保留该时间步的特征；  
- $g$ 在 $(0,1)$ 之间时，相当于“软选择”，灵活调节特征强度。