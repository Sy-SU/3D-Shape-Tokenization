## ✅ 全面复现这篇论文你需要完成的模块

### 🧱 模块 1：**Shape Tokenizer**（VQ-VAE）

> ✅ 状态：你已经在理解和准备复现
> 🎯 作用：把 voxel/SDF 转成离散的 token grid
> 📥 输入：32³ voxel / SDF
> 📤 输出：8³ token grid（int）+ 还原 voxel
> 🛠️ 需要做：

* encoder（3D CNN）
* codebook（VQ 机制）
* decoder（3D CNN）
* 重建损失 + commitment loss

---

### 🧠 模块 2：**Shape Prior 模型（Transformer）**

> 🎯 作用：用 Transformer 模型建模 token 的先验分布（做生成、补全等任务）
> 📥 输入：flatten 的 token 序列（或带 mask 的序列）
> 📤 输出：完整 token 序列（分类任务）
> 🛠️ 需要做：

* 位置编码（3D → 1D 序列）
* decoder-only Transformer（或 masked Transformer）
* 训练时用 cross-entropy loss 预测 token ID
* 采样 / 推理策略（如 greedy 或 top-k）

---

### 🎨 模块 3：**Shape Decoder / Generator**

> 🎯 作用：把预测的 token 还原成 3D 形状
> 📥 输入：token index grid
> 📤 输出：还原 voxel / SDF → mesh
> 🛠️ 做法：

* 复用 tokenizer 的 decoder
* 或者重新训练一个解码器网络
* 可以接上 neural renderer（可选）

---

### 🔧 模块 4：**数据预处理与加载**

> 🎯 作用：把 ShapeNet 数据处理为可训练格式
> 🛠️ 需要做：

* 下载 ShapeNetCore.v2
* mesh → voxel grid 或 SDF
* 保存为 `.npy` / `.pt` 格式
* 训练集 / 验证集 / 测试集划分

---

### 📊 模块 5：**评估指标与实验脚本**

> 🎯 作用：还原论文中的评估任务：重建、生成、补全、插值
> 🛠️ 需要做：

* Chamfer Distance（用于点云）
* IoU（体素重建）
* FID（生成质量评估）
* 补全任务：mask + 预测
* 插值任务：两个 token 做 linear interpolate

---

### 🧪 模块 6（可选）：**消融实验**

> 🎯 作用：理解哪些组件重要
> 🛠️ 可做：

* 改变 tokenizer 的 token 数量 / codebook size
* 换不同的 transformer 深度
* 换不同位置编码方式
* 对比 raw voxel AE vs token-based 方法

---

## ✅ 总览表（复现路线图）

| 阶段  | 模块          | 关键任务                 |
| --- | ----------- | -------------------- |
| 1️⃣ | Tokenizer   | VQ-VAE 结构 + 训练       |
| 2️⃣ | Transformer | token 建模 + 补全训练      |
| 3️⃣ | Decoder     | 还原 shape             |
| 4️⃣ | 数据预处理       | ShapeNet → voxel/SDF |
| 5️⃣ | 评估          | Chamfer、IoU、FID      |
| 6️⃣ | 实验对比        | 插值、补全、消融分析           |

---
