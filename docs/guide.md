
# 复现指南

## 1. 实验环境

* **硬件环境**

    * GPU：NVIDIA RTX 4090 (24GB) × 1
    * CPU：25 vCPU Intel(R) Xeon(R) Platinum 8481C
    * 内存：90 GB
    * 硬盘：系统盘 30GB，数据盘 50GB SSD

* **软件环境**

    * OS：Ubuntu 22.04
    * Python：3.12
    * PyTorch：2.3.0
    * CUDA：12.1

---

## 2. 克隆仓库

```bash
# HTTPS
git clone https://github.com/Sy-SU/3D-Shape-Tokenization.git

# 或 SSH（需提前配置 SSH Key）
git clone git@github.com:Sy-SU/3D-Shape-Tokenization.git
```

进入项目目录：

```bash
cd 3D-Shape-Tokenization/
```

---

## 3. 安装依赖

建议在虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

如需加速：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
```

---

## 4. 数据准备

将 ShapeNetCore.v2.PC15k 或子集解压至数据盘，例如：

```
/root/autodl-fs/chair/
/root/autodl-fs/demo/
```

目录结构示例：

```
chair/
  ├── train/*.npy
  ├── val/*.npy
  └── test/*.npy
```

---

## 5. 训练

项目提供统一入口脚本 `scripts/task.sh`。使用前先授予执行权限：

```bash
chmod +x ./scripts/task.sh
```

### 示例一：在 Chair 数据集上训练

```bash
./scripts/task.sh --data_root /root/autodl-fs/chair --timestamp chair_run_001 --max_batches -1
```

### 示例二：在 Demo 数据集上训练

```bash
./scripts/task.sh --data_root /root/autodl-fs/demo --timestamp demo_run_001 --max_batches -1
```

参数说明：

* `--data_root`：数据集路径
* `--timestamp`：运行标识，用于区分实验结果
* `--max_batches`：最大训练批次数，`-1` 表示不限制

---

## 6. 评估与可视化

在训练完成后，可以运行提供的脚本进行评估和可视化。

### 评估模型

```bash
chmod +x ./scripts/eval.sh
./scripts/eval.sh --data_root '/root/autodl-fs/chair' --timestamp chair_eval_001
```

### 可视化结果

```bash
chmod +x ./scripts/visualize.sh
./scripts/visualize.sh --data_root '/root/autodl-fs/chair' --timestamp chair_vis_001
```

### 重建过程可视化

你还可以直接生成逐步的重建可视化结果：

```bash
python tools/integration.py --data_root '/root/autodl-fs/chair'
```

> 输出的点云结果会保存到 `outs/` 目录下，包含重建可视化和日志文件。

---
