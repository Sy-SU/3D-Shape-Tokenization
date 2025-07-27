# tools/train.py

"""
联合训练 ShapeTokenizer 与 VelocityEstimator 的主脚本。

功能：
- 从 ShapeNet 加载点云数据
- 使用 ShapeTokenizer 提取 latent tokens
- 使用 VelocityEstimator 拟合噪声点到目标点之间的速度场
- 训练目标包括：Flow Matching Loss + KL 正则项

该脚本为训练流程搭建框架，后续将逐步补充损失实现与验证逻辑。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 添加 tqdm 进度条模块

# 添加 utils 模块所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.dataloader import get_dataloader
from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator


# ========== 单轮训练 ==========
def train_one_epoch(tokenizer: ShapeTokenizer,
                    estimator: VelocityEstimator,
                    dataloader,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    kl_weight: float = 1e-4):
    """
    训练一个 epoch，执行一次完整的数据遍历。

    参数:
        tokenizer: ShapeTokenizer 模型
        estimator: VelocityEstimator 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 当前设备
        kl_weight: KL 正则项的权重
    """
    tokenizer.train()
    estimator.train()

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Step 1: 提取点云并移动到设备 (Tensor[B, N, 3])
        pointclouds = batch['points'].to(device)  # shape: [B, N, 3]
        B, N, _ = pointclouds.shape

        # Step 2: 获取每个样本的 latent shape token (Tensor[B, K, D])
        # 注意：tokenizer 接收单个点云，因此逐个处理
        token_list = []
        for i in range(B):
            p = pointclouds[i]  # shape: [N, 3]
            tokens = tokenizer(p)  # shape: [K, D]
            token_list.append(tokens)

        shape_tokens = torch.stack(token_list, dim=0)  # [B, K, D]

        # Step 3: 构造每个样本的 (x0, x1, t) 并计算中间点 x_t
        # x0: 噪声点 ∈ Uniform([-1, 1]^3), shape: [B, 3]
        x0 = torch.rand(B, 3, device=device) * 2 - 1

        # x1: 从每个 shape 点云中随机采样一个目标点，shape: [B, 3]
        idx = torch.randint(0, N, size=(B,), device=device)  # 每个样本的采样索引
        x1 = pointclouds[torch.arange(B), idx, :]  # 高维索引采样

        # t: 随机时间点 ∈ [0, 1]，shape: [B]
        t = torch.rand(B, device=device)

        # x_t: 线性插值得到中间点，shape: [B, 3]
        t_expand = t.unsqueeze(1)  # [B, 1]
        x_t = (1 - t_expand) * x0 + t_expand * x1
        # Step 4: 使用 velocity estimator 预测 v_theta(x_t; s, t)
        v_pred = estimator(x_t, shape_tokens, t)  # [B, 3]

        # Step 5: 计算 Flow Matching Loss
        target_v = x1 - x0  # 理想速度 [B, 3]
        loss_flow = nn.functional.mse_loss(v_pred, target_v)

        # Step 6: 一致性 KL 散度项 KL(q(s|Y) || q(s|Z))
        # 为此我们需要再次采样一组子点云 Z
        token_list_z = []
        num_subpoints = min(1024, N)
        for i in range(B):
            idx_z = torch.randperm(N)[:num_subpoints]
            p = pointclouds[i][idx_z]  # 对每个 shape 再采样 N 个点
            tokens = tokenizer(p)  # [K, D]
            token_list_z.append(tokens)
        shape_tokens_z = torch.stack(token_list_z, dim=0)  # [B, K, D]

        # KL 近似为两个 token 平均值之差的平方
        mu_y = shape_tokens.mean(dim=1)     # [B, D]
        mu_z = shape_tokens_z.mean(dim=1)   # [B, D]
        sigma = 1e-3
        loss_kl = (1.0 / (2 * sigma ** 2)) * ((mu_y - mu_z) ** 2).sum(dim=1).mean()

        # 总损失
        loss = loss_flow + kl_weight * loss_kl

        # Step 7: 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失用于计算平均
        if 'total_flow_loss' not in locals():
            total_flow_loss = 0.0
            total_kl_loss = 0.0
            num_batches = 0
        total_flow_loss += loss_flow.item()
        total_kl_loss += loss_kl.item()
        num_batches += 1

        # tqdm 展示当前损失
        tqdm.write(f"FlowLoss: {loss_flow.item():.4f} | KL: {loss_kl.item():.4f}")

    # 返回平均损失
    avg_flow_loss = total_flow_loss / num_batches if num_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0.0
    return avg_flow_loss, avg_kl_loss


# ========== 主训练函数 ==========
def main():
    """
    设置模型、数据、优化器，运行完整训练流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--d_f', type=int, default=128)
    parser.add_argument('--num_tokens', type=int, default=32)
    parser.add_argument('--kl_weight', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/ShapeNetCore.v2.PC15k')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    tokenizer = ShapeTokenizer(
        num_tokens=args.num_tokens,
        d_in=3,
        d_f=args.d_f,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    ).to(device)

    estimator = VelocityEstimator(
        d=args.d_f,
        num_frequencies=16,
        n_blocks=3
    ).to(device)

    # 优化器
    optimizer = optim.AdamW(
        list(tokenizer.parameters()) + list(estimator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 加载训练数据
    train_loader = get_dataloader(
        root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_points=args.num_points,
        shuffle=True,
        num_workers=4
    )

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        avg_flow, avg_kl = train_one_epoch(tokenizer, estimator, train_loader, optimizer, device, kl_weight=args.kl_weight)
        print(f"[Epoch {epoch}] Avg Flow Loss: {avg_flow:.6f} | Avg KL Loss: {avg_kl:.6f}")


if __name__ == '__main__':
    main()
