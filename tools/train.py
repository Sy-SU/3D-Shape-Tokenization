# tools/train.py

"""
联合训练 ShapeTokenizer 与 VelocityEstimator 的主脚本。

功能：
- 从 ShapeNet 加载点云数据
- 使用 ShapeTokenizer 提取 latent tokens
- 使用 VelocityEstimator 拟合噪声点到目标点之间的速度场
- 训练目标包括：Flow Matching Loss + KL 正则项
- 支持训练集与验证集评估
- 添加早停与最佳模型保存逻辑
- 将训练日志写入 logs/ 目录

"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import json
from datetime import datetime

# 添加 utils 模块所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.dataloader import get_dataloader
from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator

# ========== Transformer LR Scheduler ==========
def get_transformer_lr_schedule(warmup_steps, d_model):
    def lr_lambda(step):
        step += 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda


# ========== 单轮训练 ==========
def run_one_epoch(tokenizer: ShapeTokenizer,
                  estimator: VelocityEstimator,
                  dataloader,
                  optimizer: optim.Optimizer = None,
                  scheduler: LambdaLR = None,
                  device: torch.device = None,
                  kl_weight: float = 1e-4,
                  mode: str = "train"):
    is_train = (mode == "train")
    tokenizer.train(is_train)
    estimator.train(is_train)

    total_flow_loss, total_kl_loss, num_batches = 0.0, 0.0, 0

    # 遍历每个 batch（点云集合）进行训练或验证
    for batch in tqdm(dataloader, desc=f"{mode.capitalize()}", leave=False):
        # Step 1: 提取点云并移动到设备 (Tensor[B, N, 3])
        pointclouds = batch['points'].to(device)
        B, N, _ = pointclouds.shape

        # Step 2: 获取每个样本的 latent shape token (Tensor[B, K, D])
        shape_tokens = tokenizer(pointclouds)

        # Step 3: 构造每个样本的 (x0, x1, t) 并计算中间点 x_t
        # x0: 噪声点 ∈ Uniform([-1, 1]^3), shape: [B, 3]
        x0 = torch.rand(B, 3, device=device) * 2 - 1
        idx = torch.randint(0, N, size=(B,), device=device)
        x1 = pointclouds[torch.arange(B), idx, :]
        t = torch.rand(B, device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1

        # Step 4: 使用 velocity estimator 预测 v_theta(x_t; s, t)
        v_pred = estimator(x_t, shape_tokens, t)
        # Step 5: 计算 Flow Matching Loss
        target_v = x1 - x0
        loss_flow = nn.functional.mse_loss(v_pred, target_v)

        # 为此我们需要再次采样一组子点云 Z
        # Step 6: 一致性 KL 散度项 KL(q(s|Y) || q(s|Z))
        num_subpoints = min(1024, N)
        idx_z = torch.randint(0, N, (B, num_subpoints), device=device)  # [B, 1024]
        subpoints_z = torch.gather(
            pointclouds, 1,
            idx_z.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, 1024, 3]
        shape_tokens_z = tokenizer(subpoints_z)  # [B, K, D]

        mu_y = shape_tokens.mean(dim=1)         # [B, D]
        mu_z = shape_tokens_z.mean(dim=1)       # [B, D]
        sigma = 1e-3
        loss_kl = (1.0 / (2 * sigma ** 2)) * ((mu_y - mu_z) ** 2).sum(dim=1).mean()

        # 总损失 = Flow Matching + KL
        loss = loss_flow + kl_weight * loss_kl

        # Step 7: 反向传播与优化，仅在训练模式下执行
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_flow_loss += loss_flow.item()
        total_kl_loss += loss_kl.item()
        num_batches += 1

    # 返回平均损失指标
    avg_flow = total_flow_loss / num_batches
    avg_kl = total_kl_loss / num_batches
    return avg_flow, avg_kl


# ========== 主训练函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--d_f', type=int, default=512)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--num_tokens', type=int, default=32)
    parser.add_argument('--kl_weight', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='data/ShapeNetCore.v2.PC15k')
    parser.add_argument('--patience', type=int, default=9999)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # tokenizer: n = 2048, k = 32, d_f = 512, d = 64, n_head = 8
    # 5490 万训练参数

    tokenizer = ShapeTokenizer(
        num_tokens=args.num_tokens,
        d_in=3,
        d_f=args.d_f,
        d=args.d,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    ).to(device)

    estimator = VelocityEstimator(
        d=args.d_f, # TODO d_f 还是 d? 
        num_frequencies=16,
        n_blocks=3
    ).to(device)

    # ===== 打印 Tokenizer 结构与参数 =====
    print("\n📦 ShapeTokenizer Architecture:\n")
    print(tokenizer)
    tokenizer_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"📊 ShapeTokenizer trainable parameters: {tokenizer_params / 1e6:.2f} M")

    # ===== 打印 VelocityEstimator 结构与参数 =====
    print("\n📦 VelocityEstimator Architecture:\n")
    print(estimator)
    estimator_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f"📊 VelocityEstimator trainable parameters: {estimator_params / 1e6:.2f} M")

    total_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad) \
             + sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f"📊 Total ShapeNet LFM trainable parameters: {total_params / 1e6:.2f} M")


    optimizer = optim.AdamW(
        list(tokenizer.parameters()) + list(estimator.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),     
        weight_decay=args.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=get_transformer_lr_schedule(args.warmup_steps, args.d_f))

    train_loader = get_dataloader(
        root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_points=args.num_points,
        shuffle=True,
        num_workers=4
    )

    val_loader = get_dataloader(
        root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_points=args.num_points,
        shuffle=False,
        num_workers=4
    )

    os.makedirs("checkpoints", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    log = {
        "config": vars(args),
        "train": []
    }
    best_val = float("inf")
    patience_counter = 0
    print(f"Training ShapeTokenizer and VelocityEstimator with config: {args}")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_flow, train_kl = run_one_epoch(tokenizer, estimator, train_loader, optimizer, scheduler, device, args.kl_weight, mode="train")
        val_flow, val_kl = run_one_epoch(tokenizer, estimator, val_loader, optimizer=None, scheduler=None, device=device, kl_weight=args.kl_weight, mode="val")

        print(f"[Epoch {epoch}] Train Flow: {train_flow:.6f} | Train KL: {train_kl:.6f} || Val Flow: {val_flow:.6f} | Val KL: {val_kl:.6f}")

        log["train"].append({
            "epoch": epoch,
            "train_flow": train_flow,
            "train_kl": train_kl,
            "val_flow": val_flow,
            "val_kl": val_kl
        })

        if val_flow < best_val:
            best_val = val_flow
            patience_counter = 0
            torch.save(tokenizer.state_dict(), "checkpoints/best_tokenizer.pt")
            torch.save(estimator.state_dict(), "checkpoints/best_estimator.pt")
            print("✅ Best model updated.")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("⏹️ Early stopping triggered.")
                break

        log_path = os.path.join(log_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
