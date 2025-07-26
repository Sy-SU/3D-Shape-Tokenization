# train.py
"""
本脚本用于训练 Shape Tokenizer 模块，后续可扩展至联合训练 Flow-Matching Velocity Estimator。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.shape_tokenizer import ShapeTokenizer  # ✅ 用户已实现模块
from utils.dataloader import get_pointcloud_datasets  # 假设你准备将 dataset 放这里
from utils.utils import set_seed, RMSNorm  # 可选辅助工具

# 添加 utils 模块所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ========================================
# TokenizerTrainer: 封装训练逻辑
# ========================================


class TokenizerTrainer:
    """
    用于训练 Shape Tokenizer 的类。

    参数:
        model (nn.Module): ShapeTokenizer 实例
        dataloader (DataLoader): 提供训练数据
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 计算设备

    输入点云维度: (B, N, 3)
    输出 token 维度: (B, K, D)
    """
    def __init__(self, model, dataloader, optimizer, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.MSELoss()  # 仅作占位，需按论文目标替换

    def train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, data in enumerate(self.dataloader):
            pointcloud = data.to(self.device)  # shape: (B, N, 3)
            self.optimizer.zero_grad()

            shape_tokens = self.model(pointcloud)  # shape: (B, K, D)
            # TODO: 暂无 reconstruction loss，后续加入 decoder 时实现
            loss = torch.tensor(0.0, device=self.device)  # placeholder

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f}")


# ========================================
# main 函数：训练入口
# ========================================
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_dataset = get_pointcloud_datasets(split='train', num_points=args.num_points)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = ShapeTokenizer(num_tokens=args.num_tokens, token_dim=args.token_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 训练器
    trainer = TokenizerTrainer(model, train_loader, optimizer, device)

    for epoch in range(1, args.epochs + 1):
        trainer.train_one_epoch(epoch)

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "shape_tokenizer.pth"))


# ========================================
# argparse 配置
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--token_dim", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
