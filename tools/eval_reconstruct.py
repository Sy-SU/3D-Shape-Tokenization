# tools/eval_reconstruct.py

"""
用于评估训练好的 ShapeTokenizer 与 VelocityEstimator 模型在 ShapeNet 测试集上的重建性能。

评估指标：
- Chamfer Distance（2048 points）

执行步骤：
1. 加载测试集点云（2048 points）
2. 使用训练好的 tokenizer 生成 shape token
3. 使用 velocity estimator 从噪声解码生成点云（Heun 解 ODE）
4. 与原始点云计算 Chamfer Distance
5. 报告平均指标
6. 将生成的点云保存到 outs/reconstruct/<时间戳>/
"""

import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse

# 添加 utils 模块所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator
from datasets.dataloader import get_dataloader

# ========== 设置全局保存目录时间戳 ==========
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
# SAVE_DIR = os.path.join("outs/reconstruct", TIMESTAMP)


# ========== 加载模型结构和权重 ==========
def load_models(device, num_tokens, d, d_f, tokenizer_ckpt='checkpoints/best_tokenizer.pt', estimator_ckpt='checkpoints/best_estimator.pt'):
    tokenizer = ShapeTokenizer(
        num_tokens=num_tokens,
        d_in=3,
        d_f=d_f,
        d=d,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    ).to(device)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt))
    tokenizer.eval()

    estimator = VelocityEstimator(
        d=d,
        d_f=d_f,
        num_frequencies=16,
        n_blocks=3
    ).to(device)
    estimator.load_state_dict(torch.load(estimator_ckpt))
    estimator.eval()

    return tokenizer, estimator


# ========== 重建函数（Heun ODE） ==========
def decode_from_token(estimator, tokens, num_points=2048, steps=1000):
    """
    使用 Heun 方法从 shape token 中重建点云。
    参数:
        estimator: VelocityEstimator
        tokens: Tensor[B, K, D]
        num_points: int, 重建点数
        steps: int, ODE 积分步数
    返回:
        Tensor[B, num_points, 3]，重建点云
    """
    # print(f"num_points = {num_points}, steps = {steps}")

    device = tokens.device
    B, K, D = tokens.shape  # 添加 batch 维度
    # print(f"B = {B}, K = {K}, D = {D}")

    h = 1.0 / steps  # 时间步长
    
    # 扩展 shape token s 为 [B, num_points, K, D] → [B * num_points, K, D]
    s = tokens.unsqueeze(1).expand(-1, num_points, -1, -1).reshape(B * num_points, K, D)

    # 初始化重建点 x: [B, num_points, 3]，范围 [-1, 1]
    x_raw = torch.rand(B, num_points, 3, device=device) * 2 - 1
    t = torch.zeros(B, num_points, device=device)  # 初始时间 [B, num_points]

    # 展平 x 和 t 为 [B * num_points, 3] 和 [B * num_points]
    x_raw = x_raw.view(B * num_points, 3)
    t = t.view(B * num_points)

    for _ in range(steps):
        v1 = estimator(x_raw, s, t)                      # [B * num_points, 3]
        x_temp = x_raw + h * v1                          # Euler 第一步
        t_b = t + h
        v2 = estimator(x_temp, s, t_b)                   # Heun 第二步

        x_raw = x_raw + 0.5 * h * (v1 + v2)
        t = t_b

    # 恢复形状为 [B, num_points, 3]
    recon_points = x_raw.view(B, num_points, 3)
    return recon_points.detach().cpu()



# ========== Chamfer Distance 实现 ==========
def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算对称 Chamfer 距离
    参数:
        x: Tensor[M, 3]
        y: Tensor[N, 3]
    返回:
        float: chamfer distance
    """
    d = torch.cdist(x, y, p=2)                  # 欧氏距离
    d = d ** 2

    # 对每个 batch 分别取最近点，再在 N/M 上平均，最后再在 batch 上平均
    cd_xy = d.min(dim=2).values.mean(dim=1)     # x->y
    cd_yx = d.min(dim=1).values.mean(dim=1)     # y->x
    cd = (cd_xy + cd_yx).mean()
    return cd.item()


# ========== 评估函数 ==========
def evaluate(tokenizer, estimator, dataloader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # print(tokenizer)
    # print(estimator)
    num_points = dataloader.dataset.num_points
    print(f"Evaluating on {len(dataloader)} batches, each with {num_points} points...")

    cds = []
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # batch['points'] ：[B, N, 3]，点云数据
        # batch['label_name']：类别标签
        # print(batch['points'].shape)
        gt = batch['points'].to(device)
        # print(gt.shape)
        tokens = tokenizer(gt)
        # print(f"tokens.shape = {tokens.shape}") #[B, K, D]

        pred = decode_from_token(estimator, tokens, num_points=num_points, steps=1000)


        cd = chamfer_distance(pred, gt.cpu())
        cds.append(cd)

        np.save(os.path.join(save_dir, f"{i:05d}_recon.npy"), pred.numpy())
        np.save(os.path.join(save_dir, f"{i:05d}_gt.npy"), gt.cpu().numpy())

    avg_cd = sum(cds) / len(cds)
    print(f"✅ Chamfer Distance over {len(cds)} samples: {avg_cd:.6f}")


# ========== 主入口 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 ShapeTokenizer 与 VelocityEstimator 的重建性能")
    parser.add_argument('--data_root', type=str, default='/root/autodl-fs/demo')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default=None, help='保存输出目录，若为空则使用当前时间戳')
    parser.add_argument('--num_tokens', type=int, default=32)
    parser.add_argument('--d_f', type=int, default=512) # shape tokenizer 和 velocity estimator 中的隐藏维度
    parser.add_argument('--d', type=int, default=64) # shape tokenizer 的输出维度, velocity estimator 中的输入维度, 表示 shape token 的维度
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, estimator = load_models(device, num_tokens=args.num_tokens, d_f=args.d_f, d=args.d)

    if args.save_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("outs/reconstruct", timestamp)
    else:
        save_dir = os.path.join("outs/reconstruct", args.save_dir)

    dataloader = get_dataloader(
        root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_points=args.num_points,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    with torch.no_grad():
        evaluate(tokenizer, estimator, dataloader, device, save_dir)

