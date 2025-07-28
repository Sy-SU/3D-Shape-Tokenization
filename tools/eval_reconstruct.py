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
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator
from datasets.dataloader import get_dataloader

# ========== 设置全局保存目录时间戳 ==========
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("outs/reconstruct", TIMESTAMP)


# ========== 加载模型结构和权重 ==========
def load_models(device, tokenizer_ckpt='checkpoints/tokenizer.pth', estimator_ckpt='checkpoints/estimator.pth'):
    tokenizer = ShapeTokenizer(
        num_tokens=32, d_in=3, d_f=128, n_heads=8,
        num_frequencies=16, num_blocks=6
    ).to(device)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt))
    tokenizer.eval()

    estimator = VelocityEstimator(
        d=128, num_frequencies=16, n_blocks=3
    ).to(device)
    estimator.load_state_dict(torch.load(estimator_ckpt))
    estimator.eval()

    return tokenizer, estimator


# ========== 重建函数（Heun ODE） ==========
def decode_from_token(estimator, tokens, num_points=2048, steps=100):
    """
    使用 Heun 方法从 shape token 中重建点云。
    参数:
        estimator: VelocityEstimator
        tokens: Tensor[k, d]
        num_points: int, 重建点数
        steps: int, ODE 积分步数
    返回:
        Tensor[num_points, 3]，重建点云
    """
    device = tokens.device
    B, K, D = 1, *tokens.shape  # 添加 batch 维度
    s = tokens.unsqueeze(0)     # [1, K, D]

    x = torch.rand(num_points, 3, device=device) * 2 - 1  # x0 ∈ Uniform([-1,1]^3)
    x = x.to(device)
    h = 1.0 / steps
    t = torch.zeros(num_points, device=device)

    for i in range(steps):
        t_a = t.clone()
        v1 = estimator(x, s.expand(num_points, -1, -1), t_a)

        x_temp = x + h * v1
        t_b = t + h
        v2 = estimator(x_temp, s.expand(num_points, -1, -1), t_b)

        x = x + 0.5 * h * (v1 + v2)
        t = t_b

    return x.detach().cpu()


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
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    dist = ((x - y) ** 2).sum(dim=2)
    cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
    return cd.item()


# ========== 评估函数 ==========
def evaluate(tokenizer, estimator, dataloader, device):
    os.makedirs(SAVE_DIR, exist_ok=True)

    cds = []
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        gt = batch['points'].squeeze(0).to(device)
        tokens = tokenizer(gt)
        pred = decode_from_token(estimator, tokens, num_points=2048, steps=100)

        cd = chamfer_distance(pred, gt.cpu())
        cds.append(cd)

        np.save(os.path.join(SAVE_DIR, f"{i:05d}_recon.npy"), pred.numpy())
        np.save(os.path.join(SAVE_DIR, f"{i:05d}_gt.npy"), gt.cpu().numpy())

    avg_cd = sum(cds) / len(cds)
    print(f"✅ Chamfer Distance over {len(cds)} samples: {avg_cd:.6f}")


# ========== 主入口 ==========
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, estimator = load_models(device)

    dataloader = get_dataloader(
        root='data/ShapeNetCore.v2.PC15k',
        split='test',
        batch_size=1,
        num_points=2048,
        shuffle=False,
        num_workers=2
    )

    evaluate(tokenizer, estimator, dataloader, device)
