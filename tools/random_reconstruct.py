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
from torch.utils.data import DataLoader, SubsetRandomSampler
from collections import defaultdict
import math
import random


# 添加 utils 模块所在路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator
from datasets.dataloader import get_dataloader

# ========== 设置全局保存目录时间戳 ==========
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


# ========== 加载模型结构和权重 ==========
def load_models(device, num_tokens, d, d_f,
                tokenizer_ckpt='checkpoints/best_tokenizer.pt',
                estimator_ckpt='checkpoints/best_estimator.pt'):
    """
    说明：
        为了与训练时的结构保持一致，这里显式接收 num_tokens/d/d_f 三个结构超参。
        （你之前的脚本版本就是这么做的，这样能避免维度不一致导致的 load_state_dict 失败）
    """
    tokenizer = ShapeTokenizer(
        num_tokens=num_tokens,
        d_in=3,
        d_f=d_f,
        d=d,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    ).to(device)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    estimator = VelocityEstimator(
        d=d,
        d_f=d_f,
        num_frequencies=16,
        n_blocks=3
    ).to(device)
    estimator.load_state_dict(torch.load(estimator_ckpt, map_location=device))
    estimator.eval()

    return tokenizer, estimator


# ========== 重建函数（Heun ODE） ==========
def decode_from_token(estimator, tokens, num_points=2048, steps=1000):
    """
    使用 Heun 方法从 shape token 中重建点云。
    参数:
        estimator: VelocityEstimator
        tokens: Tensor[B, K, D]（此处 D = d，注意估计器内部通道是 d_f，外部 token 维度是 d）
        num_points: int, 重建点数
        steps: int, ODE 积分步数
    返回:
        Tensor[B, num_points, 3]，重建点云
    """
    device = tokens.device
    B, K, D = tokens.shape

    h = 1.0 / steps  # 时间步长

    # 扩展 shape token s 为 [B, num_points, K, D] → [B * num_points, K, D]
    s = tokens.unsqueeze(1).expand(-1, num_points, -1, -1).reshape(B * num_points, K, D)

    # 初始化重建点 x: [B, num_points, 3]，范围 [-1, 1]
    x_raw = (torch.rand(B, num_points, 3, device=device) * 2 - 1).view(B * num_points, 3)
    t = torch.zeros(B * num_points, device=device)  # 初始时间 t=0

    for _ in range(steps):
        v1 = estimator(x_raw, s, t)              # [B*num_points, 3]
        x_temp = x_raw + h * v1                  # Euler 第一步
        t_b = t + h
        v2 = estimator(x_temp, s, t_b)           # Heun 第二步

        x_raw = x_raw + 0.5 * h * (v1 + v2)
        t = t_b

    recon_points = x_raw.view(B, num_points, 3)
    return recon_points.detach().cpu()


# ========== Chamfer Distance（支持 batch） ==========
def chamfer_distance_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算对称 Chamfer 距离（支持 batch）
    参数:
        x: Tensor[B, M, 3]
        y: Tensor[B, N, 3]
    返回:
        float: 平均 chamfer distance（先对每个 batch 求，再对 batch 求均值）
    备注：
        采用与 PointFlow/LION 相同的“平方欧氏距离 + 双向最近点平均”的定义（参见你的参考表述）。 
    """
    # [B, M, N]
    d = torch.cdist(x, y, p=2) ** 2
    cd_xy = d.min(dim=2).values.mean(dim=1)   # x->y, 按 M 平均
    cd_yx = d.min(dim=1).values.mean(dim=1)   # y->x, 按 N 平均
    cd = (cd_xy + cd_yx).mean()
    return cd.item()


# ========== 评估函数 ==========
def evaluate(tokenizer, estimator, dataloader, device, save_dir, ode_steps=1000):
    os.makedirs(save_dir, exist_ok=True)

    num_points = dataloader.dataset.num_points
    print(f"Evaluating on {len(dataloader)} batches, each with {num_points} points...")

    cds = []
    running_idx = 0  # 用于给保存文件编号（确保不被 batch 维度混淆）

    for batch in tqdm(dataloader, desc="Evaluating"):
        # batch['points'] ：[B, N, 3]，点云数据
        gt = batch['points'].to(device)
        tokens = tokenizer(gt)  # [B, K, d]

        pred = decode_from_token(estimator, tokens, num_points=num_points, steps=ode_steps)

        # 计算 Chamfer Distance（按 batch）
        cd = chamfer_distance_batch(pred, gt.cpu())
        cds.append(cd)

        # 逐样本保存
        B = pred.shape[0]
        for b in range(B):
            np.save(os.path.join(save_dir, f"{running_idx:05d}_recon.npy"), pred[b].numpy())
            np.save(os.path.join(save_dir, f"{running_idx:05d}_gt.npy"), gt[b].detach().cpu().numpy())
            running_idx += 1

    avg_cd = sum(cds) / len(cds)
    print(f"✅ Chamfer Distance over {running_idx} samples: {avg_cd:.6f}")


# ========== 主入口 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 ShapeTokenizer 与 VelocityEstimator 的重建性能")
    parser.add_argument('--data_root', type=str, default='/root/autodl-fs/demo')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default=None, help='保存输出目录，若为空则使用当前时间戳')

    # ===== 新增：在测试集中随机抽取若干样本 =====
    parser.add_argument('--num_samples', type=int, default=64, help='在测试集内随机抽取的样本数；<=0 表示使用全部测试集')
    parser.add_argument('--seed', type=int, default=42, help='随机抽样的随机种子')

    # ===== 结构/权重 & ODE 步数参数（保持与训练侧一致）=====
    parser.add_argument('--num_tokens', type=int, default=32)
    parser.add_argument('--d_f', type=int, default=512)  # hidden 维度（Tokenizer/Estimator 内部通道）
    parser.add_argument('--d', type=int, default=64)     # shape token 维度（Tokenizer 输出 & Estimator 输入的 token 维）
    parser.add_argument('--ode_steps', type=int, default=1000, help='Heun ODE 采样步数')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, estimator = load_models(
        device,
        num_tokens=args.num_tokens,
        d=args.d,
        d_f=args.d_f
    )

    # 保存目录
    if args.save_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("outs/reconstruct", timestamp)
    else:
        save_dir = os.path.join("outs/reconstruct", args.save_dir)

    # 先创建一个用于拿到 dataset 的 DataLoader
    base_loader = get_dataloader(
        root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_points=args.num_points,
        shuffle=False,
        num_workers=args.num_workers
    )

    dataset = base_loader.dataset
    total = len(dataset)

    # ===== 均匀按类采样 =====
    target_total = args.num_samples if (args.num_samples is not None and args.num_samples > 0) else 0
    if target_total and target_total < total:
        # 1) 按 label_name 分桶
        label_to_indices = defaultdict(list)
        for idx in range(total):
            item = dataset[idx]
            # 兼容：item 可能是 dict
            label = item['label_name'] if isinstance(item, dict) else getattr(item, 'label_name', None)
            if label is None:
                raise KeyError(f"Sample {idx} has no 'label_name' field.")
            label_to_indices[label].append(idx)

        num_classes = len(label_to_indices)
        rng = random.Random(args.seed)

        # 2) 每类分配配额（上取整），优先均匀覆盖各类
        per_class_quota = max(1, math.ceil(target_total / num_classes))

        selected = []
        leftovers = []

        for label, idxs in label_to_indices.items():
            rng.shuffle(idxs)  # 以 seed 打乱
            take = min(per_class_quota, len(idxs))
            selected.extend(idxs[:take])
            # 记录该类剩余备用
            leftovers.extend(idxs[take:])

        # 3) 如果超过目标总数，截断；如果不足则从剩余里补齐
        if len(selected) > target_total:
            selected = selected[:target_total]
        elif len(selected) < target_total:
            rng.shuffle(leftovers)
            need = target_total - len(selected)
            selected.extend(leftovers[:need])

        sampler = SubsetRandomSampler(selected)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=False
        )

        # 打印一个简短分布统计
        dist = defaultdict(int)
        for i in selected:
            lab = dataset[i]['label_name'] if isinstance(dataset[i], dict) else getattr(dataset[i], 'label_name', 'UNK')
            dist[lab] += 1
        dist_str = ", ".join([f"{k}:{v}" for k, v in list(dist.items())[:10]])
        print(f"🔎 Class-balanced subset: {len(selected)}/{total} samples "
              f"(classes={num_classes}, quota≈{per_class_quota}).")
        print(f"   Sampled per-class (first 10): {dist_str}")
    else:
        dataloader = base_loader
        print(f"🔎 Using the full test set: {total} samples.")


    with torch.no_grad():
        evaluate(tokenizer, estimator, dataloader, device, save_dir, ode_steps=args.ode_steps)
