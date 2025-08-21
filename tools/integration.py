# tools/visualize_integration.py
"""
可视化 Heun ODE 采样的“生长过程”（点从噪声运动成形状）。
- 从测试集取一个样本
- tokenizer -> tokens
- estimator 进行 ODE 解码，并按间隔记录中间点云
- 生成 GIF/MP4，也可把中间快照保存为 .npy

输出目录：outs/vis_ode/<timestamp>/
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.dataloader import get_dataloader
from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator

# -------------------- 工具：加载模型 --------------------
def load_models(device, num_tokens, d, d_f,
                tokenizer_ckpt='checkpoints/best_tokenizer.pt',
                estimator_ckpt='checkpoints/best_estimator.pt'):
    tokenizer = ShapeTokenizer(
        num_tokens=num_tokens, d_in=3, d_f=d_f, d=d,
        n_heads=8, num_frequencies=16, num_blocks=6
    ).to(device)
    tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    # ✅ 
    estimator = VelocityEstimator(
        d=d, d_f=d_f, num_frequencies=16, n_blocks=3
    ).to(device)
    estimator.load_state_dict(torch.load(estimator_ckpt, map_location=device))
    estimator.eval()
    return tokenizer, estimator

# -------------------- 解码（记录轨迹） --------------------
@torch.no_grad()
def decode_with_traj(estimator, tokens, num_points=2048, steps=1000, record_every=10):
    """
    Heun 采样 + 记录中间快照（轨迹）。
    输入:
        tokens: [B, K, d]（建议 B=1 做可视化）
    返回:
        traj: np.ndarray [T, N, 3]，T 为记录帧数
        t_list: list[float]，对应每帧的时间 t
    """
    device = tokens.device
    B, K, D = tokens.shape
    assert B == 1, "建议可视化时传入单样本（B=1），否则内存很快爆炸"

    h = 1.0 / steps
    N = num_points

    # 展开 token 到 [B*N, K, d]
    s = tokens.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, K, D)

    # 初始化 x 与 t
    x = (torch.rand(B, N, 3, device=device) * 2 - 1).view(B * N, 3)
    t = torch.zeros(B * N, device=device)

    frames = []
    t_list = []

    # 首帧（t=0）
    frames.append(x.view(B, N, 3)[0].detach().cpu().numpy())
    t_list.append(0.0)

    for i in range(1, steps + 1):
        v1 = estimator(x, s, t)               # [B*N, 3]
        x_euler = x + h * v1
        t_next = t + h
        v2 = estimator(x_euler, s, t_next)

        x = x + 0.5 * h * (v1 + v2)
        t = t_next

        if i % record_every == 0 or i == steps:
            frames.append(x.view(B, N, 3)[0].detach().cpu().numpy())
            t_list.append(float(i * h))

    traj = np.stack(frames, axis=0)  # [T, N, 3]
    return traj, t_list

# -------------------- 可视化（生成 GIF/MP4） --------------------
def make_growth_gif_with_spin(traj, out_path, title="ODE Integration + Spin",
                              max_points=4000, fps=15,
                              pause_sec=2, spin_deg=360, spin_step=3):
    """
    综合生长 + 放大 + 旋转的 GIF 动画
    - traj: np.ndarray [T, N, 3]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    T, N, _ = traj.shape
    if N > max_points:
        idx = np.random.default_rng(42).choice(N, size=max_points, replace=False)
        traj = traj[:, idx, :]

    final_points = traj[-1]
    max_range = np.max(final_points.max(axis=0) - final_points.min(axis=0))
    target_half_size = max_range / 2.0

    # 计算点云中心
    center = final_points.mean(axis=0)
    # 计算所需缩放边界，使得点云最大边长正好占据画布边长一半
    lim_end = target_half_size / 0.5

    lim_start = np.max(np.abs(traj.reshape(-1, 3))) * 1.05

    pause_frames = int(fps * pause_sec)
    spin_frames = spin_deg // spin_step
    total_frames = T + pause_frames + spin_frames

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], s=1)
    ax.set_axis_off()

    def update(frame_idx):
        nonlocal scat
        if frame_idx < T:
            P = traj[frame_idx]
            lim = lim_start
            azim = 30
        elif frame_idx < T + pause_frames:
            P = final_points
            scale_ratio = (frame_idx - T + 1) / pause_frames
            lim = lim_start + (lim_end - lim_start) * scale_ratio
            azim = 30
        else:
            P = final_points
            lim = lim_end
            azim = 30 + spin_step * (frame_idx - T - pause_frames)

        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(center[0] - lim, center[0] + lim)
        ax.set_ylim(center[1] - lim, center[1] + lim)
        ax.set_zlim(center[2] - lim, center[2] + lim)
        ax.view_init(elev=20, azim=azim)
        ax.set_title(title + f"\nframe {frame_idx}/{total_frames}")
        scat = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
        return scat,

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps)
    anim.save(out_path, writer='pillow', dpi=160)
    plt.close(fig)


# -------------------- 可视化（最终点云旋转 GIF） --------------------
def make_rotation_gif(points, out_path, title="Final Reconstruction (spin)",
                      fps=20, step_deg=3, fig_size=5):
    """
    points: np.ndarray [N, 3] - 最终一帧点云
    保存一个 360° 旋转的 GIF
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 坐标轴范围
    lim = np.max(np.abs(points)) * 1.05 if points.size > 0 else 1.0

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.set_title(title)

    def rotate(az):
        ax.view_init(elev=20, azim=az)
        return scat,

    frames = list(range(0, 360, step_deg))
    anim = animation.FuncAnimation(fig, rotate, frames=frames, interval=1000/fps)
    anim.save(out_path, writer='pillow', dpi=160)
    plt.close(fig)

# -------------------- 主入口 --------------------
def main():
    parser = argparse.ArgumentParser(description="可视化点从噪声运动成形状的 ODE 采样过程")
    parser.add_argument('--data_root', type=str, default='/root/autodl-fs/chair')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1, help='建议=1 以免显存/内存开销过大')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--num_tokens', type=int, default=32)
    parser.add_argument('--d_f', type=int, default=512)
    parser.add_argument('--d', type=int, default=64)

    parser.add_argument('--steps', type=int, default=1000, help='Heun ODE 步数')
    parser.add_argument('--record_every', type=int, default=1, help='每多少步记录一帧')
    parser.add_argument('--max_points', type=int, default=4000, help='GIF 中最多显示的点数（仅可视化用）')

    parser.add_argument('--sample_index', type=int, default=1, help='从测试集中取第几个样本可视化')
    parser.add_argument('--save_traj_npy', action='store_true', help='同时把每帧保存成 .npy')
    parser.add_argument('--out_name', type=str, default=None, help='输出名（默认自动时间戳）')

    parser.add_argument('--tokenizer_ckpt', type=str, default='checkpoints/best_tokenizer.pt')
    parser.add_argument('--estimator_ckpt', type=str, default='checkpoints/best_estimator.pt')

    parser.add_argument('--spin_fps', type=int, default=20, help='最终点云旋转 GIF 的帧率')
    parser.add_argument('--spin_step_deg', type=int, default=3, help='每帧旋转角度（度）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, estimator = load_models(
        device, args.num_tokens, args.d, args.d_f,
        tokenizer_ckpt=args.tokenizer_ckpt,
        estimator_ckpt=args.estimator_ckpt
    )

    # 只取一个样本用于可视化
    loader = get_dataloader(
        root=args.data_root, split=args.split,
        batch_size=1, num_points=args.num_points,
        shuffle=False, num_workers=args.num_workers
    )
    dataset = loader.dataset
    assert args.sample_index < len(dataset), "sample_index 超出数据集大小"

    # 取出指定样本：✅ 同时支持 numpy / torch.Tensor
    sample = dataset[args.sample_index]
    pts = sample['points']
    if isinstance(pts, np.ndarray):
        gt = torch.from_numpy(pts).unsqueeze(0).to(device)  # [1, N, 3]
    elif torch.is_tensor(pts):
        gt = pts.unsqueeze(0).to(device)  # [1, N, 3]
    else:
        raise TypeError(f"Unsupported points type: {type(pts)}")

    # token 化
    with torch.no_grad():
        tokens = tokenizer(gt)  # [1, K, d]

    # 采样并记录轨迹
    traj, t_list = decode_with_traj(
        estimator, tokens,
        num_points=args.num_points,
        steps=args.steps,
        record_every=args.record_every
    )  # [T, N, 3]

    # 输出路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = args.out_name or f"{timestamp}_idx{args.sample_index}_N{args.num_points}_S{args.steps}_R{args.record_every}"
    out_dir = os.path.join("outs", "vis_ode", base)
    os.makedirs(out_dir, exist_ok=True)

    # 保存 GIF
    gif_path = os.path.join(out_dir, "growth.gif")
    make_growth_gif_with_spin(traj, gif_path,
        title="Heun ODE growth + Spin",
        max_points=args.max_points,
        fps=24,
        pause_sec=10,
        spin_deg=360,
        spin_step=3
    )

    print(f"🎬 Saved: {gif_path}")

    # 保存 最终点云的旋转 GIF
    final_points = traj[-1]  # [N, 3]
    final_spin_path = os.path.join(out_dir, "final_rotation.gif")
    make_rotation_gif(final_points, final_spin_path,
                      title="Final Reconstruction (spin)",
                      fps=args.spin_fps, step_deg=args.spin_step_deg)
    print(f"🌀 Saved: {final_spin_path}")

    # 可选：保存每帧 .npy，便于后处理/复现
    if args.save_traj_npy:
        for i, P in enumerate(traj):
            np.save(os.path.join(out_dir, f"step_{i:04d}.npy"), P)
        np.save(os.path.join(out_dir, "t_list.npy"), np.array(t_list, dtype=np.float32))
        print(f"💾 Saved {len(traj)} npy frames to: {out_dir}")

if __name__ == "__main__":
    main()
