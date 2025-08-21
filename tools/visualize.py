# tools/visualize.py
"""
功能：
- 遍历指定目录下的 *_recon.npy 和 *_gt.npy 文件
- 对比每个样本的重建点云和真实点云
- 仅保存动态旋转图（.gif），并在标题中标注 Chamfer Distance

输出路径：outs/vis_compare/<timestamp>/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation
import random


# ========== Chamfer Distance ==========
def chamfer_distance_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """
    使用 NumPy 实现对称 Chamfer 距离
    参数:
        x: [M, 3] numpy array
        y: [N, 3] numpy array
    返回:
        float: chamfer distance
    """
    x = x[:, np.newaxis, :]  # [M, 1, 3]
    y = y[np.newaxis, :, :]  # [1, N, 3]
    dist = np.sum((x - y) ** 2, axis=2)  # [M, N]
    cd = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    return float(cd)


# ========== 动图生成函数 ==========
def save_rotation_gif(points, filename, title):
    """
    保存点云为旋转视角的 gif 图。
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_axis_off()
    ax.set_title(title)

    def rotate(i):
        ax.view_init(elev=20, azim=i)
        return scat,

    anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 4), interval=40)
    anim.save(filename, writer='pillow', dpi=150)
    plt.close(fig)


# ========== 主函数：对比重建和GT ==========
def visualize_recon_vs_gt(npy_dir, out_dir="outs/vis_compare", max_batches=10):
    """
    遍历指定目录下的 *_recon.npy 和 *_gt.npy 文件，对比重建和GT。
    参数:
        npy_dir: 保存 .npy 文件的目录
        out_dir: 保存可视化图像的目录
        max_batches: 最多可视化的 batch 数量；-1 表示全部
    """
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(npy_dir) if f.endswith("_recon.npy"))

    random.seed(20040303)
    random.shuffle(files)

    if max_batches != -1:
        files = files[:max_batches]

    for f in files:
        batch_id = f[:5]
        recon_path = os.path.join(npy_dir, f)
        gt_path = os.path.join(npy_dir, f"{batch_id}_gt.npy")

        if not os.path.exists(gt_path):
            print(f"⚠️ Missing GT for {f}, skipping...")
            continue

        recon = np.load(recon_path)  # [B, N, 3]
        gt = np.load(gt_path)        # [B, N, 3]

        # 如果只有两维，自动加 batch 维
        if recon.ndim == 2 and recon.shape[1] == 3:
            recon = recon[None, ...]   # -> [1, N, 3]
        if gt.ndim == 2 and gt.shape[1] == 3:
            gt = gt[None, ...]         # -> [1, N, 3]

        B = recon.shape[0]
        for i in range(B):
            cd = chamfer_distance_numpy(recon[i], gt[i])

            # 保存 GIF：重建 & GT
            recon_gif_path = os.path.join(out_dir, f"{batch_id}_{i:03d}_recon.gif")
            gt_gif_path    = os.path.join(out_dir, f"{batch_id}_{i:03d}_gt.gif")

            save_rotation_gif(recon[i], recon_gif_path, title=f"Reconstruction\nCD = {cd:.6f}")
            save_rotation_gif(gt[i],    gt_gif_path,    title="Ground Truth")

            print(f"✅ Saved: {recon_gif_path}, {gt_gif_path}")


# ========== 命令行入口 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化 Shape 重建结果与 GT 对比图像")
    parser.add_argument("timestamp", type=str, help="指定重建结果的时间戳子目录")
    parser.add_argument("--max_batches", type=int, default=10, help="最多可视化的 batch 数量，-1 表示全部")
    args = parser.parse_args()

    input_dir = os.path.join("outs/reconstruct", args.timestamp)
    output_dir = os.path.join("outs/vis_compare", args.timestamp)
    visualize_recon_vs_gt(npy_dir=input_dir, out_dir=output_dir, max_batches=args.max_batches)
