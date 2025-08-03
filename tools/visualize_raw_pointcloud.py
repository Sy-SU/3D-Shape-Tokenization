# tools/visualize_raw_pointcloud.py

"""
用于可视化原始训练集中不含 batch 维的点云（形如 [N, 3] 的 .npy 文件）

功能：
- 遍历指定目录下前 N 个 .npy 文件
- 对每个点云生成静态图（.png）和动态旋转图（.gif）
- 图像保存到 outs/vis_raw/<子目录名>/
"""

# python tools/visualize_raw_pointcloud.py ~/autodl-fs/demo/02747177/train --max_files 10

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ========== 可视化函数 ==========
def plot_point_cloud(ax, points, title):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)

# ========== 动图函数 ==========
def save_rotation_gif(points, filename, title):
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

# ========== 主函数 ==========
def visualize_raw_pointclouds(npy_dir, out_dir, max_files=10):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])[:max_files]

    for i, fname in enumerate(files):
        path = os.path.join(npy_dir, fname)
        points = np.load(path)  # shape: [N, 3]

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        plot_point_cloud(ax, points, title=f"Raw PointCloud #{i}")

        png_path = os.path.join(out_dir, f"{i:03d}.png")
        gif_path = os.path.join(out_dir, f"{i:03d}.gif")

        plt.savefig(png_path, dpi=300)
        plt.close(fig)

        save_rotation_gif(points, gif_path, title=f"Raw PointCloud #{i}")

        print(f"✅ Saved: {png_path}, {gif_path}")

# ========== 命令行入口 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化原始训练集点云")
    parser.add_argument("npy_dir", type=str, help="包含 [N,3] .npy 文件的目录")
    parser.add_argument("--max_files", type=int, default=10, help="最多可视化文件数")
    args = parser.parse_args()

    subdir_name = os.path.basename(os.path.normpath(args.npy_dir))
    out_dir = os.path.join("outs", "vis_raw", subdir_name)

    visualize_raw_pointclouds(args.npy_dir, out_dir, args.max_files)
