# eval.py

import torch
from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator
from datasets.dataloader import get_dataloader
import open3d as o3d


def visualize_point_cloud(points: torch.Tensor, name="input"):
    """
    用 open3d 可视化点云，支持 CPU 或 GPU tensor。
    参数:
        points: Tensor[n, 3]
    """
    if points.is_cuda:
        points = points.cpu()
    np_points = points.detach().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    print(f"📌 可视化 {name} 点云，共 {np_points.shape[0]} 个点")
    o3d.visualization.draw_geometries([pcd])


def main():
    # 配置参数
    num_points = 2048
    num_tokens = 32
    d_f = 128
    B = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: 读取数据
    dataloader = get_dataloader(
        root='data/ShapeNetCore.v2.PC15k',
        split='test',
        batch_size=1,
        num_points=num_points,
        shuffle=True
    )
    batch = next(iter(dataloader))
    points = batch['points'][0].to(device)  # [2048, 3]，选第一条数据

    # Step 2: 初始化 ShapeTokenizer
    tokenizer = ShapeTokenizer(
        num_tokens=num_tokens,
        d_in=3,
        d_f=d_f,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    ).to(device)

    visualize_point_cloud(points, name="原始点云 (ShapeTokenizer 输入)")

    # Step 3: 提取 shape tokens
    with torch.no_grad():
        shape_tokens = tokenizer(points)  # [k, d_f]

    print("✅ Shape Tokens:", shape_tokens.shape)

    # Step 4: 构造一批 query 点
    x = torch.randn(B, 3).to(device)  # [B, 3]
    t = torch.rand(B).to(device)     # [B]

    # Step 5: 扩展 token 为 batch
    s = shape_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, k, d_f]

    # Step 6: 初始化 VelocityEstimator
    estimator = VelocityEstimator(
        d=d_f,
        num_frequencies=16,
        n_blocks=3
    ).to(device)

    # Step 7: 估计速度
    with torch.no_grad():
        velocity = estimator(x, s, t)  # [B, 3]

    print("✅ Velocity:", velocity.shape)
    print("🎉 eval.py 执行成功！")


if __name__ == '__main__':
    main()
