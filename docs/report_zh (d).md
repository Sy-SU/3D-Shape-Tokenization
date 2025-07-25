# 

## 3D 形状的表示方法

- graph applications: mesh / 3D Gaussian representation
- scientific or physics simulations: distance fields (continuous representation) 
- **mechine learing: voxels, meshes, point clouds, (un-)signed distance fields, Gaussian splats, latent representations**

## 我们希望我们的表示方法 (shape tokens) 满足以下的特性

- 连续的和紧凑的, 使用低维潜空间, 使得 shape tokens 易于被集成
- 对三维形状的假设极少, 只需要从三维表面采样的点云, 即 $p(xyz) 的独立同分布样本. 训练时只需要点云的数据 (现有的神经 3D 表示需要)
- shape tokens 可以提供几何功能 (?)