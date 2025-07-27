# 

## 3D 形状的表示方法

- graph applications: mesh / 3D Gaussian representation
- scientific or physics simulations: distance fields (continuous representation) 
- **mechine learing: voxels, meshes, point clouds, (un-)signed distance fields, Gaussian splats, latent representations**

## 我们希望我们的表示方法 (shape tokens) 满足以下的特性

- 连续的和紧凑的, 使用低维潜空间, 使得 shape tokens 易于被集成
- 对三维形状的假设极少, 只需要从三维表面采样的点云, 即 $p(xyz) 的独立同分布样本. 训练时只需要点云的数据 (现有的神经 3D 表示需要)
- shape tokens 可以提供几何功能 (?)

## shape tokenizer 和 flow matching
**shape S (初始数据预处理后得到的) 经过 shape tokenizer 得到 shape token (类似于 latent space 中的数据)，这个 token 作为条件 (或者说是 prompt) 去引导一个噪声 x 经过 flow mathcing 还原成 shape。我们可以通过对比还原后的shape和初始shape，来联合训练 shape tokenizer 和 flow matcing 网络，它们相当于 encoder 和 decoder**