# datasets/dataloader.py

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import json
from glob import glob


class ShapeNetDataset(Dataset):
    """
    用于加载 ShapeNetCore.v2.PC15k 点云数据的 PyTorch Dataset

    参数:
        root (str): 根目录，如 'data/ShapeNetCore.v2.PC15k'
        split (str): 数据划分，'train' / 'val' / 'test'
        num_points (int): 每个 shape 随机采样的点数
    返回:
        dict，包含:
            'points': torch.FloatTensor [num_points, 3]
            'label_name': 类别名称（字符串）
    """

    def __init__(self, root, split='train', num_points=2048, taxonomy_path='data/taxonomy.json'):
        self.root = root
        self.split = split
        self.num_points = num_points

        # 加载 synset_id 到类别名的映射
        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
        self.synsetid_to_name = {
            entry["synsetId"]: entry["name"].split(",")[0]
            for entry in taxonomy
        }

        # 遍历所有 .npy 文件
        self.files = []
        for synset_id in self.synsetid_to_name.keys():
            path = os.path.join(root, synset_id, split)
            if os.path.exists(path):
                for npy_file in glob(os.path.join(path, '*.npy')):
                    self.files.append((npy_file, synset_id))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, synset_id = self.files[idx]
        points = np.load(path)  # [N, 3]
        if points.shape[0] >= self.num_points:
            indices = np.random.choice(
                points.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(
                points.shape[0], self.num_points, replace=True)
        sampled_points = points[indices]

        return {
            # [num_points, 3]
            'points': torch.from_numpy(sampled_points).float(),
            'label_name': self.synsetid_to_name[synset_id]
        }


def get_dataloader(root, split='train', batch_size=32, num_points=2048, shuffle=True, num_workers=4):
    """
    构建 DataLoader

    参数:
        root (str): 数据集根路径
        split (str): 'train' / 'val' / 'test'
        batch_size (int): 批大小
        num_points (int): 每个点云采样点数
        shuffle (bool): 是否打乱数据
        num_workers (int): 多线程数
    返回:
        PyTorch DataLoader
    """
    dataset = ShapeNetDataset(root, split, num_points)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    # 用于测试 DataLoader 的主函数
    root_dir = '/root/autodl-fs/chair'
    split = 'train'
    batch_size = 20
    num_points = 1024

    dataloader = get_dataloader(
        root=root_dir, split=split, batch_size=batch_size, num_points=num_points)

    for i, batch in enumerate(dataloader):
        points = batch['points']  # [B, num_points, 3]

        # ===== 中心化到原点 =====
        points -= points.mean(dim=1, keepdim=True)  # 按每个 shape 平移

        # ===== 缩放到 [-0.5, 0.5] =====
        max_abs = points.abs().max(dim=1, keepdim=True)[0]  # [B, 1, 3]
        max_abs_val = max_abs.max(dim=2, keepdim=True)[0]   # [B, 1, 1]
        points = points / (2 * max_abs_val.clamp(min=1e-8))

        # ===== 断言范围 =====
        assert torch.all(points >= -0.5 - 1e-6) and torch.all(points <= 0.5 + 1e-6), \
            f"Point cloud out of range: min={points.min().item():.4f}, max={points.max().item():.4f}"

        print(f"\n=== Batch {i} ===")
        print("points shape:", points.shape)  # 应该是 [B, num_points, 3]
        print("label names:", batch['label_name'])
        print(f"range: [{points.min().item():.4f}, {points.max().item():.4f}]")

        if i >= 2:  # 只打印前3个 batch
            break