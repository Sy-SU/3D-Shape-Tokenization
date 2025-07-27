"""
示例：
from src.dataset import ShapeNetDataset
from torch.utils.data import DataLoader

dataset = ShapeNetDataset(root='ShapeNetCore.v2.PC15k', split='train')
print(len(dataset))
points, label = dataset[0]
print(points.shape, label)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
for pts, lbl in dataloader:
    print(pts.shape)  # [16, N, 3]
    print(lbl)
    break

似乎不需要，PC15k里面的数据就是 npy 的格式
可以注意一下需不需要完成label的映射
tokenizer里面需要dataloader，具体的抽样方式看原论文5.1部分
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        :param root: Root path to ShapeNetCore.v2.PC15k directory
        :param split: 'train', 'val', or 'test'
        :param transform: Optional transform to apply to point cloud
        """
        self.root = root
        self.split = split
        self.transform = transform

        # 加载 synset_id → label_name 映射表
        mapping_path = os.path.join('data', 'label.json')
        with open(mapping_path, 'r') as f:
            self.synset_to_name = json.load(f)

        # 收集所有数据路径和标签
        self.data = []
        for synset_id in sorted(os.listdir(root)):
            synset_dir = os.path.join(root, synset_id, split)
            if not os.path.isdir(synset_dir):
                continue
            label_name = self.synset_to_name.get(synset_id, 'unknown')
            for file in os.listdir(synset_dir):
                if file.endswith('.npy'):
                    file_path = os.path.join(synset_dir, file)
                    self.data.append((file_path, label_name))

        if not self.data:
            raise RuntimeError(
                f"No .npy files found under split '{split}' in {root}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label_name = self.data[idx]
        points = np.load(path)  # shape: (N, 3)
        if self.transform:
            points = self.transform(points)
        return points, label_name
