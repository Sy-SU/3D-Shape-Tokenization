# tools/visualize_token.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import matplotlib.cm as cm
from matplotlib import colormaps  # 修复 get_cmap 报错
from collections import defaultdict

# 加载模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.shape_tokenizer import ShapeTokenizer
from datasets.dataloader import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1. 加载模型 =====
tokenizer = ShapeTokenizer(
    num_tokens=32, d_in=3, d_f=512, n_heads=8,
    num_frequencies=16, num_blocks=6
).to(device)
tokenizer.load_state_dict(torch.load("checkpoints/best_tokenizer.pt"))
tokenizer.eval()

# ===== 2. 加载数据集 =====
dataloader = get_dataloader(
    root='/root/autodl-fs/demo',
    split='test',
    batch_size=32,
    num_points=2048,
    shuffle=False,
    num_workers=2
)

features = []
labels = []

# ===== 3. 提取 Token 向量 =====
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting tokens"):
        x = batch['points'].to(device)         # [B, N, 3]
        label = batch['label_name']            # list[str]
        tokens = tokenizer(x)                  # [B, K, D]
        pooled = tokens.mean(dim=1).cpu().numpy()  # [B, D]

        features.append(pooled)
        labels += label

features = np.concatenate(features, axis=0)  # [B, D]

# 打印 mean/std 不覆盖 features 本身
print("Token mean (per dim):", features.mean(axis=0))
print("Token std (per dim):", features.std(axis=0))

# ===== 新增：打印单个样本的 token 分布 =====
with torch.no_grad():
    for batch in dataloader:
        x = batch['points'].to(device)
        tokens = tokenizer(x)  # [B, K, D]
        single = tokens[0].cpu().numpy()  # [K, D]

        print("\n[Debug] Single Shape Token mean per dim:", single.mean(axis=0))
        print("[Debug] Single Shape Token std per dim:", single.std(axis=0))

        proj_single = PCA(n_components=2).fit_transform(single)
        plt.figure()
        plt.scatter(proj_single[:, 0], proj_single[:, 1], s=15)
        plt.title("Single Shape Token Distribution (PCA)")
        plt.tight_layout()
        plt.savefig("outs/token_single_debug.png")
        plt.show()
        break

# ===== 4. 只保留前 N 类别用于可视化 =====
label_counts = defaultdict(int)
for lbl in labels:
    label_counts[lbl] += 1

top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
top_labels = {l for l, _ in top_labels}

filtered_idx = [i for i, lbl in enumerate(labels) if lbl in top_labels]
features = features[filtered_idx]
labels = [labels[i] for i in filtered_idx]

# ===== 5. 降维 & 可视化（对象级 pooling） =====
if args.method == "tsne":
    print("Applying t-SNE projection...")
    proj = TSNE(n_components=2, perplexity=30, init='random', random_state=42).fit_transform(features)
else:
    print("Applying PCA projection...")
    proj = PCA(n_components=2).fit_transform(features)

plt.figure(figsize=(8, 6))
colors = colormaps["tab10"](np.linspace(0, 1, len(set(labels))))
label_to_color = {l: colors[i] for i, l in enumerate(sorted(set(labels)))}

for l in sorted(set(labels)):
    idx = [i for i, lbl in enumerate(labels) if lbl == l]
    plt.scatter(proj[idx, 0], proj[idx, 1], label=l, s=10, alpha=0.6, color=label_to_color[l])

plt.title("Shape Token Embedding ({} projection)".format(args.method.upper()))
plt.legend(fontsize=6, loc='best', markerscale=2)
plt.tight_layout()
os.makedirs("outs", exist_ok=True)
plt.savefig("outs/token_vis.png")
plt.show()

# ===== 6. 降维 & 可视化（原始所有 token） =====
raw_tokens = []
raw_labels = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting raw tokens"):
        x = batch['points'].to(device)
        label = batch['label_name']
        tokens = tokenizer(x).cpu().numpy()  # [B, K, D]
        B, K, D = tokens.shape

        raw_tokens.append(tokens.reshape(B * K, D))  # flatten token per object
        raw_labels += [l for l in label for _ in range(K)]

raw_tokens = np.concatenate(raw_tokens, axis=0)  # [B*K, D]
proj = PCA(n_components=2).fit_transform(raw_tokens)

plt.figure(figsize=(8, 6))
colors = colormaps["tab10"](np.linspace(0, 1, len(set(raw_labels))))
label_to_color = {l: colors[i] for i, l in enumerate(sorted(set(raw_labels)))}

for l in sorted(set(raw_labels)):
    idx = [i for i, lbl in enumerate(raw_labels) if lbl == l]
    plt.scatter(proj[idx, 0], proj[idx, 1], label=l, s=2, alpha=0.3, color=label_to_color[l])

plt.title("All Shape Tokens (raw, no pooling)")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("outs/token_vis_all.png")
plt.show()
