# models/shape2clip_mlp.py
"""
作用：
- 定义一个将 ShapeTokenizer 输出的 tokens（[B, K, D_tok]）拼接成向量并映射到 CLIP 文本空间的 MLP 适配器。
- 提供可直接运行的训练入口（只训练 MLP；冻结 ShapeTokenizer 和 CLIP 文本编码器），用于对齐到 zero-shot 分类可用的空间。

训练目标：
- 给定每个样本的类别名称（由 dataloader 提供 label_name），构造文本 prompt，
- 计算 CLIP 文本编码器的文本嵌入 z_text ∈ R^{D_clip}（冻结），
- 将 tokens 拼接成 [B, K*D_tok]，经 MLP 映射为 z_shape ∈ R^{D_clip}（训练），
- 使用 InfoNCE（CLIP 风格对比损失）拉近匹配的 (z_shape, z_text_label) 距离，推远其他类别文本。
"""

import os
import sys
import json
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使得可以 import 项目内模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.shape_tokenizer import ShapeTokenizer
from datasets.dataloader import get_dataloader


# ========== 工具：加载 CLIP 文本编码器 ==========
def load_clip_text_encoder(model_name: str, pretrained: str, device: torch.device):
    """
    加载 CLIP 文本编码器（优先 open_clip，其次 openai/clip），并返回：
        encode_text_fn(text_list) -> Tensor[L, D_clip]（已 L2 归一）
        D_clip: 文本嵌入维度
        clip_id: 字符串标识
    """
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)

        @torch.no_grad()
        def encode_text_fn(text_list: List[str]) -> torch.Tensor:
            toks = tokenizer(text_list).to(device)
            feats = model.encode_text(toks).float()
            feats = F.normalize(feats, dim=-1)
            return feats

        dim = encode_text_fn(["test"]).shape[-1]
        return encode_text_fn, dim, f"open_clip::{model_name}@{pretrained}"
    except Exception:
        import clip  # openai/clip
        model, _ = clip.load(model_name, device=device, jit=False)

        @torch.no_grad()
        def encode_text_fn(text_list: List[str]) -> torch.Tensor:
            toks = clip.tokenize(text_list).to(device)
            feats = model.encode_text(toks).float()
            feats = F.normalize(feats, dim=-1)
            return feats

        dim = encode_text_fn(["test"]).shape[-1]
        return encode_text_fn, dim, f"openai/clip::{model_name}"


# ========== Prompt 生成 ==========
def build_prompts(labels: List[str], template: str) -> List[str]:
    """
    作用：将类别名转换为 zero-shot 文本 prompt。
    例如：template="a 3D model of a {}"
    """
    out = []
    for name in labels:
        out.append(template.format(name.strip()))
    return out


# ========== 标签读取 ==========
def load_label_list(path: str) -> List[str]:
    """
    支持：
    1) ["airplane","chair",...]
    2) [{"synsetId":"0269...","name":"airplane,aeroplane"}, ...]  取 name 的第一个别名
    3) {id: name, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = []
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            labels = data
        elif all(isinstance(x, dict) for x in data) and "name" in data[0]:
            for item in data:
                raw = item.get("name", "")
                lab = raw.split(",")[0].strip()
                if lab:
                    labels.append(lab)
    elif isinstance(data, dict):
        for _, v in data.items():
            lab = str(v).split(",")[0].strip()
            if lab:
                labels.append(lab)
    else:
        raise ValueError(f"Unknown label.json structure: {type(data)}")
    if not labels:
        raise ValueError("Empty labels parsed from label.json")
    return labels


# ========== MLP 适配器定义 ==========
class Shape2CLIPMLP(nn.Module):
    """
    作用：
        将 ShapeTokenizer 输出的 tokens [B, K, D_tok] 拼接为 [B, K*D_tok]，
        通过 4 层隐藏层（各 4096 维）的 MLP，映射到 CLIP 文本空间维度 D_clip。

    参数：
        d_in: int，输入向量维度（K*D_tok）
        d_out: int，输出向量维度（等于 CLIP 文本嵌入维度）
        hidden: int，隐藏层宽度（默认 4096）
        dropout: float，Dropout 概率

    前向：
        tokens: Tensor[B, K, D_tok]
        返回：Tensor[B, D_out]，已 L2 归一化
    """
    def __init__(self, d_in: int, d_out: int, hidden: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, K, D_tok], got {tokens.shape}")
        B, K, Dtok = tokens.shape
        if K * Dtok != self.d_in:
            raise ValueError(f"Flatten dim mismatch: K*D_tok={K*Dtok} != d_in={self.d_in}")
        x = tokens.reshape(B, self.d_in)      # 拼接
        z = self.net(x)                       # [B, d_out]
        z = F.normalize(z, dim=-1)            # L2 归一
        return z


# ========== CLIP 风格 InfoNCE（仅 shape->text 单向） ==========
class InfoNCECategorical(nn.Module):
    """
    作用：
        将 zero-shot 训练视为“按所有类别文本原型做 softmax 分类”。
        给定形状嵌入 z_shape[B, D] 与 全类别文本嵌入 z_text[L, D]（L 为类别数），
        取每个样本的真实类别 idx，计算交叉熵：
            logits = (z_shape @ z_text^T) / tau
            loss = CrossEntropy(logits, target=class_idx)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature))

    def forward(self, z_shape: torch.Tensor, z_text_all: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        if z_shape.ndim != 2 or z_text_all.ndim != 2:
            raise ValueError("z_shape and z_text_all must be 2D tensors")
        if class_idx.ndim != 1 or class_idx.shape[0] != z_shape.shape[0]:
            raise ValueError("class_idx must be [B] aligning with z_shape batch")
        # 归一化假设外部已做；此处再保险
        z_shape = F.normalize(z_shape, dim=-1)
        z_text_all = F.normalize(z_text_all, dim=-1)
        logits = self.logit_scale.exp() * (z_shape @ z_text_all.t())  # [B, L]
        loss = F.cross_entropy(logits, class_idx, reduction="mean")
        return loss


# ========== 训练入口 ==========
def main():
    parser = argparse.ArgumentParser(description="训练 Shape2CLIP MLP（冻结 ShapeTokenizer 与 CLIP 文本编码器）")
    # 数据与标签
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/ShapeNetCore.v2.PC15k")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_json", type=str, default="configs/label.json")
    parser.add_argument("--prompt_tpl", type=str, default="a 3D model of a {}")

    # Tokenizer 配置（需与训练时一致）
    parser.add_argument("--tokenizer_ckpt", type=str, default="checkpoints/best_tokenizer.pt")
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--d_f", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_frequencies", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=6)

    # CLIP 配置
    parser.add_argument("--clip_model", type=str, default="ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, default="openai")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)

    # 输出
    parser.add_argument("--save_path", type=str, default="checkpoints/shape2clip_mlp.pt")
    parser.add_argument("--log_dir", type=str, default=None)  # 若为空则 logs/<timestamp> 自动创建
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # ========== 标签与文本原型 ==========
    labels = load_label_list(args.label_json)           # L
    prompts = build_prompts(labels, args.prompt_tpl)

    encode_text, d_clip, clip_id = load_clip_text_encoder(args.clip_model, args.clip_pretrained, device)
    print(f"[CLIP] loaded: {clip_id}, dim={d_clip}, #labels={len(labels)}")

    with torch.no_grad():
        text_emb_all = encode_text(prompts).to(device)  # [L, d_clip]

    # 构造 label->idx 映射，便于从 batch['label_name'] 找到类别下标
    label2idx = {name: i for i, name in enumerate(labels)}

    # ========== DataLoader ==========
    loader = get_dataloader(
        root=args.data_root, split=args.split, batch_size=args.batch_size,
        num_points=args.num_points, shuffle=True, num_workers=args.num_workers
    )

    # ========== 冻结 ShapeTokenizer ==========
    tokenizer = ShapeTokenizer(
        num_tokens=args.num_tokens, d_in=3, d_f=args.d_f, d=args.d,
        n_heads=args.n_heads, num_frequencies=args.num_frequencies, num_blocks=args.num_blocks
    ).to(device)
    if not os.path.exists(args.tokenizer_ckpt):
        raise FileNotFoundError(f"Tokenizer ckpt not found: {args.tokenizer_ckpt}")
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # 确定 d_in = K * D_tok
    with torch.no_grad():
        dummy = torch.zeros(1, args.num_points, 3, device=device)
        dummy_tok = tokenizer(dummy)  # [1, K, D_tok]
        if dummy_tok.ndim != 3:
            raise RuntimeError(f"Tokenizer forward should return [B,K,D], got {dummy_tok.shape}")
        K, D_tok = dummy_tok.shape[1], dummy_tok.shape[2]
        d_in = K * D_tok

    # ========== 定义 MLP（仅此可训练） ==========
    mlp = Shape2CLIPMLP(d_in=d_in, d_out=d_clip, hidden=4096, dropout=args.dropout).to(device)

    # ========== 优化器与损失 ==========
    optim = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = InfoNCECategorical(temperature=args.temperature).to(device)

    # 日志目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join("logs", timestamp)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_loss = float("inf")
    patience = 0

    print(f"[Train] Start training Shape2CLIP MLP. Save to: {args.save_path}")
    for epoch in range(1, args.epochs + 1):
        mlp.train()
        running = 0.0
        n_samples = 0

        for batch in loader:
            points = batch["points"].to(device)            # [B, N, 3]
            names = batch.get("label_name", None)
            if names is None:
                raise RuntimeError("Your dataloader must provide 'label_name' for each sample.")

            # 将标签名映射到类别下标（若不存在则跳过样本）
            idxs = []
            for name in names:
                idxs.append(label2idx.get(name, -1))
            idxs = torch.tensor(idxs, device=device, dtype=torch.long)
            valid_mask = idxs >= 0
            if valid_mask.sum() == 0:
                continue

            with torch.no_grad():
                tokens = tokenizer(points)                  # [B, K, D_tok]

            z_shape = mlp(tokens[valid_mask])               # [Bv, d_clip], L2
            class_idx = idxs[valid_mask]                    # [Bv]
            loss = criterion(z_shape, text_emb_all, class_idx)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item() * valid_mask.sum().item()
            n_samples += valid_mask.sum().item()

        epoch_loss = running / max(1, n_samples)
        print(f"[Epoch {epoch:03d}] loss={epoch_loss:.6f}")

        # 早停 & 保存
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience = 0
            torch.save(mlp.state_dict(), args.save_path)
            print(f"  ✅ Saved best MLP -> {args.save_path}")
        else:
            patience += 1
            print(f"  ⏳ patience {patience}/{args.patience}")
            if patience >= args.patience:
                print("  ⏹️ Early stopping.")
                break

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()
