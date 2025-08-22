# tools/zero_shot_cls.py
"""
作用：
- 使用冻结的 ShapeTokenizer 与 训练好的 Shape2CLIP MLP，
- 与 CLIP 文本编码器的类别原型做相似度，计算 zero-shot Top-1 / Top-5。

不修改任何现有模块。
"""

import os
import sys
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F

# 使得可以 import 项目内模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.shape_tokenizer import ShapeTokenizer
from models.shape2clip import Shape2CLIPMLP, load_clip_text_encoder, load_label_list, build_prompts
from datasets.dataloader import get_dataloader


def topk_accuracy(sim: torch.Tensor, gt_labels: List[str], label_list: List[str], ks: Tuple[int, ...] = (1, 5)) -> dict:
    preds = sim.topk(k=max(ks), dim=-1).indices  # [B, Kmax]
    results = {}
    if gt_labels is None or any(x is None for x in gt_labels):
        for k in ks:
            results[f"top{k}"] = float("nan")
        return results

    label2idx = {name: i for i, name in enumerate(label_list)}
    gt_idx = torch.tensor([label2idx.get(name, -1) for name in gt_labels], device=sim.device)

    for k in ks:
        topk = preds[:, :k]
        correct = (topk == gt_idx.unsqueeze(1)).any(dim=1) & (gt_idx >= 0)
        results[f"top{k}"] = correct.float().mean().item()
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot 分类（ShapeTokenizer + Shape2CLIP MLP + CLIP 文本）")
    # 数据
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/ShapeNetCore.v2.PC15k")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)

    # 标签 & prompt
    parser.add_argument("--label_json", type=str, default="configs/label.json")
    parser.add_argument("--prompt_tpl", type=str, default="a 3D model of a {}")

    # Tokenizer
    parser.add_argument("--tokenizer_ckpt", type=str, default="checkpoints/best_tokenizer.pt")
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--d_f", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_frequencies", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=6)

    # CLIP
    parser.add_argument("--clip_model", type=str, default="ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, default="openai")

    # MLP
    parser.add_argument("--mlp_ckpt", type=str, default="checkpoints/shape2clip_mlp.pt")
    parser.add_argument("--save_pred", type=str, default=None)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 文本原型
    labels = load_label_list(args.label_json)
    prompts = build_prompts(labels, args.prompt_tpl)
    encode_text, d_clip, clip_id = load_clip_text_encoder(args.clip_model, args.clip_pretrained, device)
    print(f"[CLIP] loaded: {clip_id}, dim={d_clip}, #labels={len(labels)}")
    with torch.no_grad():
        text_emb = encode_text(prompts).to(device)  # [L, d_clip], L2

    # 2) 数据
    loader = get_dataloader(
        root=args.data_root, split=args.split,
        batch_size=args.batch_size, num_points=args.num_points,
        shuffle=False, num_workers=args.num_workers
    )

    # 3) Tokenizer（冻结）
    tok = ShapeTokenizer(
        num_tokens=args.num_tokens, d_in=3, d_f=args.d_f, d=args.d,
        n_heads=args.n_heads, num_frequencies=args.num_frequencies, num_blocks=args.num_blocks
    ).to(device)
    if not os.path.exists(args.tokenizer_ckpt):
        raise FileNotFoundError(f"Tokenizer ckpt not found: {args.tokenizer_ckpt}")
    tok.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tok.eval()
    for p in tok.parameters():
        p.requires_grad = False

    # 4) MLP
    with torch.no_grad():
        dummy = torch.zeros(1, args.num_points, 3, device=device)
        dtok = tok(dummy)  # [1, K, D_tok]
        if dtok.ndim != 3:
            raise RuntimeError(f"Tokenizer forward should return [B,K,D], got {dtok.shape}")
        K, D_tok = dtok.shape[1], dtok.shape[2]
        d_in = K * D_tok

    mlp = Shape2CLIPMLP(d_in=d_in, d_out=d_clip, hidden=4096, dropout=0.0).to(device)
    if not os.path.exists(args.mlp_ckpt):
        raise FileNotFoundError(f"MLP ckpt not found: {args.mlp_ckpt}")
    mlp.load_state_dict(torch.load(args.mlp_ckpt, map_location=device))
    mlp.eval()

    # 5) 评估
    all_top1, all_top5, n = 0.0, 0.0, 0
    pred_records = []

    for batch in loader:
        pts = batch["points"].to(device)          # [B, N, 3]
        names = batch.get("label_name", None)
        if names is None:
            names = [None] * pts.shape[0]

        with torch.no_grad():
            toks = tok(pts)                        # [B, K, D_tok]
            z_shape = mlp(toks)                    # [B, d_clip], L2
            sim = z_shape @ text_emb.t()           # [B, L]

        metrics = topk_accuracy(sim, names, labels, ks=(1, 5))
        bsz = pts.shape[0]
        if not (math.isnan(metrics["top1"]) or math.isnan(metrics["top5"])):
            all_top1 += metrics["top1"] * bsz
            all_top5 += metrics["top5"] * bsz
        n += bsz

        if args.save_pred is not None:
            top5_idx = sim.topk(5, dim=-1).indices.tolist()
            for i in range(bsz):
                preds_i = [labels[j] for j in top5_idx[i]]
                pred_records.append({
                    "gt": names[i],
                    "pred_top1": preds_i[0],
                    "pred_top5": preds_i
                })

    if n > 0 and all_top1 > 0:
        print(f"[Result] Top-1: {all_top1 / n:.4f} | Top-5: {all_top5 / n:.4f}")
    else:
        print("[Result] 无法计算准确率（可能缺少 GT 标签），但已完成相似度预测。")

    if args.save_pred is not None:
        os.makedirs(os.path.dirname(args.save_pred), exist_ok=True)
        with open(args.save_pred, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, ensure_ascii=False, indent=2)
        print(f"[Saved] predictions -> {args.save_pred}")


if __name__ == "__main__":
    import math
    main()
