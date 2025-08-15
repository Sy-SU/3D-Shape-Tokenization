# models/shape_tokenizer.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import FourierPositionalEncoding3D, PointFeatureProjector, RMSNorm

# ========== Cross Attention Block for Shape Tokenizer ==========


class CrossAttentionBlock(nn.Module):
    """
    实现 shape tokenizer 中的 Cross Attention 模块。

    输入:
        query_tokens: Tensor[B, k, d_f]，可学习的初始 token
        point_features: Tensor[B, n, d_f]，位置编码并投影后的点云特征
    输出:
        attended_tokens: Tensor[B, k, d_f]，聚合了点云信息的 token 表示

    架构:
        - LayerNorm (query / key-value 分开)
        - Linear_q / Linear_k / Linear_v
        - RMSNorm (query / key 分开)
        - Scaled Dot-Product Attention
        - Linear_o 投影
        - Residual + FeedForward: LayerNorm → Linear → GELU → Linear → Residual
    """
    def __init__(self, d_f, n_heads=8):
        super().__init__()
        self.d_f = d_f
        self.n_heads = n_heads
        self.head_dim = d_f // n_heads

        assert d_f % n_heads == 0, "d_f 必须可以被 n_heads 整除"

        # LayerNorm
        self.layernorm_q = nn.LayerNorm(d_f)
        self.layernorm_kv = nn.LayerNorm(d_f)

        # Linear projections
        self.linear_q = nn.Linear(d_f, d_f)
        self.linear_k = nn.Linear(d_f, d_f)
        self.linear_v = nn.Linear(d_f, d_f)

        # RMSNorm (标准实现)
        self.rmsnorm_q = RMSNorm(d_f)
        self.rmsnorm_k = RMSNorm(d_f)

        # Output projection
        self.linear_o = nn.Linear(d_f, d_f)

        # ===== MLP + Residual Block=====
        self.mlp_norm = nn.LayerNorm(d_f)  # MLP 前的 LayerNorm
        self.mlp = nn.Sequential(          # 前馈网络：Linear → GELU → Linear
            nn.Linear(d_f, d_f * 4),
            nn.GELU(),
            nn.Linear(d_f * 4, d_f)
        )

    def forward(self, query_tokens, point_features):
        """
        参数:
            query_tokens: Tensor[B, k, d_f]
            point_features: Tensor[B, n, d_f]
        返回:
            Tensor[B, k, d_f]
        """
        B, k_token, d_f = query_tokens.shape
        _, n_point, _ = point_features.shape

        # LayerNorm
        q_input = self.layernorm_q(query_tokens)
        kv_input = self.layernorm_kv(point_features)

        # Linear projections
        q_proj = self.linear_q(q_input)
        k_proj = self.linear_k(kv_input)
        v_proj = self.linear_v(kv_input)

        # RMSNorm
        q_proj = self.rmsnorm_q(q_proj)
        k_proj = self.rmsnorm_k(k_proj)

        # Reshape for multi-head attention
        q_proj = q_proj.view(B, k_token, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, k, d_h]
        k_proj = k_proj.view(B, n_point, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, n, d_h]
        v_proj = v_proj.view(B, n_point, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, n, d_h]

        # Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, k, n]
        attn = torch.softmax(scores, dim=-1)  # [B, h, k, n]
        out = torch.matmul(attn, v_proj)  # [B, h, k, d_h]

        # 合并多头输出
        out = out.transpose(1, 2).contiguous().view(B, k_token, d_f)  # [B, k, d_f]

        # Residual 1: attention + query_tokens
        out = self.linear_o(out) + query_tokens

        # Residual 2: MLP(layernorm(attn + query)) + (attn + query)
        mlp_out = self.mlp(self.mlp_norm(out))
        out = out + mlp_out

        return out

# ========== Self Attention Block for Shape Tokenizer ==========


class SelfAttentionBlock(nn.Module):
    """
    实现 shape tokenizer 中的 Self Attention 模块（重复 2 次）。

    输入:
        tokens: Tensor[B, k, d_f]，跨 token 的上下文建模
    输出:
        tokens: Tensor[B, k, d_f]，融合上下文信息的 token 表示

    架构:
        - LayerNorm
        - Linear_q / Linear_k / Linear_v（来自相同输入）
        - RMSNorm_q / RMSNorm_k（分开）
        - 多头注意力（query, key, value 都是 tokens）
        - Linear_o 投影
        - Residual + FeedForward: LayerNorm → Linear → GELU → Linear → Residual
    """
    def __init__(self, d_f, d, n_heads=8):
        super().__init__()
        self.d_f = d_f
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d_f // n_heads

        assert d_f % n_heads == 0, "d_f 必须可以被 n_heads 整除"

        # LayerNorm
        self.layernorm = nn.LayerNorm(d_f)

        # Linear projections
        self.linear_q = nn.Linear(d_f, d_f)
        self.linear_k = nn.Linear(d_f, d_f)
        self.linear_v = nn.Linear(d_f, d_f)

        # RMSNorm
        self.rmsnorm_q = RMSNorm(d_f)
        self.rmsnorm_k = RMSNorm(d_f)

        # Output projection
        self.linear_o = nn.Linear(d_f, d_f)

        # MLP + Residual Block
        self.mlp_norm = nn.LayerNorm(d_f)
        self.mlp = nn.Sequential(
            nn.Linear(d_f, d_f * 4),
            nn.GELU(),
            nn.Linear(d_f * 4, d_f)
        )

    def forward(self, tokens):
        """
        参数:
            tokens: Tensor[B, k, d_f]
        返回:
            Tensor[B, k, d_f]
        """
        B, k_token, d_f = tokens.shape

        # LayerNorm
        x_norm = self.layernorm(tokens)

        # Linear projections
        q_proj = self.linear_q(x_norm)
        k_proj = self.linear_k(x_norm)
        v_proj = self.linear_v(x_norm)

        # RMSNorm
        q_proj = self.rmsnorm_q(q_proj)
        k_proj = self.rmsnorm_k(k_proj)

        # Multi-head attention reshape
        q_proj = q_proj.view(B, k_token, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, k, d_h]
        k_proj = k_proj.view(B, k_token, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, k, d_h]
        v_proj = v_proj.view(B, k_token, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, k, d_h]

        # Attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, k, k]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_proj)  # [B, h, k, d_h]

        # 合并多头输出
        out = out.transpose(1, 2).contiguous().view(B, k_token, d_f)  # [k, d_f]

        # Residual 1
        out = self.linear_o(out) + tokens

        # Residual 2
        mlp_out = self.mlp(self.mlp_norm(out))
        out = out + mlp_out

        return out

# ========== Token Block ==========


class TokenBlock(nn.Module):
    """
    封装一个 block: CrossAttention + SelfAttention ×2
    每次迭代从点云提取信息并融合上下文
    """
    def __init__(self, d_f, d, n_heads):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(d_f, n_heads)
        self.self_attn1 = SelfAttentionBlock(d_f, d, n_heads)
        self.self_attn2 = SelfAttentionBlock(d_f, d, n_heads)

    def forward(self, tokens, point_features):
        tokens = self.cross_attn(tokens, point_features)
        tokens = self.self_attn1(tokens)
        tokens = self.self_attn2(tokens)
        return tokens

# ========== Shape Tokenizer ==========


class ShapeTokenizer(nn.Module):
    """
    实现 Shape Tokenizer。

    输入:
        points: Tensor[n, 3]，原始点云坐标
    输出:
        shape_tokens: Tensor[k, d_f]，用于生成的形状 token 表达

    模块结构:
        - FourierPositionalEncoding3D
        - PointFeatureProjector
        - 初始化 learnable tokens
        - 6 × (CrossAttentionBlock + SelfAttentionBlock × 2)
    """
    def __init__(self, num_tokens: int = 32, d_in: int = 3, d_f: int = 512, d: int = 64, n_heads: int = 8, num_frequencies: int = 16, num_blocks: int = 16):
        super().__init__()
        self.k = num_tokens
        self.d_f = d_f
        self.d = d
        self.output_proj = nn.Linear(d_f, d)

        # Fourier 编码器 + 投影模块
        self.pos_encoder = FourierPositionalEncoding3D(num_frequencies=num_frequencies, include_input=True)
        pe_dim = 3 + 3 * 2 * num_frequencies  # 输入为3维点，每维产生 2 × freq 个新特征
        self.projector = PointFeatureProjector(pe_dim, d_f)

        # 初始化可学习 token 向量 [k, d_f]
        self.token_embed = nn.Parameter(torch.randn(num_tokens, d_f))

        # 6 个重复 block
        self.blocks = nn.ModuleList([TokenBlock(d_f, d, n_heads) for _ in range(num_blocks)])

    def forward(self, x):
        """
        参数:
            x: Tensor[B, n, 3]，原始点云输入
        返回:
            shape_tokens: Tensor[B, k, d_f]
        """
        B, N, _ = x.shape

        # Step 1: Fourier 编码 → [n, pe_dim]
        x_encoded = self.pos_encoder(x.view(-1, 3))  # [B*N, pe_dim]

        # Step 2: 投影 → [n, d_f]
        x_projected = self.projector(x_encoded).view(B, N, self.d_f)  # [B, N, d_f]

        # Step 3: 6 × (Cross Attention + Self Attention × 2)
        tokens = self.token_embed.unsqueeze(0).expand(B, -1, -1)   # [B, k, d_f]
        for block in self.blocks:
            tokens = block(tokens, x_projected)

        token_out = self.output_proj(tokens)

        return token_out

# ========== Unit Test ==========


if __name__ == '__main__':
    # ========== Unit Test of Cross Attention Block ==========
    torch.manual_seed(42)

    B = 4
    d_f = 128
    k = 32    # number of learned tokens
    n = 2048  # number of input point features

    query = torch.randn(B, k, d_f)
    points = torch.randn(B, n, d_f)

    print("✅ Input shapes:")
    print("  Query shape:", query.shape)
    print("  Point feature shape:", points.shape)

    cross_attn = CrossAttentionBlock(d_f=d_f, n_heads=8)
    out = cross_attn(query, points)

    # 检查输出维度正确
    assert out.shape == (B, k, d_f), f"Output shape mismatch: got {out.shape}, expected ({k}, {d_f})"

    # 检查是否为有限值（无 inf / nan）
    assert torch.isfinite(out).all(), "Output contains NaN or Inf!"

    print("✅ Output shape:", out.shape)
    print("✅ Output sample:", out[0][:6])
    print("🎉 CrossAttentionBlock unit test passed.")
    # ========== End Unit Test of Cross Attention Block ==========

    # ========== Unit Test of Self Attention Block ==========
    tokens = torch.randn(B, k, d_f)
    self_attn = SelfAttentionBlock(d_f=d_f, n_heads=8)
    out2 = self_attn(tokens)

    # 检查输出维度正确
    assert out2.shape == (B, k, d_f), f"SelfAttention output shape mismatch: got {out2.shape}, expected ({k}, {d_f})"

    # 检查是否为有限值
    assert torch.isfinite(out2).all(), "SelfAttention output contains NaN or Inf!"

    print("✅ SelfAttention output shape:", out2.shape)
    print("✅ SelfAttention output sample:", out2[0][:6])
    print("🎉 SelfAttentionBlock unit test passed.")
    # ========== End Unit Test of Self Attention Block ==========

    # ========== Unit Test of Shape Tokenizer ==========
    dummy_points = torch.randn(B, n, 3)  # 原始点云输入 [n, 3]
    tokenizer = ShapeTokenizer(num_tokens=k, d_in=3, d_f=d_f, n_heads=8, num_frequencies=16, num_blocks=6)
    shape_tokens = tokenizer(dummy_points)

    # 检查输出维度
    assert shape_tokens.shape == (B, k, d_f), f"ShapeTokenizer output shape mismatch: got {shape_tokens.shape}, expected ({k}, {d_f})"

    # 检查是否为有限值
    assert torch.isfinite(shape_tokens).all(), "ShapeTokenizer output contains NaN or Inf!"

    print("✅ ShapeTokenizer output shape:", shape_tokens.shape)
    print("✅ ShapeTokenizer output sample:", shape_tokens[0][:6])
    print("🎉 ShapeTokenizer unit test passed.")
    # ========== End Unit Test of Shape Tokenizer ==========
