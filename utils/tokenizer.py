import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Cross Attention Block for Shape Tokenizer ==========


class CrossAttentionBlock(nn.Module):
    """
    实现 shape tokenizer 中的 Cross Attention 模块。

    输入:
        query_tokens: Tensor[k, d_f]，可学习的初始 token
        point_features: Tensor[n, d_f]，位置编码并投影后的点云特征
    输出:
        attended_tokens: Tensor[k, d_f]，聚合了点云信息的 token 表示

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
            query_tokens: Tensor[k, d_f]
            point_features: Tensor[n, d_f]
        返回:
            Tensor[k, d_f]
        """
        k_token, d_f = query_tokens.shape
        n_point, _ = point_features.shape

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
        q_proj = q_proj.view(k_token, self.n_heads, self.head_dim).transpose(0, 1)  # [h, k, d_h]
        k_proj = k_proj.view(n_point, self.n_heads, self.head_dim).transpose(0, 1)  # [h, n, d_h]
        v_proj = v_proj.view(n_point, self.n_heads, self.head_dim).transpose(0, 1)  # [h, n, d_h]

        # Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [h, k, n]
        attn = torch.softmax(scores, dim=-1)  # [h, k, n]
        out = torch.matmul(attn, v_proj)  # [h, k, d_h]

        # 合并多头输出
        out = out.transpose(0, 1).contiguous().view(k_token, d_f)  # [k, d_f]

        # Residual 1: attention + query_tokens
        out = self.linear_o(out) + query_tokens

        # Residual 2: MLP(layernorm(attn + query)) + (attn + query)
        mlp_out = self.mlp(self.mlp_norm(out))
        out = out + mlp_out

        return out


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    输入: Tensor[..., d]
    输出: 同形状张量，乘以可学习缩放因子
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ========== Unit Test ==========

if __name__ == '__main__':
    # ========== Unit Test of Cross Attention Block ==========
    torch.manual_seed(42)

    d_f = 128
    k = 32    # number of learned tokens
    n = 2048  # number of input point features

    query = torch.randn(k, d_f)
    points = torch.randn(n, d_f)

    print("✅ Input shapes:")
    print("  Query shape:", query.shape)
    print("  Point feature shape:", points.shape)

    cross_attn = CrossAttentionBlock(d_f=d_f, n_heads=8)
    out = cross_attn(query, points)

    # 检查输出维度正确
    assert out.shape == (k, d_f), f"Output shape mismatch: got {out.shape}, expected ({k}, {d_f})"

    # 检查是否为有限值（无 inf / nan）
    assert torch.isfinite(out).all(), "Output contains NaN or Inf!"

    print("✅ Output shape:", out.shape)
    print("✅ Output sample:", out[0][:6])
    print("🎉 CrossAttentionBlock unit test passed.")
    # ========== End Unit Test of Cross Attention Block ==========
