# utils/velocity_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FourierPositionalEncoding3D, RMSNorm

# ========== Cross Attention Block for Shape Tokenizer ==========


class CrossAttentionBlock(nn.Module):
    """
    实现 velocity estimator 中使用的 Cross Attention 模块。

    输入:
        query: Tensor[B, d]，表示每个样本的查询向量
        point_features: Tensor[B, k, d]，每个样本的 k 个 shape tokens

    输出:
        attended_output: Tensor[B, d]，聚合了 shape tokens 信息的输出向量

    模块结构:
        - LayerNorm（query / key 分开）
        - Linear_q / Linear_k / Linear_v（分头）
        - RMSNorm（q / k）
        - Multi-head Attention（batch 版）
        - Linear_o 输出
        - 残差连接 + 前馈 MLP（LayerNorm → Linear → GELU → Linear → Residual）
    """
    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        assert d % n_heads == 0, "d 必须能整除 n_heads"

        self.layernorm_q = nn.LayerNorm(d)
        self.layernorm_kv = nn.LayerNorm(d)

        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)

        self.rmsnorm_q = RMSNorm(d)
        self.rmsnorm_k = RMSNorm(d)

        self.linear_o = nn.Linear(d, d)

        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )

    def forward(self, query: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query: Tensor[B, d]
            point_features: Tensor[B, k, d]
        返回:
            Tensor[B, d]
        """
        B, d = query.shape
        B2, k, d2 = point_features.shape

        if B != B2 or d != d2:
            raise ValueError(f"输入维度不匹配: query {query.shape}, point_features {point_features.shape}")

        # 1. LayerNorm & projection
        q_input = self.layernorm_q(query)                      # [B, d]
        kv_input = self.layernorm_kv(point_features)           # [B, k, d]

        q_proj = self.rmsnorm_q(self.linear_q(q_input))        # [B, d]
        k_proj = self.rmsnorm_k(self.linear_k(kv_input))       # [B, k, d]
        v_proj = self.linear_v(kv_input)                       # [B, k, d]

        # 2. Reshape to multi-head
        q_proj = q_proj.view(B, self.n_heads, 1, self.head_dim)     # [B, h, 1, d_h]
        k_proj = k_proj.view(B, self.n_heads, k, self.head_dim)     # [B, h, k, d_h]
        v_proj = v_proj.view(B, self.n_heads, k, self.head_dim)     # [B, h, k, d_h]

        # 3. Scaled Dot Product Attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, 1, k]
        attn = torch.softmax(scores, dim=-1)                      # [B, h, 1, k]
        out = torch.matmul(attn, v_proj)                          # [B, h, 1, d_h]

        # 4. 合并多头输出
        out = out.squeeze(2).transpose(1, 2).reshape(B, d)        # [B, d]

        # 5. 输出 projection + residual
        out = self.linear_o(out) + query                         # [B, d]

        # 6. FeedForward MLP + residual
        mlp_out = self.mlp(self.mlp_norm(out))                   # [B, d]
        out = out + mlp_out

        return out

# ========== Velocity Estimator ==========


class TimeEncoder(nn.Module):
    """
    编码 flow matching 时间 t 的模块。

    输入:
        t: Tensor[B] 或 [B, 1]，取值范围在 [0, 1]
    输出:
        t_emb: Tensor[B, d]，时间嵌入向量
    """
    def __init__(self, d):
        super().__init__()
        self.freqs = 2 * torch.pi * torch.logspace(0, 16, steps=16, base=2.0)
        self.linear1 = nn.Linear(32, 64)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(64, d)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]  # [B, 1]
        device = t.device
        freqs = self.freqs.to(device)[None, :]  # [1, 16]
        emb = torch.cat([torch.sin(freqs * t), torch.cos(freqs * t)], dim=-1)  # [B, 32]
        out = self.linear2(self.act(self.linear1(emb)))  # [B, d]
        return out


class AdaLayerNorm(nn.Module):
    """
    自适应归一化：根据时间嵌入 t_emb 对输入 x 进行调制。

    输入:
        x: Tensor[B, d]
        t_emb: Tensor[B, d]
    输出:
        Tensor[B, d]
    """
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.shift = nn.Linear(d, d)
        self.scale = nn.Linear(d, d)

    def forward(self, x, t_emb):
        x_norm = self.norm(x)
        shift = self.shift(t_emb)
        scale = self.scale(t_emb)
        return x_norm * (1 + scale) + shift


# ========== VelocityEstimatorBlock ==========
class VelocityEstimatorBlock(nn.Module):
    """
    Flow Matching Decoder 中的基本模块。

    包含：
        - 输入点特征线性变换
        - LayerNorm + 调制（shift1 & scale1）
        - Cross Attention（调制后的 query）
        - Gating 1（门控）
        - AdaLayerNorm + MLP（前馈网络）
        - LayerNorm → shift2 & scale2 + Gating 2（门控）
        - x_norm 与 h 残差连接，最终加上 h2
    """
    def __init__(self, d, pe_dim):
        super().__init__()
        self.point_proj = nn.Linear(pe_dim, d)
        self.layernorm1 = nn.LayerNorm(d)
        self.shift1 = nn.Linear(d, d)
        self.scale1 = nn.Linear(d, d)
        self.gate1 = nn.Linear(d, d)

        self.cross_attn = CrossAttentionBlock(d, n_heads=8)

        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        self.layernorm2 = AdaLayerNorm(d)
        self.shift2 = nn.Linear(d, d)
        self.scale2 = nn.Linear(d, d)
        self.gate2 = nn.Linear(d, d)

    def forward(self, x_pe, s, t_emb):
        """
        参数:
            x_pe: Tensor[B, pe_dim]，Fourier 编码后的点
            s: Tensor[B, k, d]，Shape Tokens
            t_emb: Tensor[B, d]，TimeEncoder 的输出
        返回:
            Tensor[B, d]，预测的点速度特征
        """
        # 调制 query
        x_proj = self.point_proj(x_pe)         # [B, d]
        x_norm = self.layernorm1(x_proj)       # [B, d]
        shift1 = self.shift1(t_emb)            # [B, d]
        scale1 = self.scale1(t_emb)            # [B, d]
        query = x_norm * (1 + scale1) + shift1  # [B, d]

        # Cross Attention
        attn_out = self.cross_attn(query, s)   # [B, d]

        # Gating 1（乘以 sigmoid）
        gate1 = torch.sigmoid(self.gate1(t_emb))
        h = attn_out * gate1                   # [B, d]

        # 与 x_norm 残差连接
        h = h + x_norm                         # [B, d]

        # LayerNorm 后进行 shift2, scale2, gate2 调制
        h2 = self.layernorm2(h, t_emb)               # [B, d]
        shift2 = self.shift2(t_emb)            # [B, d]
        scale2 = self.scale2(t_emb)            # [B, d]
        h2 = self.mlp(h2)                      # [B, d]
        gate2 = torch.sigmoid(self.gate2(t_emb))  # [B, d]
        h2 = h2 * (1 + scale2) + shift2        # [B, d]
        h2 = h2 * gate2                        # [B, d]

        return h + h2  # 最终输出：包含 x_norm、h、h2 的完整残差路径


# ========== VelocityEstimator ==========
class VelocityEstimator(nn.Module):
    """
    Flow Matching Velocity Decoder。

    输入:
        x: Tensor[B, 3]，原始 3D 点
        s: Tensor[B, k, d]，Shape Tokens
        t: Tensor[B]，flow matching 时间 [0, 1]

    输出:
        velocity: Tensor[B, 3]
    """
    def __init__(self, d=512, num_frequencies=16, n_blocks=3):
        super().__init__()
        self.pos_encoder = FourierPositionalEncoding3D(num_frequencies=num_frequencies, include_input=True)
        self.pe_dim = 3 + 3 * 2 * num_frequencies
        self.time_encoder = TimeEncoder(d)

        self.blocks = nn.ModuleList([
            VelocityEstimatorBlock(d, self.pe_dim) for _ in range(n_blocks)
        ])

        self.final = nn.Linear(d, 3)  # 输出 3D 速度

    def forward(self, x, s, t):
        x_pe = self.pos_encoder(x)         # [B, pe_dim]
        t_emb = self.time_encoder(t)       # [B, d]

        h = None
        for block in self.blocks:
            h = block(x_pe, s, t_emb)      # [B, d]

        v = self.final(h)                  # [B, 3]
        return v


# ========== Unit Test ==========
if __name__ == "__main__":
    # 测试 VelocityEstimator 是否能正常 forward
    B, k, d = 4, 16, 512
    pe_dim = 3 + 3 * 2 * 16  # 与 pos_encoder 一致

    x = torch.randn(B, 3)              # 原始点
    s = torch.randn(B, k, d)          # shape tokens
    t = torch.rand(B)                 # 时间 t ∈ [0, 1]

    model = VelocityEstimator(d=d, num_frequencies=16, n_blocks=3)
    v = model(x, s, t)                # [B, 3]

    print("Output shape:", v.shape)
    assert v.shape == (B, 3), "VelocityEstimator 输出维度应为 [B, 3]"
    print("✅ VelocityEstimator 单元测试通过！")
