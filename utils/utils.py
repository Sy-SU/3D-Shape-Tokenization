import torch
import torch.nn as nn
import math

# ========== Positional Encoding Modules ==========


class FourierPositionalEncoding3D(nn.Module):
    """
    将3D点云坐标编码为高维傅里叶特征。

    输入: Tensor[n, 3]，每行为 (x, y, z)
    输出: Tensor[n, out_dim]，其中 out_dim = 3 * 2 * num_frequencies + (3 if include_input)
    """
    def __init__(self, num_frequencies=32, include_input=True, log_scale=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if log_scale:
            self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

    def forward(self, coords):
        """
        参数:
            coords: Tensor[n, 3]，每行为一个点的坐标 (x, y, z)
        返回:
            Tensor[n, out_dim]：傅里叶位置编码特征
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected input of shape [n, 3], but got {coords.shape}")

        coords = coords.unsqueeze(-1)  # [n, 3, 1]
        freq = self.freq_bands.to(coords.device).reshape(1, 1, -1) * math.pi  # [1, 1, L]

        scaled = coords * freq  # [n, 3, L]
        sin_feat = torch.sin(scaled)  # [n, 3, L]
        cos_feat = torch.cos(scaled)  # [n, 3, L]

        encoding = torch.cat([sin_feat, cos_feat], dim=-1)  # [n, 3, 2L]
        encoding = encoding.view(coords.shape[0], -1)  # [n, 3*2L]

        if self.include_input:
            encoding = torch.cat([coords.squeeze(-1), encoding], dim=-1)  # [n, 3 + 3*2L]

        return encoding


class PointFeatureProjector(nn.Module):
    """
    将 Fourier 编码后的特征投影到统一维度 d_f

    输入: Tensor[n, in_dim]
    输出: Tensor[n, d_f]
    """
    def __init__(self, in_dim, d_f):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_f)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected input of shape [n, in_dim], but got {x.shape}")
        return self.linear(x)


# ========== Unit Test ==========

if __name__ == '__main__':
    # ========== Unit Test of Positional Encoding Modules ==========

    num_points = 10
    num_frequencies = 16
    d_f = 128

    dummy_input = torch.randn(num_points, 3)  # [10, 3]
    print("Input shape:", dummy_input.shape)

    # 初始化位置编码器
    encoder = FourierPositionalEncoding3D(num_frequencies=num_frequencies, include_input=True)
    encoded = encoder(dummy_input)
    print("Encoded shape:", encoded.shape)  # 应为 [10, 3 + 3×2×16] = [10, 99]

    # 初始化投影器
    projector = PointFeatureProjector(encoded.shape[1], d_f)
    projected = projector(encoded)
    print("Projected shape:", projected.shape)  # 应为 [10, 128]

    # 显示一个样本特征
    print("First encoded vector:", encoded[0][:6])
    print("First projected vector:", projected[0][:6])

    # ========== End Unit Test of Positional Encoding Modules ==========
