# inspect_model.py

"""
打印 ShapeTokenizer 与 VelocityEstimator 的每层参数维度与参数量
适用于在项目根目录下调试运行
"""

import os
import sys
import torch
from torch import nn

# 添加路径，确保可以导入模块
sys.path.append(os.path.abspath('.'))

from models.shape_tokenizer import ShapeTokenizer
from models.velocity_estimator import VelocityEstimator


def print_model_parameters(model: nn.Module, name: str = "Model"):
    """
    递归打印每一层模块的参数数量与权重维度
    """
    print(f"\n{name} Architecture Summary:\n{'=' * 60}")
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        print(f"{name:60} | shape: {str(tuple(param.shape)):20} | params: {param_count}")
        total_params += param_count
    print(f"{'=' * 60}\nTotal Parameters: {total_params:,}\n")


if __name__ == '__main__':
    # 初始化模型
    tokenizer = ShapeTokenizer(
        num_tokens=32,
        d_in=3,
        d_f=128,
        n_heads=8,
        num_frequencies=16,
        num_blocks=6
    )

    estimator = VelocityEstimator(
        d=128,
        num_frequencies=16,
        n_blocks=3
    )

    # 打印参数信息
    print_model_parameters(tokenizer, name="ShapeTokenizer")
    print_model_parameters(estimator, name="VelocityEstimator")
