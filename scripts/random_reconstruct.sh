#!/bin/bash

echo "🚀 随机抽取测试样本..."

# 随机抽 100 个测试样本评估（可复现实验）
python tools/random_reconstruct.py \
  --data_root /root/autodl-fs/demo \
  --batch_size 4 --num_points 2048 --num_workers 2 \
  --num_samples 50 --seed 2025 \
  --num_tokens 32 --d_f 512 --d 64 --ode_steps 1000
