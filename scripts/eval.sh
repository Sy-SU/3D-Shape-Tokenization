#!/bin/bash

echo "🚀 开始模型评估..."

python tools/reconstruct.py \
  --data_root "/root/autodl-fs/chair" \
  --batch_size 4 \
  --num_points 2048 \
  --num_workers 4 \
  --save_dir "chair"