#!/bin/bash

echo "🚀 开始模型评估..."

python tools/eval_reconstruct.py \
  --data_root "/root/autodl-fs/demo" \
  --batch_size 1 \
  --num_points 1024 \
  --num_workers 1 \
  --save_dir "demo" \