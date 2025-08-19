#!/bin/bash

echo "🚀 启动训练..."

python tools/train.py \
  --batch_size 32 \
  --num_points 2048 \
  --epochs 1000 \
  --lr 0.4 \
  --weight_decay 0 \
  --patience 15 \
  --d_f 512 \
  --d 64 \
  --num_tokens 32 \
  --kl_weight 1e-3 \
  --kl_prior_weight 1e-4\
  --data_root ~/autodl-fs/demo
