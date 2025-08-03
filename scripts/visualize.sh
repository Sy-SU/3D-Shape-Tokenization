#!/bin/bash

# ========== Visualization Script ==========
# 可视化重建点云与 Ground Truth 的对比图（静态 PNG + 动态 GIF）

# 用法:
#   ./scripts/visualize.sh <timestamp> [max_batches]
# 说明:
#   <timestamp>: 重建结果目录名（outs/reconstruct/<timestamp>）
#   [max_batches]: 最多可视化的 batch 数量，默认值为 10

TIMESTAMP=$1
MAX_BATCHES=${2:-10}

if [ -z "$TIMESTAMP" ]; then
  echo "❌ 错误：缺少 timestamp 参数"
  echo "用法: ./scripts/visualize.sh <timestamp> [max_batches]"
  exit 1
fi

echo "🎨 正在可视化重建结果 timestamp=$TIMESTAMP, 最多显示 $MAX_BATCHES 个 batch..."
python tools/visualize.py "$TIMESTAMP" --max_batches "$MAX_BATCHES"
