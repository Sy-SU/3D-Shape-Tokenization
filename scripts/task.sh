#!/bin/bash
set -euo pipefail

# =========================
# 一键：训练 -> 评估 -> 可视化
# 必填参数：
#   --data_root <path>    数据集根目录
#   --timestamp <name>    保存/可视化用的时间戳目录名（outs/reconstruct/<timestamp>）
# 可选：
#   --max_batches <N>     可视化最多展示的 batch 数，默认 10
# =========================

DATA_ROOT=""
TIMESTAMP=""
MAX_BATCHES=10

usage() {
  cat <<EOF
用法:
  $0 --data_root <PATH> --timestamp <NAME> [--max_batches N]

示例:
  $0 --data_root ~/autodl-fs/demo --timestamp 20250821-1200 --max_batches 12
EOF
}

# --- 解析参数 ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root)
      DATA_ROOT="${2:-}"; shift 2 ;;
    --timestamp)
      TIMESTAMP="${2:-}"; shift 2 ;;
    --max_batches)
      MAX_BATCHES="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "未知参数: $1"; usage; exit 1 ;;
  esac
done

# --- 校验必填 ---
if [[ -z "$DATA_ROOT" || -z "$TIMESTAMP" ]]; then
  echo "❌ 缺少必填参数：--data_root 和 --timestamp 都要指定"
  usage; exit 1
fi

echo "📦 data_root      = $DATA_ROOT"
echo "🕒 timestamp      = $TIMESTAMP"
echo "🖼  max_batches   = $MAX_BATCHES"
echo "-----------------------------------------"

# ========== Step 1: 训练 ==========
echo "🚀 Step 1: 启动训练..."
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
  --kl_prior_weight 1e-4 \
  --data_root "$DATA_ROOT"

# ========== Step 2: 评估（重建）==========
# 将结果固定保存到 outs/reconstruct/<timestamp>
echo "🧪 Step 2: 开始模型评估并保存到 outs/reconstruct/$TIMESTAMP ..."
python tools/reconstruct.py \
  --data_root "$DATA_ROOT" \
  --batch_size 4 \
  --num_points 2048 \
  --num_workers 4 \
  --save_dir "$TIMESTAMP"

# 确认输出目录存在
OUT_DIR="outs/reconstruct/$TIMESTAMP"
if [[ ! -d "$OUT_DIR" ]]; then
  echo "❌ 预期输出目录不存在：$OUT_DIR"
  echo "   请检查 tools/reconstruct.py 是否按 --save_dir 写入到 outs/reconstruct/<save_dir>"
  exit 1
fi

# ========== Step 3: 可视化 ==========
echo "🎨 Step 3: 可视化 timestamp=$TIMESTAMP，最多 $MAX_BATCHES 个 batch ..."
python tools/visualize.py "$TIMESTAMP" --max_batches "$MAX_BATCHES"

echo "✅ 全流程完成！"
echo "   重建与可视化结果目录：$OUT_DIR"
