#!/bin/bash
# Ablation: tail_ratio / tau_thr / rank, only on 7scenes-dense.
# 所有 run 都 append 到 ${output_dir}/tre_diff/all_metrics.csv,
# tag 列区分 (e.g. base / tail_005 / tau_007 / rank_32 ...)。
#
# 用法:
#   bash run_ablation_7sd.sh                # 默认全跑 13 个组合
#   bash run_ablation_7sd.sh tail_ratio     # 只跑 tail_ratio 那一组
#   GPU=2 bash run_ablation_7sd.sh tau      # 指定 GPU
#
# 默认值: tail_ratio=0.01, tau_thr=0.01, rank=64, skip_p=0.0 (纯阈值模式)。

set -e
GPU="${GPU:-0}"
CKPT="${CKPT:-/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt}"
WHICH="${1:-all}"   # all / base / tail_ratio / tau / rank

# taptq.py 用 "./configs" 这种相对路径, 所以 cwd 必须是 Pi3-evaluation 根。
# 优先用容器内符号链接, fallback 到 /data 直挂路径。
if [[ -d /root/Pi3-evaluation/configs ]]; then
  PI3_ROOT=/root/Pi3-evaluation
elif [[ -d /data/minimax-dialogue/users/boli/autodl_pull/Pi3-evaluation/configs ]]; then
  PI3_ROOT=/data/minimax-dialogue/users/boli/autodl_pull/Pi3-evaluation
else
  echo "ERROR: 找不到 Pi3-evaluation 根目录 (需要 configs/ 子目录)" >&2
  exit 1
fi
cd "$PI3_ROOT"
echo "cwd = $PI3_ROOT"
SCRIPT="mv_recon/taptq.py"

run () {
  local tag="$1"; shift
  echo "==================== [$(date +%H:%M:%S)] tag=${tag} ===================="
  CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} ++mode=compensate_eval \
      ++ckpt.path=${CKPT} ++compensate.strategy=module \
      "++eval_datasets=[7scenes-dense]" \
      ++save_suffix=${tag} "$@"
}

# Baseline: tail_ratio=0.01 / tau_thr=0.01 / rank=64 (默认, 不传 override)。
# 其他 sweep 都对照这个 baseline。
if [[ "$WHICH" == "all" || "$WHICH" == "base" ]]; then
  run base
fi

# --- tail_ratio sweep (TRE 取 top-K 比例) ---
# 默认 0.01 (= top 1% magnitude). 调大让 TRE 看更广分布, 调小聚焦 outlier。
if [[ "$WHICH" == "all" || "$WHICH" == "tail_ratio" || "$WHICH" == "tail" ]]; then
  run tail_0001  ++compensate.tail_ratio=0.001
  run tail_0005  ++compensate.tail_ratio=0.005
  run tail_001   ++compensate.tail_ratio=0.01
  run tail_002   ++compensate.tail_ratio=0.02
  run tail_005   ++compensate.tail_ratio=0.05
  run tail_01    ++compensate.tail_ratio=0.1
  run tail_02    ++compensate.tail_ratio=0.2
  run tail_05    ++compensate.tail_ratio=0.5
  run tail_1     ++compensate.tail_ratio=1.0
fi

# # --- tau_thr sweep (legacy 阈值, skip 低 TRE 模块) ---
# # 默认 0.01. 注意 skip_p 默认 0.0 = 纯阈值模式 (rand_skip 全 True, 阈值才生效)。
# if [[ "$WHICH" == "all" || "$WHICH" == "tau" ]]; then
#   run tau_0      ++compensate.tau_thr=0.000    # ~不skip任何模块
  # run tau_001    ++compensate.tau_thr=0.001    
  # run tau_003    ++compensate.tau_thr=0.003
  # run tau_005    ++compensate.tau_thr=0.005
  # run tau_007    ++compensate.tau_thr=0.007
  # run tau_01     ++compensate.tau_thr=0.01
  # run tau_02     ++compensate.tau_thr=0.02
  # run tau_05     ++compensate.tau_thr=0.05     # 激进, 几乎全skip
  # run tau_07     ++compensate.tau_thr=0.07     # 激进, 几乎全skip
  # run tau_1      ++compensate.tau_thr=0.1     # 激进, 几乎全skip
#   run tau_10     ++compensate.tau_thr=1     # 全skip
# fi

# --- rank sweep (SVD 低秩补偿的秩) ---
# 默认 64. 影响 QwT 表达能力 / 显存。
if [[ "$WHICH" == "all" || "$WHICH" == "rank" ]]; then
  run rank_8     ++compensate.rank=8
  run rank_16    ++compensate.rank=16
  run rank_32    ++compensate.rank=32
  run rank_64    ++compensate.rank=64
  run rank_128   ++compensate.rank=128
  run rank_256    ++compensate.rank=256
fi

echo "==================== [$(date +%H:%M:%S)] DONE ===================="
echo "Results in: \${output_dir}/tre_diff/all_metrics.csv (filter dataset=7scenes-dense)"
