#!/usr/bin/env bash
# Train the official ShuttleNet with 3 seeds and score each on our shared test set.
# Usage: bash scripts/train_shuttlenet.sh [seed ...]      (default: 42 43 44)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SF="third_party/CoachAI-Projects/Stroke Forecasting"
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(42 43 44); fi

for s in "${SEEDS[@]}"; do
  echo "=== ShuttleNet seed $s ==="
  (cd "$SF" && python train.py --model_type ShuttleNet \
      --output_folder_name "./model/shuttlenet_seed${s}/" \
      --encode_length 4 --max_ball_round 35 \
      --shot_dim 32 --area_dim 32 --player_dim 32 --encode_dim 32 \
      --seed_value "$s")
  python -m badminton_coach.shuttlenet_eval --model_dir "$ROOT/$SF/model/shuttlenet_seed${s}/1" --seed "$s"
done
