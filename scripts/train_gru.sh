#!/usr/bin/env bash
# Train the GRU ablation (3 input settings x 3 seeds) and score on the shared test set.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in type type+loc type+loc+player; do
  echo "=== GRU features: $f ==="
  python -m badminton_coach.gru_model --features "$f" --seeds 42 43 44
done
