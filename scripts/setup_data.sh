#!/usr/bin/env bash
# Download ShuttleSet and the official ShuttleNet code from the upstream repository.
# Nothing downloaded here is committed to this repo (see .gitignore).
set -euo pipefail

UPSTREAM="https://github.com/wywyWang/CoachAI-Projects.git"
COMMIT="45517f7d4cb936b03f3eabf939cc7959d39226fe"   # pinned for reproducibility
DEST="third_party/CoachAI-Projects"

cd "$(dirname "$0")/.."

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout "$UPSTREAM" "$DEST"
fi

git -C "$DEST" sparse-checkout set "ShuttleSet" "Stroke Forecasting"
git -C "$DEST" checkout -q "$COMMIT"

# Local fix for macOS: worker processes re-import train.py (no __main__ guard),
# so use num_workers=0. Applied to the downloaded copy only; not committed.
perl -pi -e 's/num_workers=8/num_workers=0/g' "$DEST/Stroke Forecasting/badmintoncleaner.py"

mkdir -p data
ln -sfn "../$DEST/ShuttleSet/set" data/raw

echo "Done."
echo "  ShuttleSet  -> data/raw/ (match.csv, homography.csv, <match>/set*.csv)"
echo "  ShuttleNet  -> $DEST/Stroke Forecasting/"
