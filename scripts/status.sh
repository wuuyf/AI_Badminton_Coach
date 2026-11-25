#!/usr/bin/env bash
# Print a compact snapshot of the project: file tree, git state and code fingerprints.
# Usage: bash scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== Files tracked by git (what is on GitHub) =="
git ls-files | sed 's/^/  /'
echo
echo "== Local-only folders (ignored by git) =="
for d in .env data/raw data/processed third_party/CoachAI-Projects model; do
  [ -e "$d" ] && echo "  ✓ $d" || echo "  – $d (missing)"
done
echo
echo "== Git =="
echo "  branch: $(git branch --show-current)   last commit: $(git log -1 --format='%h %s')"
changes=$(git status --short)
[ -z "$changes" ] && echo "  no uncommitted changes" || { echo "  uncommitted changes:"; echo "$changes" | sed 's/^/    /'; }
echo
echo "== Code fingerprints (first 12 chars of SHA-256) =="
for f in src/badminton_coach/*.py scripts/*.sh requirements.txt pyproject.toml; do
  [ -f "$f" ] && printf "  %s  %s\n" "$(shasum -a 256 "$f" | cut -c1-12)" "$f"
done
