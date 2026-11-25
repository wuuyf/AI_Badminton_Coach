#!/usr/bin/env bash
# Block commits that contain an API key or a forbidden file.
# Install once:  ln -sf ../../scripts/check_secrets.sh .git/hooks/pre-commit
set -uo pipefail

fail=0
files=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$files" ] && exit 0

# 1) Google / Gemini API keys start with "AIza" (39 chars total)
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if git show ":$f" | grep -qE 'AIza[0-9A-Za-z_-]{35}'; then
    echo "BLOCKED: possible Google API key in $f"; fail=1
  fi
  if git show ":$f" | grep -qiE 'GEMINI_API_KEY *= *["'"'"']?[A-Za-z0-9_-]{20,}'; then
    echo "BLOCKED: hard-coded GEMINI_API_KEY in $f"; fail=1
  fi
done <<< "$files"

# 2) Files that must never be committed
if echo "$files" | grep -qE '(^|/)\.env$|\.(pt|pth|ckpt)$|^data/raw/|^third_party/CoachAI'; then
  echo "BLOCKED: .env, model weights, raw data or third-party code is staged:"
  echo "$files" | grep -E '(^|/)\.env$|\.(pt|pth|ckpt)$|^data/raw/|^third_party/CoachAI'
  fail=1
fi

[ $fail -eq 1 ] && echo "Commit aborted. Unstage with: git restore --staged <file>"
exit $fail
