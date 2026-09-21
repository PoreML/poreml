#!/bin/bash
# Export the repository for double-blind review: a fresh directory holding the tracked tree of
# a branch — no git history (every commit carries its author), none of the paths marked
# `export-ignore` in .gitattributes (the development archive, the agent instructions) — as one
# commit by a neutral author, swept for identifying strings before it is declared ready.
#
#   util/release/export_anonymous.sh [BRANCH] [OUT_DIR]     # defaults: main, ../poreml_release
#
# Push OUT_DIR, never this repository. util/release/deny.txt (gitignored) holds the sweep's patterns.
set -euo pipefail
BRANCH="${1:-main}"
OUT="${2:-../poreml_release}"
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

[ -e "$OUT" ] && { echo "$OUT exists; remove it or name another directory"; exit 1; }
mkdir -p "$OUT"
git archive --format=tar "$BRANCH" | tar -x -C "$OUT"

python3 util/release/check_anonymous.py "$OUT" --deny util/release/deny.txt

cd "$OUT"
git init -q -b main
git add -A
GIT_AUTHOR_NAME="PoreML authors" GIT_AUTHOR_EMAIL="poreml@users.noreply.github.com" \
GIT_COMMITTER_NAME="PoreML authors" GIT_COMMITTER_EMAIL="poreml@users.noreply.github.com" \
    git commit -q -m "PoreML: a benchmark for machine learning on pore-scale multiphase flow"
echo "exported $(git ls-files | wc -l) files of $BRANCH to $OUT as one anonymous commit"
