#!/usr/bin/env bash
set -euo pipefail

REMOTE="nuseg-server"
REMOTE_DIR="/hy-tmp/NuSeg"

ssh "$REMOTE" "
set -e
cd '$REMOTE_DIR'

echo '===== BEFORE ====='
git status --short
git log -1 --oneline

if [ -n \"\$(git status --porcelain)\" ]; then
    echo
    echo 'ERROR: GPU server working tree is not clean.'
    echo 'Refusing to pull to avoid overwriting remote changes.'
    exit 1
fi

git fetch origin
git switch main
git pull --ff-only origin main

echo
echo '===== AFTER ====='
git status --short
git log -1 --oneline
"
