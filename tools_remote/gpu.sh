#!/usr/bin/env bash
set -euo pipefail

REMOTE="nuseg-server"
REMOTE_DIR="/hy-tmp/NuSeg"

if [ "$#" -eq 0 ]; then
    exec ssh -t "$REMOTE" "cd '$REMOTE_DIR' && exec bash"
fi

exec ssh -t "$REMOTE" "cd '$REMOTE_DIR' && $*"
