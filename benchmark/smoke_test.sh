#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/vllm-venv/bin/python}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"

"$PYTHON" -u -m benchmark \
    --title "smoke test" \
    --kinds mixed short \
    --rates 8 inf \
    --repeats 2 \
    --n-requests 32 \
    --n-slots 5 \
    --token-budget 256
