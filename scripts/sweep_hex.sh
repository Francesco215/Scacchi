#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BOARD_SIZES=(3 4 5 6 8 9)
NUM_CHANNELS=(8 32 64 128 512 1024)
NUM_LAYERS=(4 4 8 8 8 8)
MAX_NUM_ITERS=(100 150 300 300 1000 2000)

for i in "${!BOARD_SIZES[@]}"; do
    bs="${BOARD_SIZES[$i]}"
    nc="${NUM_CHANNELS[$i]}"
    nl="${NUM_LAYERS[$i]}"
    ni="${MAX_NUM_ITERS[$i]}"

    echo "=== Training hex board_size=${bs} num_channels=${nc} num_layers=${nl} max_num_iters=${ni} ==="

    uv run python -m scacchi.train \
        board_size="${bs}" \
        num_channels="${nc}" \
        num_layers="${nl}" \
        max_num_iters="${ni}" \
        "$@"
done
