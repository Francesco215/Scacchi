#!/usr/bin/env bash

# Train every numbered Hex recipe sequentially on one CUDA device.
#
# Usage:
#   scripts/train_all_hex.sh
#   BOARD_SIZES="3 5 7" scripts/train_all_hex.sh logging.wandb.enabled=false

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BOARD_SIZES_VALUE="${BOARD_SIZES:-3 4 5 6 7 8 9 11}"
read -r -a BOARD_SIZE_LIST <<< "$BOARD_SIZES_VALUE"

NUM_CHANNELS=128
NUM_LAYERS=6
RUN_TAG="${RUN_TAG:-uniform_c128_l6}"
LOG_DIR="${LOG_DIR:-logs/train_all_hex/${RUN_TAG}}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/scacchi-uv-cache}"

mkdir -p "$LOG_DIR"

for board_size in "${BOARD_SIZE_LIST[@]}"; do
    config_name="hex${board_size}"
    config_path="scacchi/configs/${config_name}.yaml"
    if [[ ! -f "$config_path" ]]; then
        echo "Missing config for board size ${board_size}: ${config_path}" >&2
        exit 2
    fi

    checkpoint_dir="checkpoints/${config_name}_${RUN_TAG}"
    log_path="${LOG_DIR}/${config_name}.log"
    status_path="${LOG_DIR}/${config_name}.status"

    echo "running" > "$status_path"
    echo "Starting ${config_name}: ${NUM_CHANNELS} channels x ${NUM_LAYERS} layers"
    echo "Checkpoint: ${checkpoint_dir}"
    echo "Log: ${log_path}"

    if uv run scacchi-train \
        --config-name "$config_name" \
        "$@" \
        "model.num_channels=${NUM_CHANNELS}" \
        "model.num_layers=${NUM_LAYERS}" \
        "checkpointing.max_to_keep=0" \
        "checkpointing.directory=${checkpoint_dir}" \
        2>&1 | tee "$log_path"; then
        echo "completed" > "$status_path"
    else
        exit_code=$?
        echo "failed:${exit_code}" > "$status_path"
        exit "$exit_code"
    fi
done

echo "All Hex training runs completed."
