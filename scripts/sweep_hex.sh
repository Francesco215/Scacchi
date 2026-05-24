#!/usr/bin/env bash
#
# Architecture sweep for hex following Andy Jones' "Scaling Scaling Laws with
# Board Games" (Table II / Fig. 5). For each board size we sweep a grid of
# (width, depth), capping each run at the per-board-size sample budget.
#
# FLOP-s caps from the paper are recorded as comments next to each board size;
# this script enforces only the sample-based stop (max_num_iters = sample_limit
# / samples-per-iter), which is the easiest knob to turn from the existing
# trainer.
#
# Usage:
#   scripts/sweep_hex.sh                       # all board sizes 3..9
#   scripts/sweep_hex.sh --board-sizes 3,4,5   # subset (comma- or space-sep)
#   scripts/sweep_hex.sh -b "3 4" logging.wandb.enabled=false   # + hydra overrides

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Read selfplay.batch_size / selfplay.max_num_steps from the hex config so the iter
# budget stays in sync with whatever the trainer actually consumes.
CONFIG_PATH="${ROOT_DIR}/scacchi/configs/hex.yaml"
read_yaml_int() {
    uv run python - "$CONFIG_PATH" "$1" <<'PY'
import sys

import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    value = yaml.safe_load(fh)
for part in sys.argv[2].split("."):
    value = value[part]
print(str(value).replace("_", ""))
PY
}
SELFPLAY_BATCH_SIZE="$(read_yaml_int selfplay.batch_size)"
MAX_NUM_STEPS="$(read_yaml_int selfplay.max_num_steps)"
: "${SELFPLAY_BATCH_SIZE:?could not read selfplay.batch_size from ${CONFIG_PATH}}"
: "${MAX_NUM_STEPS:?could not read selfplay.max_num_steps from ${CONFIG_PATH}}"
SAMPLES_PER_ITER=$(( SELFPLAY_BATCH_SIZE * MAX_NUM_STEPS ))

# Widths and depths per board size — powers of two, capped by Table II.
WIDTHS_3="1 2"
WIDTHS_4="1 2 4 8 16"
WIDTHS_5="1 2 4 8 16"
WIDTHS_6="1 2 4 8 16 32 64 128"
WIDTHS_7="1 2 4 8 16 32 64 128 256 512"
WIDTHS_8="1 2 4 8 16 32 64 128 256 512"
WIDTHS_9="1 2 4 8 16 32 64 128 256 512 1024"

DEPTHS_3="1 2 4"
DEPTHS_4="1 2 4"
DEPTHS_5="1 2 4 8"
DEPTHS_6="1 2 4 8"
DEPTHS_7="1 2 4 8"
DEPTHS_8="1 2 4 8"
DEPTHS_9="1 2 4 8"

# Sample stop limits (FLOP-s caps in comments — not enforced here).
SAMPLES_3=400000000     # 4e8 samples / 1e12 FLOP-s
SAMPLES_4=200000000     # 2e8 samples / 1e13 FLOP-s
SAMPLES_5=300000000     # 3e8 samples / 3e13 FLOP-s
SAMPLES_6=600000000     # 6e8 samples / 4e14 FLOP-s
SAMPLES_7=1000000000    # 1e9 samples / 1e16 FLOP-s
SAMPLES_8=1000000000    # 1e9 samples / 3e16 FLOP-s
SAMPLES_9=2000000000    # 2e9 samples / 1e17 FLOP-s

BOARD_SIZES=(3 4 5 6 7 8 9)

# Parse leading --board-sizes / -b flag; everything after is forwarded to hydra.
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --board-sizes|-b)
            IFS=', ' read -r -a BOARD_SIZES <<< "$2"
            shift 2
            ;;
        --board-sizes=*)
            IFS=', ' read -r -a BOARD_SIZES <<< "${1#*=}"
            shift
            ;;
        --)
            shift
            HYDRA_ARGS+=("$@")
            break
            ;;
        *)
            HYDRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate that every requested board size has a sweep grid defined.
for bs in "${BOARD_SIZES[@]}"; do
    if ! declare -p "WIDTHS_${bs}" >/dev/null 2>&1; then
        echo "error: no sweep grid defined for board_size=${bs}" >&2
        exit 1
    fi
done

echo "sweeping board sizes: ${BOARD_SIZES[*]}"

for bs in "${BOARD_SIZES[@]}"; do
    widths_var="WIDTHS_${bs}"
    depths_var="DEPTHS_${bs}"
    samples_var="SAMPLES_${bs}"
    widths="${!widths_var}"
    depths="${!depths_var}"
    samples="${!samples_var}"

    # Ceiling division so we don't under-shoot the sample budget.
    max_num_iters=$(( (samples + SAMPLES_PER_ITER - 1) / SAMPLES_PER_ITER ))

    for nc in $widths; do
        for nl in $depths; do
            echo "=== hex bs=${bs} width=${nc} depth=${nl} iters=${max_num_iters} (sample cap=${samples}) ==="
            uv run python -m scacchi.train \
                env.board_size="${bs}" \
                model.num_channels="${nc}" \
                model.num_layers="${nl}" \
                run.max_num_iters="${max_num_iters}" \
                "${HYDRA_ARGS[@]}"
        done
    done
done
