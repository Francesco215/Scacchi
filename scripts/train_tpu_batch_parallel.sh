#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${SCACCHI_REPO_DIR:-/home/francescosacco/Scacchi-pod}"
PYTHON="${SCACCHI_TPU_PYTHON:-/home/francescosacco/Scacchi-pod/.venv/bin/python}"
EOPOD="${SCACCHI_EOPOD:-/home/francescosacco/Scacchi/.venv/bin/eopod}"

REMOTE_CMD="cd ${REPO_DIR} && PYTHONPATH=${REPO_DIR} JAX_PLATFORMS=tpu,cpu ${PYTHON} -m scacchi.train $*"

exec "${EOPOD}" run --worker all "${REMOTE_CMD}"
