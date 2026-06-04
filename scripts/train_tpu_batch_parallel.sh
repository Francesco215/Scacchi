#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script with bash, not source: bash ${BASH_SOURCE[0]} ..." >&2
  return 2
fi

set -euo pipefail

REPO_DIR="${SCACCHI_REPO_DIR:-/home/francescosacco/Scacchi-pod}"
PYTHON="${SCACCHI_TPU_PYTHON:-/home/francescosacco/Scacchi-pod/.venv/bin/python}"
LOCAL_REPO_DIR="${SCACCHI_LOCAL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TPU_NAME="${SCACCHI_TPU_NAME:-e_server_spot}"
TPU_ZONE="${SCACCHI_TPU_ZONE:-us-central2-b}"
TPU_PROJECT="${SCACCHI_TPU_PROJECT:-my-phd-research-o}"
SYNC="${SCACCHI_SYNC:-1}"
DRY_RUN="${SCACCHI_DRY_RUN:-0}"

if [[ -n "${SCACCHI_EOPOD:-}" ]]; then
  EOPOD="${SCACCHI_EOPOD}"
elif command -v eopod >/dev/null 2>&1; then
  EOPOD="$(command -v eopod)"
elif [[ -x /home/francescosacco/orchestrator-venv/bin/eopod ]]; then
  EOPOD="/home/francescosacco/orchestrator-venv/bin/eopod"
elif [[ -x /home/francescosacco/easy-venv/bin/eopod ]]; then
  EOPOD="/home/francescosacco/easy-venv/bin/eopod"
else
  echo "Could not find eopod. Set SCACCHI_EOPOD=/path/to/eopod." >&2
  exit 1
fi

if [[ "${SYNC}" == "1" ]]; then
  archive="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync.XXXXXX.tar.gz")"
  file_list="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync-files.XXXXXX")"
  delete_list="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync-delete.XXXXXX")"
  cleanup() {
    rm -f "${archive}" "${file_list}" "${delete_list}"
  }
  trap cleanup EXIT

  (
    cd "${LOCAL_REPO_DIR}"
    git diff --name-only --diff-filter=ACMRT HEAD -- \
      | awk '!/^(baselines|checkpoints|wandb)\//' \
      > "${file_list}"
    git diff --name-only --diff-filter=D HEAD -- \
      | awk '!/^(baselines|checkpoints|wandb)\//' \
      > "${delete_list}"
    if [[ -s "${file_list}" ]]; then
      tar -czf "${archive}" --files-from "${file_list}"
    fi
  )

  if [[ -s "${file_list}" ]]; then
    echo "Syncing changed files:"
    sed 's/^/  /' "${file_list}"
  fi
  if [[ -s "${delete_list}" ]]; then
    echo "Deleting remote files:"
    sed 's/^/  /' "${delete_list}"
  fi
  if [[ ! -s "${file_list}" && ! -s "${delete_list}" ]]; then
    echo "No git diff changes to sync."
  fi

  if [[ "${DRY_RUN}" != "1" && -s "${file_list}" ]]; then
    remote_archive="/tmp/$(basename "${archive}")"
    gcloud compute tpus tpu-vm scp \
      "${archive}" \
      "${TPU_NAME}:${remote_archive}" \
      --zone="${TPU_ZONE}" \
      --project="${TPU_PROJECT}" \
      --worker=all

    "${EOPOD}" run --worker all \
      "mkdir -p ${REPO_DIR} && tar -xzf ${remote_archive} -C ${REPO_DIR} && rm -f ${remote_archive}"
  fi

  if [[ "${DRY_RUN}" != "1" && -s "${delete_list}" ]]; then
    remote_delete_list="/tmp/$(basename "${delete_list}")"
    gcloud compute tpus tpu-vm scp \
      "${delete_list}" \
      "${TPU_NAME}:${remote_delete_list}" \
      --zone="${TPU_ZONE}" \
      --project="${TPU_PROJECT}" \
      --worker=all

    "${EOPOD}" run --worker all \
      "cd ${REPO_DIR} && xargs -r rm -f < ${remote_delete_list} && rm -f ${remote_delete_list}"
  fi
fi

quote_remote() {
  printf "%q" "$1"
}

REMOTE_ENV=("PYTHONPATH=${REPO_DIR}" "JAX_PLATFORMS=tpu,cpu")
for name in \
  SCACCHI_PROFILE_DIR \
  SCACCHI_PROFILE_START_ITER \
  SCACCHI_PROFILE_NUM_ITERS \
  SCACCHI_PROFILE_PROCESS \
  SCACCHI_PROFILE_PERFETTO
do
  if [[ -n "${!name:-}" ]]; then
    REMOTE_ENV+=("${name}=${!name}")
  fi
done

REMOTE_ENV_STR=""
for assignment in "${REMOTE_ENV[@]}"; do
  REMOTE_ENV_STR+=" $(quote_remote "${assignment}")"
done

REMOTE_CMD="cd $(quote_remote "${REPO_DIR}") &&${REMOTE_ENV_STR} $(quote_remote "${PYTHON}") -u -m scacchi.train $*"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Would run:"
  echo "  ${EOPOD} run --worker all ${REMOTE_CMD}"
  exit 0
fi

"${EOPOD}" run --worker all "${REMOTE_CMD}"

if [[ -n "${SCACCHI_PROFILE_DIR:-}" ]]; then
  echo "Profile traces were written under ${SCACCHI_PROFILE_DIR} on the selected TPU workers."
fi
