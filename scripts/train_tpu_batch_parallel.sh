#!/usr/bin/env bash

 eopod kill-tpu --force

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
SYNC_EVAL_CHECKPOINTS="${SCACCHI_SYNC_EVAL_CHECKPOINTS:-auto}"
DRY_RUN="${SCACCHI_DRY_RUN:-0}"
TPU_RUNTIME_METRICS_PORTS="${TPU_RUNTIME_METRICS_PORTS:-8431,8432,8433,8434}"
CLEAR_TPU_LOCKFILE="${SCACCHI_CLEAR_TPU_LOCKFILE:-1}"

quote_remote() {
  printf "%q" "$1"
}

CONFIG_NAME="config"
NEXT_ARG_IS_CONFIG_NAME=0
for arg in "$@"; do
  if [[ "${NEXT_ARG_IS_CONFIG_NAME}" == "1" ]]; then
    CONFIG_NAME="${arg}"
    NEXT_ARG_IS_CONFIG_NAME=0
    continue
  fi
  case "${arg}" in
    --config-name|-cn)
      NEXT_ARG_IS_CONFIG_NAME=1
      ;;
    --config-name=*|-cn=*)
      CONFIG_NAME="${arg#*=}"
      ;;
  esac
done

config_file_for_name() {
  local config_name="$1"
  if [[ "${config_name}" == *.yaml || "${config_name}" == *.yml ]]; then
    printf '%s/scacchi/configs/%s\n' "${LOCAL_REPO_DIR}" "${config_name}"
  else
    printf '%s/scacchi/configs/%s.yaml\n' "${LOCAL_REPO_DIR}" "${config_name}"
  fi
}

yaml_section_value() {
  local file="$1"
  local wanted_section="$2"
  local wanted_key="$3"
  awk -v wanted_section="${wanted_section}" -v wanted_key="${wanted_key}" '
    /^[[:space:]]*(#|$)/ { next }
    /^[[:alnum:]_][[:alnum:]_-]*:/ {
      section=$1
      sub(/:.*/, "", section)
      next
    }
    section == wanted_section {
      line=$0
      sub(/^[[:space:]]+/, "", line)
      if (index(line, wanted_key ":") == 1) {
        sub("^[^:]+:[[:space:]]*", "", line)
        sub(/[[:space:]]+#.*/, "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
        print line
        exit
      }
    }
  ' "${file}"
}

infer_eval_checkpoint_path() {
  local config_file
  local board_size=""
  local eval_baseline="checkpoint"
  local eval_interval="5"
  local checkpoint_path=""

  config_file="$(config_file_for_name "${CONFIG_NAME}")"
  if [[ -f "${config_file}" ]]; then
    board_size="$(yaml_section_value "${config_file}" env board_size)"
    eval_baseline="$(yaml_section_value "${config_file}" eval baseline)"
    eval_interval="$(yaml_section_value "${config_file}" eval interval)"
    checkpoint_path="$(yaml_section_value "${config_file}" eval checkpoint_path)"
    eval_baseline="${eval_baseline:-checkpoint}"
    eval_interval="${eval_interval:-5}"
  fi

  for arg in "$@"; do
    arg="${arg#+}"
    case "${arg}" in
      env.board_size=*)
        board_size="${arg#env.board_size=}"
        ;;
      eval.baseline=*)
        eval_baseline="${arg#eval.baseline=}"
        ;;
      eval.interval=*)
        eval_interval="${arg#eval.interval=}"
        ;;
      eval.checkpoint_path=*)
        checkpoint_path="${arg#eval.checkpoint_path=}"
        ;;
    esac
  done

  if [[ "${eval_interval}" == "0" || "${eval_baseline}" != "checkpoint" ]]; then
    return 0
  fi

  if [[ -n "${checkpoint_path}" && "${checkpoint_path}" != "null" ]]; then
    printf '%s\n' "${checkpoint_path%/}"
    return 0
  fi

  if [[ -n "${board_size}" && "${board_size}" != "null" ]]; then
    printf 'checkpoints/%s_solved\n' "${board_size}"
  fi
}

sync_eval_checkpoint_baseline() {
  local checkpoint_path="$1"
  local local_checkpoint_path
  local local_checkpoint_parent
  local checkpoint_name
  local remote_checkpoint_parent
  local checkpoint_archive
  local remote_archive

  if [[ -z "${checkpoint_path}" ]]; then
    return 0
  fi

  if [[ "${checkpoint_path}" = /* ]]; then
    local_checkpoint_path="${checkpoint_path}"
    remote_checkpoint_parent="$(dirname "${checkpoint_path}")"
  else
    local_checkpoint_path="${LOCAL_REPO_DIR}/${checkpoint_path}"
    remote_checkpoint_parent="${REPO_DIR}/$(dirname "${checkpoint_path}")"
  fi
  local_checkpoint_parent="$(dirname "${local_checkpoint_path}")"
  checkpoint_name="$(basename "${local_checkpoint_path}")"

  if [[ ! -d "${local_checkpoint_path}" ]]; then
    echo "Eval checkpoint baseline not found locally: ${local_checkpoint_path}" >&2
    echo "Remote training may still fail unless that checkpoint already exists on every TPU worker." >&2
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Would sync eval checkpoint baseline:"
    echo "  ${local_checkpoint_path} -> ${remote_checkpoint_parent}/${checkpoint_name}"
    return 0
  fi

  checkpoint_archive="$(mktemp "${TMPDIR:-/tmp}/scacchi-eval-checkpoint.XXXXXX.tar.gz")"
  eval_checkpoint_archive="${checkpoint_archive}"
  tar -czf "${checkpoint_archive}" -C "${local_checkpoint_parent}" "${checkpoint_name}"
  remote_archive="/tmp/$(basename "${checkpoint_archive}")"

  echo "Syncing eval checkpoint baseline:"
  echo "  ${local_checkpoint_path} -> ${remote_checkpoint_parent}/${checkpoint_name}"

  gcloud compute tpus tpu-vm scp \
    "${checkpoint_archive}" \
    "${TPU_NAME}:${remote_archive}" \
    --zone="${TPU_ZONE}" \
    --project="${TPU_PROJECT}" \
    --worker=all

  "${EOPOD}" run --worker all \
    "mkdir -p $(quote_remote "${remote_checkpoint_parent}") && tar -xzf $(quote_remote "${remote_archive}") -C $(quote_remote "${remote_checkpoint_parent}") && rm -f $(quote_remote "${remote_archive}")"
}

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
  eval_checkpoint_archive=""
  cleanup() {
    rm -f "${archive}" "${file_list}" "${delete_list}"
    if [[ -n "${eval_checkpoint_archive}" ]]; then
      rm -f "${eval_checkpoint_archive}"
    fi
  }
  trap cleanup EXIT

  (
    cd "${LOCAL_REPO_DIR}"
    {
      git ls-files --cached
      git ls-files --others --exclude-standard
    } \
      | awk '!/^(baselines|checkpoints|wandb)\//' \
      | sort -u \
      | while IFS= read -r path; do
          [[ -f "${path}" ]] && printf '%s\n' "${path}"
        done \
      > "${file_list}"
    git diff --name-only --diff-filter=D HEAD -- \
      | awk '!/^(baselines|checkpoints|wandb)\//' \
      > "${delete_list}"
    if [[ -s "${file_list}" ]]; then
      tar -czf "${archive}" --files-from "${file_list}"
    fi
  )

  if [[ -s "${file_list}" ]]; then
    echo "Syncing files:"
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
      "mkdir -p $(quote_remote "${REPO_DIR}") && tar -xzf $(quote_remote "${remote_archive}") -C $(quote_remote "${REPO_DIR}") && rm -f $(quote_remote "${remote_archive}")"
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
      "cd $(quote_remote "${REPO_DIR}") && xargs -r rm -f < $(quote_remote "${remote_delete_list}") && rm -f $(quote_remote "${remote_delete_list}")"
  fi

  case "${SYNC_EVAL_CHECKPOINTS}" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      ;;
    *)
      sync_eval_checkpoint_baseline "$(infer_eval_checkpoint_path "$@")"
      ;;
  esac
fi

REMOTE_PYTHONPATH="${REPO_DIR}"
if [[ -n "${SCACCHI_REMOTE_PYTHONPATH_PREFIX:-}" ]]; then
  REMOTE_PYTHONPATH="${SCACCHI_REMOTE_PYTHONPATH_PREFIX}:${REMOTE_PYTHONPATH}"
fi
REMOTE_ENV=(
  "PYTHONPATH=${REMOTE_PYTHONPATH}"
  "JAX_PLATFORMS=tpu,cpu"
  "TPU_RUNTIME_METRICS_PORTS=${TPU_RUNTIME_METRICS_PORTS}"
)
for name in \
  SCACCHI_RAW_SNAPSHOT_DIR \
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
  echo "  ${EOPOD} run --worker all --retry 1 ${REMOTE_CMD}"
  exit 0
fi

if [[ "${CLEAR_TPU_LOCKFILE}" == "1" ]]; then
  "${EOPOD}" run --worker all "sudo rm -f /tmp/libtpu_lockfile"
fi

# In this eopod version, --retry is implemented as the total attempt count.
"${EOPOD}" run --worker all --retry 1 "${REMOTE_CMD}"

if [[ -n "${SCACCHI_PROFILE_DIR:-}" ]]; then
  echo "Profile traces were written under ${SCACCHI_PROFILE_DIR} on the selected TPU workers."
fi
