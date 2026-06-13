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
TPU_WORKERS="${SCACCHI_TPU_WORKERS:-${SCACCHI_TPU_WORKER:-all}}"
SYNC="${SCACCHI_SYNC:-1}"
SYNC_EVAL_CHECKPOINTS="${SCACCHI_SYNC_EVAL_CHECKPOINTS:-auto}"
DRY_RUN="${SCACCHI_DRY_RUN:-0}"
TPU_RUNTIME_METRICS_PORTS="${TPU_RUNTIME_METRICS_PORTS:-8431,8432,8433,8434}"
CLEAR_TPU_LOCKFILE="${SCACCHI_CLEAR_TPU_LOCKFILE:-1}"

archive=""
file_list=""
delete_list=""
eval_checkpoint_archive=""
launch_script=""

cleanup() {
  [[ -n "${archive}" ]] && rm -f "${archive}"
  [[ -n "${file_list}" ]] && rm -f "${file_list}"
  [[ -n "${delete_list}" ]] && rm -f "${delete_list}"
  [[ -n "${eval_checkpoint_archive}" ]] && rm -f "${eval_checkpoint_archive}"
  [[ -n "${launch_script}" ]] && rm -f "${launch_script}"
  return 0
}
trap cleanup EXIT

quote_remote() {
  printf "%q" "$1"
}

TPU_WORKER_LIST=()
TPU_WORKER_LIST_CSV=""

topology_worker_spec() {
  local topology="$1"

  case "${topology}" in
    topo2-0|topology2-0) printf '12,3\n' ;;
    topo2-1|topology2-1) printf '3,11\n' ;;
    topo2-2|topology2-2) printf '11,2\n' ;;
    topo2-3|topology2-3) printf '8,0\n' ;;
    topo2-4|topology2-4) printf '0,1\n' ;;
    topo2-5|topology2-5) printf '1,13\n' ;;
    topo2-6|topology2-6) printf '10,5\n' ;;
    topo2-7|topology2-7) printf '5,9\n' ;;
    topo2-8|topology2-8) printf '9,14\n' ;;
    topo2-9|topology2-9) printf '15,7\n' ;;
    topo2-10|topology2-10) printf '7,4\n' ;;
    topo2-11|topology2-11) printf '4,6\n' ;;
    topo4-0|topology4-0) printf '12,3,11,2\n' ;;
    topo4-1|topology4-1) printf '8,0,1,13\n' ;;
    topo4-2|topology4-2) printf '10,5,9,14\n' ;;
    topo4-3|topology4-3) printf '15,7,4,6\n' ;;
    topo8-0|topology8-0|topo8-b|topo8b|v4-64-b|v4-64b) printf '10,12,5,3,9,11,14,2\n' ;;
    topo8-1|topology8-1|topo8-a|topo8a|v4-64-a|v4-64a) printf '8,15,0,7,1,4,13,6\n' ;;
    topo16-0|topology16-0) printf '8,10,15,12,0,5,7,3,1,9,4,11,13,14,6,2\n' ;;
    *)
      return 1
      ;;
  esac
}

print_topology_usage() {
  cat >&2 <<'EOF'
Use --worker topoN-I with one of:
  topo2-0..topo2-11
  topo4-0 -> 12,3,11,2
  topo4-1 -> 8,0,1,13
  topo4-2 -> 10,5,9,14
  topo4-3 -> 15,7,4,6
  topo8-0 -> 10,12,5,3,9,11,14,2
  topo8-1 -> 8,15,0,7,1,4,13,6
  topo16-0 -> full v4-128 topology order
EOF
}

tpu_process_bounds_for_worker_count() {
  local worker_count="$1"

  if [[ -n "${SCACCHI_TPU_PROCESS_BOUNDS:-}" ]]; then
    printf '%s\n' "${SCACCHI_TPU_PROCESS_BOUNDS}"
    return 0
  fi

  case "${worker_count}" in
    1) printf '1,1,1\n' ;;
    2) printf '1,1,2\n' ;;
    4) printf '1,1,4\n' ;;
    8) printf '1,2,4\n' ;;
    16) printf '2,2,4\n' ;;
    *)
      echo "Unsupported TPU worker count for automatic v4 sub-slice setup: ${worker_count}" >&2
      echo "Use 1, 2, 4, 8, or 16 workers, or set SCACCHI_TPU_PROCESS_BOUNDS explicitly." >&2
      exit 2
      ;;
  esac
}

parse_worker_spec() {
  local spec="$1"
  local part
  local start
  local end
  local worker
  local topology_spec
  declare -A seen=()

  TPU_WORKER_LIST=()
  TPU_WORKER_LIST_CSV=""

  if topology_spec="$(topology_worker_spec "${spec}")"; then
    spec="${topology_spec}"
    TPU_WORKERS="${spec}"
  elif [[ "${spec}" =~ ^(topo|topology)[0-9]+- ]]; then
    echo "invalid topology worker value: ${spec}" >&2
    print_topology_usage
    exit 2
  fi

  if [[ "${spec}" == "all" ]]; then
    return 0
  fi

  IFS=',' read -r -a parts <<< "${spec}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -z "${part}" ]]; then
      continue
    fi
    if [[ "${part}" =~ ^[0-9]+$ ]]; then
      worker="${part}"
      if [[ -n "${seen[${worker}]:-}" ]]; then
        echo "duplicate worker in --worker ${spec}: ${worker}" >&2
        exit 2
      fi
      seen["${worker}"]=1
      TPU_WORKER_LIST+=("${worker}")
      continue
    fi
    if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start > end )); then
        echo "invalid descending worker range: ${part}" >&2
        exit 2
      fi
      for (( worker=start; worker<=end; worker++ )); do
        if [[ -n "${seen[${worker}]:-}" ]]; then
          echo "duplicate worker in --worker ${spec}: ${worker}" >&2
          exit 2
        fi
        seen["${worker}"]=1
        TPU_WORKER_LIST+=("${worker}")
      done
      continue
    fi
    echo "invalid --worker value: ${spec}" >&2
    echo "Use all, topo4-0, topo8-a, topo8-b, a single worker like 3, a range like 0-7, or a comma list like 0,2,4-7." >&2
    exit 2
  done

  if [[ "${#TPU_WORKER_LIST[@]}" -eq 0 ]]; then
    echo "invalid empty --worker value: ${spec}" >&2
    exit 2
  fi

  printf -v TPU_WORKER_LIST_CSV '%s,' "${TPU_WORKER_LIST[@]}"
  TPU_WORKER_LIST_CSV="${TPU_WORKER_LIST_CSV%,}"
}

validate_known_v4_topology_selection() {
  local selected="${TPU_WORKER_LIST_CSV}"
  local valid
  local -a valid_selections=()

  if [[ "${TPU_WORKERS}" == "all" || "${#TPU_WORKER_LIST[@]}" -le 1 ]]; then
    return 0
  fi
  if [[ -n "${SCACCHI_TPU_PROCESS_BOUNDS:-}" || "${SCACCHI_SKIP_TOPOLOGY_VALIDATION:-0}" == "1" ]]; then
    return 0
  fi

  case "${#TPU_WORKER_LIST[@]}" in
    2)
      valid_selections=(
        "8,0" "0,1" "1,13"
        "10,5" "5,9" "9,14"
        "15,7" "7,4" "4,6"
        "12,3" "3,11" "11,2"
      )
      ;;
    4)
      valid_selections=(
        "8,0,1,13"
        "10,5,9,14"
        "15,7,4,6"
        "12,3,11,2"
      )
      ;;
    8)
      valid_selections=(
        "8,15,0,7,1,4,13,6"
        "10,12,5,3,9,11,14,2"
      )
      ;;
    16)
      valid_selections=("8,10,15,12,0,5,7,3,1,9,4,11,13,14,6,2")
      ;;
    *)
      return 0
      ;;
  esac

  for valid in "${valid_selections[@]}"; do
    if [[ "${selected}" == "${valid}" ]]; then
      return 0
    fi
  done

  echo "TPU worker selection is not a known topology-ordered v4 sub-slice: ${selected}" >&2
  echo "Numeric worker ids are SSH ids, not JAX topology order; ranges such as 0-7 or 2-3 can connect but fail TPU mesh initialization." >&2
  echo "Use a topology-ordered list or preset, for example:" >&2
  case "${#TPU_WORKER_LIST[@]}" in
    2)
      echo "  --worker topo2-0" >&2
      echo "  --worker topo2-4" >&2
      echo "  --worker topo2-11" >&2
      ;;
    4)
      echo "  --worker topo4-0" >&2
      echo "  --worker topo4-1" >&2
      ;;
    8)
      echo "  --worker topo8-0" >&2
      echo "  --worker topo8-1" >&2
      ;;
    16)
      echo "  --worker topo16-0" >&2
      ;;
  esac
  echo "Set SCACCHI_SKIP_TOPOLOGY_VALIDATION=1 to bypass this check for probes." >&2
  exit 2
}

write_remote_launch_script() {
  local path="$1"
  local coordinator_port="${SCACCHI_JAX_COORDINATOR_PORT:-8476}"
  local tpu_accelerator_type=""
  local tpu_chips_per_process_bounds="${SCACCHI_TPU_CHIPS_PER_PROCESS_BOUNDS:-2,2,1}"
  local tpu_process_bounds=""
  local assignment
  local arg

  if [[ "${TPU_WORKERS}" != "all" ]]; then
    tpu_accelerator_type="${SCACCHI_TPU_ACCELERATOR_TYPE:-v4-$((${#TPU_WORKER_LIST[@]} * 8))}"
    tpu_process_bounds="$(tpu_process_bounds_for_worker_count "${#TPU_WORKER_LIST[@]}")"
  fi

  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf 'cd %s\n' "$(quote_remote "${REPO_DIR}")"

    if [[ "${TPU_WORKERS}" != "all" ]]; then
      printf 'SCACCHI_JAX_WORKER_LIST=%s\n' "$(quote_remote "${TPU_WORKER_LIST_CSV}")"
      printf 'SCACCHI_JAX_NUM_PROCESSES=%s\n' "$(quote_remote "${#TPU_WORKER_LIST[@]}")"
      printf 'SCACCHI_JAX_COORDINATOR_WORKER=%s\n' "$(quote_remote "${TPU_WORKER_LIST[0]}")"
      printf 'SCACCHI_JAX_COORDINATOR_PORT=%s\n' "$(quote_remote "${coordinator_port}")"
      printf 'SCACCHI_TPU_ACCELERATOR_TYPE=%s\n' "$(quote_remote "${tpu_accelerator_type}")"
      printf 'SCACCHI_TPU_CHIPS_PER_PROCESS_BOUNDS=%s\n' "$(quote_remote "${tpu_chips_per_process_bounds}")"
      printf 'SCACCHI_TPU_PROCESS_BOUNDS=%s\n' "$(quote_remote "${tpu_process_bounds}")"
      printf '%s\n' 'SCACCHI_TPU_WORKER_ID="$(curl -sfH '\''Metadata-Flavor: Google'\'' http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)"'
      printf '%s\n' 'IFS='\'','\'' read -r -a SCACCHI_JAX_WORKERS <<< "${SCACCHI_JAX_WORKER_LIST}"'
      printf '%s\n' 'SCACCHI_JAX_PROCESS_ID=""'
      printf '%s\n' 'for idx in "${!SCACCHI_JAX_WORKERS[@]}"; do'
      printf '%s\n' '  if [[ "${SCACCHI_JAX_WORKERS[$idx]}" == "${SCACCHI_TPU_WORKER_ID}" ]]; then'
      printf '%s\n' '    SCACCHI_JAX_PROCESS_ID="${idx}"'
      printf '%s\n' '    break'
      printf '%s\n' '  fi'
      printf '%s\n' 'done'
      printf '%s\n' ': "${SCACCHI_JAX_PROCESS_ID:?missing_process_id}"'
      printf '%s\n' 'SCACCHI_TPU_WORKER_ENDPOINTS="$(curl -sfH '\''Metadata-Flavor: Google'\'' http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints)"'
      printf '%s\n' 'IFS='\'','\'' read -r -a SCACCHI_TPU_ENDPOINTS <<< "${SCACCHI_TPU_WORKER_ENDPOINTS}"'
      printf '%s\n' 'declare -a SCACCHI_TPU_ENDPOINT_HOSTS=()'
      printf '%s\n' 'for endpoint_index in "${!SCACCHI_TPU_ENDPOINTS[@]}"; do'
      printf '%s\n' '  IFS='\'':'\'' read -r _endpoint_name _endpoint_port endpoint_host _endpoint_rest <<< "${SCACCHI_TPU_ENDPOINTS[$endpoint_index]}"'
      printf '%s\n' '  SCACCHI_TPU_ENDPOINT_HOSTS[$endpoint_index]="${endpoint_host}"'
      printf '%s\n' 'done'
      printf '%s\n' 'SCACCHI_TPU_WORKER_HOSTNAMES=""'
      printf '%s\n' 'for worker in "${SCACCHI_JAX_WORKERS[@]}"; do'
      printf '%s\n' '  host="${SCACCHI_TPU_ENDPOINT_HOSTS[$worker]:-}"'
      printf '%s\n' '  if [[ -z "${host}" ]]; then'
      printf '%s\n' '    echo "missing worker hostname for TPU worker ${worker}" >&2'
      printf '%s\n' '    exit 1'
      printf '%s\n' '  fi'
      printf '%s\n' '  SCACCHI_TPU_WORKER_HOSTNAMES+="${SCACCHI_TPU_WORKER_HOSTNAMES:+,}${host}"'
      printf '%s\n' 'done'
      printf '%s\n' ': "${SCACCHI_TPU_WORKER_HOSTNAMES:?missing_worker_hostnames}"'
      printf '%s\n' 'SCACCHI_JAX_COORDINATOR_HOST="${SCACCHI_TPU_WORKER_HOSTNAMES%%,*}"'
      printf '%s\n' ': "${SCACCHI_JAX_COORDINATOR_HOST:?missing_coordinator_host}"'
      printf '%s\n' 'export SCACCHI_JAX_NUM_PROCESSES'
      printf '%s\n' 'export SCACCHI_JAX_PROCESS_ID'
      printf '%s\n' 'export SCACCHI_JAX_COORDINATOR_ADDRESS="${SCACCHI_JAX_COORDINATOR_HOST}:${SCACCHI_JAX_COORDINATOR_PORT}"'
      printf '%s\n' 'export TPU_SKIP_MDS_QUERY=1'
      printf '%s\n' 'export TPU_WORKER_HOSTNAMES="${SCACCHI_TPU_WORKER_HOSTNAMES}"'
      printf '%s\n' 'export TPU_WORKER_ID="${SCACCHI_JAX_PROCESS_ID}"'
      printf '%s\n' 'export TPU_ACCELERATOR_TYPE="${SCACCHI_TPU_ACCELERATOR_TYPE}"'
      printf '%s\n' 'export TPU_CHIPS_PER_PROCESS_BOUNDS="${SCACCHI_TPU_CHIPS_PER_PROCESS_BOUNDS}"'
      printf '%s\n' 'export TPU_PROCESS_BOUNDS="${SCACCHI_TPU_PROCESS_BOUNDS}"'
      printf '%s\n' 'echo "Scacchi TPU worker ${SCACCHI_TPU_WORKER_ID} -> JAX process ${SCACCHI_JAX_PROCESS_ID}/${SCACCHI_JAX_NUM_PROCESSES} at ${SCACCHI_JAX_COORDINATOR_ADDRESS}"'
    fi

    for assignment in "${REMOTE_ENV[@]}"; do
      printf 'export %s\n' "$(quote_remote "${assignment}")"
    done

    printf '%s\n' 'set +e'
    printf '%s -u -m scacchi.train' "$(quote_remote "${PYTHON}")"
    for arg in "${TRAIN_ARGS[@]}"; do
      printf ' %s' "$(quote_remote "${arg}")"
    done
    printf '\n'
    printf '%s\n' 'status=$?'
    printf '%s\n' 'if [[ "${status}" -ne 0 ]]; then'
    printf '%s\n' '  latest="$(ls -t /tmp/tpu_logs/*ERROR* 2>/dev/null | head -1)"'
    printf '%s\n' '  if [[ -n "${latest}" ]]; then'
    printf '%s\n' '    printf '\''\n__SCACCHI_TPU_ERROR_LOG__ %s\n'\'' "${latest}"'
    printf '%s\n' '    tail -80 "${latest}" 2>/dev/null || true'
    printf '%s\n' '  fi'
    printf '%s\n' 'fi'
    printf '%s\n' 'printf '\''\n__SCACCHI_REMOTE_STATUS__=%s\n'\'' "${status}"'
    printf '%s\n' 'exit "${status}"'
  } > "${path}"
}

CONFIG_NAME="config"
TRAIN_ARGS=()
NEXT_ARG_IS_CONFIG_NAME=0
NEXT_ARG_IS_WORKER=0
for arg in "$@"; do
  if [[ "${NEXT_ARG_IS_WORKER}" == "1" ]]; then
    TPU_WORKERS="${arg}"
    NEXT_ARG_IS_WORKER=0
    continue
  fi
  if [[ "${NEXT_ARG_IS_CONFIG_NAME}" == "1" ]]; then
    CONFIG_NAME="${arg}"
    NEXT_ARG_IS_CONFIG_NAME=0
    TRAIN_ARGS+=("${arg}")
    continue
  fi
  case "${arg}" in
    --worker|--workers|--tpu-worker|--tpu-workers)
      NEXT_ARG_IS_WORKER=1
      ;;
    --worker=*|--workers=*|--tpu-worker=*|--tpu-workers=*)
      TPU_WORKERS="${arg#*=}"
      ;;
    --topology)
      echo "--topology is not supported; use --worker topo4-0 or another topology alias." >&2
      print_topology_usage
      exit 2
      ;;
    --topology=*)
      echo "--topology is not supported; use --worker ${arg#*=} only if it is a valid --worker value." >&2
      print_topology_usage
      exit 2
      ;;
    --config-name|-cn)
      NEXT_ARG_IS_CONFIG_NAME=1
      TRAIN_ARGS+=("${arg}")
      ;;
    --config-name=*|-cn=*)
      CONFIG_NAME="${arg#*=}"
      TRAIN_ARGS+=("${arg}")
      ;;
    *)
      TRAIN_ARGS+=("${arg}")
      ;;
  esac
done
if [[ "${NEXT_ARG_IS_WORKER}" == "1" ]]; then
  echo "--worker requires a value, for example: --worker 0-7" >&2
  exit 2
fi
if [[ "${NEXT_ARG_IS_CONFIG_NAME}" == "1" ]]; then
  echo "--config-name requires a value." >&2
  exit 2
fi
parse_worker_spec "${TPU_WORKERS}"
validate_known_v4_topology_selection

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
    --worker="${TPU_WORKERS}"

  run_eopod "mkdir -p $(quote_remote "${remote_checkpoint_parent}") && tar -xzf $(quote_remote "${remote_archive}") -C $(quote_remote "${remote_checkpoint_parent}") && rm -f $(quote_remote "${remote_archive}")"
}

if [[ -n "${SCACCHI_EOPOD:-}" ]]; then
  EOPOD="${SCACCHI_EOPOD}"
elif [[ -x /home/francescosacco/eopod-github-venv/bin/eopod ]]; then
  EOPOD="/home/francescosacco/eopod-github-venv/bin/eopod"
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

remote_status_wrapper() {
  local command="$1"

  printf 'set +e; %s; status=$?; if [ "${status}" -ne 0 ]; then latest="$(ls -t /tmp/tpu_logs/*ERROR* 2>/dev/null | head -1)"; if [ -n "${latest}" ]; then printf '"'"'\\n__SCACCHI_TPU_ERROR_LOG__ %%s\\n'"'"' "${latest}"; tail -80 "${latest}" 2>/dev/null || true; fi; fi; printf '"'"'\\n__SCACCHI_REMOTE_STATUS__=%%s\\n'"'"' "${status}"; exit 0' "${command}"
}

run_eopod() {
  local command="$1"
  local wrapped_command
  local log_dir
  local worker
  local log
  local pid
  local failed=0
  local -a pids=()
  local -a logs=()
  local -a workers=()

  if [[ "${TPU_WORKERS}" == "all" ]]; then
    "${EOPOD}" run --worker "${TPU_WORKERS}" --retry 1 "${command}"
    return
  fi

  wrapped_command="$(remote_status_wrapper "${command}")"
  log_dir="$(mktemp -d "${TMPDIR:-/tmp}/scacchi-eopod.XXXXXX")"

  for worker in "${TPU_WORKER_LIST[@]}"; do
    log="${log_dir}/worker-${worker}.log"
    logs+=("${log}")
    workers+=("${worker}")
    (
      "${EOPOD}" run --worker "${worker}" --retry 1 --no-stream "${wrapped_command}"
    ) > >(sed -u "s/^/[worker ${worker}] /" | tee "${log}") 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  for idx in "${!logs[@]}"; do
    if ! grep -q '^.*__SCACCHI_REMOTE_STATUS__=0$' "${logs[$idx]}"; then
      echo "Remote command failed on TPU worker ${workers[$idx]}." >&2
      failed=1
    fi
  done

  rm -rf "${log_dir}"
  return "${failed}"
}

run_eopod_stream() {
  local command="$1"
  local log_dir
  local worker
  local log
  local pid
  local failed=0
  local -a pids=()
  local -a logs=()
  local -a workers=()

  if [[ "${TPU_WORKERS}" == "all" ]]; then
    "${EOPOD}" run --worker "${TPU_WORKERS}" --retry 1 "${command}"
    return
  fi

  log_dir="$(mktemp -d "${TMPDIR:-/tmp}/scacchi-eopod.XXXXXX")"

  for worker in "${TPU_WORKER_LIST[@]}"; do
    log="${log_dir}/worker-${worker}.log"
    logs+=("${log}")
    workers+=("${worker}")
    (
      "${EOPOD}" run --worker "${worker}" --retry 1 "${command}"
    ) > >(sed -u "s/^/[worker ${worker}] /" | tee "${log}") 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  for idx in "${!logs[@]}"; do
    if ! grep -q '^.*__SCACCHI_REMOTE_STATUS__=0$' "${logs[$idx]}"; then
      echo "Remote command failed on TPU worker ${workers[$idx]}." >&2
      failed=1
    fi
  done

  rm -rf "${log_dir}"
  return "${failed}"
}

if [[ "${SYNC}" == "1" ]]; then
  archive="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync.XXXXXX.tar.gz")"
  file_list="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync-files.XXXXXX")"
  delete_list="$(mktemp "${TMPDIR:-/tmp}/scacchi-sync-delete.XXXXXX")"
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
      --worker="${TPU_WORKERS}"

    run_eopod "mkdir -p $(quote_remote "${REPO_DIR}") && tar -xzf $(quote_remote "${remote_archive}") -C $(quote_remote "${REPO_DIR}") && rm -f $(quote_remote "${remote_archive}")"
  fi

  if [[ "${DRY_RUN}" != "1" && -s "${delete_list}" ]]; then
    remote_delete_list="/tmp/$(basename "${delete_list}")"
    gcloud compute tpus tpu-vm scp \
      "${delete_list}" \
      "${TPU_NAME}:${remote_delete_list}" \
      --zone="${TPU_ZONE}" \
      --project="${TPU_PROJECT}" \
      --worker="${TPU_WORKERS}"

    run_eopod "cd $(quote_remote "${REPO_DIR}") && xargs -r rm -f < $(quote_remote "${remote_delete_list}") && rm -f $(quote_remote "${remote_delete_list}")"
  fi

  case "${SYNC_EVAL_CHECKPOINTS}" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      ;;
    *)
      sync_eval_checkpoint_baseline "$(infer_eval_checkpoint_path "${TRAIN_ARGS[@]}")"
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
  SCACCHI_PROFILE_PERFETTO \
  SCACCHI_DISABLE_DISTRIBUTED \
  SCACCHI_JAX_INITIALIZATION_TIMEOUT \
  SCACCHI_JAX_LOCAL_DEVICE_IDS \
  SCACCHI_JAX_COORDINATOR_BIND_ADDRESS
do
  if [[ -n "${!name:-}" ]]; then
    REMOTE_ENV+=("${name}=${!name}")
  fi
done

launch_script="$(mktemp "${TMPDIR:-/tmp}/scacchi-tpu-launch.XXXXXX.sh")"
remote_launch_script="/tmp/$(basename "${launch_script}")"
write_remote_launch_script "${launch_script}"
REMOTE_CMD="bash $(quote_remote "${remote_launch_script}")"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Would run:"
  echo "  gcloud compute tpus tpu-vm scp ${launch_script} ${TPU_NAME}:${remote_launch_script} --zone=${TPU_ZONE} --project=${TPU_PROJECT} --worker=${TPU_WORKERS}"
  if [[ "${TPU_WORKERS}" == "all" ]]; then
    echo "  ${EOPOD} run --worker ${TPU_WORKERS} --retry 1 ${REMOTE_CMD}"
  else
    for worker in "${TPU_WORKER_LIST[@]}"; do
      echo "  ${EOPOD} run --worker ${worker} --retry 1 ${REMOTE_CMD}"
    done
  fi
  echo "Remote launch script:"
  sed 's/^/  /' "${launch_script}"
  exit 0
fi

gcloud compute tpus tpu-vm scp \
  "${launch_script}" \
  "${TPU_NAME}:${remote_launch_script}" \
  --zone="${TPU_ZONE}" \
  --project="${TPU_PROJECT}" \
  --worker="${TPU_WORKERS}"

if [[ "${CLEAR_TPU_LOCKFILE}" == "1" ]]; then
  run_eopod "sudo rm -f /tmp/libtpu_lockfile"
fi

# In this eopod version, --retry is implemented as the total attempt count.
run_eopod_stream "${REMOTE_CMD}"

if [[ -n "${SCACCHI_PROFILE_DIR:-}" ]]; then
  echo "Profile traces were written under ${SCACCHI_PROFILE_DIR} on the selected TPU workers."
fi
