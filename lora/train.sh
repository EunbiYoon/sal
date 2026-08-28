#!/usr/bin/env bash
# Train every paper LoRA variant using hyperparameters from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Values explicitly supplied before ./lora/train.sh override .env.
CALLER_RUN_ID_SET="${RUN_ID+x}"
CALLER_RUN_ID="${RUN_ID-}"
CALLER_TRAIN_VARIANTS_SET="${TRAIN_VARIANTS+x}"
CALLER_TRAIN_VARIANTS="${TRAIN_VARIANTS-}"
CALLER_TRAIN_NUM_GPUS_SET="${TRAIN_NUM_GPUS+x}"
CALLER_TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS-}"
CALLER_TRAIN_GPU_IDS_SET="${TRAIN_GPU_IDS+x}"
CALLER_TRAIN_GPU_IDS="${TRAIN_GPU_IDS-}"
CALLER_TRAIN_AUTO_MERGE_SET="${TRAIN_AUTO_MERGE+x}"
CALLER_TRAIN_AUTO_MERGE="${TRAIN_AUTO_MERGE-}"
CALLER_TRAIN_VISIBLE_DEVICES_SET="${TRAIN_VISIBLE_DEVICES+x}"
CALLER_TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

[[ -n "$CALLER_RUN_ID_SET" ]] && export RUN_ID="$CALLER_RUN_ID"
[[ -n "$CALLER_TRAIN_VARIANTS_SET" ]] && export TRAIN_VARIANTS="$CALLER_TRAIN_VARIANTS"
[[ -n "$CALLER_TRAIN_NUM_GPUS_SET" ]] && export TRAIN_NUM_GPUS="$CALLER_TRAIN_NUM_GPUS"
[[ -n "$CALLER_TRAIN_GPU_IDS_SET" ]] && export TRAIN_GPU_IDS="$CALLER_TRAIN_GPU_IDS"
[[ -n "$CALLER_TRAIN_AUTO_MERGE_SET" ]] && export TRAIN_AUTO_MERGE="$CALLER_TRAIN_AUTO_MERGE"
[[ -n "$CALLER_TRAIN_VISIBLE_DEVICES_SET" ]] && export TRAIN_VISIBLE_DEVICES="$CALLER_TRAIN_VISIBLE_DEVICES"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.conda/pkgs}"
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/${USER}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-126}"

module load conda/latest cuda/12.6
if [[ -n "${CUDA_HOME:-}" ]]; then
  NVHPC_ROOT="$(cd "${CUDA_HOME}/../.." && pwd)"
  CUDA_PATHS="${NVHPC_ROOT}/math_libs/lib64:${CUDA_HOME}/lib64"
  export LD_LIBRARY_PATH="${CUDA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sal

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda is unavailable'"
python -c "import bitsandbytes.cextension as e; assert e.lib is not None and getattr(e.lib, 'compiled_with_cuda', False), 'bitsandbytes CUDA is unavailable'"

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/paper}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-3}"
TRAIN_VARIANTS="${TRAIN_VARIANTS:-filter_on,filter_off,core,aux,all,rw}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-}"
TRAIN_AUTO_MERGE="${TRAIN_AUTO_MERGE:-true}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-}"
LOG_DIR="runs/${RUN_ID}/train_logs"
mkdir -p "$LOG_DIR"

if ! [[ "$TRAIN_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || ((TRAIN_NUM_GPUS > 6)); then
  echo "ERROR: TRAIN_NUM_GPUS must be an integer from 1 to 6" >&2
  exit 1
fi

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if ((VISIBLE_GPUS < TRAIN_NUM_GPUS)); then
  echo "ERROR: TRAIN_NUM_GPUS=${TRAIN_NUM_GPUS}, but only ${VISIBLE_GPUS} GPU(s) are visible" >&2
  exit 1
fi

if [[ -n "$TRAIN_GPU_IDS" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$TRAIN_GPU_IDS"
else
  GPU_IDS=()
  for ((gpu = 0; gpu < TRAIN_NUM_GPUS; gpu++)); do
    GPU_IDS+=("$gpu")
  done
fi
if ((${#GPU_IDS[@]} != TRAIN_NUM_GPUS)); then
  echo "ERROR: TRAIN_GPU_IDS must contain exactly TRAIN_NUM_GPUS entries" >&2
  exit 1
fi
if [[ -n "$TRAIN_VISIBLE_DEVICES" ]] && ((TRAIN_NUM_GPUS != 1)); then
  echo "ERROR: TRAIN_VISIBLE_DEVICES model sharding requires TRAIN_NUM_GPUS=1" >&2
  exit 1
fi

train_variant() {
  local gpu="$1"
  local variant="$2"
  local pairs="$3"
  local log_file="${LOG_DIR}/${variant}.log"
  local visible_devices="${TRAIN_VISIBLE_DEVICES:-$gpu}"
  local run_info="runs/${RUN_ID}/lora/${variant}/run_info.json"
  local -a resume_args=()

  if [[ -f "$run_info" ]] && python - "$run_info" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "completed" else 1)
PY
  then
    echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: already completed; skipping"
    return 0
  fi

  if compgen -G "runs/${RUN_ID}/lora/${variant}/adapter/checkpoint-*" >/dev/null; then
    resume_args+=(--resume)
    echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: resuming latest checkpoint"
  fi

  echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: ${pairs}"
  CUDA_VISIBLE_DEVICES="$visible_devices" python -m lora.train \
    --paper \
    --tensorboard \
    --pairs "$pairs" \
    --out "$variant" \
    "${resume_args[@]}" \
    2>&1 | sed -u "s/^/[GPU ${visible_devices}][${variant}] /" | tee "$log_file"
}

[[ -f "${DATA_DIR}/a_beta_all.jsonl" ]] || {
  echo "ERROR: missing ${DATA_DIR}/a_beta_all.jsonl" >&2
  exit 1
}

declare -A PAIRS_BY_VARIANT=(
  [filter_on]="${DATA_DIR}/filter_on.jsonl"
  [filter_off]="${DATA_DIR}/filter_off.jsonl"
  [core]="${DATA_DIR}/a_beta_core.jsonl"
  [aux]="${DATA_DIR}/a_beta_aux.jsonl"
  [all]="${DATA_DIR}/a_beta_all.jsonl"
  [rw]="${DATA_DIR}/a_beta_rw.jsonl"
)

IFS=',' read -r -a REQUESTED_VARIANTS <<< "$TRAIN_VARIANTS"
VARIANTS=()
PAIR_FILES=()
for raw_variant in "${REQUESTED_VARIANTS[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  if [[ -z "$variant" || -z "${PAIRS_BY_VARIANT[$variant]+x}" ]]; then
    echo "ERROR: invalid TRAIN_VARIANTS entry: ${raw_variant}" >&2
    exit 1
  fi
  VARIANTS+=("$variant")
  PAIR_FILES+=("${PAIRS_BY_VARIANT[$variant]}")
done

worker() {
  local slot="$1"
  local gpu="$2"
  local index
  for ((index = slot; index < ${#VARIANTS[@]}; index += WORKER_COUNT)); do
    train_variant "$gpu" "${VARIANTS[index]}" "${PAIR_FILES[index]}"
  done
}

WORKER_COUNT="$TRAIN_NUM_GPUS"
if ((WORKER_COUNT > ${#VARIANTS[@]})); then
  WORKER_COUNT="${#VARIANTS[@]}"
fi

echo "Session: ${RUN_ID}; variants: ${VARIANTS[*]}; parallel workers: ${WORKER_COUNT}"
PIDS=()
for ((gpu = 0; gpu < WORKER_COUNT; gpu++)); do
  worker "$gpu" "${GPU_IDS[gpu]}" &
  PIDS+=("$!")
done

cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=1
  fi
done
trap - INT TERM

if ((FAILED)); then
  echo "ERROR: one or more training workers failed; merge skipped" >&2
  exit 1
fi

if [[ "$TRAIN_AUTO_MERGE" == "true" ]] \
  && [[ -f "runs/${RUN_ID}/lora/aux/adapter_config.json" ]] \
  && [[ -f "runs/${RUN_ID}/lora/all/adapter_config.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m eval.run_table \
    --table 5 \
    --merge \
    --checkpoint-dir "runs/${RUN_ID}/lora"
else
  echo "Merge skipped (TRAIN_AUTO_MERGE=${TRAIN_AUTO_MERGE}; requires completed aux and all)"
fi

echo "Training complete: runs/${RUN_ID}/lora"
echo "Worker logs: ${LOG_DIR}"
