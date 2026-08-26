#!/usr/bin/env bash
# Train every paper LoRA variant using hyperparameters from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

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

train_variant() {
  local gpu="$1"
  local variant="$2"
  local pairs="$3"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%H:%M:%S')] GPU ${gpu} -> ${variant}: ${pairs}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m lora.train \
    --paper \
    --tensorboard \
    --pairs "$pairs" \
    --out "$variant" \
    2>&1 | sed -u "s/^/[GPU ${gpu}][${variant}] /" | tee "$log_file"
}

[[ -f "${DATA_DIR}/a_beta_all.jsonl" ]] || {
  echo "ERROR: missing ${DATA_DIR}/a_beta_all.jsonl" >&2
  exit 1
}

VARIANTS=(filter_on filter_off core aux all rw)
PAIR_FILES=(
  "${DATA_DIR}/filter_on.jsonl"
  "${DATA_DIR}/filter_off.jsonl"
  "${DATA_DIR}/a_beta_core.jsonl"
  "${DATA_DIR}/a_beta_aux.jsonl"
  "${DATA_DIR}/a_beta_all.jsonl"
  "${DATA_DIR}/a_beta_rw.jsonl"
)

worker() {
  local gpu="$1"
  local index
  for ((index = gpu; index < ${#VARIANTS[@]}; index += TRAIN_NUM_GPUS)); do
    train_variant "$gpu" "${VARIANTS[index]}" "${PAIR_FILES[index]}"
  done
}

echo "Session: ${RUN_ID}; parallel workers: ${TRAIN_NUM_GPUS}"
PIDS=()
for ((gpu = 0; gpu < TRAIN_NUM_GPUS; gpu++)); do
  worker "$gpu" &
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

CUDA_VISIBLE_DEVICES=0 python -m eval.run_table \
  --table 5 \
  --merge \
  --checkpoint-dir "runs/${RUN_ID}/lora"

echo "Training complete: runs/${RUN_ID}/lora"
echo "Worker logs: ${LOG_DIR}"
