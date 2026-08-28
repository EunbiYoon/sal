#!/usr/bin/env bash
# Train one paper variant with torch DistributedDataParallel on one node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: $0 --variant {filter_on|filter_off|core|aux|all|rw} [--resume]" >&2
}

VARIANT=""
RESUME=false
while (($#)); do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      VARIANT="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$VARIANT" in
  filter_on) PAIRS="data/paper/filter_on.jsonl" ;;
  filter_off) PAIRS="data/paper/filter_off.jsonl" ;;
  core) PAIRS="data/paper/a_beta_core.jsonl" ;;
  aux) PAIRS="data/paper/a_beta_aux.jsonl" ;;
  all) PAIRS="data/paper/a_beta_all.jsonl" ;;
  rw) PAIRS="data/paper/a_beta_rw.jsonl" ;;
  *)
    echo "ERROR: invalid or missing --variant: ${VARIANT:-<empty>}" >&2
    usage
    exit 2
    ;;
esac

CALLER_RUN_ID_SET="${RUN_ID+x}"
CALLER_RUN_ID="${RUN_ID-}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [[ -n "$CALLER_RUN_ID_SET" && -n "$CALLER_RUN_ID" ]]; then
  export RUN_ID="$CALLER_RUN_ID"
else
  # RUN_ID in .env is intentionally ignored: an unspecified run starts a new session.
  export RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
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

DDP_NUM_GPUS="${DDP_NUM_GPUS:-3}"
DDP_GRAD_ACCUM="${DDP_GRAD_ACCUM:-3}"
if ! [[ "$DDP_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DDP_NUM_GPUS must be a positive integer" >&2
  exit 1
fi
VISIBLE_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if ((VISIBLE_COUNT != DDP_NUM_GPUS)); then
  echo "ERROR: DDP_NUM_GPUS=${DDP_NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes ${VISIBLE_COUNT}" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "$RESUME" == "true" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "DDP session=${RUN_ID} variant=${VARIANT} GPUs=${DDP_NUM_GPUS} grad_accum=${DDP_GRAD_ACCUM} resume=${RESUME}"

exec python -m torch.distributed.run --standalone --nproc_per_node="$DDP_NUM_GPUS" -m train \
  --paper \
  --tensorboard \
  --pairs "$PAIRS" \
  --out "$VARIANT" \
  --grad-accum "$DDP_GRAD_ACCUM" \
  "${RESUME_ARGS[@]}"
