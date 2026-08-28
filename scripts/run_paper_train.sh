#!/usr/bin/env bash
# Compatibility entry point for training one paper variant on one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "Usage: $0 --variant {filter_on|filter_off|core|aux|all|rw}" >&2
}

VARIANT=""
while (($#)); do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      VARIANT="$2"
      shift 2
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
  filter_on|filter_off|core|aux|all|rw) ;;
  *)
    echo "ERROR: invalid or missing --variant: ${VARIANT:-<empty>}" >&2
    usage
    exit 2
    ;;
esac

# CUDA_VISIBLE_DEVICES is intentionally inherited from the caller. Within that
# restricted view, train/train.sh uses logical GPU 0 for this single worker.
export TRAIN_VARIANTS="$VARIANT"
export TRAIN_NUM_GPUS=1
export TRAIN_GPU_IDS=0
export TRAIN_VISIBLE_DEVICES=""
export TRAIN_AUTO_MERGE=false

exec "$ROOT/train/train.sh"
