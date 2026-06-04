#!/usr/bin/env bash
# Temporary: NK @ p0.5, GPU 5.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=5
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.5"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="NK"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
