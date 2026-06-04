#!/usr/bin/env bash
# Temporary: B @ p0, GPU 0.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="B"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
