#!/usr/bin/env bash
# Temporary: B @ p0.5, GPU 1.
set -euo pipefail
DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
export CUDA_VISIBLE_DEVICES=1
export FIG2_TASK2_PLUS_SCDIFF_CTRL_SLUGS="p0.5"
export FIG2_TASK2_PLUS_HOLDOUT_TYPES="B"
exec bash "${DIR}/fig2_task2_plus_scdiff.sh"
